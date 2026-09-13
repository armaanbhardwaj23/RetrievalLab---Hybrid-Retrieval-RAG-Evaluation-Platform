"""A bounded retrieval tool-calling answer path, not a general-purpose agent."""

import json
from collections.abc import Callable, Sequence
from typing import Any

from retrievallab.models import RetrievalResult
from retrievallab.openrouter import openrouter_client
from retrievallab.rag import GroundedAnswer, SourceCitation, _configured_chat_model, _source_label

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": "Search the indexed handbooks for passages that can answer the user's question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A focused handbook search query."}},
            "required": ["query"], "additionalProperties": False,
        },
    },
}


class ToolCallingAnswerGenerator:
    """Let a model make exactly one bounded call to the retrieval tool."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or _configured_chat_model()
        self._client = client or openrouter_client(api_key)

    def answer(self, question: str, search: Callable[[str], Sequence[RetrievalResult]], *, max_tokens: int = 400) -> GroundedAnswer:
        """Force one `search_handbook` call, then answer only from its passages."""
        if not question.strip():
            raise ValueError("question cannot be empty.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use search_handbook exactly once. Answer only from its returned passages. If evidence is insufficient, say so. Cite factual claims as [S1], [S2], etc."},
            {"role": "user", "content": question},
        ]
        first_response = self._client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=128, messages=messages, tools=[SEARCH_TOOL],
            tool_choice={"type": "function", "function": {"name": "search_handbook"}},
        )
        tool_call = _single_search_call(first_response.choices[0].message)
        query = _tool_query(tool_call.function.arguments)
        results = tuple(search(query))
        if not results:
            raise RuntimeError("The retrieval tool returned no sources.")
        citations = tuple(SourceCitation(label=f"S{position}", chunk=result.chunk) for position, result in enumerate(results, start=1))
        messages.extend([_assistant_tool_message(tool_call), {"role": "tool", "tool_call_id": tool_call.id, "content": _tool_result(citations)}])
        final_response = self._client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=max_tokens, messages=messages,
        )
        text = final_response.choices[0].message.content
        if not text:
            raise RuntimeError("The model returned an empty answer.")
        usage = final_response.usage
        return GroundedAnswer(text=text, model=final_response.model, citations=citations,
                              input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)


def _single_search_call(message: Any) -> Any:
    calls = getattr(message, "tool_calls", None) or []
    if len(calls) != 1 or calls[0].function.name != "search_handbook":
        raise RuntimeError("The model must make exactly one search_handbook call.")
    return calls[0]


def _tool_query(arguments: str) -> str:
    try:
        query = json.loads(arguments)["query"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("search_handbook requires a JSON string field named query.") from error
    if not isinstance(query, str) or not query.strip():
        raise RuntimeError("search_handbook query must be a non-empty string.")
    return query


def _assistant_tool_message(tool_call: Any) -> dict[str, Any]:
    return {"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": "function", "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}}]}


def _tool_result(citations: Sequence[SourceCitation]) -> str:
    return "\n\n".join(f"[{citation.label}] {_source_label(citation.chunk)}\n{citation.chunk.text}" for citation in citations)
