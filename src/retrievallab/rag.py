"""Grounded answer generation from retrieved chunks."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from retrievallab.models import Chunk, RetrievalResult
from retrievallab.openrouter import openrouter_client


DEFAULT_CHAT_MODEL = "openai/gpt-4.1-mini"


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """A stable label for one retrieved source passage."""

    label: str
    chunk: Chunk


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer plus the passages supplied as its citable evidence."""

    text: str
    model: str
    citations: tuple[SourceCitation, ...]
    input_tokens: int
    output_tokens: int


class OpenRouterAnswerGenerator:
    """Generate a constrained answer from already retrieved passages."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or _configured_chat_model()
        self._client = client or openrouter_client(api_key)

    def answer(
        self,
        question: str,
        results: Sequence[RetrievalResult],
        *,
        max_tokens: int = 400,
    ) -> GroundedAnswer:
        """Answer using supplied retrieval results and return their citations."""

        if not question.strip():
            raise ValueError("question cannot be empty.")
        if not results:
            raise ValueError("At least one retrieved result is required.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        citations = tuple(
            SourceCitation(label=f"S{position}", chunk=result.chunk)
            for position, result in enumerate(results, start=1)
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied sources. If the sources do not "
                        "establish an answer, begin exactly with: Insufficient evidence in the retrieved sources. Cite each factual claim with its "
                        "source label, such as [S1]."
                    ),
                },
                {
                    "role": "user",
                    "content": _answer_prompt(question, citations),
                },
            ],
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("The model returned an empty answer.")

        return GroundedAnswer(
            text=text,
            model=response.model,
            citations=citations,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


def _configured_chat_model() -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def _answer_prompt(question: str, citations: Sequence[SourceCitation]) -> str:
    sources = "\n\n".join(
        f"[{citation.label}] {_source_label(citation.chunk)}\n{citation.chunk.text}"
        for citation in citations
    )
    return f"Question: {question}\n\nSources:\n{sources}"


def _source_label(chunk: Chunk) -> str:
    pages = ", ".join(str(page) for page in chunk.page_numbers)
    location = f"page {pages}" if pages else chunk.heading or "source location unavailable"
    return f"{chunk.source_path.name}, {location}, chunk {chunk.chunk_id}"
