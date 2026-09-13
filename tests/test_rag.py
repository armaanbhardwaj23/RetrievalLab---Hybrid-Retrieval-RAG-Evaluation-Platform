"""Tests for grounded answer prompts and source citations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from retrievallab.models import Chunk, RetrievalResult
from retrievallab.rag import OpenRouterAnswerGenerator


class FakeCompletionsEndpoint:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            model="test-chat-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer. [S1]"))],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=4),
        )


def _result() -> RetrievalResult:
    chunk = Chunk(
        chunk_id="guide:0007",
        document_id="guide",
        text="CSC2701H is required.",
        source_path=Path("guide.pdf"),
        source_type="pdf",
        index=7,
        page_numbers=(7,),
    )
    return RetrievalResult(chunk=chunk, rank=1, score=0.8, method="hybrid")


def test_answer_generation_labels_sources_for_citation() -> None:
    endpoint = FakeCompletionsEndpoint()
    client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    generator = OpenRouterAnswerGenerator(
        api_key="test-key", model="test-chat-model", client=client
    )

    answer = generator.answer("What course is required?", [_result()])

    assert answer.text == "Answer. [S1]"
    assert answer.citations[0].label == "S1"
    assert "[S1] guide.pdf, page 7, chunk guide:0007" in endpoint.request["messages"][1]["content"]


def test_answer_generation_requires_retrieved_sources() -> None:
    generator = OpenRouterAnswerGenerator(api_key="test-key", client=SimpleNamespace())

    with pytest.raises(ValueError, match="retrieved"):
        generator.answer("What course is required?", [])
