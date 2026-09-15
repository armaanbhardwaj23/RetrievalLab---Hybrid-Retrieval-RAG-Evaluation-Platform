import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrievallab.citation_verification import OpenRouterCitationVerifier
from retrievallab.models import Chunk
from retrievallab.rag import GroundedAnswer, SourceCitation


class FakeCompletionsEndpoint:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            model="judge-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )


def _answer() -> GroundedAnswer:
    chunk = Chunk(
        chunk_id="guide:0007", document_id="guide", text="The program lasts 16 months.",
        source_path=Path("guide.pdf"), source_type="pdf", index=7, page_numbers=(7,),
    )
    return GroundedAnswer(
        text="The program lasts 16 months [S1].", model="answer-model",
        citations=(SourceCitation("S1", chunk),), input_tokens=1, output_tokens=1,
    )


def test_citation_verifier_parses_structured_supported_claim() -> None:
    content = json.dumps({"reviews": [{
        "claim": "The program lasts 16 months.", "labels": ["S1"],
        "verdict": "supported", "rationale": "The source explicitly says so."
    }]})
    endpoint = FakeCompletionsEndpoint(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))

    report = OpenRouterCitationVerifier(client=client, model="judge-model").verify(_answer())

    assert report.supported_count == 1
    assert report.reviews[0].verdict == "supported"
    assert "Cited source passages" in endpoint.request["messages"][1]["content"]


def test_citation_verifier_rejects_unknown_source_label() -> None:
    content = json.dumps({"reviews": [{
        "claim": "The program lasts 16 months.", "labels": ["S2"],
        "verdict": "supported", "rationale": "Wrong label."
    }]})
    endpoint = FakeCompletionsEndpoint(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))

    with pytest.raises(RuntimeError, match="source label"):
        OpenRouterCitationVerifier(client=client).verify(_answer())


def test_save_citation_verification_records_writes_answer_and_reviews(tmp_path: Path) -> None:
    from retrievallab.answer_evaluation import AnswerEvaluationCase
    from retrievallab.citation_verification import (
        CitationVerificationRecord,
        CitationVerificationReport,
        ClaimCitationReview,
        save_citation_verification_records,
    )

    record = CitationVerificationRecord(
        case=AnswerEvaluationCase("duration", "How long?", (), category="lookup"),
        answer=_answer(),
        report=CitationVerificationReport(
            reviews=(ClaimCitationReview("The program lasts 16 months.", ("S1",), "supported", "Explicit."),),
            model="judge-model",
        ),
    )
    output = tmp_path / "report.json"

    save_citation_verification_records((record,), output)

    payload = json.loads(output.read_text())
    assert payload["case_count"] == 1
    assert payload["records"][0]["sources"][0]["page_numbers"] == [7]
    assert payload["records"][0]["reviews"][0]["verdict"] == "supported"
