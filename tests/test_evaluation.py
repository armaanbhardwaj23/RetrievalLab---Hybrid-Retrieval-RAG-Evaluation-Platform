"""Tests for retrieval-quality metrics and durable source labels."""

from pathlib import Path

import pytest

from retrievallab.evaluation import (
    EvaluationCase,
    RelevantEvidence,
    analyze_retrieval,
    evaluate_retrieval,
    load_evaluation_cases,
)
from retrievallab.models import Chunk, RetrievalResult


def _result(page: int, rank: int) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"guide:{page:04d}",
        document_id="guide",
        text="Text",
        source_path=Path("MScAC-Handbook.pdf"),
        source_type="pdf",
        index=page,
        page_numbers=(page,),
    )
    return RetrievalResult(chunk=chunk, rank=rank, score=1.0, method="bm25")


def test_evaluation_calculates_recall_and_mrr() -> None:
    case = EvaluationCase(
        case_id="courses",
        question="What course is required?",
        relevant_evidence=(
            RelevantEvidence("MScAC-Handbook.pdf", (1,)),
            RelevantEvidence("MScAC-Handbook.pdf", (2,)),
        ),
    )

    report = evaluate_retrieval([case], lambda _: [_result(2, 1), _result(3, 2)], k=2)

    assert report.recall_at_k == pytest.approx(0.5)
    assert report.mrr == pytest.approx(1.0)
    assert report.mean_latency_ms >= 0


def test_load_evaluation_cases_uses_source_filename_and_pages(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        """[
          {
            "id": "courses",
            "question": "What course is required?",
            "relevant_evidence": [
              {"source_filename": "MScAC-Handbook.pdf", "page_numbers": [7]}
            ]
          }
        ]"""
    )

    cases = load_evaluation_cases(path)

    assert cases[0].relevant_evidence == (
        RelevantEvidence("MScAC-Handbook.pdf", (7,)),
    )


def test_retrieval_analysis_keeps_failures_and_ranked_results() -> None:
    case = EvaluationCase(
        case_id="courses",
        question="What course is required?",
        relevant_evidence=(
            RelevantEvidence("MScAC-Handbook.pdf", (1,)),
            RelevantEvidence("MScAC-Handbook.pdf", (2,)),
        ),
    )

    analysis = analyze_retrieval([case], lambda _: [_result(2, 1), _result(3, 2)], k=2)

    inspected_case = analysis.cases[0]
    assert inspected_case.evidence_found == (False, True)
    assert inspected_case.first_relevant_rank == 1
    assert analysis.failed_cases == (inspected_case,)
