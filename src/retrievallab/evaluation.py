"""Evaluate retrieval quality against manually labeled source evidence."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from retrievallab.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class RelevantEvidence:
    """A stable source location known to support an evaluation question."""

    source_filename: str
    page_numbers: tuple[int, ...] = ()

    def matches(self, result: RetrievalResult) -> bool:
        if result.chunk.source_path.name != self.source_filename:
            return False
        return not self.page_numbers or bool(
            set(self.page_numbers).intersection(result.chunk.page_numbers)
        )


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One question and the source locations that should answer it."""

    case_id: str
    question: str
    relevant_evidence: tuple[RelevantEvidence, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate retrieval metrics for one strategy and one labeled dataset."""

    case_count: int
    recall_at_k: float
    mrr: float
    mean_latency_ms: float


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Load manually authored evaluation cases from a JSON array."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation data must be a JSON array.")

    cases = tuple(
        EvaluationCase(
            case_id=item["id"],
            question=item["question"],
            relevant_evidence=tuple(
                RelevantEvidence(
                    source_filename=evidence["source_filename"],
                    page_numbers=tuple(evidence.get("page_numbers", [])),
                )
                for evidence in item["relevant_evidence"]
            ),
        )
        for item in payload
    )
    if not cases:
        raise ValueError("Evaluation data must contain at least one case.")
    return cases


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    retrieve: Callable[[str], Sequence[RetrievalResult]],
    *,
    k: int = 5,
) -> EvaluationReport:
    """Measure evidence recall, first-relevant rank, and retrieval latency."""

    if not cases:
        raise ValueError("At least one evaluation case is required.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for case in cases:
        started = perf_counter()
        results = tuple(retrieve(case.question))[:k]
        latencies_ms.append((perf_counter() - started) * 1_000)

        found = [
            any(evidence.matches(result) for result in results)
            for evidence in case.relevant_evidence
        ]
        recalls.append(sum(found) / len(case.relevant_evidence))

        first_relevant_rank = next(
            (
                result.rank
                for result in results
                if any(evidence.matches(result) for evidence in case.relevant_evidence)
            ),
            None,
        )
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)

    return EvaluationReport(
        case_count=len(cases),
        recall_at_k=sum(recalls) / len(recalls),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        mean_latency_ms=sum(latencies_ms) / len(latencies_ms),
    )


@dataclass(frozen=True, slots=True)
class CaseAnalysis:
    """The retrieved evidence and match outcome for one labeled question."""

    case: EvaluationCase
    results: tuple[RetrievalResult, ...]
    evidence_found: tuple[bool, ...]
    first_relevant_rank: int | None

    @property
    def has_missed_evidence(self) -> bool:
        """Whether at least one labeled source location was absent from top-K."""

        return not all(self.evidence_found)


@dataclass(frozen=True, slots=True)
class RetrievalAnalysis:
    """Inspectable per-case outcomes for a retrieval run."""

    cases: tuple[CaseAnalysis, ...]

    @property
    def failed_cases(self) -> tuple[CaseAnalysis, ...]:
        """Return only cases with missing labeled evidence."""

        return tuple(case for case in self.cases if case.has_missed_evidence)


def analyze_retrieval(
    cases: Sequence[EvaluationCase],
    retrieve: Callable[[str], Sequence[RetrievalResult]],
    *,
    k: int = 5,
) -> RetrievalAnalysis:
    """Return per-question evidence matches so aggregate failures can be inspected."""

    if not cases:
        raise ValueError("At least one evaluation case is required.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    analyses: list[CaseAnalysis] = []
    for case in cases:
        results = tuple(retrieve(case.question))[:k]
        evidence_found = tuple(
            any(evidence.matches(result) for result in results)
            for evidence in case.relevant_evidence
        )
        first_relevant_rank = next(
            (
                result.rank
                for result in results
                if any(evidence.matches(result) for evidence in case.relevant_evidence)
            ),
            None,
        )
        analyses.append(
            CaseAnalysis(
                case=case,
                results=results,
                evidence_found=evidence_found,
                first_relevant_rank=first_relevant_rank,
            )
        )
    return RetrievalAnalysis(cases=tuple(analyses))
