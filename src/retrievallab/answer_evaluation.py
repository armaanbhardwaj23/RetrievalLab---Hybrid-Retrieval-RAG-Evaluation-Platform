"""Small, transparent answer checks for grounded-RAG evaluation cases.

These checks are deliberately deterministic. They validate expected answer
assertions and citation-label structure; they do not claim to prove semantic
faithfulness, which still requires human or model-judge review.
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from retrievallab.evaluation import RelevantEvidence
from retrievallab.rag import GroundedAnswer

_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True, slots=True)
class AnswerEvaluationCase:
    """One question with observable answer expectations and source evidence."""

    case_id: str
    question: str
    expected_evidence: tuple[RelevantEvidence, ...]
    required_phrases: tuple[str, ...] = ()
    required_any_phrases: tuple[tuple[str, ...], ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expects_abstention: bool = False
    category: str = "general"


@dataclass(frozen=True, slots=True)
class AnswerCheckResult:
    """Deterministic checks plus data needed for manual grounding review."""

    case: AnswerEvaluationCase
    answer_text: str
    missing_required_phrases: tuple[str, ...]
    found_forbidden_phrases: tuple[str, ...]
    referenced_labels: tuple[str, ...]
    invalid_labels: tuple[str, ...]
    expected_evidence_supplied: bool
    abstention_detected: bool
    citations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether all configured deterministic checks passed."""

        return (
            not self.missing_required_phrases
            and not self.found_forbidden_phrases
            and not self.invalid_labels
            and self.expected_evidence_supplied
            and self.abstention_detected == self.case.expects_abstention
        )


@dataclass(frozen=True, slots=True)
class AnswerEvaluationReport:
    """Aggregate result for a deliberately small answer-quality smoke suite."""

    results: tuple[AnswerCheckResult, ...]

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)


def load_answer_evaluation_cases(path: Path) -> tuple[AnswerEvaluationCase, ...]:
    """Load human-authored answer checks from a JSON array."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Answer evaluation data must be a non-empty JSON array.")
    return tuple(
        AnswerEvaluationCase(
            case_id=item["id"],
            question=item["question"],
            expected_evidence=tuple(
                RelevantEvidence(
                    source_filename=evidence["source_filename"],
                    page_numbers=tuple(evidence.get("page_numbers", [])),
                )
                for evidence in item.get("expected_evidence", [])
            ),
            required_phrases=tuple(item.get("required_phrases", [])),
            required_any_phrases=tuple(
                tuple(group) for group in item.get("required_any_phrases", [])
            ),
            forbidden_phrases=tuple(item.get("forbidden_phrases", [])),
            expects_abstention=item.get("expects_abstention", False),
            category=item.get("category", "general"),
        )
        for item in payload
    )


def evaluate_answers(
    cases: Sequence[AnswerEvaluationCase],
    answer_question: Callable[[str], GroundedAnswer],
) -> AnswerEvaluationReport:
    """Run deterministic checks while retaining citations for human review."""

    if not cases:
        raise ValueError("At least one answer evaluation case is required.")
    return AnswerEvaluationReport(
        results=tuple(_check_answer(case, answer_question(case.question)) for case in cases)
    )


def _check_answer(case: AnswerEvaluationCase, answer: GroundedAnswer) -> AnswerCheckResult:
    normalized_answer = answer.text.casefold()
    missing_required = tuple(
        phrase for phrase in case.required_phrases if phrase.casefold() not in normalized_answer
    )
    missing_required += tuple(
        " OR ".join(group)
        for group in case.required_any_phrases
        if not any(phrase.casefold() in normalized_answer for phrase in group)
    )
    found_forbidden = tuple(
        phrase for phrase in case.forbidden_phrases if phrase.casefold() in normalized_answer
    )
    labels = tuple(f"S{number}" for number in _CITATION_PATTERN.findall(answer.text))
    known_labels = {citation.label for citation in answer.citations}
    invalid_labels = tuple(label for label in labels if label not in known_labels)
    expected_evidence_supplied = all(
        any(
            citation.chunk.source_path.name == evidence.source_filename
            and (
                not evidence.page_numbers
                or bool(set(citation.chunk.page_numbers).intersection(evidence.page_numbers))
            )
            for citation in answer.citations
        )
        for evidence in case.expected_evidence
    )
    abstention_detected = "insufficient evidence" in normalized_answer
    citations = tuple(
        f"[{citation.label}] {citation.chunk.source_path.name}, pages "
        f"{', '.join(map(str, citation.chunk.page_numbers)) or 'unknown'}"
        for citation in answer.citations
    )
    return AnswerCheckResult(
        case=case,
        answer_text=answer.text,
        missing_required_phrases=missing_required,
        found_forbidden_phrases=found_forbidden,
        referenced_labels=labels,
        invalid_labels=invalid_labels,
        expected_evidence_supplied=expected_evidence_supplied,
        abstention_detected=abstention_detected,
        citations=citations,
    )


def save_answer_evaluation_report(report: AnswerEvaluationReport, path: Path) -> None:
    """Persist deterministic answer checks for later review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_count": len(report.results),
        "passed_count": report.passed_count,
        "results": [
            {
                "case_id": result.case.case_id,
                "category": result.case.category,
                "question": result.case.question,
                "answer": result.answer_text,
                "passed": result.passed,
                "missing_required_phrases": list(result.missing_required_phrases),
                "found_forbidden_phrases": list(result.found_forbidden_phrases),
                "invalid_labels": list(result.invalid_labels),
                "expected_evidence_supplied": result.expected_evidence_supplied,
                "abstention_detected": result.abstention_detected,
                "citations": list(result.citations),
            }
            for result in report.results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
