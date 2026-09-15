"""Model-assisted claim-to-citation verification for grounded answers.

A verifier judges whether each factual claim carrying a source label is supported
by the corresponding supplied chunk. Its verdict is a review signal, not proof
of factual truth.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from retrievallab.answer_evaluation import AnswerEvaluationCase

from dotenv import load_dotenv

from retrievallab.openrouter import openrouter_client
from retrievallab.rag import GroundedAnswer, _source_label

CitationVerdict = Literal["supported", "unsupported", "unclear"]
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"


@dataclass(frozen=True, slots=True)
class ClaimCitationReview:
    """One model-judged claim/citation relationship."""

    claim: str
    labels: tuple[str, ...]
    verdict: CitationVerdict
    rationale: str


@dataclass(frozen=True, slots=True)
class CitationVerificationReport:
    """Structured evidence review retained alongside the generated answer."""

    reviews: tuple[ClaimCitationReview, ...]
    model: str

    @property
    def supported_count(self) -> int:
        return sum(review.verdict == "supported" for review in self.reviews)


class OpenRouterCitationVerifier:
    """Ask a model to assess only the answer claims that cite supplied chunks."""

    def __init__(
        self, *, api_key: str | None = None, model: str | None = None, client: Any | None = None
    ) -> None:
        self.model = model or _configured_judge_model()
        self._client = client or openrouter_client(api_key)

    def verify(self, answer: GroundedAnswer) -> CitationVerificationReport:
        """Return structured support judgments; reject malformed judge output."""

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict citation verifier. Inspect only factual claims "
                        "in the answer that have [S#] labels. For each, decide whether the "
                        "listed cited source text supports it. Return JSON only, with "
                        '{"reviews":[{"claim":"...","labels":["S1"],"verdict":' 
                        '"supported|unsupported|unclear","rationale":"..."}]}.'
                    ),
                },
                {"role": "user", "content": _verification_prompt(answer)},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The citation verifier returned an empty response.")
        return CitationVerificationReport(
            reviews=_parse_reviews(content, {citation.label for citation in answer.citations}),
            model=response.model,
        )


def _configured_judge_model() -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _verification_prompt(answer: GroundedAnswer) -> str:
    sources = "\n\n".join(
        f"[{citation.label}] {_source_label(citation.chunk)}\n{citation.chunk.text}"
        for citation in answer.citations
    )
    return f"Answer:\n{answer.text}\n\nCited source passages:\n{sources}"


def _parse_reviews(content: str, known_labels: set[str]) -> tuple[ClaimCitationReview, ...]:
    try:
        payload = json.loads(content)
        raw_reviews = payload["reviews"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("Citation verifier returned invalid JSON.") from error
    if not isinstance(raw_reviews, list):
        raise RuntimeError("Citation verifier reviews must be a JSON array.")

    reviews: list[ClaimCitationReview] = []
    for item in raw_reviews:
        labels = tuple(item["labels"])
        verdict = item["verdict"]
        if not item["claim"].strip() or not labels or not set(labels).issubset(known_labels):
            raise RuntimeError("Citation verifier returned an invalid claim or source label.")
        if verdict not in {"supported", "unsupported", "unclear"}:
            raise RuntimeError("Citation verifier returned an invalid verdict.")
        reviews.append(
            ClaimCitationReview(
                claim=item["claim"], labels=labels, verdict=verdict, rationale=item["rationale"]
            )
        )
    return tuple(reviews)


@dataclass(frozen=True, slots=True)
class CitationVerificationRecord:
    """One answer and its verifier output, suitable for durable JSON reporting."""

    case: AnswerEvaluationCase
    answer: GroundedAnswer
    report: CitationVerificationReport


def save_citation_verification_records(
    records: tuple[CitationVerificationRecord, ...], path: Path
) -> None:
    """Write completed cases after each verification so interrupted runs retain evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_count": len(records),
        "records": [
            {
                "case_id": record.case.case_id,
                "category": record.case.category,
                "question": record.case.question,
                "answer": record.answer.text,
                "answer_model": record.answer.model,
                "citation_verifier_model": record.report.model,
                "sources": [
                    {
                        "label": citation.label,
                        "source_filename": citation.chunk.source_path.name,
                        "page_numbers": list(citation.chunk.page_numbers),
                        "chunk_id": citation.chunk.chunk_id,
                    }
                    for citation in record.answer.citations
                ],
                "reviews": [
                    {
                        "claim": review.claim,
                        "labels": list(review.labels),
                        "verdict": review.verdict,
                        "rationale": review.rationale,
                    }
                    for review in record.report.reviews
                ],
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
