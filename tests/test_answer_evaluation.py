from pathlib import Path

from retrievallab.answer_evaluation import (
    AnswerEvaluationCase,
    evaluate_answers,
    load_answer_evaluation_cases,
)
from retrievallab.evaluation import RelevantEvidence
from retrievallab.models import Chunk
from retrievallab.rag import GroundedAnswer, SourceCitation


def _answer(text: str, *, page: int = 19) -> GroundedAnswer:
    chunk = Chunk(
        chunk_id="guide:0019",
        document_id="guide",
        text="Registration steps.",
        source_path=Path("guide.pdf"),
        source_type="pdf",
        index=19,
        page_numbers=(page,),
    )
    return GroundedAnswer(
        text=text,
        model="test-model",
        citations=(SourceCitation("S1", chunk),),
        input_tokens=1,
        output_tokens=1,
    )


def test_answer_checks_required_text_and_citation_labels() -> None:
    case = AnswerEvaluationCase(
        case_id="registration",
        question="How do I register?",
        expected_evidence=(RelevantEvidence("guide.pdf", (19,)),),
        required_phrases=("documentation",),
    )

    report = evaluate_answers([case], lambda _: _answer("Provide documentation [S1]."))

    result = report.results[0]
    assert result.passed
    assert result.answer_text == "Provide documentation [S1]."
    assert result.referenced_labels == ("S1",)
    assert result.expected_evidence_supplied


def test_answer_check_accepts_one_phrase_from_a_requirement_group() -> None:
    case = AnswerEvaluationCase(
        case_id="academic-standard",
        question="Must standards remain the same?",
        expected_evidence=(RelevantEvidence("guide.pdf", (19,)),),
        required_any_phrases=(("same level", "same course standards"),),
    )

    result = evaluate_answers(
        [case], lambda _: _answer("Students meet the same course standards [S1].")
    ).results[0]

    assert result.passed
    assert not result.missing_required_phrases


def test_answer_check_fails_unknown_citation_and_missing_abstention() -> None:
    case = AnswerEvaluationCase(
        case_id="unknown",
        question="Unknown question?",
        expected_evidence=(),
        expects_abstention=True,
    )

    result = evaluate_answers([case], lambda _: _answer("I do not know. [S2]")).results[0]

    assert not result.passed
    assert result.invalid_labels == ("S2",)
    assert not result.abstention_detected


def test_load_answer_evaluation_cases(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text('[{"id":"one","question":"Q?","expected_evidence":[]}]')

    cases = load_answer_evaluation_cases(path)

    assert cases[0].case_id == "one"
    assert cases[0].category == "general"


def test_save_answer_evaluation_report_writes_check_details(tmp_path: Path) -> None:
    from retrievallab.answer_evaluation import save_answer_evaluation_report
    import json

    case = AnswerEvaluationCase("registration", "How do I register?", (), required_phrases=("documentation",))
    report = evaluate_answers([case], lambda _: _answer("Provide documentation [S1]."))
    output = tmp_path / "answer_report.json"

    save_answer_evaluation_report(report, output)

    payload = json.loads(output.read_text())
    assert payload["passed_count"] == 1
    assert payload["results"][0]["answer"] == "Provide documentation [S1]."
