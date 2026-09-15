"""Command-line entry point for indexing, searching, and grounded answers."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from retrievallab.answer_evaluation import (
    evaluate_answers,
    load_answer_evaluation_cases,
    save_answer_evaluation_report,
)
from retrievallab.citation_verification import (
    CitationVerificationRecord,
    OpenRouterCitationVerifier,
    save_citation_verification_records,
)
from retrievallab.bm25 import Bm25Retriever
from retrievallab.deduplication import find_near_duplicate_chunks
from retrievallab.dense import DenseRetriever
from retrievallab.embeddings import OpenRouterEmbedder
from retrievallab.evaluation import analyze_retrieval, evaluate_retrieval, load_evaluation_cases
from retrievallab.fusion import reciprocal_rank_fusion
from retrievallab.indexing import build_dense_index, load_dense_index, save_dense_index
from retrievallab.ingest import ingest_path
from retrievallab.models import Chunk, ChunkEmbedding, RetrievalResult
from retrievallab.rag import OpenRouterAnswerGenerator
from retrievallab.tool_calling import ToolCallingAnswerGenerator
from retrievallab.vector_store import ChromaVectorStore


DEFAULT_SOURCE_DIRECTORY = Path("data/raw")
DEFAULT_INDEX_PATH = Path("data/processed/dense_index.json")
DEFAULT_CHROMA_DIRECTORY = Path("data/processed/chroma")
DEFAULT_EVALUATION_PATH = Path("data/evaluation/questions.json")
SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".txt"})


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RetrievalLab command-line interface."""

    args = _build_parser().parse_args(argv)
    if args.command == "index":
        return _index_sources(args.source_directory, args.index_path)
    if args.command == "vector-index":
        index = load_dense_index(args.index_path)
        ChromaVectorStore.create_from_index(
            index, persist_directory=args.chroma_directory, overwrite=args.overwrite
        )
        print(f"Stored {len(index.chunks)} vectors in Chroma: {args.chroma_directory}")
        return 0
    if args.command == "duplicates":
        index = load_dense_index(args.index_path)
        pairs = find_near_duplicate_chunks(index, similarity_threshold=args.threshold)
        print(f"Candidate pairs: {len(pairs)}")
        for pair in pairs:
            print(f"{pair.similarity:.4f} {_source_description(pair.first)} <-> {_source_description(pair.second)}")
        return 0
    if args.command == "vector-search":
        results = ChromaVectorStore.open(
            persist_directory=args.chroma_directory
        ).search(args.question, embedder=OpenRouterEmbedder(), top_k=args.top_k)
        _print_results(results)
        return 0

    index = load_dense_index(args.index_path)
    if args.command == "evaluate":
        cases = load_evaluation_cases(args.evaluation_path)
        report = evaluate_retrieval(
            cases,
            lambda question: _retrieve(
                index.chunks, index.embeddings, question, args.strategy, args.top_k
            ),
            k=args.top_k,
        )
        print(f"Cases: {report.case_count}")
        print(f"Recall@{args.top_k}: {report.recall_at_k:.3f}")
        print(f"MRR: {report.mrr:.3f}")
        print(f"Mean retrieval latency: {report.mean_latency_ms:.1f} ms")
        return 0
    if args.command == "evaluate-answers":
        cases = load_answer_evaluation_cases(args.answer_evaluation_path)
        generator = OpenRouterAnswerGenerator()
        report = evaluate_answers(
            cases,
            lambda question: generator.answer(
                question,
                _retrieve(index.chunks, index.embeddings, question, args.strategy, args.top_k),
            ),
        )
        save_answer_evaluation_report(report, args.output)
        _print_answer_evaluation(report)
        return 0

    if args.command == "verify-citations":
        cases = load_answer_evaluation_cases(args.answer_evaluation_path)
        generator = OpenRouterAnswerGenerator()
        verifier = OpenRouterCitationVerifier()
        records: list[CitationVerificationRecord] = []
        for case in cases:
            answer = generator.answer(
                case.question,
                _retrieve(index.chunks, index.embeddings, case.question, args.strategy, args.top_k),
            )
            report = verifier.verify(answer)
            records.append(CitationVerificationRecord(case=case, answer=answer, report=report))
            save_citation_verification_records(tuple(records), args.output)
            print(f"\n[{case.case_id}] {len(report.reviews)} cited claims reviewed")
            for review in report.reviews:
                print(f"  {review.verdict}: {review.claim} ({", ".join(review.labels)})")
        return 0

    if args.command == "analyze":
        cases = load_evaluation_cases(args.evaluation_path)
        analysis = analyze_retrieval(
            cases,
            lambda question: _retrieve(
                index.chunks, index.embeddings, question, args.strategy, args.top_k
            ),
            k=args.top_k,
        )
        _print_analysis(analysis)
        return 0

    if args.command == "answer" and args.mode == "tool":
        answer = ToolCallingAnswerGenerator().answer(
            args.question,
            lambda query: _retrieve(
                index.chunks, index.embeddings, query, args.strategy, args.top_k
            ),
        )
        print(answer.text)
        print("\nSources:")
        for citation in answer.citations:
            print(f"[{citation.label}] {_source_description(citation.chunk)}")
        print(f"\nModel: {answer.model}")
        print(f"Tokens: input={answer.input_tokens}, output={answer.output_tokens}")
        return 0


    results = _retrieve(index.chunks, index.embeddings, args.question, args.strategy, args.top_k)
    if args.command == "search":
        _print_results(results)
        return 0

    answer = OpenRouterAnswerGenerator().answer(args.question, results)
    print(answer.text)
    print("\nSources:")
    for citation in answer.citations:
        print(f"[{citation.label}] {_source_description(citation.chunk)}")
    print(f"\nModel: {answer.model}")
    print(f"Tokens: input={answer.input_tokens}, output={answer.output_tokens}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect hybrid RAG retrieval.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index = subcommands.add_parser("index", help="Chunk and embed supported raw documents.")
    index.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE_DIRECTORY)
    index.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)

    vector_index = subcommands.add_parser(
        "vector-index", help="Persist the existing dense vectors in Chroma without re-embedding."
    )
    vector_index.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    vector_index.add_argument("--chroma-directory", type=Path, default=DEFAULT_CHROMA_DIRECTORY)
    vector_index.add_argument("--overwrite", action="store_true")

    duplicates = subcommands.add_parser(
        "duplicates", help="Inspect likely duplicate chunks without changing the index."
    )
    duplicates.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    duplicates.add_argument("--threshold", type=float, default=0.95)

    vector_search = subcommands.add_parser(
        "vector-search", help="Search the persisted Chroma dense-vector collection."
    )
    vector_search.add_argument("question")
    vector_search.add_argument("--chroma-directory", type=Path, default=DEFAULT_CHROMA_DIRECTORY)
    vector_search.add_argument("--top-k", type=int, default=5)

    command_help = {
        "search": "Search the saved index and display ranked chunks.",
        "answer": "Generate a cited answer from retrieved chunks.",
    }
    for command, help_text in command_help.items():
        query = subcommands.add_parser(command, help=help_text)
        query.add_argument("question")
        query.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
        query.add_argument("--top-k", type=int, default=5)
        query.add_argument(
            "--strategy",
            choices=("dense", "bm25", "hybrid"),
            default="hybrid",
            help="Retrieval strategy. Answer generation uses the chosen strategy.",
        )
        if command == "answer":
            query.add_argument(
                "--mode", choices=("direct", "tool"), default="direct",
                help="direct retrieves first; tool lets the model make one bounded search call.",
            )

    evaluate = subcommands.add_parser(
        "evaluate", help="Measure one strategy against labeled source evidence."
    )
    evaluate.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    evaluate.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument(
        "--strategy",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
    )
    analyze = subcommands.add_parser(
        "analyze", help="Show the top-K evidence for every failed evaluation case."
    )
    analyze.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    analyze.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    analyze.add_argument("--top-k", type=int, default=5)
    analyze.add_argument("--strategy", choices=("dense", "bm25", "hybrid"), default="hybrid")
    answer_evaluation = subcommands.add_parser(
        "evaluate-answers",
        help="Run deterministic answer and citation-label checks; review citations manually.",
    )
    answer_evaluation.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    answer_evaluation.add_argument(
        "--answer-evaluation-path", type=Path, default=Path("data/evaluation/answer_questions.json")
    )
    answer_evaluation.add_argument("--top-k", type=int, default=5)
    answer_evaluation.add_argument(
        "--output", type=Path, default=Path("data/processed/answer_evaluation.json")
    )
    answer_evaluation.add_argument(
        "--strategy", choices=("dense", "bm25", "hybrid"), default="hybrid"
    )
    citation_verification = subcommands.add_parser(
        "verify-citations", help="Generate answers and have a judge review cited claims."
    )
    citation_verification.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    citation_verification.add_argument(
        "--answer-evaluation-path", type=Path, default=Path("data/evaluation/answer_questions.json")
    )
    citation_verification.add_argument("--top-k", type=int, default=5)
    citation_verification.add_argument(
        "--output", type=Path, default=Path("data/processed/citation_verification.json")
    )
    citation_verification.add_argument(
        "--strategy", choices=("dense", "bm25", "hybrid"), default="hybrid"
    )
    return parser


def _index_sources(source_directory: Path, index_path: Path) -> int:
    paths = _source_paths(source_directory)
    documents = [ingest_path(path) for path in paths]
    index = build_dense_index(documents, embedder=OpenRouterEmbedder())
    save_dense_index(index, index_path)
    print(f"Indexed {len(paths)} source files into {len(index.chunks)} chunks.")
    print(f"Saved: {index_path}")
    return 0


def _source_paths(source_directory: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            path
            for path in source_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"No supported documents found in {source_directory}.")
    return paths


def _retrieve(
    chunks: Sequence[Chunk],
    embeddings: Sequence[ChunkEmbedding],
    question: str,
    strategy: str,
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    if strategy == "bm25":
        return Bm25Retriever(chunks).search(question, top_k=top_k)

    dense_results = DenseRetriever(chunks, embeddings).search(
        question,
        embedder=OpenRouterEmbedder(),
        top_k=max(top_k, 10),
    )
    if strategy == "dense":
        return dense_results[:top_k]

    bm25_results = Bm25Retriever(chunks).search(question, top_k=max(top_k, 10))
    return reciprocal_rank_fusion([dense_results, bm25_results], top_k=top_k)


def _print_results(results: Sequence[RetrievalResult]) -> None:
    for result in results:
        print(f"\n#{result.rank} {result.method} score={result.score:.4f}")
        print(_source_description(result.chunk))
        print(result.chunk.text[:500])


def _source_description(chunk: Chunk) -> str:
    pages = ", ".join(str(page) for page in chunk.page_numbers)
    location = f"page(s) {pages}" if pages else chunk.heading or "source location unavailable"
    return f"{chunk.source_path.name}, {location}, chunk {chunk.chunk_id}"


def _print_analysis(analysis: object) -> None:
    """Print enough provenance to diagnose misses without hiding ranked results."""

    print(f"Cases inspected: {len(analysis.cases)}")
    print(f"Failed cases: {len(analysis.failed_cases)}")
    if not analysis.failed_cases:
        print("No labeled evidence was missed in the requested top-K.")
        return

    for case_analysis in analysis.failed_cases:
        print(f"\n[{case_analysis.case.case_id}] {case_analysis.case.question}")
        missing_evidence = (
            evidence
            for evidence, found in zip(
                case_analysis.case.relevant_evidence, case_analysis.evidence_found
            )
            if not found
        )
        for evidence in missing_evidence:
            pages = ", ".join(str(page) for page in evidence.page_numbers) or "any page"
            print(f"Expected but not retrieved: {evidence.source_filename}, page(s) {pages}")
        print("Top results:")
        for result in case_analysis.results:
            print(
                f"  #{result.rank} score={result.score:.4f} "
                f"{_source_description(result.chunk)}"
            )


def _print_answer_evaluation(report: object) -> None:
    """Print deterministic checks and citations that require human review."""

    print(f"Cases: {len(report.results)}")
    print(f"Deterministic checks passed: {report.passed_count}/{len(report.results)}")
    for result in report.results:
        status = "PASS" if result.passed else "CHECK"
        print(f"\n[{status}] {result.case.category} / {result.case.case_id}: {result.case.question}")
        print(f"Answer: {result.answer_text}")
        if result.missing_required_phrases:
            print(f"Missing required phrases: {', '.join(result.missing_required_phrases)}")
        if result.found_forbidden_phrases:
            print(f"Forbidden phrases found: {', '.join(result.found_forbidden_phrases)}")
        if result.invalid_labels:
            print(f"Unknown citation labels: {', '.join(result.invalid_labels)}")
        if not result.expected_evidence_supplied:
            print("Expected evidence was not among supplied citations.")
        if result.abstention_detected != result.case.expects_abstention:
            print(f"Abstention detected: {result.abstention_detected}")
        print("Human citation review:")
        for citation in result.citations:
            print(f"  {citation}")
