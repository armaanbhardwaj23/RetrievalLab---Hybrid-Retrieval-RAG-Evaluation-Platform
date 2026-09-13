"""Command-line entry point for indexing, searching, and grounded answers."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from retrievallab.bm25 import Bm25Retriever
from retrievallab.dense import DenseRetriever
from retrievallab.embeddings import OpenRouterEmbedder
from retrievallab.evaluation import evaluate_retrieval, load_evaluation_cases
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
