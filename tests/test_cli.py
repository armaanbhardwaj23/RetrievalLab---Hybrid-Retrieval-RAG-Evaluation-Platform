"""Tests for the public command-line interface configuration."""

from pathlib import Path

from retrievallab.cli import _build_parser, _source_paths


def test_search_defaults_to_hybrid_retrieval() -> None:
    args = _build_parser().parse_args(["search", "What courses are required?"])

    assert args.strategy == "hybrid"
    assert args.top_k == 5
    assert args.index_path == Path("data/processed/dense_index.json")


def test_source_paths_discovers_only_supported_files(tmp_path: Path) -> None:
    (tmp_path / "guide.pdf").write_bytes(b"PDF")
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "ignored.html").write_text("<h1>Ignored</h1>")

    assert [path.name for path in _source_paths(tmp_path)] == ["guide.pdf", "notes.md"]
