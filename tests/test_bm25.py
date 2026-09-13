"""Tests for transparent BM25 retrieval behavior."""

from pathlib import Path

import pytest

from retrievallab.bm25 import Bm25Retriever, tokenize
from retrievallab.models import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="guide",
        text=text,
        source_path=Path("guide.pdf"),
        source_type="pdf",
        index=int(chunk_id[-1]),
    )


def test_tokenize_preserves_course_codes_and_underscores() -> None:
    assert tokenize("CSC2701H and ERR_AUTH_104") == ("csc2701h", "err_auth_104")


def test_bm25_ranks_exact_course_code_match_first() -> None:
    retriever = Bm25Retriever(
        [
            _chunk("guide:0000", "CSC2701H is a required communication course."),
            _chunk("guide:0001", "The program includes elective courses."),
        ]
    )

    results = retriever.search("What is CSC2701H?", top_k=1)

    assert results[0].chunk.chunk_id == "guide:0000"
    assert results[0].method == "bm25"


def test_bm25_rejects_a_query_without_searchable_terms() -> None:
    retriever = Bm25Retriever([_chunk("guide:0000", "Course information")])

    with pytest.raises(ValueError, match="searchable"):
        retriever.search("?!")
