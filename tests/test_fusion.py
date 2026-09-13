"""Tests for Reciprocal Rank Fusion."""

from pathlib import Path

import pytest

from retrievallab.fusion import reciprocal_rank_fusion
from retrievallab.models import Chunk, RetrievalMethod, RetrievalResult


def _result(chunk_id: str, rank: int, method: RetrievalMethod) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="guide",
        text=chunk_id,
        source_path=Path("guide.pdf"),
        source_type="pdf",
        index=int(chunk_id[-1]),
    )
    return RetrievalResult(chunk=chunk, rank=rank, score=float(rank), method=method)


def test_rrf_promotes_a_chunk_ranked_by_both_retrievers() -> None:
    dense = [_result("guide:0000", 1, "dense"), _result("guide:0001", 2, "dense")]
    bm25 = [_result("guide:0001", 1, "bm25"), _result("guide:0002", 2, "bm25")]

    fused = reciprocal_rank_fusion([dense, bm25], rank_constant=60, top_k=3)

    assert [result.chunk.chunk_id for result in fused] == [
        "guide:0001",
        "guide:0000",
        "guide:0002",
    ]
    assert all(result.method == "hybrid" for result in fused)


def test_rrf_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="rank_constant"):
        reciprocal_rank_fusion([], rank_constant=0)
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=0)
