from pathlib import Path

import pytest

from retrievallab.deduplication import find_near_duplicate_chunks
from retrievallab.indexing import DenseIndex
from retrievallab.models import Chunk, ChunkEmbedding


def _index() -> DenseIndex:
    chunks = tuple(
        Chunk(f"guide:{index:04d}", "guide", f"Text {index}", Path("guide.pdf"), "pdf", index)
        for index in range(3)
    )
    embeddings = (
        ChunkEmbedding("guide:0000", "model", (1.0, 0.0)),
        ChunkEmbedding("guide:0001", "model", (0.99, 0.01)),
        ChunkEmbedding("guide:0002", "model", (0.0, 1.0)),
    )
    return DenseIndex(chunks, embeddings)


def test_duplicate_analysis_returns_only_pairs_above_threshold() -> None:
    pairs = find_near_duplicate_chunks(_index(), similarity_threshold=0.95)

    assert len(pairs) == 1
    assert pairs[0].first.chunk_id == "guide:0000"
    assert pairs[0].second.chunk_id == "guide:0001"
    assert pairs[0].similarity == pytest.approx(1.0, abs=0.001)


def test_duplicate_analysis_validates_threshold() -> None:
    with pytest.raises(ValueError, match="between"):
        find_near_duplicate_chunks(_index(), similarity_threshold=1.1)
