"""Tests for exact dense retrieval and cosine similarity."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from retrievallab.dense import DenseRetriever, cosine_similarity
from retrievallab.embeddings import EmbeddingBatch
from retrievallab.models import Chunk, ChunkEmbedding


class FakeEmbedder:
    def __init__(self, vector: tuple[float, ...], model: str = "test-model") -> None:
        self.vector = vector
        self.model = model

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        assert texts == ["refund deadline"]
        return EmbeddingBatch(model=self.model, vectors=(self.vector,), input_tokens=2)


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="guide",
        text=chunk_id,
        source_path=Path("guide.pdf"),
        source_type="pdf",
        index=int(chunk_id[-1]),
    )


def test_cosine_similarity_ranks_aligned_vectors_higher() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_dense_retrieval_returns_ranked_results() -> None:
    chunks = [_chunk("guide:0000"), _chunk("guide:0001")]
    embeddings = [
        ChunkEmbedding("guide:0000", "test-model", (0.9, 0.1)),
        ChunkEmbedding("guide:0001", "test-model", (0.1, 0.9)),
    ]
    retriever = DenseRetriever(chunks, embeddings)

    results = retriever.search(
        "refund deadline", embedder=FakeEmbedder((1.0, 0.0)), top_k=1
    )

    assert [(result.chunk.chunk_id, result.rank, result.method) for result in results] == [
        ("guide:0000", 1, "dense")
    ]
    assert results[0].score == pytest.approx(0.9938837)


def test_dense_retrieval_rejects_model_mismatch() -> None:
    chunks = [_chunk("guide:0000")]
    embeddings = [ChunkEmbedding("guide:0000", "test-model", (1.0, 0.0))]
    retriever = DenseRetriever(chunks, embeddings)

    with pytest.raises(ValueError, match="does not match"):
        retriever.search(
            "refund deadline", embedder=FakeEmbedder((1.0, 0.0), "other-model")
        )
