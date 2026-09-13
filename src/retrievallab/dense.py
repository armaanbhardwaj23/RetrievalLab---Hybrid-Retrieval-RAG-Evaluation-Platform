"""Exact, inspectable dense retrieval over stored chunk embeddings."""

import math
from collections.abc import Sequence

from retrievallab.embeddings import OpenRouterEmbedder
from retrievallab.models import Chunk, ChunkEmbedding, RetrievalResult


class DenseRetriever:
    """Rank every indexed chunk by cosine similarity to a query embedding."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[ChunkEmbedding],
    ) -> None:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        embeddings_by_id = {embedding.chunk_id: embedding for embedding in embeddings}

        if len(chunks_by_id) != len(chunks):
            raise ValueError("Chunk IDs must be unique.")
        if len(embeddings_by_id) != len(embeddings):
            raise ValueError("Embedding chunk IDs must be unique.")
        if chunks_by_id.keys() != embeddings_by_id.keys():
            raise ValueError("Chunks and embeddings must have matching chunk IDs.")
        if not embeddings:
            raise ValueError("At least one embedding is required.")

        models = {embedding.model for embedding in embeddings}
        dimensions = {len(embedding.vector) for embedding in embeddings}
        if len(models) != 1:
            raise ValueError("All chunk embeddings must use the same model.")
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError("All chunk embeddings need one non-empty vector dimension.")

        self._chunks_by_id = chunks_by_id
        self._embeddings_by_id = embeddings_by_id
        self._model = next(iter(models))

    def search(
        self,
        query: str,
        *,
        embedder: OpenRouterEmbedder,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Embed one query and return its highest-scoring chunks."""

        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_batch = embedder.embed_texts([query])
        if query_batch.model != self._model:
            raise ValueError(
                "Query embedding model does not match indexed chunk embeddings."
            )

        query_vector = query_batch.vectors[0]
        scores = [
            (
                chunk_id,
                cosine_similarity(query_vector, embedding.vector),
            )
            for chunk_id, embedding in self._embeddings_by_id.items()
        ]
        scores.sort(key=lambda item: (-item[1], item[0]))

        return tuple(
            RetrievalResult(
                chunk=self._chunks_by_id[chunk_id],
                rank=rank,
                score=score,
                method="dense",
            )
            for rank, (chunk_id, score) in enumerate(scores[:top_k], start=1)
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, rejecting incompatible or zero vectors."""

    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension.")
    if not left:
        raise ValueError("Vectors cannot be empty.")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return dot_product / (left_norm * right_norm)
