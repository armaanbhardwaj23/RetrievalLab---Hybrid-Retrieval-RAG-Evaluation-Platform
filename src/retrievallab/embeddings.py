"""Hosted embedding support through OpenRouter's embeddings endpoint."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from retrievallab.models import Chunk, ChunkEmbedding
from retrievallab.openrouter import openrouter_client


DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors and usage returned from one embeddings API request."""

    model: str
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int


class OpenRouterEmbedder:
    """Embed text with a configurable OpenRouter-hosted model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client or openrouter_client(api_key)

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed text in one batch and preserve the caller's input ordering."""

        if not texts:
            raise ValueError("texts must contain at least one item.")
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty or whitespace-only text.")

        response = self._client.embeddings.create(
            input=list(texts),
            model=self.model,
            encoding_format="float",
        )
        items = sorted(response.data, key=lambda item: item.index)
        vectors = tuple(tuple(item.embedding) for item in items)

        if len(vectors) != len(texts):
            raise RuntimeError("Embedding response count did not match input count.")
        if not vectors or any(not vector for vector in vectors):
            raise RuntimeError("Embedding response contained an empty vector.")

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError("Embedding response used inconsistent vector dimensions.")

        return EmbeddingBatch(
            model=response.model,
            vectors=vectors,
            input_tokens=response.usage.prompt_tokens,
        )

    def embed_chunks(self, chunks: Sequence[Chunk]) -> tuple[ChunkEmbedding, ...]:
        """Embed chunks while keeping each vector tied to its stable chunk ID."""

        batch = self.embed_texts([chunk.text for chunk in chunks])
        return tuple(
            ChunkEmbedding(chunk_id=chunk.chunk_id, model=batch.model, vector=vector)
            for chunk, vector in zip(chunks, batch.vectors, strict=True)
        )
