"""Tests for OpenRouter embedding requests without calling the network."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from retrievallab.embeddings import DEFAULT_EMBEDDING_MODEL, OpenRouterEmbedder
from retrievallab.models import Chunk


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            model="openai/text-embedding-3-small",
            data=[
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            ],
            usage=SimpleNamespace(prompt_tokens=7),
        )


def test_embed_texts_preserves_input_order_and_usage() -> None:
    endpoint = FakeEmbeddingsEndpoint()
    client = SimpleNamespace(embeddings=endpoint)
    embedder = OpenRouterEmbedder(api_key="test-key", client=client)

    batch = embedder.embed_texts(["first", "second"])

    assert endpoint.request == {
        "input": ["first", "second"],
        "model": DEFAULT_EMBEDDING_MODEL,
        "encoding_format": "float",
    }
    assert batch.vectors == ((0.1, 0.2), (0.3, 0.4))
    assert batch.input_tokens == 7


def test_embed_chunks_keeps_chunk_ids_attached_to_vectors() -> None:
    endpoint = FakeEmbeddingsEndpoint()
    embedder = OpenRouterEmbedder(
        api_key="test-key", client=SimpleNamespace(embeddings=endpoint)
    )
    chunks = [
        Chunk("guide:0000", "guide", "first", Path("guide.pdf"), "pdf", 0),
        Chunk("guide:0001", "guide", "second", Path("guide.pdf"), "pdf", 1),
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert [embedding.chunk_id for embedding in embeddings] == [
        "guide:0000",
        "guide:0001",
    ]
    assert embeddings[0].vector == (0.1, 0.2)


def test_embed_texts_rejects_empty_input() -> None:
    embedder = OpenRouterEmbedder(api_key="test-key", client=SimpleNamespace())

    with pytest.raises(ValueError, match="at least one"):
        embedder.embed_texts([])
