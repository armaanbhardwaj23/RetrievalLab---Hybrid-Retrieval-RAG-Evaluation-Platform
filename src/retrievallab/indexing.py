"""Build and persist a small exact dense-retrieval index."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retrievallab.chunking import chunk_document
from retrievallab.embeddings import OpenRouterEmbedder
from retrievallab.models import Chunk, ChunkEmbedding, Document


INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DenseIndex:
    """Chunks and their embeddings, ready to construct a dense retriever."""

    chunks: tuple[Chunk, ...]
    embeddings: tuple[ChunkEmbedding, ...]


def build_dense_index(
    documents: Sequence[Document], *, embedder: OpenRouterEmbedder
) -> DenseIndex:
    """Chunk documents and embed every chunk in one inspectable operation."""

    chunks = tuple(chunk for document in documents for chunk in chunk_document(document))
    if not chunks:
        raise ValueError("Cannot build an index without non-empty chunks.")
    return DenseIndex(chunks=chunks, embeddings=embedder.embed_chunks(chunks))


def save_dense_index(index: DenseIndex, path: Path) -> None:
    """Write an index as readable JSON so it can be loaded without re-embedding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "source_path": str(chunk.source_path),
                "source_type": chunk.source_type,
                "index": chunk.index,
                "page_numbers": list(chunk.page_numbers),
                "heading": chunk.heading,
            }
            for chunk in index.chunks
        ],
        "embeddings": [
            {
                "chunk_id": embedding.chunk_id,
                "model": embedding.model,
                "vector": list(embedding.vector),
            }
            for embedding in index.embeddings
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_dense_index(path: Path) -> DenseIndex:
    """Load a previously saved dense index and validate its internal links."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("Unsupported dense-index schema version.")

    chunks = tuple(
        Chunk(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            text=item["text"],
            source_path=Path(item["source_path"]),
            source_type=item["source_type"],
            index=item["index"],
            page_numbers=tuple(item["page_numbers"]),
            heading=item["heading"],
        )
        for item in payload["chunks"]
    )
    embeddings = tuple(
        ChunkEmbedding(
            chunk_id=item["chunk_id"],
            model=item["model"],
            vector=tuple(item["vector"]),
        )
        for item in payload["embeddings"]
    )

    _validate_index(chunks, embeddings)
    return DenseIndex(chunks=chunks, embeddings=embeddings)


def _validate_index(
    chunks: Sequence[Chunk], embeddings: Sequence[ChunkEmbedding]
) -> None:
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    embedding_ids = {embedding.chunk_id for embedding in embeddings}
    if not chunks or chunk_ids != embedding_ids:
        raise ValueError("Index chunks and embeddings must have matching IDs.")
    if len(chunk_ids) != len(chunks) or len(embedding_ids) != len(embeddings):
        raise ValueError("Index chunk IDs and embedding IDs must be unique.")
