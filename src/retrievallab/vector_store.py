"""A persistent vector-store adapter, separate from the exact-search baseline."""

import json
from pathlib import Path
from typing import Any

from retrievallab.embeddings import OpenRouterEmbedder
from retrievallab.indexing import DenseIndex
from retrievallab.models import Chunk, RetrievalResult

DEFAULT_COLLECTION_NAME = "retrievallab_chunks"


class ChromaVectorStore:
    """Persist dense vectors in Chroma and search them by cosine distance.

    RetrievalLab retains exact cosine search as its transparent baseline. This
    adapter demonstrates the boundary a vector database adds: persistence,
    metadata storage, and a database query API.
    """

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    @classmethod
    def create_from_index(
        cls, index: DenseIndex, *, persist_directory: Path,
        collection_name: str = DEFAULT_COLLECTION_NAME, overwrite: bool = False,
        client: Any | None = None,
    ) -> "ChromaVectorStore":
        """Persist an existing index without re-embedding its chunks."""
        chroma_client = client or _persistent_client(persist_directory)
        existing_names = {item.name for item in chroma_client.list_collections()}
        if collection_name in existing_names:
            if not overwrite:
                raise ValueError(f"Chroma collection {collection_name!r} already exists; pass overwrite=True to replace it.")
            chroma_client.delete_collection(collection_name)
        model = index.embeddings[0].model
        collection = chroma_client.create_collection(
            name=collection_name, metadata={"embedding_model": model, "hnsw:space": "cosine"},
        )
        vectors_by_id = {embedding.chunk_id: list(embedding.vector) for embedding in index.embeddings}
        collection.add(
            ids=[chunk.chunk_id for chunk in index.chunks],
            documents=[chunk.text for chunk in index.chunks],
            embeddings=[vectors_by_id[chunk.chunk_id] for chunk in index.chunks],
            metadatas=[_chunk_metadata(chunk) for chunk in index.chunks],
        )
        return cls(collection)

    @classmethod
    def open(cls, *, persist_directory: Path,
             collection_name: str = DEFAULT_COLLECTION_NAME, client: Any | None = None) -> "ChromaVectorStore":
        """Open a previously persisted collection."""
        chroma_client = client or _persistent_client(persist_directory)
        return cls(chroma_client.get_collection(collection_name))

    def search(self, query: str, *, embedder: OpenRouterEmbedder, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Embed a query once and ask Chroma for its nearest stored passages."""
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        expected_model = self._collection.metadata.get("embedding_model")
        batch = embedder.embed_texts([query])
        if batch.model != expected_model:
            raise ValueError("Query embedding model does not match the Chroma collection.")
        response = self._collection.query(
            query_embeddings=[list(batch.vectors[0])], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids, documents, metadatas, distances = (
            response["ids"][0], response["documents"][0],
            response["metadatas"][0], response["distances"][0],
        )
        return tuple(
            RetrievalResult(
                chunk=_chunk_from_record(chunk_id, text, metadata), rank=rank,
                score=1 - distance, method="dense",
            )
            for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances), start=1
            )
        )


def _persistent_client(persist_directory: Path) -> Any:
    try:
        import chromadb
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Chroma is not installed. Run `uv sync` to install RetrievalLab dependencies.") from error
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "document_id": chunk.document_id, "source_path": str(chunk.source_path),
        "source_type": chunk.source_type, "index": chunk.index,
        "page_numbers": json.dumps(chunk.page_numbers), "heading": chunk.heading or "",
    }


def _chunk_from_record(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id=metadata["document_id"], text=text,
        source_path=Path(metadata["source_path"]), source_type=metadata["source_type"],
        index=metadata["index"], page_numbers=tuple(json.loads(metadata["page_numbers"])),
        heading=metadata["heading"] or None,
    )
