"""Tests for saving and loading a reusable dense index."""

from pathlib import Path

from retrievallab.indexing import DenseIndex, load_dense_index, save_dense_index
from retrievallab.models import Chunk, ChunkEmbedding


def test_dense_index_round_trip_preserves_vectors_and_provenance(tmp_path: Path) -> None:
    chunk = Chunk(
        chunk_id="guide:0000",
        document_id="guide",
        text="Refunds are due within 30 days.",
        source_path=Path("data/raw/guide.pdf"),
        source_type="pdf",
        index=0,
        page_numbers=(2,),
    )
    index = DenseIndex(
        chunks=(chunk,),
        embeddings=(
            ChunkEmbedding("guide:0000", "test-model", (0.1, 0.2, 0.3)),
        ),
    )
    path = tmp_path / "dense_index.json"

    save_dense_index(index, path)
    loaded = load_dense_index(path)

    assert loaded == index
