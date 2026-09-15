"""Inspect an existing dense index for likely duplicate chunks without modifying it."""

from dataclasses import dataclass

from retrievallab.dense import cosine_similarity
from retrievallab.indexing import DenseIndex
from retrievallab.models import Chunk


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two chunks whose embedding similarity meets the chosen review threshold."""

    first: Chunk
    second: Chunk
    similarity: float


def find_near_duplicate_chunks(
    index: DenseIndex, *, similarity_threshold: float = 0.95
) -> tuple[DuplicatePair, ...]:
    """Return candidate duplicates for human review; never delete either chunk."""

    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1.")
    embeddings = {embedding.chunk_id: embedding.vector for embedding in index.embeddings}
    duplicates: list[DuplicatePair] = []
    for position, first in enumerate(index.chunks):
        for second in index.chunks[position + 1 :]:
            similarity = cosine_similarity(embeddings[first.chunk_id], embeddings[second.chunk_id])
            if similarity >= similarity_threshold:
                duplicates.append(DuplicatePair(first, second, similarity))
    return tuple(sorted(duplicates, key=lambda pair: -pair.similarity))
