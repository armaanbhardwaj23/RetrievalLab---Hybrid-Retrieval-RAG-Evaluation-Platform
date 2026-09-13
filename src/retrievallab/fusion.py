"""Rank-list fusion for hybrid retrieval."""

from collections.abc import Sequence

from retrievallab.models import RetrievalResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]],
    *,
    rank_constant: int = 60,
    top_k: int = 5,
) -> tuple[RetrievalResult, ...]:
    """Fuse ranked lists without comparing their incompatible raw scores."""

    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    fused_scores: dict[str, float] = {}
    chunks_by_id = {}
    for ranking in rankings:
        for result in ranking:
            chunk_id = result.chunk.chunk_id
            chunks_by_id[chunk_id] = result.chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (
                rank_constant + result.rank
            )

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        RetrievalResult(
            chunk=chunks_by_id[chunk_id],
            rank=rank,
            score=score,
            method="hybrid",
        )
        for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
    )
