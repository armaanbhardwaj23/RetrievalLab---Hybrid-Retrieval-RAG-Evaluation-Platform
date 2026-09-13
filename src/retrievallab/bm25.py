"""A small, inspectable BM25 retriever over RetrievalLab chunks."""

import math
import re
from collections import Counter
from collections.abc import Sequence

from retrievallab.models import Chunk, RetrievalResult


STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


class Bm25Retriever:
    """Rank chunks by exact term matches with BM25 scoring."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required.")
        if k1 < 0:
            raise ValueError("k1 cannot be negative.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        self._term_frequencies = tuple(
            Counter(tokenize(chunk.text)) for chunk in self._chunks
        )
        self._lengths = tuple(sum(frequencies.values()) for frequencies in self._term_frequencies)
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._document_frequencies = Counter(
            term
            for frequencies in self._term_frequencies
            for term in frequencies
        )

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Return chunks ranked by BM25 score for the literal query terms."""

        query_terms = tokenize(query)
        if not query_terms:
            raise ValueError("query must contain at least one searchable term.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        scores = [
            (chunk, self._score(query_terms, frequencies, length))
            for chunk, frequencies, length in zip(
                self._chunks, self._term_frequencies, self._lengths, strict=True
            )
        ]
        scores.sort(key=lambda item: (-item[1], item[0].chunk_id))

        return tuple(
            RetrievalResult(chunk=chunk, rank=rank, score=score, method="bm25")
            for rank, (chunk, score) in enumerate(scores[:top_k], start=1)
        )

    def _score(
        self,
        query_terms: Sequence[str],
        frequencies: Counter[str],
        length: int,
    ) -> float:
        return sum(
            self._idf(term)
            * (
                frequency * (self._k1 + 1)
                / (
                    frequency
                    + self._k1
                    * (1 - self._b + self._b * length / self._average_length)
                )
            )
            for term in query_terms
            if (frequency := frequencies[term])
        )

    def _idf(self, term: str) -> float:
        documents_with_term = self._document_frequencies[term]
        document_count = len(self._chunks)
        return math.log(
            1 + (document_count - documents_with_term + 0.5) / (documents_with_term + 0.5)
        )


def tokenize(text: str) -> tuple[str, ...]:
    """Keep meaningful lowercase lexical terms, including course/error codes."""

    return tuple(
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in STOP_WORDS
    )
