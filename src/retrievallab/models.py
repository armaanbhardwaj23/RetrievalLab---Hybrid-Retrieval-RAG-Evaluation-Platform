"""Data structures shared by RetrievalLab pipeline stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SourceType = Literal["pdf", "markdown", "txt"]
RetrievalMethod = Literal["dense", "bm25", "hybrid"]


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text extracted from one page of a PDF source."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class Document:
    """A normalized source document, before it is split into chunks."""

    document_id: str
    text: str
    source_path: Path
    source_type: SourceType
    pages: tuple[DocumentPage, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable passage and the provenance needed to cite it."""

    chunk_id: str
    document_id: str
    text: str
    source_path: Path
    source_type: SourceType
    index: int
    page_numbers: tuple[int, ...] = ()
    heading: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkEmbedding:
    """A vector generated for one chunk by a specific embedding model."""

    chunk_id: str
    model: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk returned by a retrieval strategy."""

    chunk: Chunk
    rank: int
    score: float
    method: RetrievalMethod
