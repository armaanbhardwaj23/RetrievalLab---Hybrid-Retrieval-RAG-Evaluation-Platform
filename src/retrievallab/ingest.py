"""Read supported source files into normalized :class:`Document` objects."""

import re
from pathlib import Path

from pypdf import PdfReader

from retrievallab.models import Document, DocumentPage, SourceType


def document_id_from_path(path: Path) -> str:
    """Return a stable, readable identifier derived from a source filename."""

    normalized = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    if not normalized:
        raise ValueError(f"Cannot derive a document ID from {path.name!r}.")
    return normalized


def ingest_path(path: Path) -> Document:
    """Dispatch a supported source file to its format-specific reader."""

    readers = {
        ".pdf": ingest_pdf,
        ".md": ingest_markdown,
        ".txt": ingest_txt,
    }
    try:
        reader = readers[path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported source file: {path.name!r}.") from error
    return reader(path)


def ingest_pdf(path: Path) -> Document:
    """Extract each PDF page while preserving its one-based page number."""

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received {path.name!r}.")

    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDFs are not supported: {path.name!r}.")

    pages = tuple(
        DocumentPage(number=number, text=(page.extract_text() or "").strip())
        for number, page in enumerate(reader.pages, start=1)
    )
    text = "\n\n".join(page.text for page in pages if page.text)

    return Document(
        document_id=document_id_from_path(path),
        text=text,
        source_path=path,
        source_type="pdf",
        pages=pages,
    )


def ingest_markdown(path: Path) -> Document:
    """Read UTF-8 Markdown while retaining its heading syntax and paragraphs."""

    return _ingest_text_file(path, expected_suffix=".md", source_type="markdown")


def ingest_txt(path: Path) -> Document:
    """Read a UTF-8 plain-text source document."""

    return _ingest_text_file(path, expected_suffix=".txt", source_type="txt")


def _ingest_text_file(
    path: Path,
    *,
    expected_suffix: str,
    source_type: SourceType,
) -> Document:
    if path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"Expected a {expected_suffix} file, received {path.name!r}."
        )

    text = _normalize_text(path.read_text(encoding="utf-8"))
    return Document(
        document_id=document_id_from_path(path),
        text=text,
        source_path=path,
        source_type=source_type,
    )


def _normalize_text(text: str) -> str:
    """Normalize line endings without flattening meaningful document structure."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
