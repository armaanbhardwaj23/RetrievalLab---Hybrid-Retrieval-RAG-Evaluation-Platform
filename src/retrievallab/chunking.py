"""Split normalized documents into small, traceable retrieval chunks."""

from dataclasses import dataclass

from retrievallab.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class _Section:
    """A structured region that chunks must not cross."""

    text: str
    page_numbers: tuple[int, ...] = ()
    heading: str | None = None


def chunk_document(
    document: Document,
    *,
    max_characters: int = 2_000,
    overlap_characters: int = 200,
) -> tuple[Chunk, ...]:
    """Create chunks while retaining page and heading provenance.

    ``max_characters`` is intentionally a temporary, explicit proxy for a
    token limit. We will switch to model-specific token counting only after an
    embedding model has been selected.
    """

    _validate_chunking_options(max_characters, overlap_characters)
    chunks: list[Chunk] = []

    for section in _sections_for_document(document):
        prefix = f"{section.heading}\n\n" if section.heading else ""
        content_limit = max(1, max_characters - len(prefix))
        content_overlap = min(overlap_characters, max(0, content_limit - 1))

        for text in _split_text(
            section.text,
            max_characters=content_limit,
            overlap_characters=content_overlap,
        ):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{index:04d}",
                    document_id=document.document_id,
                    text=f"{prefix}{text}",
                    source_path=document.source_path,
                    source_type=document.source_type,
                    index=index,
                    page_numbers=section.page_numbers,
                    heading=section.heading,
                )
            )

    return tuple(chunks)


def _validate_chunking_options(max_characters: int, overlap_characters: int) -> None:
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1.")
    if overlap_characters < 0:
        raise ValueError("overlap_characters cannot be negative.")
    if overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be smaller than max_characters.")


def _sections_for_document(document: Document) -> tuple[_Section, ...]:
    if document.source_type == "pdf":
        return tuple(
            _Section(text=page.text, page_numbers=(page.number,))
            for page in document.pages
            if page.text
        )
    if document.source_type == "markdown":
        return _markdown_sections(document.text)
    if document.source_type == "txt":
        return (_Section(text=document.text),) if document.text else ()
    raise ValueError(f"Unsupported source type: {document.source_type!r}.")


def _markdown_sections(text: str) -> tuple[_Section, ...]:
    sections: list[_Section] = []
    heading_parts: list[str] = []
    body_lines: list[str] = []

    def flush_body() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            heading = " > ".join(heading_parts) or None
            sections.append(_Section(text=body, heading=heading))
        body_lines.clear()

    for line in text.splitlines():
        heading = _parse_markdown_heading(line)
        if heading is None:
            body_lines.append(line)
            continue

        level, title = heading
        flush_body()
        heading_parts[:] = heading_parts[: level - 1]
        heading_parts.append(title)

    flush_body()
    return tuple(sections)


def _parse_markdown_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None

    level = len(stripped) - len(stripped.lstrip("#"))
    if level > 6 or len(stripped) == level or stripped[level] != " ":
        return None

    title = stripped[level:].strip().rstrip("#").strip()
    return (level, title) if title else None


def _split_text(
    text: str,
    *,
    max_characters: int,
    overlap_characters: int,
) -> tuple[str, ...]:
    text = text.strip()
    if not text:
        return ()

    pieces: list[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_characters:
            pieces.append(text[start:].strip())
            break

        end = _preferred_break(text, start, start + max_characters)
        pieces.append(text[start:end].strip())
        start = max(end - overlap_characters, start + 1)

    return tuple(piece for piece in pieces if piece)


def _preferred_break(text: str, start: int, limit: int) -> int:
    paragraph_break = text.rfind("\n\n", start + 1, limit + 1)
    sentence_break = text.rfind(". ", start + 1, limit + 1)
    word_break = text.rfind(" ", start + 1, limit + 1)

    if paragraph_break != -1:
        return paragraph_break
    if sentence_break != -1:
        return sentence_break + 1
    if word_break != -1:
        return word_break
    return limit
