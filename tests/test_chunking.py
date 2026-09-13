"""Tests for chunk boundaries and citation provenance."""

from pathlib import Path

import pytest

from retrievallab.chunking import chunk_document
from retrievallab.models import Document, DocumentPage


def test_pdf_chunks_do_not_cross_page_boundaries() -> None:
    document = Document(
        document_id="guide",
        text="Page one source.\n\nPage two source.",
        source_path=Path("guide.pdf"),
        source_type="pdf",
        pages=(
            DocumentPage(number=1, text="Page one source."),
            DocumentPage(number=2, text="Page two source."),
        ),
    )

    chunks = chunk_document(document, max_characters=100, overlap_characters=0)

    assert [chunk.chunk_id for chunk in chunks] == ["guide:0000", "guide:0001"]
    assert [chunk.page_numbers for chunk in chunks] == [(1,), (2,)]
    assert [chunk.text for chunk in chunks] == ["Page one source.", "Page two source."]


def test_markdown_chunks_include_their_heading_path() -> None:
    document = Document(
        document_id="program-guide",
        text="# Program\n\n## Requirements\n\nComplete the required course.",
        source_path=Path("program-guide.md"),
        source_type="markdown",
    )

    chunks = chunk_document(document, max_characters=100, overlap_characters=0)

    assert len(chunks) == 1
    assert chunks[0].heading == "Program > Requirements"
    assert chunks[0].text == "Program > Requirements\n\nComplete the required course."


def test_long_text_uses_overlap_between_chunks() -> None:
    document = Document(
        document_id="notice",
        text="Alpha beta gamma delta epsilon zeta eta theta.",
        source_path=Path("notice.txt"),
        source_type="txt",
    )

    chunks = chunk_document(document, max_characters=20, overlap_characters=5)

    assert len(chunks) > 1
    assert chunks[0].text[-5:] in chunks[1].text


def test_invalid_overlap_is_rejected() -> None:
    document = Document(
        document_id="notice",
        text="Text.",
        source_path=Path("notice.txt"),
        source_type="txt",
    )

    with pytest.raises(ValueError, match="smaller"):
        chunk_document(document, max_characters=10, overlap_characters=10)


def test_overlap_always_makes_forward_progress() -> None:
    document = Document(
        document_id="short-words",
        text="a b c d e f g h i j k l m n o p",
        source_path=Path("short-words.txt"),
        source_type="txt",
    )

    chunks = chunk_document(document, max_characters=10, overlap_characters=9)

    assert chunks
    assert len(chunks) < len(document.text)
