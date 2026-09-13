"""Tests for PDF ingestion behavior that must remain stable."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from retrievallab.ingest import (
    document_id_from_path,
    ingest_markdown,
    ingest_path,
    ingest_pdf,
    ingest_txt,
)


def test_document_id_from_path_is_readable_and_stable() -> None:
    path = Path("MScAC Handbook 2026_27 (Final).pdf")

    assert document_id_from_path(path) == "mscac-handbook-2026-27-final"


def test_ingest_pdf_preserves_page_numbers_and_empty_pages() -> None:
    first_page = Mock()
    first_page.extract_text.return_value = " First page text. \n"
    second_page = Mock()
    second_page.extract_text.return_value = None
    reader = Mock(is_encrypted=False, pages=[first_page, second_page])

    with patch("retrievallab.ingest.PdfReader", return_value=reader):
        document = ingest_pdf(Path("Student Guide.pdf"))

    assert document.document_id == "student-guide"
    assert document.text == "First page text."
    assert document.pages[0].number == 1
    assert document.pages[0].text == "First page text."
    assert document.pages[1].number == 2
    assert document.pages[1].text == ""


def test_ingest_pdf_rejects_non_pdf_paths() -> None:
    with pytest.raises(ValueError, match="Expected a PDF"):
        ingest_pdf(Path("guide.md"))


def test_ingest_markdown_preserves_headings_and_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "Program Guide.md"
    path.write_text("# Program\r\n\r\n## Requirements\r\nTake a course.\r\n")

    document = ingest_markdown(path)

    assert document.document_id == "program-guide"
    assert document.source_type == "markdown"
    assert document.text == "# Program\n\n## Requirements\nTake a course."
    assert document.pages == ()


def test_ingest_txt_normalizes_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "notice.txt"
    path.write_bytes(b"First line.\r\n\r\nSecond line.\r\n")

    document = ingest_txt(path)

    assert document.source_type == "txt"
    assert document.text == "First line.\n\nSecond line."
    assert document.pages == ()


def test_ingest_path_dispatches_each_supported_format(tmp_path: Path) -> None:
    markdown_path = tmp_path / "guide.md"
    text_path = tmp_path / "notice.txt"
    markdown_path.write_text("# Guide")
    text_path.write_text("Notice")

    assert ingest_path(markdown_path).source_type == "markdown"
    assert ingest_path(text_path).source_type == "txt"


def test_ingest_path_rejects_unsupported_formats(tmp_path: Path) -> None:
    path = tmp_path / "guide.html"
    path.write_text("<h1>Guide</h1>")

    with pytest.raises(ValueError, match="Unsupported"):
        ingest_path(path)
