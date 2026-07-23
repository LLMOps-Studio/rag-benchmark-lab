import io

import pytest

from rag_benchmark_lab.document_extraction import extract_text


def test_extracts_plain_txt():
    assert extract_text("notes.txt", b"Hello world.") == "Hello world."


def test_extracts_markdown():
    assert extract_text("readme.md", b"# Title\n\nBody text.") == "# Title\n\nBody text."


def test_extracts_latin1_fallback():
    # A byte sequence that isn't valid UTF-8 but is valid latin-1 (e.g. a
    # smart quote written by an old Windows editor).
    content = "café".encode("latin-1")
    assert extract_text("notes.txt", content) == "café"


def test_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("image.png", b"\x89PNG\r\n")


def test_rejects_empty_content():
    with pytest.raises(ValueError, match="no extractable text"):
        extract_text("empty.txt", b"   \n  ")


def test_extracts_real_pdf_text():
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "Finwise Scribe technical documentation.")
    c.save()

    text = extract_text("doc.pdf", buf.getvalue())
    assert "Finwise Scribe" in text


def test_rejects_pdf_with_no_extractable_text():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(ValueError, match="no extractable text"):
        extract_text("blank.pdf", buf.getvalue())
