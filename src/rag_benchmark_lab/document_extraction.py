"""Extracts plain text from an uploaded document so it can feed the same
`raw_text` field RAGBenchmarkRunner/RAGPipeline already expect.

Kept deliberately separate from api.py so the extraction logic is testable
without spinning up FastAPI, and separate from RAGPipeline itself since
"get text out of a file" and "chunk + embed + retrieve" are different
concerns.
"""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(filename: str, content: bytes) -> str:
    """Returns extracted plain text, or raises ValueError with a message
    safe to show directly in the UI (unsupported type / empty / corrupt)."""
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix or '(none)'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if suffix in (".txt", ".md"):
        text = _decode_text(content, filename)
    else:  # .pdf
        text = _extract_pdf(content, filename)

    text = text.strip()
    if not text:
        raise ValueError(f"'{filename}' contains no extractable text.")
    return text


def _decode_text(content: bytes, filename: str) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return content.decode("latin-1")
        except UnicodeDecodeError as e:
            raise ValueError(f"Could not decode '{filename}' as text: {e}")


def _extract_pdf(content: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError(
            "PDF support requires the 'pypdf' package, which is not installed."
        ) from e

    import io

    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"'{filename}' could not be read as a PDF: {e}")

    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
