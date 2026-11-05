import io
import re

from pypdf import PdfReader


class PDFError(ValueError):
    pass


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    if not data.startswith(b"%PDF"):
        raise PDFError("File is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    except Exception as e:  # noqa: BLE001
        raise PDFError(f"Could not read PDF: {e}") from e
    if not any(t.strip() for _, t in pages):
        raise PDFError("No extractable text found (scanned PDF? OCR is not supported in this version).")
    return pages


def chunk_pages(pages: list[tuple[int, str]], size: int = 1800, overlap: int = 300) -> list[dict]:
    """Page-aware sliding-window chunker that prefers paragraph / sentence boundaries."""
    chunks: list[dict] = []
    for page_no, text in pages:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                window = text[start:end]
                cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
                if cut > size * 0.5:
                    end = start + cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
