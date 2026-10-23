"""
document_parser.py
-------------------
Extracts raw text from uploaded resume / job description files.
Supports PDF, DOCX, and plain TXT.
"""

import io
from pypdf import PdfReader
import docx


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file given as bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file given as bytes."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file given as bytes."""
    return file_bytes.decode("utf-8", errors="ignore")


def parse_document(filename: str, file_bytes: bytes) -> str:
    """
    Dispatch to the right parser based on file extension.
    Raises ValueError for unsupported file types.
    """
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext == "docx":
        return parse_docx(file_bytes)
    elif ext == "txt":
        return parse_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. Supported types: pdf, docx, txt"
        )


def clean_text(text: str) -> str:
    """Basic cleanup: collapse excess whitespace and blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
