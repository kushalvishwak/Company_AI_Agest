"""Extract searchable text from supported office and plain-text documents."""

from __future__ import annotations

import csv
import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".xlsx", ".pdf", ".docx"}
MAX_EXTRACTED_CHARACTERS = 2_000_000


def extract_document(filename: str, content: bytes) -> str:
    """Return plain text from a supported document, raising on invalid input."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Supported extensions: {supported}")
    if not content:
        raise ValueError("The uploaded file is empty")

    if extension in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
    elif extension == ".csv":
        rows = csv.reader(io.StringIO(content.decode("utf-8-sig")))
        text = "\n".join("\t".join(cell.strip() for cell in row) for row in rows)
    elif extension == ".xlsx":
        text = _extract_xlsx(content)
    elif extension == ".pdf":
        text = _extract_pdf(content)
    else:
        text = _extract_docx(content)

    text = text.strip()
    if not text:
        raise ValueError("No readable text was found in the document")
    if len(text) > MAX_EXTRACTED_CHARACTERS:
        raise ValueError(
            f"Extracted text is too large (over {MAX_EXTRACTED_CHARACTERS:,} characters)"
        )
    return text


def _extract_xlsx(content: bytes) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sections: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows = [
                "\t".join("" if value is None else str(value) for value in row).strip()
                for row in sheet.iter_rows(values_only=True)
            ]
            nonempty_rows = [row for row in rows if row.strip()]
            if nonempty_rows:
                sections.append(f"Sheet: {sheet.title}\n" + "\n".join(nonempty_rows))
    finally:
        workbook.close()
    return "\n\n".join(sections)


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    if reader.is_encrypted:
        raise ValueError("Password-protected PDF files are not supported")
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(content: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(content))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)
