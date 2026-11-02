import io

import pytest

from agent_stack.layer2_rag.document_loader import extract_document


def test_extracts_utf8_text_and_csv():
    assert extract_document("policy.txt", b"\xef\xbb\xbfLeave is accrued monthly.") == (
        "Leave is accrued monthly."
    )
    assert extract_document("policy.csv", b"topic,days\nannual leave,18\n") == (
        "topic\tdays\nannual leave\t18"
    )


def test_extracts_xlsx_sheet_names_and_values():
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Leave"
    sheet.append(["Policy", "Days"])
    sheet.append(["Annual leave", 18])
    stream = io.BytesIO()
    workbook.save(stream)

    text = extract_document("policy.xlsx", stream.getvalue())

    assert "Sheet: Leave" in text
    assert "Annual leave\t18" in text


def test_extracts_docx_paragraphs_and_tables():
    from docx import Document

    document = Document()
    document.add_paragraph("Leave is accrued monthly.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Annual leave"
    table.cell(0, 1).text = "18 days"
    stream = io.BytesIO()
    document.save(stream)

    text = extract_document("policy.docx", stream.getvalue())

    assert "Leave is accrued monthly." in text
    assert "Annual leave\t18 days" in text


def test_extracts_text_from_pdf():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_ref}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 20 100 Td (Taxi reimbursement is INR 850.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    pdf = io.BytesIO()
    writer.write(pdf)

    assert extract_document("travel_policy.pdf", pdf.getvalue()) == (
        "Taxi reimbursement is INR 850."
    )


def test_rejects_unsupported_and_empty_documents():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_document("policy.xls", b"data")
    with pytest.raises(ValueError, match="empty"):
        extract_document("policy.txt", b"")
    with pytest.raises(ValueError, match="No readable text"):
        extract_document("policy.txt", b" \n")
