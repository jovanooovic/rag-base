import pymupdf
import pytest

from app.ingest.loaders import load_pdf


def _text_pdf(path, heading: str, body: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), heading, fontsize=18)
    page.insert_text((72, 110), body, fontsize=11)
    doc.save(str(path))
    doc.close()


def _scanned_pdf(path, heading: str, body: str) -> None:
    """A PDF with no text layer at all -- an image of text, like a real scan."""
    src = pymupdf.open()
    src_page = src.new_page()
    src_page.insert_text((72, 72), heading, fontsize=18)
    src_page.insert_text((72, 110), body, fontsize=11)
    pixmap = src_page.get_pixmap(dpi=200)
    image_bytes = pixmap.tobytes("png")
    src.close()

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=image_bytes)
    doc.save(str(path))
    doc.close()


def test_load_pdf_extracts_text_and_keeps_the_heading(tmp_path):
    pdf_path = tmp_path / "policy.pdf"
    _text_pdf(pdf_path, "Refund Policy", "Refunds are issued within 30 days of purchase.")

    text = load_pdf(pdf_path)

    assert "Refund Policy" in text
    assert "30 days" in text
    assert text.strip().startswith("#")  # structure-first chunking splits on this


def test_load_pdf_ocrs_a_scanned_page_with_no_text_layer(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _scanned_pdf(pdf_path, "International Shipping", "We ship to 40 countries.")

    text = load_pdf(pdf_path)

    assert "International Shipping" in text
    assert "40 countries" in text


def test_load_pdf_fails_loudly_on_a_genuinely_blank_page(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    with pytest.raises(RuntimeError, match="extracted no text"):
        load_pdf(pdf_path)
