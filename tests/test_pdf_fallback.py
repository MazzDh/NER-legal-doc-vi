from __future__ import annotations

from pathlib import Path

from ocr_pipeline.run_pipeline import OCRPipelineConfig, process_pdf_document
from ocr_pipeline.types import TextBlock


class _FakeRect:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height


class _FakePixmap:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-image")


class _FakePage:
    def __init__(self, text: str, width: float = 900, height: float = 1200):
        self._text = text
        self.rect = _FakeRect(width, height)

    def get_text(self, mode: str) -> str:
        assert mode == "text"
        return self._text

    def get_pixmap(self, dpi: int = 300) -> _FakePixmap:
        assert dpi == 300
        return _FakePixmap()


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]):
        self._pages = pages
        self.page_count = len(pages)

    def load_page(self, index: int) -> _FakePage:
        return self._pages[index]

    def close(self) -> None:
        return None


class _FakeFitzModule:
    @staticmethod
    def open(path: Path) -> _FakeDoc:  # noqa: ARG004
        pages = [
            _FakePage(
                "Van ban nay da co selectable text voi du noi dung cho threshold fallback test."
            ),
            _FakePage("too short"),
        ]
        return _FakeDoc(pages)


class _FakeEngine:
    def __init__(self, lang: str = "vi"):  # noqa: ARG002
        pass

    def ocr_page(self, image: str, page_index: int) -> list[TextBlock]:
        assert image.endswith(".png")
        return [
            TextBlock(
                text="QUYET DINH BO SUNG QUY DINH",
                bbox=(50.0, 50.0, 650.0, 95.0),
                confidence=0.96,
                source="ocr",
                page_index=page_index,
            ),
            TextBlock(
                text="Nguoi ky: Tran Van B",
                bbox=(420.0, 950.0, 850.0, 995.0),
                confidence=0.81,
                source="ocr",
                page_index=page_index,
            ),
        ]


def test_process_pdf_document_uses_digital_text_then_ocr_fallback(monkeypatch):
    import ocr_pipeline.run_pipeline as rp

    monkeypatch.setattr(rp, "_require_pymupdf", lambda: _FakeFitzModule)
    monkeypatch.setattr(rp, "PaddleOCREngine", _FakeEngine)

    result = process_pdf_document(
        pdf_path=Path("dummy.pdf"),
        cfg=OCRPipelineConfig(min_pdf_text_chars=20, min_pdf_text_words=5),
    )

    assert result.metadata["page_count"] == 2
    assert result.metadata["source_counts"]["digital_text"] == 1
    assert result.metadata["source_counts"]["ocr"] == 1
    assert "QUYET DINH BO SUNG QUY DINH" in result.title_candidates
    assert "Nguoi ky: Tran Van B" not in result.full_text
