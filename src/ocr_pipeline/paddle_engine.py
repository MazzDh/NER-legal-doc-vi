from __future__ import annotations

from typing import Any

from .types import TextBlock


class PaddleOCREngine:
    def __init__(self, lang: str = "vi", use_angle_cls: bool = True):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "PaddleOCR is not installed. Install optional deps with `pip install -e \".[ocr]\"` "
                "and ensure PaddlePaddle is installed for your platform."
            ) from exc
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def ocr_page(self, image: Any, page_index: int) -> list[TextBlock]:
        result = self._ocr.ocr(image, cls=True)
        blocks: list[TextBlock] = []
        for line in result[0] if result else []:
            quad = line[0]
            text, conf = line[1]
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            blocks.append(
                TextBlock(
                    text=str(text),
                    bbox=bbox,
                    confidence=float(conf),
                    source="ocr",
                    page_index=page_index,
                )
            )
        return blocks
