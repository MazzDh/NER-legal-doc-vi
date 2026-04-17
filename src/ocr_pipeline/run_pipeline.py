from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layout_filter import LayoutFilterConfig, filter_layout_blocks
from .normalize import normalize_text
from .paddle_engine import PaddleOCREngine
from .pdf_extract import has_sufficient_text
from .types import DocumentResult, PageResult, TextBlock


@dataclass
class OCRPipelineConfig:
    lang: str = "vi"
    min_pdf_text_chars: int = 80
    min_pdf_text_words: int = 15
    save_intermediate_json: bool = True


def _require_pymupdf() -> Any:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "PyMuPDF is not installed. Install optional deps with `pip install -e \".[ocr]\"`."
        ) from exc
    return fitz


def _digital_text_block(page_index: int, text: str, width: float, height: float) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=(0.0, 0.0, float(width), float(height)),
        confidence=1.0,
        source="digital_text",
        page_index=page_index,
    )


def process_pdf_document(
    pdf_path: Path,
    cfg: OCRPipelineConfig | None = None,
    layout_cfg: LayoutFilterConfig | None = None,
) -> DocumentResult:
    config = cfg or OCRPipelineConfig()
    fitz = _require_pymupdf()
    engine = PaddleOCREngine(lang=config.lang)

    doc = fitz.open(pdf_path)
    page_results: list[PageResult] = []
    doc_titles: list[str] = []

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            extracted = normalize_text(page.get_text("text"))

            if has_sufficient_text(
                extracted,
                min_chars=config.min_pdf_text_chars,
                min_words=config.min_pdf_text_words,
            ):
                blocks = [_digital_text_block(page_index, extracted, page_width, page_height)]
                source = "digital_text"
            else:
                pix = page.get_pixmap(dpi=300)
                fd, tmp_name = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                Path(tmp_name).unlink(missing_ok=True)
                try:
                    pix.save(tmp_name)
                    blocks = engine.ocr_page(image=tmp_name, page_index=page_index)
                finally:
                    Path(tmp_name).unlink(missing_ok=True)
                source = "ocr"

            filtered = filter_layout_blocks(
                blocks=blocks,
                page_width=page_width,
                page_height=page_height,
                cfg=layout_cfg,
            )

            page_text = normalize_text("\n".join(block.text for block in filtered.kept_blocks if block.text.strip()))
            page_result = PageResult(
                page_index=page_index,
                source=source,
                text=page_text,
                title_candidates=filtered.title_candidates,
                removed_blocks=filtered.removed_blocks,
                kept_blocks=filtered.kept_blocks,
            )
            page_results.append(page_result)
            doc_titles.extend(filtered.title_candidates)
    finally:
        doc.close()

    full_text = normalize_text("\n\n".join(page.text for page in page_results if page.text))
    dedup_titles: list[str] = []
    seen: set[str] = set()
    for title in doc_titles:
        key = title.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup_titles.append(title)

    metadata = {
        "page_count": len(page_results),
        "source_counts": {
            "digital_text": sum(1 for p in page_results if p.source == "digital_text"),
            "ocr": sum(1 for p in page_results if p.source == "ocr"),
        },
        "config": {
            "lang": config.lang,
            "min_pdf_text_chars": config.min_pdf_text_chars,
            "min_pdf_text_words": config.min_pdf_text_words,
        },
    }
    return DocumentResult(
        input_path=str(pdf_path),
        full_text=full_text,
        title_candidates=dedup_titles,
        pages=page_results,
        metadata=metadata,
    )


def save_document_result(result: DocumentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
