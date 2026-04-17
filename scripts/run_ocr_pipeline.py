from __future__ import annotations

import argparse
from pathlib import Path

from ocr_pipeline.run_pipeline import OCRPipelineConfig, process_pdf_document, save_document_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Vietnamese OCR pipeline for PDF documents (digital text first, OCR fallback)."
    )
    parser.add_argument("--input-pdf", required=True, help="Input PDF path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--lang", default="vi", help="PaddleOCR language code (default: vi)")
    parser.add_argument("--min-pdf-text-chars", type=int, default=80, help="Minimum chars to trust digital text")
    parser.add_argument("--min-pdf-text-words", type=int, default=15, help="Minimum words to trust digital text")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = OCRPipelineConfig(
        lang=args.lang,
        min_pdf_text_chars=args.min_pdf_text_chars,
        min_pdf_text_words=args.min_pdf_text_words,
    )
    result = process_pdf_document(pdf_path=Path(args.input_pdf), cfg=config)
    save_document_result(result=result, output_path=Path(args.output_json))

    print("OCR pipeline complete:")
    print(f"- input: {args.input_pdf}")
    print(f"- output: {args.output_json}")
    print(f"- pages: {result.metadata['page_count']}")
    print(f"- source_counts: {result.metadata['source_counts']}")
    print(f"- title_candidates: {len(result.title_candidates)}")


if __name__ == "__main__":
    main()
