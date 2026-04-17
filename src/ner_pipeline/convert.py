from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None

CORE_ENTITIES = [
    "LEGAL_TYPE",
    "DOC_NUM",
    "ISSUANCE_DATE",
    "ISSUING_AUTHORITY",
    "SIGNER",
    "TITLE",
]

REQUIRED_COLUMNS = [
    "id",
    "document_number",
    "title",
    "legal_type",
    "issuing_authority",
    "issuance_date",
    "signers",
    "content",
]


@dataclass
class ConverterConfig:
    xlsx_path: Path
    output_dir: Path
    fuzzy_threshold: float = 85.0
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ConversionOutputs:
    train_path: Path
    val_path: Path
    test_path: Path
    review_queue_path: Path
    label_map_path: Path
    summary_path: Path


def _normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    out = unicodedata.normalize("NFC", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _parse_signer(raw: Any) -> str:
    text = _normalize_text(raw)
    if not text:
        return ""
    return text.split(":", 1)[0].strip()


def _to_iso_date(raw: Any) -> str | None:
    txt = _normalize_text(raw)
    if not txt:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(txt, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Simple fallback: dd/mm/yy variants
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", txt)
    if not m:
        return None
    day, month, year = m.groups()
    if len(year) == 2:
        year = f"20{year}"
    try:
        dt = datetime(int(year), int(month), int(day))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _find_all_exact(text: str, value: str) -> list[tuple[int, int, float]]:
    if not value:
        return []
    matches: list[tuple[int, int, float]] = []
    start = 0
    lower_text = text.lower()
    lower_value = value.lower()
    while True:
        idx = lower_text.find(lower_value, start)
        if idx == -1:
            break
        end = idx + len(value)
        matches.append((idx, end, 100.0))
        start = idx + 1
    return matches


def _find_fuzzy_windows(text: str, value: str, threshold: float) -> list[tuple[int, int, float]]:
    if not value or fuzz is None:
        return []
    # Character window scan with coarse step for speed.
    value_len = len(value)
    if value_len < 4:
        return []
    results: list[tuple[int, int, float]] = []
    max_step = max(1, value_len // 6)
    for i in range(0, max(0, len(text) - value_len + 1), max_step):
        window = text[i : i + value_len]
        score = float(fuzz.ratio(window.lower(), value.lower()))
        if score >= threshold:
            results.append((i, i + value_len, score))
    # Merge near-duplicate spans, keep higher score.
    results.sort(key=lambda x: (x[0], -x[2]))
    dedup: list[tuple[int, int, float]] = []
    for s, e, sc in results:
        if dedup and abs(s - dedup[-1][0]) <= 3:
            if sc > dedup[-1][2]:
                dedup[-1] = (s, e, sc)
            continue
        dedup.append((s, e, sc))
    return dedup


def _candidate_map(row: pd.Series) -> dict[str, str]:
    return {
        "DOC_NUM": _normalize_text(row.get("document_number", "")),
        "TITLE": _normalize_text(row.get("title", "")),
        "LEGAL_TYPE": _normalize_text(row.get("legal_type", "")),
        "ISSUING_AUTHORITY": _normalize_text(row.get("issuing_authority", "")),
        "ISSUANCE_DATE": _normalize_text(row.get("issuance_date", "")),
        "SIGNER": _parse_signer(row.get("signers", "")),
    }


def _detect_spans(
    doc_id: str,
    text: str,
    candidates: dict[str, str],
    fuzzy_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    for entity_type, value in candidates.items():
        if not value:
            review.append(
                {
                    "doc_id": doc_id,
                    "entity_type": entity_type,
                    "candidate_span": None,
                    "suggested_value": value,
                    "reason": "empty_candidate",
                }
            )
            continue

        exact = _find_all_exact(text, value)
        if exact:
            for s, e, conf in exact:
                entities.append(
                    {
                        "type": entity_type,
                        "start": s,
                        "end": e,
                        "text": text[s:e],
                        "confidence": conf,
                        "source": "exact",
                    }
                )
            continue

        fuzzy_found = _find_fuzzy_windows(text, value, fuzzy_threshold)
        if fuzzy_found:
            for s, e, conf in fuzzy_found:
                entities.append(
                    {
                        "type": entity_type,
                        "start": s,
                        "end": e,
                        "text": text[s:e],
                        "confidence": conf,
                        "source": "fuzzy",
                    }
                )
            continue

        review.append(
            {
                "doc_id": doc_id,
                "entity_type": entity_type,
                "candidate_span": None,
                "suggested_value": value,
                "reason": "not_found",
            }
        )

    # Remove overlaps preferring higher confidence and longer spans.
    entities.sort(key=lambda x: (x["start"], -x["confidence"], -(x["end"] - x["start"])))
    accepted: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for ent in entities:
        s, e = ent["start"], ent["end"]
        has_overlap = any(not (e <= os or s >= oe) for os, oe in occupied)
        if has_overlap:
            review.append(
                {
                    "doc_id": doc_id,
                    "entity_type": ent["type"],
                    "candidate_span": {"start": s, "end": e, "text": ent["text"]},
                    "suggested_value": ent["text"],
                    "reason": "conflict_overlap",
                }
            )
            continue
        occupied.append((s, e))
        accepted.append(ent)

    accepted.sort(key=lambda x: x["start"])
    return accepted, review


def _whitespace_tokens_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def _entities_to_bio(
    tokens: list[str],
    offsets: list[tuple[int, int]],
    entities: list[dict[str, Any]],
) -> list[str]:
    tags = ["O"] * len(tokens)
    for ent in entities:
        ent_type = ent["type"]
        start = ent["start"]
        end = ent["end"]
        token_ids = []
        for i, (ts, te) in enumerate(offsets):
            if te <= start or ts >= end:
                continue
            token_ids.append(i)
        if not token_ids:
            continue
        tags[token_ids[0]] = f"B-{ent_type}"
        for tid in token_ids[1:]:
            tags[tid] = f"I-{ent_type}"
    return tags


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_label_map() -> dict[str, int]:
    labels = ["O"]
    for ent in CORE_ENTITIES:
        labels.append(f"B-{ent}")
        labels.append(f"I-{ent}")
    return {label: idx for idx, label in enumerate(labels)}


def convert_excel_to_bio_jsonl(cfg: ConverterConfig) -> ConversionOutputs:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(cfg.xlsx_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        doc_id = str(row["id"])
        text = _normalize_text(row["content"])
        if not text:
            review_rows.append(
                {
                    "doc_id": doc_id,
                    "entity_type": None,
                    "candidate_span": None,
                    "suggested_value": None,
                    "reason": "empty_content",
                }
            )
            continue

        candidates = _candidate_map(row)
        entities, review = _detect_spans(
            doc_id=doc_id,
            text=text,
            candidates=candidates,
            fuzzy_threshold=cfg.fuzzy_threshold,
        )
        review_rows.extend(review)

        tokens, offsets = _whitespace_tokens_with_offsets(text)
        ner_tags = _entities_to_bio(tokens=tokens, offsets=offsets, entities=entities)

        # Low-confidence entities should be manually reviewed.
        for ent in entities:
            if ent["confidence"] < cfg.fuzzy_threshold:
                review_rows.append(
                    {
                        "doc_id": doc_id,
                        "entity_type": ent["type"],
                        "candidate_span": {
                            "start": ent["start"],
                            "end": ent["end"],
                            "text": ent["text"],
                        },
                        "suggested_value": ent["text"],
                        "reason": "low_confidence",
                    }
                )

        record = {
            "doc_id": doc_id,
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": entities,
            "confidence_meta": {
                "fuzzy_threshold": cfg.fuzzy_threshold,
            },
            "issuance_date_raw": _normalize_text(row.get("issuance_date", "")),
            "issuance_date_iso": _to_iso_date(row.get("issuance_date", "")),
        }
        records.append(record)

    random.Random(cfg.seed).shuffle(records)
    n = len(records)
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.val_ratio)
    train_rows = records[:train_end]
    val_rows = records[train_end:val_end]
    test_rows = records[val_end:]

    train_path = cfg.output_dir / "train.jsonl"
    val_path = cfg.output_dir / "val.jsonl"
    test_path = cfg.output_dir / "test.jsonl"
    review_path = cfg.output_dir / "review_queue.jsonl"
    label_map_path = cfg.output_dir / "label_map.json"
    summary_path = cfg.output_dir / "summary.json"

    _jsonl_write(train_path, train_rows)
    _jsonl_write(val_path, val_rows)
    _jsonl_write(test_path, test_rows)
    _jsonl_write(review_path, review_rows)

    label_map = _build_label_map()
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_path": str(cfg.xlsx_path),
        "total_docs": n,
        "train_docs": len(train_rows),
        "val_docs": len(val_rows),
        "test_docs": len(test_rows),
        "review_items": len(review_rows),
        "entities": CORE_ENTITIES,
        "fuzzy_threshold": cfg.fuzzy_threshold,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return ConversionOutputs(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        review_queue_path=review_path,
        label_map_path=label_map_path,
        summary_path=summary_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert legal_documents.xlsx to BIO JSONL dataset.")
    parser.add_argument("--xlsx-path", required=True, help="Path to source Excel file")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--fuzzy-threshold", type=float, default=85.0, help="Fuzzy match threshold [0,100]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = ConverterConfig(
        xlsx_path=Path(args.xlsx_path),
        output_dir=Path(args.output_dir),
        fuzzy_threshold=args.fuzzy_threshold,
        seed=args.seed,
    )
    outputs = convert_excel_to_bio_jsonl(cfg)
    print("Conversion complete:")
    print(f"- train: {outputs.train_path}")
    print(f"- val: {outputs.val_path}")
    print(f"- test: {outputs.test_path}")
    print(f"- review: {outputs.review_queue_path}")
    print(f"- labels: {outputs.label_map_path}")
    print(f"- summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
