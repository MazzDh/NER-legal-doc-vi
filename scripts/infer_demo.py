from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass
class PredEntity:
    entity_type: str
    start: int
    end: int
    text: str
    score: float


def parse_args():
    parser = argparse.ArgumentParser(description="Inference demo for PhoBERT NER model.")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory (best_model)")
    parser.add_argument("--text", help="Raw input text for inference")
    parser.add_argument("--text-file", help="Path to UTF-8 text file for inference")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-json", help="Optional path to save prediction JSON")
    return parser.parse_args()


def load_input_text(args) -> str:
    if args.text and args.text_file:
        raise ValueError("Use only one of --text or --text-file")
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --text or --text-file")


def decode_entities(
    text: str,
    offsets: list[tuple[int, int]],
    labels: list[str],
    probs: list[float],
) -> list[PredEntity]:
    entities: list[PredEntity] = []
    cur_type = None
    cur_start = None
    cur_end = None
    cur_scores: list[float] = []

    def flush():
        nonlocal cur_type, cur_start, cur_end, cur_scores
        if cur_type is None or cur_start is None or cur_end is None:
            return
        span_text = text[cur_start:cur_end]
        if span_text.strip():
            entities.append(
                PredEntity(
                    entity_type=cur_type,
                    start=cur_start,
                    end=cur_end,
                    text=span_text,
                    score=sum(cur_scores) / max(1, len(cur_scores)),
                )
            )
        cur_type = None
        cur_start = None
        cur_end = None
        cur_scores = []

    for (start, end), tag, p in zip(offsets, labels, probs):
        if start == end:
            continue
        if tag == "O":
            flush()
            continue

        if "-" in tag:
            prefix, entity_type = tag.split("-", 1)
        else:
            prefix, entity_type = "B", tag

        if prefix == "B":
            flush()
            cur_type = entity_type
            cur_start = start
            cur_end = end
            cur_scores = [p]
            continue

        # I-*
        if cur_type == entity_type and cur_end is not None and start <= cur_end + 1:
            cur_end = end
            cur_scores.append(p)
        else:
            # Broken I-* sequence: treat as new B-* span.
            flush()
            cur_type = entity_type
            cur_start = start
            cur_end = end
            cur_scores = [p]

    flush()
    return entities


def main():
    args = parse_args()
    text = load_input_text(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs_all = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs_all, dim=-1).tolist()
        pred_probs = probs_all.max(dim=-1).values.tolist()

    id2label = model.config.id2label
    labels = [id2label[i] for i in pred_ids]
    entities = decode_entities(text=text, offsets=offsets, labels=labels, probs=pred_probs)

    output: dict[str, Any] = {
        "text": text,
        "entities": [
            {
                "type": e.entity_type,
                "start": e.start,
                "end": e.end,
                "text": e.text,
                "score": round(e.score, 4),
            }
            for e in entities
        ],
    }

    print("Predicted entities:")
    if not entities:
        print("- (none)")
    for e in entities:
        print(f"- {e.entity_type:20s} | {e.text} | [{e.start},{e.end}] | score={e.score:.4f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
