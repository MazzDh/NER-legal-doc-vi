from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 BIO datasets.")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test JSONL files.")
    parser.add_argument("--label-map", required=True, help="Path to label_map.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
    known_labels = set(label_map.keys())

    split_paths = {
        "train": data_dir / "train.jsonl",
        "val": data_dir / "val.jsonl",
        "test": data_dir / "test.jsonl",
    }

    id_sets = {}
    global_label_counter = Counter()
    errors = []
    per_split = {}

    for split, path in split_paths.items():
        if not path.exists():
            errors.append(f"Missing split file: {path}")
            continue
        doc_ids = set()
        rows = list(read_jsonl(path))
        per_split[split] = len(rows)
        for idx, row in enumerate(rows):
            doc_id = str(row.get("doc_id", ""))
            tokens = row.get("tokens", [])
            tags = row.get("ner_tags", [])
            if not doc_id:
                errors.append(f"{split}[{idx}] missing doc_id")
            if doc_id in doc_ids:
                errors.append(f"{split}[{idx}] duplicate doc_id: {doc_id}")
            doc_ids.add(doc_id)
            if len(tokens) != len(tags):
                errors.append(f"{split}[{idx}] tokens/tags length mismatch ({len(tokens)} vs {len(tags)})")
            for tag in tags:
                if tag not in known_labels:
                    errors.append(f"{split}[{idx}] unknown label: {tag}")
                global_label_counter[tag] += 1
        id_sets[split] = doc_ids

    if id_sets.get("train") and id_sets.get("val"):
        overlap = id_sets["train"] & id_sets["val"]
        if overlap:
            errors.append(f"Doc overlap train/val: {len(overlap)}")
    if id_sets.get("train") and id_sets.get("test"):
        overlap = id_sets["train"] & id_sets["test"]
        if overlap:
            errors.append(f"Doc overlap train/test: {len(overlap)}")
    if id_sets.get("val") and id_sets.get("test"):
        overlap = id_sets["val"] & id_sets["test"]
        if overlap:
            errors.append(f"Doc overlap val/test: {len(overlap)}")

    result = {
        "ok": len(errors) == 0,
        "splits": per_split,
        "label_counts": dict(global_label_counter),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
