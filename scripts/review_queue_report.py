from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize review_queue.jsonl for Phase 1 QA.")
    parser.add_argument("--review-queue", required=True, help="Path to review_queue.jsonl")
    parser.add_argument("--top-k", type=int, default=5, help="Top K entities/reasons")
    args = parser.parse_args()

    path = Path(args.review_queue)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    reason_counter = Counter()
    entity_counter = Counter()
    reason_by_entity = defaultdict(Counter)
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            reason = row.get("reason", "unknown")
            entity = row.get("entity_type", "unknown")
            reason_counter[reason] += 1
            entity_counter[entity] += 1
            reason_by_entity[entity][reason] += 1

    report = {
        "total_review_items": total,
        "top_reasons": reason_counter.most_common(args.top_k),
        "top_entities": entity_counter.most_common(args.top_k),
        "reason_by_entity": {k: dict(v) for k, v in reason_by_entity.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
