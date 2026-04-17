# Phase 1 Completion Report

## Objective
Complete data foundation and auto-label bootstrapping for Vietnamese legal NER (core entities):
- `LEGAL_TYPE`
- `DOC_NUM`
- `ISSUANCE_DATE`
- `ISSUING_AUTHORITY`
- `SIGNER`
- `TITLE`

## Deliverables
- Converter pipeline: `scripts/convert_legal_xlsx.py`
- Core logic: `src/ner_pipeline/convert.py`
- Produced artifacts under `data/processed/`:
  - `train.jsonl`
  - `val.jsonl`
  - `test.jsonl`
  - `review_queue.jsonl`
  - `label_map.json`
  - `summary.json`
- Dataset validator: `scripts/validate_phase1_dataset.py`
- Review queue reporter: `scripts/review_queue_report.py`

## Completion Criteria
- Required columns validated from source XLSX
- BIO files generated for train/val/test
- Label map generated for core entities
- Review queue generated for unresolved/low-confidence/conflict cases
- Dataset split has no document overlap
- Token/tag consistency validated

## Handoff to Phase 2
- Use `data/processed/` as training input
- Run baseline fine-tuning with `scripts/train_phobert_ner.py`
- Keep `review_queue.jsonl` as the first active-learning backlog
