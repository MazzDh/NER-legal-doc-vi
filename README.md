# Vietnamese Legal NER (PhoBERT) - v1 Scaffold

This repository contains:
- A semi-automatic converter from `legal_documents.xlsx` to BIO JSONL (`train/val/test`)
- A review queue for low-confidence/conflicting labels
- A PhoBERT token-classification training script
- A local OCR pipeline for Vietnamese administrative PDFs (digital text first, OCR fallback)

## 1) Install

```bash
pip install -e .
pip install -e ".[train,fuzzy,dev]"
pip install -e ".[ocr]"
```

## 2) Convert Excel -> BIO JSONL

```bash
python scripts/convert_legal_xlsx.py \
  --xlsx-path "C:\Users\LENOVO\.cache\huggingface\hub\datasets--minhdoan17--vietnamese-legal-documents\legal_documents.xlsx" \
  --output-dir "E:\NER\data\processed" \
  --fuzzy-threshold 85
```

Outputs:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `review_queue.jsonl`
- `label_map.json`
- `summary.json`

## 3) Train PhoBERT NER

```bash
python scripts/train_phobert_ner.py \
  --data-dir "E:\NER\data\processed" \
  --label-map "E:\NER\data\processed\label_map.json" \
  --output-dir "E:\NER\runs\phobert-v1" \
  --model-name "vinai/phobert-base-v2" \
  --num-train-epochs 4 \
  --learning-rate 3e-5 \
  --batch-size 8
```

## 3.2) Inference demo (single text)

```bash
python scripts/infer_demo.py \
  --model-dir "E:\NER\runs\phobert-v1\best_model" \
  --text "Nghị quyết số 115/NQ-HĐBCQG do Hội đồng bầu cử quốc gia ban hành ngày 29/01/2026, người ký Trần Thanh Mẫn." \
  --output-json "E:\NER\runs\phobert-v1\demo_prediction.json"
```

## 3.1) Validate Phase 1 dataset quality

```bash
python scripts/validate_phase1_dataset.py \
  --data-dir "E:\NER\data\processed" \
  --label-map "E:\NER\data\processed\label_map.json"
```

```bash
python scripts/review_queue_report.py \
  --review-queue "E:\NER\data\processed\review_queue.jsonl"
```

## 4) Entity set (v1 core)

- `LEGAL_TYPE`
- `DOC_NUM`
- `ISSUANCE_DATE`
- `ISSUING_AUTHORITY`
- `SIGNER`
- `TITLE`

## 5) OCR pipeline for PDF -> clean text

Run OCR extraction with strategy `digital-text-first + PaddleOCR fallback`:

```bash
python scripts/run_ocr_pipeline.py \
  --input-pdf "E:\NER\data\raw\sample_admin_doc.pdf" \
  --output-json "E:\NER\data\processed\ocr\sample_admin_doc.json" \
  --lang vi \
  --min-pdf-text-chars 80 \
  --min-pdf-text-words 15
```

Output JSON includes:
- `full_text` (cleaned text for downstream NER)
- `title_candidates` (title lines preserved for title entity extraction)
- `pages[].source` (`digital_text` or `ocr`)
- `pages[].removed_blocks` (stamp/signature-like text that was removed by layout rules)

## Notes

- Converter labels all occurrences when a candidate appears multiple times.
- `signers` is parsed as text before `:`.
- `issuance_date` is kept as both raw and ISO (`YYYY-MM-DD`).
