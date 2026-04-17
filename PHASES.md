# NER Project Phases (Locked)

1. **Phase 1: Data Foundation & Auto-Label Bootstrapping**
   - Standardize source data
   - Convert Excel to BIO JSONL
   - Generate review queue and dataset QA report

2. **Phase 2: PhoBERT Baseline Training & Evaluation**
   - Fine-tune PhoBERT on core entities
   - Produce baseline metrics and error breakdown

3. **Phase 3: Human-in-the-Loop QA & Active Learning Loop**
   - Review low-confidence/conflict samples
   - Retrain with corrected labels iteratively

4. **Phase 4: User-Facing Inference & Highlight UX**
   - Build inference flow for PDF/Word/Text input
   - Deliver highlighted output and friendly summary panel
