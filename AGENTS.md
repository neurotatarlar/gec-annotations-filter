# Agent Notes

Context for Codex/AI agents working on this repo.

- Code lives under `app/`.
- SQLite pipeline is immutable: each command reads a source table and writes to a new output table; defaults:
  - `ingest` -> `clean_segments`
  - `dedup` -> `dedup_segments`
  - `toxicity` -> `toxicity_segments`
  - LLM scoring -> `llm_segments`
- Dedup uses SimHash (near-dup distance configurable, stricter for short texts). It seeds the index from existing output rows so reruns are incremental.
- Toxicity uses `textdetox/bert-multilingual-toxicity-classifier`; `_is_toxic_label` maps `LABEL_1`/“toxic”/similar to toxic. Incremental: skips ids already in the output table.
- LLM scoring:
  - `gemini` command batches multiple rows per API request and writes to `llm_segments` by default.
  - `llm-step` drives a single prompt/response file (`reports/llm_batch.txt`): on each run, if RESPONSE is filled, it applies to `llm_segments`; otherwise it writes the next prompt batch for manual copy/paste.
- Storage helpers validate table names; `ensure_llm_table` mirrors gemini schema.
- Ingest uses `ingest_log` to skip already-processed rows for the same dataset + output table; no resume offsets.
- Caching: dataset cache under `data/hf_cache`; revision check against HF Hub; forced redownload when revision changes.
- Defaults: DB at `data/pipeline.sqlite`, reports under `reports/`.

Recent removals: NDJSON-based `llm-render`/`llm-apply` were removed; use `llm-step` instead for manual LLM interaction.
