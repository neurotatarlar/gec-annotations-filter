# GEC Annotation Filter (Tatar)

Typer-based CLI to stream VK messages from Hugging Face, clean them for Tatar GEC annotation, deduplicate, score toxicity, and export to Parquet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

Defaults: sqlite at `data/pipeline.sqlite`, reports under `reports/`.

Pipeline is immutable: every stage reads from a source table and writes to a new output table; by default each stage uses a different table name. Put the fastest, most-dropping steps first.

1) **Ingest & clean**
```bash
python -m app.cli ingest --dataset yasalma/vk-messages --split train --output-table clean_segments
# reports default to reports/ingest_report.json, HF cache to data/hf_cache
# runs full cleaning/splitting; uses ingest_log to skip rows already processed for the same output table
```

2) **Deduplicate (exact + SimHash near-dups)**
```bash
python -m app.cli dedup --source-table clean_segments --output-table dedup_segments --distance 1
# reports default to reports/dedup_report.json
```

3) **Toxicity filtering (textdetox)**
```bash
python -m app.cli toxicity --source-table dedup_segments --output-table toxicity_segments --batch-size 32 --device -1
# By default keeps toxic texts but stores toxicity_label/score.
# To drop above a threshold: add --threshold 0.5
```

4) **LLM scoring (Gemini API or manual copy/paste)**
```bash
python -m app.cli gemini --source-table toxicity_segments --output-table llm_segments --model models/gemini-3-flash-preview ; spd-say complete
python -m app.cli gemini --source-table toxicity_segments --output-table llm_segments --keys-path data/gemini_keys.yaml --model models/gemini-3-flash-preview
# custom prompt: --prompt-path path/to/prompt.txt
# adaptive batching: --batch-size 64 --max-batch-size 512
```

5) **Export to Parquet**
```bash
python -m app.cli export output.parquet
```

6) **Prepare annotation batch (JSON)**
```bash
python -m app.cli prepare-import --source-table llm_segments --limit 5000 data/import.json
# Use --dry-run to see counts without writing.
# Use --keep-proportions to preserve low/medium/high error_density ratios.
# Use --extreme-share to control low/high share (default 0.15 each).
```

Notes:
- Use `--source-table`/`--output-table` to chain stages without mutating inputs; tables must differ.
- `ingest` skips rows already logged for the same dataset+output table (ingest_log) so re-runs only add new rows.
- LLM scoring writes to `llm_segments` by default, whether via Gemini API or `llm-step`.
- `prepare-import` is incremental: it records exported ids in `export_log` and only writes new eligible rows on rerun.
- Gemini scoring adapts batch size on timeouts/invalid JSON (grows after consecutive successes; allows up to 5% missing items) and marks `gemini_skipped` only after repeated failures at batch size 1.
- Safe for incremental updates: `ingest`, `dedup`, `toxicity`, `gemini`, `prepare-import`. `export` always writes a full parquet snapshot.

## Notes on cleaning rules

- Cyrillic-only letters; keep emojis.
- Word count 5–50; char count 20–300; at least 10 letters and 5 Tatar-specific letters (`ӘәҮүҖҗҢңӨөҺһ`).
- Split long texts by sentences into compliant chunks.
- Remove system messages (`Post is not in text format.`, `Comment is not in text format.`).
- Drop messages that are URL/phone/email-only.
- Replace URLs/phones/emails with deterministic fakes.
- Replace `[[Name]]` placeholders with deterministic Tatar names from `data/tatar_names.json`.
- Drop messages with single `[` `]` pairs (keeps `[[`/`]]` placeholders).

Tables in `data.sqlite`:
- `clean_segments`: cleaned + split texts with exact deduplication.
- `dedup_segments`: after near-duplicate filtering.
- `toxicity_segments`: after toxicity scoring (stores label + score; optional dropping if threshold set).
- `llm_segments`: after LLM scoring (stores response JSON per row; Gemini is the default).
- `export_log`: ids already included in `prepare-import` to keep exports incremental.

## Gemini keys
- Put your Gemini API keys into `data/gemini_keys.yaml` as a list or mapping:
```yaml
- key1_here
- key2_here
```
  or
```yaml
primary: key1_here
backup: key2_here
```
- Keys are used round-robin on each request. Prompt defaults to a structured JSON request; override with `--prompt-path` if desired.
