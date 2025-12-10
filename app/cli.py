from pathlib import Path
from typing import Optional

import datasets
import json
import sqlite3
import hashlib
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from .config import DEFAULT_CONFIG
from .dedup import simhash_from_text
from .gemini import (
    DEFAULT_PROMPT,
    KeyRotator,
    call_gemini,
    configure_client,
    load_keys,
    parse_json_response,
)
from .processing import (
    ProcessedSegment,
    digest_from_url,
    is_contact_only,
    load_name_list,
    normalize_whitespace,
    passes_filters,
    process_text,
    tokenize_words,
)
from .storage import (
    copy_rows_to_dedup,
    copy_rows_to_gemini,
    copy_rows_to_toxicity,
    collect_ids,
    ensure_ingest_log_table,
    insert_clean_segments,
    iterate_table,
    open_db,
    ensure_clean_table,
    ensure_dedup_table,
    ensure_toxicity_table,
    ensure_gemini_table,
    try_log_row,
)
from .toxicity import load_toxicity_pipeline, predict_toxicity
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.progress import track

app = typer.Typer(add_completion=False, help="Pipeline to prepare texts for Tatar GEC annotation.")


def _ensure_distinct_tables(source: str, output: str, stage: str) -> None:
    if source == output:
        raise typer.BadParameter(f"{stage}: output_table must differ from source_table to keep pipeline immutable.")


def resolve_field(row: dict, field: str, fallback: str, idx: int) -> str:
    for key in (field, fallback, "message_url", "permalink", "id"):
        value = row.get(key)
        if value:
            return str(value)
    return f"{fallback}:{idx}"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _revision_file(cache_dir: Path, dataset: str, split: str) -> Path:
    return cache_dir / "revisions" / f"{_safe_token(dataset)}__{_safe_token(split)}.txt"


def _fetch_remote_revision(dataset: str) -> Optional[str]:
    """
    Try to retrieve the latest dataset commit sha from the Hugging Face Hub.
    """
    if Path(dataset).exists():
        return None
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        info = HfApi().dataset_info(dataset)
        return getattr(info, "sha", None)
    except Exception:
        return None


def _prepare_dataset_cache(dataset: str, split: str, cache_dir: Path) -> tuple[datasets.DownloadMode, str]:
    """
    Download the dataset to cache if needed and return the streaming download mode.

    If the cached revision differs from the latest on the Hub, force a re-download.
    """
    _ensure_parent(cache_dir)
    rev_path = _revision_file(cache_dir, dataset, split)
    cached_rev = rev_path.read_text(encoding="utf-8").strip() if rev_path.exists() else None
    latest_rev = _fetch_remote_revision(dataset)

    download_mode = datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
    if latest_rev and latest_rev != cached_rev:
        download_mode = datasets.DownloadMode.FORCE_REDOWNLOAD
        typer.echo(f"New dataset revision detected ({cached_rev or 'none'} -> {latest_rev}); forcing re-download.")
    elif latest_rev:
        typer.echo(f"Dataset cache matches remote revision {latest_rev}.")
    elif cached_rev:
        typer.echo("Using cached dataset; remote revision check unavailable.")
    else:
        typer.echo("No cached dataset found; downloading fresh copy.")

    # Materialize dataset locally to populate cache.
    datasets.load_dataset(
        dataset,
        split=split,
        cache_dir=str(cache_dir),
        streaming=False,
        download_mode=download_mode,
    )

    if latest_rev:
        rev_path.parent.mkdir(parents=True, exist_ok=True)
        rev_path.write_text(latest_rev, encoding="utf-8")

    revision_tag = latest_rev or cached_rev or "local"
    return download_mode, revision_tag


def _row_hash(source_id: str, raw_text: str) -> str:
    payload = (source_id or "") + "\n" + raw_text
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@app.command("fast-filter")
def fast_filter(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Path to sqlite database for intermediate data."),
    dataset: str = typer.Option("yasalma/vk-messages", help="HF dataset identifier or local path."),
    split: str = typer.Option("train", help="Dataset split to stream."),
    cache_dir: Path = typer.Option(Path("data/hf_cache"), help="Local cache dir for streamed dataset shards."),
    limit: Optional[int] = typer.Option(None, help="Optional limit for debugging."),
    batch_size: int = typer.Option(1000, help="Number of rows to insert per transaction."),
    report_path: Optional[Path] = typer.Option(Path("reports/fast_filter_report.json"), help="Write JSON counters about filter reasons."),
    output_table: str = typer.Option("raw_segments", "--output-table", help="Destination table for fast-filtered rows."),
):
    """
    Fast, minimal filtering of the source dataset (no replacements or splitting).
    """
    from collections import Counter
    import json

    cfg = DEFAULT_CONFIG
    _ensure_parent(cache_dir)
    download_mode, revision_tag = _prepare_dataset_cache(dataset, split, cache_dir)
    builder = datasets.load_dataset_builder(dataset, cache_dir=str(cache_dir))
    ds = datasets.load_dataset(
        dataset,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
        download_mode=download_mode,
    )
    total = builder.info.splits[split].num_examples if split in builder.info.splits else None

    buffered = []
    kept_rows = 0
    stats = Counter()
    processed_rows = 0
    _ensure_parent(db_path)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with open_db(db_path) as conn, progress:
        ensure_clean_table(conn, output_table)
        ensure_ingest_log_table(conn)

        dataset_label = f"{dataset}:{output_table}"
        task = progress.add_task("Fast filter", total=total)
        for idx, row in enumerate(ds):
            if limit is not None and processed_rows >= limit:
                break
            progress.advance(task)

            raw_text = row["message"]
            source_url = row["url"]
            row_hash = _row_hash(source_url, str(raw_text))
            if not try_log_row(conn, row_hash, dataset_label, split, revision_tag, source_url):
                stats["rows_skipped_seen"] += 1
                continue

            stats["rows_total"] += 1
            text = normalize_whitespace(str(raw_text))
            if not text or text in cfg.system_messages:
                stats["reject_system_or_empty"] += 1
                continue
            if is_contact_only(text):
                stats["reject_contact_only"] += 1
                continue
            if not passes_filters(text, cfg):
                stats["reject_filters"] += 1
                continue

            parent_digest = digest_from_url(source_url) if source_url else digest_from_url(text)
            seg_id = f"{parent_digest}#0"
            buffered.append(ProcessedSegment(segment_id=seg_id, parent_digest=parent_digest, text=text))
            processed_rows += 1

            if len(buffered) >= batch_size:
                kept_rows += insert_clean_segments(conn, buffered, output_table)
                buffered = []
                conn.commit()

        if buffered:
            kept_rows += insert_clean_segments(conn, buffered, output_table)
        conn.commit()

    typer.echo(f"Fast-filtered {kept_rows} rows into {output_table} (db: {db_path})")
    if report_path:
        _ensure_parent(report_path)
        stats["rows_inserted"] = kept_rows
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote fast-filter report to {report_path}")


@app.command()
def ingest(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Path to sqlite database for intermediate data."),
    dataset: str = typer.Option("yasalma/vk-messages", help="HF dataset identifier or local path."),
    split: str = typer.Option("train", help="Dataset split to stream."),
    cache_dir: Path = typer.Option(Path("data/hf_cache"), help="Local cache dir for streamed dataset shards."),
    # text_field: str = typer.Option("text", help="Field containing the raw message."),
    # url_field: str = typer.Option("url", help="Field containing the message URL."),
    limit: Optional[int] = typer.Option(None, help="Optional limit for debugging."),
    batch_size: int = typer.Option(500, help="Number of cleaned segments to insert per transaction."),
    report_path: Optional[Path] = typer.Option(Path("reports/ingest_report.json"), help="Write JSON counters about filter reasons."),
    output_table: str = typer.Option("clean_segments", "--output-table", help="Destination table to write cleaned segments."),
):
    """
    Stream raw data, clean it, split long messages, and store into sqlite.
    """
    from collections import Counter
    import json

    cfg = DEFAULT_CONFIG
    name_list = load_name_list(cfg.name_list_path)
    _ensure_parent(cache_dir)
    download_mode, revision_tag = _prepare_dataset_cache(dataset, split, cache_dir)
    builder = datasets.load_dataset_builder(dataset, cache_dir=str(cache_dir))
    ds = datasets.load_dataset(
        dataset,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
        download_mode=download_mode,
    )
    total = builder.info.splits[split].num_examples if split in builder.info.splits else None

    buffered = []
    kept_segments = 0
    stats = Counter()
    processed_rows = 0
    _ensure_parent(db_path)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with open_db(db_path) as conn, progress:
        ensure_clean_table(conn, output_table)
        ensure_ingest_log_table(conn)

        dataset_label = f"{dataset}:{output_table}"
        task = progress.add_task("Ingesting", total=total)
        for _, row in enumerate(ds):
            if limit is not None and processed_rows >= limit:
                break
            progress.advance(task)

            raw_text = row['message']
            source_url = row['url']
            row_hash = _row_hash(source_url, str(raw_text))
            if not try_log_row(conn, row_hash, dataset_label, split, revision_tag, source_url):
                stats["rows_skipped_seen"] += 1
                continue

            segments = process_text(str(raw_text), str(source_url), cfg, name_list, stats=stats)
            buffered.extend(segments)
            processed_rows += 1

            if len(buffered) >= batch_size:
                inserted = insert_clean_segments(conn, buffered, output_table)
                kept_segments += inserted
                buffered = []
                conn.commit()

        if buffered:
            inserted = insert_clean_segments(conn, buffered, output_table)
            kept_segments += inserted
        conn.commit()

    typer.echo(f"Ingested {kept_segments} segments into {db_path}")
    if report_path:
        _ensure_parent(report_path)
        stats["segments_inserted"] = kept_segments
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote ingest report to {report_path}")


@app.command()
def dedup(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("clean_segments", help="Table to read from."),
    output_table: str = typer.Option("dedup_segments", "--output-table", help="Destination table for deduplicated rows."),
    distance: int = typer.Option(DEFAULT_CONFIG.near_dup_distance, help="SimHash Hamming distance for near-duplicate removal."),
    short_distance: int = typer.Option(1, help="Stricter SimHash distance for short texts."),
    short_token_threshold: int = typer.Option(8, help="Token count threshold for short text handling."),
    batch_size: int = typer.Option(1000, help="Rows to process per batch."),
    report_path: Optional[Path] = typer.Option(Path("reports/dedup_report.json"), help="Write JSON counters for dedup stage."),
):
    """
    Run exact+near duplicate filtering from clean_segments into dedup_segments.
    """
    total = 0
    kept = 0
    from collections import Counter
    import json
    stats = Counter()
    _ensure_parent(db_path)
    _ensure_distinct_tables(source_table, output_table, "dedup")
    with open_db(db_path) as conn:
        ensure_clean_table(conn, source_table)
        ensure_dedup_table(conn, output_table)

        # Build index incrementally to avoid loading everything at once.
        from simhash import SimhashIndex  # lazy import to keep CLI startup fast

        search_k = max(distance, short_distance)
        index = SimhashIndex([], k=search_k)
        seen_norm = set()
        id_to_sh = {}

        # Seed index with existing output rows to keep incremental runs immutable.
        existing_ids = collect_ids(conn, output_table)
        for rows in iterate_table(conn, output_table, batch_size):
            for row in rows:
                seen_norm.add(row["norm_hash"])
                sh = simhash_from_text(row["text"])
                index.add(row["id"], sh)
                id_to_sh[row["id"]] = sh

        for rows in track(iterate_table(conn, source_table, batch_size), description="Dedup"):
            accepted = []
            for row in rows:
                total += 1
                stats["rows_total"] += 1
                if row["id"] in existing_ids:
                    stats["already_present"] += 1
                    continue
                norm_hash = row["norm_hash"]
                if norm_hash in seen_norm:
                    stats["exact_dups"] += 1
                    continue
                sh = simhash_from_text(row["text"])
                token_len = len(tokenize_words(row["text"]))
                effective_k = short_distance if token_len <= short_token_threshold else distance
                near_ids = index.get_near_dups(sh)
                dup_found = False
                for nid in near_ids:
                    other = id_to_sh.get(nid)
                    if other is None:
                        continue
                    if sh.distance(other) <= effective_k:
                        dup_found = True
                        break
                if dup_found:
                    stats["near_dups"] += 1
                    continue
                index.add(row["id"], sh)
                id_to_sh[row["id"]] = sh
                seen_norm.add(norm_hash)
                accepted.append(row)

            if accepted:
                kept += copy_rows_to_dedup(conn, accepted, output_table)
                stats["rows_kept"] += len(accepted)

    typer.echo(f"Deduped: kept {kept} of {total} rows (distance={distance})")
    if report_path:
        _ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote dedup report to {report_path}")


@app.command()
def toxicity(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("dedup_segments", help="Table to read from."),
    output_table: str = typer.Option("toxicity_segments", "--output-table", help="Destination table for toxicity-scored rows."),
    threshold: Optional[float] = typer.Option(None, help="Optional threshold: drop texts with toxicity_score >= threshold."),
    batch_size: int = typer.Option(32, help="Batch size for the model."),
    device: int = typer.Option(-1, help="Transformers device id (-1 = CPU)."),
    report_path: Optional[Path] = typer.Option(Path("reports/toxicity_report.json"), help="Write JSON counters for toxicity stage."),
):
    """
    Score toxicity; keep all by default, optionally drop at given threshold. Stores label and score.
    """
    from collections import Counter
    import json
    classifier = load_toxicity_pipeline(device=device)
    total = 0
    kept = 0
    stats = Counter()
    _ensure_parent(db_path)
    _ensure_distinct_tables(source_table, output_table, "toxicity")
    with open_db(db_path) as conn:
        ensure_toxicity_table(conn, output_table)
        existing_ids = collect_ids(conn, output_table)
        for rows in track(iterate_table(conn, source_table, batch_size), description="toxicity"):
            total += len(rows)
            stats["rows_total"] += len(rows)
            # Filter out already-processed rows.
            pending_rows = [row for row in rows if row["id"] not in existing_ids]
            preds = predict_toxicity(classifier, [row["text"] for row in pending_rows])
            accepted = []
            for row, (label, score) in zip(pending_rows, preds):
                drop = threshold is not None and score >= threshold
                if drop:
                    stats["dropped_threshold"] += 1
                    continue
                row_data = dict(row)
                row_data["toxicity_label"] = label
                row_data["toxicity_score"] = score
                stats["toxic" if label else "non_toxic"] += 1
                accepted.append(row_data)
                # if label == 1:
                    # print(f"toxicity score: {score}: {row['text']}")
            if accepted:
                kept += copy_rows_to_toxicity(conn, accepted, output_table)
            conn.commit()
    typer.echo(
        f"Toxicity scoring: kept {kept} of {total} rows"
        + (f" (threshold={threshold})" if threshold is not None else " (no dropping)")
    )
    if report_path:
        _ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote toxicity report to {report_path}")


@app.command("gemini")
def gemini_score(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("toxicity_segments", help="Table to read from."),
    output_table: str = typer.Option("gemini_segments", "--output-table", help="Destination table for Gemini-scored rows."),
    keys_path: Path = typer.Option(DEFAULT_CONFIG.gemini_keys_path, help="YAML file containing Gemini API keys (list or mapping)."),
    model: str = typer.Option(DEFAULT_CONFIG.gemini_model, help="Gemini model name."),
    prompt_path: Optional[Path] = typer.Option(None, help="Optional path to custom prompt; defaults to built-in."),
    batch_size: int = typer.Option(8, help="Rows per batch."),
    max_rows: Optional[int] = typer.Option(None, help="Optional limit for debugging."),
    report_path: Optional[Path] = typer.Option(Path("reports/gemini_report.json"), help="Write JSON counters for Gemini stage."),
):
    """
    Call Gemini to score texts; stores JSON response per row in gemini_segments.
    """
    from collections import Counter

    _ensure_parent(db_path)
    prompt = DEFAULT_PROMPT
    if prompt_path:
        prompt = prompt_path.read_text(encoding="utf-8")

    keys = load_keys(keys_path)
    rotator = KeyRotator(keys)
    stats = Counter()
    processed = 0
    stop = False

    with open_db(db_path) as conn:
        _ensure_distinct_tables(source_table, output_table, "gemini")
        ensure_gemini_table(conn, output_table)
        existing_ids = collect_ids(conn, output_table)
        for rows in track(iterate_table(conn, source_table, batch_size), description="gemini"):
            pending_rows = [row for row in rows if row["id"] not in existing_ids]
            accepted = []
            for row in pending_rows:
                if max_rows and processed >= max_rows:
                    stop = True
                    break
                processed += 1
                stats["rows_total"] += 1
                last_error = None
                parsed = {}

                for _ in range(len(keys)):
                    key = rotator.next_key()
                    try:
                        client = configure_client(key, model)
                        resp_text = call_gemini(client, prompt, row["text"])
                        parsed = parse_json_response(resp_text)
                        stats["rows_ok"] += 1
                        break
                    except Exception as exc:  # pragma: no cover - network errors
                        last_error = str(exc)
                        stats["errors"] += 1
                        continue

                row_data = dict(row)
                row_data["gemini_json"] = json.dumps(parsed, ensure_ascii=False) if parsed else None
                row_data["gemini_error"] = last_error
                accepted.append(row_data)

            if accepted:
                copy_rows_to_gemini(conn, accepted, output_table)

            if stop:
                break

    typer.echo(
        f"Gemini scoring finished; processed {processed} rows; ok={stats.get('rows_ok',0)}, errors={stats.get('errors',0)}"
    )
    if report_path:
        _ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote Gemini report to {report_path}")


def _stream_table_to_parquet(
    conn: sqlite3.Connection, table: str, output: Path, batch_size: int = 2000
) -> int:
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    written = 0
    with pq.ParquetWriter(output, schema) as writer:
        for rows in iterate_table(conn, table, batch_size):
            data = {"id": [r["id"] for r in rows], "text": [r["text"] for r in rows]}
            batch = pl.DataFrame(data)
            writer.write_table(batch.to_arrow())
            written += len(rows)
    return written


@app.command()
def export(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    output: Path = typer.Argument(..., help="Destination parquet file."),
    table: str = typer.Option(
        "toxicity_segments",
        help="Table to export. Will fall back to dedup_segments if empty.",
    ),
):
    """
    Export the chosen table to a parquet file containing id and text columns.
    """
    _ensure_parent(db_path)
    with open_db(db_path) as conn:
        # Fallback if toxicity stage has not been run.
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        selected_table = table
        if count == 0 and table != "dedup_segments":
            selected_table = "dedup_segments"
            count = conn.execute(f"SELECT COUNT(*) FROM {selected_table}").fetchone()[0]
        if count == 0:
            typer.echo(f"No data to export in {table} or dedup_segments.")
            raise typer.Exit(code=1)

        written = _stream_table_to_parquet(conn, selected_table, output)
    typer.echo(f"Exported {written} rows from {selected_table} to {output}")


if __name__ == "__main__":
    app()
