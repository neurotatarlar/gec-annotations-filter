import json
import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sqlite3
from pydantic import TypeAdapter

import datasets
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from google import genai
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.progress import track

from .config import DEFAULT_CONFIG
from .dedup import simhash_from_text
from pydantic import TypeAdapter

from .gemini import (
    DEFAULT_PROMPT,
    KeyRotator,
    call_gemini,
    load_keys,
    build_batch_prompt,
    ScoredItem
)
from .processing import (
    ProcessedSegment,
    load_name_list,
    process_text,
    tokenize_words,
)
from .storage import (
    copy_rows_to_dedup,
    copy_rows_to_gemini,
    copy_rows_to_toxicity,
    ensure_llm_table,
    collect_ids,
    ensure_ingest_log_table,
    insert_clean_segments,
    iterate_table,
    open_db,
    ensure_clean_table,
    ensure_dedup_table,
    ensure_toxicity_table,
    try_log_row,
)
from .toxicity import load_toxicity_pipeline, predict_toxicity


# Utility helpers
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_field(row: dict, field: str, fallback: str, idx: int) -> str:
    for key in (field, fallback, "message_url", "permalink", "id"):
        value = row.get(key)
        if value:
            return str(value)
    return f"{fallback}:{idx}"


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _revision_file(cache_dir: Path, dataset: str, split: str) -> Path:
    return cache_dir / "revisions" / f"{_safe_token(dataset)}__{_safe_token(split)}.txt"


def _fetch_remote_revision(dataset: str) -> Optional[str]:
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


def prepare_dataset_cache(dataset: str, split: str, cache_dir: Path) -> tuple[datasets.DownloadMode, str]:
    ensure_parent(cache_dir)
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


def row_hash(source_id: str, raw_text: str) -> str:
    payload = (source_id or "") + "\n" + raw_text
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# Command implementations
def ingest_cmd(
    db_path: Path,
    dataset: str,
    split: str,
    cache_dir: Path,
    text_field: str,
    url_field: str,
    limit: Optional[int],
    batch_size: int,
    report_path: Optional[Path],
    output_table: str,
):
    from collections import Counter

    cfg = DEFAULT_CONFIG
    name_list = load_name_list(cfg.name_list_path)
    ensure_parent(cache_dir)
    download_mode, revision_tag = prepare_dataset_cache(dataset, split, cache_dir)
    builder = datasets.load_dataset_builder(dataset, cache_dir=str(cache_dir))
    ds = datasets.load_dataset(
        dataset,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
        download_mode=download_mode,
    )
    total = builder.info.splits[split].num_examples if split in builder.info.splits else None

    buffered: List[ProcessedSegment] = []
    kept_segments = 0
    stats = Counter()
    processed_rows = 0
    ensure_parent(db_path)
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

        task = progress.add_task("Ingesting", total=total)
        for idx, row in enumerate(ds):
            if limit is not None and processed_rows >= limit:
                break
            progress.advance(task)

            raw_text = row.get(text_field) or row.get("text") or row.get("message") or ""
            source_url = resolve_field(row, url_field, "url", idx)
            rhash = row_hash(source_url, str(raw_text))
            if not try_log_row(conn, rhash, dataset + ":" + output_table, split, revision_tag, source_url):
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
        ensure_parent(report_path)
        stats["segments_inserted"] = kept_segments
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote ingest report to {report_path}")


def dedup_cmd(
    db_path: Path,
    source_table: str,
    output_table: str,
    distance: int,
    short_distance: int,
    short_token_threshold: int,
    batch_size: int,
    report_path: Optional[Path],
):
    total = 0
    kept = 0
    from collections import Counter

    stats = Counter()
    ensure_parent(db_path)
    with open_db(db_path) as conn:
        ensure_clean_table(conn, source_table)
        ensure_dedup_table(conn, output_table)

        from simhash import SimhashIndex

        search_k = max(distance, short_distance)
        index = SimhashIndex([], k=search_k)
        seen_norm = set()
        id_to_sh = {}

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

    typer.echo(f"Deduped: kept {kept} of {total} rows (distance={distance}, short_distance={short_distance})")
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote dedup report to {report_path}")


def toxicity_cmd(
    db_path: Path,
    source_table: str,
    output_table: str,
    threshold: Optional[float],
    batch_size: int,
    device: int,
    report_path: Optional[Path],
):
    from collections import Counter

    classifier = load_toxicity_pipeline(device=device)
    total = 0
    kept = 0
    stats = Counter()
    ensure_parent(db_path)
    with open_db(db_path) as conn:
        ensure_toxicity_table(conn, output_table)
        existing_ids = collect_ids(conn, output_table)
        for rows in track(iterate_table(conn, source_table, batch_size), description="toxicity"):
            total += len(rows)
            stats["rows_total"] += len(rows)
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
            if accepted:
                kept += copy_rows_to_toxicity(conn, accepted, output_table)
            conn.commit()
    typer.echo(
        f"Toxicity scoring: kept {kept} of {total} rows"
        + (f" (threshold={threshold})" if threshold is not None else " (no dropping)")
    )
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote toxicity report to {report_path}")


def gemini_cmd(
    db_path: Path,
    source_table: str,
    output_table: str,
    keys_path: Path,
    model: str,
    prompt_path: Optional[Path],
    batch_size: int,
    max_rows: Optional[int],
    report_path: Optional[Path],
):
    from collections import Counter

    ensure_parent(db_path)
    prompt = DEFAULT_PROMPT
    if prompt_path:
        prompt = prompt_path.read_text(encoding="utf-8")

    expired_path = keys_path.with_suffix(keys_path.suffix + ".expired")

    def _load_expired() -> set[str]:
        if not expired_path.exists():
            return set()
        try:
            return set(json.loads(expired_path.read_text()))
        except Exception:
            try:
                return set(k.strip() for k in expired_path.read_text().splitlines() if k.strip())
            except Exception:
                return set()

    def _save_expired(expired: set[str]) -> None:
        expired_path.write_text(json.dumps(sorted(expired), ensure_ascii=False, indent=2), encoding="utf-8")

    expired_keys = _load_expired()
    keys = [k for k in load_keys(keys_path) if k not in expired_keys]
    if not keys:
        typer.echo("No usable Gemini keys after filtering expired keys.")
        raise typer.Exit(code=1)
    rotator = KeyRotator(keys)
    stats = Counter()
    processed = 0
    stop = False
    batch_size = max(1, batch_size)

    def score_chunk(chunk: List[Dict]) -> Tuple[Dict[str, Dict], Optional[str]]:
        last_error: Optional[str] = None
        tries = 0
        nonlocal keys, rotator, expired_keys
        while keys and tries < len(keys) + 1:
            tries += 1
            key = rotator.next_key()
            try:
                client = genai.Client(api_key=key)
                prompt_text = build_batch_prompt(prompt, [{"id": r["id"], "text": r["text"]} for r in chunk])
                print(prompt_text)
                resp_text = call_gemini(client, model, prompt_text, schema=list[ScoredItem])
                print(resp_text)
                ta = TypeAdapter(List[ScoredItem])
                return ta.validate_python(json.loads(resp_text))
            except Exception as exc:  # pragma: no cover - network errors
                last_error = str(exc)
                if "UNAVAILABLE" in last_error or "overloaded" in last_error.lower():
                    typer.echo("Gemini overloaded (503). Sleeping 60s before retry...")
                    time.sleep(60)
                    continue
                if "DEADLINE_EXCEEDED" in last_error or "timed out" in last_error.lower():
                    typer.echo("Gemini timed out (504); retrying with next key...")
                    continue
                if "RESOURCE_EXHAUSTED" in last_error or "quota" in last_error.lower():
                    typer.echo(f"Key appears exhausted; marking as expired: {key}")
                    expired_keys.add(key)
                    _save_expired(expired_keys)
                    keys = [k for k in keys if k != key]
                    if not keys:
                        typer.echo("All Gemini keys exhausted.")
                        return {}, last_error
                    rotator = KeyRotator(keys)
                    continue
                raise exc
        return {}, last_error

    with open_db(db_path) as conn:
        ensure_llm_table(conn, output_table)
        existing_ids = collect_ids(conn, output_table)
        for rows in track(iterate_table(conn, source_table, batch_size), description="Scoring by Gemini"):
            pending_rows = []
            for row in rows:
                row_d = dict(row)
                if row_d["id"] in existing_ids:
                    continue
                if row_d.get("toxicity_label") == 1:
                    continue
                pending_rows.append(row_d)
            accepted: List[Dict] = []
            idx = 0
            while idx < len(pending_rows):
                if max_rows and processed >= max_rows:
                    stop = True
                    break
                chunk = pending_rows[idx : idx + batch_size]
                processed += len(chunk)
                stats["rows_total"] += len(chunk)

                scored_items = {s.id: s.labels for s in score_chunk(chunk)}

                for row in chunk:
                    payload = scored_items.get(str(row["id"]))
                    if not payload:
                        continue
                    row_data = dict(row)
                    row_data["gemini_json"]=payload.model_dump_json(indent=None, by_alias=True, exclude_none=True, exclude_unset=True, ensure_ascii=False)
                    if payload:
                        stats["rows_ok"] += 1
                    else:
                        stats["errors"] += 1
                    accepted.append(row_data)

                idx += len(chunk)

            if accepted:
                copy_rows_to_gemini(conn, accepted, output_table)

            if stop:
                break

    typer.echo(
        f"Gemini scoring finished; processed {processed} rows; ok={stats.get('rows_ok',0)}, errors={stats.get('errors',0)}"
    )
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote Gemini report to {report_path}")


def export_cmd(
    db_path: Path,
    output: Path,
    table: str,
):
    ensure_parent(db_path)
    with open_db(db_path) as conn:
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
