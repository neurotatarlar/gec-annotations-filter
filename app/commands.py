import json
import hashlib
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sqlite3
from pydantic import TypeAdapter
import time

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
    ScoredItem,
    LabelPayload,
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
    ensure_gemini_table,
    ensure_export_log_table,
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
    from rich.console import Console
    
    console = Console()

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
        console.print("No usable Gemini keys after filtering expired keys.")
        raise typer.Exit(code=1)
    rotator = KeyRotator(keys)
    stats = Counter()
    processed = 0
    stop = False
    batch_size = max(1, batch_size)
    token_totals = Counter()

    def _usage_val(usage, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            try:
                return int(usage.get(key) or 0)
            except Exception:
                return 0
        try:
            return int(getattr(usage, key, 0) or 0)
        except Exception:
            return 0

    def _record_usage(usage) -> None:
        prompt_tokens = _usage_val(usage, "prompt_token_count")
        output_tokens = _usage_val(usage, "candidates_token_count")
        total_tokens = _usage_val(usage, "total_token_count")
        token_totals["input"] += prompt_tokens
        token_totals["output"] += output_tokens
        token_totals["total"] += total_tokens
        token_totals["requests"] += 1
        console.print(
            f"Gemini tokens: input={prompt_tokens}, output={output_tokens}, total={total_tokens}",
            style='green',
        )

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
                # console.print(prompt_text, style="honeydew2")
                resp_text, usage = call_gemini(client, model, prompt_text, schema=list[ScoredItem])
                # console.print(resp_text, style="thistle1")
                _record_usage(usage)
                ta = TypeAdapter(List[ScoredItem])
                res = ta.validate_python(json.loads(resp_text))
                time.sleep(5)
                return res
            except Exception as exc:  # pragma: no cover - network errors
                import traceback
                last_error = str(exc)
                if "UNAVAILABLE" in last_error or "overloaded" in last_error.lower():
                    console.print("Gemini overloaded (503). Sleeping 60s before retry...")
                    time.sleep(60)
                    continue
                if "DEADLINE_EXCEEDED" in last_error or "timed out" in last_error.lower():
                    console.print(f"Gemini timed out (504) on chunk size {len(chunk)}; will reduce batch size.")
                    raise TimeoutError(last_error)
                if "RESOURCE_EXHAUSTED" in last_error or "quota" in last_error.lower():
                    console.print(f"Key appears exhausted; marking as expired: {key}")
                    traceback.print_exc()
                    expired_keys.add(key)
                    _save_expired(expired_keys)
                    keys = [k for k in keys if k != key]
                    if not keys:
                        console.print("All Gemini keys exhausted.")
                        raise typer.Exit(code=1)
                    rotator = KeyRotator(keys)
                    time.sleep(5)
                    continue
                traceback.print_exc()
                time.sleep(5)
                continue
        if last_error and ("DEADLINE_EXCEEDED" in last_error or "timed out" in last_error.lower()):
            raise TimeoutError(last_error)
        return {}, last_error

    with open_db(db_path) as conn:
        if source_table == "toxicity_segments":
            ensure_toxicity_table(conn, source_table)
        ensure_llm_table(conn, output_table)
        pending_sql = f"""
        SELECT s.*
        FROM {source_table} AS s
        LEFT JOIN {output_table} AS o ON s.id = o.id
        WHERE o.id IS NULL AND COALESCE(s.toxicity_label, 0) != 1 AND COALESCE(s.gemini_skipped,0)=0
        ORDER BY s.rowid
        LIMIT ?
        """
        chunk_num = 0
        current_batch_size = max(1, batch_size)
        try:
            while True:
                if max_rows and processed >= max_rows:
                    console.print("Reached max_rows limit.")
                    break

                rows = conn.execute(pending_sql, (current_batch_size,)).fetchall()
                if not rows:
                    console.print("No more rows to process.")
                    break

                chunk = [dict(r) for r in rows]
                console.print(
                    f"Sending chunk {chunk_num + 1} with batch_size={len(chunk)} (current limit={current_batch_size})"
                )
                try:
                    chunk_num += 1
                    accepted: List[Dict] = []
                    scored_items = {s.id: s.labels for s in score_chunk(chunk) if hasattr(s, "id") and hasattr(s, "labels")}

                    processed += len(chunk)
                    stats["rows_total"] += len(chunk)

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

                    chunk_matched = len(accepted)
                    chunk_missing = len(chunk) - chunk_matched
                    console.print(
                        f"Chunk {chunk_num} processed: total={len(chunk)}, matched={chunk_matched}, missing={chunk_missing}, cumulative_ok={stats.get('rows_ok',0)}"
                    )

                    if accepted:
                        copy_rows_to_gemini(conn, accepted, output_table)
                        conn.commit()

                    if stop:
                        break
                except TimeoutError:
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        console.print(f"Timeout encountered; reducing batch size to {current_batch_size}")
                        continue
                    else:
                        # batch size already 1; mark these rows as skipped
                        ids_to_skip = [r["id"] for r in rows]
                        conn.executemany(
                            f"UPDATE {source_table} SET gemini_skipped=1 WHERE id=?",
                            [(rid,) for rid in ids_to_skip],
                        )
                        conn.commit()
                        console.print(f"Marked {len(ids_to_skip)} rows as gemini_skipped after repeated timeouts.")
                        current_batch_size = 1
                        continue
        finally:
            reqs = token_totals.get("requests", 0)
            if reqs:
                prompt_total = token_totals.get("input", 0)
                output_total = token_totals.get("output", 0)
                total_total = token_totals.get("total", 0)
                console.print(
                    f"Gemini token usage totals: requests={reqs}, input={prompt_total}, output={output_total}, total={total_total}",
                    style='green',
                )
                console.print(
                    "Gemini token usage averages: "
                    f"input={prompt_total/reqs:.2f}, output={output_total/reqs:.2f}, total={total_total/reqs:.2f}",
                    style='green',
                )
            else:
                console.print("Gemini token usage: no requests made.", style='green')

    console.print(
        f"Gemini scoring finished; processed {processed} rows; ok={stats.get('rows_ok',0)}, errors={stats.get('errors',0)}",
        style='green',
    )
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"Wrote Gemini report to {report_path}")


def export_parquet_cmd(
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


def prepare_import_cmd(
    db_path: Path,
    source_table: str,
    output: Path,
    limit: Optional[int],
    extreme_share: float,
    dry_run: bool = False,
    keep_proportions: bool = False,
):
    """
    Produce a JSON array of {id, text} ready for annotation, with basic quality filters,
    non-toxic only, and a small proportion of low/high error_density examples.
    """
    extreme_share = max(0.0, min(0.5, extreme_share))
    ensure_parent(db_path)
    if not dry_run:
        ensure_parent(output)

    allowed_langs = {"tatar", "mixed"}
    ta_label = TypeAdapter(LabelPayload)
    target = "prepare_import"

    with open_db(db_path) as conn:
        ensure_gemini_table(conn, source_table)
        ensure_export_log_table(conn)

        exported_ids = {
            row["id"]
            for row in conn.execute("SELECT id FROM export_log WHERE target = ?", (target,)).fetchall()
        }

        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({source_table})")}
        has_gemini_skipped = "gemini_skipped" in columns
        skip_clause = "AND COALESCE(s.gemini_skipped,0)=0" if has_gemini_skipped else ""

        rows = conn.execute(
            f"""
            SELECT s.id, s.text, s.gemini_json
            FROM {source_table} AS s
            WHERE s.gemini_json IS NOT NULL
              AND COALESCE(s.toxicity_label, 0) = 0
              {skip_clause}
            ORDER BY s.rowid
            """
        ).fetchall()

        buckets = {"low": [], "medium": [], "high": []}
        skipped = 0
        for row in rows:
            if row["id"] in exported_ids:
                continue
            try:
                labels = ta_label.validate_json(row["gemini_json"])
            except Exception:
                skipped += 1
                continue
            if labels.main_language not in allowed_langs:
                continue
            if labels.noise_score > 0.35:
                continue
            if labels.meaning_clarity < 0.35:
                continue
            if labels.overall_gec_usefulness < 0.4:
                continue
            density = labels.error_density
            if density not in buckets:
                continue
            score = (
                0.50 * labels.overall_gec_usefulness
                + 0.20 * labels.meaning_clarity
                + 0.15 * labels.tatar_prob
                + 0.10 * labels.error_share
                + 0.05 * labels.non_fluent_prob
                - 0.20 * labels.noise_score
            )
            buckets[density].append({"id": row["id"], "text": row["text"], "score": score})

        total_candidates = sum(len(v) for v in buckets.values())
        if total_candidates == 0:
            typer.echo("No candidates available for prepare_import after filtering.")
            return

        if not keep_proportions and limit is None:
            selected_med = sorted(buckets["medium"], key=lambda x: (-x["score"], x["id"]))
            selected_low = sorted(buckets["low"], key=lambda x: (-x["score"], x["id"]))
            selected_high = sorted(buckets["high"], key=lambda x: (-x["score"], x["id"]))
            selected = selected_med + selected_low + selected_high
            if dry_run:
                typer.echo(
                    f"Dry-run prepare_import: total_candidates={total_candidates}, selected={len(selected)} "
                    f"(med={len(selected_med)}, low={len(selected_low)}, high={len(selected_high)}), skipped_invalid={skipped}"
                )
                return
            random.shuffle(selected)
            output_data = [{"id": r["id"], "text": r["text"]} for r in selected]
            output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
            typer.echo(f"Wrote {len(selected)} rows to {output} (skipped {skipped} invalid Gemini payloads).")
            typer.echo(
                f"Selection breakdown: medium={len(selected_med)}, low={len(selected_low)}, high={len(selected_high)}; "
                f"kept_proportions={keep_proportions}, limit={limit or 'none'}"
            )
            conn.executemany(
                "INSERT OR IGNORE INTO export_log (id, target) VALUES (?, ?)",
                [(r["id"], target) for r in selected],
            )
            conn.commit()
            return

        target_total = limit if limit is not None else total_candidates
        target_total = max(0, target_total)
        share = extreme_share
        low_target = min(len(buckets["low"]), int(round(target_total * share)))
        high_target = min(len(buckets["high"]), int(round(target_total * share)))
        remaining = max(0, target_total - low_target - high_target)
        med_target = min(len(buckets["medium"]), remaining)
        remaining -= med_target

        sorted_low = sorted(buckets["low"], key=lambda x: (-x["score"], x["id"]))
        sorted_high = sorted(buckets["high"], key=lambda x: (-x["score"], x["id"]))
        sorted_med = sorted(buckets["medium"], key=lambda x: (-x["score"], x["id"]))

        selected_low = sorted_low[:low_target]
        selected_high = sorted_high[:high_target]
        selected_med = sorted_med[:med_target]

        if not keep_proportions and remaining > 0:
            # fill from leftover low/high evenly
            low_left = sorted_low[low_target:]
            high_left = sorted_high[high_target:]
            take_more_low = min(len(low_left), (remaining + 1) // 2)
            take_more_high = min(len(high_left), remaining - take_more_low)
            selected_low += low_left[:take_more_low]
            selected_high += high_left[:take_more_high]
            remaining -= take_more_low + take_more_high

        selected = selected_med + selected_low + selected_high
        if not keep_proportions and remaining > 0 and len(selected) < target_total:
            # If still short, fill from any remaining medium first, then others.
            med_left = sorted_med[med_target:]
            extra = med_left[:remaining]
            selected += extra

        selected = selected[:target_total]

        if not selected:
            typer.echo("No rows selected for prepare_import.")
            return

        if dry_run:
            typer.echo(
                f"Dry-run prepare_import: total_candidates={total_candidates}, selected={len(selected)} "
                f"(med={len(selected_med)}, low={len(selected_low)}, high={len(selected_high)}), skipped_invalid={skipped}"
            )
            return

        random.shuffle(selected)
        output_data = [{"id": r["id"], "text": r["text"]} for r in selected]
        output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {len(selected)} rows to {output} (skipped {skipped} invalid Gemini payloads).")
        typer.echo(
            f"Selection breakdown: medium={len(selected_med)}, low={len(selected_low)}, high={len(selected_high)}; "
            f"kept_proportions={keep_proportions}, limit={limit or 'none'}"
        )

        conn.executemany(
            "INSERT OR IGNORE INTO export_log (id, target) VALUES (?, ?)",
            [(r["id"], target) for r in selected],
        )
        conn.commit()
