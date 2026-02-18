"""Pipeline command implementations for ingest, dedup, scoring, and export."""

import json
import math
import hashlib
import random
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
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

from .gemini import (
    DEFAULT_PROMPT,
    call_gemini,
    load_account_keys,
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
    """Ensure the parent directory for a path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_field(row: dict, field: str, fallback: str, idx: int) -> str:
    """Pick a stable row identifier from known fields or a fallback."""
    for key in (field, fallback, "message_url", "permalink", "id"):
        value = row.get(key)
        if value:
            return str(value)
    return f"{fallback}:{idx}"


def _safe_token(value: str) -> str:
    """Sanitize strings for use in filenames."""
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _revision_file(cache_dir: Path, dataset: str, split: str) -> Path:
    """Return the path to the revision marker file for a dataset/split."""
    return cache_dir / "revisions" / f"{_safe_token(dataset)}__{_safe_token(split)}.txt"


def _fetch_remote_revision(dataset: str) -> Optional[str]:
    """Fetch the remote revision hash for an HF dataset when available."""
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
    """
    Warm the HF dataset cache and record the revision.

    Returns the download mode used and a revision tag (remote hash, cached hash,
    or "local" if the dataset is a local path).
    """
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
    """Compute a stable SHA1 over the source id and raw text."""
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
    """
    Stream a dataset, clean and segment text, and insert into the clean table.

    Uses an ingest_log to skip previously processed rows for the same dataset
    and output table, and optionally emits a JSON report.
    """
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
    """Deduplicate rows using SimHash with stricter handling for short texts."""
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
    """Score toxicity for rows and optionally drop those above a threshold."""
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
    max_batch_size: int,
    workers: int,
    account_cooldown_seconds: int,
    max_rows: Optional[int],
    report_path: Optional[Path],
):
    """
    Score rows with Gemini and store JSON labels in the output table.

    Handles account-aware parallel requests, adaptive batch sizing, retries,
    and incremental writes.
    """
    from collections import Counter
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from rich.console import Console

    console = Console()

    ensure_parent(db_path)
    prompt = DEFAULT_PROMPT
    if prompt_path:
        prompt = prompt_path.read_text(encoding="utf-8")

    expired_path = keys_path.with_suffix(keys_path.suffix + ".expired")

    def _load_expired() -> set[str]:
        """Load keys already marked exhausted on previous runs."""
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
        """Persist exhausted keys so retries skip them."""
        expired_path.write_text(json.dumps(sorted(expired), ensure_ascii=False, indent=2), encoding="utf-8")

    def _log(message: str, style: Optional[str] = None, account: Optional[str] = None) -> None:
        """Print a log line with thread/account context."""
        thread_name = threading.current_thread().name
        context = thread_name if account is None else f"{thread_name}|{account}"
        text = f"{context}: {message}"
        if style:
            console.print(text, style=style)
        else:
            console.print(text)

    class ResponseParseError(RuntimeError):
        """Raised when Gemini returns invalid JSON for the expected schema."""
        pass

    def _usage_val(usage, key: str) -> int:
        """Read an integer usage field from dict- or object-like metadata."""
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

    def _usage_from_meta(usage) -> tuple[int, int, int]:
        """Extract usage counters from Gemini metadata."""
        prompt_tokens = _usage_val(usage, "prompt_token_count")
        output_tokens = _usage_val(usage, "candidates_token_count")
        total_tokens = _usage_val(usage, "total_token_count")
        return prompt_tokens, output_tokens, total_tokens

    def _record_usage(prompt_tokens: int, output_tokens: int, total_tokens: int) -> None:
        """Accumulate and print per-request token usage counters."""
        token_totals["input"] += prompt_tokens
        token_totals["output"] += output_tokens
        token_totals["total"] += total_tokens
        token_totals["requests"] += 1
        _log(
            f"Gemini tokens: input={prompt_tokens}, output={output_tokens}, total={total_tokens}",
            style='green',
        )

    def _next_account_key(account: Dict[str, Any]) -> str:
        """Rotate keys within one account."""
        key = account["keys"][account["idx"] % len(account["keys"])]
        account["idx"] = (account["idx"] + 1) % len(account["keys"])
        return key

    def score_chunk(
        account: Dict[str, Any],
        chunk: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Score one chunk with retry logic within a single account.

        Returns a structured result so the caller can update counters and decide
        whether to shrink batch size or disable exhausted accounts.
        """
        last_error: Optional[str] = None
        exhausted_in_call: List[str] = []
        tries = 0
        while account["keys"] and tries < len(account["keys"]) + 1:
            tries += 1
            key = _next_account_key(account)
            try:
                client = genai.Client(api_key=key)
                prompt_text = build_batch_prompt(prompt, [{"id": r["id"], "text": r["text"]} for r in chunk])
                resp_text, usage = call_gemini(client, model, prompt_text, schema=list[ScoredItem])
                prompt_tokens, output_tokens, total_tokens = _usage_from_meta(usage)
                ta = TypeAdapter(List[ScoredItem])
                try:
                    items = ta.validate_python(json.loads(resp_text))
                except Exception as exc:
                    raise ResponseParseError(str(exc)) from exc
                time.sleep(5)
                return {
                    "status": "ok",
                    "items": items,
                    "error": None,
                    "usage": (prompt_tokens, output_tokens, total_tokens),
                    "exhausted_keys": exhausted_in_call,
                }
            except ResponseParseError as exc:
                return {
                    "status": "parse_error",
                    "items": [],
                    "error": str(exc),
                    "usage": (0, 0, 0),
                    "exhausted_keys": exhausted_in_call,
                }
            except Exception as exc:  # pragma: no cover - network errors
                import traceback

                last_error = str(exc)
                if "UNAVAILABLE" in last_error or "overloaded" in last_error.lower():
                    _log("Gemini overloaded (503). Sleeping 60s before retry...", account=account["name"])
                    time.sleep(60)
                    continue
                if "DEADLINE_EXCEEDED" in last_error or "timed out" in last_error.lower():
                    return {
                        "status": "timeout",
                        "items": [],
                        "error": last_error,
                        "usage": (0, 0, 0),
                        "exhausted_keys": exhausted_in_call,
                    }
                if "RESOURCE_EXHAUSTED" in last_error or "quota" in last_error.lower():
                    _log("Key appears exhausted; marking as expired.", account=account["name"])
                    exhausted_in_call.append(key)
                    account["keys"] = [k for k in account["keys"] if k != key]
                    if account["keys"]:
                        account["idx"] = account["idx"] % len(account["keys"])
                        if account_cooldown_seconds > 0:
                            _log(
                                f"Cooling down account for {account_cooldown_seconds}s before trying next key.",
                                account=account["name"],
                            )
                            time.sleep(account_cooldown_seconds)
                    continue
                traceback.print_exc()
                time.sleep(5)
                continue
        if not account["keys"]:
            return {
                "status": "account_exhausted",
                "items": [],
                "error": "Account has no usable keys left",
                "usage": (0, 0, 0),
                "exhausted_keys": exhausted_in_call,
            }
        return {
            "status": "error",
            "items": [],
            "error": last_error,
            "usage": (0, 0, 0),
            "exhausted_keys": exhausted_in_call,
        }

    expired_keys = _load_expired()
    raw_accounts = load_account_keys(keys_path)
    account_pool: List[Dict[str, Any]] = []
    for account_name, account_keys in raw_accounts.items():
        usable_keys = [k for k in account_keys if k not in expired_keys]
        if usable_keys:
            account_pool.append({"name": account_name, "keys": usable_keys, "idx": 0})
    if not account_pool:
        _log("No usable Gemini accounts after filtering expired keys.")
        raise typer.Exit(code=1)

    workers = max(1, workers)
    account_cooldown_seconds = max(0, account_cooldown_seconds)
    if workers > len(account_pool):
        _log(
            f"workers ({workers}) exceeds available accounts ({len(account_pool)}); clamping."
        )
    workers = min(workers, len(account_pool))

    stats = Counter()
    processed = 0
    token_totals = Counter()
    batch_size = max(1, batch_size)
    max_batch_size = max(1, max_batch_size)
    if batch_size > max_batch_size:
        _log(f"batch_size ({batch_size}) exceeds max_batch_size ({max_batch_size}); clamping.")
        batch_size = max_batch_size
    success_streak = 0
    success_threshold = 5
    accounts = deque(account_pool)
    all_accounts_exhausted = False

    with open_db(db_path) as conn:
        if source_table == "toxicity_segments":
            ensure_toxicity_table(conn, source_table)
        ensure_llm_table(conn, output_table)
        pending_count_sql = f"""
        SELECT COUNT(*)
        FROM {source_table} AS s
        LEFT JOIN {output_table} AS o ON s.id = o.id
        WHERE o.id IS NULL AND COALESCE(s.toxicity_label, 0) != 1 AND COALESCE(s.gemini_skipped,0)=0
        """
        total_count_sql = f"""
        SELECT COUNT(*)
        FROM {source_table} AS s
        WHERE COALESCE(s.toxicity_label, 0) != 1 AND COALESCE(s.gemini_skipped,0)=0
        """
        pending_sql = f"""
        SELECT s.*
        FROM {source_table} AS s
        LEFT JOIN {output_table} AS o ON s.id = o.id
        WHERE o.id IS NULL AND COALESCE(s.toxicity_label, 0) != 1 AND COALESCE(s.gemini_skipped,0)=0
        ORDER BY s.rowid
        LIMIT ?
        """
        total_rows = int(conn.execute(total_count_sql).fetchone()[0] or 0)
        remaining_rows = int(conn.execute(pending_count_sql).fetchone()[0] or 0)
        already_processed_rows = total_rows - remaining_rows
        stats["rows_total"] = total_rows
        stats["rows_remaining"] = remaining_rows
        stats["rows_already_processed"] = already_processed_rows
        _log(
            "Gemini rows: "
            f"total={total_rows}, already_processed={already_processed_rows}, remaining={remaining_rows}, "
            f"workers={workers}, accounts={len(account_pool)}, account_cooldown_seconds={account_cooldown_seconds}"
        )
        current_batch_size = batch_size
        with ThreadPoolExecutor(max_workers=workers) as executor:
            try:
                while True:
                    if max_rows is not None and processed >= max_rows:
                        _log("Reached max_rows limit.")
                        break
                    if not accounts:
                        all_accounts_exhausted = True
                        _log("All Gemini accounts are exhausted.")
                        break

                    round_workers = min(workers, len(accounts))
                    fetch_limit = current_batch_size * round_workers
                    if max_rows is not None:
                        fetch_limit = min(fetch_limit, max_rows - processed)
                        if fetch_limit <= 0:
                            _log("Reached max_rows limit.")
                            break

                    rows = conn.execute(pending_sql, (fetch_limit,)).fetchall()
                    if not rows:
                        _log("No more rows to process.")
                        break

                    row_dicts = [dict(r) for r in rows]
                    chunks: List[List[Dict[str, Any]]] = []
                    for idx in range(0, len(row_dicts), current_batch_size):
                        chunks.append(row_dicts[idx : idx + current_batch_size])

                    active_pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]], Any]] = []
                    for chunk in chunks[:round_workers]:
                        account = accounts.popleft()
                        fut = executor.submit(score_chunk, account, chunk)
                        active_pairs.append((account, chunk, fut))

                    for account, chunk, fut in active_pairs:
                        result = fut.result()
                        for exhausted_key in result.get("exhausted_keys", []):
                            if exhausted_key not in expired_keys:
                                expired_keys.add(exhausted_key)
                                _save_expired(expired_keys)

                        prompt_tokens, output_tokens, total_tokens = result.get("usage", (0, 0, 0))
                        if prompt_tokens or output_tokens or total_tokens:
                            _record_usage(prompt_tokens, output_tokens, total_tokens)

                        status = result.get("status")
                        if status == "ok":
                            processed += len(chunk)
                            accepted: List[Dict[str, Any]] = []
                            scored_items = {
                                s.id: s.labels for s in result.get("items", []) if hasattr(s, "id") and hasattr(s, "labels")
                            }
                            for row in chunk:
                                payload = scored_items.get(str(row["id"]))
                                if not payload:
                                    continue
                                row_data = dict(row)
                                row_data["gemini_json"] = payload.model_dump_json(
                                    indent=None,
                                    by_alias=True,
                                    exclude_none=True,
                                    exclude_unset=True,
                                    ensure_ascii=False,
                                )
                                stats["rows_ok"] += 1
                                accepted.append(row_data)
                            chunk_matched = len(accepted)
                            chunk_missing = len(chunk) - chunk_matched
                            remaining_rows = max(0, remaining_rows - chunk_matched)
                            stats["rows_remaining"] = remaining_rows
                            if accepted:
                                copy_rows_to_gemini(conn, accepted, output_table)
                                conn.commit()
                            missing_ratio = (chunk_missing / len(chunk)) if len(chunk) else 1.0
                            if missing_ratio <= 0.05:
                                success_streak += 1
                                if success_streak >= success_threshold and current_batch_size < max_batch_size:
                                    next_size = min(max_batch_size, int(math.ceil(current_batch_size * 1.2)))
                                    if next_size == current_batch_size and current_batch_size < max_batch_size:
                                        next_size = min(max_batch_size, current_batch_size + 1)
                                    if next_size != current_batch_size:
                                        _log(
                                            f"Success streak {success_streak} reached; increasing batch size to {next_size}"
                                        )
                                        current_batch_size = next_size
                                    success_streak = 0
                            else:
                                success_streak = 0
                        elif status == "timeout":
                            success_streak = 0
                            if current_batch_size > 1:
                                current_batch_size = max(1, int(math.floor(current_batch_size * 0.75)))
                                _log(
                                    f"Timeout encountered; reducing batch size to {current_batch_size}",
                                    account=account["name"],
                                )
                            else:
                                ids_to_skip = [r["id"] for r in chunk]
                                conn.executemany(
                                    f"UPDATE {source_table} SET gemini_skipped=1 WHERE id=?",
                                    [(rid,) for rid in ids_to_skip],
                                )
                                conn.commit()
                                remaining_rows = max(0, remaining_rows - len(ids_to_skip))
                                stats["rows_remaining"] = remaining_rows
                                _log(
                                    f"Marked {len(ids_to_skip)} rows as gemini_skipped after repeated timeouts."
                                )
                        elif status == "parse_error":
                            success_streak = 0
                            if current_batch_size > 1:
                                current_batch_size = max(1, int(math.floor(current_batch_size * 0.75)))
                                _log(
                                    f"Invalid JSON response; reducing batch size to {current_batch_size}",
                                    account=account["name"],
                                )
                            else:
                                ids_to_skip = [r["id"] for r in chunk]
                                conn.executemany(
                                    f"UPDATE {source_table} SET gemini_skipped=1 WHERE id=?",
                                    [(rid,) for rid in ids_to_skip],
                                )
                                conn.commit()
                                remaining_rows = max(0, remaining_rows - len(ids_to_skip))
                                stats["rows_remaining"] = remaining_rows
                                _log(
                                    "Marked "
                                    f"{len(ids_to_skip)} rows as gemini_skipped after repeated invalid JSON responses."
                                )
                        else:
                            success_streak = 0
                            err = result.get("error")
                            if err:
                                _log(f"Gemini worker error: {err}", account=account["name"])

                        if account["keys"]:
                            accounts.append(account)
                        else:
                            _log("Account exhausted and disabled.", account=account["name"])
                            if not accounts:
                                all_accounts_exhausted = True
            finally:
                reqs = token_totals.get("requests", 0)
                if reqs:
                    prompt_total = token_totals.get("input", 0)
                    output_total = token_totals.get("output", 0)
                    total_total = token_totals.get("total", 0)
                    _log(
                        f"Gemini token usage totals: requests={reqs}, input={prompt_total}, output={output_total}, total={total_total}",
                        style='green',
                    )
                    _log(
                        "Gemini token usage averages: "
                        f"input={prompt_total/reqs:.2f}, output={output_total/reqs:.2f}, total={total_total/reqs:.2f}",
                        style='green',
                    )
                else:
                    _log("Gemini token usage: no requests made.", style='green')

    _log(
        f"Gemini scoring finished; total={stats.get('rows_total',0)}, remaining={stats.get('rows_remaining',0)}, ok={stats.get('rows_ok',0)}, errors={stats.get('errors',0)}",
        style='green',
    )
    if all_accounts_exhausted and stats.get("rows_remaining", 0) > 0:
        raise typer.Exit(code=1)
    if report_path:
        ensure_parent(report_path)
        report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"Wrote Gemini report to {report_path}")


def export_parquet_cmd(
    db_path: Path,
    output: Path,
    table: str,
):
    """Export a table to a Parquet file (fallback to dedup if empty)."""
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
    """Stream table rows to Parquet in batches."""
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
