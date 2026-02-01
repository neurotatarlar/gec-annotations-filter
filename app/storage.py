"""SQLite helpers for pipeline tables and data movement."""

import sqlite3
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, List, Sequence, Set

import xxhash

from .dedup import normalize_for_hash
from .processing import ProcessedSegment


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


@contextmanager
def open_db(db_path: Path | str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits on exit."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _validate_table_name(name: str) -> str:
    """Ensure table names are safe for string interpolation."""
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Unsafe table name: {name}")
    return name


def ensure_clean_table(conn: sqlite3.Connection, table: str = "clean_segments") -> None:
    """Create the clean_segments table if missing."""
    table = _validate_table_name(table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            parent_digest TEXT,
            text TEXT NOT NULL,
            norm_hash TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_parent ON {table}(parent_digest)")


def ensure_dedup_table(conn: sqlite3.Connection, table: str = "dedup_segments") -> None:
    """Create the dedup_segments table if missing."""
    table = _validate_table_name(table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            parent_digest TEXT,
            text TEXT NOT NULL,
            norm_hash TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_parent ON {table}(parent_digest)")


def ensure_toxicity_table(conn: sqlite3.Connection, table: str = "toxicity_segments") -> None:
    """Create the toxicity_segments table if missing (adds gemini_skipped if needed)."""
    table = _validate_table_name(table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            parent_digest TEXT,
            text TEXT NOT NULL,
            norm_hash TEXT NOT NULL UNIQUE,
            toxicity_label INTEGER,
            toxicity_score REAL,
            gemini_skipped INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_parent ON {table}(parent_digest)")
    # Ensure new columns exist for older DBs
    existing_cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if "gemini_skipped" not in existing_cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN gemini_skipped INTEGER DEFAULT 0")


def ensure_gemini_table(conn: sqlite3.Connection, table: str = "gemini_segments") -> None:
    """Create the gemini/llm table if missing."""
    table = _validate_table_name(table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            parent_digest TEXT,
            text TEXT NOT NULL,
            norm_hash TEXT NOT NULL UNIQUE,
            toxicity_label INTEGER,
            toxicity_score REAL,
            gemini_json TEXT,
            gemini_error TEXT
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_parent ON {table}(parent_digest)")


def ensure_export_log_table(conn: sqlite3.Connection) -> None:
    """
    Track which ids have been exported for a given target to avoid duplicates on reruns.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS export_log (
            id TEXT NOT NULL,
            target TEXT NOT NULL,
            exported_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            PRIMARY KEY (id, target)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_export_log_target ON export_log(target)")


def insert_clean_segments(conn: sqlite3.Connection, segments: Sequence[ProcessedSegment], table: str = "clean_segments") -> int:
    """Insert cleaned segments into the destination table."""
    ensure_clean_table(conn, table)
    rows = [
        (
            seg.segment_id,
            seg.parent_digest,
            seg.text,
            text_hash(seg.text),
        )
        for seg in segments
    ]
    sql = f"INSERT OR IGNORE INTO {_validate_table_name(table)} (id, parent_digest, text, norm_hash) VALUES (?, ?, ?, ?)"
    cur = conn.executemany(sql, rows)
    return cur.rowcount if cur.rowcount is not None else 0


def copy_rows_to_dedup(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row], table: str = "dedup_segments") -> int:
    """Copy rows into the dedup table."""
    ensure_dedup_table(conn, table)
    payload = [(r["id"], r["parent_digest"], r["text"], r["norm_hash"]) for r in rows]
    sql = f"INSERT OR IGNORE INTO {_validate_table_name(table)} (id, parent_digest, text, norm_hash) VALUES (?, ?, ?, ?)"
    cur = conn.executemany(sql, payload)
    return cur.rowcount if cur.rowcount is not None else 0


def copy_rows_to_toxicity(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row], table: str = "toxicity_segments") -> int:
    """Copy rows into the toxicity table."""
    ensure_toxicity_table(conn, table)
    def _val(row, key):
        try:
            return row[key]
        except Exception:
            return row.get(key)
    payload = [
        (
            r["id"],
            r["parent_digest"],
            r["text"],
            r["norm_hash"],
            _val(r, "toxicity_label"),
            _val(r, "toxicity_score"),
        )
        for r in rows
    ]
    sql = f"INSERT OR REPLACE INTO {_validate_table_name(table)} (id, parent_digest, text, norm_hash, toxicity_label, toxicity_score) VALUES (?, ?, ?, ?, ?, ?)"
    cur = conn.executemany(sql, payload)
    return cur.rowcount if cur.rowcount is not None else 0


def copy_rows_to_gemini(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row], table: str) -> int:
    """Copy rows into the gemini/llm table."""
    ensure_gemini_table(conn, table)
    payload = [
        (
            r["id"],
            r["parent_digest"],
            r["text"],
            r["norm_hash"],
            r.get("toxicity_label"),
            r.get("toxicity_score"),
            r.get("gemini_json"),
        )
        for r in rows
    ]
    sql = f"INSERT OR REPLACE INTO {_validate_table_name(table)} (id, parent_digest, text, norm_hash, toxicity_label, toxicity_score, gemini_json) VALUES (?, ?, ?, ?, ?, ?, ?)"
    cur = conn.executemany(sql, payload)
    return cur.rowcount if cur.rowcount is not None else 0


def ensure_llm_table(conn: sqlite3.Connection, table: str = "llm_segments") -> None:
    """
    Mirror gemini table; used for offline LLM scoring to keep results separate.
    """
    ensure_gemini_table(conn, table)


def iterate_table(
    conn: sqlite3.Connection, table: str, batch_size: int = 1000
) -> Generator[List[sqlite3.Row], None, None]:
    """Stream rows from a table by rowid with forward-only pagination."""
    table = _validate_table_name(table)
    last_rowid = 0
    while True:
        rows = conn.execute(
            f"SELECT rowid as __rowid__, * FROM {table} WHERE rowid > ? and toxicity_label != 1 ORDER BY rowid LIMIT ?",
            (last_rowid, batch_size),
        ).fetchall()
        if not rows:
            break
        last_rowid = rows[-1]["__rowid__"]
        yield rows


def collect_ids(conn: sqlite3.Connection, table: str) -> Set[str]:
    """Collect ids from a table into a set (used for incremental stages)."""
    table = _validate_table_name(table)
    ids: Set[str] = set()
    for rows in iterate_table(conn, table, batch_size=1000):
        ids.update(r["id"] for r in rows)
    return ids


def text_hash(text: str) -> str:
    """Compute a normalized content hash for exact deduplication."""
    normalized = normalize_for_hash(text)
    return xxhash.xxh64(normalized).hexdigest()


def ensure_ingest_log_table(conn: sqlite3.Connection) -> None:
    """Create ingest_log for tracking processed dataset rows."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_log (
            row_hash TEXT PRIMARY KEY,
            dataset TEXT,
            split TEXT,
            revision TEXT,
            source_id TEXT,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingest_log_dataset_split ON ingest_log(dataset, split)"
    )


def try_log_row(
    conn: sqlite3.Connection, row_hash: str, dataset: str, split: str, revision: str | None, source_id: str
) -> bool:
    """
    Returns True if the row_hash was newly inserted (i.e., row not seen before).
    """
    ensure_ingest_log_table(conn)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO ingest_log (row_hash, dataset, split, revision, source_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (row_hash, dataset, split, revision, source_id),
    )
    return cur.rowcount is not None and cur.rowcount > 0
