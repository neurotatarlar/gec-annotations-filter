"""Unit tests for command-layer scheduler helper functions."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import typer
from pydantic import TypeAdapter

from app.gemini import LabelPayload
from app.commands import (
    _flatten_hf_export_row,
    _gemini_jittered_seconds,
    _gemini_prefetch_limits,
    _should_flush_gemini_writes,
    export_hf_parquet_cmd,
)


class _StubRng:
    """Deterministic RNG stub returning a fixed uniform value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def uniform(self, _a: float, _b: float) -> float:
        return self.value


class GeminiSchedulerHelperTests(unittest.TestCase):
    """Verify pure scheduler helpers used by Gemini parallel execution."""

    def test_jittered_seconds_handles_zero_base(self) -> None:
        self.assertEqual(_gemini_jittered_seconds(0, 0.2, rng=_StubRng(0.1)), 0.0)

    def test_jittered_seconds_applies_positive_and_negative_jitter(self) -> None:
        self.assertAlmostEqual(_gemini_jittered_seconds(60, 0.1, rng=_StubRng(0.1)), 66.0)
        self.assertAlmostEqual(_gemini_jittered_seconds(60, 0.1, rng=_StubRng(-0.1)), 54.0)

    def test_prefetch_limits_scale_with_workers_and_batch(self) -> None:
        watermark, fetch_limit = _gemini_prefetch_limits(3, 64, queue_factor=3, fetch_factor=2)
        self.assertEqual(watermark, 576)
        self.assertEqual(fetch_limit, 384)

    def test_flush_decision_respects_threshold_interval_and_force(self) -> None:
        self.assertFalse(_should_flush_gemini_writes(0, 100, 10.0, 10.1, 2.0))
        self.assertTrue(_should_flush_gemini_writes(100, 100, 10.0, 10.1, 2.0))
        self.assertTrue(_should_flush_gemini_writes(10, 100, 10.0, 12.1, 2.0))
        self.assertTrue(_should_flush_gemini_writes(1, 100, 10.0, 10.1, 2.0, force=True))
        self.assertTrue(_should_flush_gemini_writes(1, 100, 10.0, 10.1, 0.0))


class HfParquetExportTests(unittest.TestCase):
    """Verify HF Parquet export flattening and row filtering."""

    def _labels(self, **overrides) -> str:
        payload = {
            "main_language": "tatar",
            "tatar_prob": 0.95,
            "russian_share": 0.05,
            "error_share": 0.2,
            "error_density": "medium",
            "main_error_type": "spelling",
            "non_fluent_prob": 0.25,
            "meaning_clarity": 0.8,
            "noise_score": 0.1,
            "overall_gec_usefulness": 0.85,
        }
        payload.update(overrides)
        return json.dumps(payload)

    def _create_db(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """
                CREATE TABLE llm_segments (
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
            conn.execute(
                """
                CREATE TABLE toxicity_segments (
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
            llm_rows = [
                ("ok1", "p", "беренче текст", "h1", 0, 0.01, self._labels(), None),
                ("ok2", "p", "икенче текст", "h2", 0, 0.02, self._labels(error_density="high"), None),
                ("toxic", "p", "toxic text", "h3", 1, 0.99, self._labels(), None),
                ("skipped", "p", "skipped text", "h4", 0, 0.03, self._labels(), None),
                ("invalid", "p", "invalid json", "h5", 0, 0.04, "{bad", None),
                ("missing", "p", "missing labels", "h6", 0, 0.05, None, None),
            ]
            tox_rows = [
                ("ok1", "p", "беренче текст", "t1", 0, 0.01, 0),
                ("ok2", "p", "икенче текст", "t2", 0, 0.02, 0),
                ("toxic", "p", "toxic text", "t3", 1, 0.99, 0),
                ("skipped", "p", "skipped text", "t4", 0, 0.03, 1),
                ("invalid", "p", "invalid json", "t5", 0, 0.04, 0),
                ("missing", "p", "missing labels", "t6", 0, 0.05, 0),
            ]
            conn.executemany(
                """
                INSERT INTO llm_segments
                (id, parent_digest, text, norm_hash, toxicity_label, toxicity_score, gemini_json, gemini_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                llm_rows,
            )
            conn.executemany(
                """
                INSERT INTO toxicity_segments
                (id, parent_digest, text, norm_hash, toxicity_label, toxicity_score, gemini_skipped)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                tox_rows,
            )
            conn.commit()
        finally:
            conn.close()

    def test_flatten_hf_export_row(self) -> None:
        row = {
            "id": "x1",
            "text": "текст",
            "toxicity_label": 0,
            "toxicity_score": 0.12,
            "gemini_json": self._labels(error_density="low"),
        }
        flat = _flatten_hf_export_row(row, TypeAdapter(LabelPayload))
        self.assertIsNotNone(flat)
        self.assertEqual(flat["id"], "x1")
        self.assertEqual(flat["error_density"], "low")
        self.assertAlmostEqual(flat["overall_gec_usefulness"], 0.85)

    def test_flatten_hf_export_row_accepts_wrapped_labels(self) -> None:
        row = {
            "id": "x1",
            "text": "текст",
            "toxicity_label": 0,
            "toxicity_score": 0.12,
            "gemini_json": json.dumps({"labels": json.loads(self._labels(error_density="high"))}),
        }
        flat = _flatten_hf_export_row(row, TypeAdapter(LabelPayload))
        self.assertIsNotNone(flat)
        self.assertEqual(flat["error_density"], "high")

    def test_export_hf_parquet_filters_rows_and_writes_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "test.sqlite"
            out_dir = root / "hf"
            self._create_db(db_path)

            export_hf_parquet_cmd(
                db_path=db_path,
                output_dir=out_dir,
                source_table="llm_segments",
                toxicity_table="toxicity_segments",
                max_file_size_mb=0.000001,
                batch_size=1,
                limit=None,
                overwrite=False,
            )

            parts = sorted(out_dir.glob("*.parquet"))
            self.assertEqual(len(parts), 2)
            tables = [pq.read_table(part) for part in parts]
            ids = []
            for table in tables:
                ids.extend(table.column("id").to_pylist())
                self.assertIn("overall_gec_usefulness", table.column_names)
                self.assertIn("toxicity_score", table.column_names)
            self.assertEqual(sorted(ids), ["ok1", "ok2"])

            report = json.loads((out_dir / "export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rows_written"], 2)
            self.assertEqual(report["rows_skipped_invalid_gemini_json"], 1)
            self.assertEqual(report["file_count"], 2)

    def test_export_hf_parquet_refuses_existing_parts_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "test.sqlite"
            out_dir = root / "hf"
            out_dir.mkdir()
            (out_dir / "train-00000.parquet").write_bytes(b"existing")
            self._create_db(db_path)

            with self.assertRaises(typer.Exit):
                export_hf_parquet_cmd(
                    db_path=db_path,
                    output_dir=out_dir,
                    source_table="llm_segments",
                    toxicity_table="toxicity_segments",
                    max_file_size_mb=512,
                    batch_size=10,
                    limit=None,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
