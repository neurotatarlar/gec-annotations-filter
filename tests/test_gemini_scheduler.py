"""Integration-style tests for Gemini scheduler behavior using fakes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.commands import gemini_cmd
from app.storage import connect, ensure_toxicity_table


def _label_payload() -> dict:
    """Return a valid label payload accepted by the Gemini response schema."""
    return {
        "main_language": "tatar",
        "tatar_prob": 0.9,
        "russian_share": 0.0,
        "error_share": 0.2,
        "error_density": "medium",
        "main_error_type": "spelling",
        "non_fluent_prob": 0.1,
        "meaning_clarity": 0.8,
        "noise_score": 0.0,
        "overall_gec_usefulness": 0.7,
    }


class _FakeClock:
    """Deterministic clock for scheduler tests; sleep advances monotonic time."""

    def __init__(self) -> None:
        self._now = 0.0

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += max(0.0, float(seconds))


class _FakeClient:
    """Minimal stand-in for Gemini client that carries api_key for test logic."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


class GeminiSchedulerIntegrationTests(unittest.TestCase):
    """Exercise gemini_cmd scheduler retry and key-rotation behavior with fakes."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.db_path = self.root / "pipeline.sqlite"
        self.keys_path = self.root / "gemini_keys.yaml"
        self.report_path = self.root / "gemini_report.json"

    def _seed_source_rows(self, rows: list[tuple[str, str]]) -> None:
        conn = connect(self.db_path)
        try:
            ensure_toxicity_table(conn, "toxicity_segments")
            conn.executemany(
                """
                INSERT INTO toxicity_segments
                    (id, parent_digest, text, norm_hash, toxicity_label, toxicity_score, gemini_skipped)
                VALUES (?, ?, ?, ?, 0, 0.0, 0)
                """,
                [
                    (row_id, f"parent-{i}", text, f"nh-{i}")
                    for i, (row_id, text) in enumerate(rows, start=1)
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def _run_gemini_with_fakes(self, fake_call_gemini):
        clock = _FakeClock()

        def fake_build_batch_prompt(_base_prompt: str, rows: list[dict]) -> str:
            return json.dumps({"items": rows}, ensure_ascii=False)

        with (
            patch("app.commands.genai.Client", side_effect=lambda api_key: _FakeClient(api_key)),
            patch("app.commands.build_batch_prompt", side_effect=fake_build_batch_prompt),
            patch("app.commands.call_gemini", side_effect=fake_call_gemini),
            patch("app.commands.time.monotonic", side_effect=clock.monotonic),
            patch("app.commands.time.sleep", side_effect=clock.sleep),
        ):
            gemini_cmd(
                db_path=self.db_path,
                source_table="toxicity_segments",
                output_table="llm_segments",
                keys_path=self.keys_path,
                model="models/fake",
                prompt_path=None,
                batch_size=4,
                max_batch_size=4,
                workers=1,
                account_cooldown_seconds=1,
                max_rows=None,
                report_path=self.report_path,
            )

    def _load_counts(self) -> tuple[int, int]:
        conn = connect(self.db_path)
        try:
            llm_count = conn.execute("SELECT COUNT(*) FROM llm_segments").fetchone()[0]
            skipped_count = conn.execute(
                "SELECT COUNT(*) FROM toxicity_segments WHERE COALESCE(gemini_skipped,0)=1"
            ).fetchone()[0]
            return int(llm_count), int(skipped_count)
        finally:
            conn.close()

    def test_partial_response_retries_missing_rows_and_completes(self) -> None:
        self.keys_path.write_text("acc1: key1\n", encoding="utf-8")
        self._seed_source_rows(
            [
                ("r1", "text 1"),
                ("r2", "text 2"),
                ("r3", "text 3"),
                ("r4", "text 4"),
            ]
        )

        seen_batches: list[list[str]] = []

        def fake_call_gemini(_client, _model, prompt: str, schema):
            del schema
            items = json.loads(prompt)["items"]
            ids = [str(item["id"]) for item in items]
            seen_batches.append(ids)
            if len(seen_batches) == 1:
                returned_ids = ids[:2]
            else:
                returned_ids = ids
            payload = [{"id": rid, "labels": _label_payload()} for rid in returned_ids]
            return json.dumps(payload), {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            }

        self._run_gemini_with_fakes(fake_call_gemini)

        llm_count, skipped_count = self._load_counts()
        self.assertEqual(llm_count, 4)
        self.assertEqual(skipped_count, 0)
        self.assertGreaterEqual(len(seen_batches), 2)
        self.assertEqual(seen_batches[0], ["r1", "r2", "r3", "r4"])
        self.assertIn(["r3", "r4"], seen_batches[1:])

        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["rows_ok"], 4)
        self.assertEqual(report["rows_remaining"], 0)
        self.assertGreaterEqual(report.get("db_commits", 0), 1)

    def test_quota_exhausts_key_then_cools_down_and_uses_next_key(self) -> None:
        self.keys_path.write_text("acc1:\n  - key1\n  - key2\n", encoding="utf-8")
        self._seed_source_rows([("r1", "text 1")])

        used_keys: list[str] = []
        quota_raised = False

        def fake_call_gemini(client, _model, prompt: str, schema):
            nonlocal quota_raised
            del schema
            ids = [str(item["id"]) for item in json.loads(prompt)["items"]]
            used_keys.append(client.api_key)
            if client.api_key == "key1" and not quota_raised:
                quota_raised = True
                raise RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")
            payload = [{"id": ids[0], "labels": _label_payload()}]
            return json.dumps(payload), None

        self._run_gemini_with_fakes(fake_call_gemini)

        llm_count, skipped_count = self._load_counts()
        self.assertEqual(llm_count, 1)
        self.assertEqual(skipped_count, 0)
        self.assertEqual(used_keys[:2], ["key1", "key2"])

        expired_keys_path = self.keys_path.with_suffix(self.keys_path.suffix + ".expired")
        expired = json.loads(expired_keys_path.read_text(encoding="utf-8"))
        self.assertIn("key1", expired)

        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["rows_ok"], 1)
        self.assertGreaterEqual(report.get("cooldowns", 0), 1)


if __name__ == "__main__":
    unittest.main()
