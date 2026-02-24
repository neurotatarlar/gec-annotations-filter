"""Unit tests for Gemini helper utilities."""

import json
import tempfile
import unittest
from pathlib import Path

from app.gemini import load_account_keys, load_keys, parse_json_response


class GeminiKeyLoadingTests(unittest.TestCase):
    """Verify Gemini key YAML parsing supports account grouping and legacy lists."""

    def _tmp_yaml(self, text: str) -> Path:
        fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        fh.write(text)
        fh.flush()
        fh.close()
        self.addCleanup(lambda: Path(fh.name).unlink(missing_ok=True))
        return Path(fh.name)

    def test_load_account_keys_from_mapping(self) -> None:
        path = self._tmp_yaml(
            """
            acc_a:
              - key1
              - key2
            acc_b: key3
            acc_c:
              - " "
              - key4
            """
        )
        self.assertEqual(
            load_account_keys(path),
            {
                "acc_a": ["key1", "key2"],
                "acc_b": ["key3"],
                "acc_c": ["key4"],
            },
        )

    def test_load_account_keys_from_legacy_list(self) -> None:
        path = self._tmp_yaml(
            """
            - key1
            - key2
            - "  "
            """
        )
        self.assertEqual(
            load_account_keys(path),
            {
                "account_1": ["key1"],
                "account_2": ["key2"],
            },
        )

    def test_load_keys_flattens_account_groups(self) -> None:
        path = self._tmp_yaml(
            """
            a:
              - key1
            b:
              - key2
              - key3
            """
        )
        self.assertEqual(load_keys(path), ["key1", "key2", "key3"])


class GeminiResponseParsingTests(unittest.TestCase):
    """Verify JSON parsing/validation tolerates fenced output and rejects bad payloads."""

    def test_parse_json_response_from_fenced_json(self) -> None:
        payload = [
            {
                "id": "x1",
                "labels": {
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
                },
            }
        ]
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        parsed = parse_json_response(fenced)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "x1")
        self.assertEqual(parsed[0]["labels"]["main_language"], "tatar")

    def test_parse_json_response_rejects_invalid_enum(self) -> None:
        payload = [
            {
                "id": "x1",
                "labels": {
                    "main_language": "bad",
                    "tatar_prob": 0.9,
                    "russian_share": 0.0,
                    "error_share": 0.2,
                    "error_density": "medium",
                    "main_error_type": "spelling",
                    "non_fluent_prob": 0.1,
                    "meaning_clarity": 0.8,
                    "noise_score": 0.0,
                    "overall_gec_usefulness": 0.7,
                },
            }
        ]
        with self.assertRaises(ValueError):
            parse_json_response(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
