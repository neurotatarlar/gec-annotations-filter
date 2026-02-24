"""Unit tests for command-layer scheduler helper functions."""

import unittest

from app.commands import (
    _gemini_jittered_seconds,
    _gemini_prefetch_limits,
    _should_flush_gemini_writes,
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


if __name__ == "__main__":
    unittest.main()
