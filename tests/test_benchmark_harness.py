"""The round-robin schedule, and the guard that stops a ratio being quoted early.

Two published claims in this project came from pairing measurements taken at
different times: "0.0.21 is 1.24x faster than 0.0.16" (really 0.88x — a
regression) and a 3.56x headline that is really 2.48x. Both had the same cause,
so the schedule that prevents it is worth pinning.
"""

from __future__ import annotations

import pytest

from benchmarks import end_to_end, versions


class TestRoundRobinSchedule:
    def test_every_tool_is_measured_in_every_round(self, monkeypatch):
        """Interleaved, not one tool run to completion and then the next."""
        order: list[str] = []

        def fake_run(tool, suite, scale, repo, timeout):
            order.append(tool)
            return {"status": "ok", "seconds": 1.0, "peak_rss_mb": 1, "output_bytes": 1}

        monkeypatch.setattr(end_to_end, "run_one", fake_run)
        monkeypatch.setitem(end_to_end.TOOLS, "a", {})
        monkeypatch.setitem(end_to_end.TOOLS, "b", {})
        end_to_end.round_robin(["a", "b"], "mixed", 1.0, ".", 10, rounds=3)

        assert order == ["a", "b", "a", "b", "a", "b"]

    def test_the_best_round_is_reported(self, monkeypatch):
        timings = iter([3.0, 1.0, 2.0])

        def fake_run(tool, suite, scale, repo, timeout):
            return {
                "status": "ok",
                "seconds": next(timings),
                "peak_rss_mb": 1,
                "output_bytes": 1,
            }

        monkeypatch.setattr(end_to_end, "run_one", fake_run)
        monkeypatch.setitem(end_to_end.TOOLS, "a", {})
        best = end_to_end.round_robin(["a"], "mixed", 1.0, ".", 10, rounds=3)

        assert best["a"]["seconds"] == 1.0

    def test_every_round_is_retained_so_the_spread_is_visible(self, monkeypatch):
        timings = iter([3.0, 1.0, 2.0])

        def fake_run(tool, suite, scale, repo, timeout):
            return {
                "status": "ok",
                "seconds": next(timings),
                "peak_rss_mb": 1,
                "output_bytes": 1,
            }

        monkeypatch.setattr(end_to_end, "run_one", fake_run)
        monkeypatch.setitem(end_to_end.TOOLS, "a", {})
        best = end_to_end.round_robin(["a"], "mixed", 1.0, ".", 10, rounds=3)

        assert sorted(best["a"]["all_seconds"]) == [1.0, 2.0, 3.0]
        assert best["a"]["rounds"] == 3
        assert best["a"]["spread_pct"] == pytest.approx(200.0)

    def test_a_failing_tool_is_recorded_not_dropped(self, monkeypatch):
        def fake_run(tool, suite, scale, repo, timeout):
            return {"status": "timeout", "seconds": 10}

        monkeypatch.setattr(end_to_end, "run_one", fake_run)
        monkeypatch.setitem(end_to_end.TOOLS, "a", {})
        best = end_to_end.round_robin(["a"], "mixed", 1.0, ".", 10, rounds=2)

        assert best["a"]["status"] == "timeout"

    def test_a_tool_that_recovers_after_failing_is_reported_ok(self, monkeypatch):
        results = iter(
            [
                {"status": "crashed"},
                {
                    "status": "ok",
                    "seconds": 2.0,
                    "peak_rss_mb": 1,
                    "output_bytes": 1,
                },
            ]
        )

        def fake_run(tool, suite, scale, repo, timeout):
            return next(results)

        monkeypatch.setattr(end_to_end, "run_one", fake_run)
        monkeypatch.setitem(end_to_end.TOOLS, "a", {})
        best = end_to_end.round_robin(["a"], "mixed", 1.0, ".", 10, rounds=2)

        assert best["a"]["status"] == "ok"
        assert best["a"]["seconds"] == 2.0


class TestQuotabilityGuard:
    def _payload(self, rounds: int) -> dict:
        return {
            "environment": {
                "system": "x",
                "machine": "y",
                "cpu_count": 1,
                "python": "3",
                "pandas": "2",
                "numpy": "2",
                "pysuricata": "0",
                "pysuricata_core": None,
                "ydata_profiling": None,
            },
            "rounds": rounds,
            "quotable": rounds >= end_to_end.MIN_QUOTABLE_ROUNDS,
            "suites": {
                "mixed": {
                    "_shape": {"rows": 10, "cols": 2, "bytes": 100},
                    "pysuricata": {
                        "status": "ok",
                        "seconds": 1.0,
                        "peak_rss_mb": 1,
                        "output_bytes": 1,
                    },
                }
            },
        }

    def test_a_single_round_is_marked_unquotable(self):
        assert "Not quotable" in end_to_end.to_markdown(self._payload(1))

    def test_two_rounds_are_still_unquotable(self):
        assert "Not quotable" in end_to_end.to_markdown(self._payload(2))

    def test_enough_rounds_carry_no_warning(self):
        assert "Not quotable" not in end_to_end.to_markdown(self._payload(5))

    def test_the_round_count_is_always_stated(self):
        assert "5 interleaved round(s)" in end_to_end.to_markdown(self._payload(5))


class TestVersionCurve:
    def _payload(self, rounds: int = 5) -> dict:
        return {
            "environment": {
                "system": "x",
                "machine": "y",
                "cpu_count": 1,
                "python": "3",
                "pandas": "2",
                "numpy": "2",
            },
            "suite": "mixed",
            "scale": 0.2,
            "rounds": rounds,
            "quotable": rounds >= end_to_end.MIN_QUOTABLE_ROUNDS,
            "baseline": "0.0.16",
            "results": {
                "0.0.16": {
                    "status": "ok",
                    "seconds": 4.0,
                    "all_seconds": [4.0, 4.4],
                },
                "0.0.21": {"status": "ok", "seconds": 4.5, "all_seconds": [4.5, 4.6]},
                "working tree": {
                    "status": "ok",
                    "seconds": 2.0,
                    "all_seconds": [2.0, 2.1],
                },
            },
        }

    def test_a_regression_is_reported_as_below_one(self):
        """0.0.21 was published as 1.24x faster; it was slower."""
        table = versions.to_markdown(self._payload())
        assert "| 0.0.21 | 4,500.0 | 0.89x" in table

    def test_an_improvement_is_reported_against_the_baseline(self):
        assert "| working tree | 2,000.0 | 2.00x" in versions.to_markdown(
            self._payload()
        )

    def test_the_table_says_ratios_are_only_valid_within_it(self):
        assert "only comparable within this table" in versions.to_markdown(
            self._payload()
        )

    def test_too_few_rounds_is_marked(self):
        assert "Not quotable" in versions.to_markdown(self._payload(rounds=1))

    def test_a_version_that_failed_shows_its_status(self):
        payload = self._payload()
        payload["results"]["0.0.21"] = {"status": "crashed"}
        assert "| 0.0.21 | — | — | crashed |" in versions.to_markdown(payload)
