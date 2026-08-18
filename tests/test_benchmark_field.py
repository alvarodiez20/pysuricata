"""`benchmarks/field.py` (#2) and the `ydata-profiling` rename it exists partly
to get right.

`field.py` is deliberately not a general-purpose harness: it pins the exact
scenario a published comparison table is measured on, so "re-run it yourself"
is one command instead of a guess at which flags produced a headline ratio.
The one thing worth asserting about a *pinned* scenario is that it stays
pinned -- a flag or a default drifting out from under a published number would
be silent otherwise.

The other half is `end_to_end.TOOLS["ydata"]`: `ydata-profiling` renamed
itself to `fg-data-profiling` (import `data_profiling`) in its 4.18.4 release
and receives no further updates under the old name. A harness that still
imported the old name only would measure an abandoned package and call it
current -- exactly the kind of thirty-second-find that costs a benchmark post
its credibility.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks import end_to_end, field

_FAKE_OK_STDOUT = (
    '__RESULT__{"seconds": 1.0, "peak_rss_mb": 1.0, "output_bytes": 1, "error": null}'
)


class TestTheScenarioIsPinned:
    def test_the_tool_set_is_a_subset_of_end_to_ends(self):
        assert set(field.TOOLS) <= set(end_to_end.TOOLS)

    def test_pysuricata_is_measured(self):
        """The one non-negotiable entry -- a comparison table without it is
        not a comparison."""
        assert "pysuricata" in field.TOOLS

    def test_the_round_count_is_the_quotable_minimum(self):
        assert field.ROUNDS == end_to_end.MIN_QUOTABLE_ROUNDS

    def test_the_scenario_is_the_realistic_shaped_suite(self):
        """`mixed` is the suite built to read as a real analytics table's
        column mix -- not one of the isolation shapes `hotspots.py` and
        `kernels.py` use to pin down a single kernel, which would flatter or
        punish one tool by the shape of the data alone."""
        assert field.SUITE == "mixed"


class TestRunUsesThePinnedScenario:
    def test_round_robin_is_called_with_the_module_constants(self, monkeypatch):
        calls = []

        def fake_round_robin(tools, suite, scale, repo, timeout, rounds):
            calls.append((tools, suite, scale, timeout, rounds))
            return {}

        monkeypatch.setattr(field, "round_robin", fake_round_robin)
        monkeypatch.setattr(field, "load_guard", lambda force: (None, None))
        monkeypatch.setattr(field, "load_average", lambda: None)
        monkeypatch.setattr(field, "environment", lambda: {})

        field.run(rounds=3, force=True)

        assert len(calls) == 1
        tools, suite, scale, timeout, rounds = calls[0]
        assert tools == field.TOOLS
        assert suite == field.SUITE
        assert scale == field.SCALE
        assert timeout == field.TIMEOUT
        assert rounds == 3

    def test_a_refused_load_guard_stops_the_run_before_measuring(self, monkeypatch):
        monkeypatch.setattr(field, "load_guard", lambda force: (None, "machine busy"))

        def fail_if_called(*a, **k):
            raise AssertionError("round_robin should not run when the guard refuses")

        monkeypatch.setattr(field, "round_robin", fail_if_called)

        with pytest.raises(SystemExit, match="machine busy"):
            field.run()


class TestCLIWarnsBelowTheQuotableMinimum:
    def test_a_low_round_count_prints_a_warning(self, monkeypatch, capsys):
        monkeypatch.setattr(
            field, "run", lambda rounds, force: {"suites": {}, "environment": {}}
        )

        field.main(["--rounds", "1"])

        assert "below the" in capsys.readouterr().out

    def test_the_quotable_minimum_prints_no_warning(self, monkeypatch, capsys):
        monkeypatch.setattr(
            field, "run", lambda rounds, force: {"suites": {}, "environment": {}}
        )

        field.main(["--rounds", str(end_to_end.MIN_QUOTABLE_ROUNDS)])

        assert "below the" not in capsys.readouterr().out

    def test_zero_rounds_is_rejected(self):
        with pytest.raises(SystemExit):
            field.main(["--rounds", "0"])


class TestTheYdataRenameIsHandled:
    """`end_to_end.TOOLS["ydata"]` -- shared by `end_to_end.py` and
    `field.py`, so fixing it once fixes both."""

    def test_the_new_name_is_tried_before_the_old_one(self):
        candidates = end_to_end.TOOLS["ydata"]["import_candidates"]
        assert candidates == ["data_profiling", "ydata_profiling"]
        assert candidates == end_to_end.TOOLS["ydata-minimal"]["import_candidates"]

    def test_neither_installed_is_reported_as_skipped_not_crashed(self, monkeypatch):
        monkeypatch.setattr(end_to_end, "have", lambda module: False)

        result = end_to_end.run_one("ydata", "mixed", 0.01, ".", 30)

        assert result["status"] == "skipped"
        assert "data_profiling" in result["reason"]
        assert "ydata_profiling" in result["reason"]

    def test_the_new_name_carries_no_note(self, monkeypatch):
        monkeypatch.setattr(
            end_to_end, "have", lambda module: module == "data_profiling"
        )
        monkeypatch.setattr(
            end_to_end.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(
                stdout=_FAKE_OK_STDOUT, stderr="", returncode=0
            ),
        )

        result = end_to_end.run_one("ydata", "mixed", 0.01, ".", 30)

        assert "note" not in result

    def test_falling_back_to_the_old_name_attaches_a_note(self, monkeypatch):
        """Using the abandoned package must not look identical to using the
        maintained one -- the whole point of the fallback existing is that a
        reader can tell which one actually ran."""
        monkeypatch.setattr(
            end_to_end, "have", lambda module: module == "ydata_profiling"
        )
        monkeypatch.setattr(
            end_to_end.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(
                stdout=_FAKE_OK_STDOUT, stderr="", returncode=0
            ),
        )

        result = end_to_end.run_one("ydata", "mixed", 0.01, ".", 30)

        assert "note" in result
        assert "ydata_profiling" in result["note"]
        assert "fg-data-profiling" in result["note"]


class TestTheEnvironmentLineNamesWhicheverIsInstalled:
    def test_neither_installed(self):
        assert "not installed" in end_to_end._profiling_line({})

    def test_the_current_name(self):
        line = end_to_end._profiling_line({"data_profiling": "4.19.1"})
        assert "fg-data-profiling 4.19.1" in line
        assert "renamed" not in line

    def test_the_abandoned_name_is_flagged_not_presented_as_current(self):
        line = end_to_end._profiling_line({"ydata_profiling": "4.18.4"})
        assert "ydata-profiling 4.18.4" in line
        assert "renamed" in line
        assert "no longer updated" in line

    def test_the_current_name_wins_when_both_are_present(self):
        """A compatibility shim in the new package can make the old import
        resolve too (`fg-data-profiling` ships one) -- the line must not flag
        an environment that actually has the maintained package installed."""
        line = end_to_end._profiling_line(
            {"data_profiling": "4.19.1", "ydata_profiling": "4.19.1"}
        )
        assert "fg-data-profiling 4.19.1" in line
        assert "renamed" not in line
