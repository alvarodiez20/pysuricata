"""A benchmark must refuse to measure on a busy machine (#212).

The measurement rule this project runs on is: *a ratio is quotable only when
both sides were measured in the same round-robin, on the same machine, within
the same run.* Interleaving cancels drift between the versions being compared.

**It does not cancel a neighbour**, because the neighbour is not in the
round-robin — and that nearly published a claim. A round-robin put 0.0.61 at
1,599 ms against 0.0.42's 1,448: a 10.5% regression on a harness that
reproduces to ±1%, with a ready-made culprit in the abstraction boundary #108
had just added to the accumulator hot path. Bisecting seven commits refused it
— 1,203 to 1,271 ms, no trend, HEAD at 1.008× — and the real cause was the
coverage suite running in parallel, competing for two cores with the benchmark
measuring against it.

That was the fourth measurement artefact in one audit series to nearly become a
published claim, and the first caught before it was written down. A clause that
lives only in a document gets forgotten, so it lives in the harness, and this
file is what keeps it there.
"""

from __future__ import annotations

import os

import pytest

from benchmarks.end_to_end import MAX_LOAD_PER_CORE, load_average, load_guard


@pytest.fixture
def busy(monkeypatch):
    """A machine with far more runnable work than cores."""
    monkeypatch.setattr(
        "benchmarks.end_to_end.load_average",
        lambda: MAX_LOAD_PER_CORE * (os.cpu_count() or 1) * 25,
    )


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr("benchmarks.end_to_end.load_average", lambda: 0.0)


class TestItRefusesWhenTheMachineIsBusy:
    def test_a_busy_machine_is_refused(self, busy):
        load, refusal = load_guard()

        assert refusal is not None, (
            "the harness agreed to measure under contention, which is how a "
            "10.5% regression that did not exist nearly got published"
        )
        assert load is not None

    def test_the_refusal_says_what_to_do(self, busy):
        _, refusal = load_guard()

        # A refusal a reader cannot act on gets worked around with --force
        # reflexively, which is the same as not having it.
        assert "--force" in refusal
        assert "load average" in refusal

    def test_force_overrides_it(self, busy):
        load, refusal = load_guard(force=True)

        assert refusal is None
        assert load is not None, "forcing must not also discard the reading"

    def test_a_quiet_machine_is_allowed(self, quiet):
        assert load_guard()[1] is None


class TestTheReadingTravelsWithTheResult:
    """A forced or contended run has to carry its own caveat, or the number
    gets quoted clean by whoever finds the file later."""

    def test_the_load_is_returned_even_when_it_passes(self, quiet):
        load, refusal = load_guard()

        assert refusal is None
        assert load == 0.0

    @pytest.mark.parametrize("harness", ["end_to_end", "versions"])
    def test_both_harnesses_record_both_ends(self, harness):
        """#212 asks for the guard in both. `versions.py` imports it rather
        than reimplementing it, so there is one threshold, not two that drift."""
        import importlib

        module = importlib.import_module(f"benchmarks.{harness}")
        source = module.__file__
        assert source is not None
        text = open(source, encoding="utf-8").read()

        assert "load_guard" in text, f"{harness} does not check the load"
        assert '"load_start"' in text, f"{harness} does not record the start load"
        assert '"load_end"' in text, f"{harness} does not record the end load"
        assert "--force" in text, f"{harness} has no escape hatch"


class TestItDegradesRatherThanGuesses:
    def test_no_load_average_means_no_opinion(self, monkeypatch):
        """Windows has no `getloadavg`. Skipping the check and saying so is
        right; inventing a number for it is not."""
        monkeypatch.setattr("benchmarks.end_to_end.load_average", lambda: None)

        assert load_guard() == (None, None)

    def test_the_real_reading_is_a_number_or_none(self):
        load = load_average()

        assert load is None or (isinstance(load, float) and load >= 0.0)


def test_the_rule_is_written_down_where_it_is_enforced():
    """#212's fourth box. The clause was in a document and the harness ignored
    it; now the harness enforces it and the document says so."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notes = open(os.path.join(repo, "CLAUDE.md"), encoding="utf-8").read()

    # Collapsed first: the clause is a sentence in wrapped markdown, so it is
    # split across a line break and is not a contiguous substring of the file.
    flowed = " ".join(notes.split()).lower()

    assert "nothing else was running" in flowed, (
        "the measurement-discipline notes do not carry the clause the harness "
        "now enforces"
    )
    assert "--force" in flowed, (
        "the notes describe the rule without the escape hatch, so a reader who "
        "hits the refusal has to go read the source to get past it"
    )
