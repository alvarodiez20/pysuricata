"""Quantiles are estimates, and the card now says so.

`Median` rendered as `0.003684` on a 60,000-row column whose reservoir held
20,000 values, in the same typography as `Min` and `Max` -- which are exact.
The true median of that column is `-0.00252`. The report printed four
significant figures, the wrong sign, and no indication that the number was
drawn from a third of the data.

The estimate is **stable** -- `profile()` defaults to seed 0, so it does not
move between runs -- and stability is not accuracy. Change the seed and the
median moves to `0.01293`, because the value is a property of the sample drawn
rather than of the column.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.sampling import quantiles_are_sampled


def _stats(html: str) -> dict[str, str]:
    html = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)
    return dict(
        re.findall(r'vstat__cap">([^<]*)</div><div class="vstat__val">([^<]*)<', html)
    )


@pytest.fixture(scope="module")
def sampled() -> dict[str, str]:
    """60,000 rows against a 20,000-value reservoir."""
    rng = np.random.default_rng(0)
    return _stats(profile(pd.DataFrame({"a": rng.normal(0, 1, 60_000)}), seed=0).html)


@pytest.fixture(scope="module")
def whole() -> dict[str, str]:
    """Short enough that the reservoir holds the column."""
    rng = np.random.default_rng(0)
    return _stats(profile(pd.DataFrame({"a": rng.normal(0, 1, 500)}), seed=0).html)


class TestTheSampledOnesAreMarked:
    @pytest.mark.parametrize("label", ["Q1 (P25)", "Median", "Q3 (P75)"])
    def test_a_long_column_marks_its_quantiles(self, sampled, label):
        assert f"{label} (≈)" in sampled
        assert label not in sampled, "the unmarked label must not also appear"

    def test_iqr_and_mad_inherit_it(self):
        """Same reservoir, same status."""
        rng = np.random.default_rng(0)
        html = profile(pd.DataFrame({"a": rng.normal(0, 1, 60_000)}), seed=0).html
        assert "IQR (≈)" in html
        assert "MAD (≈)" in html


class TestTheExactOnesAreNot:
    """#118 made the extremes come from every value rather than the reservoir,
    precisely so they would stop being sampled. Marking them now would throw
    that away."""

    @pytest.mark.parametrize("label", ["Min", "Max", "Mean"])
    def test_they_carry_no_marker(self, sampled, label):
        assert label in sampled
        assert f"{label} (≈)" not in sampled


class TestAShortColumnClaimsNothing:
    @pytest.mark.parametrize("label", ["Q1 (P25)", "Median", "Q3 (P75)"])
    def test_no_marker_when_the_reservoir_held_everything(self, whole, label):
        assert label in whole
        assert f"{label} (≈)" not in whole

    def test_the_predicate_agrees(self):
        class S:
            count = 500
            sample_vals = list(range(500))

        assert not quantiles_are_sampled(S())
        S.count = 60_000
        assert quantiles_are_sampled(S())


class TestThePredicateIsHonestWhenItCannotTell:
    """False is the honest answer with no sample: those quantiles did not come
    from a reservoir, and a warning on them would point at the wrong thing."""

    @pytest.mark.parametrize("sample", [None, [], ()])
    def test_absent_sample_is_not_reported_as_approximate(self, sample):
        class S:
            count = 10_000

        S.sample_vals = sample
        assert not quantiles_are_sampled(S())

    def test_a_missing_count_does_not_raise(self):
        assert not quantiles_are_sampled(object())


class TestTheEstimateIsWorthMarking:
    """Why the marker is warranted, stated as the thing that is actually true.

    The first version of this file argued from run-to-run variance and was
    wrong: `profile()` defaults to seed 0, so an unseeded run is bit-identical
    to a seeded one and nothing moves between runs. That claim came from
    driving `NumericAccumulator` directly with no seed -- a configuration the
    public API never uses. Same error as measuring a kernel through a call site
    nothing calls.

    Determinism is not accuracy. The estimate is stable and still an estimate,
    and these are the two facts that show it.
    """

    def test_the_seed_changes_the_answer(self):
        """Reproducible per seed, different across seeds -- so the value is a
        property of the sample drawn, not of the column alone."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 60_000)})
        assert (
            _stats(profile(frame, seed=0).html)["Median (≈)"]
            != (_stats(profile(frame, seed=7).html)["Median (≈)"])
        )

    def test_the_printed_precision_exceeds_the_accuracy(self):
        """Four significant figures on a value that is right to about two.

        The normal case is the sharp one: the true median sits near zero, the
        sampling error is larger than it, and the report prints a positive
        median for a column whose median is negative. Every digit shown invites
        a conclusion the estimate cannot support.
        """
        values = np.random.default_rng(0).normal(0, 1, 60_000)
        reported = float(
            _stats(profile(pd.DataFrame({"a": values}), seed=0).html)["Median (≈)"]
        )
        true = float(np.median(values))

        assert reported != pytest.approx(true, abs=1e-6)
        # Not a small slip in the last digit: the error is a whole order of
        # magnitude above the precision printed.
        assert abs(reported - true) > 1e-3

    @pytest.mark.parametrize(
        ("name", "sample"),
        [
            ("gamma", np.random.default_rng(1).gamma(2, 20, 60_000)),
            ("lognormal", np.random.default_rng(2).lognormal(0, 1.5, 60_000)),
        ],
    )
    def test_the_estimate_is_close_but_not_exact(self, name, sample):
        """On distributions away from zero the error is fractions of a percent —
        good, and still not the four figures printed."""
        reported = float(
            _stats(profile(pd.DataFrame({"a": sample}), seed=0).html)["Median (≈)"]
        )
        true = float(np.median(sample))
        assert reported == pytest.approx(true, rel=0.05)
        assert reported != pytest.approx(true, rel=1e-9)

    def test_a_seeded_run_is_reproducible(self):
        """The property the whole suite relies on, asserted once so it cannot
        quietly stop being true."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 30_000)})
        assert _stats(profile(frame, seed=0).html) == _stats(
            profile(frame, seed=0).html
        )
