"""`compare(a, b)` — what changed, as a structure rather than a verdict.

#65. `check` answers "should this build fail?"; this answers "what moved?".

The two share their arithmetic, which is the thing worth protecting: a gate and
a diff that disagree about what counts as a change are worse than either alone.
Several tests below assert that agreement directly.

Where they differ is deliberate. Category churn is here and not in the gate,
because top-k membership is not a census and reshuffles on counting noise — a
poor thing to fail a build on and the right thing to show a reader.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pysuricata import Comparison, compare, summarize
from pysuricata.check import Thresholds
from pysuricata.check import compare as gate
from pysuricata.comparison import KMV_RELATIVE_ERROR_PCT, render


def _frame(n: int = 20_000, *, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.standard_normal(n) + shift,
            "region": rng.choice(["north", "south", "east"], n).astype(object),
            "active": rng.random(n) > 0.5,
            "seen_at": pd.date_range("2026-01-01", periods=n, freq="min"),
        }
    )


class TestNothingChanged:
    def test_the_same_frame_has_no_schema_delta(self):
        assert not compare(_frame(), _frame()).schema.changed

    def test_every_column_is_compared(self):
        diff = compare(_frame(), _frame())
        assert set(diff.columns) == {"amount", "region", "active", "seen_at"}

    def test_no_column_moved(self):
        diff = compare(_frame(), _frame())
        for delta in diff.columns.values():
            assert delta.missing_pct_change in (None, 0.0)
            assert delta.categories_added == ()
            assert delta.categories_removed == ()

    def test_numeric_shifts_are_zero(self):
        delta = compare(_frame(), _frame()).columns["amount"]
        assert delta.mean_shift_sigma == pytest.approx(0.0, abs=1e-12)
        assert delta.std_ratio == pytest.approx(1.0, abs=1e-12)

    def test_the_rendering_says_so(self):
        assert "amount" not in render(compare(_frame(), _frame()))


class TestSchemaDelta:
    def test_an_added_column(self):
        after = _frame()
        # Varying, not constant: a single-valued column is reclassified as
        # categorical, which would make this assert the classifier rather than
        # the schema delta.
        after["extra"] = np.random.default_rng(1).standard_normal(len(after))
        assert compare(_frame(), after).schema.added == {"extra": "numeric"}

    def test_a_removed_column(self):
        assert compare(_frame(), _frame().drop(columns=["region"])).schema.removed == {
            "region": "categorical"
        }

    def test_a_retyped_column(self):
        after = _frame()
        after["region"] = np.arange(len(after), dtype=float)
        assert compare(_frame(), after).schema.retyped["region"][0] == "categorical"

    def test_a_retyped_column_gets_no_statistical_delta(self):
        """It is a schema fact; comparing a mean against a category count would
        be noise on top of something the reader already knows."""
        after = _frame()
        after["region"] = np.arange(len(after), dtype=float)
        assert "region" not in compare(_frame(), after).columns

    def test_unchanged_columns_are_listed(self):
        assert "amount" in compare(_frame(), _frame()).schema.unchanged


class TestNumericDeltas:
    @pytest.fixture
    def delta(self):
        return compare(_frame(), _frame(shift=2.0)).columns["amount"]

    def test_the_mean_shift_is_in_sigmas(self, delta):
        assert delta.mean_shift_sigma == pytest.approx(2.0, abs=0.05)

    def test_every_quartile_is_reported(self, delta):
        """A gate wants one number; a diff wants the shape."""
        assert delta.q1_shift_sigma == pytest.approx(2.0, abs=0.1)
        assert delta.median_shift_sigma == pytest.approx(2.0, abs=0.1)
        assert delta.q3_shift_sigma == pytest.approx(2.0, abs=0.1)

    def test_the_shift_is_signed(self):
        """A diff says which way, where a gate only needs how far."""
        down = compare(_frame(), _frame(shift=-2.0)).columns["amount"]
        assert down.mean_shift_sigma < 0

    def test_spread_is_a_ratio(self):
        wider = _frame()
        wider["amount"] = wider["amount"] * 3.0
        assert compare(_frame(), wider).columns["amount"].std_ratio == pytest.approx(
            3.0, rel=0.05
        )

    def test_the_range_is_carried(self, delta):
        assert delta.range_before is not None
        assert delta.range_after is not None
        assert delta.range_after[0] > delta.range_before[0]


class TestCategoryChurn:
    """The piece the gate deliberately leaves out."""

    @pytest.fixture
    def delta(self):
        rng = np.random.default_rng(3)
        after = _frame()
        after["region"] = rng.choice(["north", "south", "west"], len(after))
        return compare(_frame(), after).columns["region"]

    def test_a_new_category_is_named(self, delta):
        assert "west" in delta.categories_added

    def test_a_vanished_category_is_named(self, delta):
        assert "east" in delta.categories_removed

    def test_the_top_category_is_reported(self, delta):
        assert delta.top_category_before is not None
        assert delta.top_category_after is not None

    def test_churn_is_marked_approximate(self, delta):
        """Top-k membership is not a census."""
        assert delta.approximate

    def test_the_rendering_says_approx(self, delta):
        rng = np.random.default_rng(3)
        after = _frame()
        after["region"] = rng.choice(["north", "south", "west"], len(after))
        assert "(approx)" in render(compare(_frame(), after))

    def test_the_gate_does_not_fail_on_churn_alone(self):
        """Which is why it lives here: a build should not break because the tail
        of a top-k table reshuffled."""
        rng = np.random.default_rng(3)
        after = _frame()
        after["region"] = rng.choice(["north", "south", "west"], len(after))
        result = gate(summarize(after, seed=0), summarize(_frame(), seed=0))
        assert not any(f.kind == "categories" for f in result.findings)


class TestOtherKinds:
    def test_a_boolean_rate_change(self):
        rng = np.random.default_rng(4)
        after = _frame()
        after["active"] = rng.random(len(after)) > 0.9
        delta = compare(_frame(), after).columns["active"]
        assert delta.true_rate_change_pp == pytest.approx(-40.0, abs=2.0)

    def test_a_datetime_range_move(self):
        after = _frame()
        after["seen_at"] = pd.date_range("2026-03-01", periods=len(after), freq="min")
        delta = compare(_frame(), after).columns["seen_at"]
        assert delta.newest_after > delta.newest_before

    def test_missing_rate_in_percentage_points(self):
        after = _frame()
        after.loc[: len(after) // 2, "amount"] = np.nan
        delta = compare(_frame(), after).columns["amount"]
        assert delta.missing_pct_change == pytest.approx(50.0, abs=1.0)


class TestGrowthIsNotShapeChange:
    """The distinction the gate had to learn the hard way, available here as two
    separate numbers rather than as one rule."""

    def test_the_count_and_the_rate_are_both_reported(self):
        doubled = pd.concat([_frame(), _frame(seed=7)], ignore_index=True)
        delta = compare(_frame(), doubled).columns["amount"]
        assert delta.unique_change_pct > 50
        assert abs(delta.distinct_rate_change_pct) < 10

    def test_a_reader_can_tell_them_apart(self):
        """For a three-level enum it is the other way round, which is exactly
        why one number cannot answer the question."""
        doubled = pd.concat([_frame(), _frame(seed=7)], ignore_index=True)
        delta = compare(_frame(), doubled).columns["region"]
        assert abs(delta.unique_change_pct) < 1
        assert delta.distinct_rate_change_pct < -40


class TestApproximationIsNotOversold:
    @staticmethod
    def _payloads(unique_after: int):
        """Two payloads differing only in the distinct estimate, so the delta is
        exactly what the test says it is rather than whatever two random draws
        happened to produce."""
        column = {
            "type": "numeric",
            "count": 20_000,
            "missing": 0,
            "unique_est": 20_000,
            "std": 1.0,
        }
        before = {"dataset": {"rows_est": 20_000}, "columns": {"x": dict(column)}}
        after = {
            "dataset": {"rows_est": 20_000},
            "columns": {"x": dict(column, unique_est=unique_after)},
        }
        return before, after

    def test_the_rendering_hides_movement_inside_the_sketch_error(self):
        """Printing a 1% distinct-count change as a finding, when the sketch's
        own error is 2.2%, is the same mistake as printing an estimate as an
        exact integer."""
        assert KMV_RELATIVE_ERROR_PCT == pytest.approx(2.2, abs=0.1)
        before, after = self._payloads(20_200)  # +1.0%
        assert "~distinct" not in render(compare(before, after))

    def test_a_real_move_is_still_shown(self):
        before, after = self._payloads(26_000)  # +30%
        assert "~distinct" in render(compare(before, after))

    def test_the_structured_delta_still_carries_the_number(self):
        """Suppressed in the text, not lost."""
        before, after = self._payloads(20_200)
        assert compare(before, after).columns["x"].unique_change_pct == pytest.approx(
            1.0
        )


class TestTheContract:
    def test_it_takes_payloads_as_well_as_frames(self):
        """Two profiles already in hand should not be re-profiled."""
        before = summarize(_frame(), seed=0)
        after = summarize(_frame(shift=2.0), seed=0)
        assert compare(before, after).columns["amount"].mean_shift_sigma > 1

    def test_the_result_is_json_serialisable(self):
        payload = compare(_frame(), _frame(shift=2.0)).to_dict()
        assert json.loads(json.dumps(payload))["dataset"]["rows_after"] == 20_000

    def test_the_json_keeps_the_three_sections(self):
        payload = compare(_frame(), _frame()).to_dict()
        assert set(payload) == {"dataset", "schema", "columns"}

    def test_the_repr_is_one_line(self):
        text = repr(compare(_frame(), _frame(shift=2.0)))
        assert text.startswith("<Comparison")
        assert "\n" not in text

    def test_it_is_exported_from_the_package(self):
        import pysuricata

        assert "compare" in pysuricata.__all__
        assert isinstance(compare(_frame(), _frame()), Comparison)

    def test_there_is_no_verdict(self):
        """`check` decides; this describes."""
        assert not hasattr(compare(_frame(), _frame()), "passed")

    def test_both_sides_are_profiled_with_the_same_settings(self):
        """Comparing two profiles taken with different settings would report the
        settings as drift."""
        diff = compare(_frame(), _frame(), sample=1_000)
        assert diff.columns["amount"].mean_shift_sigma == pytest.approx(0.0, abs=1e-12)

    def test_the_seed_is_fixed_by_default(self):
        """So comparing a dataset against itself is a no-op rather than a set of
        sampling wobbles."""
        first = compare(_frame(), _frame()).to_dict()
        second = compare(_frame(), _frame()).to_dict()
        assert first == second


class TestTheGateAndTheDiffAgree:
    """One place computes what changed; the gate is thresholds on top."""

    def test_they_read_the_missing_rate_the_same_way(self):
        after = _frame()
        after.loc[: len(after) // 3, "amount"] = np.nan
        delta = compare(_frame(), after).columns["amount"]
        result = gate(
            summarize(after, seed=0),
            summarize(_frame(), seed=0),
            Thresholds(max_missing_drift_pp=1.0),
        )
        finding = next(f for f in result.findings if f.kind == "missing")
        assert finding.current == pytest.approx(delta.missing_pct_after)

    def test_they_read_the_sigma_shift_the_same_way(self):
        result = gate(
            summarize(_frame(shift=2.0), seed=0),
            summarize(_frame(), seed=0),
            Thresholds(max_mean_shift_sigma=0.1),
        )
        delta = compare(_frame(), _frame(shift=2.0)).columns["amount"]
        message = next(f.message for f in result.findings if "mean moved" in f.message)
        assert f"{abs(delta.mean_shift_sigma):.2f}σ" in message

    def test_they_share_the_noise_floor(self):
        from pysuricata import check

        assert check._KMV_RELATIVE_ERROR_PCT == KMV_RELATIVE_ERROR_PCT
