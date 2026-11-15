"""An approximate figure says how approximate, and the interval is honest.

`(≈)` told a reader *that* a number was an estimate and nothing about how good
one. These tests pin the three things that make the interval printed beside it
worth reading:

1. **It covers.** A 95% interval that contains the truth 70% of the time is a
   decoration. Each interval is checked against an exact computation over many
   independently drawn columns, the same oracle idea as
   `test_estimate_contracts.py`, applied to the bound instead of the point.
2. **It is silent when the figure is exact.** `± 0` on a counted value is a
   claim about a sketch that never ran.
3. **The report and the payload print the same one.** Two surfaces producing
   one bound separately is how the duplicate count drifted (#202).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

import pysuricata
from pysuricata.accumulators import NumericConfig
from pysuricata.accumulators.margins import (
    distinct_interval,
    quantile_interval,
    scaled_count_interval,
    t_critical_95,
)
from pysuricata.accumulators.numeric import NumericAccumulator
from pysuricata.render.margin import (
    count_margin_html,
    format_margin,
    lower_bound_note,
    margin_html,
)

#: Columns drawn per coverage test. At 200 a true 95% interval lands between
#: 91% and 98.5% of the time 99.9% of the time, so the floor below is not noise.
ROUNDS = 200
COVERAGE_FLOOR = 0.90


def _markup(html: str) -> str:
    """The report without the CSS and JS it inlines -- see CLAUDE.md."""
    return re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)


def _cells(html: str) -> dict[str, str]:
    """Stat-row caption -> the full inner HTML of its value cell."""
    return dict(
        re.findall(
            r'vstat__cap">([^<]*)</div><div class="vstat__val">(.*?)</div>',
            _markup(html),
            flags=re.S,
        )
    )


def _summaries(make, n: int):
    for seed in range(ROUNDS):
        values = make(np.random.default_rng(seed), n)
        acc = NumericAccumulator("a", NumericConfig(), seed=seed)
        acc.update(values)
        yield values, acc.finalize()


class TestTheIntervalsCover:
    """Against truth, over columns three times the reservoir."""

    N = 60_000

    @pytest.fixture(scope="class")
    def runs(self):
        return list(_summaries(lambda rng, n: rng.lognormal(0.0, 1.0, n), self.N))

    @pytest.mark.parametrize(("name", "p"), [("q1", 25), ("median", 50), ("q3", 75)])
    def test_a_quartile_interval_contains_the_true_quartile(self, runs, name, p):
        hits = sum(
            getattr(s, f"{name}_ci_lo")
            <= np.percentile(values, p)
            <= getattr(s, f"{name}_ci_hi")
            for values, s in runs
        )
        assert hits / ROUNDS >= COVERAGE_FLOOR

    def test_the_outlier_interval_contains_the_true_count(self, runs):
        def exact(values):
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())

        hits = sum(
            s.outliers_iqr_ci_lo <= exact(values) <= s.outliers_iqr_ci_hi
            for values, s in runs
        )
        assert hits / ROUNDS >= COVERAGE_FLOOR

    def test_the_distinct_interval_contains_the_true_count(self):
        """On a column where the clamp at the row count does not bind.

        A continuous column has as many distinct values as rows, so its
        interval is cut off at the ceiling and covers trivially. 9,000 levels
        in 60,000 rows leaves both arms free.
        """
        runs = _summaries(
            lambda rng, n: rng.integers(0, 9_000, n).astype(float), self.N
        )
        hits = widths = 0
        for values, s in runs:
            assert not s.unique_est_exact
            hits += s.unique_ci_lo <= len(np.unique(values)) <= s.unique_ci_hi
            widths += s.unique_ci_hi - s.unique_ci_lo
        assert hits / ROUNDS >= COVERAGE_FLOOR
        # And not by being uselessly wide: about 2 x 1.96 x 2.2% of 9,000.
        assert widths / ROUNDS < 0.12 * 9_000

    def test_the_interval_narrows_as_the_reservoir_nears_the_column(self):
        """The finite-population correction, which is most of the answer when
        the reservoir has seen four fifths of the column."""
        ordered = np.sort(np.random.default_rng(0).normal(0, 1, 20_000))
        lo_far, hi_far = quantile_interval(ordered, 0.5, 2_000_000)
        lo_near, hi_near = quantile_interval(ordered, 0.5, 25_000)
        assert (hi_near - lo_near) < 0.6 * (hi_far - lo_far)


class TestAnExactFigureHasNoMargin:
    @pytest.fixture(scope="class")
    def short(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 500),
                "c": rng.choice(["a", "b", "c"], 500).astype(object),
                "t": pd.date_range("2024-01-01", periods=500, freq="h"),
            }
        )
        return pysuricata.summarize(frame), pysuricata.profile(frame).html

    def test_the_payload_interval_is_zero_width(self, short):
        payload, _ = short
        x = payload["columns"]["x"]
        for name in ("unique", "q1", "median", "q3", "outliers_iqr"):
            value = (
                x["unique_est" if name == "unique" else name]
                if name != ("outliers_iqr")
                else x["outliers_iqr_est"]
            )
            assert x[f"{name}_ci_lo"] == x[f"{name}_ci_hi"] == value, name
        for column in ("c", "t"):
            stats = payload["columns"][column]
            assert stats["unique_ci_lo"] == stats["unique_ci_hi"] == stats["unique_est"]

    def test_the_report_prints_none(self, short):
        _, html = short
        assert 'class="moe"' not in _markup(html)
        assert "moe-note" not in _markup(html)

    def test_and_does_not_call_the_figure_approximate(self, short):
        """Three labels that used to say `(≈)` on counted values."""
        _, html = short
        markup = _markup(html)
        assert "(≈)" not in markup.replace("Processed bytes (≈)", "")

    def test_the_helpers_agree(self):
        assert distinct_interval(599, 2_048, True, 891) == (599, 599)
        assert quantile_interval([1.0, 2.0, 3.0], 0.5, 3) is None
        assert scaled_count_interval(4, 500, 500) == (4, 4)
        assert margin_html(5.0, 5.0, 5.0) == ""
        assert margin_html(5.0, None, 6.0) == ""
        assert margin_html(5.0, float("nan"), 6.0) == ""
        assert lower_bound_note(0) == ""


class TestTheReportAndThePayloadPrintTheSameInterval:
    @pytest.fixture(scope="class")
    def long(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x": rng.lognormal(3.0, 1.2, 60_000)})
        return pysuricata.summarize(frame)["columns"]["x"], _cells(
            pysuricata.profile(frame).html
        )

    def test_the_figure_is_still_the_first_thing_in_the_cell(self, long):
        """A scraper reading up to the first tag must get what it got before."""
        stats, cells = long
        leading = cells["Median (≈)"].split("<")[0]
        assert float(leading) == pytest.approx(stats["median"], rel=1e-3)

    def test_the_unique_margin_is_the_payload_half_width(self, long):
        stats, cells = long
        half = (stats["unique_ci_hi"] - stats["unique_ci_lo"]) / 2
        printed = re.search(r'class="moe"[^>]*>(.*?)<', cells["Unique (≈)"]).group(1)
        if printed.startswith("±"):
            assert int(printed[1:].replace(",", "")) == math.ceil(half)
        else:  # clamped at the row count, so printed as its two bounds
            assert f"{stats['unique_ci_lo']:,}" in printed
            assert f"{stats['unique_ci_hi']:,}" in printed

    @pytest.mark.parametrize(
        ("label", "key"),
        [("Q1 (P25) (≈)", "q1"), ("Median (≈)", "median"), ("Q3 (P75) (≈)", "q3")],
    )
    def test_each_quartile_carries_one(self, long, label, key):
        stats, cells = long
        assert 'class="moe"' in cells[label]
        assert stats[f"{key}_ci_lo"] < stats[key] < stats[f"{key}_ci_hi"]

    def test_the_exact_neighbours_carry_none(self, long):
        """Min, Max and Mean see every value. A margin on them would say the
        profiler is unsure of something it counted."""
        _, cells = long
        for label in ("Min", "Max", "Mean", "Count", "Zeros", "Missing"):
            assert "moe" not in cells[label], label


class TestAKeyColumnCannotBeCertifiedUnique:
    """The clamp at the row count makes the interval one-sided, and says so."""

    def test_the_interval_stops_at_the_row_count(self):
        lo, hi = distinct_interval(126_000, 2_048, False, 126_000)
        assert hi == 126_000 and lo < 126_000

    def test_it_is_printed_as_bounds_not_as_plus_or_minus(self):
        html = count_margin_html(126_000, 120_541, 126_000)
        assert "±" not in html
        assert "(120,541 – 126,000)" in html

    def test_the_identifier_card_shows_what_the_sketch_cannot_rule_out(self):
        frame = pd.DataFrame({"id": np.arange(30_000)})
        markup = _markup(pysuricata.profile(frame).html)
        duplicates = re.search(
            r'Duplicates \(≈\)</div><div class="vstat__val">(.*?)</div>', markup
        ).group(1)
        # Whatever the estimate, zero is inside what the sketch allows -- and a
        # count is never printed as `791 ± 1,029`, which runs below zero.
        assert "±" not in duplicates
        assert re.search(r"\(0 – [\d,]+\)", duplicates), duplicates


class TestZeroHitsIsNotCertainlyNone:
    def test_the_interval_has_an_upper_arm(self):
        lo, hi = scaled_count_interval(0, 20_000, 126_000)
        assert lo == 0 and 0 < hi < 100

    def test_sampled_hits_are_a_hard_floor(self):
        lo, hi = scaled_count_interval(50, 20_000, 60_000)
        assert lo >= 50 and hi <= 60_000 - (20_000 - 50)


class TestTopKCountsSayTheyAreLowerBounds:
    """In the payload since #328, and printed with no bound until now."""

    @pytest.fixture(scope="class")
    def lossy(self):
        rng = np.random.default_rng(0)
        # Zipf, not uniform: a flat 1,000 levels is high-cardinality and gets
        # the Shape pane, which has no table for a caption to sit under.
        skus = [f"sku-{i:05d}" for i in rng.zipf(1.3, 120_000) % 30_000]
        frame = pd.DataFrame({"sku": skus})
        return pysuricata.summarize(frame)["columns"]["sku"], _markup(
            pysuricata.profile(frame).html
        )

    def test_the_fixture_actually_evicts(self, lossy):
        stats, _ = lossy
        assert stats["top_items_uncertainty"] > 0

    def test_the_caption_prints_the_payload_bound(self, lossy):
        stats, markup = lossy
        assert 'class="common-values-table' in markup
        assert f"may be short by up to {stats['top_items_uncertainty']:,}" in markup

    def test_the_note_is_one_sided(self):
        note = lower_bound_note(1_112)
        assert "1,112" in note and "±" not in note


class TestTheMeanIntervalUsesStudentsT:
    @pytest.mark.parametrize(
        ("df", "expected"), [(1, 12.706), (4, 2.776), (30, 2.042), (60, 2.000)]
    )
    def test_the_critical_value(self, df, expected):
        assert t_critical_95(df) == pytest.approx(expected, abs=2e-3)

    def test_it_approaches_the_normal_point(self):
        assert t_critical_95(1_000_000) == pytest.approx(1.95996, abs=1e-4)
        assert math.isnan(t_critical_95(0))

    def test_five_values_get_the_wide_interval(self):
        stats = pysuricata.summarize(pd.DataFrame({"x": [1.0, 2.5, 3.0, 4.75, 9.0]}))
        x = stats["columns"]["x"]
        half = (x["ci_hi"] - x["ci_lo"]) / 2
        assert half == pytest.approx(2.776 * x["se"], rel=1e-3)


class TestTheMarginIsFormattedToWhatItKnows:
    @pytest.mark.parametrize(
        ("value", "shown"),
        [(0.01552, "0.016"), (0.31, "0.31"), (3.14159, "3.1"), (580.4, "580")],
    )
    def test_two_significant_figures(self, value, shown):
        assert format_margin(value) == shown

    def test_a_count_margin_rounds_up(self):
        assert "± 4<" in count_margin_html(100, 97, 104)
