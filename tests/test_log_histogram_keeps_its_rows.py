"""The log histogram excludes values, not bins (#258).

`render_histogram_from_bins` computed its positive mask over **edges** and then
sliced it to index **counts**:

    positive_mask = original_edges > 0
    positive_counts = original_counts[positive_mask[:-1]]

For a column whose minimum is 0, `edges[0] == 0` makes `positive_mask[0]` false,
and `positive_mask[:-1]` therefore drops the count of the **entire first bin**.

Measured on the Titanic `Fare` column, which is right-skewed with a large mass
near zero:

| | |
|---|---|
| rows | 891 |
| fares actually `<= 0` | **15** |
| first bin, at 25 bins | `[0, 20.5]`, holding **519** |
| rows the log variants drew | **372** |
| rows silently discarded | **519 — 58% of the column** |

A log axis must exclude non-positive values. The defect was the granularity:
15 values cannot be logged, and 519 were thrown away with them, because the
exclusion was applied to a bin rather than to the values inside it. Nothing on
the chart said so, so a reader comparing the linear and log views of one column
saw two different distributions with no way to tell that one was missing more
than half its rows.

## What replaced it

A bin is drawable when *any* of it is positive, which is its **right** edge
being positive. The single bin that straddles zero is clipped to the column's
smallest positive value -- carried through `StreamingMoments`, beside the
positive-count state the geometric mean already needed -- rather than dropped
whole. Its zeros and negatives are subtracted from its count, because they lie
to the left of the new edge and keeping them would trade a 58% undercount for a
small overcount. Both are charts that do not add up.

And the caption says what is missing, which is worth doing whichever way the
first part is decided: the count is never zero for a column with zeros in it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.accumulators.algorithms import StreamingMoments
from pysuricata.render.histogram_svg import SVGHistogramRenderer

BIN_OPTIONS = (10, 25, 50)


def _drawn(svg: str) -> int:
    return sum(int(c) for c in re.findall(r'data-count="(-?\d+)"', svg))


def _caption(figure: str) -> str:
    found = re.search(r'class="hist__caption">([^<]*)<', figure)
    assert found, "no caption on the figure"
    return found.group(1)


@pytest.fixture(scope="module")
def renderer() -> SVGHistogramRenderer:
    return SVGHistogramRenderer()


@pytest.fixture(scope="module")
def skewed():
    """25 linear bins over `[0, 100]`, with the mass in the first one.

    The shape that makes the defect large: a column whose minimum is zero and
    whose first bin holds most of the rows.
    """
    edges = [float(i * 4) for i in range(26)]
    counts = [500] + [10] * 24
    return edges, counts


class TestTheFirstBinIsNotThrownAway:
    @pytest.mark.parametrize("bins", BIN_OPTIONS)
    def test_the_log_view_keeps_the_rows_that_can_be_logged(
        self, renderer, skewed, bins
    ):
        edges, counts = skewed

        svg = renderer.render_histogram_from_bins(
            edges, counts, bins, "log", "x", "c", min_positive=0.5, non_positive=3
        )

        assert _drawn(svg) == sum(counts) - 3, (
            "the log view dropped more than the rows that cannot be logged"
        )

    def test_without_the_fix_the_whole_bin_would_go(self, renderer, skewed):
        """Pins the size of the defect rather than only its absence.

        Omitting `min_positive` is the old behaviour, kept as the fallback for
        a caller that has no positive value to anchor the bin with -- and it
        still loses every row in that bin, which is what made this worth 500 of
        740 rows here.
        """
        edges, counts = skewed

        without = renderer.render_histogram_from_bins(
            edges, counts, 25, "log", "x", "c"
        )
        with_it = renderer.render_histogram_from_bins(
            edges, counts, 25, "log", "x", "c", min_positive=0.5, non_positive=3
        )

        assert _drawn(without) == sum(counts) - 500
        assert _drawn(with_it) > _drawn(without)


class TestOnlyTheUndrawableRowsAreExcluded:
    """The count has to be exact in both directions. Dropping the bin
    understates by 519; keeping it whole overstates by 15."""

    def test_the_titanic_case_from_the_issue(self):
        frame = pd.read_csv("web/sample/titanic.csv")
        zeros = int((frame["Fare"] <= 0).sum())
        html = profile(frame, seed=0).html

        drawn = {}
        for variant in re.findall(r'id="(col_Fare-\w+-bins-\d+)"', html):
            start = html.index(f'id="{variant}"')
            drawn[variant] = _drawn(html[start : html.index("</figure>", start)])

        assert drawn, "no Fare histogram variants in the report"
        for variant, rows in drawn.items():
            expected = len(frame) - (zeros if "-log-" in variant else 0)
            assert rows == expected, f"{variant} drew {rows}, expected {expected}"

    def test_a_column_with_no_zeros_loses_nothing(self, renderer):
        edges = [1.0 + i for i in range(11)]
        counts = [7] * 10

        svg = renderer.render_histogram_from_bins(
            edges, counts, 25, "log", "x", "c", min_positive=1.0, non_positive=0
        )

        assert _drawn(svg) == sum(counts)

    def test_bins_entirely_below_zero_go_whole(self, renderer):
        """A bin whose right edge is at or below zero holds nothing loggable,
        so it is excluded with its full count -- the clipping applies only to
        the one bin that straddles."""
        edges = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0]
        counts = [4, 6, 5, 40, 30]

        svg = renderer.render_histogram_from_bins(
            edges, counts, 10, "log", "x", "c", min_positive=2.0, non_positive=15
        )

        assert _drawn(svg) == 70, "only the two positive bins should survive"


class TestTheChartSaysWhatItOmits:
    def test_the_caption_states_the_excluded_rows(self, renderer, skewed):
        edges, counts = skewed

        figure = renderer.render_histogram_from_bins(
            edges, counts, 25, "log", "x", "c", min_positive=0.5, non_positive=3
        )

        assert "3 rows not shown" in _caption(figure)

    def test_a_linear_chart_says_nothing_because_it_omits_nothing(
        self, renderer, skewed
    ):
        edges, counts = skewed

        figure = renderer.render_histogram_from_bins(edges, counts, 25, "lin", "x", "c")

        assert "not shown" not in _caption(figure)
        assert _drawn(figure) == sum(counts)

    def test_one_excluded_row_is_singular(self, renderer, skewed):
        edges, counts = skewed

        figure = renderer.render_histogram_from_bins(
            edges, counts, 25, "log", "x", "c", min_positive=0.5, non_positive=1
        )

        assert "1 row not shown" in _caption(figure)


class TestTheAccumulatorCarriesTheSmallestPositiveValue:
    """The clip needs a value from the data. It goes beside the positive-count
    state the geometric mean already maintains, so it costs one `min()`."""

    def test_it_is_the_smallest_strictly_positive_value(self):
        moments = StreamingMoments()
        moments.update(np.array([0.0, 5.0, -3.0, 0.25, 9.0]))

        assert moments.get_statistics()["min_positive"] == 0.25

    def test_a_column_with_no_positive_value_reports_none(self):
        moments = StreamingMoments()
        moments.update(np.array([0.0, -1.0, -7.5]))

        assert moments.get_statistics()["min_positive"] is None

    def test_it_is_order_independent(self):
        """The invariant every accumulator in this project has to hold:
        chunked must equal unchunked."""
        values = np.array([0.0, 5.0, -3.0, 0.25, 9.0, 0.0, 100.0, 0.5])

        whole = StreamingMoments()
        whole.update(values)

        chunked = StreamingMoments()
        for piece in (values[:3], values[3:5], values[5:]):
            part = StreamingMoments()
            part.update(piece)
            chunked.merge(part)

        assert (
            chunked.get_statistics()["min_positive"]
            == whole.get_statistics()["min_positive"]
            == 0.25
        )

    def test_merging_into_an_empty_accumulator_keeps_the_value(self):
        """The `count == 0` short-circuit in `merge` copies state field by
        field, so a field added without touching it silently stays at its
        identity."""
        filled = StreamingMoments()
        filled.update(np.array([3.0, 8.0]))

        empty = StreamingMoments()
        empty.merge(filled)

        assert empty.get_statistics()["min_positive"] == 3.0

    def test_it_reaches_the_numeric_summary(self):
        from pysuricata.accumulators.numeric import NumericAccumulator

        accumulator = NumericAccumulator("fare")
        accumulator.update(np.array([0.0, 4.5, 12.0, 0.0, 3.25]))

        assert accumulator.finalize().min_positive == 3.25
