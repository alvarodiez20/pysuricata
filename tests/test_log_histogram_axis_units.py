"""A log chart labels its axis in the column's units, not in exponents (#264).

`Fare`'s log view captioned its peak bin `0.603–0.688`. Those are log10(4.01)
and log10(4.87), and **no fare is 0.603**. The axis ran roughly 0.6 to 2.7 for
a column whose values run 4 to 512, and nothing on the chart said the numbers
were exponents.

The bars are laid out in log space and have to be — that is what makes the
axis linear in the log of the value, which is the whole point of the view. So
the fix is not to stop transforming; it is that everything a *reader* sees has
to come back out of that space. Three things read `edges` and all three were
wrong in the same way, which is why `_in_data_units` exists rather than three
patches.

`HistogramData` used to carry an `original_range` field declared for exactly
this and never assigned anywhere. Carrying both ranges is the other way to fix
it, and that field is what happened to the second copy: it fell out of step
with the first by never being written at all.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

#: The report inlines its own CSS and JS, so a search of the whole document
#: finds class names in the source that references them.
_TAGS = re.compile(r"<script.*?</script>|<style.*?</style>", re.S)


@pytest.fixture(scope="module")
def body() -> str:
    """Titanic, which is what #264 was reported against.

    `Fare` spans 0 to 512 with a long right tail, so its log view is the one a
    reader would actually open — and it is the column whose caption read
    `0.603–0.688`.
    """
    frame = pd.read_csv("web/sample/titanic.csv")
    return _TAGS.sub("", profile(frame, seed=0).html)


def _variant(body: str, col: str, scale: str, bins: int) -> str:
    """One histogram variant, cut at the start of the next one.

    A fixed-size slice is not good enough here: the six variants sit next to
    each other in the document, so a window wide enough to hold the linear
    chart's labels also holds the log chart's, and a test asserting the linear
    ticks are evenly spaced then reads eighteen ticks from two charts and
    fails on a chart that is correct.
    """
    start = body.index(f'id="col_{col}-{scale}-bins-{bins}"')
    nxt = body.find('id="col_', start + 1)
    return body[start : nxt if nxt != -1 else len(body)]


def _caption(segment: str) -> str:
    return re.search(r'class="hist__caption">([^<]*)<', segment).group(1)


def _ticks(segment: str) -> list[float]:
    return [
        float(t.replace(",", ""))
        for t in re.findall(r'class="hist__tick"[^>]*>([^<]*)<', segment)
    ]


class TestTheLogAxisIsInDataUnits:
    def test_the_caption_no_longer_quotes_an_exponent(self, body: str) -> None:
        """The exact regression from the issue."""
        caption = _caption(_variant(body, "Fare", "log", 25))

        assert "0.603" not in caption, caption
        assert "0.688" not in caption, caption

    def test_the_peak_range_is_a_fare_someone_paid(self, body: str) -> None:
        caption = _caption(_variant(body, "Fare", "log", 25))
        low, high = re.search(
            r"peak [\d,]+ rows at ([\d.]+)–([\d.]+)", caption
        ).groups()

        # log10(4.01) = 0.603. The bin is the one holding fares around 4, and
        # that is what it must say.
        assert 3.0 <= float(low) <= 6.0, caption
        assert 3.0 <= float(high) <= 6.0, caption

    def test_the_ticks_span_the_column_not_its_logarithm(self, body: str) -> None:
        ticks = _ticks(_variant(body, "Fare", "log", 25))

        assert ticks, "no x ticks on the log variant"
        # Fares run to 512. In log space the axis would end near 2.7.
        assert max(ticks) > 100, ticks
        assert min(ticks) >= 0, ticks

    def test_the_ticks_are_unevenly_spaced(self, body: str) -> None:
        """What a log axis normally shows, and the visible sign that the
        labels are values rather than exponents: evenly spaced *positions*
        carrying geometrically growing *values*."""
        ticks = _ticks(_variant(body, "Fare", "log", 25))
        gaps = np.diff(ticks)

        assert (gaps > 0).all(), ticks
        # Each gap is larger than the last, because equal steps in log space
        # are multiplicative in data space.
        assert (np.diff(gaps) > 0).all(), ticks

    def test_the_tooltip_bounds_agree_with_the_axis(self, body: str) -> None:
        """`data-x0`/`data-x1` are the third consumer, and a reader comparing
        a hovered bar against the axis under it must not see two unit systems."""
        segment = _variant(body, "Fare", "log", 25)
        bounds = [
            (float(a), float(b))
            for a, b in re.findall(r'data-x0="([\d.]+)" data-x1="([\d.]+)"', segment)
        ]
        ticks = _ticks(segment)

        assert bounds, "no bars carry bounds on the log variant"
        assert min(x0 for x0, _ in bounds) >= min(ticks) * 0.95
        assert max(x1 for _, x1 in bounds) <= max(ticks) * 1.05


class TestTheLinearViewIsUnchanged:
    """The transform only applies on a log chart. A linear chart that started
    reporting `10 ** x` would be the same defect with the sign flipped."""

    def test_the_caption_still_reads_in_fares(self, body: str) -> None:
        assert "peak 519 rows at 0–20.5" in _caption(_variant(body, "Fare", "lin", 25))

    def test_the_ticks_are_evenly_spaced(self, body: str) -> None:
        ticks = _ticks(_variant(body, "Fare", "lin", 25))
        gaps = np.diff(ticks)

        assert np.allclose(gaps, gaps[0], rtol=0.02), ticks


class TestTheTwoViewsDescribeOneColumn:
    def test_they_agree_on_where_the_data_ends(self, body: str) -> None:
        """The strongest check available without reimplementing the binning:
        both views draw the same column, so their axes must end at the same
        value whatever space the bars were laid out in."""
        lin = _ticks(_variant(body, "Fare", "lin", 25))
        log = _ticks(_variant(body, "Fare", "log", 25))

        assert max(log) == pytest.approx(max(lin), rel=0.02), (lin[-1], log[-1])

    def test_the_log_view_still_says_what_it_dropped(self, body: str) -> None:
        """#258's fix, which lives in the same function. Un-logging the labels
        must not quietly take the exclusion note with it."""
        assert "rows not shown" in _caption(_variant(body, "Fare", "log", 25))


class TestTheHelperItself:
    @pytest.mark.parametrize("edge", [-1.0, 0.0, 0.5, 2.7, 3.0])
    def test_a_linear_chart_is_passed_through_untouched(self, edge: float) -> None:
        from pysuricata.render.histogram_svg import (
            HistogramData,
            SVGHistogramRenderer,
        )

        data = HistogramData(
            counts=np.array([1]),
            edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            total_count=1,
            scale="lin",
            y_max=1.0,
        )
        assert SVGHistogramRenderer._in_data_units(data, edge) == edge

    @pytest.mark.parametrize(
        "edge,expected", [(0.0, 1.0), (1.0, 10.0), (2.0, 100.0), (0.603, 4.007)]
    )
    def test_a_log_chart_is_raised_back(self, edge: float, expected: float) -> None:
        from pysuricata.render.histogram_svg import (
            HistogramData,
            SVGHistogramRenderer,
        )

        data = HistogramData(
            counts=np.array([1]),
            edges=np.array([0.0, 1.0]),
            bin_centers=np.array([0.5]),
            total_count=1,
            scale="log",
            y_max=1.0,
        )
        assert SVGHistogramRenderer._in_data_units(data, edge) == pytest.approx(
            expected, rel=1e-3
        )


def test_the_second_copy_of_the_range_is_gone() -> None:
    """`original_range` was declared for this problem and never assigned, so
    the chart mislabelled itself while carrying the field meant to prevent it.
    Formatting through `10 ** x` needs no second copy, and a second copy that
    can be forgotten is what this was."""
    from pysuricata.render.histogram_svg import HistogramData

    assert not hasattr(HistogramData, "original_range")
    assert "original_range" not in HistogramData.__dataclass_fields__
