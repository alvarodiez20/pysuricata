"""A bin count cannot be negative, and the report may not draw one (#253).

The Titanic `Fare` card's **50-bin** variant emitted this bar:

    <rect class="bar" ... height="-0.33" data-count="-1" data-pct="-0.1"/>

Three symptoms of one defect. `data-count` is what the tooltip reads, so a
reader hovering that bar was told the bin holds −1 rows; `height="-0.33"` is
invalid SVG, which Chrome logs and drops, three times per report load.

## The cause was not the one it looked like

The bars summed to exactly 891 — the row count — so one bin was −1 and another
was 1 too high, with the total conserved. That is the signature of counts
obtained by **differencing something cumulative**, and it points at the sketch
code. It was not that. `true_histogram_counts` leaves the accumulator with 25
bins and no negatives; the 50-bin variant is resampled in the *renderer*, and
the defect is in how that resampling rounds.

Splitting 25 bins over 50 gives fractional counts, and rounding each one
independently does not preserve the total, so a residual has to go somewhere.
The old code put **all** of it into a single bin chosen by
`argmax(count - round(count))`:

    diff = total_original - np.sum(new_counts_int)
    max_fractional_idx = np.argmax(fractional_parts)
    new_counts_int[max_fractional_idx] += int(diff)

That choice is only correct when the residual is positive — it finds the bin
rounded *down* hardest, the right one to give a row to and the wrong one to
take a row from. Measured on `Fare` at 50 bins: the residual is **−3**, and the
winning bin holds 2.5, which `np.round` sends to 2 under round-half-to-even,
giving it the largest fractional part (0.5) in the array. 2 − 3 = −1.

Conservation was never evidence of differencing. It was the arithmetic
identity of moving the whole residual to one place.

The fix is largest-remainder apportionment (Hare–Niemeyer): floor every bin,
then hand out the residual one row at a time, largest remainder first. Floors
are non-negative and only additions follow, so no bin can go below zero, and
exactly `residual` rows are handed out, so the total is still exact.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.render.histogram_svg import SVGHistogramRenderer

REPO = Path(__file__).resolve().parents[1]
TITANIC = REPO / "web" / "sample" / "titanic.csv"

#: The bin counts the report offers. The defect lived in 50 and not in 25,
#: because only an upsample produces fractional counts to round.
BIN_OPTIONS = (10, 25, 50)

#: A 25-bin histogram shaped like one the accumulator emits: uneven, with a
#: heavy bin, a long tail and three empty bins. The empty ones matter -- a bin
#: holding nothing is the one most easily pushed below zero, since it has no
#: rows to give back.
UNEVEN_EDGES = [float(i) for i in range(26)]
UNEVEN_COUNTS = [
    3, 17, 2, 0, 41, 9, 1, 5, 28, 6, 0, 12, 7,
    33, 2, 1, 19, 4, 0, 8, 15, 6, 2, 11, 3,
]  # fmt: skip


def _counts(svg: str) -> list[int]:
    """Every count the chart actually draws, sign included."""
    return [int(c) for c in re.findall(r'data-count="(-?\d+)"', svg)]


@pytest.fixture(scope="module")
def report() -> str:
    """A frame wide enough that several columns get numeric cards.

    Deliberately not `[1.0, 2, 3, 4, 5] * 40`: five distinct values profile as
    *categorical*, so a report built from that has no numeric card at all and
    every selector below would look dead while passing.
    """
    rng = np.random.default_rng(0)
    rows = 900
    return profile(
        pd.DataFrame(
            {
                "fare": rng.gamma(1.5, 22, rows),
                "age": rng.integers(1, 80, rows).astype(float),
                "score": rng.normal(0, 1, rows),
            }
        ),
        seed=0,
    ).html


class TestTheReportDrawsNoImpossibleBar:
    """The cheap guard the issue asked for, over a whole rendered report.

    It is deliberately not scoped to histograms: `data-count` and `height` mean
    the same thing on a temporal bar, and a negative one would be just as wrong
    there.
    """

    def test_no_rect_carries_a_negative_height(self, report):
        rects = re.findall(r"<rect[^>]*>", report)
        assert rects, "no rects in the report at all -- the fixture missed"

        offenders = [r for r in rects if re.search(r'height="-', r)]
        assert not offenders, (
            f"{len(offenders)} rect(s) carry a negative height, which is invalid "
            f"SVG -- the browser drops the element and logs an error: "
            f"{offenders[:2]}"
        )

    def test_no_element_reports_a_negative_count(self, report):
        offenders = re.findall(r'data-count="(-\d+)"', report)
        assert not offenders, (
            f"a bin reports {offenders[0]} rows. `data-count` is what the "
            f"tooltip reads out, so this is the report stating a fact about "
            f"the data that cannot be true"
        )

    def test_no_element_reports_a_negative_share(self, report):
        assert not re.findall(r'data-pct="(-[\d.]+)"', report)


class TestTheApportionmentIsExactAndNonNegative:
    """The two properties that make the rounding correct, asserted directly on
    the resampler rather than through a report."""

    @pytest.fixture(scope="class")
    def renderer(self) -> SVGHistogramRenderer:
        return SVGHistogramRenderer()

    @pytest.mark.parametrize("bins", BIN_OPTIONS)
    def test_a_resample_conserves_every_row(self, renderer, bins):
        """Rows may move between bins; none may be created or destroyed."""
        drawn = _counts(
            renderer.render_histogram_from_bins(
                UNEVEN_EDGES, UNEVEN_COUNTS, bins, "lin", "x", "c"
            )
        )

        assert sum(drawn) == sum(UNEVEN_COUNTS), (
            f"{bins} bins drew {sum(drawn)} rows out of {sum(UNEVEN_COUNTS)}"
        )

    @pytest.mark.parametrize("bins", BIN_OPTIONS)
    def test_a_resample_never_produces_a_negative_bin(self, renderer, bins):
        drawn = _counts(
            renderer.render_histogram_from_bins(
                UNEVEN_EDGES, UNEVEN_COUNTS, bins, "lin", "x", "c"
            )
        )

        assert all(c > 0 for c in drawn), [c for c in drawn if c <= 0]

    def test_the_residual_is_spread_rather_than_dumped(self, renderer):
        """The property the old code broke, stated as itself.

        Handing the whole residual to one bin distorts that bin by the size of
        the residual. Largest-remainder moves each bin by at most one row, so
        every drawn count stays within 1 of its exact fractional share.
        """
        edges = [0.0, 1.0, 2.0, 3.0]
        counts = [7, 7, 7]
        bins = 7

        drawn = _counts(
            renderer.render_histogram_from_bins(edges, counts, bins, "lin", "x", "c")
        )
        exact = sum(counts) / bins  # 3.0 rows per bin

        assert sum(drawn) == sum(counts)
        for count in drawn:
            assert abs(count - exact) <= 1, (
                f"a bin drew {count} against an exact share of {exact}; the "
                f"residual was concentrated rather than apportioned"
            )

    def test_the_residual_is_never_negative(self):
        """Why the apportionment needs no give-rows-back branch, and no clamp.

        The first draft had one, and `codecov/patch` flagged it as unreached.
        `np.floor` never rounds up, so `floors.sum() <= scaled.sum()`, which the
        scaling puts at the original total to within float noise; both sides
        are integers, so a negative residual needs an error of one part in the
        row count against noise nearer one part in 1e15. Deleted rather than
        left untested, and the bound asserted here instead -- if some future
        edit makes a residual go negative, that edit needs to handle it, and
        this is what will say so.
        """
        rng = np.random.default_rng(0)

        worst = 0
        for _ in range(20_000):
            weights = rng.random(int(rng.integers(2, 60)))
            weights *= rng.choice([1.0, 1e3, 1e6])
            total = int(rng.integers(1, 10_000_000))
            if weights.sum() <= 0:
                continue

            scaled = weights * (total / weights.sum())
            worst = min(worst, total - int(np.floor(scaled).astype(np.int64).sum()))

        assert worst >= 0, (
            f"flooring overshot the total by {-worst} rows, so the residual can "
            f"go negative and the apportionment can now drive a bin below zero"
        )


class TestTheRendererRefusesAnImpossibleCount:
    """The second line of the fix. The apportionment is what stops a negative
    count being produced; this is what stops one being *drawn* if some future
    path produces one anyway.

    Zero and negative are two different statements. Zero is a drawing decision
    -- rule 3, an empty bin draws nothing. Negative is a value that cannot
    exist, and turning it into geometry is how it reached a reader.
    """

    def test_a_negative_count_emits_no_rect(self):
        from pysuricata.render.histogram_svg import HistogramData

        renderer = SVGHistogramRenderer()
        data = HistogramData(
            counts=np.array([5, -1, 8]),
            bin_centers=np.array([0.5, 1.5, 2.5]),
            edges=np.array([0.0, 1.0, 2.0, 3.0]),
            total_count=12,
            scale="lin",
            y_max=8,
        )

        markup = "".join(renderer._render_bars(data, "c"))

        assert 'data-count="-1"' not in markup
        assert 'height="-' not in markup
        assert _counts(markup) == [5, 8]


@pytest.mark.skipif(not TITANIC.exists(), reason="the sample dataset is absent")
class TestTheReportedCaseItself:
    """#253 as filed, on the data it was filed against.

    The synthetic fixtures above assert the properties; this asserts the bug.
    A property test that happens not to hit the failing arrangement passes
    while the reported defect is still there.
    """

    @pytest.fixture(scope="class")
    def fare(self) -> dict:
        payload = summarize(pd.read_csv(TITANIC), seed=0)
        return payload["columns"]["Fare"]

    def test_the_payload_was_never_the_problem(self, fare):
        """Worth pinning: it locates the defect in the renderer, and a future
        reader who sees a negative count should not start in the sketches."""
        counts = fare["true_histogram_counts"]

        assert len(counts) == 25
        assert min(counts) >= 0, "the accumulator itself now emits a negative bin"

    def test_the_fifty_bin_variant_is_clean(self, fare):
        drawn = _counts(
            SVGHistogramRenderer().render_histogram_from_bins(
                fare["true_histogram_edges"],
                fare["true_histogram_counts"],
                50,
                "lin",
                "Fare",
                "c",
            )
        )

        assert min(drawn) > 0, f"negative bins remain: {[c for c in drawn if c <= 0]}"
        assert sum(drawn) == sum(fare["true_histogram_counts"]) == 891

    def test_the_whole_report_is_clean(self):
        html = profile(pd.read_csv(TITANIC), seed=0).html

        assert not re.findall(r'data-count="(-\d+)"', html)
        assert not re.findall(r'height="(-[\d.]+)"', html)
