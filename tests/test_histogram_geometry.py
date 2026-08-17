"""The histogram's two coordinate systems, and the rules that keep them apart.

#147, phase 5d of the design package. The measured problem: at a 1240px
viewport the `<svg>` element was 1,099px wide and the bars occupied **356px**
of it -- 68% blank -- because `preserveAspectRatio` defaults to `xMidYMid meet`
and the container's fixed height was the limiting dimension.

The bind that makes it non-trivial, worth restating because every "obvious"
fix runs into it:

    Uniform scale  =>  text size tracks the viewport.
    Fixed text     =>  the canvas has to be ~1:1 with its display size.
    One static SVG cannot be 1:1 at both 1,099px and 284px.

So the SVG holds only what should stretch, and every glyph is HTML at a
percentage offset. These tests guard the invariants that arrangement depends
on -- the ones a later edit would silently undo, since all of them still
*render* when broken.

Layout itself is not asserted here: there is no browser in the test
environment. What was measured in one, at 1240px and 390px, and what a reader
should re-measure if they doubt these:

    coverage of the plot width by bars .... 100% at both
    tick label size ....................... 11px at both
    bar separator ......................... 1px at both
    x ticks visible ....................... 9 / 5 / 3
    plot left edge, every numeric card .... identical
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

#: The same frame `test_report_layout.py` measures against.
TITANIC = Path(__file__).resolve().parent.parent / "docs" / "assets" / "titanic.csv"
from pysuricata.render.histogram_svg import (
    SVGHistogramRenderer,
    _round_preserving_total,
)

CSS = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"


@pytest.fixture(scope="module")
def report() -> str:
    rng = np.random.default_rng(0)
    return profile(
        pd.DataFrame(
            {
                "age_years": rng.integers(1, 90, 500).astype(float),
                "score": rng.normal(0, 1, 500),
                "gap": rng.gamma(2, 50, 500),
            }
        ),
        seed=0,
    ).html


@pytest.fixture(scope="module")
def figures(report: str) -> list[str]:
    found = re.findall(r'<figure class="hist".*?</figure>', report, re.S)
    assert found, "no histogram figures in the report"
    return found


def _chart(bins: int = 25) -> str:
    renderer = SVGHistogramRenderer()
    return renderer.render_histogram_from_bins(
        bin_edges=[0.0, 10.0, 20.0, 30.0, 40.0],
        bin_counts=[5, 20, 0, 8],
        bins=bins,
        scale="lin",
        title="age_years",
        col_id="c1",
    )


class TestTextNeverLivesInsideTheScaledSvg:
    """Rule 4. The load-bearing one: with `preserveAspectRatio="none"` the x
    and y scales differ, so a glyph inside the SVG is stretched by their
    ratio -- 2.6x horizontally at a 1240px viewport."""

    def test_no_text_element_in_any_histogram(self, report):
        for svg in re.findall(r'<svg class="hist-svg".*?</svg>', report, re.S):
            assert "<text" not in svg, svg[:200]

    def test_the_labels_exist_somewhere_else(self, figures):
        """Otherwise the rule above is satisfied by a chart with no labels."""
        joined = "".join(figures)
        assert 'class="hist__y"' in joined
        assert 'class="hist__tick"' in joined
        assert 'class="hist__unit"' in joined

    def test_the_svg_declares_the_stretch(self, report):
        for svg in re.findall(r'<svg class="hist-svg"[^>]*>', report):
            assert 'preserveAspectRatio="none"' in svg


class TestEveryStrokeSurvivesTheStretch:
    """A 1-unit line in viewBox space is 11px wide and 0.28px tall at a
    1,100x210 plot. `non-scaling-stroke` is what makes a hairline a hairline."""

    @pytest.mark.parametrize("element", ["rect", "line"])
    def test_marks_carry_a_mark_class(self, report, element):
        """The classes are the hook the rule below hangs on."""
        pattern = rf"<{element}[^>]*class=\"(?:bar|grid|axis)\"[^>]*>"
        found = re.findall(pattern, report) + re.findall(
            rf'<{element} class="(?:bar|grid|axis)"[^>]*>', report
        )
        assert found, f"no {element} marks found"

    @pytest.mark.parametrize("mark", ["bar", "grid", "axis"])
    def test_every_mark_class_gets_a_non_scaling_stroke(self, mark):
        """Declared once in CSS, not repeated on every element.

        This used to assert the `vector-effect="non-scaling-stroke"` attribute
        on each mark, which pinned the *mechanism* rather than the invariant.
        The attribute was 41 bytes on every one of 50 bars in each of 6
        variants of every numeric column, so #206 moved it into the `.bar`,
        `.grid` and `.axis` rules -- beside the strokes it modifies, where the
        stylesheet's own comment already explained why it was needed. Verified
        pixel-identical before and after.

        The invariant is unchanged and still guarded: drop the declaration and
        this fails. `tests/test_report_layout.py` asserts the same thing from
        the other end, on computed style in a browser, which is the form that
        cannot be satisfied by a declaration that never applies.
        """
        sheet = (
            Path(__file__).resolve().parents[1]
            / "pysuricata"
            / "static"
            / "css"
            / "_07-histogram.css"
        ).read_text(encoding="utf-8")

        rule = re.search(rf"\.hist-svg \.{mark} \{{(.*?)\}}", sheet, re.S)
        assert rule, f"no .hist-svg .{mark} rule in the histogram stylesheet"
        assert "vector-effect: non-scaling-stroke" in rule.group(1), (
            f"`.{mark}` does not get a non-scaling stroke, so its hairline "
            f"scales with the plot -- 1.1px at 1,100px and 0.28px at 284px"
        )


class TestABarPaysOnlyForWhatIsRead:
    """#206. A bar is the most repeated element in the report -- 50 of them in
    each of 6 variants of every numeric column -- so anything constant on it is
    multiplied by 300 per column.

    Two things were, and are not now: `vector-effect="non-scaling-stroke"` (41
    bytes, moved to the `.bar` rule) and a third decimal on four coordinates
    (the viewBox is 0..100, so at 1,100px that digit is a ten-thousandth of a
    pixel). A bar went from 184 bytes to 131, and the marginal cost of a
    numeric column from 73,204 to 63,596.

    What stayed is what something reads. `data-count`, `data-pct`, `data-x0`
    and `data-x1` drive the tooltip; `data-col` scopes the first two for
    `scripts/report_fingerprint.py`, which takes an element's scope from the
    same tag -- drop it and every `attr::col_age::count` collapses to
    `attr::::count`, colliding the bar counts of every numeric column under one
    key. That is a weaker invariance guard bought with 19 bytes a bar, and it
    was measured and then put back.
    """

    def _a_bar(self, report: str) -> str:
        bars = re.findall(r'<rect class="bar"[^>]*>', report)
        assert bars, "no histogram bars found"
        return max(bars, key=len)

    def test_a_bar_carries_no_repeated_constant(self, report):
        bar = self._a_bar(report)

        assert "vector-effect" not in bar, (
            "`vector-effect` is back on the bar; it belongs in the `.bar` rule, "
            "where it costs 41 bytes once instead of once per bar"
        )
        assert not re.search(r'="\d+\.\d{3,}"', bar), (
            f"a coordinate carries three or more decimals: {bar}"
        )

    def test_a_bar_still_carries_everything_that_is_read(self, report):
        bar = self._a_bar(report)

        for attribute, reader in (
            ("data-count", "the tooltip"),
            ("data-pct", "the tooltip"),
            ("data-x0", "the tooltip's range line"),
            ("data-x1", "the tooltip's range line"),
            ("data-col", "report_fingerprint.py, to scope the counts per column"),
        ):
            assert attribute in bar, f"{attribute} is gone, and {reader} reads it"


class TestTheBarGapIsNotGeometry:
    """5d.7, and not optional.

    `bar_w = max(1, bar_width - 1)` was a 1-unit gap in viewBox space, which
    scales with x: 1.1px at a 1,100px plot, 0.56px at 560px, **0.28px at
    284px** -- where the bars merge into one block.
    """

    def test_bars_are_edge_to_edge(self):
        chart = _chart()
        rects = [
            (float(m.group(1)), float(m.group(2)))
            for m in re.finditer(
                r'<rect class="bar" x="([\d.]+)"[^>]*width="([\d.]+)"', chart
            )
        ]
        assert len(rects) >= 2
        width = rects[0][1]
        for (x0, w0), (x1, _) in zip(rects, rects[1:], strict=False):
            # Bins may be skipped where the count is zero, so the step is a
            # whole number of bar widths -- but never a fraction of one.
            step = round((x1 - x0) / width)
            assert abs((x0 + step * w0) - x1) < 1e-6, (rects,)

    def test_the_separator_is_a_stroke_in_the_stylesheet(self):
        css = (CSS / "_07-histogram.css").read_text(encoding="utf-8")
        block = re.search(r"\.hist-svg \.bar \{(.*?)\}", css, re.S)
        assert block, "the bar rule is gone"
        assert "stroke: var(--paper)" in block.group(1)
        assert "stroke-width: 1" in block.group(1)


class TestAZeroCountDrawsNothing:
    """Rule 3. A 1px floor is right for a small non-zero value and wrong for
    zero -- ten empty months drawn as ten 1px bars assert data that is not
    there."""

    def test_the_empty_bin_emits_no_rect(self):
        # `bins=4` matches the four input bins, so the renderer's rebinning is
        # the identity and the empty bin stays exactly one empty bin. Asking
        # for 25 would spread four bins over twenty-five and prove nothing
        # about the one that is zero.
        chart = _chart(4)
        counts = [int(c) for c in re.findall(r'data-count="(\d+)"', chart)]
        assert counts, "no bars at all"
        assert 0 not in counts
        assert len(counts) == 3, counts

    def test_a_column_of_nothing_but_gaps_draws_nothing(self):
        renderer = SVGHistogramRenderer()
        chart = renderer.render_histogram_from_bins(
            [0.0, 1.0, 2.0, 3.0], [0, 0, 0], 3, "lin", "x", "c"
        )
        assert 'class="bar"' not in chart


class TestTheAxisMaxIsPerChart:
    """5d.8. Changing the bin count changes the peak, so a max shared across
    variants would draw the 50-bin chart half empty -- the same defect as the
    letterbox it replaced."""

    def test_a_finer_binning_gets_its_own_ceiling(self):
        rng = np.random.default_rng(0)
        edges = list(np.linspace(0, 100, 41))
        counts = list(rng.integers(5, 60, 40))
        renderer = SVGHistogramRenderer()

        def ceiling(bins: int) -> float:
            chart = renderer.render_histogram_from_bins(
                edges, counts, bins, "lin", "x", "c"
            )
            labels = re.findall(r'class="hist__y"[^>]*>([^<]+)<', chart)
            return float(labels[-1].replace(",", ""))

        coarse, fine = ceiling(10), ceiling(50)
        assert coarse > fine, (
            f"10 bins ceiling {coarse}, 50 bins ceiling {fine} -- a shared max "
            "would draw the fine chart half empty"
        )

    def test_the_tallest_bar_fills_most_of_the_plot(self):
        """The real content of 'reaches its top gridline'. `nice_ticks` rounds
        the ceiling up, so a bar never touches exactly; what matters is that it
        is not left at half height."""
        for bins in (10, 25, 50):
            chart = _chart(bins)
            heights = [float(h) for h in re.findall(r'height="([\d.]+)"', chart)]
            assert max(heights) >= 60.0, (bins, max(heights))


class TestTheGutterCanBeFixed:
    """It is fixed at 44px so the plot's left edge does not move between
    columns and bars line up down the page. That only works because a count
    label is guaranteed four glyphs (#183)."""

    def test_no_y_label_exceeds_four_glyphs(self, figures):
        labels = re.findall(r'class="hist__y"[^>]*>([^<]+)<', "".join(figures))
        assert labels
        too_long = [label for label in labels if len(label) > 4]
        assert not too_long, too_long

    def test_the_stylesheet_fixes_the_gutter(self):
        css = (CSS / "_07-histogram.css").read_text(encoding="utf-8")
        assert "--hist-gutter: 44px" in css
        assert "grid-template-columns: var(--hist-gutter)" in css


class TestTicksAreTieredByImportance:
    """5d.3. The renderer cannot know the viewport, so it writes every tick it
    would want and CSS drops tiers."""

    def test_nine_ticks_are_written(self, figures):
        ticks = re.findall(r'class="hist__tick" data-tier="(\d)"', figures[0])
        assert len(ticks) == 9, ticks

    def test_three_survive_the_narrowest_breakpoint(self, figures):
        """Tier 1 is the two ends **and the midpoint**. The first version made
        only the ends tier 1, which left a phone with a range and no middle --
        nothing to say whether the distribution is centred."""
        tiers = [int(t) for t in re.findall(r'data-tier="(\d)"', figures[0])]
        assert tiers.count(1) == 3, tiers
        assert tiers[0] == 1 and tiers[-1] == 1 and tiers[len(tiers) // 2] == 1

    def test_five_survive_the_middle_breakpoint(self, figures):
        tiers = [int(t) for t in re.findall(r'data-tier="(\d)"', figures[0])]
        assert sum(1 for t in tiers if t <= 2) == 5, tiers

    def test_the_stylesheet_drops_them_in_that_order(self):
        css = (CSS / "_07-histogram.css").read_text(encoding="utf-8")
        wide = re.search(r"@media \(max-width: 768px\) \{(.*?)\n\}", css, re.S)
        narrow = re.search(r"@media \(max-width: 480px\) \{(.*?)\n\}", css, re.S)
        assert wide and 'data-tier="3"' in wide.group(1)
        assert narrow and 'data-tier="2"' in narrow.group(1)


class TestTheCaption:
    """5d.5. The x unit used to sit at the right end of the axis, opposite
    `ROWS`; at 1,100px they are a hand-span apart and stop reading as a pair."""

    def test_it_carries_unit_bins_and_peak(self):
        caption = re.search(r'class="hist__caption"[^>]*>([^<]+)<', _chart()).group(1)
        assert "years" in caption
        assert "25 bins" in caption
        assert "peak" in caption

    def test_a_unitless_column_omits_the_unit_clause(self):
        renderer = SVGHistogramRenderer()
        chart = renderer.render_histogram_from_bins(
            [0.0, 1.0, 2.0], [3, 4], 2, "lin", "score", "c"
        )
        caption = re.search(r'class="hist__caption"[^>]*>([^<]+)<', chart).group(1)
        assert caption.startswith("2 bins"), caption

    def test_the_peak_is_the_exact_count(self):
        """The y labels abbreviate to four glyphs, so this is where the precise
        figure lives."""
        renderer = SVGHistogramRenderer()
        chart = renderer.render_histogram_from_bins(
            [0.0, 1.0, 2.0], [3, 123456], 2, "lin", "score", "c"
        )
        caption = re.search(r'class="hist__caption"[^>]*>([^<]+)<', chart).group(1)
        assert "123,456" in caption


class TestTheEmptyStateIsNotAFailedChart:
    def test_it_says_so_in_words(self):
        renderer = SVGHistogramRenderer()
        chart = renderer.render_histogram_from_bins([], [], 10, "lin", "x", "c")
        assert "No values to plot" in chart
        assert "<svg" not in chart


# --------------------------------------------------------------------------- #
# A bin count is a count of rows, and rows do not come in negative quantities
# --------------------------------------------------------------------------- #
class TestRebinningCannotInventANegativeCount:
    """#253. Every variant is re-binned from one set of 25 non-negative counts,
    so a negative can only be manufactured on the way.

    It was. Counts were rounded to nearest and the whole residual was then
    dumped into the single bin with the largest fractional part. On `Fare` at
    50 bins that residual was **-3** and the chosen bin held **2**, so the
    report shipped a bin of -1: a count that cannot exist, drawn as
    `height="-0.33"` -- which the browser rejects and logs -- and printed in
    that bar's tooltip.

    The negative was the visible half. Dumping a residual of any sign into one
    bin moves rows out of, or into, a single column of the chart; a bin holding
    5 could display 2 with nothing wrong on screen.
    """

    def test_the_largest_remainder_method_preserves_the_total(self):
        values = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5])
        out = _round_preserving_total(values, 5)
        assert out.sum() == 5
        assert (out >= 0).all()

    def test_it_never_goes_negative_when_rounding_overshoots(self):
        """Six halves round to nearest as six ones -- one over the true five.
        The old code took that one back out of a single bin."""
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        out = _round_preserving_total(values, 3)
        assert out.sum() == 3
        assert (out >= 0).all(), out

    def test_no_bin_moves_by_more_than_one(self):
        """The correction is spread, not dumped. Each bin ends within one of
        its own weight, which is the smallest change consistent with the
        total."""
        rng = np.random.default_rng(0)
        values = rng.random(50) * 20
        total = int(round(values.sum()))
        out = _round_preserving_total(values, total)
        assert out.sum() == total
        assert np.all(np.abs(out - values) < 1.0 + 1e-9), np.max(np.abs(out - values))

    def test_the_titanic_fare_case_that_produced_the_minus_one(self):
        """The exact input from the report: 891 rows over 50 bins."""
        frame = pd.read_csv(TITANIC)
        stats = profile(frame, seed=0).stats["columns"]["Fare"]
        edges = np.array(stats["true_histogram_edges"])
        counts = np.array(stats["true_histogram_counts"])
        assert (counts >= 0).all(), "the source counts were never the problem"

        new_edges = np.linspace(edges[0], edges[-1], 51)
        weights = np.zeros(50)
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            left, right = edges[i], edges[i + 1]
            for j in range(50):
                overlap = min(right, new_edges[j + 1]) - max(left, new_edges[j])
                if overlap > 0:
                    weights[j] += count * overlap / (right - left)
        weights *= counts.sum() / weights.sum()
        out = _round_preserving_total(weights, int(counts.sum()))
        assert out.sum() == 891
        assert out.min() >= 0, out.min()

    @pytest.mark.parametrize("column", ["Fare", "Age", "SibSp", "Parch"])
    def test_no_rendered_bar_carries_a_count_below_zero(self, column):
        """The check #253 asks for, over the real document: no `<rect>` in a
        generated report may carry a negative `data-count` or a negative
        `height`."""
        html = profile(pd.read_csv(TITANIC), seed=0).html
        card = re.search(
            rf'<article class="var-card" id="col_{column}".*?</article>', html, re.S
        )
        assert card, f"no card for {column}"
        assert not re.findall(r'data-count="(-\d+)"', card.group(0))
        assert not re.findall(r'<rect[^>]*height="-[\d.]+"', card.group(0))

    def test_every_linear_variant_still_accounts_for_every_row(self):
        """Preserving the total is the property the old code was reaching for
        and got at the cost of a negative. Both, or it is not a fix.

        Linear only: the log variants drop rows for a different reason, and a
        much larger one -- see #258.
        """
        html = profile(pd.read_csv(TITANIC), seed=0).html
        card = re.search(
            r'<article class="var-card" id="col_Fare".*?</article>', html, re.S
        ).group(0)
        # Split rather than match a nested block -- a non-greedy `(.*?)</div>`
        # ends at the first closing tag inside the variant, several elements
        # before its bars. Then bound each chunk at its own `</svg>`: the last
        # chunk otherwise runs to the end of the card and picks up the scale
        # buttons, which carry a `data-scale` of their own and made a log
        # variant read as a linear one.
        found = 0
        for chunk in card.split('<div class="hist variant')[1:]:
            svg = chunk.split("</svg>")[0]
            scale = re.search(r'data-scale="(\w+)"', svg)
            if not scale or scale.group(1) != "lin":
                continue
            bins = re.search(r'data-bin="(\d+)"', svg).group(1)
            counts = [int(c) for c in re.findall(r'data-count="(-?\d+)"', svg)]
            assert counts, f"{bins} bins drew nothing"
            assert sum(counts) == 891, f"{bins} bins sum to {sum(counts)}"
            found += 1
        assert found == 3, f"expected 3 linear variants, read {found}"
