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
from pysuricata.render.histogram_svg import SVGHistogramRenderer

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
    def test_marks_are_marked_non_scaling(self, report, element):
        pattern = rf"<{element}[^>]*class=\"(?:bar|grid|axis)\"[^>]*>"
        found = re.findall(pattern, report) + re.findall(
            rf'<{element} class="(?:bar|grid|axis)"[^>]*>', report
        )
        assert found, f"no {element} marks found"
        for mark in found:
            assert 'vector-effect="non-scaling-stroke"' in mark, mark


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
