"""The charts have to be *looked at*, and these are what the looking found.

Four defects, all invisible to the existing suite because the fingerprint
deliberately discards presentation and every other check reads values rather
than geometry. They were found by rendering a report in Chromium and measuring
the boxes.

1. **Every histogram variant was displayed at once.** A vestigial
   `display: block` on `.hist.variant` outranked the rule that hides the
   inactive ones, so a numeric card drew all six bins/scale combinations
   stacked -- 1,671px instead of ~570px -- and the toggles appeared to do
   nothing because every option was already on screen.
2. **The extreme y labels were nudged the wrong way.** The CSS keyed the
   correction on `:first-of-type`/`:last-of-type`, but ticks are emitted
   ascending, so the first span is the bottom. The top label overhung the plot
   and was clipped; the `0` hung down into the tick row.
3. **The caption was drawn over the x tick labels.** `.hist__area` was pinned
   to `--hist-height`, so the tick row inside it overflowed without
   contributing layout height and the caption below landed on top of it.
4. **Categorical bar thickness was inversely proportional to cardinality.** A
   fixed height divided among the bars gave two levels two 218px slabs and five
   levels 87px each, and the 420-unit viewBox stretched to a ~1,100px column
   rendered every 11px label at ~30px.

These tests read the markup and the stylesheet rather than a browser, so they
run everywhere; the browser is what found the defects, not what guards them.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

CSS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "pysuricata", "static", "css"
)


def _css() -> str:
    parts = []
    for path in sorted(glob.glob(os.path.join(CSS_DIR, "_*.css"))):
        with open(path, encoding="utf-8") as handle:
            parts.append(handle.read())
    return "".join(parts)


def _css_rules() -> list[tuple[str, str]]:
    """(selector, body) pairs, with comments removed first.

    The comments in this stylesheet quote the selectors they are about -- the
    note explaining this very bug contains `.hist-variants .variant { display:
    none }` as prose -- so a parser that keeps them reports the documentation
    as a rule.
    """
    css = re.sub(r"/\*.*?\*/", "", _css(), flags=re.S)
    return [
        (" ".join(m.group(1).split()), m.group(2))
        for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css)
    ]


def _css_body(selector: str) -> str:
    """Every declaration this selector carries, across all of its blocks.

    A selector may legitimately appear more than once -- `.dt-svg` has a sizing
    block and a separate `overflow` one -- so keying a dict on the selector
    silently keeps whichever came last and reports the other's declarations as
    absent.
    """
    return "".join(body for sel, body in _css_rules() if sel == selector)


def _strip_assets(html: str) -> str:
    """The report inlines its own CSS and JS, so a search over the whole
    document finds a class name in the very source that references it."""
    return re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)


@pytest.fixture(scope="module")
def report() -> str:
    """One frame reaching numeric and several categorical shapes."""
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            "num": rng.gamma(2, 20, n),
            "two": rng.choice(["male", "female"], n),
            "five": rng.choice(list("abcde"), n),
            "many": rng.choice([f"cat_{i}" for i in range(40)], n),
        }
    )
    return profile(frame, seed=0).html


# --------------------------------------------------------------------------- #
# 1. one variant at a time
# --------------------------------------------------------------------------- #
class TestOnlyTheActiveVariantIsShown:
    """The bug was pure specificity, so the guard is too: any rule that sets
    `display` on a variant and outranks the hide rule brings it back."""

    #: Selectors allowed to set `display` on a chart variant, and nothing else.
    _ALLOWED = (
        "#pysuricata-report .hist-variants .variant",
        "#pysuricata-report .hist-variants .variant.active",
    )

    def test_no_rule_forces_a_histogram_variant_visible(self):
        offenders = []
        for selector, body in _css_rules():
            if "display" not in body:
                continue
            # The *subject* of the selector is its last compound. A rule for
            # `.cat.variant > svg` styles the svg, not the variant, and is not
            # in this fight.
            subject = re.split(r"[\s>+~]+", selector)[-1]
            if ".variant" not in subject:
                continue
            if selector in self._ALLOWED:
                continue
            offenders.append(f"{selector} {{{' '.join(body.split())}}}")

        assert not offenders, (
            "these rules set `display` on a chart variant and will fight the "
            f"show/hide pair, whichever wins on specificity: {offenders}"
        )

    def test_exactly_one_histogram_variant_is_marked_active(self, report):
        """Per card. Splitting on `<article` rather than matching nested divs:
        the variants live several levels down and a non-greedy `</div>` stops
        at the first inner close, which silently scoped this to a fragment."""
        body = _strip_assets(report)

        cards = [c for c in body.split("<article") if 'class="hist variant' in c]
        assert cards, "no card carried histogram variants"

        for card in cards:
            total = len(re.findall(r'class="hist variant', card))
            active = len(re.findall(r'class="hist variant active"', card))
            assert active == 1, f"{active} of {total} variants marked active"

    def test_a_numeric_card_offers_several_variants(self, report):
        """A guard on the guard: if the card stopped emitting variants the
        assertions above would pass while proving nothing."""
        body = _strip_assets(report)
        assert len(re.findall(r'class="hist variant', body)) >= 6


# --------------------------------------------------------------------------- #
# 2. the extreme y labels
# --------------------------------------------------------------------------- #
class TestTheAxisLabelsAreNudgedAtTheRightEnds:
    """`:first-of-type` and `:last-of-type` read the DOM, and the DOM is
    ascending -- so they addressed the opposite ends from the ones intended."""

    def test_the_first_y_label_emitted_is_the_bottom_of_the_axis(self, report):
        body = _strip_assets(report)
        labels = re.findall(r'<span class="hist__y"([^>]*)>', body)
        assert labels, "no y labels rendered"

        first = labels[0]
        assert 'data-edge="bottom"' in first, (
            "the first y label is no longer the bottom of the axis. The CSS "
            "nudges are keyed on `data-edge` precisely so this order does not "
            "matter -- but if it changed, check the edges are still tagged."
        )

    def test_both_extremes_are_tagged(self, report):
        body = _strip_assets(report)
        block = re.search(r'<div class="hist__gutter">.*?</div>', body, re.S)
        assert block, "no gutter rendered"

        assert 'data-edge="top"' in block.group(0)
        assert 'data-edge="bottom"' in block.group(0)

    def test_only_the_extremes_are_tagged(self, report):
        """A middle label nudged inward would sit off its own gridline."""
        body = _strip_assets(report)
        block = re.search(r'<div class="hist__gutter">.*?</div>', body, re.S).group(0)

        assert block.count("data-edge=") == 2

    def test_the_stylesheet_keys_the_nudge_on_the_edge_not_the_order(self):
        css = _css()

        assert '.hist__y[data-edge="top"]' in css
        assert '.hist__y[data-edge="bottom"]' in css
        assert ".hist__y:first-of-type" not in css, (
            "DOM-order selectors are what inverted the nudges; the ticks are "
            "emitted ascending, so `:first-of-type` is the bottom label."
        )
        assert ".hist__y:last-of-type" not in css


# --------------------------------------------------------------------------- #
# 3. the caption clears the axis
# --------------------------------------------------------------------------- #
class TestTheTickRowContributesItsHeight:
    def test_the_plot_area_is_not_pinned_to_the_chart_height(self):
        """`.hist__area` holds the x-tick row. Pinning it to `--hist-height`
        made that row overflow without taking up any layout height, and the
        caption below was then drawn straight over the tick labels."""
        css = _css()
        block = re.search(r"#pysuricata-report \.hist__area \{([^}]*)\}", css)
        assert block, "`.hist__area` rule missing"

        assert "height" not in block.group(1), (
            "`.hist__area` has a height again; the tick row inside it will "
            "overflow and the caption will overlap the axis labels."
        )

    def test_the_plot_itself_carries_the_height(self):
        css = _css()
        block = re.search(r"#pysuricata-report \.hist-svg \{([^}]*)\}", css)
        assert block and "var(--hist-height)" in block.group(1), (
            "the height has to live on the plot, or the chart collapses"
        )


# --------------------------------------------------------------------------- #
# 4. categorical bars
# --------------------------------------------------------------------------- #
def _cat_svgs(html: str) -> list[str]:
    return re.findall(r"<svg class=\"cat-svg\".*?</svg>", _strip_assets(html), re.S)


def _bar_heights(svg: str) -> list[float]:
    return [
        float(h) for h in re.findall(r'<rect class="bar"[^>]*height="([\d.]+)"', svg)
    ]


class TestABarHasAHeightAndTheChartFollows:
    """Thickness was `(fixed_height - gaps) / n`, so the same chart read
    differently at every cardinality."""

    def test_every_bar_is_the_same_height_within_a_chart(self, report):
        for svg in _cat_svgs(report):
            heights = set(_bar_heights(svg))
            assert len(heights) <= 1, (
                f"bars of differing height in one chart: {heights}"
            )

    def test_bar_height_does_not_depend_on_how_many_levels_there_are(self, report):
        charts = [(len(_bar_heights(s)), _bar_heights(s)) for s in _cat_svgs(report)]
        charts = [(n, hs) for n, hs in charts if n]
        assert len({n for n, _ in charts}) > 1, (
            "fixture no longer produces charts with differing level counts, so "
            "this proves nothing"
        )

        thicknesses = {hs[0] for _, hs in charts}
        assert len(thicknesses) == 1, (
            f"bar thickness varies with level count: {sorted(thicknesses)}"
        )

    def test_the_svg_height_grows_with_the_number_of_bars(self, report):
        seen = {}
        for svg in _cat_svgs(report):
            n = len(_bar_heights(svg))
            height = float(
                re.search(r'<svg class="cat-svg"[^>]*height="([\d.]+)"', svg).group(1)
            )
            if n:
                seen[n] = height

        assert len(seen) > 1
        ordered = sorted(seen.items())
        assert all(b[1] > a[1] for a, b in zip(ordered, ordered[1:], strict=False)), (
            f"height did not grow with bar count: {ordered}"
        )


class TestTheValueTextSaysWhichBackgroundItIsOn:
    """`--muted` on the bar fill measured 1.20:1, against AA's 4.5:1 for text
    this size. The renderer is the only thing that knows whether the value
    landed inside the bar or past its end."""

    def test_a_value_inside_a_bar_is_tagged(self, report):
        svgs = _cat_svgs(report)
        assert svgs, "no categorical charts rendered"

        inside = sum(s.count("bar-value is-inside") for s in svgs)
        assert inside > 0, (
            "no value was placed inside a bar, so the contrast pairing is "
            "untested -- the fixture needs a wide bar"
        )

    def test_every_value_declares_a_placement(self, report):
        for svg in _cat_svgs(report):
            values = re.findall(r'class="bar-value([^"]*)"', svg)
            for suffix in values:
                assert "is-inside" in suffix or "is-outside" in suffix, (
                    f"bar-value with no placement class: {suffix!r}"
                )

    def test_the_stylesheet_colours_the_inside_case_separately(self):
        css = _css()
        assert ".cat-svg .bar-value.is-inside" in css
        block = re.search(
            r"#pysuricata-report \.cat-svg \.bar-value\.is-inside \{([^}]*)\}", css
        )
        assert block and "--paper" in block.group(1), (
            "text on the bar must take the paper ink, not the muted grey that "
            "is chosen against the page background"
        )


# --------------------------------------------------------------------------- #
# 5. the datetime timeline is drawn at the size it is displayed at (#217)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dt_report() -> str:
    """Two datetime columns reaching both x-label formats.

    `booked` spans months, so the labels are `%Y-%m-%d`; `tight` spans under
    three days, which crosses `short_span_ns` and produces the longer
    `%Y-%m-%d %H:%M`. The wide-label case is the one that clips.
    """
    n = 800
    frame = pd.DataFrame(
        {
            "booked": pd.date_range("2020-01-01", periods=n, freq="h"),
            "tight": pd.date_range("2021-06-01", periods=n, freq="s"),
        }
    )
    return profile(frame, seed=0).html


def _dt_svgs(html: str) -> list[str]:
    return re.findall(
        r"<svg class=\"dt-svg\".*?</svg>", _strip_assets(html), flags=re.S
    )


class TestTheTimelineIsAuthoredAtTheSizeItIsDrawn:
    """Measured in Chromium at `0f7c2c7`: a `0 0 420 180` viewBox rendered
    1146x491, a scale of 2.73, so the stylesheet's 11px tick labels came out at
    ~30px and the card stood 844px tall to hold a flat line.

    Same defect the categorical chart had, and the same fix: author the chart at
    roughly the width of the column it lands in. After: 1146x208, scale 1.04,
    labels at 11.5px, card 561px.
    """

    def test_the_viewbox_is_not_a_fraction_of_the_column(self, dt_report):
        svgs = _dt_svgs(dt_report)
        assert svgs, "no datetime charts rendered"
        for svg in svgs:
            width = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
            assert width >= 1000, (
                f"the timeline is authored {width:.0f} units wide and is drawn "
                "into a ~1,150px column, so everything inside it is scaled up "
                f"by {1146 / width:.2f}x"
            )

    def test_the_svg_carries_its_own_size(self, dt_report):
        """`width="100%" height="100%"` makes the box whatever the column is and
        stretches the drawing onto it, which is the inflation mechanism."""
        for svg in _dt_svgs(dt_report):
            head = svg[: svg.index(">")]
            vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
            assert re.search(rf'width="{vb.group(1)}"', head), (
                f"intrinsic width does not match the viewBox: {head}"
            )
            assert re.search(rf'height="{vb.group(2)}"', head), (
                f"intrinsic height does not match the viewBox: {head}"
            )

    def test_the_aspect_ratio_is_not_thrown_away(self, dt_report):
        """`preserveAspectRatio="none"` lets the x and y scales diverge, so the
        drawing is not merely inflated but distorted."""
        for svg in _dt_svgs(dt_report):
            assert 'preserveAspectRatio="none"' not in svg[: svg.index(">")]

    def test_the_stylesheet_scales_the_width_and_lets_height_follow(self):
        body = _css_body("#pysuricata-report .dt-svg")
        assert "height: auto" in body, (
            "a fixed height with a scaled width is what distorts the drawing"
        )

    def test_the_chart_is_never_scaled_below_one_to_one(self):
        """The fix cuts both ways. Fitting a 1,100-unit chart into a 694px
        column renders an 11px label at 6.9px, which is no more readable than
        30px was. The floor pins a unit to a pixel; the wrapper scrolls."""
        body = _css_body("#pysuricata-report .dt-svg")
        floor = re.search(r"min-width:\s*(\d+)px", body)
        assert floor, ".dt-svg has no min-width, so it shrinks with the column"

        width = int(floor.group(1))
        wrapper = _css_body("#pysuricata-report .timeline-chart")
        assert "overflow-x: auto" in wrapper, (
            f"the chart is floored at {width}px but its wrapper does not "
            "scroll, so a narrow column overflows the page instead"
        )


class TestTheGutterFollowsTheLabelsRatherThanTheViewBox:
    """A constant gutter and a viewBox width are not independent -- 45 units is
    11% of a 420-unit chart and 4% of a 1,100-unit one -- so the margins have to
    come from the text they hold, or widening the viewBox moves them."""

    def _margins(self, y_labels):
        from pysuricata.render.datetime_card import DateTimeCardRenderer

        return DateTimeCardRenderer()._timeline_margins(1100, 200, y_labels)

    def test_a_wider_axis_label_gets_a_wider_gutter(self):
        narrow = self._margins(["0", "5"])[0]
        wide = self._margins(["0", "250000"])[0]
        assert wide > narrow, (
            f"a six-digit axis got the same {wide}-unit gutter as a one-digit "
            "one, so the margin is not derived from its contents"
        )

    def test_the_gutter_holds_the_widest_label(self):
        from pysuricata.render.card_config import DEFAULT_DT_CONFIG as cfg

        for labels in (["0", "5"], ["0", "1200"], ["0", "250000"]):
            gutter = self._margins(labels)[0]
            needed = max(len(v) for v in labels) * cfg.char_width
            assert gutter >= min(needed, cfg.max_gutter), (
                f"{labels} needs ~{needed} units of ink and the gutter is "
                f"{gutter}; the labels run into the plot"
            )

    def test_the_gutter_is_bounded(self):
        from pysuricata.render.card_config import DEFAULT_DT_CONFIG as cfg

        assert self._margins(["0"])[0] >= cfg.min_gutter
        assert self._margins(["1" * 50])[0] <= cfg.max_gutter, (
            "an unbounded gutter lets a pathological count eat the plot"
        )

    def test_a_degenerate_width_still_leaves_a_plot(self):
        from pysuricata.render.datetime_card import DateTimeCardRenderer

        left, right, _, _ = DateTimeCardRenderer()._timeline_margins(
            60, 200, ["123456"]
        )
        assert left + right < 60, "margins consumed the whole chart"


class TestTheEndLabelsAreAnchoredInward:
    """A centred date sits half its own width past the end of the axis: the
    first ran off the left edge into the y gutter, and the last would need a
    right margin the size of half a timestamp."""

    def _x_labels(self, svg):
        return re.findall(
            r'<text class="tick-label x-tick-label"[^>]*text-anchor="(\w+)"[^>]*'
            r'data-edge="([^"]*)"[^>]*>([^<]*)</text>',
            svg,
        )

    def test_the_first_and_last_are_anchored_and_the_rest_centred(self, dt_report):
        svgs = _dt_svgs(dt_report)
        assert svgs, "no datetime charts rendered"
        for svg in svgs:
            labels = self._x_labels(svg)
            assert len(labels) >= 3, f"only {len(labels)} x labels to check"
            assert labels[0][0] == "start" and labels[0][1] == "first"
            assert labels[-1][0] == "end" and labels[-1][1] == "last"
            for anchor, edge, _ in labels[1:-1]:
                assert anchor == "middle" and edge == ""

    def test_the_stylesheet_does_not_rotate_them(self):
        """The rotation tier never worked: `transform-origin: center` on an SVG
        child resolves against the viewport element, not the label's own box, so
        every label was rotated about the middle of the chart and thrown
        275-400px below it -- measured at a 560px viewport, labels landing at
        y=2240 against an SVG bottom of 1966.

        It is also unnecessary now. Five labels across ~1,050 units do not
        collide: 0 collisions and 0 clipped labels down to a 360px viewport.
        """
        for selector, body in _css_rules():
            if ".dt-svg .x-tick-label" in selector:
                assert "rotate" not in body, (
                    f"{selector} rotates the date labels about the chart's "
                    f"centre rather than their own: {body.strip()}"
                )
