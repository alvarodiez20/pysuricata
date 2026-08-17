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
