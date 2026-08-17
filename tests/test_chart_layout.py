"""The charts have to be *looked at*, and these are what the looking found.

Five defects, all invisible to the existing suite because the fingerprint
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
5. **The datetime timeline drew its labels inside a stretched SVG** (#217), so
   an 11px date came out at ~37px in a wide column and would have come out at
   5px in a narrow one. No viewBox is right at both widths, which is why the
   last class here is a rule about structure -- text does not belong inside a
   `preserveAspectRatio="none"` SVG at all -- rather than a measurement.

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


# --------------------------------------------------------------------------- #
# 5. nothing readable inside a stretched SVG
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report_with_dates() -> str:
    """The datetime card needs its own frame; the fixture above has no dates."""
    rng = np.random.default_rng(0)
    n = 400
    return profile(
        pd.DataFrame(
            {
                "num": rng.gamma(2, 20, n),
                "when": pd.date_range("2024-01-01", periods=n, freq="7h"),
            }
        ),
        seed=0,
    ).html


class TestNoTextInsideAStretchedSvg:
    """#217, and the rule that generalises it.

    `preserveAspectRatio="none"` maps the viewBox onto whatever box CSS hands
    the element, independently on each axis. Nothing inside such an SVG has a
    size of its own: the datetime timeline authored its labels at 11px in a
    420-unit box that was painted 1,146px wide, and they came out at ~37px —
    three times the stat row beside them — while the same chart in a 470px
    column would have rendered them at 5px.

    **There is no viewBox that is correct at both widths.** That is why the fix
    is to take the text out, not to pick a better number, and why this test is
    a rule about the structure rather than a measurement of one chart.

    A uniformly scaled SVG (the default `xMidYMid meet`) is not in scope here:
    it keeps proportions, and the categorical chart is authored near its real
    width so its scale is ~1.
    """

    _SVG = re.compile(r"<svg\b[^>]*>.*?</svg>", re.S | re.I)

    def _stretched_svgs(self, html: str) -> list[str]:
        return [
            svg
            for svg in self._SVG.findall(_strip_assets(html))
            if 'preserveAspectRatio="none"' in svg
        ]

    def test_the_report_has_stretched_svgs_at_all(self, report_with_dates):
        """A guard on the guard: if the marks stopped being stretched, the
        assertion below would pass by matching nothing."""
        assert self._stretched_svgs(report_with_dates)

    def test_none_of_them_contains_a_text_element(self, report_with_dates):
        offenders = []
        for svg in self._stretched_svgs(report_with_dates):
            # `<title>` and `<desc>` are accessibility metadata, never painted.
            painted = re.findall(r"<text\b[^>]*>(.*?)</text>", svg, re.S)
            if painted:
                offenders.append(painted[:3])

        assert not offenders, (
            "text inside a non-uniformly stretched SVG has no size of its own; "
            f"it scales with the column. Move it to HTML: {offenders}"
        )


class TestTheTimelineIsBuiltLikeTheHistogram:
    """Reusing `figure.hist` rather than styling a second chart: the gutter,
    the tiered labels, the caption and the axis nudges already exist, are
    already tested, and now cannot drift apart from the histogram's."""

    def _figure(self, html: str) -> str:
        match = re.search(
            r'<figure class="hist dt-figure">.*?</figure>', _strip_assets(html), re.S
        )
        assert match, "the datetime card no longer renders a hist figure"
        return match.group(0)

    def test_the_labels_are_html_not_svg(self, report_with_dates):
        figure = self._figure(report_with_dates)

        assert '<span class="hist__y"' in figure, "count labels are not HTML"
        assert '<span class="hist__tick"' in figure, "date labels are not HTML"

    def test_the_extremes_of_the_count_axis_are_tagged(self, report_with_dates):
        gutter = re.search(
            r'<div class="hist__gutter">.*?</div>',
            self._figure(report_with_dates),
            re.S,
        )
        assert gutter

        assert 'data-edge="top"' in gutter.group(0)
        assert 'data-edge="bottom"' in gutter.group(0)
        assert gutter.group(0).count("data-edge=") == 2

    def test_the_date_labels_are_tiered_so_narrow_widths_thin_them(
        self, report_with_dates
    ):
        """Dates are ~10 glyphs where the histogram's numbers are ~6, so they
        collide sooner and have to be droppable."""
        row = re.search(
            r'<div class="hist__x">.*?</div>', self._figure(report_with_dates), re.S
        )
        assert row

        tiers = re.findall(r'data-tier="(\d)"', row.group(0))
        assert tiers, "no tick tiers emitted"
        assert "1" in tiers, "some labels must survive every breakpoint"
        assert len(set(tiers)) > 1, "all one tier means nothing ever thins"

    def test_the_ends_anchor_inside_the_plot(self, report_with_dates):
        row = re.search(
            r'<div class="hist__x">.*?</div>', self._figure(report_with_dates), re.S
        ).group(0)

        assert 'data-anchor="start"' in row
        assert 'data-anchor="end"' in row

    def test_the_caption_carries_the_exact_peak(self, report_with_dates):
        """The count labels round, so the caption is where the real figure is."""
        figure = self._figure(report_with_dates)

        caption = re.search(
            r'<figcaption class="hist__caption">(.*?)</figcaption>', figure, re.S
        )
        assert caption and "peak" in caption.group(1)

    def test_the_column_name_is_not_repeated_inside_the_chart(self, report_with_dates):
        """The card header already names the column; the SVG used to draw it
        again as a title."""
        figure = self._figure(report_with_dates)

        assert "hist-title" not in figure


# --------------------------------------------------------------------------- #
# 7. the temporal small multiples, rebuilt the same way
# --------------------------------------------------------------------------- #
class TestNoLabelIsInsideAChartTheStylesheetStretches:
    """The generalisation of the rule above, and the reason it was needed.

    The `preserveAspectRatio="none"` guard catches a chart whose *markup* says
    it may be distorted. It does not catch one that is stretched by CSS while
    keeping its proportions — and that is exactly what the hour/day/month
    charts were doing. They carried `width="400" height="160"` matching their
    viewBox, so every attribute-level check passed, and then `width: 100%` in
    the stylesheet scaled the whole thing to fill a grid cell.

    Measured in Chromium before the fix: the same 11px label rendered between
    **5.6px and 14.9px** across viewport widths, and not even monotonically —
    the grid drops from two columns to one, so a 600px viewport gave a *larger*
    label than an 820px one.

    So the rule is about neither attribute: an SVG that the stylesheet sizes to
    its container has no intrinsic size, and nothing with a font-size belongs
    inside it.
    """

    _SVG = re.compile(r"<svg\b[^>]*>.*?</svg>", re.S | re.I)

    def _classes_of_svgs_containing_text(self, html: str) -> set[str]:
        out: set[str] = set()
        for svg in self._SVG.findall(_strip_assets(html)):
            if not re.search(r"<text\b", svg):
                continue
            match = re.search(r'<svg\b[^>]*\bclass="([^"]*)"', svg)
            if match:
                out.update(match.group(1).split())
        return out

    def test_no_class_holding_text_is_given_a_container_width(self, report_with_dates):
        stretched = set()
        for selector, body in _css_rules():
            if not re.search(r"width:\s*100%", body):
                continue
            subject = re.split(r"[\s>+~]+", selector)[-1]
            for cls in re.findall(r"\.([\w-]+)", subject):
                stretched.add(cls)

        offenders = sorted(
            self._classes_of_svgs_containing_text(report_with_dates) & stretched
        )
        assert not offenders, (
            "these SVG classes contain painted text and are stretched to their "
            f"container by the stylesheet, so the text has no fixed size: "
            f"{offenders}"
        )


class TestTheTemporalChartsAreBuiltLikeTheHistogram:
    """Same treatment as the timeline in #219, for the same reason."""

    def _figures(self, html: str) -> list[str]:
        return re.findall(
            r'<figure class="hist temporal-figure">.*?</figure>',
            _strip_assets(html),
            re.S,
        )

    def test_the_card_renders_temporal_figures(self, report_with_dates):
        """A guard on the guard: every assertion below passes vacuously if the
        charts stop being emitted."""
        assert len(self._figures(report_with_dates)) >= 2

    def test_their_labels_are_html(self, report_with_dates):
        for figure in self._figures(report_with_dates):
            assert '<span class="hist__y"' in figure, "count labels are not HTML"
            assert '<span class="hist__tick"' in figure, "bucket labels are not HTML"

    def test_their_svgs_hold_no_text(self, report_with_dates):
        for figure in self._figures(report_with_dates):
            svg = re.search(r"<svg\b.*?</svg>", figure, re.S)
            assert svg, "no marks svg"
            assert not re.search(r"<text\b", svg.group(0))

    def test_the_bars_keep_the_data_the_tooltip_reads(self, report_with_dates):
        """`functionality.js` binds on `.temporal-chart .temporal-bar` and
        reads these attributes; the rebuild has to keep both the hooks."""
        for figure in self._figures(report_with_dates):
            assert 'class="bar temporal-bar"' in figure
            assert "data-count=" in figure and "data-label=" in figure

    def test_no_bar_carries_a_corner_radius(self, report_with_dates):
        """`rx` is in user units, so a stretched box rounds the horizontal and
        vertical corners by different amounts and the bars come out lopsided."""
        for figure in self._figures(report_with_dates):
            assert " rx=" not in figure


class TestTheBucketLabelsThinWithoutColliding:
    """Constant-size labels are the point of the rebuild, and they are also why
    thinning became necessary: labels that no longer shrink with the box will
    collide in it instead. Measured before the tiers: 7 overlapping labels at a
    360px viewport."""

    def _tick_rows(self, html: str) -> list[list[tuple[str, str]]]:
        rows = []
        for figure in re.findall(
            r'<figure class="hist temporal-figure">.*?</figure>',
            _strip_assets(html),
            re.S,
        ):
            row = re.search(r'<div class="hist__x">.*?</div>', figure, re.S)
            if row:
                rows.append(
                    re.findall(r'data-ttier="(\d)"[^>]*>([^<]*)</span>', row.group(0))
                )
        return [r for r in rows if r]

    def test_the_last_label_survives_every_thinning(self, report_with_dates):
        for row in self._tick_rows(report_with_dates):
            assert row[-1][0] == "1", (
                f"the axis loses its right endpoint when it thins: {row[-1]}"
            )

    def test_the_label_before_the_last_goes_first(self, report_with_dates):
        """Promoting the final label without demoting its neighbour is what put
        `18:00` on top of `21:00` and `Nov` on top of `Dec`: the two then
        survived every thinning together, side by side."""
        for row in self._tick_rows(report_with_dates):
            if len(row) < 2:
                continue
            assert row[-2][0] == "3", (
                f"the label beside the last one is tier {row[-2][0]}, so the "
                f"two collide at every width: {row[-2:]}"
            )

    def test_thinning_leaves_an_evenly_spaced_set(self, report_with_dates):
        """Tier 1 alone must not be two labels bunched at one end."""
        for row in self._tick_rows(report_with_dates):
            keep = [i for i, (tier, _) in enumerate(row) if tier == "1"]
            assert len(keep) >= 2, f"nothing left after thinning: {row}"
            gaps = [b - a for a, b in zip(keep, keep[1:], strict=False)]
            assert max(gaps) - min(gaps) <= 2, (
                f"tier-1 labels are unevenly spaced at {keep} in {row}"
            )

    def test_the_thinning_is_keyed_on_the_chart_not_the_viewport(self):
        """A media query reads the window; these charts are small multiples in
        a two-column grid, so at a 1,024px viewport each one is only 374px wide
        and a viewport rule calls that roomy while the labels already collide.
        """
        css = _css()
        assert "container-type: inline-size" in css, (
            "the chart item is not a container, so the queries below cannot "
            "resolve against it"
        )
        assert re.search(r"@container\s+temporal\s*\(max-width", css), (
            "bucket labels are not thinned by a container query"
        )

    def test_the_temporal_tiers_are_not_the_histograms(self):
        """`data-ttier`, deliberately. The histogram thins `data-tier` on
        viewport media queries, which are wrong here — at a 700px viewport this
        chart is 544px and perfectly roomy, and the inherited rule dropped half
        its labels anyway."""
        for selector, body in _css_rules():
            if "@container" in selector or "display" not in body:
                continue
            if "data-ttier" in selector:
                continue
            assert not re.search(r"\.temporal-figure[^{]*data-tier=", selector), (
                f"a viewport rule is thinning the temporal ticks: {selector}"
            )
