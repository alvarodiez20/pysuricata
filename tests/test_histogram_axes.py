"""The histogram's axes, and the label it must not invent.

Phase 5.2. The chart printed the column name inside the plot and bare numbers
on both axes — so it spent a line on a word the card header had just said, and
still left nothing to say which axis was years and which was rows.

The unit derivation is the substance here, and **its absent branch is the point**.
A column called ``score`` has no unit. Labelling its axis ``SCORE`` would add a
word and no information while looking like a unit, which is worse than a bare
axis. Anything the name does not state outright gets no label.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.histogram_svg import SVGHistogramRenderer, derive_x_unit


@pytest.fixture(scope="module")
def html() -> str:
    rng = np.random.default_rng(0)
    n = 400
    return profile(
        pd.DataFrame(
            {
                "age": rng.integers(1, 90, n).astype(float),
                "score": rng.normal(0, 1, n),
                "elapsed_ms": rng.gamma(2, 50, n),
            }
        ),
        seed=0,
    ).html


def _card(html: str, name: str) -> str:
    for chunk in html.split('<article class="var-card"')[1:]:
        if f'data-name="{name}"' in chunk:
            return chunk
    raise AssertionError(f"no card for {name}")


def _units(card: str) -> set[str]:
    """The two axis captions, wherever they now live.

    They used to be `<text class="unit-label">` inside the SVG. #147 took every
    glyph out of the SVG, so `ROWS` is an HTML span in the gutter and the x
    unit moved into the caption line -- at 1,100px the two were a hand-span
    apart and had stopped reading as a pair.
    """
    units = set(re.findall(r'class="hist__unit"[^>]*>([^<]+)<', card))
    for caption in re.findall(r'class="hist__caption"[^>]*>([^<]+)<', card):
        # `years · 25 bins · peak 83 rows at 25.9-29.1`
        head = caption.split("\u00b7")[0].strip()
        if head and "bins" not in head:
            units.add(head.upper())
    return units


# --------------------------------------------------------------------------- #
# the unit, and its absence
# --------------------------------------------------------------------------- #
class TestTheUnitIsNeverInvented:
    @pytest.mark.parametrize(
        "name",
        ["score", "fare", "value", "temperature", "x", "measure", "amount", ""],
    )
    def test_a_name_that_states_no_unit_gets_none(self, name):
        """The branch the issue calls the important one."""
        assert derive_x_unit(name) is None

    @pytest.mark.parametrize(
        ("name", "unit"),
        [
            ("age", "YEARS"),
            ("age_years", "YEARS"),
            ("duration_seconds", "SECONDS"),
            ("elapsed_ms", "MS"),
            ("size_bytes", "BYTES"),
            ("pct_missing", "%"),
            ("n_rows", "COUNT"),
        ],
    )
    def test_a_name_that_states_one_gets_it(self, name, unit):
        assert derive_x_unit(name) == unit

    def test_the_last_word_wins_over_the_first(self):
        """`age_years` is years. Reading left to right would make every
        `age_*` column years regardless of what it measures."""
        assert derive_x_unit("age_years") == "YEARS"

    def test_it_survives_odd_names(self):
        for name in ("  ", "___", "123", "!!", None):
            assert derive_x_unit(name) is None

    def test_a_unitless_column_still_labels_its_y_axis(self, html):
        """Rows is always rows, whatever the column measures."""
        units = _units(_card(html, "score"))
        assert units == {"ROWS"}

    def test_a_column_with_a_unit_labels_both(self, html):
        assert _units(_card(html, "age")) == {"ROWS", "YEARS"}

    def test_the_unit_comes_from_the_column_not_the_chart(self, html):
        assert "MS" in _units(_card(html, "elapsed_ms"))


# --------------------------------------------------------------------------- #
# the chart itself
# --------------------------------------------------------------------------- #
class TestTheChart:
    def test_the_column_name_is_no_longer_printed_inside_it(self, html):
        """The card header carries it; the plot spent a line repeating it."""
        assert 'class="hist-title"' not in html

    def test_the_accessible_name_survives(self, html):
        """`<title>` is the accessible name, not a caption -- dropping the
        drawn title must not take it."""
        assert re.search(r"<title id=\"hist-title-[^\"]+\">Histogram for", html)

    def test_the_bars_encode_only_length(self, html):
        """A stroke, a fill-opacity or a corner radius each change a bar's
        apparent length, which is the one thing it encodes."""
        svg = re.search(r'<svg class="hist-svg".*?</svg>', html, re.S).group(0)
        assert "fill-opacity" not in svg
        assert not re.search(r"<rect[^>]*\sstroke=", svg)
        assert not re.search(r"<rect[^>]*\srx=", svg)

    def test_the_end_labels_sit_inside_the_plot(self, html):
        """Centred on their tick, the first ran under the y-axis labels and the
        last ran off the chart. They anchor to the plot edge instead.

        The mechanism changed with #147 -- `text-anchor` on an SVG `<text>`
        became `data-anchor` on an HTML span, since the labels are no longer in
        the SVG at all -- but the rule is the same one.
        """
        figure = re.search(r'<figure class="hist".*?</figure>', html, re.S).group(0)
        anchors = re.findall(r'data-anchor="(\w+)"', figure)
        assert {"start", "end"} <= set(anchors), anchors

    def test_it_carries_no_legacy_hue(self):
        source = (
            __import__("pathlib").Path("pysuricata/render/histogram_svg.py").read_text()
        )
        for legacy in ("4ea3f1", "8ac926", "ffca3a", "ff595e"):
            assert legacy not in source.lower()

    def test_the_figures_are_monospace(self):
        source = (
            __import__("pathlib").Path("pysuricata/render/histogram_svg.py").read_text()
        )
        assert "var(--font-mono)" in source
        assert "Arial" not in source


class TestEdgeCases:
    def test_a_single_valued_range_does_not_divide_by_zero(self):
        renderer = SVGHistogramRenderer()
        out = renderer.render_histogram_from_bins(
            bin_edges=[2.0, 2.0],
            bin_counts=[5],
            bins=1,
            scale="lin",
            title="c",
            col_id="c",
        )
        assert "<svg" in out

    def test_a_range_spanning_zero_renders(self):
        rng = np.random.default_rng(0)
        out = profile(pd.DataFrame({"v": rng.normal(0, 5, 300)}), seed=0).html
        assert 'class="hist-svg"' in out

    def test_a_negative_only_range_renders(self):
        rng = np.random.default_rng(0)
        out = profile(pd.DataFrame({"v": -rng.gamma(2, 3, 300)}), seed=0).html
        assert 'class="hist-svg"' in out

    def test_very_large_magnitudes_stay_compact(self):
        rng = np.random.default_rng(0)
        out = profile(pd.DataFrame({"v": rng.normal(0, 1, 300) * 1e15}), seed=0).html
        svg = re.search(r'<svg class="hist-svg".*?</svg>', out, re.S).group(0)
        labels = re.findall(r'class="tick-label"[^>]*>([^<]+)<', svg)
        assert all(len(label) <= 12 for label in labels), labels

    def test_an_empty_column_renders_a_chart_or_none_but_does_not_raise(self):
        out = profile(pd.DataFrame({"v": [np.nan] * 200}), seed=0).html
        assert "<html" in out
