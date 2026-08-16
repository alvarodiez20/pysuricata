"""The boolean split bar, and the fixed axes of the temporal charts.

Phases 5.5 and 5.6. Most of what #117 asks for was already true by the time it
was reached — the palette work made the boolean bar two steps of one hue, and
`temporal_charts.py` already allocated a fixed slot per hour, day and month.
Verifying that rather than assuming it is the point of the first two classes
here; the third covers what was actually wrong.

The fixed axis is the subtle one. Two populated months drawn as two half-width
slabs reads as *spread evenly across the timeline* instead of *2 of 12*, so the
chart must allocate all twelve slots and simply leave ten of them empty.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

TEMPORAL = Path(__file__).resolve().parents[1] / "pysuricata/render/temporal_charts.py"


@pytest.fixture(scope="module")
def html() -> str:
    rng = np.random.default_rng(0)
    n = 600
    return profile(
        pd.DataFrame(
            {
                "survived": rng.integers(0, 2, n).astype(bool),
                # ~175 days, so only six months are populated -- the case the
                # fixed axis exists for.
                "when": pd.date_range("2026-01-01", periods=n, freq="7h"),
            }
        ),
        seed=0,
    ).html


def _card(html: str, kind: str) -> str:
    for chunk in html.split('<article class="var-card"')[1:]:
        if f'data-type="{kind}"' in chunk:
            return chunk
    raise AssertionError(f"no {kind} card")


def _bool_svg(html: str) -> str:
    return re.search(r"<svg[^>]*bool-svg.*?</svg>", html, re.S).group(0)


# --------------------------------------------------------------------------- #
# the boolean bar
# --------------------------------------------------------------------------- #
class TestSurvivedIsNotRedAndGreen:
    """Colouring `false` rust and `true` olive reads as bad-versus-good — the
    report passing judgement on someone's data."""

    def test_the_segments_are_two_steps_of_one_hue(self, html):
        svg = _bool_svg(html)
        fills = set(re.findall(r'<rect[^>]*fill="var\((--[\w-]+)', svg))
        assert fills <= {"--data-2", "--data-4", "--track"}, fills

    def test_no_quality_colour_touches_it(self, html):
        svg = _bool_svg(html)
        for quality in ("--q-good", "--q-bad", "--q-warn-fill", "--q-warn-text"):
            assert quality not in svg, quality

    def test_the_labels_are_legible_on_what_is_under_them(self, html):
        """Every label was `fill="white"`. That is fine on `--data-2` and close
        to illegible on `--data-4`: white on #A8BECD is about 1.8:1, against
        the 4.5:1 a label needs. `--on-data-*` states which ink goes with which
        step, and the pairing has to be used, not just defined."""
        svg = _bool_svg(html)
        assert 'fill="white"' not in svg
        inks = set(re.findall(r'<text[^>]*fill="var\((--[\w-]+)', svg))
        assert inks <= {"--on-data-2", "--on-data-4", "--ink"}, inks

    def test_the_ink_matches_the_step_beneath_it(self, html):
        svg = _bool_svg(html)
        # The pale step takes dark ink, the dark step takes pale ink.
        assert "--on-data-4" in svg or "false" not in svg
        assert "--on-data-2" in svg or "true" not in svg

    def test_it_is_a_bar_not_a_band(self, html):
        """38px, not the 52px it was -- that is a chart-sized block for a
        column with two values."""
        svg = _bool_svg(html)
        seg = int(re.search(r'class="seg[^"]*"[^>]*height="(\d+)"', svg).group(1))
        assert 36 <= seg <= 40, seg

    def test_the_share_is_labelled_in_place(self, html):
        svg = _bool_svg(html)
        labels = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
        assert any("true" in label for label in labels), labels
        assert any("%" in label for label in labels), labels


class TestBooleanEdgeCases:
    def test_a_column_with_no_true_values(self):
        out = profile(pd.DataFrame({"b": [False] * 300}), seed=0).html
        svg = _bool_svg(out)
        assert "<svg" in svg
        assert "false" in svg

    def test_a_column_with_no_false_values(self):
        out = profile(pd.DataFrame({"b": [True] * 300}), seed=0).html
        assert "<svg" in _bool_svg(out)

    def test_an_entirely_missing_column_does_not_divide_by_zero(self):
        frame = pd.DataFrame({"b": pd.Series([None] * 200, dtype="object")})
        out = profile(frame, seed=0).html
        assert "<html" in out


# --------------------------------------------------------------------------- #
# the fixed categorical axis
# --------------------------------------------------------------------------- #
class TestTheAxesAreFixedNotPopulated:
    """Two populated months as two half-width slabs reads as *spread evenly
    across the timeline* rather than *2 of 12*."""

    @pytest.mark.parametrize(
        ("chart", "slots"),
        [("hour of day", 24), ("day of week", 7), ("month", 12)],
    )
    def test_every_bucket_gets_a_slot(self, html, chart, slots):
        card = _card(html, "datetime")
        found = re.findall(
            rf"Bar chart showing {chart} distribution with (\d+) bars", card
        )
        assert found, chart
        assert all(int(n) == slots for n in found), (chart, found)

    def test_the_month_chart_keeps_twelve_when_six_are_populated(self, html):
        """The fixture spans about 175 days, so half the year is empty."""
        card = _card(html, "datetime")
        assert "month distribution with 12 bars" in card

    def test_the_labels_survive(self, html):
        """`Mon…Sun` is what tells a reader whether the week starts Monday."""
        card = _card(html, "datetime")
        for day in ("Mon", "Sun"):
            assert day in card, day
        for month in ("Jan", "Dec"):
            assert month in card, month

    def test_a_single_day_of_data_still_gets_24_hours(self):
        frame = pd.DataFrame({"t": pd.date_range("2026-03-01", periods=24, freq="h")})
        out = profile(frame, seed=0).html
        assert "hour of day distribution with 24 bars" in out

    def test_a_birth_date_column_is_all_zeros_at_midnight_and_that_is_fine(self):
        """Correct output, and it must not look like a failure: the slots are
        still there, they are simply empty."""
        frame = pd.DataFrame(
            {"born": pd.date_range("1980-01-01", periods=200, freq="D")}
        )
        out = profile(frame, seed=0).html
        assert "hour of day distribution with 24 bars" in out


class TestTheTemporalChartsSayWhatTheyCount:
    def test_the_y_axis_names_its_unit(self, html):
        """Nothing said these were records per bucket -- a reader could as
        easily have taken the axis for a share, or for the column's values."""
        assert ">RECORDS<" in _card(html, "datetime")

    def test_the_chart_carries_no_legacy_hue(self):
        source = TEMPORAL.read_text().lower()
        for legacy in ("4ea3f1", "8ac926", "ffca3a", "ff595e"):
            assert legacy not in source, legacy

    def test_its_figures_are_monospace(self):
        assert "var(--font-mono)" in TEMPORAL.read_text()
