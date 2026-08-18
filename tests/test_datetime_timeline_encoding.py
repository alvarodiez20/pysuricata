"""The timeline draws bars, because a bucket count is a quantity per interval.

Phase 5e.4 (#293). The timeline was a `<polyline>` through bucket centres, so
between "84 records on 8 Jan" and "83 on 9 Jan" it drew a continuous slope --
asserting every intermediate value, when the data holds values only at the
buckets. The card's own temporal panes and the numeric histogram already drew
counts as bars, so one report carried two encodings for one quantity.

The issue was filed as a decision with three options, and the plan called it
"the one genuine trade in the phase": bars are honest, a line reads a trend
better, and the proposal was to switch between them at ~180 buckets where bars
go sub-pixel. Two measurements settled it, and both are pinned below because
the argument is only as good as they are:

* **The threshold could not have fired.** `DEFAULT_DT_CONFIG.default_bins` is
  60 and is not reachable from `ProfileConfig` or `ComputeOptions`, so
  `min(bins, 180)` is always 60. The line branch would have been unreachable
  code -- no input reaches it, so no test could either.
* **The sub-pixel risk is a viewport width, not a bucket count.** Those 60
  buckets are ~12.5px each at 1240 and ~3.8px at 390. A static report cannot
  branch on the width it will be read at, and the numeric histogram already
  ships bars of about that width at 390 on the same screen.

So: bars, always, at every bucket count this renderer produces.

Geometry belongs to `test_chart_layout.py`, which owns the figure's structure.
This file is about what the mark *means*.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.card_config import DEFAULT_DT_CONFIG


def _figure(html: str) -> str:
    """The timeline figure, with the inlined stylesheet and scripts gone.

    The report inlines its own CSS and JS, so searching the document for a
    class name finds it in the source that defines it.
    """
    stripped = re.sub(r"(?s)<(script|style)\b.*?</\1>", "", html)
    match = re.search(r'<figure class="hist dt-figure">.*?</figure>', stripped, re.S)
    assert match, "the datetime card no longer renders a hist figure"
    return match.group(0)


def _bars(figure: str) -> list[dict[str, float]]:
    return [
        {k: float(v) for k, v in re.findall(r'(\w+)="([-\d.]+)"', tag)}
        for tag in re.findall(r'<rect class="bar"[^/]*/>', figure)
    ]


def _hotspots(figure: str) -> list[str]:
    return re.findall(r'<rect class="hot"[^/]*/>', figure)


def _profile(frame: pd.DataFrame) -> str:
    return profile(frame, seed=0).html


@pytest.fixture(scope="module")
def dense() -> str:
    """A year of records with no empty stretch: every bucket draws."""
    rng = np.random.default_rng(0)
    n = 4000
    return _profile(
        pd.DataFrame(
            {
                "seen_at": pd.to_datetime("2024-01-01")
                + pd.to_timedelta(
                    np.sort(rng.integers(0, 365 * 24 * 3600, n)), unit="s"
                ),
                "amount": rng.normal(50, 12, n),
            }
        )
    )


@pytest.fixture(scope="module")
def gappy() -> str:
    """Two ten-day bursts ten months apart.

    The case the polyline was worst at, and the one a uniform fixture cannot
    reach: between the bursts the line sloped down to zero and back up again,
    drawing a gradual decline and recovery across ten months in which nothing
    happened at all.
    """
    rng = np.random.default_rng(0)
    first = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 10 * 86400, 900), unit="s"
    )
    second = pd.to_datetime("2024-11-01") + pd.to_timedelta(
        rng.integers(0, 10 * 86400, 900), unit="s"
    )
    return _profile(
        pd.DataFrame(
            {
                "seen_at": np.sort(np.concatenate([first.values, second.values])),
                "amount": rng.normal(50, 12, 1800),
            }
        )
    )


class TestTheEncodingIsBars:
    def test_the_timeline_draws_bars(self, dense):
        assert len(_bars(_figure(dense))) > 0

    def test_no_polyline_survives_anywhere_in_the_report(self, dense):
        """Not just in the figure. A `<polyline>` left elsewhere in the card
        would still be a line drawn through counts."""
        stripped = re.sub(r"(?s)<(script|style)\b.*?</\1>", "", dense)
        assert "<polyline" not in stripped

    def test_the_bars_are_the_report_s_one_count_mark(self, dense):
        """`class="bar"`, the same class the numeric histogram and the temporal
        panes use, so the separator, the fill token and the hover state are
        defined once and cannot drift apart from the histogram's.

        The figure holds exactly two kinds of rect and no third: the bars that
        encode the counts, and the transparent hover columns over them. A new
        class here would be a second way to draw a count in the one chart this
        phase was about unifying.
        """
        classes = set(re.findall(r'<rect class="([^"]*)"', _figure(dense)))

        assert classes == {"bar", "hot"}, sorted(classes)


class TestAZeroCountDrawsNothing:
    """Design-system rule 3, and the reason a 1px floor is wrong here: ten
    empty months as ten 1px bars assert data that is not there."""

    def test_an_empty_bucket_has_no_bar(self, gappy):
        figure = _figure(gappy)
        empty = sum(1 for h in _hotspots(figure) if 'data-count="0"' in h)

        assert empty > 0, "the fixture has no empty bucket, so it proves nothing"
        assert len(_bars(figure)) == len(_hotspots(figure)) - empty

    def test_the_gap_really_is_most_of_the_chart(self, gappy):
        """A guard on the fixture. If the bursts ever merged into one bucket
        this file would still pass while testing nothing."""
        figure = _figure(gappy)

        assert len(_bars(figure)) < 10
        assert len(_hotspots(figure)) == DEFAULT_DT_CONFIG.default_bins

    def test_a_dense_column_draws_every_bucket(self, dense):
        """The rule has to be able to *not* fire, or it is a deletion."""
        figure = _figure(dense)

        assert len(_bars(figure)) == len(_hotspots(figure))


class TestTheHoverTargetsOutliveTheBars:
    """The design proposed the hotspot rects *become* the bars. They cannot.

    A hotspot is full height so the tooltip answers anywhere in its column; a
    bar is count height. Merging them would make an empty bucket unhoverable --
    exactly the bucket whose `0 rows` is worth reading, since a gap in bars
    looks identical to a gap in the axis. `functionality.js` matches `.hot`,
    and it had already been dead once for a class rename (#219).
    """

    def test_every_bucket_keeps_a_hotspot(self, gappy):
        assert len(_hotspots(_figure(gappy))) == DEFAULT_DT_CONFIG.default_bins

    def test_an_empty_bucket_still_answers_with_a_count(self, gappy):
        empty = [h for h in _hotspots(_figure(gappy)) if 'data-count="0"' in h]

        assert empty
        for hotspot in empty:
            assert "data-label=" in hotspot
            assert "data-pct=" in hotspot

    def test_the_hotspots_are_full_height(self, gappy):
        for hotspot in _hotspots(_figure(gappy)):
            assert 'y="0"' in hotspot
            assert 'height="100"' in hotspot


class TestTheBarsEncodeTheCounts:
    def test_they_share_one_baseline(self, dense):
        """`y + height` is the axis. A bar floating off it encodes nothing."""
        for bar in _bars(_figure(dense)):
            assert bar["y"] + bar["height"] == pytest.approx(100.0, abs=0.02)

    def test_height_is_proportional_to_the_count(self, dense):
        """Read off the rendered geometry against the counts the hotspots
        carry, rather than against the same expression that drew them."""
        figure = _figure(dense)
        counts = [
            int(re.search(r'data-count="(\d+)"', h).group(1)) for h in _hotspots(figure)
        ]
        bars = _bars(figure)
        assert len(bars) == len(counts), "the fixture has an empty bucket"

        tallest = max(counts)
        for bar, count in zip(bars, counts, strict=True):
            assert bar["height"] == pytest.approx(count / tallest * 100, abs=0.02)

    def test_the_bars_tile_the_axis_without_overlapping(self, dense):
        bars = sorted(_bars(_figure(dense)), key=lambda b: b["x"])

        assert bars[0]["x"] == pytest.approx(0.0, abs=0.02)
        for left, right in zip(bars, bars[1:], strict=False):
            assert left["x"] + left["width"] == pytest.approx(right["x"], abs=0.02)
        assert bars[-1]["x"] + bars[-1]["width"] == pytest.approx(100.0, abs=0.02)


class TestTheThresholdTheDecisionRejected:
    """Pinning the measurement the decision rests on, not the decision.

    If `default_bins` ever becomes reachable, or its value changes enough that
    a bucket really could go sub-pixel at a normal width, this fails and the
    trade is worth re-opening. That is the whole reason #293 was a decision.
    """

    def test_the_bucket_count_is_fixed_well_under_the_proposed_threshold(self):
        assert DEFAULT_DT_CONFIG.default_bins == 60

    def test_the_bucket_count_is_not_reachable_from_the_public_config(self):
        from pysuricata import api, config

        for module in (api, config):
            source = __import__("inspect").getsource(module)
            assert "default_bins" not in source, (
                f"{module.__name__} can now set the bucket count, so a bucket "
                f"can go sub-pixel and #293's threshold is worth re-opening"
            )
