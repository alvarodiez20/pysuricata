"""The column-type composition bar, which replaced the donut. Closes #104.

Rewritten from ``test_donut_chart.py``. Most of that file asserted arc
geometry — the 100%-share special case that had to draw a ``<circle>`` because
a 360° arc has identical start and end points, the inner ring drawn at 0.6
opacity for depth — and none of it survives, because none of the shapes do.

What replaces those assertions is the property the old chart could not state:
**the segments must account for exactly 100% of the columns.** A donut whose
arcs summed to 359.9° looked fine; a bar whose segments sum to 99.9% has a gap
at the right edge, which reads as a rendering bug. That is the difference
between a chart you check by eye and one you can check by arithmetic.
"""

from __future__ import annotations

import re

import pytest

from pysuricata.render.composition_bar import (
    _MIN_SHARE_FOR_LABEL,
    CompositionBarRenderer,
    apportion,
)


def _segments(html: str) -> list[dict]:
    out = []
    for tag in re.findall(r"<div class=\"composition__seg\"[^>]*>", html):
        out.append(
            {
                "type": re.search(r'data-type="([^"]+)"', tag).group(1),
                "count": int(re.search(r'data-count="(\d+)"', tag).group(1)),
                "percent": float(
                    re.search(r'data-percentage="([\d.]+)"', tag).group(1)
                ),
                "width": float(re.search(r"width:([\d.]+)%", tag).group(1)),
            }
        )
    return out


@pytest.fixture
def renderer() -> CompositionBarRenderer:
    return CompositionBarRenderer()


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #
class TestTheWidthsSumToExactlyOneHundred:
    """Rounding each share on its own leaves the bar short or long: a third,
    three times, rounds to 99.9. The largest-remainder method fixes the total
    first and hands the leftover to whichever shares lost most to the floor."""

    @pytest.mark.parametrize(
        "counts",
        [
            (3, 8, 1, 0),
            (1, 1, 1, 0),  # thirds — the case that motivates the method
            (1, 1, 1, 1),
            (75, 0, 0, 0),
            (1, 0, 0, 0),
            (2, 8, 1, 1),
            (7, 7, 7, 7),
            (1, 2, 3, 4),
            (999, 1, 1, 1),
            (1, 1, 1, 997),
        ],
    )
    def test_for_every_shape(self, renderer, counts):
        segments = _segments(renderer.render(*counts))
        assert sum(segment["percent"] for segment in segments) == pytest.approx(100.0)

    def test_thirds_do_not_lose_a_tenth(self):
        assert sum(apportion([1, 1, 1])) == pytest.approx(100.0)
        assert sorted(apportion([1, 1, 1])) == [33.3, 33.3, 33.4]

    def test_sevenths_do_not_lose_a_tenth(self):
        shares = apportion([1] * 7)
        assert sum(shares) == pytest.approx(100.0)

    def test_an_exact_split_is_left_exact(self):
        assert apportion([1, 2, 3, 4]) == [10.0, 20.0, 30.0, 40.0]

    def test_no_columns_is_zero_not_a_division(self):
        assert apportion([0, 0, 0, 0]) == [0.0, 0.0, 0.0, 0.0]

    def test_the_rendered_width_matches_the_published_percentage(self, renderer):
        for segment in _segments(renderer.render(3, 8, 1, 2)):
            assert segment["width"] == segment["percent"]


class TestTheSegmentsMatchTheCounts:
    def test_each_type_carries_its_own_count(self, renderer):
        segments = {
            s["type"]: s["count"] for s in _segments(renderer.render(3, 8, 1, 2))
        }
        assert segments == {"numeric": 3, "categorical": 8, "datetime": 1, "boolean": 2}

    def test_they_are_ordered_largest_first(self, renderer):
        counts = [s["count"] for s in _segments(renderer.render(3, 8, 1, 2))]
        assert counts == sorted(counts, reverse=True)

    def test_a_tie_is_broken_the_same_way_every_time(self, renderer):
        """Order must not depend on dict iteration or on the input order, or
        the same frame renders differently between runs."""
        first = [s["type"] for s in _segments(renderer.render(1, 1, 1, 1))]
        second = [s["type"] for s in _segments(renderer.render(1, 1, 1, 1))]
        assert first == second


# --------------------------------------------------------------------------- #
# a type with no columns
# --------------------------------------------------------------------------- #
class TestAZeroCountType:
    """A zero-width segment is an artifact rather than information, and the
    palest step of the data scale sits close enough to ``--track`` that a
    hairline of it reads as a seam in the bar."""

    def test_it_gets_no_segment(self, renderer):
        types = [s["type"] for s in _segments(renderer.render(3, 8, 0, 0))]
        assert types == ["categorical", "numeric"]

    def test_it_still_appears_in_the_legend(self, renderer):
        html = renderer.render(3, 8, 0, 0)
        assert html.count("composition__item") == 4
        assert "datetime" in html
        assert "boolean" in html

    def test_the_legend_marks_it_as_empty(self, renderer):
        html = renderer.render(3, 8, 0, 0)
        assert html.count("is-zero") >= 2

    def test_its_swatch_is_not_filled(self, renderer):
        """A filled swatch would imply a share of the bar that is not there."""
        html = renderer.render(3, 8, 0, 0)
        assert '<span class="composition__swatch is-zero"></span>' in html


# --------------------------------------------------------------------------- #
# edge cases named in #112
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_every_column_one_type(self, renderer):
        """One segment at 100%. The count still has to fit inside it."""
        segments = _segments(renderer.render(75, 0, 0, 0))
        assert len(segments) == 1
        assert segments[0]["percent"] == 100.0
        assert ">75<" in renderer.render(75, 0, 0, 0)

    def test_a_single_column(self, renderer):
        segments = _segments(renderer.render(1, 0, 0, 0))
        assert len(segments) == 1
        assert segments[0]["percent"] == 100.0

    def test_no_columns_at_all(self, renderer):
        """`profile(pd.DataFrame())` must not divide by zero, and must not draw
        a bar of nothing."""
        html = renderer.render(0, 0, 0, 0)
        assert "composition__seg" not in html
        assert "No columns" in html
        assert 'role="img"' in html

    def test_a_sliver_omits_its_count_rather_than_overlapping(self, renderer):
        """Below the threshold the number would spill over its neighbour. It is
        in the legend regardless, so nothing is lost."""
        html = renderer.render(999, 1, 0, 0)
        segments = _segments(html)
        narrow = [s for s in segments if s["percent"] < _MIN_SHARE_FOR_LABEL]
        assert narrow, "expected a segment too narrow to label"
        bar = html.split('class="composition__bar"', 1)[1].split("</div></div>", 1)[0]
        assert bar.count("composition__count") == len(segments) - len(narrow)

    def test_a_wide_segment_keeps_its_count(self, renderer):
        html = renderer.render(3, 8, 1, 2)
        bar = html.split('class="composition__bar"', 1)[1]
        assert "composition__count" in bar


# --------------------------------------------------------------------------- #
# the palette, and the accessible text
# --------------------------------------------------------------------------- #
class TestItStaysOnTheDataScale:
    def test_every_fill_is_a_step_of_the_data_scale(self, renderer):
        html = renderer.render(3, 8, 1, 2)
        fills = re.findall(r"background:var\((--[\w-]+)", html)
        assert fills, "no fills found"
        assert all(fill.startswith("--data-") for fill in fills), fills

    def test_no_quality_colour_labels_a_type(self, renderer):
        """Type is not a colour, and the quality scale means something else.
        Olive meaning both `categorical` and `passes` is the collision the
        palette exists to remove."""
        html = renderer.render(3, 8, 1, 2)
        for quality in ("--q-good", "--q-warn-fill", "--q-warn-text", "--q-bad"):
            assert quality not in html, quality

    def test_the_text_on_each_segment_is_paired_with_its_fill(self, renderer):
        """`--on-data-*` exists so a label is never dark ink on a dark step.

        `--data-3` is the exception and has no partner: at 4.03:1 on the paper
        and 3.83:1 on the ink it reaches neither text minimum, so it carries no
        label at all. A pairing would be a promise the palette cannot keep.
        """
        html = renderer.render(3, 8, 1, 2)
        for index in (1, 2, 4):
            if f"--data-{index}," in html:
                assert f"--on-data-{index}," in html
        assert "--on-data-3" not in html

    def test_a_data_3_segment_carries_no_count_and_no_text_colour(self, renderer):
        """The count goes to the legend, the way a too-narrow segment's does."""
        html = renderer.render(3, 8, 1, 2)
        segment = re.search(
            r'<div class="composition__seg"[^>]*--data-3[^>]*>(.*?)</div>', html
        )
        assert segment, "no --data-3 segment in a four-type bar"
        assert "composition__count" not in segment.group(1)
        assert "color:" not in segment.group(0)

    def test_the_count_is_still_reachable_in_the_legend(self, renderer):
        """Dropping the in-segment label must not drop the number."""
        html = renderer.render(3, 8, 1, 2)
        legend = (
            html[html.index("composition__legend") :]
            if "composition__legend" in html
            else html
        )
        for count in (3, 8, 1, 2):
            assert f">{count}<" in legend


class TestTheAccessibleDescription:
    """The donut carried a `<desc>`; the bar carries an `aria-label`. Same
    sentence, and it survives being read rather than seen."""

    def test_it_states_every_drawn_type_with_its_count(self, renderer):
        html = renderer.render(3, 8, 1, 2)
        label = re.search(r'aria-label="([^"]+)"', html).group(1)
        for word in ("numeric 3", "categorical 8", "datetime 1", "boolean 2"):
            assert word in label

    def test_it_says_so_when_there_is_nothing(self, renderer):
        label = re.search(r'aria-label="([^"]+)"', renderer.render(0, 0, 0, 0)).group(1)
        assert "no columns" in label.lower()

    def test_the_bar_is_one_image_not_four(self, renderer):
        """Four segments announced separately is four announcements of a thing
        that only means something as a whole."""
        html = renderer.render(3, 8, 1, 2)
        assert html.count('role="img"') == 1
