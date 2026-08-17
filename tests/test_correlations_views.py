"""The three correlation views, and the one that actually renders.

Phase 6 (#119).

The empty state is the common case — both example reports hit it — and it read
``No significant correlations found``. But nothing was missing: the pairs *were*
computed and every one came back weak. That is a finding, not an absence, and
the numbers to say so were already to hand.

The other theme is that **sign is position, not colour**. A red bar for a
negative correlation reads as *bad*, and a negative correlation is often the
interesting one. A diverging bar survives greyscale and needs no legend.

#119 asks specifically that both populated views be checked against a dataset
with genuine correlations, because none had ever been seen in a real report
here — so the fixtures build real structure rather than illustrative numbers.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.correlations_section import CorrelationsSectionRenderer


def _section(html: str) -> str:
    start = html.index('id="correlations"')
    end = html.find('id="missing-values"', start)
    return html[start : end if end != -1 else len(html)]


def _bars(section: str) -> list[tuple[float, float]]:
    return [
        (float(left), float(width))
        for left, width in re.findall(
            r"corr-bar__fill\" style=\"left:([\d.]+)%;width:([\d.]+)%", section
        )
    ]


def _rows(section: str) -> list[tuple[float, float, float]]:
    """(left, width, value) per list row.

    Parsed per row rather than with one pattern spanning the whole row: the bar
    is wrapped in a `<span class="corr-bar">` whose closing tag sits between the
    fill and the value, and a single regex that has to describe everything in
    between breaks on any markup change -- which it did.
    """
    out = []
    for chunk in section.split('<div class="correlation-row">')[1:]:
        bar = re.search(r"left:([\d.]+)%;width:([\d.]+)%", chunk)
        value = re.search(r'correlation-value">\s*([+-][\d.]+)', chunk)
        if bar and value:
            out.append(
                (float(bar.group(1)), float(bar.group(2)), float(value.group(1)))
            )
    return out


@pytest.fixture(scope="module")
def weak() -> str:
    """No pair above threshold — what both example reports produce."""
    rng = np.random.default_rng(0)
    return profile(pd.DataFrame({c: rng.normal(0, 1, 600) for c in "abc"}), seed=0).html


@pytest.fixture(scope="module")
def matrix() -> str:
    """Five columns with real structure, so the matrix has something to show."""
    rng = np.random.default_rng(0)
    n = 900
    x = rng.normal(0, 1, n)
    return profile(
        pd.DataFrame(
            {
                "x": x,
                "y": x * 0.95 + rng.normal(0, 0.15, n),
                "z": -x * 0.8 + rng.normal(0, 0.3, n),
                "w": rng.normal(0, 1, n),
                "v": x * 0.55 + rng.normal(0, 0.8, n),
            }
        ),
        seed=0,
    ).html


@pytest.fixture(scope="module")
def ranked() -> str:
    """Twelve columns, half positively and half negatively related."""
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(0, 1, n)
    cols = {f"p{i}": x * (0.9 - 0.05 * i) + rng.normal(0, 0.2, n) for i in range(6)}
    cols.update(
        {f"n{i}": -x * (0.9 - 0.05 * i) + rng.normal(0, 0.2, n) for i in range(6)}
    )
    return profile(pd.DataFrame(cols), seed=0).html


# --------------------------------------------------------------------------- #
# the common case
# --------------------------------------------------------------------------- #
class TestWeakIsAFindingNotAnAbsence:
    def test_it_says_how_many_were_checked(self, weak):
        section = _section(weak)
        assert "weakly related" in section
        assert re.search(r"All <strong>\d+</strong> numeric pairs", section)

    def test_it_names_the_strongest(self, weak):
        """`No significant correlations found` withheld a number it had."""
        assert re.search(r"strongest is <strong>[\d.]+</strong>", _section(weak))

    def test_it_states_the_threshold_it_fell_under(self, weak):
        assert "reporting threshold" in _section(weak)

    def test_it_lists_the_pairs_with_their_real_values(self, weak):
        section = _section(weak)
        assert section.count("corr-weak__row") == 3
        assert re.search(r'corr-weak__value">[+-][\d.]+<', section)

    def test_the_pairs_get_the_same_diverging_bar(self, weak):
        assert "corr-bar__fill" in _section(weak)

    def test_a_single_pair_reads_as_singular(self):
        rng = np.random.default_rng(0)
        out = profile(
            pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)}),
            seed=0,
        ).html
        section = _section(out)
        assert "numeric pair is weakly related" in section

    def test_fewer_than_two_numeric_columns_says_why(self):
        """It says *why* now rather than restating the rule.

        This used to assert `"at least 2 numeric columns"`, which is the
        sentence #243 was filed about: it tells a reader what a correlation
        needs, which they know, and nothing about the frame in front of them.

        Note what this fixture actually is -- `a` is constant, so it is
        reclassified as categorical and the frame reaches the section with
        **zero** numeric columns, not one. The copy has to hold for that
        without telling the reader their float column is not a number.
        """
        section = _section(
            profile(pd.DataFrame({"a": [1.0] * 50, "t": list("xy") * 25}), seed=0).html
        )
        assert "profiled as numeric" in section
        assert "at least 2 numeric columns" not in section


# --------------------------------------------------------------------------- #
# sign is position
# --------------------------------------------------------------------------- #
class TestSignIsPositionNotColour:
    def test_a_positive_correlation_runs_right_of_centre(self, ranked):
        positives = [(l, w) for l, w, v in _rows(_section(ranked)) if v > 0]
        assert positives, "no positive pairs in the fixture"
        assert all(left == pytest.approx(50.0) for left, _ in positives)

    def test_a_negative_correlation_runs_left_of_centre(self, ranked):
        negatives = [(l, w) for l, w, v in _rows(_section(ranked)) if v < 0]
        assert negatives, "no negative pairs in the fixture"
        for left, width in negatives:
            assert left < 50.0
            assert left + width == pytest.approx(50.0, abs=0.01)

    def test_no_bar_overflows_its_track(self, ranked):
        for left, width in _bars(_section(ranked)):
            assert left >= 0.0
            assert left + width <= 100.001

    def test_a_perfect_correlation_reaches_the_edge_without_passing_it(self):
        rng = np.random.default_rng(0)
        n = 400
        x = rng.normal(0, 1, n)
        cols = {f"c{i}": x * (1 if i % 2 else -1) for i in range(12)}
        out = profile(pd.DataFrame(cols), seed=0).html
        for left, width in _bars(_section(out)):
            assert left >= 0.0
            assert left + width <= 100.001

    def test_the_scale_says_where_zero_is(self, ranked):
        assert "corr-scale" in _section(ranked)
        assert "← 0 →" in _section(ranked)


class TestTheRankedList:
    def test_the_rank_badges_are_gone(self, ranked):
        """The list is ordered; `#1` beside the first row is noise."""
        assert "rank-badge" not in _section(ranked)

    def test_the_strength_bands_are_steps_of_one_hue(self, ranked):
        fills = set(re.findall(r"background:var\((--data-\d)\)", _section(ranked)))
        assert fills
        assert all(f.startswith("--data-") for f in fills)

    def test_the_count_says_what_was_checked_not_only_what_passed(self, ranked):
        badge = re.search(r'correlation-count-badge">([^<]+)<', _section(ranked))
        assert badge
        assert "checked" in badge.group(1)
        assert re.search(r"\d+ pairs? above [\d.]+, of [\d,]+ checked", badge.group(1))


# --------------------------------------------------------------------------- #
# the matrix
# --------------------------------------------------------------------------- #
class TestTheMatrixIsALowerTriangle:
    def test_the_diagonal_is_gone(self, matrix):
        """It said 1.00 once per column — half the ink for none of the
        information."""
        assert "corr-cell diagonal" not in _section(matrix)

    def test_no_pair_appears_twice(self, matrix):
        section = _section(matrix)
        values = re.findall(r'data-corr="([-\d.]+)"', section)
        # Five columns is ten pairs, each drawn once.
        assert len(values) == 10

    def test_a_weak_pair_stays_visible(self, matrix):
        """Hidden, it is indistinguishable from a pair that could not be
        computed — and an all-weak row is information."""
        assert "corr-cell weak" in _section(matrix)

    def test_the_sign_is_the_printed_number(self, matrix):
        section = _section(matrix)
        assert re.search(r'data-corr="-[\d.]+"', section)
        assert re.search(r">[+-][\d.]+</td>", section)

    def test_it_falls_back_to_the_list_above_ten_columns(self, ranked):
        assert "correlation-matrix" not in _section(ranked)

    def test_two_numeric_columns_use_the_list_not_a_single_cell(self):
        rng = np.random.default_rng(0)
        n = 400
        x = rng.normal(0, 1, n)
        out = profile(
            pd.DataFrame({"x": x, "y": x * 0.95 + rng.normal(0, 0.1, n)}), seed=0
        ).html
        assert "correlation-matrix" not in _section(out)


class TestTheEmojiAreGone:
    """Not part of this brand, and they render inconsistently across
    platforms."""

    @pytest.mark.parametrize("emoji", ["📊", "📈", "📉"])
    def test_none_reach_the_report(self, matrix, ranked, weak, emoji):
        for html in (matrix, ranked, weak):
            assert emoji not in html


# --------------------------------------------------------------------------- #
# The two states that mean "nothing to compare"
# --------------------------------------------------------------------------- #
class TestAnEmptyStateSaysWhichEmptyItIs:
    """#243. Phase 6.1's enriched copy landed on the path where pairs exist and
    all come back weak -- the interesting case, and the one both example reports
    hit. The two paths that mean *nothing to compare* kept a single bare
    sentence, and they are the ones a small frame lands on.

    "Correlation analysis requires at least 2 numeric columns" states the rule
    and none of the case. A reader looking at a correlations section already
    knows a correlation needs two things; what they cannot see is how many this
    frame has, which one it is when it has one, or -- when it has several and
    still shows nothing -- why.
    """

    @staticmethod
    def _text(frame: pd.DataFrame) -> str:
        section = _section(profile(frame, seed=0).html)
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", section)).strip()

    def test_one_numeric_column_is_named(self):
        frame = pd.DataFrame({"age": np.arange(60.0), "grade": list("ab") * 30})
        text = self._text(frame)
        assert "age" in text, text
        assert "only numeric column" in text, text

    def test_no_numeric_column_points_at_the_typing(self):
        """And says it about *the report*, not about the data.

        "This dataset has no numeric columns" is a claim about the frame and it
        can be false: a column that never varies is reclassified as
        categorical, so two constant float columns reach here and would be told
        the dataset holds no numbers. The report's own Summary says 0 numeric
        for that frame, so the sentence has to agree with the classification
        rather than contradict the input.
        """
        frame = pd.DataFrame({"a": list("xyz") * 20, "b": list("pq") * 30})
        text = self._text(frame)
        assert "profiled as numeric" in text, text
        assert "dataset has no numeric" not in text, text

    def test_two_constant_floats_are_not_told_they_are_not_numbers(self):
        frame = pd.DataFrame({"a": [1.0] * 60, "b": [2.0] * 60})
        text = self._text(frame)
        assert "profiled as numeric" in text, text

    def test_numeric_columns_with_no_usable_pair_say_why(self):
        """Reachable with the estimator absent -- correlations switched off, or
        a frame the estimator never ran on -- and the bare copy could not tell
        that apart from having no numeric columns at all."""
        renderer = CorrelationsSectionRenderer()
        out = renderer.render_section(None, ["a", "b"])
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", out)).strip()
        assert "no pair produced a usable coefficient" in text, text
        assert "never varies" in text, text

    def test_the_weak_state_is_untouched(self):
        """The path that was already enriched must stay that way -- it is the
        common one."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)})
        text = self._text(frame)
        assert "weakly related" in text, text
        assert "strongest is" in text, text

    def test_no_state_prints_the_old_bare_rule(self):
        for frame in (
            pd.DataFrame({"age": np.arange(60.0), "g": list("ab") * 30}),
            pd.DataFrame({"a": list("xyz") * 20}),
            pd.DataFrame({"a": [1.0] * 60, "b": [2.0] * 60}),
        ):
            assert "requires at least 2 numeric columns" not in self._text(frame)
