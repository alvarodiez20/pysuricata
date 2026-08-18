"""A statistic that cannot say anything is not rendered at all.

Phase 5f.1 (#295). `Entropy`, `Rare levels` and `Top 5 coverage` describe how a
distribution spreads across its levels, and the categorical card rendered all
three for every categorical column. Categorical is the most common column type
-- eight of Titanic's twelve -- and one card face was doing duty for four
different things: a boolean in a string (`Sex`, 2 levels), a true category
(`Embarked`, 3), a sparse identifier (`Cabin`, 147) and a primary key (`Name`,
891). The three statistics were written for the second.

`Sex` got entropy 0.936, rare levels 0 and top-5 coverage 100% -- three
confident figures about the spread of a distribution with two members and no
spread. None of them is *wrong*, which is the problem: `top 5 coverage 100%` on
a two-level column is the top **five**, so it is 100% by arithmetic and reads
as a measurement.

**This file asserts absence, not emptiness.** The distinction is the whole
point of the phase, and it is one step past `_unknown_cell`:

* an **em dash** means *unknown* -- the sketch could not answer, and
  `tests/test_empty_topk.py` is the file that guards it;
* an **absent row** means *does not apply* -- there was no question.

A test that accepted an empty cell would pass on a card that prints a bare dash
under `Entropy`, which tells a reader their report failed to measure something
rather than that there was nothing to measure. Opposite conclusions about their
data, from the same markup.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.categorical_card import (
    ENTROPY,
    RARE_LEVELS,
    TOP5_COVERAGE,
    suppressed_statistics,
)

#: The three under test, by the caption each renders.
SPREAD = {"Entropy", "Rare levels", "Top 5 coverage"}


def _body(html: str) -> str:
    """The report with its inlined CSS and JS removed.

    The report inlines its own stylesheet and scripts, so searching the whole
    document for a caption finds it in the source that generates it. Every
    assertion below is about markup, so every one of them needs this first.
    """
    return re.sub(r"(?s)<(script|style)\b.*?</\1>", "", html)


def _card(html: str, column: str) -> str:
    """One column's card, by id."""
    found = re.search(
        rf'<article[^>]*id="col_{re.escape(column)}".*?</article>', _body(html), re.S
    )
    assert found, f"no card for {column} -- the markup moved, not a pass"
    return found.group(0)


def _slots(card: str) -> list[tuple[str, str]]:
    """The stat row as (caption, value) pairs, in document order."""
    return re.findall(
        r'<div class="vstat__cap">(.*?)</div><div class="vstat__val">(.*?)</div>',
        card,
        re.S,
    )


def _captions(card: str) -> list[str]:
    return [caption for caption, _ in _slots(card)]


def _profile(frame: pd.DataFrame) -> str:
    return profile(frame, seed=0).html


# --------------------------------------------------------------------------- #
# the fixtures, and what each one is for
# --------------------------------------------------------------------------- #
# A frame that misses a branch reports *absent*, and absent is what this whole
# file asserts -- so a fixture that quietly fails to produce the column it
# names would make every test here pass while proving nothing. Each one is
# checked against the rule it is meant to trip, in
# `TestTheFixturesReachTheBranchesTheyClaim` at the bottom.
FRAME = pd.DataFrame(
    {
        # Two levels. Loses all three: no spread, and the top five are both of
        # them.
        "sex": ["male", "female"] * 400,
        # Three levels, none rare. Keeps entropy and rare levels, loses top-5
        # coverage -- suppression is per statistic, not per column.
        "port": ["s"] * 500 + ["c"] * 200 + ["q"] * 100,
        # Two levels, one of them genuinely under the 1% rare threshold. Still
        # loses `Rare levels`: the bar chart above prints `0.4%` next to the
        # level itself, so the summary of the tail is a summary of nothing the
        # reader cannot already see.
        "lopsided": ["yes"] * 797 + ["no"] * 3,
        # Eight levels. Clear of every rule; keeps all three, which is what
        # stops the suppression from being a blanket.
        "grade": list("abcdefgh") * 100,
        # Eight hundred values, every one of them seen exactly once. Entropy
        # over values that never repeat is `log2(n)` -- a restatement of
        # `Unique`, three cells to its left, dressed as a measure of how the
        # values repeat.
        "token": [f"t{i:03d}" for i in range(800)],
    }
)


@pytest.fixture(scope="module")
def html() -> str:
    return _profile(FRAME)


# --------------------------------------------------------------------------- #
# 1. the rule, without rendering
# --------------------------------------------------------------------------- #
class TestTheRuleItself:
    """`suppressed_statistics` is a pure function of the derived stats, so it
    can be pinned without building a frame for every case."""

    @staticmethod
    def _facts(**overrides) -> dict:
        facts = {
            "tracked": True,
            "levels_complete": True,
            "n_levels": 8,
            "all_singletons": False,
        }
        facts.update(overrides)
        return facts

    def test_a_full_distribution_keeps_everything(self):
        assert suppressed_statistics(self._facts()) == frozenset()

    def test_two_levels_lose_all_three(self):
        assert suppressed_statistics(self._facts(n_levels=2)) == {
            ENTROPY,
            RARE_LEVELS,
            TOP5_COVERAGE,
        }

    @pytest.mark.parametrize("levels", [3, 4, 5])
    def test_five_or_fewer_lose_only_top_five_coverage(self, levels):
        """The top five of five levels is all of them, so the figure is 100%
        by construction. Entropy and rare levels still have something to
        measure at three levels, and keeping them is what makes this a
        per-statistic rule rather than a threshold on the column."""
        assert suppressed_statistics(self._facts(n_levels=levels)) == {TOP5_COVERAGE}

    def test_six_levels_keep_top_five_coverage(self):
        """The boundary from the other side: at six levels the figure is a
        measurement again, because one level is outside it."""
        assert suppressed_statistics(self._facts(n_levels=6)) == frozenset()

    def test_all_singletons_lose_entropy_only(self):
        """Entropy over values that never repeat collapses to `log2(n)`.
        Nothing else does, so nothing else goes."""
        assert suppressed_statistics(self._facts(n_levels=40, all_singletons=True)) == {
            ENTROPY
        }

    def test_an_unproven_level_count_suppresses_nothing(self):
        """The level rules need `levels_complete`, and without it the sketch
        is a sample rather than a census -- `Top 5 coverage` is then a real
        measurement of a real tail. Suppressing on an estimate would make the
        card change shape between runs of the same data."""
        assert (
            suppressed_statistics(self._facts(levels_complete=False, n_levels=2))
            == frozenset()
        )

    def test_an_untracked_column_is_left_to_the_dash(self):
        """*Unknown* and *absent* are different statements and must not be
        collapsed into one. An empty sketch is the first, and
        `test_empty_topk.py` owns it."""
        assert (
            suppressed_statistics(self._facts(tracked=False, n_levels=2)) == frozenset()
        )


# --------------------------------------------------------------------------- #
# 2. the card
# --------------------------------------------------------------------------- #
class TestTheRowsAreAbsentRatherThanEmpty:
    def test_a_two_level_column_renders_none_of_the_three(self, html):
        captions = _captions(_card(html, "sex"))
        assert SPREAD.isdisjoint(captions), sorted(SPREAD.intersection(captions))

    def test_it_renders_nine_slots_rather_than_twelve(self, html):
        assert len(_slots(_card(html, "sex"))) == 9

    def test_no_slot_is_a_placeholder(self, html):
        """The row closes up. A suppressed statistic leaves no cell at all --
        not an empty one, and not one holding the em dash that means the
        sketch could not answer."""
        for caption, value in _slots(_card(html, "sex")):
            assert value.strip(), f"{caption} rendered an empty cell"
            assert "—" not in value, f"{caption} rendered a dash, not an absence"

    def test_a_singleton_column_loses_entropy_and_keeps_the_rest(self, html):
        captions = _captions(_card(html, "token"))
        assert "Entropy" not in captions
        assert "Top 5 coverage" in captions

    def test_a_three_level_column_loses_only_top_five_coverage(self, html):
        captions = _captions(_card(html, "port"))
        assert "Top 5 coverage" not in captions
        assert "Entropy" in captions
        assert "Rare levels" in captions

    def test_a_rare_level_is_suppressed_with_the_rest_at_two_levels(self, html):
        """The one case where the suppressed figure would have been non-zero.

        `lopsided` really does have a level under the 1% threshold, so `Rare
        levels 1` would be true. It goes anyway: with two levels the chart
        prints both shares directly above, and a statistic that summarises a
        tail is not earning its slot when there is no tail to summarise.
        """
        assert "Rare levels" not in _captions(_card(html, "lopsided"))

    def test_a_column_with_a_real_spread_keeps_all_three(self, html):
        """The rule has to be able to *not* fire, or it is a deletion."""
        captions = _captions(_card(html, "grade"))
        assert SPREAD.issubset(captions), sorted(SPREAD.difference(captions))


class TestTheRestOfTheCardIsUntouched:
    """Suppression takes three cells and nothing else."""

    @pytest.mark.parametrize("column", ["sex", "port", "grade", "token", "lopsided"])
    def test_every_card_still_carries_the_counting_half(self, html, column):
        captions = _captions(_card(html, column))
        for required in ("Count", "Unique", "Missing", "Mode", "Empty or zero"):
            assert required in captions, f"{column} lost {required}"

    def test_the_levels_are_still_on_the_chart(self, html):
        """What justifies dropping the three is that a two-level column shows
        both of its levels, with their exact shares, immediately above the
        stat row. If that ever stops being true the justification goes with
        it."""
        card = _card(html, "sex")
        assert "male" in card and "female" in card
        assert re.search(r"2 of 2 levels shown", card)


class TestProcessedBytesReportsTheColumn:
    """Found by looking at the card this phase rebuilt.

    `_right_stats` read `mem_bytes` off the derived-stats dict, which has never
    had that key, so the `.get()` default rendered and **every** categorical
    column in every report claimed to have processed `0.0 B`. It is the third
    field in this one function to go the same way -- `avg_len` and `len_p90`
    were the first two (#155, 5c.2).
    """

    @pytest.mark.parametrize("column", ["sex", "port", "grade", "token"])
    def test_it_is_not_zero(self, html, column):
        slots = dict(_slots(_card(html, column)))
        assert "Processed bytes (≈)" in slots
        assert slots["Processed bytes (≈)"] not in ("0.0 B", "0 B"), (
            f"{column} reports no processed bytes"
        )


# --------------------------------------------------------------------------- #
# 3. the guard on the fixtures
# --------------------------------------------------------------------------- #
class TestTheFixturesReachTheBranchesTheyClaim:
    """Every assertion above is an absence, and a column that never rendered
    would satisfy all of them. These check the frame produced what it said."""

    @pytest.mark.parametrize(
        "column,levels", [("sex", 2), ("port", 3), ("lopsided", 2), ("grade", 8)]
    )
    def test_the_column_has_the_level_count_the_rule_needs(self, html, column, levels):
        slots = dict(_slots(_card(html, column)))
        assert slots["Unique"] == f"{levels:,}"

    def test_every_column_rendered_as_a_categorical_card(self, html):
        body = _body(html)
        kinds = set(re.findall(r'<article[^>]*data-type="(\w+)"', body))
        assert kinds == {"categorical"}, (
            f"a fixture column profiled as something else: {sorted(kinds)}"
        )

    def test_the_singleton_column_is_tracked_and_all_singletons(self, html):
        """It has to reach the entropy branch, not the untracked one -- those
        produce the same absence for opposite reasons."""
        card = _card(html, "token")
        assert "no value repeats often enough" not in card
        assert "Entropy" not in _captions(card)
