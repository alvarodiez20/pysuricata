"""When a top-values chart has nothing to say, say that instead.

Phase 5.3. `Name`, `Ticket` and `Cabin` rendered ten bars of one row each — a
chart drawn at full size, carrying no information, indistinguishable at a glance
from one that carries plenty.

Two things make the rule harder than a threshold on ``unique_est``:

**The inputs are approximate.** ``unique_est`` carries about 2.2% of KMV error,
so a column on the boundary can change shape between runs of the same data —
worse than either shape. Coverage comes from Misra-Gries counts, which are
*lower bounds*, so the test can only under-state coverage and therefore errs
towards the sentence.

**``top_items`` may be empty rather than full of singletons.** Misra-Gries is
gated off entirely on high-cardinality columns (#62), so the branch has to
handle *no top values at all* — the case that falls through to an empty chart
and looks like a bug.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.card_types import CategoricalStats
from pysuricata.render.categorical_card import (
    describe_high_cardinality,
    high_cardinality_sentence,
)


def _stats(count: int, unique: int, items) -> CategoricalStats:
    return CategoricalStats(
        name="c",
        dtype_str="object",
        count=count,
        missing=0,
        unique_est=unique,
        approx=True,
        mem_bytes=0,
        top_items=items,
        empty_zero=0,
        case_variants_est=0,
        trim_variants_est=0,
    )


def _card(html: str, name: str) -> str:
    for chunk in html.split('<article class="var-card"')[1:]:
        if f'data-name="{name}"' in chunk:
            return chunk
    raise AssertionError(f"no card for {name}")


@pytest.fixture(scope="module")
def titanic_shaped() -> str:
    """The three columns the issue names, at their real proportions."""
    rng = np.random.default_rng(0)
    n = 891
    cabins = [None] * 687 + [f"C{i % 147}" for i in range(204)]
    rng.shuffle(cabins)
    return profile(
        pd.DataFrame(
            {
                "cabin": cabins,
                "ticket": [f"T{i % 681}" for i in range(n)],
                "name": [f"passenger {i}" for i in range(n)],
                "sex": rng.choice(["male", "female"], n),
                "embarked": rng.choice(["S", "C", "Q"], n),
            }
        ),
        seed=0,
    ).html


# --------------------------------------------------------------------------- #
# when it fires
# --------------------------------------------------------------------------- #
class TestTheBranchFiresWhereItShould:
    @pytest.mark.parametrize("column", ["name", "ticket", "cabin"])
    def test_on_the_three_columns_the_issue_names(self, titanic_shaped, column):
        card = _card(titanic_shaped, column)
        assert 'class="nochart"' in card, column

    @pytest.mark.parametrize("column", ["name", "ticket", "cabin"])
    def test_and_emits_no_chart_for_them(self, titanic_shaped, column):
        """Not an empty chart box -- no box. A container the height of a chart
        with nothing in it reads as a failed render."""
        card = _card(titanic_shaped, column)
        assert "cat-svg" not in card, column

    def test_cabin_is_the_hard_one(self):
        """147 distinct in 204 rows, with the top five covering 8.8% -- so it
        clears the coverage arm and only the cardinality arm catches it. A rule
        written for `Name` alone lets it through."""
        facts = describe_high_cardinality(
            _stats(
                204, 147, [("C23", 4), ("G6", 4), ("B96", 4), ("F2", 3), ("E101", 3)]
            )
        )
        assert facts is not None
        assert facts["coverage"] > 0.02  # the coverage arm does not catch it
        assert facts["distinct_ratio"] > 0.5  # this one does

    def test_it_handles_having_no_counters_at_all(self):
        """Misra-Gries is switched off on high-cardinality columns (#62), so
        `top_items` is empty. The absence is the signal, not a reason to fall
        through to a chart of nothing."""
        assert describe_high_cardinality(_stats(891, 880, [])) is not None

    def test_an_empty_counter_list_is_not_enough_on_its_own(self):
        """An all-missing column also has no top values."""
        assert describe_high_cardinality(_stats(0, 0, [])) is None
        assert describe_high_cardinality(_stats(500, 1, [])) is None


class TestTheBranchDoesNotFireWhereItShouldNot:
    @pytest.mark.parametrize("rows", [100, 1_000, 100_000, 10_000_000])
    def test_never_on_a_three_level_column_at_any_row_count(self, rows):
        """The acceptance criterion, stated as the issue states it."""
        items = [
            ("a", rows // 2),
            ("b", rows // 3),
            ("c", rows - rows // 2 - rows // 3),
        ]
        assert describe_high_cardinality(_stats(rows, 3, items)) is None

    @pytest.mark.parametrize("column", ["sex", "embarked"])
    def test_not_on_an_ordinary_column(self, titanic_shaped, column):
        card = _card(titanic_shaped, column)
        assert 'class="nochart"' not in card
        assert "cat-svg" in card

    def test_not_on_fifty_even_levels(self):
        assert (
            describe_high_cardinality(
                _stats(10_000, 50, [(f"l{i}", 200) for i in range(5)])
            )
            is None
        )

    def test_not_on_a_single_level(self):
        assert describe_high_cardinality(_stats(500, 1, [("x", 500)])) is None

    def test_not_on_an_empty_column(self):
        assert describe_high_cardinality(_stats(0, 0, [])) is None


class TestTheRuleIsStableAgainstSketchError:
    """`unique_est` carries about 2.2% error. A column whose shape flips
    between runs of the same data is worse than either shape."""

    def test_the_cardinality_arm_sits_far_from_the_error(self):
        """0.5, not 0.95: a 2.2% wobble cannot carry a column across it unless
        the column was already ambiguous by a wide margin."""
        near = _stats(1000, 480, [(f"l{i}", 2) for i in range(5)])
        far = _stats(1000, 520, [(f"l{i}", 2) for i in range(5)])
        # Both are caught by the coverage arm anyway -- which is the point: the
        # two arms overlap, so neither boundary is load-bearing alone.
        assert describe_high_cardinality(near) is not None
        assert describe_high_cardinality(far) is not None

    def test_coverage_is_a_lower_bound_so_it_errs_toward_the_sentence(self):
        """Misra-Gries counts under-state. Under-stating coverage makes the
        chart *more* likely to be replaced, which is the safe direction: a
        sentence about a chartable column is a lesser failure than a chart of
        slivers."""
        facts = describe_high_cardinality(
            _stats(1000, 900, [(f"v{i}", 1) for i in range(5)])
        )
        assert facts is not None
        assert facts["coverage"] < 0.02


class TestWhatTheSentenceSays:
    def test_it_claims_uniqueness_only_when_that_is_true(self):
        every = high_cardinality_sentence(
            {
                "unique": 891,
                "count": 891,
                "coverage": 0.006,
                "distinct_ratio": 1.0,
                "identifier_like": True,
            }
        )
        assert "Every value is different" in every

    def test_it_does_not_claim_it_for_a_merely_high_cardinality_column(self):
        """`Cabin` is 147 in 204. Saying every value is different there is
        simply false."""
        some = high_cardinality_sentence(
            {
                "unique": 147,
                "count": 204,
                "coverage": 0.088,
                "distinct_ratio": 0.72,
                "identifier_like": False,
            }
        )
        assert "Every value is different" not in some
        assert "147 distinct values in 204 rows" in some
        assert "8.8%" in some

    def test_the_identifier_flag_agrees_with_the_uniqueness_claim(self, titanic_shaped):
        card = _card(titanic_shaped, "name")
        assert "identifier-like" in card

    def test_a_high_cardinality_column_that_is_not_a_key_is_not_flagged(
        self, titanic_shaped
    ):
        card = _card(titanic_shaped, "cabin")
        assert "identifier-like" not in card


class TestTheCoverageNote:
    def test_an_ordinary_column_states_what_the_bars_account_for(self, titanic_shaped):
        """Without it there is nothing to say whether the bars are the whole
        column or a tenth of it."""
        card = _card(titanic_shaped, "sex")
        note = re.search(r'class="coverage-note">([^<]+)<', card)
        assert note, "no coverage note"
        assert "levels shown" in note.group(1)
        assert "of non-missing rows" in note.group(1)

    def test_a_two_level_column_reads_correctly(self, titanic_shaped):
        card = _card(titanic_shaped, "sex")
        note = re.search(r'class="coverage-note">([^<]+)<', card).group(1)
        assert note.startswith("2 of 2 levels shown")
        assert "100%" in note

    def test_a_single_level_column_says_level_not_levels(self):
        out = profile(pd.DataFrame({"c": ["only"] * 300}), seed=0).html
        card = _card(out, "c")
        note = re.search(r'class="coverage-note">([^<]+)<', card)
        if note:
            assert " 1 level shown" in note.group(1)

    def test_an_all_missing_column_says_nothing_rather_than_zero_of_zero(self):
        out = profile(pd.DataFrame({"c": [None] * 200}), seed=0).html
        card = _card(out, "c")
        assert "0 of 0" not in card
