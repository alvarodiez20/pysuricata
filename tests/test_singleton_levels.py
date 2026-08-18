"""#297 — how many of a column's levels occur exactly once.

A 147-level column is two very different columns depending on the answer: a
handful of crowded levels, or a drift of near-singletons. Only the second is a
column where a top-values ranking would have meant anything, and the distinct
count alone cannot separate them.

Neither sketch already in `CategoricalAccumulator` can answer it. Misra-Gries
is gated off entirely on high-cardinality columns and its counts are lower
bounds in any case, so a tracked count of 1 is not evidence of a singleton;
KMV keeps values and not counts. `SingletonCounter` counts exactly inside its
capacity and **refuses** outside it, because a singleton count is a claim about
the rarest thing in a column and that is precisely what a frequency sketch is
worst at.

The invariant that matters most here is the one `CLAUDE.md` names as most
likely to break: a chunked result must equal an unchunked one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.accumulators.sketches import SingletonCounter

TITANIC = "docs/assets/titanic.csv"


def _truth(series: pd.Series) -> tuple[int, int]:
    """`(levels, singletons)` straight from pandas."""
    counts = series.dropna().astype(str).value_counts()
    return len(counts), int((counts == 1).sum())


class TestItCountsExactly:
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(list("aaabbc"), id="mixed"),
            pytest.param(list("abcdef"), id="all-singletons"),
            pytest.param(["x"] * 40, id="one-level"),
            pytest.param([], id="empty"),
        ],
    )
    def test_against_a_plain_count(self, values):
        counter = SingletonCounter(capacity=64)
        counter.add_many(values)
        counts = pd.Series(values, dtype=object).value_counts()

        assert counter.levels() == len(counts)
        assert counter.singletons() == int((counts == 1).sum())

    def test_a_weighted_add_matches_the_same_values_one_at_a_time(self):
        """The accumulator takes the vectorised `value_counts` path on a
        pandas chunk and the per-value path on its fallback. The two must not
        disagree."""
        one_at_a_time = SingletonCounter(capacity=64)
        one_at_a_time.add_many(["a", "a", "a", "b", "c"])
        weighted = SingletonCounter(capacity=64)
        weighted.add("a", w=3)
        weighted.add("b", w=1)
        weighted.add("c", w=1)

        assert weighted.singletons() == one_at_a_time.singletons() == 2
        assert weighted.levels() == one_at_a_time.levels() == 3


class TestItRefusesRatherThanEstimating:
    def test_past_capacity_the_answer_is_unknown_and_not_zero(self):
        counter = SingletonCounter(capacity=4)
        counter.add_many(["a", "b", "c", "d", "e"])

        assert counter.exact is False
        assert counter.singletons() is None
        assert counter.levels() is None

    def test_exactly_at_capacity_is_still_exact(self):
        counter = SingletonCounter(capacity=4)
        counter.add_many(["a", "b", "c", "d"])

        assert counter.exact is True
        assert counter.singletons() == 4

    def test_a_repeat_at_capacity_does_not_trip_it(self):
        """Only a *new* level past the capacity turns it off. Counting more of
        a level already held costs nothing."""
        counter = SingletonCounter(capacity=2)
        counter.add_many(["a", "b", "a", "b", "a"])

        assert counter.exact is True
        assert counter.singletons() == 0


class TestMergingIsExactAndOrderIndependent:
    """`CLAUDE.md`: accumulators must be mergeable and order-independent where
    the statistic allows it, and chunked results must equal unchunked ones.
    This statistic allows it exactly."""

    def test_two_halves_equal_the_whole(self):
        values = list("aabbbcdefgh")
        whole = SingletonCounter(capacity=64)
        whole.add_many(values)

        left = SingletonCounter(capacity=64)
        left.add_many(values[:5])
        right = SingletonCounter(capacity=64)
        right.add_many(values[5:])
        left.merge(right)

        assert left.singletons() == whole.singletons()
        assert left.levels() == whole.levels()

    def test_a_level_split_across_the_seam_is_not_two_singletons(self):
        """The case that makes this worth asserting: `a` appears once in each
        half and is a singleton in neither the whole nor the merge."""
        left = SingletonCounter(capacity=64)
        left.add_many(["a", "b"])
        right = SingletonCounter(capacity=64)
        right.add_many(["a", "c"])
        left.merge(right)

        assert left.singletons() == 2  # b and c, not a
        assert left.levels() == 3

    def test_the_merge_order_does_not_matter(self):
        one = SingletonCounter(capacity=64)
        one.add_many(list("aabc"))
        two = SingletonCounter(capacity=64)
        two.add_many(list("cdde"))

        forward = SingletonCounter(capacity=64)
        forward.add_many(list("aabc"))
        forward.merge(two)
        backward = SingletonCounter(capacity=64)
        backward.add_many(list("cdde"))
        backward.merge(one)

        assert forward.singletons() == backward.singletons()
        assert forward.levels() == backward.levels()

    def test_merging_an_inexact_counter_makes_the_result_inexact(self):
        """Otherwise a merge could launder an unknown into a number."""
        over = SingletonCounter(capacity=2)
        over.add_many(["a", "b", "c"])
        under = SingletonCounter(capacity=2)
        under.add_many(["a"])
        under.merge(over)

        assert under.singletons() is None

    def test_a_merge_that_would_exceed_the_capacity_turns_it_off(self):
        left = SingletonCounter(capacity=3)
        left.add_many(["a", "b"])
        right = SingletonCounter(capacity=3)
        right.add_many(["c", "d"])
        left.merge(right)

        assert left.singletons() is None


class TestTheReportSaysIt:
    @pytest.fixture(scope="class")
    def titanic(self) -> pd.DataFrame:
        return pd.read_csv(TITANIC)

    @pytest.fixture(scope="class")
    def html(self, titanic) -> str:
        return profile(titanic, seed=0).html

    @pytest.mark.parametrize("column", ["Cabin", "Ticket"])
    def test_the_figure_matches_pandas(self, titanic, html, column):
        levels, singletons = _truth(titanic[column])

        assert f"{singletons:,} of {levels:,} levels occur exactly once" in html

    def test_a_three_level_column_is_unaffected(self, html):
        """#297's fourth acceptance line. `Embarked` has three levels, none of
        them a singleton, and gets a bar chart and no clause."""
        card = html[html.find('id="col_Embarked"') :][:6000]

        assert "levels occur exactly once" not in card

    def test_the_clause_is_silent_when_no_level_is_a_singleton(self):
        """Zero is the ordinary case; a clause saying so on every column is
        noise, not information."""
        frame = pd.DataFrame({"a": [f"v{i}" for i in range(60)] * 2})

        assert "occur exactly once" not in profile(frame, seed=0).html


class TestChunkingDoesNotChangeTheAnswer:
    """The invariant `benchmarks/accuracy.py` exists for, exercised through the
    whole engine rather than the sketch alone."""

    @pytest.mark.parametrize("chunk_size", [200, 1_000, 100_000])
    def test_the_count_is_the_same_at_every_chunk_size(self, chunk_size):
        rng = np.random.default_rng(0)
        # 300 levels, deliberately mixed: a third are singletons and the rest
        # repeat, so the answer is neither 0 nor the level count.
        values = [f"v{i}" for i in range(100)] + [
            f"w{i}" for i in rng.integers(0, 200, 900)
        ]
        frame = pd.DataFrame({"a": values})
        levels, singletons = _truth(frame["a"])

        html = profile(frame, seed=0, chunk_size=chunk_size).html

        assert f"{singletons:,} of {levels:,} levels occur exactly once" in html
