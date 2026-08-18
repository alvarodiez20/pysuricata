"""Misra-Gries counts are lower bounds, and the report has to say so.

The sketch keeps `top_k_size` counters, 50 by default. Below that the counts
are exact. Above it every new value evicts weight from every counter, so a
reported count can only undercount, and the *ranking* goes with it: on a
near-uniform column of 1,000 categories over a million rows the true top value
held 1,107 occurrences and the report named a different value with 35 (#328).

The flag meant to warn about this read `len(top_items) >= top_k_size`, which
gets the dangerous case exactly backwards. Eviction *deletes* counters, so the
list shrinks below the budget precisely when the sketch is under most pressure:
that same column published nine items and `approx=False`. The report claimed
exactness where it was least entitled to it.

`approx` is now derived from the sketch's own decrement mass, which is the only
thing that knows. That mass is also the error bound Misra-Gries guarantees:

    true_count(x) ∈ [reported(x), reported(x) + decremented]

so it is published as `top_items_uncertainty` rather than left implicit.

The sizes here sit at 49, 50, 51 and 500 distinct values because the defect
lives entirely in the gap between 50 and 51: a fixture on either side alone
reports "absent", and absent reads as fixed.
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.accumulators.sketches import MisraGries

#: The default `top_k_size`. Stated as the number the tests straddle rather
#: than imported, so a change to the default fails here and gets looked at.
TOP_K = 50


def _frame(rows: int, distinct: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = np.array([f"v{i}" for i in range(distinct)])[
        rng.integers(0, distinct, rows)
    ]
    return pd.DataFrame({"c": values})


class TestTheSketchKnowsWhetherItIsExact:
    def test_no_eviction_means_exact(self):
        mg = MisraGries(k=4)
        for value in ["a", "a", "b", "c"]:
            mg.add(value)

        assert mg.is_exact
        assert mg.error_bound == 0

    def test_eviction_is_counted(self):
        mg = MisraGries(k=2)
        for value in ["a", "b", "c", "d"]:
            mg.add(value)

        assert not mg.is_exact
        assert mg.error_bound > 0

    def test_the_batch_path_counts_it_too(self):
        """`add_many` is the path a chunked run takes. A bound that missed it
        would read exact on almost every real column."""
        mg = MisraGries(k=2)
        mg.add_many(["a", "b", "c", "d", "e"])

        assert not mg.is_exact, "add_many evicted without recording the mass"

    def test_a_merge_carries_both_sides_of_the_error(self):
        left, right = MisraGries(k=2), MisraGries(k=2)
        for value in ["a", "b", "c", "d"]:
            right.add(value)
        before = right.error_bound

        left.merge(right)

        assert left.error_bound >= before


@pytest.mark.parametrize("distinct", [TOP_K - 1, TOP_K, TOP_K + 1, TOP_K * 10])
def test_approx_is_true_exactly_when_a_count_is_a_lower_bound(distinct):
    """The contract, stated both ways round: `approx is False` has to mean
    every published count is the truth."""
    frame = _frame(20_000, distinct)
    truth = collections.Counter(frame["c"])

    column = summarize(frame, seed=0)["columns"]["c"]
    reported = dict(column["top_items"] or [])
    exact = all(count == truth[value] for value, count in reported.items())

    if column["approx"]:
        # Approximate is allowed to be exact; the flag is a warning, not a
        # promise of error. The reverse is what must never happen.
        return
    assert exact, (
        f"{distinct} distinct values: approx=False while a count is a lower "
        f"bound. Worst gap "
        f"{max(truth[v] - c for v, c in reported.items())}"
    )


@pytest.mark.parametrize(
    ("rows", "distinct"), [(5_000, 100), (200_000, 500), (1_000_000, 1_000)]
)
def test_the_uncertainty_brackets_the_truth(rows, distinct):
    """The Misra-Gries guarantee, exercised rather than cited."""
    frame = _frame(rows, distinct)
    truth = collections.Counter(frame["c"])

    column = summarize(frame, seed=0)["columns"]["c"]
    bound = column["top_items_uncertainty"]

    assert column["approx"], "a column this wide cannot have exact counters"
    assert bound > 0
    for value, count in column["top_items"] or []:
        assert count <= truth[value] <= count + bound, (
            f"{value}: reported {count:,}, true {truth[value]:,}, bound {bound:,}"
        )


def test_an_exact_column_publishes_a_zero_bound():
    """A range on an exact count would be its own kind of lie."""
    column = summarize(_frame(20_000, TOP_K - 1), seed=0)["columns"]["c"]

    assert column["top_items_uncertainty"] == 0
    assert not column["approx"]


def test_the_dominant_share_is_not_overstated_by_lossy_counters():
    """`most_common_ratio` divided one decremented count by the sum of the
    decremented counts. Both shrink, the denominator faster, so the ratio grew
    as the counters lost information: 0.132 for a value whose true share was
    0.0011. Over the exact row count it can only understate."""
    rows, distinct = 1_000_000, 1_000
    frame = _frame(rows, distinct)
    truth = collections.Counter(frame["c"])
    true_share = truth.most_common(1)[0][1] / rows

    ratio = summarize(frame, seed=0)["columns"]["c"]["most_common_ratio"]

    assert ratio <= true_share + 1e-9, (
        f"most_common_ratio {ratio:.4f} overstates a true share of {true_share:.4f}"
    )


def test_a_numeric_column_makes_the_same_promise():
    """`top_values` comes from the same sketch and was covered by an `approx`
    that never consulted it.

    1,000 distinct values, not 5,000: above a coverage floor the numeric path
    switches top-k off entirely and publishes nothing, so a wider frame would
    test that the bound on an absent table is zero -- true, and not the point.
    """
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"n": rng.integers(0, 1_000, 200_000)})

    column = summarize(frame, seed=0)["columns"]["n"]

    assert column["top_values"], "top-k was switched off; pick a narrower frame"
    assert column["approx"]
    assert column["top_values_uncertainty"] > 0


def test_the_counter_budget_did_not_grow():
    """The cheap fix for the counts is a bigger `k`, and it is the wrong one:
    memory is `top_k * 100 B` per column, so 8,192 counters across 600 columns
    is ~490 MB of counters alone, against the very axis #207 is about. The
    honest bound is what ships; the budget stays put."""
    from pysuricata.accumulators.config import CategoricalConfig, NumericConfig

    assert CategoricalConfig().top_k_size == TOP_K
    assert NumericConfig().top_k_size == TOP_K
