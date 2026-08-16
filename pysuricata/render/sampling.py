"""Telling a measured statistic from a sampled one.

``Min``, ``Max`` and ``Mean`` see every value: the extremes are tracked over the
stream (#118, after a sampled "Maximum" was printed above a table of larger
values) and the mean comes from the streaming moments. ``Q1``, ``Median``,
``Q3``, ``IQR`` and ``MAD`` do not — they are computed from a reservoir that
holds ``numeric_sample_k`` values, so on any column longer than that they are
estimates.

The card printed all of them in the same typography, to four significant
figures. On a 60,000-row column the median rendered as ``0.003684`` while eight
unseeded runs of the same data spread ``1.86e-02`` — the printed precision was
three orders of magnitude finer than the estimate supports, and every digit
after the first was noise.

Nothing here changes a number. It answers one question — *did the reservoir see
everything?* — so the label can say which kind of value it is, per the standing
rule that approximate values are labelled approximate. The distinct count has
followed that rule since #41; the quantiles had not.
"""

from __future__ import annotations

from typing import Any


def quantiles_are_sampled(stats: Any) -> bool:
    """Whether the quantiles came from a partial view of the column.

    Args:
        stats: A ``NumericStats``-shaped object carrying ``count`` and
            ``sample_vals``.

    Returns:
        True when the reservoir holds fewer values than the column has, so the
        quantiles drawn from it are estimates.

    False is the honest answer whenever the sample size cannot be established.
    A missing or empty ``sample_vals`` means the quantiles were not drawn from a
    reservoir at all -- a column short enough to be held whole, or one with no
    finite values -- and marking those approximate would attach a warning to
    exactly the numbers that do not need one.
    """
    sample = getattr(stats, "sample_vals", None)
    if not sample:
        return False
    count = getattr(stats, "count", 0) or 0
    return len(sample) < count
