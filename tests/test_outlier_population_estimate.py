"""Outlier counts are estimates for the column, not counts inside the reservoir.

The IQR fence is fitted over the reservoir sample and the crossings counted
there. That count used to be published unscaled as ``outliers_iqr_est``, beside
a ``count`` that means the whole column: two scales in one struct. The ratio
between them was exactly ``numeric_sample_size / n``, so at a million rows a
column that is 7.7% outliers reported 1,495 of them, or 0.2% (#327).

The percentage was the part that hurt. ``render/card_base.py`` divides the
outlier count by ``stats.count`` to key the ``many outliers`` flag, so the flag
went quiet as the frame grew: it fired on a 10,000-row frame and stayed silent
on the same distribution at a million, which is failing silent on exactly the
datasets a profiler is for.

These cases straddle the reservoir size (20,000 by default) rather than sitting
at one shape, because everything below it is unscaled by definition and passes
whether or not the fix is in place. The two sizes above it are where the bug
lived, and a fixture that only used the small one would report "absent".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.accumulators.numeric import NumericAccumulator

#: Below, at, and well above the default reservoir.
SIZES = (5_000, 20_000, 60_000, 200_000)


def _lognormal(n: int) -> pd.DataFrame:
    """A heavy right tail, so the fence has something real to catch."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"v": rng.lognormal(0, 1, n)})


def _true_iqr_outliers(x: np.ndarray) -> int:
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())


def _true_mad_outliers(x: np.ndarray) -> int:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        return 0
    return int((np.abs(0.6745 * (x - med) / mad) > 3.5).sum())


@pytest.mark.parametrize("n", SIZES)
def test_the_iqr_estimate_tracks_the_exact_count(n):
    frame = _lognormal(n)
    exact = _true_iqr_outliers(frame["v"].to_numpy())

    est = summarize(frame, seed=0)["columns"]["v"]["outliers_iqr_est"]

    # 10% covers reservoir noise at every size here; the bug was a factor of
    # 50, so this does not need to be tight to catch it coming back.
    assert est == pytest.approx(exact, rel=0.10), (
        f"{n:,} rows: estimated {est:,} against an exact {exact:,}, a ratio of "
        f"{est / max(1, exact):.3f}"
    )


@pytest.mark.parametrize("n", SIZES)
def test_the_modified_zscore_estimate_tracks_the_exact_count(n):
    """Same shape of arithmetic, same slip, and it plateaued the same way."""
    frame = _lognormal(n)
    exact = _true_mad_outliers(frame["v"].to_numpy())

    est = summarize(frame, seed=0)["columns"]["v"]["outliers_mod_zscore_est"]

    assert est == pytest.approx(exact, rel=0.10), (
        f"{n:,} rows: estimated {est:,} against an exact {exact:,}"
    )


def test_the_estimate_does_not_collapse_as_the_frame_grows():
    """The signature of the bug, stated directly.

    An unscaled count plateaus at the reservoir's own outlier count whatever
    the row count, so the *share* falls as 1/n. Holding the distribution fixed
    and growing the frame, the share has to stay put.
    """
    shares = []
    for n in (10_000, 100_000, 400_000):
        frame = _lognormal(n)
        col = summarize(frame, seed=0)["columns"]["v"]
        shares.append(col["outliers_iqr_est"] / col["count"])

    assert max(shares) - min(shares) < 0.01, (
        f"the outlier share moves with the row count: {shares}"
    )


def test_the_many_outliers_flag_still_fires_on_a_large_frame():
    """The user-visible half. The flag is keyed off the percentage, so an
    unscaled numerator over a population denominator silenced it above the
    reservoir size: the warning disappeared as the data got big."""
    from pysuricata.render.card_base import QualityAssessor

    rng = np.random.default_rng(0)
    # 10% of the column parked far past any fence.
    body = rng.normal(0, 1, 900_000)
    tail = rng.normal(500, 1, 100_000)
    frame = pd.DataFrame({"v": np.concatenate([body, tail])})

    stats = _finalize(frame)
    flags = QualityAssessor().assess_numeric_quality(stats)

    assert flags.many_outliers, (
        "a column that is 10% outliers at a million rows did not raise the flag"
    )


def _finalize(frame: pd.DataFrame):
    """Run the column through the accumulator the report uses."""
    acc = NumericAccumulator("v", seed=0)
    acc.update(frame["v"].to_numpy())
    return acc.finalize()


@pytest.mark.parametrize("n", (5_000, 60_000))
def test_the_sample_counts_are_kept_beside_the_estimates(n):
    """The fence pane lists the sampled rows, so it needs the sampled count.

    Below the reservoir the two are the same number; above it the sample count
    is the smaller one, and neither is allowed to go missing.
    """
    stats = _finalize(_lognormal(n))

    assert stats.outliers_iqr_sample > 0
    assert stats.outliers_iqr >= stats.outliers_iqr_sample
    if n <= 20_000:
        assert stats.outliers_iqr == stats.outliers_iqr_sample
    else:
        assert stats.outliers_iqr > stats.outliers_iqr_sample
