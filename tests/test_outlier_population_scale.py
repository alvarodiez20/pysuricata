"""Outlier counts must describe the column, not the reservoir (#327).

`outliers_iqr_est` was a count made inside the reservoir and published beside
`count`, which covers every row. Nothing in the suite compared it to truth, so
a field that was 49x low at 1M rows passed every test. These are the contract
tests that fail on the pre-fix code: scale invariance across `n`, and the
percentage the card and the quality flag are built on.

`n` is chosen to straddle `numeric_sample_size` (20,000) -- the defect lives
entirely above that threshold and is invisible below it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysuricata

#: The reservoir is a sample, so the estimate carries sampling error. Measured
#: spread over these cases is under 4%; 15% leaves room for the tail without
#: admitting the 60-98% error the unscaled count had.
_TOLERANCE = 0.15


def _exact_iqr_outliers(values: np.ndarray) -> int:
    """The IQR fence over every value, which is what the estimate estimates."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


def _lognormal(n: int) -> np.ndarray:
    """A column with a real tail. A normal one has too few outliers to divide by."""
    return np.random.default_rng(0).lognormal(0.0, 1.0, n)


@pytest.mark.parametrize("n", [5_000, 20_000, 50_000, 200_000])
def test_outlier_estimate_tracks_truth_at_every_size(n: int) -> None:
    """The ratio to truth must not drift with `n`.

    A ratio that tracks `numeric_sample_size / n` is the signature of the bug:
    it was 0.396 at 50,000 and 0.101 at 200,000.
    """
    values = _lognormal(n)
    summary = pysuricata.summarize(pd.DataFrame({"x": values}))
    reported = summary["columns"]["x"]["outliers_iqr_est"]
    exact = _exact_iqr_outliers(values)

    assert exact > 0, "fixture must actually have outliers, or this proves nothing"
    ratio = reported / exact
    assert abs(ratio - 1.0) < _TOLERANCE, (
        f"n={n:,}: reported {reported:,} against a true {exact:,} (ratio {ratio:.3f}). "
        "A ratio tracking sample_size/n means the reservoir count is unscaled."
    )


def test_outlier_count_is_exact_when_the_sample_holds_everything() -> None:
    """Below the reservoir size there is nothing to estimate, so do not.

    Guards the other direction: a scale factor applied where it does not belong
    would make a small column's exact count approximate.
    """
    values = _lognormal(5_000)
    summary = pysuricata.summarize(pd.DataFrame({"x": values}))
    assert summary["columns"]["x"]["outliers_iqr_est"] == _exact_iqr_outliers(values)


def test_outlier_percentage_reaches_the_report_at_scale() -> None:
    """The card's percentage, and the flag keyed off it, must survive sampling.

    At 200,000 rows the unscaled count put this column at 0.2%, under the 1%
    threshold, so the `Many outliers` flag did not render -- on a column that
    is over 10% outliers.
    """
    values = _lognormal(200_000)
    exact_pct = _exact_iqr_outliers(values) / len(values) * 100.0
    assert exact_pct > 1.0, "fixture must clear the flag threshold"

    summary = pysuricata.summarize(pd.DataFrame({"x": values}))
    col = summary["columns"]["x"]
    reported_pct = col["outliers_iqr_est"] / col["count"] * 100.0

    assert abs(reported_pct - exact_pct) / exact_pct < _TOLERANCE, (
        f"card would print {reported_pct:.1f}% for a column that is "
        f"{exact_pct:.1f}% outliers"
    )
    assert reported_pct > 1.0, "the outlier flag would not fire"
