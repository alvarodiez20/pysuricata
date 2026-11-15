"""How far off an approximate figure may be, as an interval a reader can use.

The standing rule is that an approximate value is labelled approximate. A label
says *that* a number is uncertain and nothing about *how* uncertain, and the two
readings a bare `(≈)` invites are both wrong half the time: a median drawn from
20,000 of 60,000 values is good to about 1% of the spread, and a distinct count
from a 2,048-hash sketch is good to about 4% of itself. Same marker, a factor
of four between them, and neither figure was on the page.

Every interval here is **95%**, the level `ci_lo` / `ci_hi` already use for the
mean, and every one is derived from the mechanism that produced the estimate
rather than from the data's shape:

* **Distinct counts** -- KMV's estimator `(k - 1) / U_(k)` has relative standard
  error `1 / sqrt(k - 2)`, whatever the column holds.
* **Quantiles** -- the reservoir is a uniform sample without replacement, so the
  number of sampled values below the column's true p-quantile is hypergeometric.
  That turns into a pair of *order statistics* that bracket the truth, with no
  assumption about the distribution. No density estimate, no normality.
* **Scaled counts** -- a count of sampled values past a fence, scaled to the
  column, is a sampled proportion; Wilson's interval stays sensible at zero
  hits, where the textbook one collapses to `0 ± 0`.

What is deliberately **not** here: an interval for anything computed over every
value. `Min`, `Max`, `Mean`, `Zeros` and the missing counts are exact, and
attaching a margin to them would say the profiler is unsure of something it
counted. The mean's own interval is a different statement -- about the
population the column was drawn from, not about the profiler's arithmetic --
which is why it stays in the Statistics pane and off the card face.

Nothing in this module draws a random number or touches an accumulator's state,
so every function is safe to call from `finalize()` and from the renderers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: The two-sided 95% point of the standard normal.
Z95 = 1.959963984540054

#: The two-sided 95% point of Student's t, for 1 to 30 degrees of freedom.
#: A table rather than a formula because the series below is poor exactly here,
#: and rather than scipy because scipy is not a dependency and has no WASM-sized
#: reason to become one for thirty constants.
_T95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def t_critical_95(df: int) -> float:
    """The two-sided 95% critical value of Student's t.

    Args:
        df: Degrees of freedom, `n - 1` for a mean.

    Returns:
        The multiplier for a standard error. NaN below one degree of freedom,
        where no interval exists.

    The mean's interval used 1.96 at every sample size, with a branch for
    `n < 30` that assigned the same constant. At five values the right figure
    is 2.776, so the published interval was 29% too narrow precisely where a
    reader most needs to know the mean is loosely pinned.
    """
    if df < 1:
        return float("nan")
    if df <= len(_T95):
        return _T95[df - 1]
    # Cornish-Fisher around the normal point. Past 30 degrees of freedom the
    # two terms are within 2e-4 of the exact quantile.
    z = Z95
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df * df)
    )


def distinct_interval(
    estimate: int, k: int, exact: bool, ceiling: int
) -> tuple[int, int]:
    """95% interval on a KMV distinct count.

    Args:
        estimate: The published count, already clamped to the row count.
        k: The sketch size.
        exact: Whether the sketch counted rather than estimated.
        ceiling: The number of values the count was taken over. More distinct
            values than values is impossible, so the interval stops there.

    Returns:
        `(lo, hi)`, both inclusive. Equal to each other, and to `estimate`,
        when the count is exact.

    The interval is clipped on both sides by things the sketch *knows* rather
    than estimates. It cannot report an estimate until it holds `k` distinct
    hashes, so `k` is a hard floor; and `ceiling` is a hard roof. On a key
    column that makes the interval one-sided -- `[120,541, 126,000]` -- which
    is the honest shape: the sketch cannot certify that a key is unique, only
    that it is within a few percent of being so.
    """
    estimate = int(estimate)
    if exact or k <= 2 or estimate <= 0:
        return estimate, estimate
    margin = Z95 * estimate / math.sqrt(k - 2)
    lo = max(int(math.floor(estimate - margin)), min(k, estimate))
    hi = min(int(math.ceil(estimate + margin)), max(int(ceiling), estimate))
    return min(lo, estimate), max(hi, estimate)


def _finite_population_correction(sample_n: int, population: int) -> float:
    """The variance factor for sampling without replacement."""
    if population <= 1 or sample_n >= population:
        return 0.0
    return (population - sample_n) / (population - 1)


def quantile_interval(
    sorted_sample: Sequence[float], p: float, population: int
) -> tuple[float, float] | None:
    """95% interval on a quantile read from a uniform sample.

    Args:
        sorted_sample: The reservoir, ascending.
        p: The quantile, in `[0, 1]`.
        population: How many values the reservoir was drawn from.

    Returns:
        `(lo, hi)` -- two values *from the sample* that bracket the column's
        true quantile -- or None when the sample is the whole column and the
        quantile is therefore exact.

    The count of sampled values at or below the true p-quantile has mean `n p`
    and variance `n p (1 - p)` times the finite-population correction, so the
    true quantile sits between the order statistics that many ranks either
    side. The interval is asymmetric on skewed data because the data is: on a
    lognormal column the upper arm of the median's interval is longer than the
    lower one, and a symmetric `±` computed from a density estimate would
    flatten that.

    The correction matters more than it looks. A 20,000-value reservoir over a
    25,000-row column has seen 80% of it, and without the correction this would
    publish an interval 2.2x too wide.
    """
    n = len(sorted_sample)
    if n == 0 or population <= n:
        return None
    half = Z95 * math.sqrt(
        n * p * (1.0 - p) * _finite_population_correction(n, population)
    )
    centre = (n - 1) * p
    lo = max(0, int(math.floor(centre - half)))
    hi = min(n - 1, int(math.ceil(centre + half)))
    return float(sorted_sample[lo]), float(sorted_sample[hi])


def scaled_count_interval(hits: int, sample_n: int, population: int) -> tuple[int, int]:
    """95% interval on a sample count scaled up to the column.

    Args:
        hits: Sampled values that met the condition.
        sample_n: Size of the sample.
        population: Size of the column the sample stands in for.

    Returns:
        `(lo, hi)` in column rows. Both equal the count when the sample is the
        whole column.

    Wilson's score interval, on an effective sample size that folds in the
    finite-population correction. Wald's `p ± z sqrt(p (1 - p) / n)` gives
    `0 ± 0` at zero hits, which reads as "certainly none" from a sample that
    could not have ruled out a handful.

    Two bounds are arithmetic rather than statistics and are applied last: the
    sampled hits are in the column, so there are at least `hits`; and the
    sampled misses are in it too, so there are at most `population` minus those.
    """
    hits, sample_n, population = int(hits), int(sample_n), int(population)
    if sample_n <= 0 or population <= sample_n:
        return hits, hits
    fpc = _finite_population_correction(sample_n, population)
    n_eff = sample_n / fpc
    p_hat = hits / sample_n
    z2 = Z95 * Z95
    denom = 1.0 + z2 / n_eff
    centre = (p_hat + z2 / (2.0 * n_eff)) / denom
    half = (
        Z95
        * math.sqrt(p_hat * (1.0 - p_hat) / n_eff + z2 / (4.0 * n_eff * n_eff))
        / denom
    )
    lo = max(hits, int(math.floor((centre - half) * population)))
    hi = min(
        population - (sample_n - hits), int(math.ceil((centre + half) * population))
    )
    return lo, max(lo, hi)
