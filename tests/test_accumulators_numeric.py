import math

import numpy as np

from pysuricata.accumulators.numeric import NumericAccumulator


def test_a_single_finite_value_takes_the_short_arm_of_percentile():
    """`percentile` divides by `n - 1`, so n == 1 has its own arm and every
    quantile is that one value. Reached by a column whose other rows are all
    missing, which is common enough in real data and was untested."""
    acc = NumericAccumulator("x")
    acc.update(np.array([float("nan"), 7.5, float("nan")]))
    s = acc.finalize()
    assert s.count == 1
    assert (s.min, s.q1, s.median, s.q3, s.max) == (7.5, 7.5, 7.5, 7.5, 7.5)
    assert s.iqr == 0.0


def test_numeric_accumulator_basic_stats():
    acc = NumericAccumulator("x")
    arr = np.array([1.0, 2.0, 3.0, float("nan"), float("inf"), -1.0, 0.0])
    acc.update(arr)
    s = acc.finalize()
    assert s.name == "x"
    # count excludes NaN/inf in moments but tracks missing/inf
    assert s.count >= 4
    assert s.missing >= 1
    assert s.inf >= 1
    assert math.isfinite(s.mean)
    assert math.isfinite(s.std) or math.isnan(s.std)
    assert s.min <= s.max
    # zeros/negatives tracked
    assert s.zeros >= 1
    assert s.negatives >= 1
