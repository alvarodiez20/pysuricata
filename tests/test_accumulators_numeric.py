import math
import numpy as np

from pysuricata.accumulators.numeric import NumericAccumulator


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

