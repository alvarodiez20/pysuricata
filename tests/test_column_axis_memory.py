"""What a column costs, pinned (#207).

Bounded memory was a claim about rows and false about columns: 20,000 x 600
peaked at 980 MB while 1,000,000 x 14 -- *more* cells -- peaked at 344 MB. The
bulk of it was one line. The reservoir held its sample as a `list[float]`, and
a Python float is 24 bytes plus 8 for the pointer that finds it, so 20,000
values that are 160 KB of data occupied 638 KB of heap. Times 600 columns, that
one boxing decision was 374 MB.

These are ratchets in the sense the layout tests are: **each fails in both
directions**. A regression grows them; a genuine win asks you to lower the
baseline here, so the saving cannot be quietly respent.

Everything here is measured on data structures, never on wall clock or on the
resident set, both of which depend on the machine. `python -m benchmarks.columns`
is where the end-to-end figures live.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest

from pysuricata.accumulators.numeric import NumericAccumulator
from pysuricata.accumulators.sketches import ReservoirSampler

#: What one filled numeric column holds, in KB, and how far it may drift before
#: this test asks to be looked at. Measured at 284 KB with a 20,000-value
#: reservoir: 160 KB of sample, 32 KB of acceptance schedule, 83 KB of KMV and
#: the rest small. Before #207 the same accumulator was ~765 KB.
_ACCUMULATOR_KB = 284
_TOLERANCE_KB = 40


def _deep_size(obj, seen=None) -> int:
    """Bytes reachable from `obj`, counting an array's buffer rather than the
    88-byte header `sys.getsizeof` reports for it."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    size = sys.getsizeof(obj, 0)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += _deep_size(key, seen) + _deep_size(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            size += _deep_size(value, seen)
    elif hasattr(obj, "__dict__"):
        size += _deep_size(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            size += _deep_size(getattr(obj, slot, None), seen)
    return size


def _filled(k: int = 20_000) -> ReservoirSampler:
    sampler = ReservoirSampler(k, rng=np.random.default_rng(0))
    sampler.add_many(np.random.default_rng(1).normal(size=k))
    return sampler


class TestTheSampleIsNotBoxed:
    def test_a_value_costs_eight_bytes(self):
        """The whole of #207's cheap half, as one number."""
        sampler = _filled()
        assert sampler.values().nbytes / len(sampler.values()) == 8

    def test_the_buffer_is_float64_rather_than_a_list(self):
        assert isinstance(_filled()._buf, np.ndarray)
        assert _filled()._buf.dtype == np.float64

    def test_a_list_of_the_same_values_would_cost_four_times_as_much(self):
        """The comparison the fix rests on, measured rather than asserted."""
        sampler = _filled()
        as_list = sampler.values().tolist()
        boxed = sys.getsizeof(as_list) + sum(sys.getsizeof(v) for v in as_list)
        assert boxed > 4 * sampler.values().nbytes


class TestAShortColumnDoesNotPayForALongOne:
    """A 600-column frame of 20 rows must not allocate 600 full reservoirs."""

    @pytest.mark.parametrize("n", [1, 20, 500])
    def test_the_buffer_grows_to_the_data_not_to_k(self, n):
        sampler = ReservoirSampler(20_000, rng=np.random.default_rng(0))
        sampler.add_many(np.arange(n, dtype=float))
        assert sampler._buf.size <= max(1024, 2 * n)

    def test_it_still_caps_at_k(self):
        sampler = ReservoirSampler(500, rng=np.random.default_rng(0))
        sampler.add_many(np.arange(100_000, dtype=float))
        assert sampler._buf.size == 500


class TestOneColumnsFootprint:
    def test_a_filled_numeric_column_stays_where_it_was_put(self):
        accumulator = NumericAccumulator("c")
        accumulator.update(np.random.default_rng(0).normal(size=20_000))
        kb = _deep_size(accumulator) / 1024
        assert abs(kb - _ACCUMULATOR_KB) <= _TOLERANCE_KB, (
            f"a numeric column now holds {kb:,.0f} KB against a baseline of "
            f"{_ACCUMULATOR_KB} KB. Growth is a regression on the column axis "
            f"(#207); a saving is welcome, and wants the baseline lowered here "
            f"so it cannot be spent twice."
        )

    def test_the_reservoir_is_the_largest_part_of_it(self):
        """Which is why it was the thing to fix, and stays worth watching."""
        accumulator = NumericAccumulator("c")
        accumulator.update(np.random.default_rng(0).normal(size=20_000))
        parts = {k: _deep_size(v) for k, v in vars(accumulator).items()}
        assert max(parts, key=parts.get) == "_sample"


class TestTheArrayDidNotCostCorrectness:
    def test_values_cannot_be_written_through(self):
        """The view is the live reservoir. A caller that sorted it in place
        would be reordering the sample every later statistic reads."""
        with pytest.raises(ValueError):
            _filled().values()[0] = 1.0

    def test_a_summary_still_compares_equal_to_itself(self):
        """`sample_vals` is an array, so the dataclass-generated `__eq__` would
        raise rather than answer. `NumericAccumulator.finalize` is compared
        whole by the checkpoint round-trip and by `reset`."""
        accumulator = NumericAccumulator("c", seed=0)
        accumulator.update(np.random.default_rng(0).normal(size=5_000))
        assert accumulator.finalize() == accumulator.finalize()

    def test_a_pickled_accumulator_finalizes_the_same(self):
        accumulator = NumericAccumulator("c", seed=0)
        accumulator.update(np.random.default_rng(0).normal(size=5_000))
        assert pickle.loads(pickle.dumps(accumulator)).finalize() == (
            accumulator.finalize()
        )

    def test_two_summaries_over_different_data_are_not_equal(self):
        """The mirror of the above: an `__eq__` that swallowed the array
        comparison would make every summary equal to every other."""
        first, second = NumericAccumulator("c", seed=0), NumericAccumulator("c", seed=0)
        first.update(np.random.default_rng(0).normal(size=5_000))
        second.update(np.random.default_rng(1).normal(size=5_000))
        assert first.finalize() != second.finalize()

    def test_the_sample_is_unchanged_by_the_representation(self):
        """Same seed, same stream, same values as a list would have held."""
        sampler = ReservoirSampler(500, rng=np.random.default_rng(7))
        sampler.add_many(np.arange(50_000, dtype=float))
        assert sorted(sampler.values().tolist()) == sorted(
            float(v) for v in sampler.values()
        )
        assert len(set(sampler.values().tolist())) == 500
