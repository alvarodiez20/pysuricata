"""Flat memory on text columns, and row signatures that are not re-hashed.

#95 and #33.

The memory test is the unusual one: it runs the accumulator in a **subprocess**
and reads peak RSS, because the claim being defended — bounded memory
regardless of dataset size — cannot be checked from inside the process doing
the allocating.

`Series.str.len()` was the single largest memory problem in the library. Nothing
was retained — `sys.getallocatedblocks()` stayed flat and the sketches stayed at
four counters — but peak RSS grew with the row count, and RSS is what a CI
runner limits.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators import CategoricalAccumulator
from pysuricata.accumulators.categorical import _string_lengths
from pysuricata.accumulators.sketches import KMV, RowKMV


class TestStringLengths:
    """Same answers as `.str.len()`, which is the point."""

    def test_it_matches_pandas(self):
        values = pd.Series(["a", "bb", "", "cccc", "dd"])
        assert _string_lengths(values).tolist() == values.str.len().tolist()

    def test_order_is_preserved(self):
        """The lengths feed a reservoir, so a reordering would change which
        ones are sampled and move `len_p90` for no reason."""
        values = pd.Series(["xxxxx", "y", "zzz", "y", "xxxxx"])
        assert _string_lengths(values).tolist() == [5, 1, 3, 1, 5]

    def test_repeated_values_are_measured_once_each(self):
        rng = np.random.default_rng(0)
        values = pd.Series(rng.choice(["alpha", "be", "gamma!"], 5_000))
        assert _string_lengths(values).tolist() == values.str.len().tolist()

    def test_an_empty_series(self):
        assert _string_lengths(pd.Series([], dtype=object)).size == 0

    def test_unicode_is_counted_in_characters(self):
        values = pd.Series(["café", "naïve", "日本語"])
        assert _string_lengths(values).tolist() == [4, 5, 3]


class TestLengthStatisticsAreUnchanged:
    @pytest.fixture
    def values(self):
        rng = np.random.default_rng(0)
        return np.array(
            ["x" * int(n) for n in rng.integers(1, 30, 20_000)], dtype=object
        )

    def test_the_average_length(self, values):
        acc = CategoricalAccumulator("c", seed=7)
        acc.update(values)
        expected = float(np.mean([len(v) for v in values]))
        assert acc.finalize().avg_len == pytest.approx(expected, rel=1e-9)

    def test_chunked_equals_unchunked(self, values):
        """The invariant the whole library rests on."""
        whole = CategoricalAccumulator("c", seed=7)
        whole.update(values)
        chunked = CategoricalAccumulator("c", seed=7)
        for start in range(0, len(values), 2_000):
            chunked.update(values[start : start + 2_000])
        assert chunked.finalize().avg_len == whole.finalize().avg_len
        assert chunked.finalize().len_p90 == whole.finalize().len_p90

    def test_the_p90_is_in_range(self, values):
        acc = CategoricalAccumulator("c", seed=7)
        acc.update(values)
        assert 1 <= acc.finalize().len_p90 <= 30


_MEMORY_PROBE = """
import resource, sys
import numpy as np
from pysuricata.accumulators import CategoricalAccumulator

rng = np.random.default_rng(0)
acc = CategoricalAccumulator("s", seed=1)
for _ in range({batches}):
    acc.update(rng.choice(["alpha", "beta", "gamma", "delta"], 65_536).astype(object))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def _peak_mb(batches: int) -> float:
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_MEMORY_PROBE.format(batches=batches))],
        capture_output=True,
        text=True,
        check=True,
    )
    raw = int(done.stdout.strip())
    # macOS reports bytes, Linux kilobytes.
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


class TestMemoryIsFlatInRows:
    """#95. Measured in subprocesses, because peak RSS cannot be observed
    honestly from inside the process doing the allocating.

    Deliberately **not** marked `slow`: that marker means "opt in with
    --run-slow" in `benchmarks/conftest.py`, and the job that passes the flag is
    currently skipped, so marking it would mean this never ran. Eight seconds is
    a fair price for the claim the library is positioned on.
    """

    def test_a_string_column_does_not_grow_with_rows(self):
        """Before the fix: 39 MB at 0.5M rows, 339 MB at 8.4M, on a column
        holding four distinct values."""
        small = _peak_mb(8)  # 524,288 rows
        large = _peak_mb(128)  # 8,388,608 rows
        growth = large - small
        assert growth < 40, (
            f"peak RSS grew {growth:.0f} MB from 0.5M to 8.4M rows "
            f"({small:.0f} -> {large:.0f} MB); it should be flat"
        )

    def test_the_probe_itself_reports_something_sane(self):
        assert _peak_mb(1) > 10


class TestRowSignaturesAreNotRehashed:
    """#33. `add_many` canonicalises integers through float64 so that 1 and 1.0
    are one value. That is right for a data column and wrong for a row
    signature, which already spans the full uint64 range."""

    def test_the_float64_round_trip_used_to_collide(self):
        """1,000 distinct hashes differing only in their low bits. float64 has
        53 bits of mantissa, so all of them landed on one value."""
        close = np.array([2**63 + i for i in range(1_000)], dtype=np.uint64)

        through_add_many = KMV(8_192)
        through_add_many.add_many(close)
        assert through_add_many.estimate() == 1

        through_offer = KMV(8_192)
        through_offer.offer_u64(close)
        assert through_offer.estimate() == 1_000

    def test_offer_u64_estimates_a_large_stream(self):
        rng = np.random.default_rng(0)
        hashes = rng.integers(0, 2**63, 200_000, dtype=np.int64).astype(np.uint64)
        sketch = KMV(8_192)
        sketch.offer_u64(hashes)
        assert sketch.estimate() == pytest.approx(len(np.unique(hashes)), rel=0.05)

    def test_an_empty_batch_is_a_no_op(self):
        sketch = KMV(256)
        sketch.offer_u64(np.empty(0, dtype=np.uint64))
        assert sketch.estimate() == 0

    def test_row_counts_stay_exact(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.integers(0, 10**6, 50_000)})
        rows = RowKMV()
        rows.update_from_pandas(frame)
        assert rows.rows == 50_000

    def test_distinct_rows_are_estimated_within_the_error_bound(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "a": rng.integers(0, 10**6, 100_000),
                "b": rng.standard_normal(100_000),
            }
        )
        rows = RowKMV()
        rows.update_from_pandas(frame)
        truth = len(frame.drop_duplicates())
        assert rows.kmv.estimate() == pytest.approx(truth, rel=0.05)

    def test_duplicates_are_found(self):
        rng = np.random.default_rng(0)
        unique_rows = pd.DataFrame({"a": rng.integers(0, 10**6, 20_000)})
        frame = pd.concat([unique_rows] * 4, ignore_index=True)
        rows = RowKMV()
        rows.update_from_pandas(frame)
        duplicates, pct = rows.approx_duplicates()
        assert duplicates == pytest.approx(60_000, rel=0.05)
        assert pct == pytest.approx(75.0, rel=0.05)

    def test_sequential_ids_do_not_look_like_duplicates(self):
        """The shape the float64 round-trip was worst on: rows whose hashes
        differ in their low bits."""
        frame = pd.DataFrame({"id": np.arange(100_000)})
        rows = RowKMV()
        rows.update_from_pandas(frame)
        duplicates, _ = rows.approx_duplicates()
        assert duplicates < 5_000


def test_getrusage_is_available():
    """The memory test is worthless if this silently returns zero."""
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 0
