"""The KMV pre-filter, the top-k gate, and the chunk-size band.

Three Phase-1 changes that trade work for nothing observable — except the top-k
gate, which deliberately removes output. What each must not change:

- the pre-filter must leave every distinct estimate bit-identical;
- the gate must keep the table wherever it carries information, drop it
  wherever it would list singletons, and reach the same verdict however the
  column was chunked;
- the auto-chosen chunk size must stay in a band, so the heuristic cannot drift
  back to a value that makes the sketch merges superlinear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators.numeric import NumericAccumulator, should_track_top_k
from pysuricata.accumulators.sketches import KMV
from pysuricata.api import ComputeOptions, ProfileConfig
from pysuricata.config import EngineConfig
from pysuricata.io.base import ChunkingConfig


def _kmv_estimate_without_prefilter(values: np.ndarray, k: int, n_chunks: int) -> int:
    """Reference KMV: merge every hash, never reject against the threshold."""
    from pysuricata.accumulators.sketches import _hash_numeric_array

    retained = np.empty(0, dtype=np.uint64)
    for chunk in np.array_split(values, n_chunks):
        hashes = np.unique(_hash_numeric_array(np.ascontiguousarray(chunk)))
        retained = np.union1d(retained, hashes) if retained.size else hashes
        retained = retained[:k]
    n = retained.size
    if n < k:
        return n
    t = (int(retained[-1]) + 1) / 2**64
    return max(n, int(round((k - 1) / t))) if t > 0 else n


class TestKmvPrefilterChangesNothing:
    @pytest.mark.parametrize("n,k", [(50_000, 1024), (300_000, 2048), (300_000, 4096)])
    def test_estimate_matches_the_unfiltered_merge(self, n, k):
        values = np.random.default_rng(0).standard_normal(n)
        sketch = KMV(k)
        for chunk in np.array_split(values, 5):
            sketch.add_many(np.ascontiguousarray(chunk))
        assert sketch.estimate() == _kmv_estimate_without_prefilter(values, k, 5)

    def test_estimate_is_still_independent_of_chunking(self):
        values = np.random.default_rng(1).standard_normal(200_000)
        estimates = []
        for n_chunks in (1, 7, 113):
            sketch = KMV(2048)
            for chunk in np.array_split(values, n_chunks):
                sketch.add_many(np.ascontiguousarray(chunk))
            estimates.append(sketch.estimate())
        assert estimates[1] == estimates[0]
        assert estimates[2] == estimates[0]

    def test_low_cardinality_stays_exact(self):
        """Below k distinct, the answer must still be a count, not an estimate."""
        sketch = KMV(2048)
        for _ in range(20):
            sketch.add_many(np.arange(300, dtype=float))
        assert sketch.estimate() == 300

    def test_repeats_after_the_sketch_fills_do_not_inflate(self):
        """The rejected majority must not re-enter through a later batch."""
        values = np.arange(100_000, dtype=float)
        sketch = KMV(1024)
        sketch.add_many(values)
        first = sketch.estimate()
        for _ in range(5):
            sketch.add_many(values)
        assert sketch.estimate() == first


class TestTopKGateRule:
    def test_an_empty_column_stays_enabled(self):
        assert should_track_top_k(0.0, 0, 50) is True

    def test_fewer_distinct_than_counters_is_always_kept(self):
        assert should_track_top_k(12.0, 200_000, 50) is True
        assert should_track_top_k(50.0, 200_000, 50) is True

    def test_a_column_the_counters_can_cover_is_kept(self):
        assert should_track_top_k(500.0, 200_000, 50) is True

    def test_a_column_of_singletons_is_dropped(self):
        assert should_track_top_k(47_810.0, 200_000, 50) is False
        assert should_track_top_k(200_923.0, 200_000, 50) is False

    def test_the_boundary_is_two_percent_coverage(self):
        assert should_track_top_k(2_500.0, 200_000, 50) is True
        assert should_track_top_k(2_501.0, 200_000, 50) is False


class TestTopKGateInTheAccumulator:
    @staticmethod
    def _top_values(values: np.ndarray, n_chunks: int = 1) -> list:
        acc = NumericAccumulator("x")
        for chunk in np.array_split(values, n_chunks):
            acc.update(np.ascontiguousarray(chunk))
        return acc.finalize().top_values or []

    def test_a_discrete_column_keeps_a_complete_table(self):
        values = np.random.default_rng(0).integers(0, 12, 200_000).astype(float)
        top = self._top_values(values)
        assert len(top) == 12
        assert sum(c for _, c in top) == 200_000

    def test_a_continuous_column_gets_no_table(self):
        values = np.random.default_rng(0).standard_normal(200_000)
        assert self._top_values(values) == []

    def test_a_high_cardinality_column_gets_no_table(self):
        values = np.random.default_rng(0).integers(0, 50_000, 200_000) / 8.0
        assert self._top_values(values) == []

    @pytest.mark.parametrize("n_chunks", [1, 4, 37])
    def test_the_verdict_does_not_depend_on_chunking(self, n_chunks):
        """The gate latches mid-stream, so its result must still be invariant."""
        rng = np.random.default_rng(0)
        continuous = rng.standard_normal(200_000)
        discrete = rng.integers(0, 12, 200_000).astype(float)
        assert self._top_values(continuous, n_chunks) == []
        assert sorted(self._top_values(discrete, n_chunks)) == sorted(
            self._top_values(discrete, 1)
        )

    def test_the_gate_never_re_enables(self):
        """Partial counts gathered after a cutoff would be worse than none."""
        acc = NumericAccumulator("x")
        acc.update(np.random.default_rng(0).standard_normal(200_000))
        assert acc._track_top_k is False
        acc.update(np.zeros(1000, dtype=float))
        assert acc._track_top_k is False
        assert acc.finalize().top_values == []

    def test_reset_re_enables_the_gate(self):
        acc = NumericAccumulator("x")
        acc.update(np.random.default_rng(0).standard_normal(200_000))
        acc.reset()
        assert acc._track_top_k is True

    def test_merging_a_gated_column_gives_up_the_table(self):
        gated = NumericAccumulator("x")
        gated.update(np.random.default_rng(0).standard_normal(200_000))
        discrete = NumericAccumulator("x")
        discrete.update(np.zeros(1000, dtype=float))
        discrete.merge(gated)
        assert discrete.finalize().top_values == []


class TestChunkSizeStaysInBand:
    """A default that was never exercised is how the old 200,000 survived."""

    def test_the_documented_defaults_agree(self):
        assert EngineConfig().chunk_size == ComputeOptions().chunk_size
        assert ChunkingConfig().chunk_size == EngineConfig().chunk_size

    def test_the_default_is_in_the_measured_band(self):
        assert 25_000 <= EngineConfig().chunk_size <= 100_000

    def test_a_frame_larger_than_the_default_is_actually_chunked(self):
        from pysuricata.compute.processing.chunking import AdaptiveChunker

        df = pd.DataFrame({"a": np.arange(200_000, dtype=float)})
        result = AdaptiveChunker().chunks_from_source(df, EngineConfig().chunk_size)
        assert result.success
        assert len(list(result.data)) > 1

    @pytest.mark.parametrize("n_rows", [1_000, 60_000, 500_000])
    @pytest.mark.parametrize("n_cols", [1, 14, 120])
    def test_the_auto_chosen_size_stays_in_band(self, n_rows, n_cols):
        from pysuricata.compute.processing.chunking import AdaptiveChunker

        df = pd.DataFrame(
            {f"c{i}": np.zeros(n_rows, dtype=float) for i in range(n_cols)}
        )
        chosen = AdaptiveChunker().adaptive_chunk_size(df)
        assert 1_000 <= chosen <= 100_000

    def test_an_explicit_size_is_still_honoured_exactly(self):
        from pysuricata.compute.processing.chunking import AdaptiveChunker

        df = pd.DataFrame({"a": np.arange(200_000, dtype=float)})
        chunker = AdaptiveChunker()
        assert chunker._determine_optimal_chunk_size(df, 30_000) == 30_000

    def test_profiling_is_unaffected_by_the_smaller_default(self):
        df = pd.DataFrame({"a": np.arange(120_000, dtype=float)})
        stats = ComputeOptions()
        assert stats.chunk_size == 50_000
        from pysuricata import summarize

        summary = summarize(df, config=ProfileConfig(compute=stats))
        assert summary["dataset"]["rows_est"] == 120_000


class TestDatetimeFallbackWindow:
    def test_a_pre_1906_timestamp_survives_the_fallback_path(self):
        from pysuricata.accumulators.datetime import DatetimeAccumulator

        acc = DatetimeAccumulator("d")
        # 1880-01-01, comfortably inside datetime64[ns] and outside the old
        # -2e18 bound this path used to carry.
        ts = int(pd.Timestamp("1880-01-01").value)
        acc._update_fallback([ts])
        assert acc.count == 1
        assert acc.missing == 0
        assert acc.min_ts == ts

    def test_an_unrepresentable_timestamp_is_still_rejected(self):
        from pysuricata.accumulators.datetime import DatetimeAccumulator

        acc = DatetimeAccumulator("d")
        acc._update_fallback([2**63])
        assert acc.count == 0
        assert acc.missing == 1

    def test_the_fallback_agrees_with_the_vectorised_path(self):
        from pysuricata.accumulators.datetime import DatetimeAccumulator

        stamps = [
            int(pd.Timestamp(s).value)
            for s in ("1700-01-01", "1880-06-15", "1969-12-31", "2026-08-15")
        ]
        fast = DatetimeAccumulator("d")
        fast.update(stamps)
        slow = DatetimeAccumulator("d")
        slow._update_fallback(stamps)
        assert (fast.count, fast.missing) == (slow.count, slow.missing)
        assert (fast.min_ts, fast.max_ts) == (slow.min_ts, slow.max_ts)
