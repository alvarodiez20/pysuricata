"""Unit tests for the statistical core: sketches and accumulator invariants.

The accumulators are ~2,600 lines carrying every number the report prints, and
they were among the least-covered modules in the package. Two properties matter
more than any individual statistic and are asserted throughout:

* **Mergeability** — merging two accumulators must equal accumulating the
  concatenation. This is what makes chunked processing valid at all.
* **Chunk-invariance** — the same data fed in different chunk sizes must give
  the same answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysuricata.accumulators.algorithms import StreamingMoments
from pysuricata.accumulators.sketches import (
    KMV,
    MisraGries,
    ReservoirSampler,
    _hash_numeric_array,
    _hash_value,
    _mix64_array,
)

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_scalar_and_vectorised_paths_agree(self):
        """The exact counter is keyed by hash, so the two paths must not
        disagree about a value or it gets counted twice."""
        values = [1.0, -3.5, 0.0, 1e300, -1e-300, 42.0]
        scalar = [_hash_value(v) for v in values]
        vectorised = _hash_numeric_array(np.array(values, dtype=float)).tolist()
        assert scalar == vectorised

    def test_negative_zero_is_not_a_distinct_value(self):
        assert _hash_value(-0.0) == _hash_value(0.0)
        pair = _hash_numeric_array(np.array([0.0, -0.0]))
        assert pair[0] == pair[1]

    def test_int_and_equal_float_hash_alike(self):
        assert _hash_value(42) == _hash_value(42.0)

    def test_avalanche(self):
        """A single input bit flip should change about half the output bits.

        This is the only property a distinct-count sketch actually needs from
        its hash, and the reason SHA-1 was overkill.
        """
        base = np.arange(1, 4001, dtype=np.uint64)
        h0 = _mix64_array(base)
        for bit in (0, 1, 7, 31, 62, 63):
            h1 = _mix64_array(base ^ np.uint64(1 << bit))
            diff = np.bitwise_xor(h0, h1)
            frac = np.mean([bin(int(v)).count("1") for v in diff]) / 64.0
            assert 0.45 < frac < 0.55, f"bit {bit}: {frac}"

    def test_handles_none_bytes_and_strings(self):
        for value in (None, b"raw", "text", True, False):
            assert isinstance(_hash_value(value), int)
        assert _hash_value(True) != _hash_value(False)
        assert _hash_value(None) != _hash_value("None")


# ---------------------------------------------------------------------------
# KMV distinct counting
# ---------------------------------------------------------------------------


class TestKMV:
    @pytest.mark.parametrize("n", [1, 50, 1000, 2047])
    def test_exact_below_k(self, n):
        s = KMV(2048)
        s.add_many(np.arange(n, dtype=float))
        assert s.estimate() == n

    @pytest.mark.parametrize("n,k", [(10_000, 1024), (100_000, 2048), (500_000, 4096)])
    def test_within_error_bound(self, n, k):
        s = KMV(k)
        s.add_many(np.arange(n, dtype=float))
        rse = 1.0 / np.sqrt(k)
        assert abs(s.estimate() - n) / n < 4 * rse

    def test_duplicates_do_not_inflate(self):
        """Repeated values must not each occupy a slot in the sketch."""
        s = KMV(2048)
        s.add_many(np.tile(np.arange(5000, dtype=float), 20))
        assert abs(s.estimate() - 5000) / 5000 < 0.15

    def test_estimate_is_invariant_to_chunking(self):
        values = np.arange(300_000, dtype=float)
        whole = KMV(2048)
        whole.add_many(values)
        chunked = KMV(2048)
        for chunk in np.array_split(values, 37):
            chunked.add_many(chunk)
        assert chunked.estimate() == whole.estimate()

    def test_add_and_add_many_agree(self):
        values = np.arange(3000, dtype=float)
        batch = KMV(2048)
        batch.add_many(values)
        scalar = KMV(2048)
        for v in values:
            scalar.add(float(v))
        assert scalar.estimate() == batch.estimate()

    def test_empty_input_is_a_no_op(self):
        s = KMV(1024)
        s.add_many(np.array([], dtype=float))
        assert s.estimate() == 0

    def test_single_repeated_value_counts_once(self):
        s = KMV(1024)
        s.add_many(np.full(10_000, 7.0))
        assert s.estimate() == 1

    def test_values_stay_sorted_distinct_and_bounded(self):
        s = KMV(256)
        s.add_many(np.arange(50_000, dtype=float))
        assert len(s._values) <= s.k
        assert s._values == sorted(s._values)
        assert len(set(s._values)) == len(s._values)

    def test_spill_out_of_exact_mode_does_not_double_count(self):
        """Crossing max_exact_tracking used to re-offer already-counted hashes."""
        s = KMV(4096, max_exact_tracking=100)
        s.add_many(np.arange(500, dtype=float))
        assert not s._use_exact
        assert s.estimate() == 500

    def test_strings_are_counted(self):
        s = KMV(1024)
        s.add_many([f"cat-{i % 37}" for i in range(5000)])
        assert s.estimate() == 37


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------


class TestReservoirSampler:
    def test_keeps_everything_below_k(self):
        r = ReservoirSampler(100)
        r.add_many(np.arange(10, dtype=float))
        assert r.values() == list(range(10))

    def test_never_exceeds_k(self):
        r = ReservoirSampler(64)
        r.add_many(np.arange(100_000, dtype=float))
        assert len(r.values()) == 64

    def test_sample_is_independent_of_chunking(self):
        values = np.arange(100_000, dtype=float)
        samples = []
        for n_chunks in (1, 7, 100):
            np.random.seed(11)
            r = ReservoirSampler(500)
            for chunk in np.array_split(values, n_chunks):
                r.add_many(chunk)
            samples.append(sorted(r.values()))
        assert samples[1] == samples[0]
        assert samples[2] == samples[0]

    def test_every_slot_is_reachable(self):
        """Acceptance picks a slot uniformly; a scaling bug would strand slots."""
        np.random.seed(1)
        r = ReservoirSampler(100)
        r.add_many(np.arange(100_000, dtype=float))
        assert len(set(r.values())) == 100

    def test_empty_batch_is_a_no_op(self):
        r = ReservoirSampler(10)
        r.add_many(np.array([], dtype=float))
        assert r.values() == []

    def test_interleaved_add_and_add_many(self):
        np.random.seed(2)
        r = ReservoirSampler(50)
        for i in range(100):
            r.add(float(i))
        r.add_many(np.arange(100, 300, dtype=float))
        for i in range(300, 400):
            r.add(float(i))
        assert len(r.values()) == 50
        assert r._seen == 400


# ---------------------------------------------------------------------------
# Misra-Gries top-k
# ---------------------------------------------------------------------------


class TestMisraGries:
    def test_finds_the_heavy_hitter(self):
        mg = MisraGries(k=8)
        for _ in range(1000):
            mg.add("dominant")
        for i in range(200):
            mg.add(f"rare-{i}")
        top = max(mg.counters.items(), key=lambda kv: kv[1])
        assert top[0] == "dominant"

    def test_counter_count_is_bounded(self):
        mg = MisraGries(k=10)
        for i in range(5000):
            mg.add(f"v{i}")
        assert len(mg.counters) <= 10

    def test_exact_below_capacity(self):
        mg = MisraGries(k=50)
        for label, count in (("a", 3), ("b", 2), ("c", 1)):
            for _ in range(count):
                mg.add(label)
        assert mg.counters == {"a": 3, "b": 2, "c": 1}


# ---------------------------------------------------------------------------
# Streaming moments — the mergeability invariant
# ---------------------------------------------------------------------------


class TestStreamingMomentsInvariants:
    @pytest.fixture
    def data(self):
        return np.random.default_rng(99).lognormal(0.0, 1.2, 50_000)

    @pytest.mark.parametrize("n_chunks", [1, 2, 9, 128])
    def test_matches_numpy_for_any_chunking(self, data, n_chunks):
        m = StreamingMoments()
        for chunk in np.array_split(data, n_chunks):
            m.update(chunk)
        got = m.get_statistics()

        centred = data - data.mean()
        m2 = (centred**2).mean()
        assert got["mean"] == pytest.approx(data.mean(), rel=1e-9)
        assert got["std"] == pytest.approx(data.std(ddof=1), rel=1e-9)
        # Looser on the higher moments by design. On a heavy-tailed sample the
        # fourth moment is dominated by a handful of extreme values, so a
        # streaming merge and NumPy's two-pass computation legitimately differ in
        # the last few digits -- and numpy's Generator stream is not guaranteed
        # stable across versions, so the actual draw varies by environment.
        assert got["skew"] == pytest.approx((centred**3).mean() / m2**1.5, rel=1e-4)
        assert got["kurtosis"] == pytest.approx(
            (centred**4).mean() / m2**2 - 3.0, rel=1e-4
        )

    @pytest.mark.parametrize("n_chunks", [1, 3, 37])
    def test_chunking_does_not_change_the_answer(self, data, n_chunks):
        """The invariant that does not depend on NumPy agreeing: however the
        stream is split, the accumulator must land in the same place."""
        whole = StreamingMoments()
        whole.update(data)
        split = StreamingMoments()
        for chunk in np.array_split(data, n_chunks):
            split.update(chunk)

        reference, got = whole.get_statistics(), split.get_statistics()
        for key in ("mean", "std", "skew", "kurtosis"):
            assert got[key] == pytest.approx(reference[key], rel=1e-9), key

    def test_merge_equals_single_stream(self, data):
        left, right = data[:20_000], data[20_000:]
        a, b, whole = StreamingMoments(), StreamingMoments(), StreamingMoments()
        a.update(left)
        b.update(right)
        whole.update(data)
        a.merge(b)

        merged, direct = a.get_statistics(), whole.get_statistics()
        for key in ("mean", "std", "skew", "kurtosis"):
            assert merged[key] == pytest.approx(direct[key], rel=1e-9), key

    def test_merging_an_empty_accumulator_changes_nothing(self, data):
        a, empty = StreamingMoments(), StreamingMoments()
        a.update(data)
        before = a.get_statistics()
        a.merge(empty)
        assert a.get_statistics() == before

    def test_merging_into_an_empty_accumulator_adopts_the_other(self, data):
        a, b = StreamingMoments(), StreamingMoments()
        b.update(data)
        a.merge(b)
        assert a.get_statistics() == b.get_statistics()

    def test_constant_column_has_zero_spread(self):
        m = StreamingMoments()
        m.update(np.full(1000, 3.5))
        got = m.get_statistics()
        assert got["mean"] == pytest.approx(3.5)
        assert got["std"] == pytest.approx(0.0)
        assert got["skew"] == 0.0

    def test_non_finite_values_are_excluded(self):
        m = StreamingMoments()
        m.update(np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0]))
        got = m.get_statistics()
        assert got["count"] == 3
        assert got["mean"] == pytest.approx(2.0)

    def test_empty_update_is_safe(self):
        m = StreamingMoments()
        m.update(np.array([], dtype=float))
        assert m.get_statistics()["count"] == 0

    def test_large_mean_does_not_destroy_variance(self):
        """The naive sum-of-squares form collapses here; Welford must not."""
        values = 1e9 + np.random.default_rng(4).standard_normal(50_000)
        m = StreamingMoments()
        for chunk in np.array_split(values, 20):
            m.update(chunk)
        assert m.get_statistics()["std"] == pytest.approx(values.std(ddof=1), rel=1e-6)


class TestHashingAtScale:
    """Large-array regression guards.

    A build of numpy without wheels for the running Python computed uint64
    arithmetic wrongly for large arrays, collapsing every hash to the same
    value. That reached the test suite only as an absurd distinct-count
    estimate; these assert the property directly, at a size that exercises the
    vectorised paths.
    """

    @pytest.mark.parametrize("n", [10_000, 100_000, 300_000])
    def test_distinct_inputs_give_distinct_hashes(self, n):
        hashes = _hash_numeric_array(np.arange(n, dtype=float))
        assert len(np.unique(hashes)) == n

    def test_hashes_spread_across_the_64_bit_range(self):
        """A collapsed or truncated mixer shows up as a degenerate spread."""
        hashes = _hash_numeric_array(np.arange(100_000, dtype=float))
        top_byte = (hashes >> np.uint64(56)).astype(np.int64)
        assert len(np.unique(top_byte)) == 256
        assert hashes.max() > (1 << 63)

    def test_large_distinct_count_is_not_degenerate(self):
        s = KMV(2048)
        s.add_many(np.arange(300_000, dtype=float))
        assert s.estimate() > 200_000


class TestGranularityDegenerateRanges:
    """Granularity detection histograms the gaps between sorted values.

    A strictly positive spread is not enough to bin: when the gaps differ only
    in the last bits of a float, every computed edge rounds to the same value.
    numpy >= 2.5 raises "Too many bins for data range" rather than returning
    degenerate bins, so this has to be handled before the call.
    """

    @staticmethod
    def _granularity(values):
        from pysuricata.accumulators.numeric import NumericAccumulator

        acc = NumericAccumulator("c")
        return acc._compute_granularity(list(values))

    def test_denormal_scale_values_do_not_raise(self):
        step, _ = self._granularity([1e-15 * i for i in range(10)])
        assert step == pytest.approx(1e-15, rel=1e-6)

    @pytest.mark.parametrize("scale", [1e-300, 1e-15, 1.0, 1e15])
    def test_evenly_spaced_values_report_their_step(self, scale):
        step, _ = self._granularity([scale * i for i in range(10)])
        assert step == pytest.approx(scale, rel=1e-6)

    def test_constant_column_has_no_granularity(self):
        assert self._granularity([3.0] * 10) == (None, None)

    def test_single_value_has_no_granularity(self):
        assert self._granularity([3.0]) == (None, None)

    def test_ordinary_integers_still_work(self):
        step, _ = self._granularity([0.0, 5.0, 10.0, 15.0, 20.0])
        assert step == pytest.approx(5.0, rel=1e-6)


class TestReservoirSchedule:
    """The acceptance schedule is precomputed in blocks.

    Correctness depends on the block boundaries falling at points determined by
    the draw sequence alone. If a refill were triggered by where a chunk happens
    to end, the sample would stop being chunk-invariant.
    """

    def test_invariant_across_many_schedule_refills(self):
        """Enough acceptances to cross the block boundary several times."""
        from pysuricata.accumulators.sketches import _SCHEDULE_BLOCK

        values = np.arange(500_000, dtype=np.float64)
        samples = []
        for n_chunks in (1, 13, 977):
            np.random.seed(5)
            r = ReservoirSampler(4000)
            for chunk in np.array_split(values, n_chunks):
                r.add_many(chunk)
            samples.append(sorted(r.values()))

        # k * ln(n/k) acceptances, comfortably more than one block.
        assert 4000 * np.log(500_000 / 4000) > 2 * _SCHEDULE_BLOCK
        assert samples[1] == samples[0]
        assert samples[2] == samples[0]

    def test_schedule_indices_are_strictly_increasing(self):
        np.random.seed(3)
        r = ReservoirSampler(200)
        r.add_many(np.arange(50_000, dtype=np.float64))
        idx = r._sched_idx
        assert np.all(np.diff(idx) > 0)

    def test_slots_stay_in_range(self):
        np.random.seed(4)
        r = ReservoirSampler(128)
        r.add_many(np.arange(200_000, dtype=np.float64))
        assert r._sched_slot.min() >= 0
        assert r._sched_slot.max() < r.k

    def test_sample_mean_tracks_the_population(self):
        """End-to-end statistical check, independent of implementation."""
        n = 400_000
        values = np.arange(n, dtype=np.float64)
        means = []
        for seed in range(25):
            np.random.seed(seed)
            r = ReservoirSampler(2000)
            for chunk in np.array_split(values, 11):
                r.add_many(chunk)
            means.append(float(np.mean(r.values())))
        expected = (n - 1) / 2
        # se of one sample mean is (n/sqrt(12))/sqrt(k); across 25 trials it
        # shrinks by another sqrt(25).
        se = (n / np.sqrt(12)) / np.sqrt(2000) / np.sqrt(25)
        assert abs(float(np.mean(means)) - expected) < 4 * se
