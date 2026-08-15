"""Statistical accuracy oracle.

This is the file that has to exist *before* any kernel is rewritten. Without
it, swapping a Python implementation for a native one is unverifiable: you can
measure that it got faster, but not that it stayed right.

Three classes of check:

1. ``TestNative*``     — native kernel vs NumPy ground truth. These must pass.
2. ``TestInvariants``  — chunked result == unchunked result, for every backend.
                         This is the property a streaming profiler lives or
                         dies by, and it is currently not tested anywhere in
                         the repo.
3. ``TestPythonReference`` — the *current shipped* Python implementations vs
                         ground truth. Several of these are expected to fail
                         today; they are marked ``xfail`` with the reason and
                         a pointer to the code. When a fix lands, pytest
                         reports XPASS and you flip the marker off. That makes
                         each bug a visible, closeable line item instead of a
                         note in a document.

Run:  pytest benchmarks/accuracy.py -v
      pytest benchmarks/accuracy.py -v -m "not slow"
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import pysuricata_core as native
except ImportError:
    native = None

pytestmark = []
requires_native = pytest.mark.skipif(
    native is None, reason="pysuricata-core not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def samples() -> dict[str, np.ndarray]:
    g = np.random.default_rng(20260814)
    return {
        "normal": g.standard_normal(200_000),
        "lognormal": g.lognormal(0.0, 1.3, 200_000),
        "large_mean": 1e9 + g.standard_normal(200_000),
        "integers": g.integers(-500, 500, 200_000).astype(np.float64),
        "constant": np.full(50_000, 3.5),
        "with_nan": np.where(
            g.random(200_000) < 0.1, np.nan, g.standard_normal(200_000)
        ),
        "tiny": np.array([1.0, 2.0, 3.0]),
    }


def rel(a: float, b: float, scale: float = 1.0) -> float:
    """Relative error against an explicit scale, so near-zero values do not
    produce meaningless ratios."""
    denom = max(abs(b), scale)
    return abs(a - b) / denom if denom else abs(a - b)


# ---------------------------------------------------------------------------
# 1. Native vs NumPy
# ---------------------------------------------------------------------------


@requires_native
class TestNativeMoments:
    @pytest.mark.parametrize(
        "key", ["normal", "lognormal", "large_mean", "integers", "constant", "tiny"]
    )
    def test_matches_numpy(self, samples, key):
        arr = samples[key]
        got = native.scan_numeric(arr)
        finite = arr[np.isfinite(arr)]
        n = len(finite)
        sd = float(finite.std()) or 1.0

        assert got["n_valid"] == n
        assert rel(got["mean"], float(finite.mean()), sd) < 1e-12
        # 1e-8, not 1e-12, and the "large_mean" case is why. Centring on the
        # tile mean cancels ~9 significant digits when the data is 1e9 +/- 1,
        # leaving ~7 for the deviation. NumPy's own two-pass std has the same
        # budget, so this is agreement at the limit of float64, not slack.
        assert rel(got["std"], float(finite.std(ddof=1)), sd) < 1e-8
        assert rel(got["min"], float(finite.min()), sd) < 1e-15
        assert rel(got["max"], float(finite.max()), sd) < 1e-15

    def test_skew_kurtosis_match_scipy_definition(self, samples):
        arr = samples["lognormal"]
        got = native.scan_numeric(arr)
        m = arr.mean()
        d = arr - m
        m2 = (d**2).mean()
        m3 = (d**3).mean()
        m4 = (d**4).mean()
        assert rel(got["skewness"], m3 / m2**1.5, 1.0) < 1e-8
        assert rel(got["kurtosis"], m4 / m2**2 - 3.0, 1.0) < 1e-8

    def test_counts_nan_inf_zeros_negatives(self):
        arr = np.array([1.0, -1.0, 0.0, np.nan, np.inf, -np.inf, 0.0, -3.0])
        got = native.scan_numeric(arr)
        assert got["n_total"] == 8
        assert got["n_nan"] == 1
        assert got["n_inf"] == 2
        assert got["n_zeros"] == 2
        assert got["n_negatives"] == 2
        assert got["n_valid"] == 5

    def test_all_nan_does_not_crash(self):
        got = native.scan_numeric(np.full(1000, np.nan))
        assert got["n_valid"] == 0
        assert math.isnan(got["min"])
        assert got["last_finite"] is None

    def test_empty_array(self):
        got = native.scan_numeric(np.array([], dtype=np.float64))
        assert got["n_total"] == 0
        assert got["n_valid"] == 0


@requires_native
class TestNativeKmv:
    def test_exact_below_k(self):
        s = native.KmvSketch(4096)
        s.offer_f64(np.arange(1000, dtype=np.float64))
        assert s.is_exact
        assert s.estimate() == 1000.0

    @pytest.mark.parametrize(
        "truth,k", [(10_000, 1024), (100_000, 2048), (1_000_000, 4096)]
    )
    def test_within_error_bound(self, truth, k):
        s = native.KmvSketch(k)
        s.offer_u64(np.arange(truth, dtype=np.uint64))
        est, err = s.estimate_with_error()
        assert abs(est - truth) / truth < 4 * err, f"{est} vs {truth}, rse={err:.4f}"

    def test_duplicates_do_not_inflate(self):
        s = native.KmvSketch(2048)
        vals = np.tile(np.arange(5000, dtype=np.uint64), 20)
        s.offer_u64(vals)
        assert abs(s.estimate() - 5000) / 5000 < 0.15
        assert s.seen == len(vals)

    def test_merge_equals_single_stream(self):
        a, b, both = (
            native.KmvSketch(2048),
            native.KmvSketch(2048),
            native.KmvSketch(2048),
        )
        left = np.arange(0, 300_000, dtype=np.uint64)
        right = np.arange(200_000, 500_000, dtype=np.uint64)
        a.offer_u64(left)
        b.offer_u64(right)
        both.offer_u64(left)
        both.offer_u64(right)
        a.merge(b)
        assert a.estimate() == both.estimate()

    def test_negative_zero_not_distinct(self):
        s = native.KmvSketch(1024)
        s.offer_f64(np.array([0.0, -0.0, 0.0]))
        assert s.estimate() == 1.0

    def test_pickle_roundtrip(self):
        import pickle

        s = native.KmvSketch(1024)
        s.offer_u64(np.arange(50_000, dtype=np.uint64))
        s2 = pickle.loads(pickle.dumps(s))
        assert s2.estimate() == s.estimate()
        assert s2.k == s.k and s2.seen == s.seen


@requires_native
class TestNativeReservoir:
    def test_reproducible(self):
        arr = np.random.default_rng(0).standard_normal(100_000)
        a, b = native.Reservoir(1000, 42), native.Reservoir(1000, 42)
        a.add_many(arr)
        b.add_many(arr)
        np.testing.assert_array_equal(a.values(), b.values())

    def test_seed_changes_sample(self):
        arr = np.random.default_rng(0).standard_normal(100_000)
        a, b = native.Reservoir(1000, 1), native.Reservoir(1000, 2)
        a.add_many(arr)
        b.add_many(arr)
        assert not np.array_equal(a.values(), b.values())

    def test_does_not_touch_global_rng(self):
        """The whole point: profiling must not perturb the caller's RNG."""
        np.random.seed(1234)
        before = np.random.random()
        np.random.seed(1234)
        r = native.Reservoir(1000, 7)
        r.add_many(np.random.default_rng(0).standard_normal(50_000))
        after = np.random.random()
        assert before == after

    def test_uniform_over_position(self):
        """Detects the batch-uniform-draw bias directly.

        If early elements of each batch are under-sampled, the mean position of
        the sample drifts above n/2.
        """
        n = 500_000
        arr = np.arange(n, dtype=np.float64)
        means = []
        for seed in range(30):
            r = native.Reservoir(2000, seed)
            r.add_many(arr)
            means.append(float(np.mean(r.values())))
        observed = float(np.mean(means))
        expected = (n - 1) / 2
        # SE of one sample mean ~ n/sqrt(12*2000) = 3227; over 30 trials ~589.
        assert abs(observed - expected) < 5 * 589, f"{observed} vs {expected}"

    @pytest.mark.parametrize("q", [0.01, 0.25, 0.5, 0.75, 0.99])
    def test_quantiles_track_numpy(self, q):
        arr = np.random.default_rng(3).lognormal(0, 1, 1_000_000)
        r = native.Reservoir(20_000, 0)
        r.add_many(arr)
        est = float(r.quantiles([q])[0])
        truth = float(np.percentile(arr, q * 100))
        # A 20k sample gives a few tenths of a percent on interior quantiles
        # and more in the tails; 5% is a loose but honest bound at p99.
        assert rel(est, truth, abs(truth) * 0.05) < 0.05, f"q{q}: {est} vs {truth}"


# ---------------------------------------------------------------------------
# 2. Invariants that must hold for any backend
# ---------------------------------------------------------------------------


@requires_native
class TestInvariants:
    @pytest.mark.parametrize("n_chunks", [1, 3, 7, 64])
    def test_moments_invariant_to_chunking(self, n_chunks):
        """Profiling the same data in k chunks must give the same moments.

        This is the check that catches the M3/M4 batch-merge formula in
        ``accumulators/algorithms.py::StreamingMoments._update_vectorized``.
        """
        arr = np.random.default_rng(11).lognormal(0, 1.2, 100_003)
        whole = native.NumericKernel(20_000, 2048, 0)
        whole.update(arr)
        w = whole.finalize()

        parts = native.NumericKernel(20_000, 2048, 0)
        for chunk in np.array_split(arr, n_chunks):
            parts.update(np.ascontiguousarray(chunk))
        p = parts.finalize()

        for field in ("count", "mean", "variance", "skew", "kurtosis", "min", "max"):
            assert rel(p[field], w[field], abs(w[field]) * 1e-9 or 1e-9) < 1e-8, (
                f"{field} drifted with {n_chunks} chunks: {p[field]} vs {w[field]}"
            )

    def test_counts_invariant_to_chunking(self):
        arr = np.where(
            np.random.default_rng(12).random(50_000) < 0.1,
            np.nan,
            np.random.default_rng(13).standard_normal(50_000),
        )
        whole = native.NumericKernel(5000, 1024, 0)
        whole.update(arr)
        parts = native.NumericKernel(5000, 1024, 0)
        for chunk in np.array_split(arr, 17):
            parts.update(np.ascontiguousarray(chunk))
        assert whole.finalize()["missing"] == parts.finalize()["missing"]
        assert whole.finalize()["count"] == parts.finalize()["count"]

    def test_kernel_merge_matches_sequential(self):
        arr = np.random.default_rng(14).standard_normal(80_000)
        seq = native.NumericKernel(20_000, 2048, 0)
        seq.update(arr)

        a = native.NumericKernel(20_000, 2048, 0)
        b = native.NumericKernel(20_000, 2048, 0)
        a.update(np.ascontiguousarray(arr[:40_000]))
        b.update(np.ascontiguousarray(arr[40_000:]))
        a.merge(b)

        s, m = seq.finalize(), a.finalize()
        for field in ("count", "mean", "variance", "skew", "kurtosis", "min", "max"):
            assert rel(m[field], s[field], abs(s[field]) * 1e-9 or 1e-9) < 1e-8, field

    def test_large_mean_variance_is_not_cancelled(self):
        """Naive sum-of-squares formulas collapse here; Welford does not."""
        arr = 1e9 + np.random.default_rng(15).standard_normal(200_000)
        got = native.NumericKernel(20_000, 2048, 0)
        got.update(arr)
        f = got.finalize()
        assert rel(f["variance"], float(arr.var(ddof=1)), 1.0) < 1e-6
        assert f["variance"] > 0.5  # a cancelled formula returns ~0 here


# ---------------------------------------------------------------------------
# 3. The shipped Python implementations vs ground truth
# ---------------------------------------------------------------------------

pysuricata = pytest.importorskip("pysuricata", reason="pysuricata not installed")


class TestPythonReference:
    """These run against the installed pysuricata package.

    Each ``xfail`` below is a live bug with a code pointer. Remove the marker
    when the fix lands; if it starts passing while the marker is still there,
    pytest reports XPASS and tells you so.
    """

    def test_kmv_estimate_reasonable(self):
        from pysuricata.accumulators.sketches import KMV

        s = KMV(4096)
        s.add_many([f"v{i}" for i in range(200_000)])
        est = s.estimate()
        assert abs(est - 200_000) / 200_000 < 0.10, est

    def test_moments_invariant_to_batching(self):
        from pysuricata.accumulators.algorithms import StreamingMoments

        arr = np.random.default_rng(21).lognormal(0, 1.2, 100_000)
        whole = StreamingMoments()
        whole.update(arr)
        w = whole.get_statistics()

        parts = StreamingMoments()
        for chunk in np.array_split(arr, 10):
            parts.update(chunk)
        p = parts.get_statistics()

        assert rel(p["skew"], w["skew"], 1e-9) < 1e-6, (
            f"skew {p['skew']} vs {w['skew']}"
        )
        assert rel(p["kurtosis"], w["kurtosis"], 1e-9) < 1e-6

    def test_skewness_matches_definition(self):
        from pysuricata.accumulators.algorithms import StreamingMoments

        arr = np.random.default_rng(22).lognormal(0, 1.0, 50_000)
        m = StreamingMoments()
        m.update(arr)
        d = arr - arr.mean()
        truth = (d**3).mean() / (d**2).mean() ** 1.5
        assert rel(m.get_statistics()["skew"], truth, 1e-9) < 1e-6

    def test_reservoir_is_uniform_over_position(self):
        from pysuricata.accumulators.sketches import ReservoirSampler

        n = 500_000
        arr = np.arange(n, dtype=np.float64)
        means = []
        for seed in range(30):
            r = ReservoirSampler(2000, rng=np.random.default_rng(seed))
            for chunk in np.array_split(arr, 5):
                r.add_many(chunk)
            means.append(float(np.mean(r.values())))
        observed = float(np.mean(means))
        expected = (n - 1) / 2
        assert abs(observed - expected) < 5 * 589, f"{observed} vs {expected}"

    def test_profile_does_not_reset_global_rng(self):
        import pandas as pd

        from pysuricata import profile

        np.random.seed(999)
        expected = np.random.random()
        np.random.seed(999)
        profile(pd.DataFrame({"a": np.arange(1000, dtype=float)}))
        assert np.random.random() == expected

    def test_generator_source_keeps_first_chunk(self):
        import pandas as pd

        from pysuricata import summarize

        def gen():
            for i in range(4):
                yield pd.DataFrame(
                    {"a": np.arange(i * 1000, (i + 1) * 1000, dtype=float)}
                )

        stats = summarize(gen())
        assert stats["dataset"]["rows_est"] == 4000
        assert stats["columns"]["a"]["min"] == 0.0
        assert stats["columns"]["a"]["max"] == 3999.0

    def test_single_chunk_generator_is_not_empty(self):
        """A one-chunk generator used to report "Empty source": sniffing consumed
        the only chunk, leaving nothing for the chunk loop."""
        import pandas as pd

        from pysuricata import summarize

        def gen():
            yield pd.DataFrame({"a": np.arange(500, dtype=float)})

        stats = summarize(gen())
        assert stats["dataset"]["rows_est"] == 500
        assert stats["columns"]["a"]["min"] == 0.0

    def test_generator_matches_equivalent_dataframe(self):
        """The chunked-vs-unchunked invariant, for a generator source."""
        import pandas as pd

        from pysuricata import summarize

        g = np.random.default_rng(17)
        values = g.standard_normal(4000)
        whole = summarize(pd.DataFrame({"a": values}))
        streamed = summarize(
            pd.DataFrame({"a": values[i : i + 1000]}) for i in range(0, 4000, 1000)
        )

        assert streamed["dataset"]["rows_est"] == whole["dataset"]["rows_est"]
        for stat in ("min", "max", "mean"):
            assert streamed["columns"]["a"][stat] == pytest.approx(
                whole["columns"]["a"][stat]
            )

    def test_correlation_survives_large_mean(self):
        import pandas as pd

        from pysuricata import summarize

        g = np.random.default_rng(31)
        base = g.standard_normal(50_000)
        df = pd.DataFrame(
            {"x": 1e9 + base, "y": 1e9 + base * 0.9 + g.standard_normal(50_000) * 0.1}
        )
        stats = summarize(df)
        top = stats["columns"]["x"].get("corr_top") or []
        assert top, "no correlation reported for two strongly correlated columns"
        assert abs(top[0][1]) > 0.8


@pytest.mark.slow
class TestChunkedVsUnchunkedEndToEnd:
    """The headline invariant, end to end: same data, different chunk sizes,
    same numbers. Not currently asserted anywhere in the repo's test suite."""

    @pytest.mark.parametrize("chunk_size", [10_000, 50_000, 250_000])
    def test_summarize_stable_across_chunk_sizes(self, chunk_size):
        import pandas as pd

        from pysuricata import ProfileConfig, summarize

        g = np.random.default_rng(41)
        df = pd.DataFrame(
            {"a": g.lognormal(0, 1, 200_000), "b": g.standard_normal(200_000)}
        )

        cfg = ProfileConfig()
        cfg.compute.chunk_size = chunk_size
        cfg.compute.random_seed = 7
        got = summarize(df, config=cfg)["columns"]["a"]

        assert rel(got["mean"], float(df["a"].mean()), 1e-9) < 1e-9
        assert rel(got["std"], float(df["a"].std()), 1e-9) < 1e-6
        # Median comes from the reservoir, so it is an estimate; 2% is the
        # honest tolerance for a 10k-20k sample of a lognormal.
        assert rel(got["median"], float(df["a"].median()), 1e-9) < 0.02


class TestReservoirInvariants:
    """Algorithm L guarantees these; the previous batch sampler broke all three."""

    def test_sample_is_independent_of_chunking(self):
        from pysuricata.accumulators.sketches import ReservoirSampler

        arr = np.arange(200_000, dtype=np.float64)
        samples = []
        for n_chunks in (1, 5, 50, 500):
            r = ReservoirSampler(2000, rng=np.random.default_rng(7))
            for chunk in np.array_split(arr, n_chunks):
                r.add_many(chunk)
            samples.append(sorted(r.values()))
        for other in samples[1:]:
            assert other == samples[0]

    def test_no_early_late_bias(self):
        from pysuricata.accumulators.sketches import ReservoirSampler

        n = 200_000
        arr = np.arange(n, dtype=np.float64)
        fracs = []
        for seed in range(40):
            r = ReservoirSampler(1000, rng=np.random.default_rng(seed))
            for chunk in np.array_split(arr, 20):
                r.add_many(chunk)
            fracs.append(float((np.asarray(r.values()) < n / 2).mean()))
        se = float(np.std(fracs)) / np.sqrt(len(fracs))
        assert abs(float(np.mean(fracs)) - 0.5) < 4 * se

    def test_short_stream_is_kept_exactly(self):
        from pysuricata.accumulators.sketches import ReservoirSampler

        r = ReservoirSampler(100)
        r.add_many(np.arange(10, dtype=np.float64))
        assert r.values() == list(range(10))

    def test_never_exceeds_k(self):
        from pysuricata.accumulators.sketches import ReservoirSampler

        r = ReservoirSampler(100)
        r.add_many(np.arange(10_000, dtype=np.float64))
        assert len(r.values()) == 100

    def test_add_and_add_many_stay_in_sync(self):
        from pysuricata.accumulators.sketches import ReservoirSampler

        r = ReservoirSampler(50, rng=np.random.default_rng(3))
        for i in range(200):
            r.add(float(i))
        r.add_many(np.arange(200, 400, dtype=np.float64))
        for i in range(400, 600):
            r.add(float(i))
        assert len(r.values()) == 50
        assert r._seen == 600


class TestRngIsolation:
    """profile() must never read or write the process-global RNG.

    Not by snapshotting and restoring it -- by never touching it. Every sketch
    draws from a generator it owns, seeded per column from ``random_seed``.
    """

    def test_stdlib_rng_preserved(self):
        import random

        import pandas as pd

        from pysuricata import profile

        random.seed(7)
        expected = random.random()
        random.seed(7)
        profile(pd.DataFrame({"a": np.arange(1000, dtype=float)}))
        assert random.random() == expected

    def test_global_rng_untouched_when_profiling_fails(self):
        from pysuricata import profile

        np.random.seed(5)
        before = np.random.get_state()[1][0]
        try:
            profile("not a dataframe")
        except Exception:
            pass
        assert np.random.get_state()[1][0] == before

    def test_seeding_the_global_rng_does_not_change_the_report(self):
        """A caller's seed must not leak into the profile's sampling."""
        import pandas as pd

        from pysuricata import profile
        from pysuricata.api import ComputeOptions, ProfileConfig

        df = pd.DataFrame({"a": np.random.default_rng(2).lognormal(0, 1, 50_000)})
        cfg = ProfileConfig(compute=ComputeOptions(random_seed=42))

        np.random.seed(1)
        first = profile(df, config=cfg).stats["columns"]["a"]["median"]
        np.random.seed(9999)
        second = profile(df, config=cfg).stats["columns"]["a"]["median"]
        assert first == second

    def test_column_seed_does_not_depend_on_the_other_columns(self):
        """A column samples the same rows whether or not its neighbours exist.

        Seeds are derived from the column *name*, so profiling a subset must
        reproduce the numbers from profiling the whole frame -- the property
        that makes per-column threading reproducible.
        """
        import pandas as pd

        from pysuricata import profile
        from pysuricata.api import ComputeOptions, ProfileConfig

        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "a": rng.lognormal(0, 1, 60_000),
                "b": rng.standard_normal(60_000),
                "c": rng.integers(0, 1000, 60_000).astype(float),
            }
        )
        cfg = ProfileConfig(compute=ComputeOptions(random_seed=42))

        full = profile(df, config=cfg).stats["columns"]["a"]["median"]
        alone = profile(df[["a"]], config=cfg).stats["columns"]["a"]["median"]
        reordered = profile(df[["c", "b", "a"]], config=cfg).stats["columns"]["a"][
            "median"
        ]
        assert full == alone == reordered

    def test_same_seed_still_reproducible(self):
        import pandas as pd

        from pysuricata import profile
        from pysuricata.api import ComputeOptions, ProfileConfig

        df = pd.DataFrame({"a": np.random.default_rng(1).lognormal(0, 1, 50_000)})
        cfg = ProfileConfig(compute=ComputeOptions(random_seed=42))
        first = profile(df, config=cfg).stats["columns"]["a"]["median"]
        second = profile(df, config=cfg).stats["columns"]["a"]["median"]
        assert first == second
