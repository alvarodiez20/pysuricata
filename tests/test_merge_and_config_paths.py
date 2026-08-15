"""Merging shards, replacing accumulators, and reporting a bad setting.

Five issues with one thing in common: a path nobody ran. `merge()` exists for
distributed use and nothing in the pipeline calls it (#67). The adapters replace
an accumulator only for forced or reclassified columns (#61). The config
fallback only fires when validation fails (#89). `outlier_methods` was read by a
detector that was never called (#60). And the missing-cell total was recomputed
by a second pass rather than read from the accumulators that had just counted it
(#36).

The merge tests are written as *comparisons against folding both value sets into
one accumulator*, because that is the invariant that matters and the one the old
implementation broke: it replayed one side's 20,000-value reservoir buffer as
though it were the whole stream, so a 90,000-row shard merged into a 60,000-row
one reported a median of 0.17 where the true value was 4.03.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import ComputeOptions, ConfigurationError, ProfileConfig, summarize
from pysuricata.accumulators import CategoricalAccumulator, NumericAccumulator
from pysuricata.accumulators.config import NumericConfig
from pysuricata.accumulators.factory import build_accumulator
from pysuricata.accumulators.sketches import KMV, MisraGries, ReservoirSampler


def _numeric(seed: int = 1, **config) -> NumericAccumulator:
    return NumericAccumulator(
        "x", config=NumericConfig(**config) if config else None, seed=seed
    )


class TestNumericMerge:
    """#67. `a.merge(b)` must equal folding both value sets into one."""

    @pytest.fixture
    def shards(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(60_000), rng.standard_normal(90_000) + 5.0

    @pytest.fixture
    def merged_and_whole(self, shards):
        left, right = shards
        a, b, whole = _numeric(1), _numeric(2), _numeric(1)
        a.update(left)
        b.update(right)
        whole.update(np.concatenate([left, right]))
        a.merge(b)
        return a.finalize(), whole.finalize()

    def test_counts_are_exact(self, merged_and_whole):
        merged, whole = merged_and_whole
        assert merged.count == whole.count

    def test_moments_are_exact(self, merged_and_whole):
        merged, whole = merged_and_whole
        assert merged.mean == pytest.approx(whole.mean, rel=1e-9)
        assert merged.std == pytest.approx(whole.std, rel=1e-9)

    def test_extremes_are_exact(self, merged_and_whole):
        merged, whole = merged_and_whole
        assert merged.min == whole.min
        assert merged.max == whole.max

    def test_the_distinct_estimate_is_exact(self, merged_and_whole):
        """KMV composes: the k smallest hashes of the union are a subset of the
        two sides' k-smallest sets, so nothing is lost by merging."""
        merged, whole = merged_and_whole
        assert merged.unique_est == whole.unique_est

    def test_the_median_reflects_both_shards(self, merged_and_whole):
        """The old merge gave 0.17 here, against a true 4.03, because the
        larger shard's 90,000 rows entered as 20,000 sampled values."""
        merged, whole = merged_and_whole
        assert merged.median == pytest.approx(whole.median, abs=0.1)

    def test_the_quartiles_reflect_both_shards(self, merged_and_whole):
        merged, whole = merged_and_whole
        assert merged.q1 == pytest.approx(whole.q1, abs=0.15)
        assert merged.q3 == pytest.approx(whole.q3, abs=0.15)

    def test_the_histogram_covers_the_merged_range(self, merged_and_whole):
        merged, whole = merged_and_whole
        assert sum(merged.true_histogram_counts) == sum(whole.true_histogram_counts)
        assert merged.true_histogram_edges[0] <= whole.min
        assert merged.true_histogram_edges[-1] >= whole.max

    def test_merging_a_tiny_shard_into_a_large_one_barely_moves_it(self):
        """Weighting, in the direction that used to fail hardest."""
        rng = np.random.default_rng(1)
        big, small = rng.standard_normal(200_000), rng.standard_normal(50) + 100.0
        a, b = _numeric(1), _numeric(2)
        a.update(big)
        b.update(small)
        before = a.finalize().median
        a.merge(b)
        assert a.finalize().median == pytest.approx(before, abs=0.05)

    def test_top_values_come_from_both_sides(self):
        """The counters were not merged at all: a merged column reported only
        the left-hand side's common values."""
        a, b = _numeric(1, top_k_size=10), _numeric(2, top_k_size=10)
        a.update(np.full(5_000, 1.0))
        b.update(np.full(5_000, 2.0))
        a.merge(b)
        assert {value for value, _ in a.finalize().top_values} == {1.0, 2.0}

    def test_a_side_that_gave_up_on_top_k_still_wins(self):
        """#62's gate must survive the merge, or a merged column claims a
        complete table of common values it does not have."""
        a, b = _numeric(1), _numeric(2)
        a.update(np.full(50_000, 1.0))
        b.update(np.arange(200_000.0))
        a.merge(b)
        assert a.finalize().top_values in (None, [])

    def test_merging_is_not_quadratic_in_the_counts(self):
        """The categorical merge replayed one `add()` per counted occurrence."""
        a, b = _numeric(1), _numeric(2)
        a.update(np.full(2_000_000, 7.0))
        b.update(np.full(2_000_000, 8.0))
        a.merge(b)  # would be four million calls if replayed
        assert a.count == 4_000_000


class TestCategoricalMerge:
    def test_the_distinct_estimate_reflects_both_sides(self):
        """It used to be seeded from at most 100 top-k keys, on the belief that
        KMV sketches cannot be merged."""
        left = [f"v{i}" for i in range(5_000)]
        right = [f"w{i}" for i in range(5_000)]
        a, b, whole = (
            CategoricalAccumulator("c", seed=1),
            CategoricalAccumulator("c", seed=2),
            CategoricalAccumulator("c", seed=1),
        )
        a.update(np.array(left, dtype=object))
        b.update(np.array(right, dtype=object))
        whole.update(np.array(left + right, dtype=object))
        a.merge(b)
        assert a.finalize().unique_est == whole.finalize().unique_est

    def test_top_items_come_from_both_sides(self):
        a, b = CategoricalAccumulator("c", seed=1), CategoricalAccumulator("c", seed=2)
        a.update(np.array(["north"] * 900 + ["south"] * 100, dtype=object))
        b.update(np.array(["east"] * 900 + ["west"] * 100, dtype=object))
        a.merge(b)
        assert {item for item, _ in a.finalize().top_items} >= {"north", "east"}

    def test_counts_are_exact(self):
        a, b = CategoricalAccumulator("c", seed=1), CategoricalAccumulator("c", seed=2)
        a.update(np.array(["x"] * 400, dtype=object))
        b.update(np.array(["y"] * 600, dtype=object))
        a.merge(b)
        assert a.finalize().count == 1_000


class TestSketchMergeProperties:
    """The guarantees the accumulator merges rest on."""

    def test_kmv_merges_exactly(self):
        left, right = KMV(256), KMV(256)
        whole = KMV(256)
        a_vals = [f"a{i}" for i in range(5_000)]
        b_vals = [f"b{i}" for i in range(5_000)]
        left.add_many(a_vals)
        right.add_many(b_vals)
        whole.add_many(a_vals + b_vals)
        left.merge(right)
        assert left.estimate() == whole.estimate()

    def test_kmv_stays_exact_while_the_union_is_small(self):
        left, right = KMV(256), KMV(256)
        left.add_many(["a", "b", "c"])
        right.add_many(["c", "d"])
        left.merge(right)
        assert left.is_exact
        assert left.estimate() == 4

    def test_kmv_merging_an_empty_sketch_changes_nothing(self):
        left = KMV(256)
        left.add_many(["a", "b"])
        left.merge(KMV(256))
        assert left.estimate() == 2

    def test_misra_gries_never_overcounts_after_a_merge(self):
        """The guarantee: a reported count may undercount, never overcount."""
        rng = np.random.default_rng(3)
        left = rng.choice(list(range(30)), 4_000).tolist()
        right = rng.choice(list(range(30)), 4_000).tolist()
        a, b = MisraGries(8), MisraGries(8)
        a.add_many(left)
        b.add_many(right)
        a.merge(b)
        truth = {v: (left + right).count(v) for v in set(left + right)}
        assert all(count <= truth[value] for value, count in a.items())

    def test_misra_gries_keeps_the_heaviest_hitter(self):
        a, b = MisraGries(4), MisraGries(4)
        a.add_many(["heavy"] * 900 + [f"x{i}" for i in range(100)])
        b.add_many(["heavy"] * 900 + [f"y{i}" for i in range(100)])
        a.merge(b)
        assert a.items()[0][0] == "heavy"

    def test_misra_gries_stays_within_k_counters(self):
        a, b = MisraGries(5), MisraGries(5)
        a.add_many([f"a{i}" for i in range(50)])
        b.add_many([f"b{i}" for i in range(50)])
        a.merge(b)
        assert len(a.counters) <= 5

    def test_a_complete_reservoir_merges_exactly(self):
        """Under k values seen, the buffer *is* the stream."""
        a = ReservoirSampler(100, rng=np.random.default_rng(0))
        b = ReservoirSampler(100, rng=np.random.default_rng(1))
        a.add_many(np.arange(10.0))
        b.add_many(np.arange(10.0, 25.0))
        a.merge(b)
        assert sorted(a.values()) == list(np.arange(25.0))

    def test_the_merged_sample_is_weighted_by_stream_length(self):
        """Each side should appear in proportion to what it saw, not to what it
        retained. Ten runs, because this is a random quantity."""
        shares = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            a = ReservoirSampler(1_000, rng=rng)
            b = ReservoirSampler(1_000, rng=np.random.default_rng(seed + 100))
            a.add_many(np.zeros(90_000))
            b.add_many(np.ones(10_000))
            a.merge(b)
            shares.append(sum(a.values()) / len(a.values()))
        assert np.mean(shares) == pytest.approx(0.1, abs=0.03)


class TestReplacementAccumulatorsKeepTheirConfig:
    """#61. Forced and reclassified columns fell back to library defaults."""

    def test_a_forced_column_honours_uniques_k(self):
        cfg = ComputeOptions(max_uniques=8_192, force_column_types={"x": "categorical"})
        acc = build_accumulator("categorical", "x", _engine(cfg))
        assert acc._uniques.k == 8_192

    def test_a_forced_column_honours_top_k(self):
        acc = build_accumulator("numeric", "x", _engine(ComputeOptions(top_k=17)))
        assert acc._topk.k == 17

    def test_a_forced_column_honours_the_sample_size(self):
        acc = build_accumulator(
            "numeric", "x", _engine(ComputeOptions(numeric_sample_size=1_234))
        )
        assert acc._sample.k == 1_234

    def test_seeds_still_flow(self):
        """#30's per-column seeds must not be lost while fixing the config."""
        first = build_accumulator(
            "numeric", "x", _engine(ComputeOptions(random_seed=7))
        )
        second = build_accumulator(
            "numeric", "x", _engine(ComputeOptions(random_seed=7))
        )
        assert first._seed == second._seed is not None

    def test_two_columns_get_different_seeds(self):
        cfg = _engine(ComputeOptions(random_seed=7))
        assert (
            build_accumulator("numeric", "a", cfg)._seed
            != build_accumulator("numeric", "b", cfg)._seed
        )

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown column kind"):
            build_accumulator("spreadsheet", "x", None)

    def test_a_reclassified_column_honours_uniques_k_end_to_end(self):
        """The real path: a numeric column with few distinct integers becomes
        categorical, and its replacement must carry the run's settings."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"grade": rng.integers(0, 12, 4_000)})
        config = ProfileConfig(compute=ComputeOptions(max_uniques=8_192))
        stats = summarize(frame, config=config)
        assert stats["columns"]["grade"]["type"] == "categorical"
        assert stats["columns"]["grade"]["unique_est"] == 12


def _engine(options: ComputeOptions):
    from pysuricata.config import EngineConfig

    return EngineConfig.from_options(options)


class TestOutlierMethodsAreHonoured:
    """#60. The option was read by a detector that was never called."""

    @pytest.fixture
    def values(self):
        rng = np.random.default_rng(0)
        return np.concatenate([rng.standard_normal(5_000), np.full(50, 50.0)])

    def test_both_methods_by_default(self, values):
        acc = _numeric()
        acc.update(values)
        summary = acc.finalize()
        assert summary.outliers_iqr > 0
        assert summary.outliers_mod_zscore > 0

    def test_iqr_only(self, values):
        acc = _numeric(outlier_methods=["iqr"])
        acc.update(values)
        summary = acc.finalize()
        assert summary.outliers_iqr > 0
        assert summary.outliers_mod_zscore == 0

    def test_mad_only(self, values):
        acc = _numeric(outlier_methods=["mad"])
        acc.update(values)
        summary = acc.finalize()
        assert summary.outliers_iqr == 0
        assert summary.outliers_mod_zscore > 0


class TestMissingCellsComeFromTheAccumulators:
    """#36. The total was recomputed by a full pass per chunk."""

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        n = 6_000
        frame = pd.DataFrame(
            {
                "num": rng.standard_normal(n),
                "cat": rng.choice(["x", "y"], n).astype(object),
                "when": pd.to_datetime(rng.integers(0, 10**9, n), unit="s"),
                "flag": (rng.random(n) > 0.5).astype(object),
            }
        )
        frame.loc[:200, "num"] = np.nan
        frame.loc[300:400, "when"] = pd.NaT
        frame.loc[500:520, "cat"] = None
        frame.loc[600:610, "flag"] = None
        return frame

    def test_the_total_matches_a_full_pass(self, frame):
        expected = int(frame.isnull().sum().sum())
        assert summarize(frame)["dataset"]["missing_cells"] == expected

    @pytest.mark.parametrize("chunk_size", [500, 2_000, 100_000])
    def test_the_total_is_the_same_at_every_chunk_size(self, frame, chunk_size):
        expected = int(frame.isnull().sum().sum())
        stats = summarize(frame, chunk_size=chunk_size)
        assert stats["dataset"]["missing_cells"] == expected

    def test_a_frame_with_no_nulls_reports_none(self):
        frame = pd.DataFrame({"a": np.arange(1_000.0)})
        assert summarize(frame)["dataset"]["missing_cells"] == 0


class TestBadConfigIsReported:
    """#89. A value that failed validation became a different configuration."""

    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0)})

    def test_an_invalid_value_raises_instead_of_being_dropped(self, frame):
        options = ComputeOptions(columns=("a",))
        object.__setattr__(options, "columns", 5)
        with pytest.raises(ConfigurationError):
            summarize(frame, config=ProfileConfig(compute=options))

    def test_the_error_is_still_a_valueerror(self, frame):
        options = ComputeOptions(columns=("a",))
        object.__setattr__(options, "columns", 5)
        with pytest.raises(ValueError):
            summarize(frame, config=ProfileConfig(compute=options))

    def test_the_column_subset_survives_on_the_happy_path(self, frame):
        """The fallback silently dropped it, so a caller asking for one column
        got the whole frame and a successful-looking run."""
        stats = summarize(
            frame, config=ProfileConfig(compute=ComputeOptions(columns=("a",)))
        )
        assert set(stats["columns"]) == {"a"}

    def test_correlation_settings_survive(self, frame):
        options = ComputeOptions(compute_correlations=False)
        stats = summarize(frame, config=ProfileConfig(compute=options))
        assert stats["columns"]["a"]["corr_top"] == []
