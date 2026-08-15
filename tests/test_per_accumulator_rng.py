"""Sampling draws from per-accumulator generators, never the global RNG.

The previous design seeded ``numpy.random`` and ``random`` globally and undid
the damage with a snapshot/restore wrapper. These tests pin the replacement:
every sketch owns its generator, the seed is derived per column from the run
seed, and the process-global state is neither read nor written.
"""

from __future__ import annotations

import random as pyrandom

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.accumulators.categorical import CategoricalAccumulator
from pysuricata.accumulators.datetime import DatetimeAccumulator
from pysuricata.accumulators.factory import (
    build_accumulators,
    derive_column_seed,
    seed_for_column,
)
from pysuricata.accumulators.numeric import NumericAccumulator
from pysuricata.accumulators.sketches import ReservoirSampler
from pysuricata.api import ComputeOptions, ProfileConfig
from pysuricata.compute.core.types import ColumnKinds
from pysuricata.config import EngineConfig


def _seeded_config(seed: int | None) -> ProfileConfig:
    return ProfileConfig(compute=ComputeOptions(random_seed=seed))


class TestReservoirOwnsItsGenerator:
    def test_explicit_generator_is_used(self):
        a = ReservoirSampler(50, rng=np.random.default_rng(11))
        b = ReservoirSampler(50, rng=np.random.default_rng(11))
        values = np.arange(20_000, dtype=float)
        a.add_many(values)
        b.add_many(values)
        assert a.values() == b.values()

    def test_different_generators_give_different_samples(self):
        a = ReservoirSampler(50, rng=np.random.default_rng(1))
        b = ReservoirSampler(50, rng=np.random.default_rng(2))
        values = np.arange(20_000, dtype=float)
        a.add_many(values)
        b.add_many(values)
        assert a.values() != b.values()

    def test_global_seed_does_not_steer_the_sample(self):
        values = np.arange(20_000, dtype=float)
        np.random.seed(1)
        a = ReservoirSampler(50, rng=np.random.default_rng(7))
        a.add_many(values)
        np.random.seed(999_999)
        b = ReservoirSampler(50, rng=np.random.default_rng(7))
        b.add_many(values)
        assert a.values() == b.values()

    def test_no_generator_still_avoids_the_global_rng(self):
        np.random.seed(4)
        before = np.random.get_state()[1][0]
        r = ReservoirSampler(50)
        r.add_many(np.arange(20_000, dtype=float))
        assert np.random.get_state()[1][0] == before

    def test_two_unseeded_samplers_are_independent(self):
        values = np.arange(50_000, dtype=float)
        a, b = ReservoirSampler(64), ReservoirSampler(64)
        a.add_many(values)
        b.add_many(values)
        assert a.values() != b.values()


class TestSeedDerivation:
    def test_none_run_seed_gives_no_column_seed(self):
        assert derive_column_seed(None, "a") is None

    def test_derivation_is_stable_across_calls(self):
        assert derive_column_seed(42, "price") == derive_column_seed(42, "price")

    def test_columns_get_distinct_seeds(self):
        seeds = {derive_column_seed(42, name) for name in ("a", "b", "c", "d")}
        assert len(seeds) == 4

    def test_run_seed_changes_every_column_seed(self):
        for name in ("a", "b", "c"):
            assert derive_column_seed(42, name) != derive_column_seed(43, name)

    def test_seed_fits_in_uint64(self):
        seed = derive_column_seed(42, "a")
        assert 0 <= seed < 2**64

    def test_seed_for_column_reads_random_seed_off_any_config(self):
        cfg = EngineConfig(random_seed=42)
        assert seed_for_column(cfg, "a") == derive_column_seed(42, "a")

    def test_seed_for_column_tolerates_a_config_without_the_attribute(self):
        assert seed_for_column(object(), "a") is None


class TestAccumulatorsAcceptSeeds:
    @pytest.mark.parametrize(
        "cls,values",
        [
            (NumericAccumulator, np.arange(30_000, dtype=float)),
            (CategoricalAccumulator, np.array([f"v{i % 997}" for i in range(30_000)])),
            (
                DatetimeAccumulator,
                pd.date_range("2020-01-01", periods=30_000, freq="min").to_numpy(),
            ),
        ],
    )
    def test_same_seed_gives_the_same_summary(self, cls, values):
        a, b = cls("x", seed=99), cls("x", seed=99)
        a.update(values)
        b.update(values)
        assert a.finalize() == b.finalize()

    def test_reset_rewinds_the_generator(self):
        values = np.arange(30_000, dtype=float)
        acc = NumericAccumulator("x", seed=5)
        acc.update(values)
        first = acc.finalize()
        acc.reset()
        acc.update(values)
        assert acc.finalize() == first

    def test_reset_works_on_the_default_config(self):
        """Outlier detection and metrics tracking are on by default."""
        acc = NumericAccumulator("x")
        acc.update(np.arange(100, dtype=float))
        acc.reset()
        assert acc.count == 0


class TestFactoryPlumbing:
    def test_seed_reaches_every_sampling_accumulator(self):
        kinds = ColumnKinds(
            numeric=["n"], categorical=["c"], datetime=["d"], boolean=[]
        )
        accs = build_accumulators(kinds, EngineConfig(random_seed=42))
        for name in ("n", "c", "d"):
            assert accs[name]._seed == derive_column_seed(42, name)

    def test_no_run_seed_leaves_column_seeds_unset(self):
        kinds = ColumnKinds(numeric=["n"], categorical=[], datetime=[], boolean=[])
        accs = build_accumulators(kinds, EngineConfig(random_seed=None))
        assert accs["n"]._seed is None


class TestProfileLeavesGlobalRngAlone:
    def test_numpy_state_is_untouched(self):
        df = pd.DataFrame({"a": np.arange(5_000, dtype=float)})
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        profile(df, config=_seeded_config(42))
        assert np.random.random() == expected

    def test_stdlib_state_is_untouched(self):
        df = pd.DataFrame({"a": np.arange(5_000, dtype=float)})
        pyrandom.seed(123)
        expected = pyrandom.random()
        pyrandom.seed(123)
        profile(df, config=_seeded_config(42))
        assert pyrandom.random() == expected

    def test_unseeded_profiling_is_also_invisible(self):
        """No seed configured must not mean "fall back to the global RNG"."""
        df = pd.DataFrame({"a": np.arange(5_000, dtype=float)})
        np.random.seed(321)
        expected = np.random.random()
        np.random.seed(321)
        profile(df, config=_seeded_config(None))
        assert np.random.random() == expected


class TestSeededProfilesReproduce:
    @staticmethod
    def _median(df, seed):
        return profile(df, config=_seeded_config(seed)).stats["columns"]["a"]["median"]

    def test_same_seed_same_quantiles(self):
        df = pd.DataFrame({"a": np.random.default_rng(0).lognormal(0, 1, 60_000)})
        assert self._median(df, 42) == self._median(df, 42)

    def test_different_seed_different_quantiles(self):
        df = pd.DataFrame({"a": np.random.default_rng(0).lognormal(0, 1, 60_000)})
        assert self._median(df, 42) != self._median(df, 43)

    def test_sample_preview_is_reproducible(self):
        df = pd.DataFrame({"a": np.arange(1_000, dtype=float)})
        first = profile(df, config=_seeded_config(42)).html
        second = profile(df, config=_seeded_config(42)).html
        marker = '<table class="sample-table">'
        assert first.split(marker)[1][:2000] == second.split(marker)[1][:2000]
