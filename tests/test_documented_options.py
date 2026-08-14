"""Documented options must do what they document.

Each of these was declared, documented and validated, then ignored: `columns`
never reached the engine, `corr_max_cols` was copied into the config and never
read, and `chunk_size` was blended with a heuristic so the caller never got the
size they asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.api import ComputeOptions, ProfileConfig
from pysuricata.compute.processing.chunking import AdaptiveChunker
from pysuricata.config import EngineConfig


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.standard_normal(2_000) for c in "abcd"})


class TestColumnsOption:
    def test_subset_is_honoured(self, frame):
        cfg = ProfileConfig(compute=ComputeOptions(columns=["a", "b"]))
        assert sorted(summarize(frame, config=cfg)["columns"]) == ["a", "b"]

    def test_none_profiles_everything(self, frame):
        assert sorted(summarize(frame)["columns"]) == ["a", "b", "c", "d"]

    def test_subset_reaches_the_engine_config(self):
        assert EngineConfig.from_options(
            ComputeOptions(columns=["a", "b"])
        ).columns == ("a", "b")
        assert EngineConfig.from_options(ComputeOptions()).columns is None

    def test_unknown_names_are_ignored_not_fatal(self, frame):
        cfg = ProfileConfig(compute=ComputeOptions(columns=["a", "nonexistent"]))
        assert sorted(summarize(frame, config=cfg)["columns"]) == ["a"]

    def test_subset_applies_to_a_streamed_source(self):
        def gen():
            rng = np.random.default_rng(1)
            for _ in range(3):
                yield pd.DataFrame({c: rng.standard_normal(500) for c in "abcd"})

        cfg = ProfileConfig(compute=ComputeOptions(columns=["a", "c"]))
        result = summarize(gen(), config=cfg)
        assert sorted(result["columns"]) == ["a", "c"]
        assert result["dataset"]["rows_est"] == 1_500

    def test_statistics_are_unaffected_by_subsetting(self, frame):
        cfg = ProfileConfig(compute=ComputeOptions(columns=["a"]))
        subset = summarize(frame, config=cfg)["columns"]["a"]
        full = summarize(frame)["columns"]["a"]
        assert subset["mean"] == pytest.approx(full["mean"])
        assert subset["count"] == full["count"]


class TestCorrMaxColsOption:
    @pytest.fixture
    def correlated(self):
        rng = np.random.default_rng(2)
        base = rng.standard_normal(2_000)
        return pd.DataFrame(
            {f"c{i}": base * 0.95 + rng.standard_normal(2_000) * 0.1 for i in range(30)}
        )

    @pytest.mark.parametrize("cap", [2, 5, 30])
    def test_cap_limits_the_columns_correlated(self, correlated, cap):
        cfg = ProfileConfig(compute=ComputeOptions(corr_max_cols=cap))
        result = summarize(correlated, config=cfg)["columns"]
        reporting = [c for c, v in result.items() if v.get("corr_top")]
        assert len(reporting) == cap

    def test_cap_above_the_column_count_is_a_no_op(self, correlated):
        cfg = ProfileConfig(compute=ComputeOptions(corr_max_cols=1_000))
        result = summarize(correlated, config=cfg)["columns"]
        assert len([c for c, v in result.items() if v.get("corr_top")]) == 30


class TestChunkSizeOption:
    @pytest.mark.parametrize("requested", [1_000, 3_000, 50_000, 250_000])
    def test_requested_size_is_used_exactly(self, frame, requested):
        """Previously blended as 0.7*optimal + 0.3*requested, so the caller
        never got the size they asked for."""
        chunker = AdaptiveChunker()
        assert chunker._determine_optimal_chunk_size(frame, requested) == requested

    def test_request_is_clamped_to_the_chunker_bounds(self, frame):
        chunker = AdaptiveChunker(min_chunk_size=1_000, max_chunk_size=10_000)
        assert chunker._determine_optimal_chunk_size(frame, 50) == 1_000
        assert chunker._determine_optimal_chunk_size(frame, 999_999) == 10_000

    def test_no_request_falls_back_to_adaptive_sizing(self, frame):
        chunker = AdaptiveChunker()
        assert chunker._determine_optimal_chunk_size(frame, 0) > 0

    def test_chunk_size_actually_splits_the_frame(self):
        """End to end: the requested size must govern how many chunks appear."""
        df = pd.DataFrame({"a": np.arange(9_000.0)})
        chunker = AdaptiveChunker()
        result = chunker.chunks_from_source(df, 3_000, False)
        assert result.success
        assert [len(c) for c in result.data] == [3_000, 3_000, 3_000]
