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

    @pytest.mark.parametrize("requested", [1, 50, 100, 500, 999])
    def test_a_request_below_the_floor_is_still_honoured(self, frame, requested):
        """#173. The floor silently raised anything under 1,000 to 1,000.

        The caller was told nothing and the value they passed never took
        effect, which makes chunk-dependent behaviour impossible to reason
        about -- and impossible to *test*, since a small deterministic fixture
        is exactly where a small chunk size is wanted. Two guards passed for
        free because of it (#139's, and #201's chunking-invariance check).
        """
        chunker = AdaptiveChunker(min_chunk_size=1_000, max_chunk_size=10_000)
        assert chunker._determine_optimal_chunk_size(frame, requested) == requested

    def test_a_request_above_the_ceiling_is_still_honoured(self, frame):
        """The ceiling was silent in the same way: a request above
        `max_chunk_size` was lowered without a word."""
        chunker = AdaptiveChunker(min_chunk_size=1_000, max_chunk_size=10_000)
        assert chunker._determine_optimal_chunk_size(frame, 999_999) == 999_999

    def test_the_bounds_still_apply_when_no_size_is_requested(self, frame):
        """Honouring the caller does not mean abandoning the heuristic bounds.

        They constrain the size the chunker picks for itself, which is the job
        they were added for; they were never a validation of the caller's.
        """
        chunker = AdaptiveChunker(min_chunk_size=7_000, max_chunk_size=8_000)
        chosen = chunker._determine_optimal_chunk_size(frame, 0)
        assert 7_000 <= chosen <= 8_000

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


class TestChunkSizeGovernsHowManyChunksAppear:
    """#173, end to end: counting the chunks a real `profile()` run consumes.

    The unit tests above check what the chunker *decides*. This counts what the
    engine actually does, which is the property no test could express while the
    floor was in place — and the property both #139's guard and #201's
    chunking-invariance test needed and silently did not have.

    The table is the one measured in #173; every row read "no" before the fix.
    """

    @staticmethod
    def _chunk_sizes(rows: int, requested: int) -> list[int]:
        from pysuricata.compute.adapters import pandas as adapter_module

        seen: list[int] = []
        original = adapter_module.PandasAdapter.consume_chunk

        def counting(self, data, *args, **kwargs):
            seen.append(len(data))
            return original(self, data, *args, **kwargs)

        adapter_module.PandasAdapter.consume_chunk = counting
        try:
            summarize(
                pd.DataFrame({"x": np.arange(float(rows))}),
                seed=0,
                chunk_size=requested,
            )
        finally:
            adapter_module.PandasAdapter.consume_chunk = original
        return seen

    @pytest.mark.parametrize(
        "rows,requested,expected_chunks",
        [
            (5_000, 100, 50),
            (5_000, 500, 10),
            (5_000, 999, 6),  # 5 full chunks and a remainder
            (5_000, 1_000, 5),
            (5_000, 1_250, 4),
            (10_000, 250, 40),
        ],
    )
    def test_the_requested_size_governs_the_chunk_count(
        self, rows, requested, expected_chunks
    ):
        sizes = self._chunk_sizes(rows, requested)

        assert len(sizes) == expected_chunks
        assert max(sizes) == min(requested, rows), (
            f"asked for {requested} rows per chunk, largest chunk was {max(sizes)}"
        )
        assert sum(sizes) == rows, "every row must still be consumed exactly once"

    def test_a_chunk_size_of_one_is_the_callers_choice(self):
        """Pathological, and not the library's to override — the argument for
        honouring the request is that a silent correction is worse than a slow
        run the caller asked for."""
        sizes = self._chunk_sizes(200, 1)

        assert len(sizes) == 200
        assert set(sizes) == {1}

    def test_a_size_larger_than_the_frame_yields_one_chunk(self):
        sizes = self._chunk_sizes(500, 2_000_000)

        assert sizes == [500]
