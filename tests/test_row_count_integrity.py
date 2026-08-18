"""Row counting must stay exact even when row hashing fails.

``RowKMV.rows`` is what the report prints as "Rows" and what
``missing_cells_pct`` divides by. The duplicate estimate is explicitly
approximate and may degrade; the row count may not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators.sketches import RowKMV


class _ExplodingFrame(pd.DataFrame):
    """A DataFrame whose columns cannot be hashed.

    ``pd.util.hash_pandas_object`` raises on the column values, which is the
    real-world trigger for the fallback path: an object column holding
    something unhashable.
    """

    @property
    def _constructor(self):
        return _ExplodingFrame


def _unhashable_frame(n_rows: int) -> pd.DataFrame:
    """A frame that defeats vectorised row hashing.

    A column of lists is the ordinary way this happens: pandas cannot hash
    unhashable cell values, so ``hash_pandas_object`` raises and the accumulator
    falls back.
    """
    return pd.DataFrame(
        {
            "ok": np.arange(n_rows),
            "bad": [[i] for i in range(n_rows)],
        }
    )


class TestRowCountExactness:
    @pytest.mark.parametrize("n_rows", [1, 100, 2_000, 2_001, 50_000])
    def test_row_count_is_exact_on_the_happy_path(self, n_rows):
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame({"a": np.arange(n_rows)}))
        assert rk.rows == n_rows

    @pytest.mark.parametrize("n_rows", [2_001, 10_000, 50_000])
    def test_row_count_is_exact_when_hashing_fails(self, n_rows):
        """Regression: the fallback added at most 2,000 rows per chunk, so a
        50,000-row chunk reported 2,000 and every missing-percentage that
        divided by it was wrong."""
        rk = RowKMV()
        rk.update_from_pandas(_unhashable_frame(n_rows))
        assert rk.rows == n_rows

    def test_row_count_is_exact_across_many_failing_chunks(self):
        rk = RowKMV()
        for _ in range(5):
            rk.update_from_pandas(_unhashable_frame(10_000))
        assert rk.rows == 50_000

    def test_mixed_success_and_failure_chunks_still_count_every_row(self):
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame({"a": np.arange(30_000)}))
        rk.update_from_pandas(_unhashable_frame(20_000))
        assert rk.rows == 50_000

    def test_failure_marks_the_duplicate_estimate_as_degraded(self):
        """The row count stays exact, but the sketch has seen only a sample, so
        the duplicate estimate must say so rather than be quietly wrong."""
        clean = RowKMV()
        clean.update_from_pandas(pd.DataFrame({"a": np.arange(10_000)}))
        assert not clean.duplicates_degraded

        degraded = RowKMV()
        degraded.update_from_pandas(_unhashable_frame(10_000))
        assert degraded.duplicates_degraded

    def test_duplicate_count_never_exceeds_the_row_count(self):
        rk = RowKMV()
        rk.update_from_pandas(_unhashable_frame(10_000))
        dups, pct = rk.approx_duplicates()
        assert 0 <= dups <= rk.rows
        assert 0.0 <= pct <= 100.0

    def test_empty_frame_is_safe(self):
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame({"a": []}))
        assert rk.rows == 0
        assert rk.approx_duplicates() == (0, 0.0)


class TestDuplicateEstimateStillWorks:
    def test_all_distinct_rows_report_no_duplicates(self):
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame({"a": np.arange(50_000)}))
        dups, pct = rk.approx_duplicates()
        assert pct < 5.0

    def test_heavily_duplicated_rows_are_detected(self):
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame({"a": np.tile(np.arange(100), 500)}))
        dups, pct = rk.approx_duplicates()
        assert pct > 90.0


class TestZeroColumnFramesMatchPandas:
    """#312. A frame with no columns has nothing to hash, which is a
    different problem from a frame whose columns fail to hash -- routing it
    through the same fallback landed on one distinct signature for every
    row and reported 90% duplicates on a 10-row frame pandas calls clean.
    """

    def test_a_zero_column_frame_reports_no_duplicates(self):
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame(index=range(10)))
        assert rk.rows == 10
        assert rk.approx_duplicates() == (0, 0.0)

    def test_it_is_reported_exact_not_degraded(self):
        """Nothing failed to hash -- there was nothing to hash -- so this
        must not carry the same flag a genuinely unhashable chunk does."""
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame(index=range(10)))
        assert not rk.duplicates_degraded

    @pytest.mark.parametrize("n_rows", [1, 100, 5_000])
    def test_the_row_count_is_exact_at_every_size(self, n_rows):
        """Within the sketch's default `k` (8,192), every row's synthetic
        identity is counted exactly -- no estimation error to allow for."""
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame(index=range(n_rows)))
        assert rk.rows == n_rows
        assert rk.approx_duplicates() == (0, 0.0)

    def test_beyond_k_the_estimate_stays_small_not_catastrophic(self):
        """Past the exact-counting budget the sketch is estimating, same as
        for any other content, and the ~2% error bound applies -- the point
        of this test is that it stays in that range rather than repeating
        #312's 90%, not that it hits exactly zero."""
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame(index=range(20_000)))
        assert rk.rows == 20_000
        _, pct = rk.approx_duplicates()
        assert pct < 5.0

    def test_two_zero_column_chunks_do_not_collide_with_each_other(self):
        """The synthetic per-row identity has to stay unique across chunks,
        or a second chunk's rows would look like duplicates of the first's --
        the same failure shape as #312 itself, one level up. Kept within `k`
        so the check is exact rather than budgeted for sketch noise."""
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame(index=range(3_000)))
        rk.update_from_pandas(pd.DataFrame(index=range(3_000)))
        assert rk.rows == 6_000
        assert rk.approx_duplicates() == (0, 0.0)

    def test_a_zero_column_chunk_amid_real_columns_does_not_corrupt_them(self):
        """Not a realistic streaming shape (a frame's column count does not
        change chunk to chunk), but the accumulator must not silently mix up
        its running state if it ever happened -- rows stay exact and the
        duplicate estimate for the real content is undisturbed."""
        rk = RowKMV()
        rk.update_from_pandas(pd.DataFrame({"a": np.arange(1_000)}))
        rk.update_from_pandas(pd.DataFrame(index=range(500)))
        assert rk.rows == 1_500
        dups, _ = rk.approx_duplicates()
        assert 0 <= dups <= rk.rows
