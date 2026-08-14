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
