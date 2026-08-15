"""Extremes, monotonicity, and the exactness of the reported min and max.

Three changes meet here. Monotonicity became a sign test on ``np.diff``, which
has to keep seeing the pair that straddles a chunk boundary. The extreme heaps
were inverted — a max-heap's job done by a min-heap plus an O(k) scan — and the
fix flips their storage convention, so the pairs coming back out must still
carry the right sign. And the reported minimum and maximum now come from the
tracker that sees every value rather than from the reservoir that samples them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators.algorithms import ExtremeTracker, MonotonicityDetector
from pysuricata.accumulators.numeric import NumericAccumulator


def _fold_tracker(values: np.ndarray, k: int = 5, n_chunks: int = 1) -> ExtremeTracker:
    tracker = ExtremeTracker(k)
    offset = 0
    for chunk in np.array_split(values, n_chunks):
        tracker.update(chunk, np.arange(offset, offset + len(chunk)))
        offset += len(chunk)
    return tracker


class TestExtremeTracker:
    VALUES = np.concatenate(
        [np.arange(1_000, dtype=float), np.arange(1_000, dtype=float) + 0.5]
    )

    @pytest.mark.parametrize("n_chunks", [1, 3, 17])
    def test_it_finds_the_true_k_smallest_and_largest(self, n_chunks):
        rng = np.random.default_rng(0)
        values = self.VALUES.copy()
        rng.shuffle(values)
        mins, maxs = _fold_tracker(values, 5, n_chunks).get_extremes()
        assert [v for _, v in mins] == sorted(values)[:5]
        assert [v for _, v in maxs] == sorted(values)[-5:][::-1]

    def test_indices_point_at_the_right_rows(self):
        values = np.arange(500, dtype=float)
        values[123] = -1.0
        values[456] = 9_999.0
        mins, maxs = _fold_tracker(values, 3, 4).get_extremes()
        assert mins[0] == (123, -1.0)
        assert maxs[0] == (456, 9_999.0)

    def test_the_heaps_stay_bounded(self):
        tracker = _fold_tracker(np.arange(50_000, dtype=float), 5, 25)
        assert len(tracker._min_heap) == 5
        assert len(tracker._max_heap) == 5

    def test_a_short_column_keeps_everything_it_has(self):
        mins, maxs = _fold_tracker(np.array([3.0, 1.0, 2.0]), 5).get_extremes()
        assert [v for _, v in mins] == [1.0, 2.0, 3.0]
        assert [v for _, v in maxs] == [3.0, 2.0, 1.0]

    def test_merge_preserves_both_ends(self):
        left = _fold_tracker(np.arange(0, 100, dtype=float), 3)
        right = _fold_tracker(np.arange(100, 200, dtype=float), 3)
        left.merge(right)
        mins, maxs = left.get_extremes()
        assert [v for _, v in mins] == [0.0, 1.0, 2.0]
        assert [v for _, v in maxs] == [199.0, 198.0, 197.0]

    def test_non_finite_values_are_ignored(self):
        values = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        mins, maxs = _fold_tracker(values, 2).get_extremes()
        assert [v for _, v in mins] == [1.0, 2.0]
        assert [v for _, v in maxs] == [2.0, 1.0]


class TestMonotonicity:
    @staticmethod
    def _detect(values, n_chunks: int = 1):
        det = MonotonicityDetector()
        for chunk in np.array_split(np.asarray(values, dtype=float), n_chunks):
            det.update(chunk)
        return det.get_monotonicity()

    @pytest.mark.parametrize("n_chunks", [1, 2, 9])
    def test_increasing(self, n_chunks):
        assert self._detect(np.arange(1_000, dtype=float), n_chunks) == (True, False)

    @pytest.mark.parametrize("n_chunks", [1, 2, 9])
    def test_decreasing(self, n_chunks):
        assert self._detect(np.arange(1_000, 0, -1, dtype=float), n_chunks) == (
            False,
            True,
        )

    def test_constant_is_both(self):
        assert self._detect(np.zeros(100)) == (True, True)

    def test_a_single_value_is_both(self):
        assert self._detect([7.0]) == (True, True)

    def test_neither(self):
        assert self._detect([1.0, 3.0, 2.0, 4.0]) == (False, False)

    def test_the_break_across_a_chunk_boundary_is_seen(self):
        """np.diff cannot see the pair that straddles two chunks."""
        det = MonotonicityDetector()
        det.update(np.array([1.0, 2.0, 3.0]))
        det.update(np.array([2.5, 4.0]))  # 3.0 -> 2.5 breaks increasing
        assert det.get_monotonicity() == (False, False)

    def test_a_boundary_that_does_not_break_it(self):
        det = MonotonicityDetector()
        det.update(np.array([1.0, 2.0, 3.0]))
        det.update(np.array([3.0, 4.0]))  # equal across the seam: still increasing
        assert det.get_monotonicity() == (True, False)

    def test_non_finite_values_do_not_break_the_run(self):
        assert self._detect([1.0, np.nan, 2.0, np.inf, 3.0]) == (True, False)

    def test_an_all_nan_chunk_does_not_reset_the_carry(self):
        det = MonotonicityDetector()
        det.update(np.array([5.0, 6.0]))
        det.update(np.array([np.nan, np.nan]))
        det.update(np.array([4.0]))  # 6.0 -> 4.0, still a break
        assert det.get_monotonicity() == (False, False)

    def test_it_matches_the_elementwise_definition(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            values = rng.integers(0, 5, 40).astype(float)
            expected = (
                bool(np.all(np.diff(values) >= 0)),
                bool(np.all(np.diff(values) <= 0)),
            )
            assert self._detect(values, 7) == expected


class TestReportedMinMaxAreExact:
    def test_a_column_larger_than_the_sample_still_reports_the_true_ends(self):
        """The reservoir holds 20,000 of these; the tracker sees all of them."""
        values = np.arange(200_000, dtype=float)
        np.random.default_rng(0).shuffle(values)
        acc = NumericAccumulator("x")
        for chunk in np.array_split(values, 5):
            acc.update(np.ascontiguousarray(chunk))
        summary = acc.finalize()
        assert summary.min == 0.0
        assert summary.max == 199_999.0

    def test_the_headline_figures_agree_with_the_extremes_table(self):
        values = np.random.default_rng(1).lognormal(0, 2, 120_000)
        acc = NumericAccumulator("x")
        acc.update(values)
        summary = acc.finalize()
        assert summary.min == summary.min_items[0][1]
        assert summary.max == summary.max_items[0][1]

    def test_end_to_end_through_a_stream(self):
        from pysuricata import summarize

        values = np.random.default_rng(2).lognormal(0, 1, 85_000)

        def gen():
            yield pd.DataFrame({"measure": values[:45_000]})
            yield pd.DataFrame({"measure": values[45_000:]})

        col = summarize(gen())["columns"]["measure"]
        assert col["min"] == pytest.approx(values.min())
        assert col["max"] == pytest.approx(values.max())


class TestNoSecondOutlierReservoir:
    def test_the_accumulator_keeps_one_sample_not_two(self):
        acc = NumericAccumulator("x")
        assert not hasattr(acc, "_outlier_detector")

    def test_outlier_counts_are_still_reported(self):
        values = np.concatenate(
            [np.random.default_rng(0).standard_normal(10_000), np.full(50, 500.0)]
        )
        summary = NumericAccumulator("x")
        summary.update(values)
        result = summary.finalize()
        assert result.outliers_iqr > 0
        assert result.outliers_mod_zscore > 0

    def test_disabling_outlier_detection_zeroes_them(self):
        from pysuricata.accumulators.config import NumericConfig

        acc = NumericAccumulator("x", NumericConfig(enable_outlier_detection=False))
        acc.update(np.concatenate([np.zeros(1_000), np.full(10, 1e6)]))
        result = acc.finalize()
        assert result.outliers_iqr == 0
        assert result.outliers_mod_zscore == 0
