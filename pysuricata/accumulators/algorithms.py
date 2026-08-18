"""Core streaming algorithms for accumulators.

This module contains the fundamental streaming algorithms used by accumulators,
extracted into separate, testable, and reusable components.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sketches import ReservoirSampler


@dataclass
class PerformanceMetrics:
    """Performance tracking for algorithms."""

    update_count: int = 0
    total_update_time: float = 0.0
    last_update_time: float = 0.0
    memory_usage_bytes: int = 0

    @property
    def avg_update_time(self) -> float:
        """Average time per update in seconds."""
        return self.total_update_time / max(1, self.update_count)

    @property
    def updates_per_second(self) -> float:
        """Estimated updates per second."""
        if self.avg_update_time == 0:
            return float("inf")
        return 1.0 / self.avg_update_time

    def reset(self) -> None:
        """Clear the collected timings."""
        self.update_count = 0
        self.total_update_time = 0.0
        self.last_update_time = 0.0
        self.memory_usage_bytes = 0


class StreamingMoments:
    """Welford's algorithm for streaming statistical moments.

    This class implements numerically stable streaming computation of mean, variance,
    skewness, and kurtosis using Welford's online algorithm.
    """

    def __init__(self, enable_performance_tracking: bool = False):
        """Initialize streaming moments calculator.

        Args:
            enable_performance_tracking: Whether to track performance metrics
        """
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0  # Sum of squared differences from mean
        self._m3 = 0.0  # Third moment
        self._m4 = 0.0  # Fourth moment
        self._log_sum_pos = 0.0  # For geometric mean
        self._pos_count = 0
        # The smallest strictly positive value seen. A log axis cannot draw a
        # zero or a negative, so this is where one honestly starts -- see #258,
        # where the log histogram threw away a bin of 519 rows because 15 of
        # them were zero. `inf` is the identity for the min that folds it in.
        self._min_positive = math.inf
        self._enable_performance_tracking = enable_performance_tracking
        self._metrics = PerformanceMetrics() if enable_performance_tracking else None

    def reset(self) -> None:
        """Return to the zero-observation state."""
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._m3 = 0.0
        self._m4 = 0.0
        self._log_sum_pos = 0.0
        self._pos_count = 0
        self._min_positive = math.inf
        if self._metrics is not None:
            self._metrics.reset()

    def update(self, values: np.ndarray) -> None:
        """Update moments with new values using vectorized batch processing.

        Args:
            values: Array of numeric values to process
        """
        if self._enable_performance_tracking and self._metrics:
            start_time = time.perf_counter()

        # Filter finite values
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]

        if len(finite_values) == 0:
            return

        # Use vectorized batch processing for better performance
        self._update_vectorized(finite_values)

        if self._enable_performance_tracking and self._metrics:
            self._metrics.update_count += 1
            self._metrics.last_update_time = time.perf_counter() - start_time
            self._metrics.total_update_time += self._metrics.last_update_time

    def _update_vectorized(self, finite_values: np.ndarray) -> None:
        """Vectorized update using batch processing for Welford's algorithm.

        This is significantly faster than per-value loops for large arrays.
        """
        n_new = len(finite_values)
        if n_new == 0:
            return

        # Calculate batch statistics
        new_sum = np.sum(finite_values)
        new_mean = new_sum / n_new

        # Update geometric mean for positive values
        pos_mask = finite_values > 0
        if pos_mask.any():
            pos_values = finite_values[pos_mask]
            self._log_sum_pos += np.sum(np.log(pos_values))
            self._pos_count += pos_mask.sum()
            self._min_positive = min(self._min_positive, float(pos_values.min()))

        # For the first batch, initialize directly
        if self.count == 0:
            self.count = n_new
            self._mean = new_mean
            # Calculate initial moments for the batch
            deviations = finite_values - new_mean
            self._m2 = np.sum(deviations * deviations)
            self._m3 = np.sum(deviations * deviations * deviations)
            self._m4 = np.sum(deviations * deviations * deviations * deviations)
            return

        # Summarise the batch about its own mean, then fold it in with the
        # pairwise merge below. merge() implements Pebay's formulas, which are
        # exact for any split; the inline "simplified" M3/M4 update that used to
        # live here was not, so skew and kurtosis drifted as soon as the data
        # arrived in more than one chunk -- and disagreed with merge() on the
        # same data. Geometric-mean state is updated above, so this temporary
        # deliberately carries none of it.
        deviations = finite_values - new_mean
        squared = deviations * deviations

        batch = StreamingMoments()
        batch.count = n_new
        batch._mean = new_mean
        batch._m2 = float(np.sum(squared))
        batch._m3 = float(np.sum(squared * deviations))
        batch._m4 = float(np.sum(squared * squared))

        self.merge(batch)

    def get_statistics(self) -> dict[str, float]:
        """Get computed statistics.

        Returns:
            Dictionary containing mean, variance, std, skewness, kurtosis, etc.
        """
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "variance": 0.0,
                "std": 0.0,
                "se": 0.0,
                "cv": 0.0,
                "skew": 0.0,
                "kurtosis": 0.0,
                "gmean": 0.0,
                "min_positive": None,
            }

        # Basic statistics
        mean = self._mean
        variance = self._m2 / max(1, self.count - 1) if self.count > 1 else 0.0
        std = math.sqrt(variance)
        se = std / math.sqrt(self.count) if self.count > 0 else 0.0
        cv = std / abs(mean) if mean != 0 else 0.0

        # Higher moments. g1 and g2 are defined against the *population* second
        # moment m2/n, not the sample variance with its n-1 denominator; using
        # the latter scales skew by ((n-1)/n)^1.5 and kurtosis by ((n-1)/n)^2,
        # a bias that never converges away and does not match scipy.
        m2_pop = self._m2 / self.count if self.count > 0 else 0.0
        if self.count > 2 and m2_pop > 0:
            skew = (self._m3 / self.count) / (m2_pop**1.5)
            kurtosis = (self._m4 / self.count) / (m2_pop**2) - 3
        else:
            skew = 0.0
            kurtosis = 0.0

        # Geometric mean
        gmean = (
            math.exp(self._log_sum_pos / self._pos_count)
            if self._pos_count > 0
            else 0.0
        )

        return {
            "count": self.count,
            "mean": mean,
            "variance": variance,
            "std": std,
            "se": se,
            "cv": cv,
            "skew": skew,
            "kurtosis": kurtosis,
            "gmean": gmean,
            "min_positive": (
                self._min_positive if self._min_positive != math.inf else None
            ),
        }

    def merge(self, other: StreamingMoments) -> None:
        """Merge another StreamingMoments instance.

        Args:
            other: Another StreamingMoments instance to merge
        """
        if other.count == 0:
            return

        if self.count == 0:
            self.count = other.count
            self._mean = other._mean
            self._m2 = other._m2
            self._m3 = other._m3
            self._m4 = other._m4
            self._log_sum_pos = other._log_sum_pos
            self._pos_count = other._pos_count
            self._min_positive = other._min_positive
            return

        # Combine using Chan's algorithm
        n1, n2 = self.count, other.count
        delta = other._mean - self._mean
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2

        n = n1 + n2
        self._mean = (n1 * self._mean + n2 * other._mean) / n

        self._m4 += (
            other._m4
            + delta4 * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n * n * n)
            + 6 * delta2 * (n1 * n1 * other._m2 + n2 * n2 * self._m2) / (n * n)
            + 4 * delta * (n1 * other._m3 - n2 * self._m3) / n
        )

        self._m3 += (
            other._m3
            + delta3 * n1 * n2 * (n1 - n2) / (n * n)
            + 3 * delta * (n1 * other._m2 - n2 * self._m2) / n
        )

        self._m2 += other._m2 + delta2 * n1 * n2 / n

        self._log_sum_pos += other._log_sum_pos
        self._pos_count += other._pos_count
        self._min_positive = min(self._min_positive, other._min_positive)
        self.count = n


class ExtremeTracker:
    """Tracks extreme values with their indices using bounded heaps.

    This class efficiently tracks the minimum and maximum values along with
    their indices, maintaining only the top K extremes to control memory usage.
    Uses heapq for O(log k) insertions and O(k) space complexity.
    """

    def __init__(self, max_extremes: int = 5):
        """Initialize extreme tracker.

        Args:
            max_extremes: Maximum number of extremes to track
        """
        self.max_extremes = max_extremes
        # Use separate heaps for min and max tracking
        # Min heap: (value, index) - tracks smallest values
        self._min_heap: list[tuple[float, Any]] = []
        # Max heap: (value, index) - tracks largest values (use max-heap by negating)
        self._max_heap: list[tuple[float, Any]] = []

    def update(self, values: np.ndarray, indices: np.ndarray | None = None) -> None:
        """Update with new values and their indices.

        Uses np.argpartition to pre-filter to only the k smallest and k largest
        candidates, then pushes only those to the heaps. This reduces the Python
        loop from O(n) to O(k) per call.

        Args:
            values: Array of values
            indices: Optional array of indices corresponding to values
        """
        if len(values) == 0:
            return

        if indices is None:
            indices = np.arange(len(values))

        # Find finite values
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return

        finite_values = values[finite_mask]
        finite_indices = indices[finite_mask]

        k = self.max_extremes
        n = len(finite_values)

        if n <= 2 * k:
            # Small array — process all values directly
            for value, index in zip(finite_values, finite_indices, strict=False):
                self._add_to_min_heap(index, float(value))
                self._add_to_max_heap(index, float(value))
        else:
            # Pre-filter with np.argpartition — O(n) numpy, then O(k) Python
            # Find k smallest candidates for min heap
            min_partition = np.argpartition(finite_values, k)[:k]
            for idx in min_partition:
                self._add_to_min_heap(finite_indices[idx], float(finite_values[idx]))

            # Find k largest candidates for max heap
            max_partition = np.argpartition(finite_values, -k)[-k:]
            for idx in max_partition:
                self._add_to_max_heap(finite_indices[idx], float(finite_values[idx]))

    def _add_to_min_heap(self, index: Any, value: float) -> None:
        """Offer a value to the k-smallest set.

        Keeping the k *smallest* values means evicting the largest, so the
        right structure is a max-heap -- stored negated, since heapq only does
        min-heaps -- whose root is the value to evict. ``heappushpop`` then does
        the whole operation in O(log k).

        The heaps used to be the other way round, which forced an O(k) ``max()``
        scan, a linear search for the matching item, and a full ``heapify`` on
        every insert: O(k log k) per value on a structure whose entire purpose
        is O(log k) inserts.
        """
        if len(self._min_heap) < self.max_extremes:
            heapq.heappush(self._min_heap, (-value, index))
        else:
            # If value is not smaller than the current largest, heappushpop
            # pops straight back what it pushed, so no guard is needed.
            heapq.heappushpop(self._min_heap, (-value, index))

    def _add_to_max_heap(self, index: Any, value: float) -> None:
        """Offer a value to the k-largest set.

        Mirror of :meth:`_add_to_min_heap`: keeping the k largest means evicting
        the smallest, so a plain min-heap on the value has the right root.
        """
        if len(self._max_heap) < self.max_extremes:
            heapq.heappush(self._max_heap, (value, index))
        else:
            heapq.heappushpop(self._max_heap, (value, index))

    def get_extremes(self) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
        """Get current extreme values.

        Returns:
            Tuple of (min_pairs, max_pairs) where each pair is (index, value)
        """
        # _min_heap stores values negated (max-heap), _max_heap stores them raw.
        min_pairs = [(index, -negated) for negated, index in self._min_heap]
        min_pairs.sort(key=lambda x: x[1])  # Sort by value

        max_pairs = [(index, value) for value, index in self._max_heap]
        max_pairs.sort(key=lambda x: -x[1])  # Sort by value descending

        return min_pairs, max_pairs

    def merge(self, other: ExtremeTracker) -> None:
        """Merge another ExtremeTracker.

        Args:
            other: Another ExtremeTracker to merge
        """
        # Undo each heap's storage convention before re-offering.
        for negated, index in other._min_heap:
            self._add_to_min_heap(index, -negated)

        for value, index in other._max_heap:
            self._add_to_max_heap(index, value)


class MonotonicityDetector:
    """Detects monotonic trends in streaming data.

    This class tracks whether values are monotonically increasing or decreasing,
    which is useful for time series analysis.
    """

    def __init__(self):
        """Initialize monotonicity detector."""
        self._last_value: float | None = None
        self._mono_inc = True
        self._mono_dec = True

    def update(self, values: np.ndarray, *, all_finite: bool = False) -> None:
        """Update monotonicity detection with new values.

        The question "does any adjacent pair go the wrong way" is a sign test on
        ``np.diff``, not a Python loop -- 66x on this kernel. The only pair the
        diff cannot see is the one straddling the chunk boundary, so that one is
        compared against the carried last value first.

        Args:
            values: Array of values to check for monotonicity.
            all_finite: Set by callers that have already dropped non-finite
                values. Re-filtering costs an ``isfinite`` pass and a full copy
                of every numeric column, per chunk, for nothing -- which is most
                of what this detector costs in situ.
        """
        finite_values = values if all_finite else values[np.isfinite(values)]
        if finite_values.size == 0:
            return

        if not (self._mono_inc or self._mono_dec):
            # Both already ruled out; only the carry-over value still matters.
            self._last_value = float(finite_values[-1])
            return

        if self._last_value is not None:
            first = float(finite_values[0])
            if first < self._last_value:
                self._mono_inc = False
            if first > self._last_value:
                self._mono_dec = False

        if finite_values.size > 1:
            deltas = np.diff(finite_values)
            if self._mono_inc and bool(np.any(deltas < 0)):
                self._mono_inc = False
            if self._mono_dec and bool(np.any(deltas > 0)):
                self._mono_dec = False

        self._last_value = float(finite_values[-1])

    def get_monotonicity(self) -> tuple[bool, bool]:
        """Get monotonicity status.

        Returns:
            Tuple of (is_increasing, is_decreasing)
        """
        return self._mono_inc, self._mono_dec

    def reset(self) -> None:
        """Reset monotonicity detection."""
        self._last_value = None
        self._mono_inc = True
        self._mono_dec = True


class OutlierDetector:
    """Detects outliers using multiple methods.

    This class implements outlier detection using IQR and MAD methods,
    providing robust outlier identification for streaming data.
    """

    def __init__(
        self, methods: list[str] = None, *, rng: np.random.Generator | None = None
    ):
        """Initialize outlier detector.

        Args:
            methods: List of methods to use ('iqr', 'mad')
            rng: Generator for the internal reservoir. Defaults to an independent
                one so the detector never draws from the global RNG.
        """
        self.methods = methods or ["iqr", "mad"]
        self._rng = rng if rng is not None else np.random.default_rng()
        self._sample = ReservoirSampler(10000, rng=self._rng)

    def reset(self, *, rng: np.random.Generator | None = None) -> None:
        """Discard the collected sample.

        Args:
            rng: Generator for the fresh reservoir. Defaults to the existing one.
        """
        if rng is not None:
            self._rng = rng
        self._sample = ReservoirSampler(10000, rng=self._rng)

    def update(self, values: np.ndarray) -> None:
        """Update outlier detection with new values.

        Args:
            values: Array of values to analyze
        """
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            self._sample.add_many(finite_values)

    def detect_outliers(self, values: np.ndarray) -> dict[str, int]:
        """Detect outliers in given values.

        Args:
            values: Array of values to check for outliers

        Returns:
            Dictionary mapping method names to outlier counts
        """
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            return dict.fromkeys(self.methods, 0)

        results = {}

        if "iqr" in self.methods:
            results["iqr"] = self._detect_iqr_outliers(finite_values)

        if "mad" in self.methods:
            results["mad"] = self._detect_mad_outliers(finite_values)

        return results

    def _detect_iqr_outliers(self, values: np.ndarray) -> int:
        """Detect outliers using IQR method."""
        if len(values) < 4:
            return 0

        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return 0

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return int(np.sum((values < lower_bound) | (values > upper_bound)))

    def _detect_mad_outliers(self, values: np.ndarray) -> int:
        """Detect outliers using MAD method."""
        if len(values) < 2:
            return 0

        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return 0

        # Use 3.5 * MAD as threshold (common choice)
        threshold = 3.5 * mad
        return int(np.sum(np.abs(values - median) > threshold))
