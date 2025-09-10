"""Core streaming algorithms for accumulators.

This module contains the fundamental streaming algorithms used by accumulators,
extracted into separate, testable, and reusable components.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .sketches import KMV, ReservoirSampler


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
        self._enable_performance_tracking = enable_performance_tracking
        self._metrics = PerformanceMetrics() if enable_performance_tracking else None

    def update(self, values: np.ndarray) -> None:
        """Update moments with new values.

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

        # Update moments using Welford's algorithm
        for value in finite_values:
            self.count += 1
            delta = value - self._mean
            delta_n = delta / self.count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (self.count - 1)

            # Update mean
            self._mean += delta_n

            # Update higher moments
            self._m4 += (
                term1 * delta_n2 * (self.count * self.count - 3 * self.count + 3)
                + 6 * delta_n2 * self._m2
                - 4 * delta_n * self._m3
            )
            self._m3 += term1 * delta_n * (self.count - 2) - 3 * delta_n * self._m2
            self._m2 += term1

            # Track positive values for geometric mean
            if value > 0:
                self._log_sum_pos += math.log(value)
                self._pos_count += 1

        if self._enable_performance_tracking and self._metrics:
            self._metrics.update_count += 1
            self._metrics.last_update_time = time.perf_counter() - start_time
            self._metrics.total_update_time += self._metrics.last_update_time

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
            }

        # Basic statistics
        mean = self._mean
        variance = self._m2 / max(1, self.count - 1) if self.count > 1 else 0.0
        std = math.sqrt(variance)
        se = std / math.sqrt(self.count) if self.count > 0 else 0.0
        cv = std / abs(mean) if mean != 0 else 0.0

        # Higher moments
        if self.count > 2 and variance > 0:
            skew = (self._m3 / self.count) / (variance**1.5)
            kurtosis = (self._m4 / self.count) / (variance**2) - 3
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
        self.count = n


class ExtremeTracker:
    """Tracks extreme values with their indices.

    This class efficiently tracks the minimum and maximum values along with
    their indices, maintaining only the top K extremes to control memory usage.
    """

    def __init__(self, max_extremes: int = 5):
        """Initialize extreme tracker.

        Args:
            max_extremes: Maximum number of extremes to track
        """
        self.max_extremes = max_extremes
        self._min_pairs: List[Tuple[Any, float]] = []
        self._max_pairs: List[Tuple[Any, float]] = []

    def update(self, values: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        """Update with new values and their indices.

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

        # Find extremes using argpartition for efficiency
        k = min(self.max_extremes, len(finite_values))

        if k > 0:
            # Find minimum values
            min_indices = np.argpartition(finite_values, k - 1)[:k]
            for i in min_indices:
                self._min_pairs.append((finite_indices[i], float(finite_values[i])))

            # Find maximum values, excluding those already in min_pairs
            max_indices = np.argpartition(-finite_values, k - 1)[:k]
            existing_min_indices = {finite_indices[i] for i in min_indices}

            for i in max_indices:
                if finite_indices[i] not in existing_min_indices:
                    self._max_pairs.append((finite_indices[i], float(finite_values[i])))

        # Remove duplicates within each list and keep only the best extremes
        # Remove duplicates from min_pairs (keep first occurrence)
        seen_min = set()
        unique_min_pairs = []
        for pair in self._min_pairs:
            if pair not in seen_min:
                seen_min.add(pair)
                unique_min_pairs.append(pair)
        self._min_pairs = unique_min_pairs

        # Remove duplicates from max_pairs (keep first occurrence)
        seen_max = set()
        unique_max_pairs = []
        for pair in self._max_pairs:
            if pair not in seen_max:
                seen_max.add(pair)
                unique_max_pairs.append(pair)
        self._max_pairs = unique_max_pairs

        # Sort and trim to max_extremes
        self._min_pairs.sort(key=lambda x: x[1])
        if len(self._min_pairs) > self.max_extremes:
            self._min_pairs = self._min_pairs[: self.max_extremes]

        self._max_pairs.sort(key=lambda x: -x[1])
        if len(self._max_pairs) > self.max_extremes:
            self._max_pairs = self._max_pairs[: self.max_extremes]

    def get_extremes(self) -> Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]:
        """Get current extreme values.

        Returns:
            Tuple of (min_pairs, max_pairs) where each pair is (index, value)
        """
        return self._min_pairs.copy(), self._max_pairs.copy()

    def merge(self, other: ExtremeTracker) -> None:
        """Merge another ExtremeTracker.

        Args:
            other: Another ExtremeTracker to merge
        """
        self._min_pairs.extend(other._min_pairs)
        self._max_pairs.extend(other._max_pairs)

        # Remove duplicates and re-sort
        # First, remove duplicates within each list
        self._min_pairs = list(dict.fromkeys(self._min_pairs))  # Preserves order
        self._max_pairs = list(dict.fromkeys(self._max_pairs))  # Preserves order

        # Remove items from max_pairs that are already in min_pairs
        min_indices = {idx for idx, _ in self._min_pairs}
        self._max_pairs = [
            (idx, val) for idx, val in self._max_pairs if idx not in min_indices
        ]

        # Re-sort and trim
        self._min_pairs.sort(key=lambda x: x[1])
        if len(self._min_pairs) > self.max_extremes:
            self._min_pairs = self._min_pairs[: self.max_extremes]

        self._max_pairs.sort(key=lambda x: -x[1])
        if len(self._max_pairs) > self.max_extremes:
            self._max_pairs = self._max_pairs[: self.max_extremes]


class MonotonicityDetector:
    """Detects monotonic trends in streaming data.

    This class tracks whether values are monotonically increasing or decreasing,
    which is useful for time series analysis.
    """

    def __init__(self):
        """Initialize monotonicity detector."""
        self._last_value: Optional[float] = None
        self._mono_inc = True
        self._mono_dec = True

    def update(self, values: np.ndarray) -> None:
        """Update monotonicity detection with new values.

        Args:
            values: Array of values to check for monotonicity
        """
        finite_values = values[np.isfinite(values)]

        for value in finite_values:
            if self._last_value is not None:
                if value < self._last_value:
                    self._mono_inc = False
                if value > self._last_value:
                    self._mono_dec = False

            self._last_value = value

    def get_monotonicity(self) -> Tuple[bool, bool]:
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

    def __init__(self, methods: List[str] = None):
        """Initialize outlier detector.

        Args:
            methods: List of methods to use ('iqr', 'mad')
        """
        self.methods = methods or ["iqr", "mad"]
        self._sample = ReservoirSampler(10000)  # Sample for outlier detection

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
            return {method: 0 for method in self.methods}

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
