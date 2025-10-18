"""High-performance numeric accumulator optimized for big data analytics.

This module provides a production-ready, scalable implementation of the numeric accumulator
using advanced algorithmic composition, vectorized operations, and performance optimizations
designed for processing massive numerical datasets efficiently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .algorithms import (
    ExtremeTracker,
    MonotonicityDetector,
    OutlierDetector,
    PerformanceMetrics,
    StreamingMoments,
)
from .config import NumericConfig
from .sketches import KMV, MisraGries, ReservoirSampler, StreamingHistogram, mad


@dataclass
class NumericSummary:
    """Comprehensive summary statistics for numeric data.

    This dataclass contains all computed statistics for a numeric column,
    including basic statistics, distribution measures, and quality indicators
    optimized for big data analytics.
    """

    name: str
    count: int
    missing: int
    unique_est: int
    mean: float
    std: float
    variance: float
    se: float
    cv: float
    gmean: float
    min: float
    q1: float
    median: float
    q3: float
    iqr: float
    mad: float
    skew: float
    kurtosis: float
    jb_chi2: float
    max: float
    zeros: int
    negatives: int
    outliers_iqr: int
    outliers_mod_zscore: int
    approx: bool
    inf: int
    # Advanced analytics (approximate)
    int_like: bool = False
    unique_ratio_approx: float = float("nan")
    hist_counts: Optional[List[int]] = None
    top_values: List[Tuple[float, int]] = field(default_factory=list)
    # Reservoir sample for advanced analytics
    sample_vals: Optional[List[float]] = None
    # True distribution histogram data
    true_histogram_edges: Optional[List[float]] = None
    true_histogram_counts: Optional[List[int]] = None
    # Quality metrics
    heap_pct: float = float("nan")
    gran_decimals: Optional[int] = None
    gran_step: Optional[float] = None
    bimodal: bool = False
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    # System metrics
    mem_bytes: int = 0
    mono_inc: bool = False
    mono_dec: bool = False
    dtype_str: str = "numeric"
    corr_top: List[Tuple[str, float]] = field(default_factory=list)
    sample_scale: float = 1.0
    # Extremes with global indices
    min_items: List[Tuple[Any, float]] = field(default_factory=list)
    max_items: List[Tuple[Any, float]] = field(default_factory=list)
    # Chunk metadata for spectrum visualization
    chunk_metadata: Optional[List[Tuple[int, int, int]]] = (
        None  # (start_row, end_row, missing_count)
    )
    corr_threshold: float = 0.5  # Threshold used for correlation filtering


class NumericAccumulator:
    """Production-grade numeric accumulator optimized for big data analytics.

    This accumulator leverages advanced algorithmic composition and vectorized operations
    to achieve maximum performance on large-scale numerical datasets while maintaining
    precision, reliability, and comprehensive statistical analysis capabilities.
    """

    def __init__(self, name: str, config: Optional[NumericConfig] = None):
        """Initialize numeric accumulator with optimized components.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
        """
        self.name = name
        self.config = config or NumericConfig()

        # Core state tracking
        self.count = 0
        self.missing = 0
        self.zeros = 0
        self.negatives = 0
        self.inf = 0
        self._int_like_all = True
        self._dtype_str = "numeric"
        self._corr_top: List[Tuple[str, float]] = []
        self._corr_threshold: float = 0.5

        # Memory tracking for big data optimization
        self._bytes_seen = 0

        # High-performance algorithm components
        self._moments = StreamingMoments(
            enable_performance_tracking=self.config.enable_memory_tracking
        )
        self._sample = ReservoirSampler(self.config.sample_size)
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._extremes = ExtremeTracker(self.config.max_extremes)
        self._topk = MisraGries(self.config.top_k_size)

        # Streaming histogram for true distribution
        self._streaming_histogram = StreamingHistogram(bins=25)

        # Optional advanced analytics components
        self._monotonicity = (
            MonotonicityDetector()
            if self.config.enable_monotonicity_detection
            else None
        )
        self.enable_outlier_detection = self.config.enable_outlier_detection
        self._outlier_detector = (
            OutlierDetector() if self.config.enable_outlier_detection else None
        )

        # Performance monitoring for production environments
        self._performance_metrics = (
            PerformanceMetrics() if self.config.enable_memory_tracking else None
        )

    def set_dtype(self, dtype_str: str) -> None:
        """Set the data type string efficiently.

        Args:
            dtype_str: String representation of the data type
        """
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "numeric"

    def set_corr_top(self, items: List[Tuple[str, float]]) -> None:
        """Set top correlated columns for analytics.

        Args:
            items: List of (column_name, correlation) tuples
        """
        self._corr_top = list(items or [])

    def set_corr_threshold(self, threshold: float) -> None:
        """Set correlation threshold for analytics.

        Args:
            threshold: Minimum absolute correlation to report
        """
        self._corr_threshold = float(threshold)

    @property
    def unique_est(self) -> int:
        """Get unique count estimate for compatibility."""
        return self._uniques.estimate()

    def update(self, arr: Sequence[Any]) -> None:
        """Update accumulator with new values using optimized vectorized processing.

        Args:
            arr: Sequence of values to process
        """
        # Handle empty arrays efficiently
        if len(arr) == 0:
            return

        # Performance tracking for production monitoring
        if self._performance_metrics:
            import time

            start_time = time.perf_counter()

        # Convert to numpy array for maximum performance
        try:
            values = np.asarray(arr, dtype=float)
            # Count missing values in the original array
            self._count_missing_values(arr)
        except (ValueError, TypeError):
            # Robust error handling for mixed-type data
            values = self._convert_to_numeric(arr)

        # Process values with optimized algorithms
        self._process_values(values)

        # Update performance metrics for monitoring
        if self._performance_metrics:
            self._performance_metrics.update_count += 1
            self._performance_metrics.last_update_time = (
                time.perf_counter() - start_time
            )
            self._performance_metrics.total_update_time += (
                self._performance_metrics.last_update_time
            )

    def _convert_to_numeric(self, arr: Sequence[Any]) -> np.ndarray:
        """Convert array to numeric with robust error handling for big data.

        Args:
            arr: Input array

        Returns:
            Numeric array with NaN for non-numeric values
        """
        result = np.full(len(arr), np.nan, dtype=float)

        for i, value in enumerate(arr):
            if value is None:
                self.missing += 1
                continue

            try:
                if isinstance(value, (int, float)):
                    if math.isnan(value):
                        self.missing += 1
                    elif math.isinf(value):
                        self.inf += 1
                        result[i] = value
                    else:
                        result[i] = float(value)
                else:
                    # Robust string to float conversion
                    result[i] = float(value)
            except (ValueError, TypeError):
                self.missing += 1

        return result

    def _count_missing_values(self, arr: Sequence[Any]) -> None:
        """Count missing values in the original array with optimized checks.

        Args:
            arr: Original input array
        """
        for value in arr:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                self.missing += 1
            elif isinstance(value, float) and math.isinf(value):
                self.inf += 1

    def _process_values(self, values: np.ndarray) -> None:
        """Process numeric values through all algorithm components with vectorized operations.

        Args:
            values: Numeric array to process
        """
        # Count special values using vectorized operations
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]

        if len(finite_values) == 0:
            return

        # Update basic counts efficiently
        self.count += len(finite_values)
        self.zeros += int(np.sum(finite_values == 0))
        self.negatives += int(np.sum(finite_values < 0))

        # Check if all values are integer-like for type inference
        if self._int_like_all:
            self._int_like_all = all(abs(v - round(v)) < 1e-10 for v in finite_values)

        # Update algorithm components with vectorized operations
        self._moments.update(finite_values)
        self._sample.add_many(finite_values)

        # Update streaming histogram for true distribution
        self._streaming_histogram.add_many(finite_values)

        # Batch update unique estimates and top values
        for value in finite_values:
            self._uniques.add(value)
            self._topk.add(value)

        # Update extremes with efficient index tracking
        indices = np.arange(len(values))[finite_mask]
        self._extremes.update(finite_values, indices)

        # Update optional advanced analytics components
        if self._monotonicity:
            self._monotonicity.update(finite_values)

        if self._outlier_detector:
            self._outlier_detector.update(finite_values)

    def update_extremes(
        self, pairs_min: List[Tuple[Any, float]], pairs_max: List[Tuple[Any, float]]
    ) -> None:
        """Update extreme values from external source with batch processing.

        Args:
            pairs_min: List of (index, value) pairs for minimum values
            pairs_max: List of (index, value) pairs for maximum values
        """
        # Convert to numpy arrays for efficient processing
        if pairs_min:
            min_values = np.array([v for _, v in pairs_min])
            min_indices = np.array([i for i, _ in pairs_min])
            self._extremes.update(min_values, min_indices)

        if pairs_max:
            max_values = np.array([v for _, v in pairs_max])
            max_indices = np.array([i for i, _ in pairs_max])
            self._extremes.update(max_values, max_indices)

    def add_mem(self, n: int) -> None:
        """Add to memory usage tracking for big data optimization.

        Args:
            n: Number of bytes to add
        """
        try:
            self._bytes_seen += int(n)
        except (ValueError, TypeError):
            pass

    def finalize(
        self, chunk_metadata: Optional[List[Tuple[int, int, int]]] = None
    ) -> NumericSummary:
        """Finalize accumulator and return comprehensive summary statistics.

        Args:
            chunk_metadata: Optional list of chunk metadata tuples (start_row, end_row, missing_count)

        Returns:
            NumericSummary containing all computed statistics
        """
        # Get comprehensive statistics from streaming moments
        stats = self._moments.get_statistics()

        # Get quantiles from reservoir sample
        sample_values = self._sample.values()
        quantiles = self._compute_quantiles(sample_values)

        # Get extremes with global indices
        min_pairs, max_pairs = self._extremes.get_extremes()

        # Get advanced monotonicity analysis if enabled
        mono_inc, mono_dec = False, False
        if self._monotonicity:
            mono_inc, mono_dec = self._monotonicity.get_monotonicity()

        # Get outlier detection results if enabled
        outliers_iqr, outliers_mod_zscore = 0, 0
        if self.enable_outlier_detection and sample_values:
            sample_arr = np.array(sample_values)
            q1, q3 = np.percentile(sample_arr, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_iqr = np.sum(
                (sample_arr < lower_bound) | (sample_arr > upper_bound)
            )

            mad_val = mad(sample_arr)
            if mad_val > 0:
                mod_z_score = 0.6745 * (sample_arr - np.median(sample_arr)) / mad_val
                outliers_mod_zscore = np.sum(np.abs(mod_z_score) > 3.5)

        # Compute advanced analytics metrics
        unique_est = self._uniques.estimate()
        unique_ratio = unique_est / max(1, self.count)

        # Compute robust statistics
        jb_chi2 = self._compute_jarque_bera(
            stats["skew"], stats["kurtosis"], self.count
        )

        # Determine if approximation was used for transparency
        approx = len(sample_values) < self.count

        # Calculate sample scale for histogram rendering
        # This is crucial for chunk mode to scale histogram counts to full dataset size
        sample_scale = self.count / len(sample_values) if sample_values else 1.0

        # Compute confidence intervals if enabled
        ci_lo, ci_hi = self._compute_confidence_interval(
            stats["mean"], stats["se"], self.count
        )

        # Compute granularity analysis
        gran_step, gran_decimals = self._compute_granularity(sample_values)

        # Compute heaping percentage
        heap_pct = self._compute_heaping_percentage(sample_values)

        return NumericSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            unique_est=unique_est,
            mean=stats["mean"],
            std=stats["std"],
            variance=stats["variance"],
            se=stats["se"],
            cv=stats["cv"],
            gmean=stats["gmean"],
            min=quantiles["min"],
            q1=quantiles["q1"],
            median=quantiles["median"],
            q3=quantiles["q3"],
            iqr=quantiles["iqr"],
            mad=mad_val if self.enable_outlier_detection else 0.0,
            skew=stats["skew"],
            kurtosis=stats["kurtosis"],
            jb_chi2=jb_chi2,
            max=quantiles["max"],
            zeros=self.zeros,
            negatives=self.negatives,
            outliers_iqr=outliers_iqr,
            outliers_mod_zscore=outliers_mod_zscore,
            approx=approx,
            inf=self.inf,
            int_like=self._int_like_all,
            unique_ratio_approx=unique_ratio,
            sample_vals=sample_values if sample_values else [],
            # True distribution histogram data
            true_histogram_edges=self._streaming_histogram.bin_edges,
            true_histogram_counts=self._streaming_histogram.counts,
            mem_bytes=self._bytes_seen,
            mono_inc=mono_inc,
            mono_dec=mono_dec,
            dtype_str=self._dtype_str,
            corr_top=self._corr_top,
            corr_threshold=self._corr_threshold,
            min_items=min_pairs,
            max_items=max_pairs,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            gran_step=gran_step,
            gran_decimals=gran_decimals,
            heap_pct=heap_pct,
            top_values=self._topk.items(),
            sample_scale=sample_scale,
            chunk_metadata=chunk_metadata,
        )

    def _compute_quantiles(self, values: List[float]) -> dict[str, float]:
        """Compute quantiles from sample values using optimized algorithms.

        Args:
            values: List of sample values

        Returns:
            Dictionary containing quantile statistics
        """
        if not values:
            return {
                "min": 0.0,
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0,
                "max": 0.0,
                "iqr": 0.0,
            }

        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            """Compute percentile using optimized linear interpolation."""
            if n == 1:
                return sorted_values[0]
            k = (n - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_values[int(k)]
            d0 = sorted_values[int(f)] * (c - k)
            d1 = sorted_values[int(c)] * (k - f)
            return d0 + d1

        min_val = sorted_values[0]
        max_val = sorted_values[-1]
        q1 = percentile(25)
        median = percentile(50)
        q3 = percentile(75)
        iqr = q3 - q1

        return {
            "min": min_val,
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": max_val,
            "iqr": iqr,
        }

    def _compute_jarque_bera(self, skew: float, kurtosis: float, n: int) -> float:
        """Compute Jarque-Bera test statistic for normality testing.

        Args:
            skew: Skewness value
            kurtosis: Kurtosis value
            n: Sample size

        Returns:
            Jarque-Bera chi-squared statistic
        """
        if n < 3:
            return 0.0

        jb = n * (skew**2 / 6 + kurtosis**2 / 24)
        return float(jb)

    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Get performance metrics for production monitoring.

        Returns:
            PerformanceMetrics object or None
        """
        return self._performance_metrics

    def merge(self, other: NumericAccumulator) -> None:
        """Merge another NumericAccumulator efficiently for distributed processing.

        Args:
            other: Another NumericAccumulator to merge
        """
        # Merge basic counts
        self.count += other.count
        self.missing += other.missing
        self.zeros += other.zeros
        self.negatives += other.negatives
        self.inf += other.inf

        # Merge algorithm components efficiently
        self._moments.merge(other._moments)

        # Merge samples (approximate for large datasets)
        other_sample = other._sample.values()
        for value in other_sample:
            self._sample.add(value)

        # Merge unique estimates
        for value in other_sample:
            self._uniques.add(value)

        # Merge extremes tracking
        self._extremes.merge(other._extremes)

        # Merge memory tracking
        self._bytes_seen += other._bytes_seen

        # Update integer-like status
        self._int_like_all = self._int_like_all and other._int_like_all

    def reset(self) -> None:
        """Reset accumulator to initial state efficiently."""
        self.count = 0
        self.missing = 0
        self.zeros = 0
        self.negatives = 0
        self.inf = 0
        self._int_like_all = True
        self._bytes_seen = 0
        self._corr_top = []

        # Reset all components efficiently
        self._moments.reset()
        self._sample = ReservoirSampler(self.config.sample_size)
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._extremes = ExtremeTracker(self.config.max_extremes)
        self._topk = MisraGries(self.config.top_k_size)

        if self._monotonicity:
            self._monotonicity.reset()
        if self._outlier_detector:
            self._outlier_detector.reset()
        if self._performance_metrics:
            self._performance_metrics.reset()

    def _compute_confidence_interval(
        self, mean: float, se: float, n: int
    ) -> Tuple[float, float]:
        """Compute 95% confidence interval for the mean.

        Args:
            mean: Sample mean
            se: Standard error
            n: Sample size

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n <= 1 or se <= 0:
            return float("nan"), float("nan")

        # Use t-distribution for small samples, normal for large samples
        if n < 30:
            # For small samples, use t-distribution (approximate with normal for simplicity)
            t_value = 1.96  # Approximate t-value for 95% CI
        else:
            t_value = 1.96  # Normal distribution for large samples

        margin_of_error = t_value * se
        return mean - margin_of_error, mean + margin_of_error

    def _compute_granularity(
        self, values: List[float]
    ) -> Tuple[Optional[float], Optional[int]]:
        """Compute granularity analysis of the data.

        Args:
            values: List of sample values

        Returns:
            Tuple of (granularity_step, decimal_places)
        """
        if not values or len(values) < 2:
            return None, None

        # Convert to numpy array for efficient processing
        arr = np.array(values)
        finite_values = arr[np.isfinite(arr)]

        if len(finite_values) < 2:
            return None, None

        # Compute differences between consecutive sorted values
        sorted_values = np.sort(finite_values)
        diffs = np.diff(sorted_values)

        # Filter out zero differences
        non_zero_diffs = diffs[diffs > 0]

        if len(non_zero_diffs) == 0:
            return None, None

        # Find the most common non-zero difference (granularity step)
        # Use histogram to find the most frequent difference
        hist, bin_edges = np.histogram(
            non_zero_diffs, bins=min(50, len(non_zero_diffs))
        )
        if len(hist) > 0:
            most_frequent_bin = np.argmax(hist)
            gran_step = (
                bin_edges[most_frequent_bin] + bin_edges[most_frequent_bin + 1]
            ) / 2
        else:
            gran_step = np.min(non_zero_diffs)

        # Calculate decimal places
        if gran_step > 0:
            # Count decimal places by finding the smallest power of 10 that makes the number an integer
            decimal_places = 0
            temp = gran_step
            while abs(temp - round(temp)) > 1e-10 and decimal_places < 10:
                temp *= 10
                decimal_places += 1
        else:
            decimal_places = None

        return float(gran_step), decimal_places

    def _compute_heaping_percentage(self, values: List[float]) -> float:
        """Compute heaping percentage (percentage of values ending in 0 or 5).

        Args:
            values: List of sample values

        Returns:
            Heaping percentage (0-100)
        """
        if not values:
            return float("nan")

        # Convert to numpy array for efficient processing
        arr = np.array(values)
        finite_values = arr[np.isfinite(arr)]

        if len(finite_values) == 0:
            return float("nan")

        # Count values ending in 0 or 5 (heaping effect)
        heaped_count = 0
        for val in finite_values:
            # Convert to string and check last digit
            val_str = f"{val:.10f}".rstrip("0").rstrip(".")
            if val_str and val_str[-1] in ["0", "5"]:
                heaped_count += 1

        return (heaped_count / len(finite_values)) * 100.0
