"""High-performance numeric accumulator optimized for big data analytics.

This module provides a production-ready, scalable implementation of the numeric accumulator
using advanced algorithmic composition, vectorized operations, and performance optimizations
designed for processing massive numerical datasets efficiently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .algorithms import (
    ExtremeTracker,
    MonotonicityDetector,
    PerformanceMetrics,
    StreamingMoments,
)
from .config import NumericConfig
from .sketches import KMV, MisraGries, ReservoirSampler, StreamingHistogram, mad

# Minimum share of a column's distinct values the top-k counters must be able to
# cover before a "Common values" table is worth building. Below it the table
# degenerates into a ranked list of values that occurred once -- sampling noise
# presented as a finding -- and Misra-Gries costs a third of this accumulator to
# produce it. Skewed columns beat this flat-distribution bound, which is the
# safe direction: they stay enabled.
_TOP_K_MIN_COVERAGE = 0.02


def should_track_top_k(unique_est: float, count: int, top_k: int) -> bool:
    """Whether a numeric column's top-k answer will carry information.

    A top-k table is worth building when the tracked values could plausibly
    cover a meaningful share of the column. When a column holds far more
    distinct values than the sketch has counters, the table lists singletons.

    Args:
        unique_est: Current distinct-count estimate for the column.
        count: Number of finite values seen so far.
        top_k: Number of Misra-Gries counters.

    Returns:
        True while top-k should keep being fed.
    """
    if count <= 0:
        return True
    if unique_est <= top_k:
        return True  # exact and complete: always worth it
    return (top_k / max(unique_est, 1.0)) >= _TOP_K_MIN_COVERAGE


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
    hist_counts: list[int] | None = None
    top_values: list[tuple[float, int]] = field(default_factory=list)
    # Reservoir sample for advanced analytics
    sample_vals: list[float] | None = None
    # True distribution histogram data
    true_histogram_edges: list[float] | None = None
    true_histogram_counts: list[int] | None = None
    # Quality metrics
    heap_pct: float = float("nan")
    gran_decimals: int | None = None
    gran_step: float | None = None
    bimodal: bool = False
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    # System metrics
    mem_bytes: int = 0
    mono_inc: bool = False
    mono_dec: bool = False
    dtype_str: str = "numeric"
    corr_top: list[tuple[str, float]] = field(default_factory=list)
    sample_scale: float = 1.0
    # Extremes with global indices
    min_items: list[tuple[Any, float]] = field(default_factory=list)
    max_items: list[tuple[Any, float]] = field(default_factory=list)
    # Chunk metadata for spectrum visualization
    chunk_metadata: list[tuple[int, int, int]] | None = (
        None  # (start_row, end_row, missing_count)
    )
    corr_threshold: float = 0.5  # Threshold used for correlation filtering


class NumericAccumulator:
    """Production-grade numeric accumulator optimized for big data analytics.

    This accumulator leverages advanced algorithmic composition and vectorized operations
    to achieve maximum performance on large-scale numerical datasets while maintaining
    precision, reliability, and comprehensive statistical analysis capabilities.
    """

    def __init__(
        self,
        name: str,
        config: NumericConfig | None = None,
        *,
        seed: int | None = None,
    ):
        """Initialize numeric accumulator with optimized components.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
            seed: Seed for this column's sampling. None draws from OS entropy.
                Sampling never touches the process-global RNG either way.
        """
        self.name = name
        self.config = config or NumericConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Core state tracking
        self.count = 0
        self.missing = 0
        self.zeros = 0
        self.negatives = 0
        self.inf = 0
        self._int_like_all = True
        self._dtype_str = "numeric"
        self._corr_top: list[tuple[str, float]] = []
        self._corr_threshold: float = 0.5

        # Memory tracking for big data optimization
        self._bytes_seen = 0

        # High-performance algorithm components
        self._moments = StreamingMoments(
            enable_performance_tracking=self.config.enable_memory_tracking
        )
        self._sample = ReservoirSampler(self.config.sample_size, rng=self._rng)
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._extremes = ExtremeTracker(self.config.max_extremes)
        self._topk = MisraGries(self.config.top_k_size)
        # Cleared once the column proves too high-cardinality for a top-k table
        # to say anything; never re-enabled.
        self._track_top_k = True

        # Streaming histogram for true distribution
        self._streaming_histogram = StreamingHistogram(bins=25)

        # Optional advanced analytics components
        self._monotonicity = (
            MonotonicityDetector()
            if self.config.enable_monotonicity_detection
            else None
        )
        # No OutlierDetector here. It kept a second 10,000-slot reservoir over
        # the same values as self._sample, fed on every chunk -- and nothing
        # ever read it: detect_outliers() was never called, and the outlier
        # counts in the report are computed in finalize() from self._sample.
        # The flag below still gates that computation.
        self.enable_outlier_detection = self.config.enable_outlier_detection

        # Performance monitoring for production environments
        self._performance_metrics = (
            PerformanceMetrics() if self.config.enable_memory_tracking else None
        )

        # Per-column chunk tracking for accurate missing value reporting
        if self.config.enable_chunk_metadata:
            # Pre-allocate arrays for bounded memory usage
            self._chunk_boundaries: list[int] = []
            self._chunk_missing: list[int] = []
            self._chunk_metadata_enabled = True
            self._chunk_count = 0
        else:
            # Disable chunk metadata tracking to save memory
            self._chunk_boundaries = None
            self._chunk_missing = None
            self._chunk_metadata_enabled = False
            self._chunk_count = 0

        self._current_chunk_missing = 0  # Missing in current chunk
        self._current_chunk_rows = 0  # Total rows in current chunk

    def set_dtype(self, dtype_str: str) -> None:
        """Set the data type string efficiently.

        Args:
            dtype_str: String representation of the data type
        """
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "numeric"

    def set_corr_top(self, items: list[tuple[str, float]]) -> None:
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

    def update(self, arr: Sequence[Any], *, row_offset: int = 0) -> None:
        """Update accumulator with new values using optimized vectorized processing.

        Args:
            arr: Sequence of values to process
            row_offset: Global index of this chunk's first row, so extreme-value
                indices refer to the dataset rather than to the chunk.
        """
        # Handle empty arrays efficiently
        if len(arr) == 0:
            return

        # Track chunk metadata before processing
        missing_before = self.missing

        # Performance tracking for production monitoring
        if self._performance_metrics:
            import time

            start_time = time.perf_counter()

        # Convert to numpy array for maximum performance
        try:
            values = np.asarray(arr, dtype=float)
            # Count missing/inf values using vectorized numpy operations
            nan_count = int(np.sum(np.isnan(values)))
            inf_count = int(np.sum(np.isinf(values)))
            self.missing += nan_count
            self.inf += inf_count
        except (ValueError, TypeError):
            # Robust error handling for mixed-type data
            values = self._convert_to_numeric(arr)

        # Process values with optimized algorithms
        self._process_values(values, row_offset)

        # Track missing values in current chunk
        missing_in_update = self.missing - missing_before
        self._current_chunk_missing += missing_in_update
        self._current_chunk_rows += len(arr)

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

    def _process_values(self, values: np.ndarray, row_offset: int = 0) -> None:
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

        # Check if all values are integer-like for type inference (vectorized)
        if self._int_like_all:
            self._int_like_all = bool(
                np.all(np.abs(finite_values - np.round(finite_values)) < 1e-10)
            )

        # Update algorithm components with vectorized operations
        self._moments.update(finite_values)
        self._sample.add_many(finite_values)

        # Update streaming histogram for true distribution
        self._streaming_histogram.add_many(finite_values)

        # Batch update unique estimates and top values using vectorized operations
        self._uniques.add_many(finite_values)

        # Top-k is only fed while its answer would mean something. The gate
        # latches off and discards what it had: keeping the partial counts
        # gathered before the cutoff would make the table depend on the chunk
        # size. Because the distinct estimate only rises, "did the gate fire"
        # is decided by the column's final cardinality, not by how it was
        # chunked -- so a given column yields the same table either way.
        if self._track_top_k:
            if should_track_top_k(
                float(self._uniques.estimate()), self.count, self.config.top_k_size
            ):
                self._topk.add_many(finite_values)
            else:
                self._track_top_k = False
                self._topk = MisraGries(self.config.top_k_size)

        # Extreme indices are global, not chunk-local. Without the offset,
        # "row 4,182 had the maximum" named a position within whichever chunk
        # the value happened to arrive in, so it was wrong for every chunk after
        # the first.
        indices = np.arange(len(values))[finite_mask] + row_offset
        self._extremes.update(finite_values, indices)

        # Update optional advanced analytics components
        if self._monotonicity:
            self._monotonicity.update(finite_values, all_finite=True)

    def update_extremes(
        self, pairs_min: list[tuple[Any, float]], pairs_max: list[tuple[Any, float]]
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

    def mark_chunk_boundary(self) -> None:
        """Mark the end of a chunk for per-column missing value tracking.

        This method records the cumulative row count and missing count for the
        current chunk, enabling accurate per-column chunk visualization.
        When chunk metadata is disabled, this method does nothing to save memory.
        """
        if not self._chunk_metadata_enabled or self._current_chunk_rows == 0:
            self._current_chunk_missing = 0
            self._current_chunk_rows = 0
            return

        # Check if we've exceeded the maximum number of chunks to track
        if self._chunk_count >= self.config.max_chunks:
            # Switch to summary mode - stop tracking individual chunks
            self._chunk_metadata_enabled = False
            self._current_chunk_missing = 0
            self._current_chunk_rows = 0
            return

        # Record chunk metadata
        cumulative_rows = self.count + self.missing
        self._chunk_boundaries.append(cumulative_rows)
        self._chunk_missing.append(self._current_chunk_missing)
        self._chunk_count += 1

        # Reset for next chunk
        self._current_chunk_missing = 0
        self._current_chunk_rows = 0

    def finalize(
        self, chunk_metadata: list[tuple[int, int, int]] | None = None
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

        # Take the reported minimum and maximum from the tracker, which sees
        # every value, rather than from the reservoir, which sees a sample of
        # them. The card printed a sampled "Maximum" directly above a table of
        # exact extreme values, so the two could disagree -- and whether they
        # did came down to whether the true extreme happened to be sampled.
        if min_pairs:
            quantiles["min"] = min_pairs[0][1]
        if max_pairs:
            quantiles["max"] = max_pairs[0][1]

        # Get advanced monotonicity analysis if enabled
        mono_inc, mono_dec = False, False
        if self._monotonicity:
            mono_inc, mono_dec = self._monotonicity.get_monotonicity()

        # Get outlier detection results if enabled
        outliers_iqr, outliers_mod_zscore = 0, 0
        # Bound up front: a column of nothing but inf leaves sample_values empty,
        # and the branch below never runs.
        mad_val = 0.0
        if self.enable_outlier_detection and sample_values:
            # `outlier_methods` used to be read by nobody: the detector that
            # consulted it was never called, and this block always computed
            # both. A configuration option that silently does nothing is worse
            # than one that does not exist, so it is honoured here.
            methods = set(self.config.outlier_methods)
            sample_arr = np.array(sample_values)

            if "iqr" in methods:
                q1, q3 = np.percentile(sample_arr, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers_iqr = np.sum(
                    (sample_arr < lower_bound) | (sample_arr > upper_bound)
                )

            if "mad" in methods:
                mad_val = mad(sample_arr)
                if mad_val > 0:
                    mod_z_score = (
                        0.6745 * (sample_arr - np.median(sample_arr)) / mad_val
                    )
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

        # Top values come from the Misra-Gries counters, or not at all.
        #
        # There used to be a fallback here: when the sketch returned fewer than
        # five entries, common values were recomputed from the reservoir sample
        # and the counts multiplied by the sampling ratio "to represent the full
        # dataset". On a continuous column that reports every sampled value as
        # having occurred sample_scale times when it occurred once -- a
        # fabricated count, presented in the report exactly like a measured one.
        # It also overrode the *exact* counters on any column with fewer than
        # five distinct values, replacing a correct answer with an estimate.
        # An absent table is the honest output when nothing is common.
        top_values = self._topk.items()

        # Build per-column chunk metadata from tracked boundaries
        # Finalize any pending chunk data first
        if self._current_chunk_rows > 0:
            self.mark_chunk_boundary()

        # Build per-column chunk metadata list (only if enabled)
        per_column_chunk_metadata = []
        if self._chunk_metadata_enabled and self._chunk_boundaries is not None:
            start_row = 0
            for i, end_cumulative in enumerate(self._chunk_boundaries):
                end_row = end_cumulative - 1
                missing_in_chunk = self._chunk_missing[i]
                per_column_chunk_metadata.append((start_row, end_row, missing_in_chunk))
                start_row = end_cumulative

        # Use per-column metadata if available, otherwise use provided global metadata
        final_chunk_metadata = (
            per_column_chunk_metadata if per_column_chunk_metadata else chunk_metadata
        )

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
            top_values=top_values,
            sample_scale=sample_scale,
            chunk_metadata=final_chunk_metadata,
        )

    def _compute_quantiles(self, values: list[float]) -> dict[str, float]:
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

    def get_performance_metrics(self) -> PerformanceMetrics | None:
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

        # Every one of these composes on its own terms; the merge delegates
        # rather than replaying values. What it used to do instead was push the
        # other side's *reservoir buffer* through `add()` -- which treats a
        # 20,000-value sample as a 20,000-value stream, so a large shard merged
        # into a small one lost its weight entirely -- and derive the distinct
        # estimate from that same sample, which cannot represent a distinct
        # count larger than the sample.
        self._sample.merge(other._sample)
        self._uniques.merge(other._uniques)
        self._streaming_histogram.merge(other._streaming_histogram)
        self._extremes.merge(other._extremes)

        # Merge memory tracking
        self._bytes_seen += other._bytes_seen

        # Update integer-like status
        self._int_like_all = self._int_like_all and other._int_like_all

        # If either side gave up on top-k, the merged column has too many
        # distinct values for the table to mean anything either.
        if not other._track_top_k and self._track_top_k:
            self._track_top_k = False
            self._topk = MisraGries(self.config.top_k_size)
        elif self._track_top_k:
            self._topk.merge(other._topk)

        # Monotonicity is deliberately not merged. It is a statement about the
        # order values arrived in, and merging two shards says nothing about
        # how they would interleave -- two ascending halves are ascending only
        # if one begins where the other ends, which the accumulators cannot
        # know. A merged column reports whatever this side observed.

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

        # A reset accumulator must replay identically, so rewind the generator
        # rather than continuing its stream.
        self._rng = np.random.default_rng(self._seed)

        # Reset all components efficiently
        self._moments.reset()
        self._sample = ReservoirSampler(self.config.sample_size, rng=self._rng)
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._extremes = ExtremeTracker(self.config.max_extremes)
        self._topk = MisraGries(self.config.top_k_size)
        self._track_top_k = True

        # Chunk metadata is per-run state: leaving it in place would append a
        # second run's chunks to the first run's list.
        if self._chunk_metadata_enabled:
            self._chunk_boundaries = []
            self._chunk_missing = []
        self._chunk_count = 0
        self._current_chunk_missing = 0
        self._current_chunk_rows = 0

        if self._monotonicity:
            self._monotonicity.reset()
        if self._performance_metrics:
            self._performance_metrics.reset()

    def _compute_confidence_interval(
        self, mean: float, se: float, n: int
    ) -> tuple[float, float]:
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
        self, values: list[float]
    ) -> tuple[float | None, int | None]:
        """Compute granularity analysis of the data.

        Args:
            values: List of sample values

        Returns:
            Tuple of (granularity_step, decimal_places)
        """
        if not values or len(values) < 2:
            return None, None

        # Convert to numpy array for efficient processing
        try:
            arr = np.array(values, dtype=float)
        except Exception:
            # Fallback for mixed types
            arr = np.array([float(x) for x in values if x is not None], dtype=float)

        finite_values = arr[np.isfinite(arr)]

        if len(finite_values) < 2:
            return None, None

        # Compute differences between consecutive sorted values
        sorted_values = np.sort(finite_values)
        diffs = np.diff(sorted_values)

        # Filter out zero differences
        non_zero_diffs = np.asarray(diffs[diffs > 0], dtype=float)

        if len(non_zero_diffs) == 0:
            return None, None

        # Find the most common non-zero difference (granularity step) by
        # histogramming the differences.
        # Specify range explicitly and use python's min/max to avoid NumPy 2.1 _NoValue pointer bug when reloaded by pytest-cov
        range_min, range_max = float(min(non_zero_diffs)), float(max(non_zero_diffs))
        n_bins = int(min(50, len(non_zero_diffs)))

        # A strictly positive spread is not sufficient. The bins must also be
        # wide enough to be representable: for values around 1e-15 whose spread
        # is ~1e-31, every computed edge rounds to the same float64 and numpy
        # (>= 2.5) rejects the call outright rather than silently returning
        # degenerate bins. When the differences are all equal to within
        # floating-point resolution there is nothing to histogram anyway -- the
        # granularity simply *is* that difference.
        spread = range_max - range_min
        min_representable = float(np.spacing(max(abs(range_min), abs(range_max))))
        if n_bins < 1 or spread <= min_representable * n_bins:
            gran_step = float(np.median(non_zero_diffs))
        else:
            hist, bin_edges = np.histogram(
                non_zero_diffs, bins=n_bins, range=(range_min, range_max)
            )
            if len(hist) > 0:
                most_frequent_bin = int(np.argmax(hist))
                gran_step = (
                    float(
                        bin_edges[most_frequent_bin] + bin_edges[most_frequent_bin + 1]
                    )
                    / 2.0
                )
            else:
                gran_step = float(np.min(non_zero_diffs))

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

    def _compute_heaping_percentage(self, values: list[float]) -> float:
        """Compute heaping percentage (percentage of values ending in 0 or 5).

        Args:
            values: List of sample values

        Returns:
            Heaping percentage (0-100)
        """
        if not values:
            return float("nan")

        # Convert to numpy array for efficient processing
        try:
            arr = np.array(values, dtype=float)
        except Exception:
            # Fallback for mixed types
            arr = np.array([float(x) for x in values if x is not None], dtype=float)

        finite_values = arr[np.isfinite(arr)]

        if len(finite_values) == 0:
            return float("nan")

        # Vectorized heaping detection: check if last significant digit is 0 or 5
        # Strategy: scale each value by powers of 10 until it's an integer,
        # then check last_digit = abs(scaled_int) % 10 in {0, 5}
        try:
            abs_vals = np.abs(finite_values)
            # Round to 10 decimal places to avoid floating-point noise
            rounded = np.round(abs_vals, 10)
            # Values >= 1e8 would overflow int64 when scaled by 1e10; use fallback.
            if rounded.size > 0 and rounded.max() >= 1e8:
                raise OverflowError("values too large for int64 scaled cast")
            # Scale up by 10^10 to convert to integers, then find trailing zeros
            scaled = np.round(rounded * 1e10).astype(np.int64)
            # Remove trailing zeros to find the last significant digit
            # Divide by 10 while divisible — but vectorized is hard for variable scales
            # Instead: for each value, the last significant digit is found by
            # removing factors of 10, then taking mod 10
            # Efficient approach: check scaled % 10, if 0 then divide by 10, repeat
            # But we can't loop per-value. Use a fixed maximum of 10 iterations.
            working = scaled.copy()
            for _ in range(10):
                divisible = (working % 10 == 0) & (working != 0)
                if not divisible.any():
                    break
                working[divisible] = working[divisible] // 10
            last_digits = np.abs(working) % 10
            heaped_mask = (last_digits == 0) | (last_digits == 5)
            heaped_count = int(np.sum(heaped_mask))
        except Exception:
            # Fallback to string-based check for edge cases
            heaped_count = 0
            for val in finite_values:
                val_str = f"{val:.10f}".rstrip("0").rstrip(".")
                if val_str and val_str[-1] in ["0", "5"]:
                    heaped_count += 1

        return (heaped_count / len(finite_values)) * 100.0
