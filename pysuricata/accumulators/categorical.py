"""High-performance categorical accumulator optimized for big data.

This module provides a production-ready, scalable implementation of the categorical accumulator
with comprehensive error handling, validation, and performance optimizations for large datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .config import CategoricalConfig
from .sketches import KMV, MisraGries, ReservoirSampler


@dataclass
class CategoricalSummary:
    """Summary statistics for categorical data.

    This dataclass contains all computed statistics for a categorical column,
    including frequency counts, string statistics, and quality indicators.
    """

    name: str
    count: int
    missing: int
    unique_est: int
    top_items: List[Tuple[str, int]]
    approx: bool
    # extras for alignment
    mem_bytes: int = 0
    avg_len: Optional[float] = None
    len_p90: Optional[int] = None
    empty_zero: int = 0
    case_variants_est: int = 0
    trim_variants_est: int = 0
    dtype_str: str = "categorical"
    # v2 additions
    entropy: float = 0.0
    gini_impurity: float = 0.0
    most_common_ratio: float = 0.0
    diversity_ratio: float = 0.0
    chunk_metadata: Optional[List[Tuple[int, int, int]]] = None


class CategoricalAccumulator:
    """Production-grade categorical accumulator optimized for big data.

    This accumulator provides comprehensive analysis of categorical data with
    superior error handling, validation, and performance optimizations for
    processing massive datasets efficiently.
    """

    def __init__(self, name: str, config: Optional[CategoricalConfig] = None):
        """Initialize categorical accumulator.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
        """
        self.name = name
        self.config = config or CategoricalConfig()

        # Core state
        self.count = 0
        self.missing = 0
        self._dtype_str = "categorical"
        self._bytes_seen = 0

        # Initialize data structures with optimized sizes for big data
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._uniques_lower = (
            KMV(self.config.uniques_sketch_size)
            if self.config.enable_case_variants
            else None
        )
        self._uniques_strip = (
            KMV(self.config.uniques_sketch_size)
            if self.config.enable_trim_variants
            else None
        )
        self._topk = MisraGries(self.config.top_k_size)

        # String length tracking with memory-efficient sampling
        self._len_sum = 0
        self._len_n = 0
        self._len_sample = (
            ReservoirSampler(self.config.length_sample_size)
            if self.config.enable_length_stats
            else None
        )

        # Special value tracking
        self._empty_zero = 0

        # Per-column chunk tracking for accurate missing value reporting
        self._chunk_boundaries: List[int] = []  # Cumulative row counts
        self._chunk_missing: List[int] = []  # Missing count per chunk
        self._current_chunk_missing = 0  # Missing in current chunk
        self._current_chunk_rows = 0  # Total rows in current chunk

    def set_dtype(self, dtype_str: str) -> None:
        """Set the data type string.

        Args:
            dtype_str: String representation of the data type
        """
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "categorical"

    @property
    def unique_est(self) -> int:
        """Get unique count estimate for compatibility."""
        return self._uniques.estimate()

    @property
    def avg_len(self) -> Optional[float]:
        """Get average string length for compatibility."""
        if self._len_n > 0:
            return self._len_sum / self._len_n
        return None

    def update(self, arr: Sequence[Any]) -> None:
        """Update accumulator with new values using optimized batch processing.

        Args:
            arr: Sequence of values to process
        """
        if not arr:
            return

        # Track chunk metadata before processing
        missing_before = self.missing

        # Process each value with comprehensive error handling
        for value in arr:
            try:
                self._process_single_value(value)
            except Exception:
                # Robust error handling - continue processing even with bad data
                self.missing += 1
                continue

        # Track missing values in current chunk
        missing_in_update = self.missing - missing_before
        self._current_chunk_missing += missing_in_update
        self._current_chunk_rows += len(arr)

    def _process_single_value(self, value: Any) -> None:
        """Process a single categorical value with optimized handling.

        Args:
            value: Value to process
        """
        # Handle missing values efficiently
        if self._is_missing(value):
            self.missing += 1
            return

        # Convert to string representation with error handling
        str_value = self._convert_to_string(value)

        # Update counts
        self.count += 1

        # Update data structures with optimized operations
        self._uniques.add(str_value)
        self._topk.add(str_value)

        # Update variant tracking if enabled
        if self.config.enable_case_variants and self._uniques_lower:
            self._uniques_lower.add(str_value.lower())

        if self.config.enable_trim_variants and self._uniques_strip:
            self._uniques_strip.add(str_value.strip())

        # Update string length statistics if enabled
        if self.config.enable_length_stats and self._len_sample:
            str_len = len(str_value)
            self._len_sum += str_len
            self._len_n += 1
            self._len_sample.add(float(str_len))

        # Track special values for data quality analysis
        if str_value == "" or str_value == "0":
            self._empty_zero += 1

    def _is_missing(self, value: Any) -> bool:
        """Check if a value is considered missing.

        Args:
            value: Value to check

        Returns:
            True if value is missing, False otherwise
        """
        if value is None:
            return True

        if isinstance(value, float) and np.isnan(value):
            return True

        return False

    def _convert_to_string(self, value: Any) -> str:
        """Convert value to string representation with robust error handling.

        Args:
            value: Value to convert

        Returns:
            String representation of the value
        """
        try:
            if isinstance(value, str):
                return value

            # Handle numeric values
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    return ""
                return str(value)

            # Handle other types
            return str(value)

        except Exception:
            # Fallback for problematic values
            return ""

    def add_mem(self, n: int) -> None:
        """Add to memory usage tracking.

        Args:
            n: Number of bytes to add
        """
        if not self.config.enable_memory_tracking:
            return

        try:
            self._bytes_seen += int(n)
        except (ValueError, TypeError):
            pass

    def mark_chunk_boundary(self) -> None:
        """Mark the end of a chunk for per-column missing value tracking.

        This method records the cumulative row count and missing count for the
        current chunk, enabling accurate per-column chunk visualization.
        """
        if self._current_chunk_rows > 0:
            cumulative_rows = self.count + self.missing
            self._chunk_boundaries.append(cumulative_rows)
            self._chunk_missing.append(self._current_chunk_missing)
            self._current_chunk_missing = 0
            self._current_chunk_rows = 0

    def finalize(self) -> CategoricalSummary:
        """Finalize accumulator and return comprehensive summary statistics.

        Returns:
            CategoricalSummary containing all computed statistics
        """
        # Get top items with optimized access
        top_items = self._topk.items()

        # Calculate string length statistics
        avg_len = self._len_sum / max(1, self._len_n) if self._len_n > 0 else None
        len_p90 = self._calculate_percentile(
            self._len_sample.values() if self._len_sample else [], 90
        )

        # Calculate variant estimates
        case_variants_est = self._uniques_lower.estimate() if self._uniques_lower else 0
        trim_variants_est = self._uniques_strip.estimate() if self._uniques_strip else 0

        # Calculate diversity metrics for data quality analysis
        entropy = self._calculate_entropy(top_items)
        gini_impurity = self._calculate_gini_impurity(top_items)
        most_common_ratio = self._calculate_most_common_ratio(top_items)
        diversity_ratio = self._calculate_diversity_ratio()

        # Determine if approximation was used
        approx = len(top_items) < self.config.top_k_size

        # Build per-column chunk metadata from tracked boundaries
        # Finalize any pending chunk data first
        if self._current_chunk_rows > 0:
            self.mark_chunk_boundary()

        # Build per-column chunk metadata list
        per_column_chunk_metadata = []
        start_row = 0
        for i, end_cumulative in enumerate(self._chunk_boundaries):
            end_row = end_cumulative - 1
            missing_in_chunk = self._chunk_missing[i]
            per_column_chunk_metadata.append((start_row, end_row, missing_in_chunk))
            start_row = end_cumulative

        return CategoricalSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            unique_est=self._uniques.estimate(),
            top_items=top_items,
            approx=approx,
            mem_bytes=self._bytes_seen,
            avg_len=avg_len,
            len_p90=len_p90,
            empty_zero=self._empty_zero,
            case_variants_est=case_variants_est,
            trim_variants_est=trim_variants_est,
            dtype_str=self._dtype_str,
            entropy=entropy,
            gini_impurity=gini_impurity,
            most_common_ratio=most_common_ratio,
            diversity_ratio=diversity_ratio,
            chunk_metadata=per_column_chunk_metadata
            if per_column_chunk_metadata
            else None,
        )

    def _calculate_percentile(
        self, values: List[float], percentile: float
    ) -> Optional[int]:
        """Calculate percentile of values efficiently.

        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value or None if insufficient data
        """
        if not values:
            return None

        sorted_values = sorted(values)
        n = len(sorted_values)
        k = (n - 1) * percentile / 100
        f = int(k)
        c = int(k) + 1

        if f == c or c >= n:
            return int(sorted_values[f])

        # Linear interpolation
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return int(d0 + d1)

    def _calculate_entropy(self, top_items: List[Tuple[str, int]]) -> float:
        """Calculate Shannon entropy of the distribution.

        Args:
            top_items: List of (value, count) tuples

        Returns:
            Entropy value in bits
        """
        if not top_items:
            return 0.0

        total_count = sum(count for _, count in top_items)
        if total_count == 0:
            return 0.0

        entropy = 0.0
        for _, count in top_items:
            if count > 0:
                p = count / total_count
                entropy -= p * np.log2(p)

        return entropy

    def _calculate_gini_impurity(self, top_items: List[Tuple[str, int]]) -> float:
        """Calculate Gini impurity of the distribution.

        Args:
            top_items: List of (value, count) tuples

        Returns:
            Gini impurity value
        """
        if not top_items:
            return 0.0

        total_count = sum(count for _, count in top_items)
        if total_count == 0:
            return 0.0

        gini = 1.0
        for _, count in top_items:
            p = count / total_count
            gini -= p * p

        return gini

    def _calculate_most_common_ratio(self, top_items: List[Tuple[str, int]]) -> float:
        """Calculate ratio of the most common value.

        Args:
            top_items: List of (value, count) tuples

        Returns:
            Ratio of most common value
        """
        if not top_items:
            return 0.0

        total_count = sum(count for _, count in top_items)
        if total_count == 0:
            return 0.0

        max_count = max(count for _, count in top_items)
        return max_count / total_count

    def _calculate_diversity_ratio(self) -> float:
        """Calculate diversity ratio (unique values / total values).

        Returns:
            Diversity ratio
        """
        if self.count == 0:
            return 0.0

        unique_est = self._uniques.estimate()
        return unique_est / self.count

    def get_quality_metrics(self) -> dict[str, Any]:
        """Get comprehensive data quality metrics.

        Returns:
            Dictionary containing quality metrics
        """
        top_items = self._topk.items()

        return {
            "total_values": self.count + self.missing,
            "valid_values": self.count,
            "missing_values": self.missing,
            "missing_ratio": self.missing / max(1, self.count + self.missing),
            "unique_estimate": self._uniques.estimate(),
            "diversity_ratio": self._calculate_diversity_ratio(),
            "entropy": self._calculate_entropy(top_items),
            "gini_impurity": self._calculate_gini_impurity(top_items),
            "most_common_ratio": self._calculate_most_common_ratio(top_items),
            "case_variants_estimate": self._uniques_lower.estimate()
            if self._uniques_lower
            else 0,
            "trim_variants_estimate": self._uniques_strip.estimate()
            if self._uniques_strip
            else 0,
            "empty_zero_count": self._empty_zero,
            "avg_string_length": self._len_sum / max(1, self._len_n)
            if self._len_n > 0
            else 0,
        }

    def merge(self, other: CategoricalAccumulator) -> None:
        """Merge another CategoricalAccumulator efficiently.

        Args:
            other: Another CategoricalAccumulator to merge
        """
        self.count += other.count
        self.missing += other.missing
        self._bytes_seen += other._bytes_seen
        self._len_sum += other._len_sum
        self._len_n += other._len_n
        self._empty_zero += other._empty_zero

        # Merge data structures (approximate for efficiency)
        for item, count in other._topk.items():
            for _ in range(count):
                self._topk.add(item)

        # Note: KMV sketches cannot be easily merged, so we approximate
        # by adding a sample of values from the other accumulator
        other_sample = [item for item, _ in other._topk.items()[:100]]
        for item in other_sample:
            self._uniques.add(item)

    def reset(self) -> None:
        """Reset accumulator to initial state efficiently."""
        self.count = 0
        self.missing = 0
        self._bytes_seen = 0
        self._len_sum = 0
        self._len_n = 0
        self._empty_zero = 0

        # Reset data structures
        self._uniques = KMV(self.config.uniques_sketch_size)
        if self._uniques_lower:
            self._uniques_lower = KMV(self.config.uniques_sketch_size)
        if self._uniques_strip:
            self._uniques_strip = KMV(self.config.uniques_sketch_size)
        self._topk = MisraGries(self.config.top_k_size)
        if self._len_sample:
            self._len_sample = ReservoirSampler(self.config.length_sample_size)
