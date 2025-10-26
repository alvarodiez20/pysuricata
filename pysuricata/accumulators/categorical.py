"""High-performance categorical accumulator optimized for big data.

This module provides a production-ready, scalable implementation of the categorical accumulator
with comprehensive error handling, validation, and performance optimizations for large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

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

        # Convert to pandas Series for vectorized operations
        try:
            import pandas as pd
            s = pd.Series(arr) if not isinstance(arr, pd.Series) else arr
        except ImportError:
            # Fallback to original implementation if pandas not available
            for value in arr:
                try:
                    self._process_single_value(value)
                except Exception:
                    self.missing += 1
                    continue
            return

        # Vectorized missing value detection
        missing_mask = s.isna()
        self.missing += missing_mask.sum()

        # Get valid (non-missing) values
        valid_values = s[~missing_mask]
        
        if len(valid_values) == 0:
            return

        # Convert to string representation (vectorized)
        try:
            str_values = valid_values.astype(str)
        except Exception:
            # Fallback to individual conversion
            for value in valid_values:
                try:
                    str_value = self._convert_to_string(value)
                    self._update_sketches(str_value)
                    self._update_length_stats(str_value)
                    self._update_special_values(str_value)
                except Exception:
                    continue
            self.count += len(valid_values)
            return

        # Update count
        self.count += len(valid_values)

        # Vectorized sketch updates using value_counts for efficiency
        try:
            value_counts = str_values.value_counts()
            
            # Batch update sketches
            for value, count in value_counts.items():
                # Update KMV sketch
                for _ in range(count):
                    self._uniques.add(value)
                
                # Update MisraGries sketch
                for _ in range(count):
                    self._topk.add(value)
                
                # Update variant tracking if enabled
                if self.config.enable_case_variants and self._uniques_lower:
                    for _ in range(count):
                        self._uniques_lower.add(value.lower())
                
                if self.config.enable_trim_variants and self._uniques_strip:
                    for _ in range(count):
                        self._uniques_strip.add(value.strip())
                
                # Update special value tracking
                if value == "" or value == "0":
                    self._empty_zero += count

        except Exception:
            # Fallback to individual processing if vectorization fails
            for value in str_values:
                try:
                    self._update_sketches(value)
                    self._update_length_stats(value)
                    self._update_special_values(value)
                except Exception:
                    continue

        # Vectorized string length statistics
        if self.config.enable_length_stats and self._len_sample:
            try:
                lengths = str_values.str.len()
                self._len_sum += lengths.sum()
                self._len_n += len(lengths)
                
                # Add lengths to reservoir sampler
                for length in lengths:
                    self._len_sample.add(float(length))
            except Exception:
                # Fallback to individual processing
                for value in str_values:
                    try:
                        str_len = len(value)
                        self._len_sum += str_len
                        self._len_n += 1
                        self._len_sample.add(float(str_len))
                    except Exception:
                        continue

    def _update_sketches(self, value: str) -> None:
        """Update sketching algorithms with a single value.
        
        Args:
            value: String value to add to sketches
        """
        self._uniques.add(value)
        self._topk.add(value)
        
        if self.config.enable_case_variants and self._uniques_lower:
            self._uniques_lower.add(value.lower())
        
        if self.config.enable_trim_variants and self._uniques_strip:
            self._uniques_strip.add(value.strip())

    def _update_length_stats(self, value: str) -> None:
        """Update string length statistics.
        
        Args:
            value: String value to process
        """
        if self.config.enable_length_stats and self._len_sample:
            str_len = len(value)
            self._len_sum += str_len
            self._len_n += 1
            self._len_sample.add(float(str_len))

    def _update_special_values(self, value: str) -> None:
        """Update special value tracking.
        
        Args:
            value: String value to check
        """
        if value == "" or value == "0":
            self._empty_zero += 1

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
