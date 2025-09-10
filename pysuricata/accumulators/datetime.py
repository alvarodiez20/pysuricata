"""High-performance datetime accumulator optimized for big data temporal analysis.

This module provides a production-ready, scalable implementation of the datetime accumulator
with vectorized operations, comprehensive temporal pattern analysis, and advanced performance
optimizations for processing massive time-series datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Sequence

import numpy as np

from .algorithms import MonotonicityDetector
from .config import DatetimeConfig
from .sketches import KMV, ReservoirSampler


@dataclass
class DatetimeSummary:
    """Summary statistics for datetime data.

    This dataclass contains all computed statistics for a datetime column,
    including temporal patterns, monotonicity, and quality indicators.
    """

    name: str
    count: int
    missing: int
    min_ts: Optional[int]
    max_ts: Optional[int]
    by_hour: List[int]  # 24 counts
    by_dow: List[int]  # 7 counts, Monday=0
    by_month: List[int]  # 12 counts, Jan=1 index => store 12-length
    # v2 additions
    dtype_str: str = "datetime"
    mono_inc: bool = False
    mono_dec: bool = False
    mem_bytes: int = 0
    sample_ts: Optional[List[int]] = None
    sample_scale: float = 1.0
    # Temporal analysis
    time_span_days: float = 0.0
    avg_interval_seconds: float = 0.0
    interval_std_seconds: float = 0.0
    weekend_ratio: float = 0.0
    business_hours_ratio: float = 0.0
    seasonal_pattern: Optional[str] = None


class DatetimeAccumulator:
    """Production-grade datetime accumulator optimized for big data temporal analysis.

    This accumulator provides comprehensive temporal analysis with maximum performance
    through vectorized numpy operations, advanced pattern detection, and memory-efficient
    processing for large-scale time-series datasets.
    """

    def __init__(self, name: str, config: Optional[DatetimeConfig] = None):
        """Initialize datetime accumulator.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
        """
        self.name = name
        self.config = config or DatetimeConfig()

        # Core state
        self.count = 0
        self.missing = 0
        self._dtype_str = "datetime"
        self._mem_bytes = 0

        # Temporal bounds for efficient range tracking
        self._min_ts: Optional[int] = None
        self._max_ts: Optional[int] = None

        # Temporal pattern tracking with pre-allocated arrays
        self.by_hour = [0] * 24
        self.by_dow = [0] * 7
        self.by_month = [0] * 12

        # Data structures optimized for big data
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._sample = ReservoirSampler(self.config.sample_size)

        # Advanced monotonicity detection
        self._monotonicity = (
            MonotonicityDetector()
            if self.config.enable_monotonicity_detection
            else None
        )

        # Interval tracking for temporal analysis with memory bounds
        self._intervals: List[float] = []
        self._last_ts: Optional[int] = None

    def set_dtype(self, dtype_str: str) -> None:
        """Set the data type string.

        Args:
            dtype_str: String representation of the data type
        """
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "datetime"

    @property
    def unique_est(self) -> int:
        """Get unique count estimate for compatibility."""
        return self._uniques.estimate()

    @property
    def min_ts(self) -> Optional[int]:
        """Get minimum timestamp for compatibility."""
        return self._min_ts

    @property
    def max_ts(self) -> Optional[int]:
        """Get maximum timestamp for compatibility."""
        return self._max_ts

    def update(self, arr_ns: Sequence[Optional[int]]) -> None:
        """Update accumulator with timestamp values in nanoseconds using vectorized processing.

        Args:
            arr_ns: Sequence of timestamps in nanoseconds since epoch
        """
        if not arr_ns:
            return

        # Convert to numpy array for maximum performance
        try:
            timestamps = np.asarray(arr_ns, dtype=object)
        except Exception:
            # Fallback to list processing for edge cases
            self._update_fallback(arr_ns)
            return

        # High-performance vectorized processing
        self._process_timestamps_vectorized(timestamps)

    def _process_timestamps_vectorized(self, timestamps: np.ndarray) -> None:
        """Process timestamps using optimized vectorized operations.

        Args:
            timestamps: Numpy array of timestamps
        """
        # Create mask for valid timestamps with optimized validation
        valid_mask = self._create_valid_mask(timestamps)
        valid_timestamps = timestamps[valid_mask]

        if len(valid_timestamps) == 0:
            self.missing += len(timestamps)
            return

        # Convert to numpy array of integers with error handling
        try:
            ts_array = np.array([int(ts) for ts in valid_timestamps], dtype=np.int64)
        except (ValueError, TypeError):
            # Fallback for problematic timestamps
            self._update_fallback(valid_timestamps)
            return

        # Update counts efficiently
        self.count += len(ts_array)
        self.missing += len(timestamps) - len(ts_array)

        # Update bounds with vectorized operations
        self._update_bounds(ts_array)

        # Update temporal patterns with optimized batch processing
        self._update_temporal_patterns(ts_array)

        # Update data structures with batch operations
        for ts in ts_array:
            self._uniques.add(ts)
            self._sample.add(float(ts))

        # Update monotonicity detection with vectorized input
        if self._monotonicity:
            self._monotonicity.update(ts_array.astype(float))

        # Update interval tracking for temporal analysis
        self._update_intervals(ts_array)

    def _create_valid_mask(self, timestamps: np.ndarray) -> np.ndarray:
        """Create mask for valid timestamps with optimized validation.

        Args:
            timestamps: Array of timestamps

        Returns:
            Boolean mask for valid timestamps
        """
        valid_mask = np.ones(len(timestamps), dtype=bool)

        for i, ts in enumerate(timestamps):
            if ts is None:
                valid_mask[i] = False
            elif isinstance(ts, float) and np.isnan(ts):
                valid_mask[i] = False
            elif isinstance(ts, (int, float)) and (ts < 0 or ts > 1e20):
                # Reasonable timestamp bounds (roughly 1970-2100)
                valid_mask[i] = False

        return valid_mask

    def _update_bounds(self, ts_array: np.ndarray) -> None:
        """Update min/max timestamp bounds using vectorized operations.

        Args:
            ts_array: Array of valid timestamps
        """
        if len(ts_array) == 0:
            return

        min_ts = np.min(ts_array)
        max_ts = np.max(ts_array)

        if self._min_ts is None or min_ts < self._min_ts:
            self._min_ts = int(min_ts)

        if self._max_ts is None or max_ts > self._max_ts:
            self._max_ts = int(max_ts)

    def _update_temporal_patterns(self, ts_array: np.ndarray) -> None:
        """Update temporal pattern counts with optimized batch processing.

        Args:
            ts_array: Array of valid timestamps
        """
        if not self.config.enable_temporal_patterns:
            return

        # Convert nanoseconds to seconds for datetime operations
        ts_seconds = ts_array / 1_000_000_000

        # Vectorized datetime operations with error handling
        try:
            # Convert to datetime objects efficiently
            datetimes = [datetime.fromtimestamp(ts) for ts in ts_seconds]

            # Batch update pattern counts for better performance
            for dt in datetimes:
                self.by_hour[dt.hour] += 1
                self.by_dow[dt.weekday()] += 1
                self.by_month[dt.month - 1] += 1  # Convert to 0-based index

        except (ValueError, OSError):
            # Handle invalid timestamps gracefully
            pass

    def _update_intervals(self, ts_array: np.ndarray) -> None:
        """Update interval tracking for temporal analysis with memory management.

        Args:
            ts_array: Array of valid timestamps
        """
        if len(ts_array) == 0:
            return

        # Sort timestamps for interval calculation
        sorted_ts = np.sort(ts_array)

        # Calculate intervals between consecutive timestamps efficiently
        if len(sorted_ts) > 1:
            intervals = np.diff(sorted_ts) / 1_000_000_000  # Convert to seconds
            self._intervals.extend(intervals.tolist())

            # Memory management: keep only recent intervals
            if len(self._intervals) > 10000:
                self._intervals = self._intervals[-5000:]

    def _update_fallback(self, arr_ns: Sequence[Optional[int]]) -> None:
        """Fallback processing for problematic timestamps with robust error handling.

        Args:
            arr_ns: Sequence of timestamps
        """
        for ts in arr_ns:
            if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                self.missing += 1
                continue

            try:
                ts_int = int(ts)
                if ts_int < 0 or ts_int > 1e20:
                    self.missing += 1
                    continue

                self.count += 1
                self._uniques.add(ts_int)
                self._sample.add(float(ts_int))

                # Update bounds
                if self._min_ts is None or ts_int < self._min_ts:
                    self._min_ts = ts_int
                if self._max_ts is None or ts_int > self._max_ts:
                    self._max_ts = ts_int

                # Update monotonicity
                if self._monotonicity:
                    self._monotonicity.update(np.array([float(ts_int)]))

            except (ValueError, TypeError):
                self.missing += 1

    def add_mem(self, n: int) -> None:
        """Add to memory usage tracking.

        Args:
            n: Number of bytes to add
        """
        if not self.config.enable_memory_tracking:
            return

        try:
            self._mem_bytes += int(n)
        except (ValueError, TypeError):
            pass

    def finalize(self) -> DatetimeSummary:
        """Finalize accumulator and return comprehensive summary statistics.

        Returns:
            DatetimeSummary containing all computed statistics
        """
        # Get monotonicity status from advanced detection
        mono_inc, mono_dec = False, False
        if self._monotonicity:
            mono_inc, mono_dec = self._monotonicity.get_monotonicity()

        # Calculate comprehensive temporal analysis metrics
        time_span_days = self._calculate_time_span()
        avg_interval, interval_std = self._calculate_interval_stats()
        weekend_ratio = self._calculate_weekend_ratio()
        business_hours_ratio = self._calculate_business_hours_ratio()
        seasonal_pattern = self._detect_seasonal_pattern()

        # Get sample values efficiently
        sample_vals = self._sample.values()
        sample_ts = [int(ts) for ts in sample_vals] if sample_vals else None

        return DatetimeSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            min_ts=self._min_ts,
            max_ts=self._max_ts,
            by_hour=self.by_hour.copy(),
            by_dow=self.by_dow.copy(),
            by_month=self.by_month.copy(),
            dtype_str=self._dtype_str,
            mono_inc=mono_inc,
            mono_dec=mono_dec,
            mem_bytes=self._mem_bytes,
            sample_ts=sample_ts,
            sample_scale=1.0,
            time_span_days=time_span_days,
            avg_interval_seconds=avg_interval,
            interval_std_seconds=interval_std,
            weekend_ratio=weekend_ratio,
            business_hours_ratio=business_hours_ratio,
            seasonal_pattern=seasonal_pattern,
        )

    def _calculate_time_span(self) -> float:
        """Calculate time span in days efficiently.

        Returns:
            Time span in days
        """
        if self._min_ts is None or self._max_ts is None:
            return 0.0

        span_seconds = (self._max_ts - self._min_ts) / 1_000_000_000
        return span_seconds / (24 * 3600)  # Convert to days

    def _calculate_interval_stats(self) -> tuple[float, float]:
        """Calculate interval statistics using vectorized operations.

        Returns:
            Tuple of (average_interval, interval_std) in seconds
        """
        if not self._intervals:
            return 0.0, 0.0

        intervals_array = np.array(self._intervals)
        avg_interval = float(np.mean(intervals_array))
        interval_std = float(np.std(intervals_array))

        return avg_interval, interval_std

    def _calculate_weekend_ratio(self) -> float:
        """Calculate ratio of weekend timestamps efficiently.

        Returns:
            Ratio of weekend timestamps (Saturday=5, Sunday=6)
        """
        if not self.config.enable_temporal_patterns:
            return 0.0

        weekend_count = self.by_dow[5] + self.by_dow[6]  # Saturday + Sunday
        total_count = sum(self.by_dow)

        return weekend_count / max(1, total_count)

    def _calculate_business_hours_ratio(self) -> float:
        """Calculate ratio of business hours timestamps with optimization.

        Returns:
            Ratio of business hours timestamps (9 AM - 5 PM, Monday-Friday)
        """
        if not self.config.enable_temporal_patterns:
            return 0.0

        business_hours_count = sum(self.by_hour[9:17])  # 9 AM to 5 PM
        business_days_count = sum(self.by_dow[:5])  # Monday to Friday

        total_count = sum(self.by_hour)
        if total_count == 0:
            return 0.0

        # Approximate business hours ratio
        business_ratio = (business_hours_count / total_count) * (
            business_days_count / max(1, sum(self.by_dow))
        )

        return business_ratio

    def _detect_seasonal_pattern(self) -> Optional[str]:
        """Detect seasonal patterns in the data with advanced analysis.

        Returns:
            String describing seasonal pattern or None
        """
        if not self.config.enable_temporal_patterns or not self.by_month:
            return None

        # Find peak months efficiently
        max_count = max(self.by_month)
        peak_months = [
            i + 1 for i, count in enumerate(self.by_month) if count == max_count
        ]

        if len(peak_months) == 1:
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            return f"Peak in {month_names[peak_months[0] - 1]}"

        return "Multiple peaks detected"

    def get_temporal_analysis(self) -> dict[str, Any]:
        """Get comprehensive temporal analysis with advanced metrics.

        Returns:
            Dictionary containing temporal analysis results
        """
        return {
            "total_timestamps": self.count + self.missing,
            "valid_timestamps": self.count,
            "missing_timestamps": self.missing,
            "time_span_days": self._calculate_time_span(),
            "unique_timestamps_estimate": self._uniques.estimate(),
            "avg_interval_seconds": self._calculate_interval_stats()[0],
            "interval_std_seconds": self._calculate_interval_stats()[1],
            "weekend_ratio": self._calculate_weekend_ratio(),
            "business_hours_ratio": self._calculate_business_hours_ratio(),
            "seasonal_pattern": self._detect_seasonal_pattern(),
            "peak_hour": self.by_hour.index(max(self.by_hour))
            if self.by_hour
            else None,
            "peak_day": self.by_dow.index(max(self.by_dow)) if self.by_dow else None,
            "peak_month": self.by_month.index(max(self.by_month)) + 1
            if self.by_month
            else None,
        }

    def merge(self, other: DatetimeAccumulator) -> None:
        """Merge another DatetimeAccumulator efficiently.

        Args:
            other: Another DatetimeAccumulator to merge
        """
        self.count += other.count
        self.missing += other.missing
        self._mem_bytes += other._mem_bytes

        # Merge bounds efficiently
        if other._min_ts is not None:
            if self._min_ts is None or other._min_ts < self._min_ts:
                self._min_ts = other._min_ts

        if other._max_ts is not None:
            if self._max_ts is None or other._max_ts > self._max_ts:
                self._max_ts = other._max_ts

        # Merge temporal patterns with vectorized operations
        for i in range(24):
            self.by_hour[i] += other.by_hour[i]
        for i in range(7):
            self.by_dow[i] += other.by_dow[i]
        for i in range(12):
            self.by_month[i] += other.by_month[i]

        # Merge intervals with memory management
        self._intervals.extend(other._intervals)
        if len(self._intervals) > 10000:
            self._intervals = self._intervals[-5000:]

        # Merge data structures efficiently (approximate)
        other_sample = other._sample.values()
        for ts in other_sample:
            self._uniques.add(int(ts))
            self._sample.add(ts)

    def reset(self) -> None:
        """Reset accumulator to initial state efficiently."""
        self.count = 0
        self.missing = 0
        self._mem_bytes = 0
        self._min_ts = None
        self._max_ts = None
        self.by_hour = [0] * 24
        self.by_dow = [0] * 7
        self.by_month = [0] * 12
        self._intervals = []
        self._last_ts = None

        # Reset data structures
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._sample = ReservoirSampler(self.config.sample_size)
        if self._monotonicity:
            self._monotonicity.reset()
