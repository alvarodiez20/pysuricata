"""High-performance datetime accumulator optimized for big data temporal analysis.

This module provides a production-ready, scalable implementation of the datetime accumulator
with vectorized operations, comprehensive temporal pattern analysis, and advanced performance
optimizations for processing massive time-series datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .algorithms import MonotonicityDetector
from .config import DatetimeConfig
from .sketches import KMV, ReservoirSampler

# Validity window for nanosecond timestamps. The old bound of -2e18 ns is
# 1906-05-13: every birthdate and historical record before it was silently
# reclassified as missing, so a column of 19th-century dates looked almost
# entirely null rather than old. The real limit is what datetime64[ns] can
# represent in an int64 -- 1677-09-21 to 2262-04-11 -- with int64 min reserved
# by pandas as the NaT sentinel.
_NS_MIN = int(np.iinfo(np.int64).min) + 1
_NS_MAX = int(np.iinfo(np.int64).max)

# 1970-01-01 was a Thursday, which datetime.weekday() numbers 3. Shifting the
# day index by 3 before the modulo puts Monday at 0 for negative days too,
# since numpy's % on integers floors rather than truncating.
_EPOCH_DOW_OFFSET = 3

_NS_PER_DAY = 86_400_000_000_000
_NS_PER_HOUR = 3_600_000_000_000


def _as_ns_int64(arr: Any) -> np.ndarray | None:
    """Return ``arr`` as a contiguous int64 nanosecond array, or None.

    None means "this input has no usable dtype" -- an object array, a list
    holding ``None``, mixed types -- and the caller should take the
    element-wise route instead. Everything else is handled without touching a
    Python object per row.

    Args:
        arr: Candidate sequence of nanosecond timestamps.

    Returns:
        int64 array, or None if the input needs element-wise handling.
    """
    if isinstance(arr, np.ndarray):
        candidate = arr
    else:
        try:
            candidate = np.asarray(arr)
        except Exception:
            return None

    kind = candidate.dtype.kind
    if kind == "i":
        return np.ascontiguousarray(candidate, dtype=np.int64)
    if kind == "u":
        # Unsigned values above int64 max cannot be a valid ns timestamp.
        if candidate.size and int(candidate.max()) > _NS_MAX:
            return None
        return candidate.astype(np.int64)
    if kind == "M":
        # NaT is int64 min, which the validity window already rejects.
        return candidate.astype("datetime64[ns]").view(np.int64)
    if kind == "f":
        # NaN would cast to an arbitrary integer, so map it to the NaT
        # sentinel first -- the window rejects that as missing.
        filled = np.where(np.isnan(candidate), float(np.iinfo(np.int64).min), candidate)
        return filled.astype(np.int64)
    return None


@dataclass
class DatetimeSummary:
    """Summary statistics for datetime data.

    This dataclass contains all computed statistics for a datetime column,
    including temporal patterns, monotonicity, and quality indicators.
    """

    name: str
    count: int
    missing: int
    min_ts: int | None
    max_ts: int | None
    by_hour: list[int]  # 24 counts
    by_dow: list[int]  # 7 counts, Monday=0
    by_month: list[int]  # 12 counts, Jan=1 index => store 12-length
    by_year: dict[int, int]  # Dynamic year counts
    # v2 additions
    dtype_str: str = "datetime"
    mono_inc: bool = False
    mono_dec: bool = False
    mem_bytes: int = 0
    sample_ts: list[int] | None = None
    sample_scale: float = 1.0
    # Temporal analysis
    time_span_days: float = 0.0
    avg_interval_seconds: float = 0.0
    interval_std_seconds: float = 0.0
    weekend_ratio: float = 0.0
    business_hours_ratio: float = 0.0
    seasonal_pattern: str | None = None
    # Source timezone (before UTC conversion)
    source_timezone: str | None = None
    # Missing fields for renderer compatibility
    unique_est: int = 0
    chunk_metadata: Sequence[tuple[int, int, int]] | None = None


class DatetimeAccumulator:
    """Production-grade datetime accumulator optimized for big data temporal analysis.

    This accumulator provides comprehensive temporal analysis with maximum performance
    through vectorized numpy operations, advanced pattern detection, and memory-efficient
    processing for large-scale time-series datasets.
    """

    def __init__(
        self,
        name: str,
        config: DatetimeConfig | None = None,
        *,
        seed: int | None = None,
    ):
        """Initialize datetime accumulator.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
            seed: Seed for this column's sampling. None draws from OS entropy.
                Sampling never touches the process-global RNG either way.
        """
        self.name = name
        self.config = config or DatetimeConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Core state
        self.count = 0
        self.missing = 0
        self._dtype_str = "datetime"
        self._mem_bytes = 0

        # Temporal bounds for efficient range tracking
        self._min_ts: int | None = None
        self._max_ts: int | None = None

        # Temporal pattern tracking with pre-allocated arrays
        self.by_hour = [0] * 24
        self.by_dow = [0] * 7
        self.by_month = [0] * 12
        self.by_year: dict[int, int] = {}  # Year -> count mapping

        # Data structures optimized for big data
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._sample = ReservoirSampler(self.config.sample_size, rng=self._rng)

        # Advanced monotonicity detection
        self._monotonicity = (
            MonotonicityDetector()
            if self.config.enable_monotonicity_detection
            else None
        )

        # Source timezone (captured before UTC conversion)
        self._source_timezone: str | None = None

        # Interval tracking for temporal analysis with memory bounds
        self._intervals: list[float] = []
        self._last_ts: int | None = None

    def set_dtype(self, dtype_str: str) -> None:
        """Set the data type string and extract timezone metadata.

        Args:
            dtype_str: String representation of the data type
        """
        try:
            self._dtype_str = str(dtype_str)
            # Extract timezone from dtype string (e.g. "datetime64[ns, US/Eastern]")
            if "," in dtype_str:
                tz_part = dtype_str.split(",", 1)[1].rstrip("]").strip()
                if tz_part and tz_part != "UTC":
                    self._source_timezone = tz_part
            elif "tz=" in dtype_str.lower():
                # Polars-style: Datetime(time_unit='us', time_zone='Europe/Berlin')
                import re

                m = re.search(r"time_zone=['\"]([^'\"]+)['\"]", dtype_str)
                if m and m.group(1) != "UTC":
                    self._source_timezone = m.group(1)
        except Exception:
            self._dtype_str = "datetime"

    @property
    def unique_est(self) -> int:
        """Get unique count estimate for compatibility."""
        return self._uniques.estimate()

    @property
    def min_ts(self) -> int | None:
        """Get minimum timestamp for compatibility."""
        return self._min_ts

    @property
    def max_ts(self) -> int | None:
        """Get maximum timestamp for compatibility."""
        return self._max_ts

    def update(self, arr_ns: Sequence[int | None]) -> None:
        """Update accumulator with timestamp values in nanoseconds.

        Args:
            arr_ns: Sequence of timestamps in nanoseconds since epoch. An int64
                or datetime64 array takes the vectorised path; anything else
                (a list containing ``None``, mixed types) falls back to the
                element-wise route.
        """
        if len(arr_ns) == 0:
            return

        ns = _as_ns_int64(arr_ns)
        if ns is not None:
            self._process_ns(ns)
            return

        # Mixed or object input: no dtype to work with, so go element-wise.
        try:
            timestamps = np.asarray(arr_ns, dtype=object)
        except Exception:
            self._update_fallback(arr_ns)
            return

        self._process_timestamps_vectorized(timestamps)

    def _process_ns(self, ns: np.ndarray) -> None:
        """Fold in a batch of int64 nanosecond timestamps, wholly vectorised.

        Args:
            ns: int64 array of nanoseconds since the epoch. Out-of-window
                values, including pandas' NaT sentinel, count as missing.
        """
        valid = (ns >= _NS_MIN) & (ns <= _NS_MAX)
        n_valid = int(np.count_nonzero(valid))
        self.missing += ns.size - n_valid
        if n_valid == 0:
            return

        vals = ns if n_valid == ns.size else ns[valid]
        self.count += n_valid

        self._update_bounds(vals)
        self._update_temporal_patterns(vals)

        # add_many, not a loop of add(): both entry points hash a 64-bit
        # integer to the same value, so the distinct estimate is unchanged.
        self._uniques.add_many(vals)
        self._sample.add_many(vals.astype(np.float64))

        if self._monotonicity:
            self._monotonicity.update(vals.astype(np.float64))

        self._update_intervals(vals)

    def _process_timestamps_vectorized(self, timestamps: np.ndarray) -> None:
        """Process timestamps using optimized vectorized operations.

        Args:
            timestamps: Numpy array of timestamps
        """
        # Create mask for valid timestamps with optimized validation
        valid_mask = self._create_valid_mask(timestamps)
        valid_timestamps = timestamps[valid_mask]
        rejected = len(timestamps) - len(valid_timestamps)

        if len(valid_timestamps) == 0:
            self.missing += len(timestamps)
            return

        # Convert to numpy array of integers with error handling
        try:
            ts_array = np.array([int(ts) for ts in valid_timestamps], dtype=np.int64)
        except (ValueError, TypeError):
            # Fallback for problematic timestamps
            self.missing += rejected
            self._update_fallback(valid_timestamps)
            return

        # Once the batch is int64 the two paths are the same problem, so it
        # folds in through the vectorised route rather than a second copy of it.
        self.missing += rejected
        self._process_ns(ts_array)

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
            elif isinstance(ts, (int, float)) and not (_NS_MIN <= ts <= _NS_MAX):
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
        if ts_array.size == 0:
            return

        # Calendar fields come from datetime64 casts rather than one Python
        # datetime object per row. Two things change as a result, both for the
        # better: the tallies are in UTC, matching the timestamps as stored,
        # where datetime.fromtimestamp() silently used the *machine's* local
        # zone -- so the same data profiled in London and Tokyo produced
        # different hour histograms; and a single out-of-range value no longer
        # makes fromtimestamp raise OSError and drop the whole chunk's patterns.
        ns = np.ascontiguousarray(ts_array, dtype=np.int64)

        # Divide in integers rather than casting datetime64[ns] to a coarser
        # unit. The cast overflows at the bottom of the window: int64.min + 1 ns
        # is 1677-09-21, and numpy reports it as day *+106750* -- sign flipped --
        # which produced hour 46 and crashed np.bincount. Floor division is
        # exact here and shrinks the magnitude instead of growing it, and it
        # floors toward negative infinity, which is what pre-1970 instants need.
        days = np.floor_divide(ns, _NS_PER_DAY)
        hours = np.floor_divide(ns, _NS_PER_HOUR) - days * 24
        dow = (days + _EPOCH_DOW_OFFSET) % 7
        # Safe to go through datetime64 from here: `days` is a small integer,
        # so the month cast has nothing left to overflow.
        months = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)

        self.by_hour = (
            np.asarray(self.by_hour) + np.bincount(hours, minlength=24)
        ).tolist()
        self.by_dow = (np.asarray(self.by_dow) + np.bincount(dow, minlength=7)).tolist()
        self.by_month = (
            np.asarray(self.by_month) + np.bincount(months % 12, minlength=12)
        ).tolist()

        years, counts = np.unique(months // 12 + 1970, return_counts=True)
        for year, count in zip(years.tolist(), counts.tolist(), strict=True):
            self.by_year[year] = self.by_year.get(year, 0) + count

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

    def _update_fallback(self, arr_ns: Sequence[int | None]) -> None:
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
                # Same window as the vectorised path. This line kept the old
                # -2e18 bound (1906-05-13) after the widening, so a pre-1906
                # timestamp that fell through to the fallback was still counted
                # as missing rather than as old.
                if not (_NS_MIN <= ts_int <= _NS_MAX):
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

    def finalize(
        self, chunk_metadata: list[tuple[int, int, int]] | None = None
    ) -> DatetimeSummary:
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
            by_year=self.by_year.copy(),
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
            source_timezone=self._source_timezone,
            unique_est=self._uniques.estimate(),
            chunk_metadata=chunk_metadata,
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

    def _detect_seasonal_pattern(self) -> str | None:
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
        # Add year merging
        for year, count in other.by_year.items():
            self.by_year[year] = self.by_year.get(year, 0) + count

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
        self.by_year = {}
        self._intervals = []
        self._last_ts = None

        # A reset accumulator must replay identically, so rewind the generator
        # rather than continuing its stream.
        self._rng = np.random.default_rng(self._seed)

        # Reset data structures
        self._uniques = KMV(self.config.uniques_sketch_size)
        self._sample = ReservoirSampler(self.config.sample_size, rng=self._rng)
        if self._monotonicity:
            self._monotonicity.reset()
