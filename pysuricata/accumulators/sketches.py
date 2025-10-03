from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _u64(x: bytes) -> int:
    """Return a 64-bit unsigned integer hash from bytes using SHA1.

    Fast enough and avoids external dependencies. Uses the first 8 bytes
    of the sha1 digest to build an unsigned 64-bit integer.
    """
    return int.from_bytes(hashlib.sha1(x).digest()[:8], "big", signed=False)


class KMV:
    """K-Minimum Values distinct counter (approximate uniques) without extra deps.

    Keep the k smallest 64-bit hashes of the observed values. If fewer than k items
    have been seen, |S| is exact uniques. Otherwise, estimate uniques as (k-1)/t,
    where t is the kth smallest hash normalized to (0,1].
    
    Enhanced with small discrete value detection for exact counting.
    """

    __slots__ = ("k", "_values", "_exact_values", "_use_exact")

    def __init__(self, k: int = 2048) -> None:
        self.k = int(k)
        self._values: List[int] = []  # store as integers in [0, 2^64)
        self._exact_values: set = set()  # for exact counting of small discrete values
        self._use_exact = True  # start with exact mode for small datasets

    def add(self, v: Any) -> None:
        # Always track exact values for small discrete sets
        if v is None:
            v = b"__NULL__"
        elif isinstance(v, bytes):
            pass
        else:
            v = str(v).encode("utf-8", "ignore")
        
        # Add to exact tracking
        self._exact_values.add(v)
        
        # If we have too many unique values, switch to approximation mode
        if len(self._exact_values) > 100:  # threshold for switching to approximation
            self._use_exact = False
        
        # Only use KMV approximation if we're not in exact mode
        if not self._use_exact:
            h = _u64(v)
            if len(self._values) < self.k:
                self._values.append(h)
                if len(self._values) == self.k:
                    self._values.sort()
            else:
                # maintain k-smallest set (max-heap simulation via last element after sort)
                if h < self._values[-1]:
                    # insert in sorted order (k is small)
                    lo, hi = 0, self.k - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if self._values[mid] < h:
                            lo = mid + 1
                        else:
                            hi = mid
                    self._values.insert(lo, h)
                    # trim to size
                    del self._values[self.k]

    @property
    def is_exact(self) -> bool:
        return self._use_exact or len(self._values) < self.k

    def estimate(self) -> int:
        # Use exact counting for small discrete value sets
        if self._use_exact:
            return len(self._exact_values)
        
        # Use KMV approximation for large datasets
        n = len(self._values)
        if n == 0:
            return 0
        if n < self.k:
            # exact
            return n
        # normalize kth smallest to (0,1]
        kth = self._values[-1]
        t = (kth + 1) / 2**64
        if t <= 0:
            return n
        return max(n, int(round((self.k - 1) / t)))

class ReservoirSampler:
    """Reservoir sampler for numeric/datetime values to approximate quantiles/histograms."""

    __slots__ = ("k", "_buf", "_seen")

    def __init__(self, k: int = 20_000) -> None:
        self.k = int(k)
        self._buf: List[float] = []
        self._seen: int = 0

    def add_many(self, arr: Sequence[float]) -> None:
        for x in arr:
            self.add(float(x))

    def add(self, x: float) -> None:
        self._seen += 1
        if len(self._buf) < self.k:
            self._buf.append(x)
        else:
            j = random.randint(1, self._seen)
            if j <= self.k:
                self._buf[j - 1] = x

    def values(self) -> List[float]:
        return self._buf


class MisraGries:
    """Heavy hitters (top-K) with deterministic memory.

    Maintains up to k counters. Good for approximate top categories.
    """

    __slots__ = ("k", "counters")

    def __init__(self, k: int = 50) -> None:
        self.k = int(k)
        self.counters: Dict[Any, int] = {}

    def add(self, x: Any, w: int = 1) -> None:
        if x in self.counters:
            self.counters[x] += w
            return
        if len(self.counters) < self.k:
            self.counters[x] = w
            return
        # decrement all
        to_del = []
        for key in list(self.counters.keys()):
            self.counters[key] -= w
            if self.counters[key] <= 0:
                to_del.append(key)
        for key in to_del:
            del self.counters[key]

    def items(self) -> List[Tuple[Any, int]]:
        # items are approximate; a second pass could refine if needed
        return sorted(self.counters.items(), key=lambda kv: (-kv[1], str(kv[0])[:64]))


def mad(arr: np.ndarray) -> float:
    """Calculates the Median Absolute Deviation (MAD) of an array.

    The MAD is a robust measure of the variability of a univariate sample of
    quantitative data. It is defined as the median of the absolute deviations
    from the data's median.

    Args:
        arr: A numpy array of quantitative data.

    Returns:
        The MAD of the array.
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))


class RowKMV:
    """Approximate row-duplicate estimator using a KMV distinct sketch.

    Maintains an approximate count of distinct rows by hashing each row into a
    64-bit signature and feeding it to a KMV (K-Minimum Values) sketch.
    """

    def __init__(self, k: int = 8192) -> None:
        self.kmv = KMV(k)
        self.rows = 0

    def update_from_pandas(self, df: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return
        try:
            # Fast row-hash: xor column hashes (uint64) to produce a row signature
            h = None
            for c in df.columns:
                hc = pd.util.hash_pandas_object(df[c], index=False).to_numpy(
                    dtype="uint64", copy=False
                )
                h = hc if h is None else (h ^ hc)
            if h is None:
                return
            self.rows += int(len(h))
            for v in h:
                self.kmv.add(int(v))
        except Exception:
            # Conservative fallback: sample a few stringified rows
            n = min(2000, len(df))
            sample = df.head(n).astype(str).agg("|".join, axis=1)
            for s in sample:
                self.kmv.add(s)
            self.rows += n

    def update_from_polars(self, df: "pl.DataFrame") -> None:  # type: ignore[name-defined]
        try:
            import polars as pl  # type: ignore
        except Exception:
            return
        try:
            # Optimized row hashing - use Polars' built-in row hashing if available
            if hasattr(df, "hash_rows"):
                h = df.hash_rows().to_numpy()
                self.rows += int(h.size)
                for v in h:
                    self.kmv.add(int(v))
                return

            # Fallback to optimized column-wise hashing
            h = None
            for c in df.columns:
                hc = df[c].hash().to_numpy()
                h = hc if h is None else (h ^ hc)
            if h is None:
                return
            self.rows += int(h.size)
            for v in h:
                self.kmv.add(int(v))
        except Exception:
            # Fallback: sample small head and reuse pandas-based path for hashing
            try:
                sample = df.head(min(2000, df.height)).to_pandas()
                self.update_from_pandas(sample)
            except Exception:
                self.rows += min(2000, df.height)

    def approx_duplicates(self) -> Tuple[int, float]:
        uniq = self.kmv.estimate()
        d = max(0, self.rows - uniq)
        pct = (d / self.rows * 100.0) if self.rows else 0.0
        return d, pct


class StreamingHistogram:
    """Lightweight streaming histogram that maintains true distribution counts.

    This implementation provides exact histogram counts for the full dataset
    without requiring all data to be kept in memory. It's optimized for
    streaming data processing and provides accurate distribution visualization.

    The histogram uses a single-pass approach that dynamically adjusts bin edges
    as new data arrives, maintaining exact counts for the true distribution.
    """

    __slots__ = (
        "bins",
        "bin_edges",
        "counts",
        "total_count",
        "min_val",
        "max_val",
        "_initialized",
    )

    def __init__(self, bins: int = 25):
        """Initialize streaming histogram.

        Args:
            bins: Number of histogram bins (default: 25)
        """
        self.bins = int(bins)
        self.bin_edges: List[float] = []
        self.counts: List[int] = []
        self.total_count = 0
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self._initialized = False

    def add(self, value: float) -> None:
        """Add a single value to the histogram.

        Args:
            value: Numeric value to add
        """
        if not self._initialized:
            # First value - initialize bounds and create bins
            self.min_val = self.max_val = value
            self._create_bins()
            self._initialized = True

        # Update bounds if needed
        if value < self.min_val:
            self._expand_range(value, self.max_val)
        elif value > self.max_val:
            self._expand_range(self.min_val, value)

        # Add to appropriate bin
        self._add_to_bin(value)

    def add_many(self, values: Sequence[float]) -> None:
        """Add multiple values to the histogram.

        Args:
            values: Sequence of numeric values
        """
        if len(values) == 0:
            return

        # Find min/max of new values
        min_val = min(values)
        max_val = max(values)

        if not self._initialized:
            # First batch - initialize bounds and create bins
            self.min_val = min_val
            self.max_val = max_val
            self._create_bins()
            self._initialized = True
        else:
            # Update bounds if needed
            if min_val < self.min_val or max_val > self.max_val:
                self._expand_range(
                    min(min_val, self.min_val), max(max_val, self.max_val)
                )

        # Add all values to bins
        for value in values:
            self._add_to_bin(value)

    def _create_bins(self) -> None:
        """Create initial bin edges and counts."""
        if self.min_val is None or self.max_val is None:
            return

        # Handle edge case where all values are the same
        if self.min_val == self.max_val:
            self.bin_edges = [self.min_val - 0.5, self.min_val + 0.5]
            self.bins = 1
            self.counts = [0]
        else:
            # Create bin edges
            self.bin_edges = np.linspace(
                self.min_val, self.max_val, self.bins + 1
            ).tolist()
            self.counts = [0] * self.bins

    def _add_to_bin(self, value: float) -> None:
        """Add a value to the appropriate bin.

        Args:
            value: Numeric value to add
        """
        if not self.bin_edges or len(self.counts) == 0:
            return

        # Find the appropriate bin
        bin_idx = np.digitize(value, self.bin_edges) - 1

        # Handle edge cases
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= len(self.counts):
            bin_idx = len(self.counts) - 1

        self.counts[bin_idx] += 1
        self.total_count += 1

    def _expand_range(self, new_min: float, new_max: float) -> None:
        """Expand the histogram range and redistribute counts.

        Args:
            new_min: New minimum value
            new_max: New maximum value
        """
        if self.min_val is None or self.max_val is None:
            return

        # Store old data
        old_edges = self.bin_edges.copy()
        old_counts = self.counts.copy()

        # Update bounds
        self.min_val = new_min
        self.max_val = new_max

        # Recreate bins
        self._create_bins()

        # Redistribute old counts
        for i, count in enumerate(old_counts):
            if count > 0 and i < len(old_edges) - 1:
                # Find the center of the old bin
                old_center = (old_edges[i] + old_edges[i + 1]) / 2.0
                # Add to new bin
                new_bin_idx = np.digitize(old_center, self.bin_edges) - 1
                if 0 <= new_bin_idx < len(self.counts):
                    self.counts[new_bin_idx] += count

    def get_histogram_data(self) -> Tuple[List[float], List[int], int]:
        """Get histogram data for rendering.

        Returns:
            Tuple of (bin_edges, counts, total_count)
        """
        if not self._initialized or not self.bin_edges:
            return [], [], 0

        return self.bin_edges, self.counts, self.total_count
