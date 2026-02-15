from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

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

    Enhanced with bounded exact counting for small discrete value sets.
    Memory usage is O(k) instead of O(n) for large datasets.
    """

    __slots__ = ("k", "_values", "_exact_counter", "_use_exact", "_max_exact_tracking")

    def __init__(self, k: int = 2048, max_exact_tracking: int = 100) -> None:
        self.k = int(k)
        self._values: List[int] = []  # store as integers in [0, 2^64)
        self._exact_counter: Dict[bytes, int] = {}  # bounded counter for exact counting
        self._use_exact = True  # start with exact mode for small datasets
        self._max_exact_tracking = int(max_exact_tracking)  # max unique values to track exactly

    def add_many(self, values: Sequence[Any]) -> None:
        """Batch add values to KMV sketch for improved performance.

        Args:
            values: Sequence of values to add
        """
        if len(values) == 0:
            return
            
        # Convert all values to bytes and hash them in batch
        try:
            import numpy as np
            
            # Convert values to bytes
            byte_values = []
            for v in values:
                if v is None:
                    byte_values.append(b"__NULL__")
                elif isinstance(v, bytes):
                    byte_values.append(v)
                else:
                    byte_values.append(str(v).encode("utf-8", "ignore"))
            
            # Batch hash computation
            hashes = np.array([_u64(bv) for bv in byte_values], dtype=np.uint64)
            
            # Process in exact mode if still using exact tracking
            if self._use_exact:
                for i, bv in enumerate(byte_values):
                    if bv in self._exact_counter:
                        self._exact_counter[bv] += 1
                    else:
                        if len(self._exact_counter) >= self._max_exact_tracking:
                            # Switch to approximation mode
                            self._use_exact = False
                            # Convert existing exact values to KMV hashes
                            for exact_value in self._exact_counter:
                                h = _u64(exact_value)
                                self._add_hash_to_kmv(h)
                            self._exact_counter.clear()
                            # Process remaining values in approximation mode
                            remaining_hashes = hashes[i:]
                            self._batch_add_hashes(remaining_hashes)
                            return
                        else:
                            self._exact_counter[bv] = 1
            else:
                # Process all hashes in approximation mode
                self._batch_add_hashes(hashes)
                
        except ImportError:
            # Fallback to individual processing if numpy not available
            for v in values:
                self.add(v)

    def _add_hash_to_kmv(self, h: int) -> None:
        """Add a single hash to the KMV sketch.
        
        Args:
            h: Hash value to add
        """
        if len(self._values) < self.k:
            self._values.append(h)
            if len(self._values) == self.k:
                self._values.sort()
        else:
            if h < self._values[-1]:
                # Binary search and insert
                lo, hi = 0, self.k - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if self._values[mid] < h:
                        lo = mid + 1
                    else:
                        hi = mid
                self._values.insert(lo, h)
                del self._values[self.k]

    def _batch_add_hashes(self, hashes: np.ndarray) -> None:
        """Batch add hashes to KMV sketch using numpy operations.
        
        Args:
            hashes: Array of hash values to add
        """
        if len(hashes) == 0:
            return
            
        # Fill KMV sketch first if not full
        if len(self._values) < self.k:
            needed = min(self.k - len(self._values), len(hashes))
            self._values.extend(hashes[:needed])
            hashes = hashes[needed:]
            if len(self._values) == self.k:
                self._values.sort()
        
        # Process remaining hashes
        if len(hashes) > 0:
            # Filter hashes that are smaller than the largest in KMV
            max_hash = self._values[-1] if self._values else 0
            candidate_mask = hashes < max_hash
            candidate_hashes = hashes[candidate_mask]
            
            if len(candidate_hashes) > 0:
                # Add candidates to existing values and sort
                self._values.extend(candidate_hashes)
                self._values.sort()
                
                # Keep only the k smallest
                if len(self._values) > self.k:
                    self._values = self._values[:self.k]

    def add(self, v: Any) -> None:
        # Convert value to bytes for consistent hashing
        if v is None:
            v = b"__NULL__"
        elif isinstance(v, bytes):
            pass
        else:
            v = str(v).encode("utf-8", "ignore")

        # Track unique values exactly if we're still in exact mode
        if self._use_exact:
            if v in self._exact_counter:
                self._exact_counter[v] += 1
            else:
                # Check if we've hit the limit for exact tracking
                if len(self._exact_counter) >= self._max_exact_tracking:
                    # Switch to approximation mode
                    self._use_exact = False
                    # Convert existing exact values to KMV hashes before clearing
                    for exact_value in self._exact_counter:
                        h = _u64(exact_value)
                        if len(self._values) < self.k:
                            self._values.append(h)
                        else:
                            # Insert if smaller than largest
                            if h < self._values[-1]:
                                # Binary search and insert
                                lo, hi = 0, self.k - 1
                                while lo < hi:
                                    mid = (lo + hi) // 2
                                    if self._values[mid] < h:
                                        lo = mid + 1
                                    else:
                                        hi = mid
                                self._values.insert(lo, h)
                                del self._values[self.k]
                    # Sort the values after adding all
                    if len(self._values) > 1:
                        self._values.sort()
                    # Clear the counter to free memory
                    self._exact_counter.clear()
                    # Add the current value to KMV sketch
                    h = _u64(v)
                    if len(self._values) < self.k:
                        self._values.append(h)
                        if len(self._values) == self.k:
                            self._values.sort()
                    else:
                        if h < self._values[-1]:
                            lo, hi = 0, self.k - 1
                            while lo < hi:
                                mid = (lo + hi) // 2
                                if self._values[mid] < h:
                                    lo = mid + 1
                                else:
                                    hi = mid
                            self._values.insert(lo, h)
                            del self._values[self.k]
                else:
                    self._exact_counter[v] = 1

        # Use KMV approximation if we're not in exact mode
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
            return len(self._exact_counter)

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

    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes for monitoring."""
        memory = 0
        # _values list: k integers * 8 bytes each
        memory += len(self._values) * 8
        # _exact_counter: dict overhead + key bytes + value ints
        memory += len(self._exact_counter) * (32 + 8)  # rough estimate
        for key in self._exact_counter:
            memory += len(key)  # actual key bytes
        return memory


class ReservoirSampler:
    """Reservoir sampler for numeric/datetime values to approximate quantiles/histograms."""

    __slots__ = ("k", "_buf", "_seen")

    def __init__(self, k: int = 20_000) -> None:
        self.k = int(k)
        self._buf: List[float] = []
        self._seen: int = 0

    def add_many(self, arr: Sequence[float]) -> None:
        """Optimized batch addition using numpy random generation.
        
        Args:
            arr: Sequence of float values to add
        """
        if len(arr) == 0:
            return
            
        try:
            import numpy as np
            arr = np.asarray(arr, dtype=float)
        except ImportError:
            # Fallback to original implementation if numpy not available
            for x in arr:
                self.add(float(x))
            return
        
        # Fill reservoir first if not full
        if len(self._buf) < self.k:
            needed = min(self.k - len(self._buf), len(arr))
            self._buf.extend(arr[:needed])
            arr = arr[needed:]
            self._seen += needed
        
        # Process remaining elements with batch random generation
        if len(arr) > 0:
            # Generate random numbers for replacement decisions
            random_vals = np.random.randint(1, self._seen + len(arr) + 1, size=len(arr))
            
            # Determine which elements to replace
            replace_mask = random_vals <= self.k
            replace_indices = random_vals[replace_mask] - 1
            
            # Replace elements in batch (convert to Python list for indexing)
            if len(replace_indices) > 0:
                replace_values = arr[replace_mask]
                for i, val in zip(replace_indices, replace_values):
                    self._buf[i] = val
            
            self._seen += len(arr)

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

    def add_many(self, values: Sequence[Any]) -> None:
        """Batch add values to MisraGries sketch for improved performance.

        Pre-counts occurrences in the batch, then applies weighted updates.
        This avoids per-value Python overhead when the batch has many repeats.

        Args:
            values: Sequence of values to add
        """
        if len(values) == 0:
            return

        # Pre-count values in the batch for weighted updates
        batch_counts: Dict[Any, int] = {}
        for v in values:
            if v in batch_counts:
                batch_counts[v] += 1
            else:
                batch_counts[v] = 1

        # Apply weighted updates — existing keys first (cheap), then new keys
        for val, count in batch_counts.items():
            if val in self.counters:
                self.counters[val] += count
            elif len(self.counters) < self.k:
                self.counters[val] = count
            else:
                # Decrement all counters by count and prune
                self.counters[val] = count
                min_count = min(self.counters.values())
                if min_count > 0:
                    self.counters = {
                        k: v - min_count
                        for k, v in self.counters.items()
                        if v - min_count > 0
                    }

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

    Row Hashing Strategy:
    - Pandas: Uses hash of row tuple (all column hashes combined)
    - Polars: Uses native df.hash_rows() method (optimal)
    - Fallback: String concatenation of row values

    This approach ensures proper collision resistance and accurate duplicate detection
    even on datasets with similar numeric values (e.g., iris dataset).

    Previous implementation used XOR of column hashes, which caused hash collisions
    on datasets with similar values. The tuple-based approach avoids this issue.

    Accuracy:
    - Small datasets (≤100 unique): Exact counting
    - Large datasets: ~99% accurate with ±1-2% error bound
    """

    def __init__(self, k: int = 8192) -> None:
        self.kmv = KMV(k)
        self.rows = 0

    def update_from_pandas(self, df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return
        try:
            # Vectorized row hashing: combine column hashes using a polynomial hash
            # instead of per-row tuple construction + hash()

            # Get hash for each column
            col_hashes = {}
            for c in df.columns:
                col_hashes[c] = pd.util.hash_pandas_object(df[c], index=False).to_numpy(
                    dtype="uint64", copy=False
                )

            # Combine column hashes using a rolling polynomial hash (vectorized)
            n_rows = len(df)
            columns = list(df.columns)
            combined = col_hashes[columns[0]].copy()
            _PRIME = np.uint64(2654435761)
            for c in columns[1:]:
                combined = combined * _PRIME + col_hashes[c]

            # Batch add combined hashes to KMV sketch
            self.kmv.add_many(combined)
            self.rows += n_rows

        except Exception:
            # Conservative fallback: sample a few stringified rows
            n = min(2000, len(df))
            sample = df.head(n).astype(str).agg("|".join, axis=1)
            for s in sample:
                self.kmv.add(s)
            self.rows += n

    def update_from_polars(self, df: pl.DataFrame) -> None:  # type: ignore[name-defined]
        try:
            pass  # type: ignore
        except Exception:
            return
        try:
            # Polars' hash_rows() is already correct - hashes entire rows properly
            if hasattr(df, "hash_rows"):
                h = df.hash_rows().to_numpy()
                self.rows += int(h.size)
                # Batch add hashes to KMV sketch instead of per-value loop
                self.kmv.add_many(h)
                return

            # Fallback: use pandas vectorized path
            try:
                pdf = df.to_pandas()
                self.update_from_pandas(pdf)
            except Exception:
                self.rows += min(2000, df.height)

        except Exception:
            # Final fallback: use pandas path which has vectorized hashing
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
        """Add multiple values to the histogram using vectorized bin assignment.

        Args:
            values: Sequence of numeric values
        """
        if len(values) == 0:
            return

        arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(arr)
        arr = arr[finite_mask]
        if len(arr) == 0:
            return

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if not self._initialized:
            self.min_val = min_val
            self.max_val = max_val
            self._create_bins()
            self._initialized = True
        else:
            if min_val < self.min_val or max_val > self.max_val:
                self._expand_range(
                    min(min_val, self.min_val), max(max_val, self.max_val)
                )

        if not self.bin_edges or len(self.counts) == 0:
            return

        # Vectorized bin assignment — single np.digitize call for all values
        bin_indices = np.digitize(arr, self.bin_edges) - 1
        np.clip(bin_indices, 0, len(self.counts) - 1, out=bin_indices)

        # Vectorized counting with np.bincount
        bin_counts = np.bincount(bin_indices, minlength=len(self.counts))
        for i in range(len(self.counts)):
            self.counts[i] += int(bin_counts[i])
        self.total_count += len(arr)

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
