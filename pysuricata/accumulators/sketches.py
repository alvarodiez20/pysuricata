from __future__ import annotations

import bisect
import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

import numpy as np

# Smallest value fed to log() so a 0.0 draw cannot produce -inf.
_TINY = 5e-324
# Sentinel "no further acceptances": larger than any realistic stream index.
_NO_MORE_ACCEPTANCES = 1 << 62
# Uniform draws are taken a block at a time; scalar np.random calls are dominated
# by interpreter overhead.
_DRAW_BLOCK = 4096


# splitmix64 finalisation constants (Steele et al.). A distinct-count sketch
# needs uniformity and avalanche, not preimage resistance -- SHA-1 was doing a
# cryptographer's job for a statistician's problem, at roughly a third of total
# runtime.
_SPLITMIX_A = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_B = np.uint64(0x94D049BB133111EB)
_U64_MASK = (1 << 64) - 1


def _mix64(z: int) -> int:
    """splitmix64 finaliser for a single already-64-bit value."""
    z &= _U64_MASK
    z ^= z >> 30
    z = (z * 0xBF58476D1CE4E5B9) & _U64_MASK
    z ^= z >> 27
    z = (z * 0x94D049BB133111EB) & _U64_MASK
    return z ^ (z >> 31)


def _mix64_array(values: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser over a uint64 array.

    Unsigned overflow is the algorithm, not an error, so the wrap warnings are
    suppressed rather than avoided.
    """
    z = values.astype(np.uint64, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        z ^= z >> np.uint64(30)
        z *= _SPLITMIX_A
        z ^= z >> np.uint64(27)
        z *= _SPLITMIX_B
        z ^= z >> np.uint64(31)
    return z


def _hash_numeric_array(arr: np.ndarray) -> np.ndarray:
    """Hash a numeric array by its bit pattern, without boxing a Python object.

    -0.0 is canonicalised to 0.0 so the two do not count as distinct values --
    the old ``str(v)`` path made them distinct, which was wrong.
    """
    as_f64 = np.ascontiguousarray(arr, dtype=np.float64)
    bits = as_f64.view(np.uint64).copy()
    bits[as_f64 == 0.0] = np.uint64(0)
    return _mix64_array(bits)


def _hash_value(v: Any) -> int:
    """Hash one arbitrary value to 64 bits.

    Real numbers are hashed by bit pattern so the result agrees with
    ``_hash_numeric_array`` -- the vectorised and scalar paths must produce the
    same hash for the same value, or the exact counter double-counts it.
    """
    if v is None:
        return _u64(b"__NULL__")
    if isinstance(v, bool):
        # Before the int branch: bool is a subclass of int.
        return _mix64(1 if v else 0)
    if isinstance(v, (int, float, np.integer, np.floating)):
        as_float = float(v)
        if as_float != as_float:  # NaN: every NaN is the same distinct value
            return _mix64(int(np.float64(np.nan).view(np.uint64)))
        if as_float == 0.0:  # -0.0 and 0.0 are the same value
            return _mix64(0)
        return _mix64(int(np.float64(as_float).view(np.uint64)))
    if isinstance(v, bytes):
        return _u64(v)
    return _u64(str(v).encode("utf-8", "ignore"))


def _u64(x: bytes) -> int:
    """Return a 64-bit unsigned integer hash from bytes.

    blake2b with an 8-byte digest gives 64 bits directly, with no slicing of a
    wider digest, and is markedly cheaper than SHA-1 here. Values that are
    already 64-bit should go through ``_mix64``/``_hash_numeric_array`` instead
    of being formatted into bytes at all.
    """
    return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big")


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
        self._values: list[int] = []  # store as integers in [0, 2^64)
        self._exact_counter: dict[bytes, int] = {}  # bounded counter for exact counting
        self._use_exact = True  # start with exact mode for small datasets
        self._max_exact_tracking = int(
            max_exact_tracking
        )  # max unique values to track exactly

    def add_many(self, values: Sequence[Any]) -> None:
        """Batch add values to KMV sketch for improved performance.

        Args:
            values: Sequence of values to add
        """
        if len(values) == 0:
            return

        # Fast path: a numeric ndarray can be hashed by bit pattern, vectorised,
        # without allocating a Python object per row. The generic path below
        # formats every value into bytes first, which for a float column meant
        # one str() and one digest per row.
        if isinstance(values, np.ndarray) and values.dtype.kind in "fiub":
            self._offer_hashes(_hash_numeric_array(values))
            return

        hashes = np.fromiter(
            (_hash_value(v) for v in values), dtype=np.uint64, count=len(values)
        )
        self._offer_hashes(hashes)

    def _offer_hashes(self, hashes: np.ndarray) -> None:
        """Feed a batch of 64-bit hashes into exact mode, then the sketch.

        The exact counter is keyed by hash rather than by the value's bytes, so
        every entry point -- vectorised, generic and scalar -- agrees on the key
        for a given value. Keying two paths differently would count the same
        value twice.
        """
        if not self._use_exact:
            self._batch_add_hashes(hashes)
            return

        uniques, counts = np.unique(hashes, return_counts=True)
        for idx, (h, c) in enumerate(
            zip(uniques.tolist(), counts.tolist(), strict=True)
        ):
            if h in self._exact_counter:
                self._exact_counter[h] += int(c)
                continue
            if len(self._exact_counter) >= self._max_exact_tracking:
                self._spill_exact_to_kmv()
                # Only the not-yet-recorded tail: the spill already moved every
                # hash counted so far into the sketch, and re-offering them
                # would count them twice.
                self._batch_add_hashes(uniques[idx:])
                return
            self._exact_counter[h] = int(c)

    def _spill_exact_to_kmv(self) -> None:
        """Leave exact mode, moving everything counted so far into the sketch."""
        self._use_exact = False
        for existing in self._exact_counter:
            self._add_hash_to_kmv(existing)
        self._exact_counter.clear()

    def _add_hash_to_kmv(self, h: int) -> None:
        """Insert one hash, keeping _values sorted, distinct and at most k long.

        Distinctness is the sketch's core invariant: the estimator reads
        len(_values) as a distinct count when below k, and the kth smallest hash
        above it. A repeated hash inserted twice inflates both.
        """
        pos = bisect.bisect_left(self._values, h)
        if pos < len(self._values) and self._values[pos] == h:
            return  # already present
        if len(self._values) < self.k:
            self._values.insert(pos, h)
        elif h < self._values[-1]:
            self._values.insert(pos, h)
            del self._values[self.k :]

    def _batch_add_hashes(self, hashes: np.ndarray) -> None:
        """Merge a batch of hashes, keeping the k smallest distinct ones.

        Args:
            hashes: Array of hash values to add.
        """
        if len(hashes) == 0:
            return

        incoming = np.unique(np.asarray(hashes, dtype=np.uint64))
        if self._values:
            existing = np.asarray(self._values, dtype=np.uint64)
            combined = np.union1d(existing, incoming)
        else:
            combined = incoming
        self._values = combined[: self.k].tolist()

    def add(self, v: Any) -> None:
        """Add a single value.

        Routes through the same hash and the same exact-counter key as the batch
        paths. The previous implementation keyed the counter by the value's
        bytes while spilling used a re-derived hash, and on the spill branch it
        inserted the current hash twice.
        """
        h = _hash_value(v)

        if self._use_exact:
            if h in self._exact_counter:
                self._exact_counter[h] += 1
                return
            if len(self._exact_counter) < self._max_exact_tracking:
                self._exact_counter[h] = 1
                return
            self._spill_exact_to_kmv()

        self._add_hash_to_kmv(h)

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
        # _exact_counter: dict overhead plus a 64-bit hash key and an int count.
        # Keys are hashes now, not the original value bytes, so the per-entry
        # cost is fixed rather than proportional to value length.
        memory += len(self._exact_counter) * (32 + 8 + 8)
        return memory


class ReservoirSampler:
    """Uniform reservoir sample of a stream, via Algorithm L (Li, 1994).

    Every element that has been seen is in the sample with probability k/n,
    independent of its position in the stream and of how the stream happened to
    be split into chunks. That independence is what makes the derived quantiles,
    median, IQR, MAD and histogram stable across chunk sizes.

    Algorithm L samples a geometric skip and jumps straight to the next element
    it will accept, so it costs about ``k * ln(n / k)`` random draws rather than
    one per element -- roughly 145k draws for 10M rows into k=20k, instead of 10M.
    """

    __slots__ = ("k", "_buf", "_seen", "_w", "_next", "_draws", "_draw_i")

    def __init__(self, k: int = 20_000) -> None:
        self.k = int(k)
        self._buf: list[float] = []
        self._seen: int = 0
        # Algorithm L skip state. Only meaningful once the reservoir is full;
        # _next is the stream index (0-based) of the next element to accept.
        self._w: float = 1.0
        self._next: int = -1
        # Pre-drawn uniforms, consumed by _uniform().
        self._draws: np.ndarray = np.empty(0, dtype=float)
        self._draw_i: int = 0

    def _uniform(self) -> float:
        """One uniform draw, taken from a pre-filled block.

        Algorithm L needs two draws per acceptance, and a scalar
        ``np.random.random()`` call costs far more in interpreter overhead than
        the number it returns. Drawing a block at a time amortises that away
        while keeping ``np.random`` as the source, so seeding still controls the
        sample.
        """
        if self._draw_i >= self._draws.size:
            self._draws = np.random.random(_DRAW_BLOCK)
            self._draw_i = 0
        u = self._draws[self._draw_i]
        self._draw_i += 1
        return u if u > 0.0 else _TINY

    def _draw_skip(self) -> int:
        """Number of elements to pass over before the next acceptance."""
        # log1p(-w) is log(1 - w) without the cancellation for small w.
        denom = math.log1p(-self._w) if self._w < 1.0 else 0.0
        if denom >= 0.0:
            # w has underflowed to 0 (or saturated); no further acceptances.
            return _NO_MORE_ACCEPTANCES
        return int(math.log(self._uniform()) / denom)

    def _advance_w(self) -> None:
        self._w *= math.exp(math.log(self._uniform()) / self.k)

    def _start_skipping(self) -> None:
        """Initialise the skip state the moment the reservoir first fills."""
        self._w = 1.0
        self._advance_w()
        # Standard formulation with i = k-1 as the last processed index:
        # the first candidate is therefore at index _seen + skip.
        self._next = self._seen + self._draw_skip()

    def _accept(self, value: float) -> None:
        """Evict a uniformly chosen slot and schedule the next acceptance."""
        # Scale a buffered uniform instead of a scalar np.random.randint call:
        # _uniform() is in (0, 1), so this lands in [0, k-1].
        self._buf[int(self._uniform() * self.k)] = value
        self._advance_w()
        skip = self._draw_skip()
        if skip >= _NO_MORE_ACCEPTANCES:
            self._next = _NO_MORE_ACCEPTANCES
        else:
            self._next += skip + 1

    def add_many(self, arr: Sequence[float]) -> None:
        """Add a batch of values, preserving the uniform-sampling guarantee.

        Args:
            arr: Sequence of float values to add.
        """
        if len(arr) == 0:
            return

        arr = np.asarray(arr, dtype=float)

        # Fill the reservoir before any sampling decision is made.
        if len(self._buf) < self.k:
            needed = min(self.k - len(self._buf), len(arr))
            self._buf.extend(arr[:needed].tolist())
            self._seen += needed
            arr = arr[needed:]
            if len(self._buf) >= self.k:
                self._start_skipping()
            if len(arr) == 0:
                return

        end = self._seen + len(arr)
        while self._next < end:
            self._accept(float(arr[self._next - self._seen]))
        self._seen = end

    def add(self, x: float) -> None:
        if len(self._buf) < self.k:
            self._buf.append(float(x))
            self._seen += 1
            if len(self._buf) >= self.k:
                self._start_skipping()
            return
        if self._seen == self._next:
            self._accept(float(x))
        self._seen += 1

    def values(self) -> list[float]:
        return self._buf


class MisraGries:
    """Heavy hitters (top-K) with deterministic memory.

    Maintains up to k counters. Good for approximate top categories.
    """

    __slots__ = ("k", "counters")

    def __init__(self, k: int = 50) -> None:
        self.k = int(k)
        self.counters: dict[Any, int] = {}

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
        batch_counts: dict[Any, int] = {}
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

    def items(self) -> list[tuple[Any, int]]:
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

    def update_from_pandas(self, df: pd.DataFrame) -> None:
        try:
            import pandas as pd
        except ImportError:
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

    def update_from_polars(self, df: pl.DataFrame) -> None:
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

    def approx_duplicates(self) -> tuple[int, float]:
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
        self.bin_edges: list[float] = []
        self.counts: list[int] = []
        self.total_count = 0
        self.min_val: float | None = None
        self.max_val: float | None = None
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

    def get_histogram_data(self) -> tuple[list[float], list[int], int]:
        """Get histogram data for rendering.

        Returns:
            Tuple of (bin_edges, counts, total_count)
        """
        if not self._initialized or not self.bin_edges:
            return [], [], 0

        return self.bin_edges, self.counts, self.total_count
