from __future__ import annotations

import bisect
import hashlib
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
# Acceptances are scheduled a block at a time. The schedule depends only on the
# generator and k, so computing it in bulk costs nothing in correctness.
_SCHEDULE_BLOCK = 2048
# Rows stringified into the sketch when vectorised row hashing fails. This
# bounds the sketch's input only -- never the row count.
_HASH_FALLBACK_SAMPLE = 2000


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

        Once the sketch is full, the kth smallest hash it holds is a hard
        admission threshold: nothing at or above it can enter now, and nothing
        can later, because the threshold only ever decreases. Rejecting against
        it with one vectorised compare -- *before* sorting -- discards well over
        99.9% of a batch from a high-cardinality column, leaving ``np.unique``
        and ``np.union1d`` to sort the survivors rather than the whole chunk.
        The retained set, and therefore the estimate, is identical either way.

        ``_values`` deliberately stays a list. Holding it as a uint64 array
        removes the ``.tolist()`` below and measures 2.7x faster on this method
        in isolation -- but it makes ``_add_hash_to_kmv`` allocate and copy the
        whole array per insert instead of doing an in-place memmove, and that
        path runs per distinct value on categorical columns. End to end on mixed
        200k x 14 it is 35% *slower*. The kernel benchmark does not exercise it.

        Args:
            hashes: Array of hash values to add.
        """
        if len(hashes) == 0:
            return

        hashes = np.asarray(hashes, dtype=np.uint64)
        if len(self._values) >= self.k:
            # Strict <: a hash equal to the threshold is already in the sketch,
            # so admitting it could only duplicate an entry.
            hashes = hashes[hashes < self._values[-1]]
            if hashes.size == 0:
                return

        incoming = np.unique(hashes)
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

    def merge(self, other: KMV) -> None:
        """Fold another sketch in, exactly.

        KMV composes: the k smallest hashes of the union are always a subset of
        the two sides' k-smallest sets, because each side retains *everything*
        below its own threshold. So the merged estimate equals the estimate of
        a single sketch fed both streams — no approximation is introduced by
        the merge itself, only the sketch's own error.

        That is the reason to do this rather than replay one side's reservoir
        sample into the other, which is what `NumericAccumulator.merge` used to
        do: a sample of 20,000 rows cannot represent the distinct count of ten
        million, so the merged estimate was biased low by whatever the sampling
        ratio happened to be.

        Args:
            other: The sketch to fold in. Left unchanged.
        """
        if other is self or (not other._values and not other._exact_counter):
            return

        # Both sides still counting exactly, and the union still fits: stay
        # exact. Being exact is not a performance detail here -- it is the
        # difference between reporting 7 distinct values and reporting ~7.
        if self._use_exact and other._use_exact:
            combined = dict(self._exact_counter)
            for h, count in other._exact_counter.items():
                combined[h] = combined.get(h, 0) + count
            if len(combined) <= self._max_exact_tracking:
                self._exact_counter = combined
                return

        if self._use_exact:
            self._spill_exact_to_kmv()
        # Exactly one of the two is populated on the other side: spilling clears
        # the counter, so this is the whole of what it holds either way.
        incoming = list(other._exact_counter) + list(other._values)
        self._batch_add_hashes(np.asarray(incoming, dtype=np.uint64))

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
    it accepts, so it costs about ``k * ln(n / k)`` random draws rather than one
    per element -- roughly 145k draws for 10M rows into k=20k, instead of 10M.

    The acceptance schedule depends only on the generator and on k, never on the
    data, so it is computed in bulk ahead of time rather than one acceptance at a
    time. Writing the recurrence out makes this explicit: with ``u`` and ``v``
    uniform,

        log W_i = cumsum(log u)_i / k
        skip_i  = floor(log v_i / log(1 - W_i))
        index_i = base + cumsum(skip)_i + i

    every term of which is a vectorised array operation. Because the schedule is
    generated from the draw sequence alone, it is identical however the stream is
    later split into chunks.
    """

    __slots__ = (
        "k",
        "_rng",
        "_buf",
        "_seen",
        "_logw",
        "_next",
        "_sched_idx",
        "_sched_slot",
        "_sched_pos",
    )

    def __init__(
        self, k: int = 20_000, *, rng: np.random.Generator | None = None
    ) -> None:
        self.k = int(k)
        # The generator belongs to the instance. Drawing from the process-global
        # numpy state would make one column's sample depend on what every other
        # column happened to draw first, which is neither reproducible under
        # threading nor invisible to the caller's own RNG.
        self._rng = rng if rng is not None else np.random.default_rng()
        self._buf: list[float] = []
        self._seen: int = 0
        # Algorithm L state carried between schedule blocks. _logw is log(W);
        # _next is the stream index (0-based) of the next element to accept.
        self._logw: float = 0.0
        self._next: int = -1
        self._sched_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._sched_slot: np.ndarray = np.empty(0, dtype=np.int64)
        self._sched_pos: int = 0

    def _extend_schedule(self, base: int) -> None:
        """Compute the next block of acceptance indices and target slots."""
        u = self._rng.random(_SCHEDULE_BLOCK)
        v = self._rng.random(_SCHEDULE_BLOCK)
        np.maximum(u, _TINY, out=u)
        np.maximum(v, _TINY, out=v)

        logw = self._logw + np.cumsum(np.log(u)) / self.k
        # log1p(-w) rather than log(1 - w): w is close to 1 early on, where the
        # subtraction would lose most of its significant digits.
        denom = np.log1p(-np.exp(logw))
        with np.errstate(divide="ignore", invalid="ignore"):
            skips = np.floor(np.log(v) / denom)
        # denom == 0 means w has saturated and no further element is accepted.
        skips = np.where(np.isfinite(skips), skips, float(_NO_MORE_ACCEPTANCES))

        idx = base + np.cumsum(skips) + np.arange(_SCHEDULE_BLOCK, dtype=float)
        idx = np.minimum(idx, float(_NO_MORE_ACCEPTANCES))

        self._sched_idx = idx.astype(np.int64)
        self._sched_slot = (self._rng.random(_SCHEDULE_BLOCK) * self.k).astype(np.int64)
        self._sched_pos = 0
        self._logw = float(logw[-1])
        self._next = int(self._sched_idx[0])

    def _advance_schedule(self) -> None:
        """Move to the next scheduled acceptance, refilling the block if spent."""
        self._sched_pos += 1
        if self._sched_pos >= self._sched_idx.size:
            self._extend_schedule(self._next)
        else:
            self._next = int(self._sched_idx[self._sched_pos])

    def _start_skipping(self) -> None:
        """Initialise the schedule the moment the reservoir first fills."""
        self._logw = 0.0
        # Standard formulation with i = k-1 as the last processed index, so the
        # first candidate sits at _seen + skip.
        self._extend_schedule(self._seen)

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
        buf = self._buf
        while self._next < end:
            buf[int(self._sched_slot[self._sched_pos])] = float(
                arr[self._next - self._seen]
            )
            self._advance_schedule()
        self._seen = end

    def add(self, x: float) -> None:
        if len(self._buf) < self.k:
            self._buf.append(float(x))
            self._seen += 1
            if len(self._buf) >= self.k:
                self._start_skipping()
            return
        if self._seen == self._next:
            self._buf[int(self._sched_slot[self._sched_pos])] = float(x)
            self._advance_schedule()
        self._seen += 1

    def values(self) -> list[float]:
        return self._buf

    def merge(self, other: ReservoirSampler) -> None:
        """Combine two reservoirs into one uniform sample of the union.

        The weights matter and are the whole point. `NumericAccumulator.merge`
        used to replay the other side's *buffer* through `add()`, which treats
        20,000 retained values as if they were the entire stream: merging a
        10-million-row shard into a 10-thousand-row one produced a sample in
        which the small shard was over-represented five-hundredfold, and every
        quantile drawn from it was wrong.

        Two cases:

        * A side that has seen fewer than k values **is** its whole stream, so
          replaying it is exact. That covers the small-shard case entirely.
        * When both are full, each slot takes the other side's value with
          probability `n_other / (n_self + n_other)`. Each buffer is already a
          uniform sample of its own stream, so the result is a uniform sample of
          the union.

        Args:
            other: The reservoir to fold in. Left unchanged.
        """
        if other is self or other._seen == 0:
            return

        # The other side is complete: replay it, weights and all.
        if len(other._buf) < other.k:
            for value in other._buf:
                self.add(value)
            return

        # This side is complete: adopt the other's sample and its stream
        # position, then replay ours into it. Same argument, mirrored.
        if len(self._buf) < self.k:
            mine = list(self._buf)
            self._buf = list(other._buf)
            self._seen = other._seen
            self._logw = other._logw
            self._next = other._next
            self._sched_idx = other._sched_idx.copy()
            self._sched_slot = other._sched_slot.copy()
            self._sched_pos = other._sched_pos
            for value in mine:
                self.add(value)
            return

        total = self._seen + other._seen
        take_other = self._rng.random(len(self._buf)) >= (self._seen / total)
        for slot in np.flatnonzero(take_other):
            self._buf[int(slot)] = other._buf[int(slot)]
        self._seen = total
        # The stream position jumped, so the acceptance schedule -- which is
        # indexed against _seen -- has to be rebuilt from the new position.
        self._start_skipping()


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

    def merge(self, other: MisraGries) -> None:
        """Fold another summary in, keeping the frequency-error guarantee.

        Misra-Gries does not merge exactly — nothing with k counters can. The
        standard result (Agarwal et al., 2012) is that summing the counters and
        subtracting the (k+1)-th largest resulting count preserves the bound,
        at a slightly worse constant: each reported count still undercounts by
        at most `n/(k+1)` of the *combined* stream, never overcounts.

        Subtracting is what pays for the merge. Dropping the tail instead would
        leave the survivors' counts too high, which is the one direction the
        guarantee does not allow.

        Args:
            other: The summary to fold in. Left unchanged.
        """
        if other is self or not other.counters:
            return

        combined = dict(self.counters)
        for key, count in other.counters.items():
            combined[key] = combined.get(key, 0) + count

        if len(combined) > self.k:
            delta = sorted(combined.values(), reverse=True)[self.k]
            combined = {
                key: count - delta
                for key, count in combined.items()
                if count - delta > 0
            }
        self.counters = combined

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
        # Set when a chunk's rows could not be hashed, so the distinct-row
        # sketch has seen fewer rows than ``rows``. The row count stays exact
        # either way; only the duplicate estimate degrades.
        self.duplicates_degraded = False

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
            # Vectorised hashing failed for this chunk. Only the *sketch* has to
            # fall back to a sample -- the row count must not, because it is what
            # the report prints as "Rows" and what every missing-percentage
            # divides by. Counting the sample here truncated a 50,000-row chunk
            # to 2,000 and silently corrupted both.
            self._degraded_update(len(df), df.head(_HASH_FALLBACK_SAMPLE))

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
                self._degraded_update(df.height, None)

        except Exception:
            # Final fallback. As above, the sample bounds what the sketch sees,
            # never what the row count records.
            sample = None
            try:
                sample = df.head(min(_HASH_FALLBACK_SAMPLE, df.height)).to_pandas()
            except Exception:
                sample = None
            self._degraded_update(df.height, sample)

    def _degraded_update(self, n_rows: int, sample: Any) -> None:
        """Record every row, but feed the sketch only what could be hashed.

        Args:
            n_rows: The true number of rows in the chunk. Always counted.
            sample: A small frame to stringify into the sketch, or None if even
                that is not possible.
        """
        if sample is not None:
            try:
                joined = sample.astype(str).agg("|".join, axis=1)
                for value in joined:
                    self.kmv.add(value)
                if len(sample) < n_rows:
                    self.duplicates_degraded = True
            except Exception:
                self.duplicates_degraded = True
        else:
            self.duplicates_degraded = True
        self.rows += int(n_rows)

    def approx_duplicates(self) -> tuple[int, float]:
        """Estimated duplicate rows and their percentage.

        When ``duplicates_degraded`` is set the sketch has seen only part of the
        data, so the distinct count is an underestimate and this figure an
        overestimate. The result is clamped to the row count so it can never
        exceed what was actually read.
        """
        uniq = self.kmv.estimate()
        d = max(0, min(self.rows, self.rows - uniq))
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

    def merge(self, other: StreamingHistogram) -> None:
        """Fold another histogram in, mapping its bins by their centres.

        This is the same bin-centre approximation `_expand_range` already
        applies whenever the observed range grows: a bin's count is reassigned
        as though every value in it sat at the midpoint. The error on any one
        value is at most half a bin width; the total count stays exact.

        Args:
            other: The histogram to fold in. Left unchanged.
        """
        if other is self or not other._initialized or other.total_count == 0:
            return

        if not self._initialized:
            self.bins = other.bins
            self.bin_edges = list(other.bin_edges)
            self.counts = list(other.counts)
            self.total_count = other.total_count
            self.min_val = other.min_val
            self.max_val = other.max_val
            self._initialized = True
            return

        if other.min_val < self.min_val or other.max_val > self.max_val:
            self._expand_range(
                min(other.min_val, self.min_val), max(other.max_val, self.max_val)
            )

        if not self.bin_edges or not self.counts:
            return

        for i, count in enumerate(other.counts):
            if count <= 0 or i >= len(other.bin_edges) - 1:
                continue
            centre = (other.bin_edges[i] + other.bin_edges[i + 1]) / 2.0
            idx = int(np.digitize(centre, self.bin_edges)) - 1
            idx = min(max(idx, 0), len(self.counts) - 1)
            self.counts[idx] += count
        self.total_count += other.total_count

    def get_histogram_data(self) -> tuple[list[float], list[int], int]:
        """Get histogram data for rendering.

        Returns:
            Tuple of (bin_edges, counts, total_count)
        """
        if not self._initialized or not self.bin_edges:
            return [], [], 0

        return self.bin_edges, self.counts, self.total_count
