"""High-performance categorical accumulator optimized for big data.

This module provides a production-ready, scalable implementation of the categorical accumulator
with comprehensive error handling, validation, and performance optimizations for large datasets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chunking import ChunkTracker
from .config import CategoricalConfig
from .protocols import AccumulatorKind, PicklableAccumulator
from .sketches import KMV, MisraGries, ReservoirSampler, SingletonCounter


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
    top_items: list[tuple[str, int]]
    approx: bool
    #: How far below the truth every count in `top_items` may sit: the total
    #: weight Misra-Gries decremented. Zero means the counts are exact, and
    #: `true_count(x) ∈ [reported(x), reported(x) + this]` otherwise. Published
    #: rather than kept private, because a count that can only undercount is a
    #: range, and the alternative is a confident wrong integer (#328).
    top_items_uncertainty: int = 0
    #: Whether `unique_est` is the exact distinct count rather than a KMV
    #: estimate. Separate from `approx`, which is true when *anything* on the
    #: column is approximate: once `approx` also covered lossy top-k counters,
    #: reading it for the Unique row marked an exact 599 as `≈ 599`, which is
    #: the same overclaim as #328 pointing the other way. Per statistic, not
    #: per column.
    unique_est_exact: bool = False
    # extras for alignment
    mem_bytes: int = 0
    avg_len: float | None = None
    len_p90: int | None = None
    #: `(length, count)` over the length reservoir, shortest first.
    #:
    #: The reservoir has been kept all along and spent on two numbers. The
    #: whole distribution was sitting in it, and on an identifier column the
    #: *shape* is the finding: `Ticket` clusters at 4-7 characters and 10-14
    #: with a tail to 18, which is two ticket formats in one column and is
    #: available no other way (#155, 5c.2).
    #:
    #: Binned here rather than in the renderer, and capped, so what crosses
    #: into the summary is a few dozen pairs instead of a 5,000-element sample.
    len_hist: list[tuple[int, int]] = field(default_factory=list)
    empty_zero: int = 0
    case_variants_est: int = 0
    trim_variants_est: int = 0
    dtype_str: str = "categorical"
    # v2 additions
    entropy: float = 0.0
    gini_impurity: float = 0.0
    most_common_ratio: float = 0.0
    diversity_ratio: float = 0.0
    #: `(start_row, end_row, missing_in_chunk)` per chunk (#193).
    #:
    #: Absent until now, which is why the Missing Values pane could not be
    #: gated on this card kind the way #154's 5b.7 gates it on numeric and
    #: datetime. Defaulting it to `None` rather than `[]` keeps "this column
    #: tracked nothing" distinguishable from "this column ran as one chunk".
    chunk_metadata: list[tuple[int, int, int]] | None = None
    #: Levels seen exactly once, and the exact level total they are out of.
    #:
    #: Both `None` together, never one without the other: they come from the
    #: same exact counting, and a singleton count is only readable against a
    #: total counted the same way. `unique_est` is KMV and carries ~2.2% of
    #: error, which is enough to make `119 of 147` arithmetic that does not
    #: quite work on the page.
    #:
    #: `None` means the column had more levels than the counter's capacity, so
    #: the answer is unknown rather than zero (#297).
    singleton_levels: int | None = None
    exact_levels: int | None = None


#: Distinct lengths above this are grouped into the widest bins the range
#: allows. Labels are short, so most columns never reach it -- `Name` is the
#: outlier at 82 distinct lengths, and 40 bars is already more than a reader
#: counts.
_MAX_LENGTH_BINS = 40


def _length_histogram(values: list[float]) -> list[tuple[int, int]]:
    """`(length, count)` pairs over the length reservoir, shortest first.

    One bar per distinct length while that stays readable, because a label
    length *is* an integer and binning it hides the thing worth seeing -- a
    column of 4-character and 7-character values is two formats, and a bin of
    4-7 is one blur.
    """
    if not values:
        return []

    counts: dict[int, int] = {}
    for value in values:
        length = int(value)
        counts[length] = counts.get(length, 0) + 1

    if len(counts) <= _MAX_LENGTH_BINS:
        return sorted(counts.items())

    low, high = min(counts), max(counts)
    width = max(1, math.ceil((high - low + 1) / _MAX_LENGTH_BINS))
    binned: dict[int, int] = {}
    for length, count in counts.items():
        bucket = low + ((length - low) // width) * width
        binned[bucket] = binned.get(bucket, 0) + count
    return sorted(binned.items())


def _string_lengths(values: Any) -> np.ndarray:
    """Length of every value, in order, without pandas' `.str` accessor.

    `Series.str.len()` looked like the obvious call and was the single largest
    memory problem in the library: a string column's peak RSS grew with the row
    count -- 39 MB at 500,000 rows, 339 MB at 8,400,000, on a column holding
    four distinct values. Nothing was retained (`sys.getallocatedblocks()` was
    flat, and the sketches stayed at four counters), so it is allocator churn
    inside the accessor rather than a leak, but RSS is RSS.

    Taking the length of each *distinct* value and gathering it back through
    the factorisation codes is flat in rows, and returns exactly the same
    lengths in exactly the same order -- so the reservoir sees an identical
    stream and no statistic moves.

    It is also 4x faster on this kernel for a low-cardinality column, because
    `len()` runs once per category instead of once per row. That is worth
    nothing end to end: a text-heavy 200,000 x 3 profile measures 1.00x either
    way, because this was never on the critical path. Recorded so the next
    reader does not go looking for a speedup that is not there.

    Args:
        values: A pandas Series of strings.

    Returns:
        An int64 array of lengths, aligned with `values`.
    """
    import pandas as pd

    codes, uniques = pd.factorize(values, sort=False)
    lengths = np.fromiter((len(u) for u in uniques), dtype=np.int64, count=len(uniques))
    if lengths.size == 0:
        return np.empty(0, dtype=np.int64)
    # -1 marks a null; the caller has already dropped them, but a gather with a
    # negative index would silently read from the end of the array.
    return lengths[np.maximum(codes, 0)]


class CategoricalAccumulator(PicklableAccumulator):
    """Production-grade categorical accumulator optimized for big data.

    This accumulator provides comprehensive analysis of categorical data with
    superior error handling, validation, and performance optimizations for
    processing massive datasets efficiently.
    """

    def __init__(
        self,
        name: str,
        config: CategoricalConfig | None = None,
        *,
        seed: int | None = None,
    ):
        """Initialize categorical accumulator.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
            seed: Seed for this column's length sampling. None draws from OS
                entropy. Sampling never touches the process-global RNG.
        """
        self.name = name
        self.config = config or CategoricalConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Core state
        self.count = 0
        self.missing = 0
        self._dtype_str = "categorical"
        self._bytes_seen = 0

        # Per-column chunk tracking (#193), so the Missing Values pane can be
        # gated on "more than one chunk" here the way it already is on numeric
        # and datetime columns.
        self._chunks = ChunkTracker(
            enabled=getattr(self.config, "enable_chunk_metadata", True),
            max_chunks=getattr(self.config, "max_chunks", 1000),
        )

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
        # #297. Exact level counts while the column stays inside the sketch
        # capacity, so a many-level column can say how many of its levels occur
        # exactly once. Sized off the same knob as the distinct sketch: past
        # that many levels the card's sentence is "every value is different"
        # and the singleton count has nothing to add.
        self._levels = SingletonCounter(self.config.uniques_sketch_size)

        # String length tracking with memory-efficient sampling
        self._len_sum = 0
        self._len_n = 0
        self._len_sample = (
            ReservoirSampler(self.config.length_sample_size, rng=self._rng)
            if self.config.enable_length_stats
            else None
        )

        # Special value tracking
        self._empty_zero = 0

    @property
    def kind(self) -> AccumulatorKind:
        """Which column kind this accumulator handles.

        Read instead of `isinstance`: a native accumulator will not be an
        instance of this class, and the isinstance chain's order was
        load-bearing without saying so anywhere.
        """
        return "categorical"

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
    def avg_len(self) -> float | None:
        """Get average string length for compatibility."""
        if self._len_n > 0:
            return self._len_sum / self._len_n
        return None

    def update(self, arr: Sequence[Any]) -> None:
        """Update accumulator with new values, recording them against the chunk.

        The chunk bookkeeping is done here, by difference, rather than inside
        `_update_values`: that method has several early returns and two
        independent paths (vectorised and a per-value fallback), and counting
        in each of them is how they would drift apart. Totals before and after
        cannot.
        """
        before_rows = self.count + self.missing
        before_missing = self.missing
        try:
            self._update_values(arr)
        finally:
            self._chunks.note(
                rows=(self.count + self.missing) - before_rows,
                missing=self.missing - before_missing,
            )

    def mark_chunk_boundary(self) -> None:
        """Tell the accumulator a chunk ended (#193).

        Duck-typed by `compute/orchestration/engine.py`, which has always
        called this on every accumulator that has it -- only the numeric one
        did, so categorical columns reached the report with no chunk metadata
        and their Missing Values pane could not be gated like the others'.
        """
        self._chunks.mark_boundary()

    def _update_values(self, arr: Sequence[Any]) -> None:
        """Update accumulator with new values using optimized batch processing.

        Args:
            arr: Sequence of values to process
        """
        if len(arr) == 0:
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

            # Update sketches — one add per unique value (KMV only needs one),
            # weighted add for MisraGries frequency tracking
            for value, count in value_counts.items():
                # KMV only tracks distinct values — one add is sufficient
                self._uniques.add(value)

                # MisraGries supports weighted add for frequency counting
                self._topk.add(value, w=int(count))

                self._levels.add(value, w=int(count))

                # Variant tracking — one add per unique variant is sufficient for KMV
                if self.config.enable_case_variants and self._uniques_lower:
                    self._uniques_lower.add(value.lower())

                if self.config.enable_trim_variants and self._uniques_strip:
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
                lengths = _string_lengths(str_values)
                self._len_sum += int(lengths.sum())
                self._len_n += int(lengths.size)

                # Batch add lengths to reservoir sampler
                self._len_sample.add_many(lengths.astype(float))
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
        self._levels.add(value)

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

    def finalize(
        self, chunk_metadata: list[tuple[int, int, int]] | None = None
    ) -> CategoricalSummary:
        """Finalize accumulator and return comprehensive summary statistics.

        Args:
            chunk_metadata: Fallback `(start_row, end_row, missing)` triples,
                used only when this column tracked none of its own -- the same
                shape the numeric and datetime accumulators accept.

        Returns:
            CategoricalSummary containing all computed statistics
        """
        per_column_chunks = self._chunks.metadata()
        # Get top items with optimized access
        top_items = self._topk.items()

        # Calculate string length statistics
        avg_len = self._len_sum / max(1, self._len_n) if self._len_n > 0 else None
        len_values = self._len_sample.values() if self._len_sample else []
        len_p90 = self._calculate_percentile(len_values, 90)
        len_hist = _length_histogram(len_values)

        # Calculate variant estimates
        case_variants_est = self._uniques_lower.estimate() if self._uniques_lower else 0
        trim_variants_est = self._uniques_strip.estimate() if self._uniques_strip else 0

        # Calculate diversity metrics for data quality analysis
        entropy = self._calculate_entropy(top_items)
        gini_impurity = self._calculate_gini_impurity(top_items)
        most_common_ratio = self._calculate_most_common_ratio(top_items)
        diversity_ratio = self._calculate_diversity_ratio()

        # Approximate if the top-k counters have evicted anything, or if the
        # distinct count came from the KMV sketch rather than its exact counter.
        #
        # The first arm used to read `len(top_items) >= top_k_size`, which is
        # the dangerous case *backwards*: eviction deletes counters, so the list
        # shrinks below the budget exactly when the sketch is under most
        # pressure. A 1M-row column over 1,000 categories came out of here with
        # 9 items, a top count 30x low, and `approx=False` (#328). The sketch
        # now reports its own decrement mass, which is the only thing that
        # knows whether a count is the truth or a lower bound.
        topk_lossy = not self._topk.is_exact
        approx = topk_lossy or not self._uniques.is_exact

        # How far below the truth any published count may sit. Zero when the
        # counters never evicted, which is the case the report may call exact.
        top_items_uncertainty = self._topk.error_bound

        return CategoricalSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            # Clamped: see the note in numeric.py. A distinct count above the
            # row count is impossible and costs the reader's trust in the page.
            unique_est=min(self._uniques.estimate(), self.count),
            top_items=top_items,
            approx=approx,
            top_items_uncertainty=top_items_uncertainty,
            unique_est_exact=self._uniques.is_exact,
            mem_bytes=self._bytes_seen,
            avg_len=avg_len,
            len_p90=len_p90,
            len_hist=len_hist,
            empty_zero=self._empty_zero,
            case_variants_est=case_variants_est,
            trim_variants_est=trim_variants_est,
            dtype_str=self._dtype_str,
            entropy=entropy,
            gini_impurity=gini_impurity,
            most_common_ratio=most_common_ratio,
            diversity_ratio=diversity_ratio,
            chunk_metadata=per_column_chunks or chunk_metadata,
            singleton_levels=self._levels.singletons(),
            exact_levels=self._levels.levels(),
        )

    def _calculate_percentile(
        self, values: list[float], percentile: float
    ) -> int | None:
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

    def _calculate_entropy(self, top_items: list[tuple[str, int]]) -> float:
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

    def _calculate_gini_impurity(self, top_items: list[tuple[str, int]]) -> float:
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

    def _calculate_most_common_ratio(self, top_items: list[tuple[str, int]]) -> float:
        """What share of the column the most common value takes.

        Against `self.count`, the exact non-null row count, not against the sum
        of the counters. Under eviction every counter is short, so their sum is
        short *faster* than any one of them: dividing one by the other measured
        0.132 for a value whose true share was 0.0011, a 120x overstatement
        (#328). The same numerator-over-a-different-denominator slip as #327,
        and in the more damaging direction, since this one feeds the dominant
        category flag.

        Over the row count the answer is a lower bound, inheriting the
        counters' own guarantee: it can only understate, and `approx` says so.

        Args:
            top_items: List of (value, count) tuples

        Returns:
            Ratio of most common value
        """
        if not top_items or self.count <= 0:
            return 0.0

        max_count = max(count for _, count in top_items)
        return max_count / self.count

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
        """Merge another CategoricalAccumulator.

        Every sketch here composes; the previous implementation assumed none of
        them did. It replayed the other side's top-k counters one `add()` call
        per counted occurrence -- so merging a column with a value seen ten
        million times ran ten million Python calls -- and it seeded the distinct
        estimate from at most a hundred top-k *keys*, on the stated belief that
        "KMV sketches cannot be easily merged". They merge exactly, by keeping
        the k smallest hashes of the union.

        Args:
            other: Another CategoricalAccumulator. Left unchanged.
        """
        self.count += other.count
        self.missing += other.missing
        self._bytes_seen += other._bytes_seen
        self._len_sum += other._len_sum
        self._len_n += other._len_n
        self._empty_zero += other._empty_zero
        # A merged column's chunks are the two runs' chunks in order, which
        # needs the second side's boundaries offset by the first's row count
        # rather than restarting at zero halfway through (#193).
        self._chunks.merge(other._chunks)

        self._topk.merge(other._topk)
        self._uniques.merge(other._uniques)
        self._levels.merge(other._levels)
        # The case- and whitespace-folded sketches drive the "looks like a
        # variant of another value" flags. They were not merged at all, so a
        # merged column silently lost the evidence for them.
        if self._uniques_lower is not None and other._uniques_lower is not None:
            self._uniques_lower.merge(other._uniques_lower)
        if self._uniques_strip is not None and other._uniques_strip is not None:
            self._uniques_strip.merge(other._uniques_strip)
        if self._len_sample is not None and other._len_sample is not None:
            self._len_sample.merge(other._len_sample)

    def reset(self) -> None:
        """Reset accumulator to initial state efficiently."""
        self.count = 0
        self.missing = 0
        self._bytes_seen = 0
        self._len_sum = 0
        self._len_n = 0
        self._empty_zero = 0
        self._chunks.reset()

        # A reset accumulator must replay identically, so rewind the generator
        # rather than continuing its stream.
        self._rng = np.random.default_rng(self._seed)

        # Reset data structures
        self._uniques = KMV(self.config.uniques_sketch_size)
        if self._uniques_lower:
            self._uniques_lower = KMV(self.config.uniques_sketch_size)
        if self._uniques_strip:
            self._uniques_strip = KMV(self.config.uniques_sketch_size)
        self._topk = MisraGries(self.config.top_k_size)
        self._levels = SingletonCounter(self.config.uniques_sketch_size)
        if self._len_sample:
            self._len_sample = ReservoirSampler(
                self.config.length_sample_size, rng=self._rng
            )
