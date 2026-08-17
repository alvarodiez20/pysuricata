"""High-performance boolean accumulator with vectorized operations.

This module provides a production-ready, efficient implementation of the boolean accumulator
optimized for big data processing with vectorized operations and comprehensive error handling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chunking import ChunkTracker
from .config import BooleanConfig
from .protocols import AccumulatorKind, PicklableAccumulator


@dataclass
class BooleanSummary:
    """Summary statistics for boolean data.

    This dataclass contains all computed statistics for a boolean column,
    including counts, ratios, and memory usage information.
    """

    name: str
    count: int
    missing: int
    true_n: int
    false_n: int
    mem_bytes: int = 0
    dtype_str: str = "boolean"
    true_ratio: float = 0.0
    false_ratio: float = 0.0
    entropy: float = 0.0
    #: `(start_row, end_row, missing_in_chunk)` per chunk (#193).
    #:
    #: Absent until now, which is why the Missing Values pane could not be
    #: gated on this card kind the way #154's 5b.7 gates it on numeric and
    #: datetime. `None` means the column tracked nothing, which is not the same
    #: as a column that ran as a single chunk.
    chunk_metadata: list[tuple[int, int, int]] | None = None


class BooleanAccumulator(PicklableAccumulator):
    """Production-grade boolean accumulator optimized for big data.

    This accumulator uses vectorized numpy operations for maximum performance
    on large datasets while maintaining reliability and comprehensive error handling.
    """

    def __init__(self, name: str, config: BooleanConfig | None = None):
        """Initialize boolean accumulator.

        Args:
            name: Column name
            config: Configuration for accumulator behavior
        """
        self.name = name
        self.config = config or BooleanConfig()

        # Core state
        self.count = 0
        self.missing = 0
        self.true_n = 0
        self.false_n = 0
        self._dtype_str = "boolean"
        self._mem_bytes = 0

        # Per-column chunk tracking (#193).
        self._chunks = ChunkTracker(
            enabled=getattr(self.config, "enable_chunk_metadata", True),
            max_chunks=getattr(self.config, "max_chunks", 1000),
        )

    @property
    def kind(self) -> AccumulatorKind:
        """Which column kind this accumulator handles.

        Read instead of `isinstance`: a native accumulator will not be an
        instance of this class, and the isinstance chain's order was
        load-bearing without saying so anywhere.
        """
        return "boolean"

    def set_dtype(self, dtype_str: str) -> None:
        """Set the data type string.

        Args:
            dtype_str: String representation of the data type
        """
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "boolean"

    @property
    def unique_est(self) -> int:
        """Get unique count estimate for compatibility."""
        return 2  # Boolean can only have 2 unique values

    def update(self, arr: Sequence[Any]) -> None:
        """Update accumulator with new values, recording them against the chunk.

        Counted by difference around `_update_values` rather than inside it:
        that method has early returns and a per-value fallback beside the
        vectorised path, and counting in each is how the two drift apart.
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

        Duck-typed by `compute/orchestration/engine.py`, which calls this on
        every accumulator that has it.
        """
        self._chunks.mark_boundary()

    def _update_values(self, arr: Sequence[Any]) -> None:
        """Update accumulator with new values using vectorized operations.

        Args:
            arr: Sequence of values to process
        """
        if len(arr) == 0:
            return

        # Convert to numpy array for efficient processing
        try:
            values = np.asarray(arr, dtype=object)
        except Exception:
            # Fallback to list processing for problematic arrays
            self._update_fallback(arr)
            return

        # Vectorized processing for better performance
        self._process_values_vectorized(values)

    def _process_values_vectorized(self, values: np.ndarray) -> None:
        """Process values using vectorized numpy operations.

        Args:
            values: Numpy array of values
        """
        # Create masks for different value types
        none_mask = np.array([v is None for v in values])
        nan_mask = np.array([isinstance(v, float) and math.isnan(v) for v in values])
        missing_mask = none_mask | nan_mask

        # Count missing values
        missing_count = int(np.sum(missing_mask))
        self.missing += missing_count

        # Process non-missing values
        valid_values = values[~missing_mask]
        if len(valid_values) == 0:
            return

        # Convert to boolean using numpy's bool conversion
        try:
            bool_values = valid_values.astype(bool)
        except Exception:
            # Fallback for problematic values
            bool_values = np.array([bool(v) for v in valid_values])

        # Count true and false values
        true_count = int(np.sum(bool_values))
        false_count = len(bool_values) - true_count

        self.true_n += true_count
        self.false_n += false_count
        self.count += len(valid_values)

    def _update_fallback(self, arr: Sequence[Any]) -> None:
        """Fallback processing for problematic arrays.

        Args:
            arr: Sequence of values to process
        """
        for value in arr:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                self.missing += 1
                continue

            try:
                if bool(value):
                    self.true_n += 1
                else:
                    self.false_n += 1
                self.count += 1
            except Exception:
                # Handle any conversion errors gracefully
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
    ) -> BooleanSummary:
        """Finalize accumulator and return summary statistics.

        Args:
            chunk_metadata: Fallback triples, used only when this column
                tracked none of its own.

        Returns:
            BooleanSummary containing all computed statistics
        """
        per_column_chunks = self._chunks.metadata()
        # Calculate ratios
        total_valid = self.true_n + self.false_n
        true_ratio = self.true_n / max(1, total_valid)
        false_ratio = self.false_n / max(1, total_valid)

        # Calculate entropy (information theory measure)
        entropy = self._calculate_entropy(true_ratio, false_ratio)

        # Estimate memory usage if not tracked
        if self._mem_bytes == 0 and self.config.enable_memory_tracking:
            # Rough estimate: ~1 byte per boolean value
            self._mem_bytes = self.count + self.missing

        return BooleanSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            true_n=self.true_n,
            false_n=self.false_n,
            mem_bytes=self._mem_bytes,
            dtype_str=self._dtype_str,
            true_ratio=true_ratio,
            false_ratio=false_ratio,
            entropy=entropy,
            chunk_metadata=per_column_chunks or chunk_metadata,
        )

    def _calculate_entropy(self, true_ratio: float, false_ratio: float) -> float:
        """Calculate Shannon entropy of the boolean distribution.

        Args:
            true_ratio: Ratio of true values
            false_ratio: Ratio of false values

        Returns:
            Entropy value in bits
        """
        if not self.config.enable_ratio_calculation:
            return 0.0

        entropy = 0.0

        if true_ratio > 0:
            entropy -= true_ratio * math.log2(true_ratio)

        if false_ratio > 0:
            entropy -= false_ratio * math.log2(false_ratio)

        return entropy

    def get_distribution_info(self) -> dict[str, Any]:
        """Get detailed distribution information.

        Returns:
            Dictionary containing distribution statistics
        """
        total_valid = self.true_n + self.false_n

        if total_valid == 0:
            return {
                "total_values": self.count + self.missing,
                "valid_values": 0,
                "missing_values": self.missing,
                "true_count": 0,
                "false_count": 0,
                "true_ratio": 0.0,
                "false_ratio": 0.0,
                "entropy": 0.0,
                "is_balanced": False,
                "is_uniform": False,
            }

        true_ratio = self.true_n / total_valid
        false_ratio = self.false_n / total_valid
        entropy = self._calculate_entropy(true_ratio, false_ratio)

        # Check if distribution is balanced (close to 50/50)
        is_balanced = abs(true_ratio - 0.5) < 0.1

        # Check if distribution is uniform (entropy close to 1.0)
        is_uniform = abs(entropy - 1.0) < 0.1

        return {
            "total_values": self.count + self.missing,
            "valid_values": total_valid,
            "missing_values": self.missing,
            "true_count": self.true_n,
            "false_count": self.false_n,
            "true_ratio": true_ratio,
            "false_ratio": false_ratio,
            "entropy": entropy,
            "is_balanced": is_balanced,
            "is_uniform": is_uniform,
        }

    def merge(self, other: BooleanAccumulator) -> None:
        """Merge another BooleanAccumulator.

        Args:
            other: Another BooleanAccumulator to merge
        """
        self.count += other.count
        self.missing += other.missing
        self.true_n += other.true_n
        self.false_n += other.false_n
        self._mem_bytes += other._mem_bytes
        # Chunks concatenate, with the second side's boundaries offset (#193).
        self._chunks.merge(other._chunks)

    def reset(self) -> None:
        """Reset accumulator to initial state."""
        self.count = 0
        self.missing = 0
        self.true_n = 0
        self.false_n = 0
        self._mem_bytes = 0
        self._chunks.reset()
