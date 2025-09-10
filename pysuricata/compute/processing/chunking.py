"""Adaptive chunking system for optimal data processing.

This module provides intelligent chunking strategies that adapt to data
characteristics, memory constraints, and processing requirements.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, Iterator, Optional, Union

from ..core.exceptions import ChunkingError
from ..core.types import ChunkMetadata, ProcessingResult

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


class ChunkingStrategy(Enum):
    """Strategy for chunking operations."""

    FIXED_SIZE = "fixed_size"  # Fixed chunk size
    ADAPTIVE = "adaptive"  # Adaptive based on data characteristics
    MEMORY_AWARE = "memory_aware"  # Memory-aware chunking
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Performance-optimized


class AdaptiveChunker:
    """Adaptive chunking with performance optimization.

    This class provides intelligent chunking strategies that adapt to
    data characteristics, memory constraints, and processing requirements.
    It supports both pandas and polars backends with optimized chunking.

    Attributes:
        strategy: Chunking strategy to use.
        logger: Logger for chunking operations.
        chunk_size_cache: Cache for optimal chunk sizes by data type.
        performance_metrics: Performance metrics for chunking operations.
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
        logger: Optional[logging.Logger] = None,
        default_chunk_size: int = 10000,
        max_chunk_size: int = 1000000,
        min_chunk_size: int = 1000,
    ):
        """Initialize the adaptive chunker.

        Args:
            strategy: Chunking strategy to use.
            logger: Logger for chunking operations.
            default_chunk_size: Default chunk size.
            max_chunk_size: Maximum allowed chunk size.
            min_chunk_size: Minimum allowed chunk size.
        """
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.default_chunk_size = default_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        self.chunk_size_cache: Dict[type, int] = {}
        self.performance_metrics: Dict[str, Any] = {
            "total_chunks": 0,
            "total_rows": 0,
            "total_time": 0.0,
            "avg_chunk_time": 0.0,
        }

    def chunks_from_source(
        self,
        source: Any,
        chunk_size: int,
        force_chunk_in_memory: bool = False,
    ) -> ProcessingResult[Iterator[Any]]:
        """Generate chunks from a data source with error handling.

        Args:
            source: Input data source (DataFrame, iterable, etc.).
            chunk_size: Size of chunks to create.
            force_chunk_in_memory: Whether to force in-memory chunking.

        Returns:
            ProcessingResult containing the chunk iterator.
        """
        start_time = time.time()

        try:
            # Validate chunk size
            if chunk_size <= 0:
                raise ChunkingError(
                    f"Invalid chunk size: {chunk_size}. Must be positive.",
                    chunk_size=chunk_size,
                )

            # Determine optimal chunk size
            optimal_size = self._determine_optimal_chunk_size(source, chunk_size)

            # Generate chunks based on strategy
            if self.strategy == ChunkingStrategy.ADAPTIVE:
                chunks = self._adaptive_chunking(
                    source, optimal_size, force_chunk_in_memory
                )
            elif self.strategy == ChunkingStrategy.MEMORY_AWARE:
                chunks = self._memory_aware_chunking(
                    source, optimal_size, force_chunk_in_memory
                )
            elif self.strategy == ChunkingStrategy.PERFORMANCE_OPTIMIZED:
                chunks = self._performance_optimized_chunking(
                    source, optimal_size, force_chunk_in_memory
                )
            else:  # FIXED_SIZE
                chunks = self._fixed_size_chunking(
                    source, optimal_size, force_chunk_in_memory
                )

            duration = time.time() - start_time

            return ProcessingResult.success_result(
                data=chunks,
                metrics={
                    "chunk_size": optimal_size,
                    "force_chunk_in_memory": force_chunk_in_memory,
                    "strategy": self.strategy.value,
                    "duration": duration,
                },
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Chunking failed: %s", e)
            return ProcessingResult.error_result(
                f"Chunking failed: {str(e)}", duration=duration
            )

    def adaptive_chunk_size(self, data: Any) -> int:
        """Determine optimal chunk size based on data characteristics.

        Args:
            data: Input data to analyze.

        Returns:
            Optimal chunk size for the data.
        """
        data_type = type(data)

        # Check cache first
        if data_type in self.chunk_size_cache:
            return self.chunk_size_cache[data_type]

        # Determine optimal size based on data characteristics
        if isinstance(data, (pd.DataFrame, pl.DataFrame)):
            optimal_size = self._analyze_dataframe_characteristics(data)
        elif hasattr(data, "__len__"):
            # For other iterables, use size-based heuristics
            data_size = len(data)
            if data_size < 1000:
                optimal_size = data_size  # Process all at once
            elif data_size < 10000:
                optimal_size = max(1000, data_size // 4)
            else:
                optimal_size = self.default_chunk_size
        else:
            optimal_size = self.default_chunk_size

        # Apply constraints
        optimal_size = max(self.min_chunk_size, min(optimal_size, self.max_chunk_size))

        # Cache the result
        self.chunk_size_cache[data_type] = optimal_size

        return optimal_size

    def _determine_optimal_chunk_size(self, source: Any, requested_size: int) -> int:
        """Determine the optimal chunk size for the source.

        Args:
            source: Data source to analyze.
            requested_size: Requested chunk size.

        Returns:
            Optimal chunk size.
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return max(self.min_chunk_size, min(requested_size, self.max_chunk_size))

        # For adaptive strategies, analyze the data
        optimal_size = self.adaptive_chunk_size(source)

        # Blend with requested size
        if self.strategy == ChunkingStrategy.ADAPTIVE:
            # Use weighted average
            return int(0.7 * optimal_size + 0.3 * requested_size)
        else:
            return optimal_size

    def _analyze_dataframe_characteristics(self, df: Any) -> int:
        """Analyze DataFrame characteristics to determine optimal chunk size.

        Args:
            df: DataFrame to analyze.

        Returns:
            Optimal chunk size based on characteristics.
        """
        try:
            if isinstance(df, pd.DataFrame):
                n_rows, n_cols = df.shape
                memory_usage = df.memory_usage(deep=True).sum()
            elif isinstance(df, pl.DataFrame):
                n_rows = df.height
                n_cols = len(df.columns)
                memory_usage = df.estimated_size()
            else:
                return self.default_chunk_size

            # Calculate optimal chunk size based on characteristics
            if memory_usage < 10 * 1024 * 1024:  # < 10MB
                # Small data, process in larger chunks
                optimal_size = min(n_rows, 50000)
            elif memory_usage < 100 * 1024 * 1024:  # < 100MB
                # Medium data, moderate chunks
                optimal_size = min(n_rows, 25000)
            else:
                # Large data, smaller chunks
                optimal_size = min(n_rows, 10000)

            # Adjust based on column count
            if n_cols > 100:
                optimal_size = int(optimal_size * 0.5)  # Reduce for wide tables
            elif n_cols < 10:
                optimal_size = int(optimal_size * 1.5)  # Increase for narrow tables

            return max(self.min_chunk_size, min(optimal_size, self.max_chunk_size))

        except Exception:
            return self.default_chunk_size

    def _adaptive_chunking(
        self, source: Any, chunk_size: int, force_chunk: bool
    ) -> Iterator[Any]:
        """Perform adaptive chunking based on data characteristics.

        Args:
            source: Data source to chunk.
            chunk_size: Base chunk size.
            force_chunk: Whether to force chunking.

        Yields:
            Data chunks.
        """
        if isinstance(source, (pd.DataFrame, pl.DataFrame)) and force_chunk:
            yield from self._chunk_dataframe(source, chunk_size)
        elif hasattr(source, "__iter__") and not isinstance(source, (str, bytes)):
            yield from iter(source)
        else:
            yield source

    def _memory_aware_chunking(
        self, source: Any, chunk_size: int, force_chunk: bool
    ) -> Iterator[Any]:
        """Perform memory-aware chunking.

        Args:
            source: Data source to chunk.
            chunk_size: Base chunk size.
            force_chunk: Whether to force chunking.

        Yields:
            Data chunks.
        """
        # Similar to adaptive but with more conservative memory usage
        conservative_size = int(chunk_size * 0.7)
        yield from self._adaptive_chunking(source, conservative_size, force_chunk)

    def _performance_optimized_chunking(
        self, source: Any, chunk_size: int, force_chunk: bool
    ) -> Iterator[Any]:
        """Perform performance-optimized chunking.

        Args:
            source: Data source to chunk.
            chunk_size: Base chunk size.
            force_chunk: Whether to force chunking.

        Yields:
            Data chunks.
        """
        # Optimize for performance with larger chunks
        performance_size = int(chunk_size * 1.3)
        yield from self._adaptive_chunking(source, performance_size, force_chunk)

    def _fixed_size_chunking(
        self, source: Any, chunk_size: int, force_chunk: bool
    ) -> Iterator[Any]:
        """Perform fixed-size chunking.

        Args:
            source: Data source to chunk.
            chunk_size: Fixed chunk size.
            force_chunk: Whether to force chunking.

        Yields:
            Data chunks.
        """
        yield from self._adaptive_chunking(source, chunk_size, force_chunk)

    def _chunk_dataframe(self, df: Any, chunk_size: int) -> Iterator[Any]:
        """Chunk DataFrame with optimal strategy.

        Args:
            df: DataFrame to chunk.
            chunk_size: Size of chunks.

        Yields:
            DataFrame chunks.
        """
        if isinstance(df, pd.DataFrame):
            yield from self._chunk_pandas(df, chunk_size)
        elif isinstance(df, pl.DataFrame):
            yield from self._chunk_polars(df, chunk_size)
        else:
            yield df

    def _chunk_pandas(
        self, df: pd.DataFrame, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Optimized pandas chunking.

        Args:
            df: Pandas DataFrame to chunk.
            chunk_size: Size of chunks.

        Yields:
            Pandas DataFrame chunks.
        """
        if pd is None:
            yield df
            return

        n = len(df)
        for i in range(0, n, chunk_size):
            chunk_start = time.time()
            chunk = df.iloc[i : i + chunk_size]
            chunk_time = time.time() - chunk_start

            # Update performance metrics
            self._update_metrics(chunk, chunk_time)

            yield chunk

    def _chunk_polars(
        self, df: pl.DataFrame, chunk_size: int
    ) -> Iterator[pl.DataFrame]:
        """Optimized polars chunking.

        Args:
            df: Polars DataFrame to chunk.
            chunk_size: Size of chunks.

        Yields:
            Polars DataFrame chunks.
        """
        if pl is None:
            yield df
            return

        n = df.height
        for i in range(0, n, chunk_size):
            chunk_start = time.time()
            chunk = df.slice(i, chunk_size)
            chunk_time = time.time() - chunk_start

            # Update performance metrics
            self._update_metrics(chunk, chunk_time)

            yield chunk

    def _update_metrics(self, chunk: Any, chunk_time: float) -> None:
        """Update performance metrics.

        Args:
            chunk: Processed chunk.
            chunk_time: Time taken to process chunk.
        """
        self.performance_metrics["total_chunks"] += 1
        self.performance_metrics["total_time"] += chunk_time

        # Update row count
        if hasattr(chunk, "__len__"):
            self.performance_metrics["total_rows"] += len(chunk)
        elif hasattr(chunk, "height"):  # Polars
            self.performance_metrics["total_rows"] += chunk.height

        # Update average chunk time
        total_chunks = self.performance_metrics["total_chunks"]
        self.performance_metrics["avg_chunk_time"] = (
            self.performance_metrics["total_time"] / total_chunks
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for chunking operations.

        Returns:
            Dictionary with performance metrics.
        """
        return self.performance_metrics.copy()

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_chunks": 0,
            "total_rows": 0,
            "total_time": 0.0,
            "avg_chunk_time": 0.0,
        }

    def clear_cache(self) -> None:
        """Clear the chunk size cache."""
        self.chunk_size_cache.clear()
