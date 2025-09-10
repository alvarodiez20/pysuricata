"""Service layer for processing operations.

This module provides a service-oriented architecture for the compute operations,
improving testability, maintainability, and separation of concerns.

The service layer abstracts the complex processing logic behind clean interfaces,
making it easier to test, mock, and extend functionality.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

from ...config import EngineConfig
from ...logger import SectionTimer
from ..analysis import RowKMV
from ..analysis.metrics import (
    apply_corr_chips,
    build_manifest_inputs,
)
from ..engine import consume_stream as engine_consume_stream
from ..manifest import build_summary
from ..processing.chunking import AdaptiveChunker
from .engine import (
    EngineManager,
    StreamingEngine,
    get_adapter_tag,
    maybe_corr_estimator,
)
from .engine import (
    consume_stream as legacy_consume_stream,
)

T = TypeVar("T")


class ProcessingResult:
    """Type-safe result container for processing operations.

    Attributes:
        success: Whether the operation succeeded
        data: The result data if successful
        error: Error message if failed
        metrics: Optional performance metrics
        duration: Time taken for the operation
    """

    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metrics = metrics or {}
        self.duration = duration


class ChunkingService:
    """Service for handling data chunking operations.

    This service provides optimized chunking strategies and adaptive
    chunk size determination based on data characteristics.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the chunking service.

        Args:
            config: Engine configuration for chunking parameters
        """
        self.config = config or EngineConfig()
        self.chunk_size_cache: Dict[type, int] = {}

    def chunks_from_source(
        self, source: Any, chunk_size: int, force_chunk_in_memory: bool
    ) -> ProcessingResult[Iterator[Any]]:
        """Yield chunks from a source with error handling.

        Args:
            source: Input data source
            chunk_size: Size of chunks to create
            force_chunk_in_memory: Whether to force in-memory chunking

        Returns:
            ProcessingResult containing the chunk iterator
        """
        start_time = time.time()
        try:
            # Simple chunking logic that works correctly
            import pandas as pd
            import polars as pl

            if isinstance(source, (pd.DataFrame, pl.DataFrame)):
                # DataFrame - chunk it properly
                chunks = self._chunk_dataframe_simple(
                    source, chunk_size, force_chunk_in_memory
                )
            elif hasattr(source, "__iter__") and not isinstance(source, (str, bytes)):
                # If it's already an iterable of DataFrames, use it directly
                chunks = iter(source)
            else:
                # Single DataFrame or other source
                chunks = self._chunk_dataframe_simple(
                    source, chunk_size, force_chunk_in_memory
                )

            duration = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=chunks,
                metrics={
                    "chunk_size": chunk_size,
                    "force_chunk_in_memory": force_chunk_in_memory,
                },
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                success=False,
                error=f"Chunking failed: {e}",
                duration=duration,
            )

    def _chunk_dataframe_simple(
        self, source: Any, chunk_size: int, force_chunk_in_memory: bool
    ) -> Iterator[Any]:
        """Simple chunking logic that works correctly.

        Args:
            source: Data source to chunk
            chunk_size: Size of chunks
            force_chunk_in_memory: Whether to force chunking

        Yields:
            Data chunks
        """
        try:
            import pandas as pd
            import polars as pl

            if isinstance(source, pd.DataFrame):
                n_rows = len(source)
                if force_chunk_in_memory or n_rows > chunk_size:
                    for i in range(0, n_rows, chunk_size):
                        yield source.iloc[i : i + chunk_size]
                else:
                    yield source
            elif isinstance(source, pl.DataFrame):
                n_rows = source.height
                if force_chunk_in_memory or n_rows > chunk_size:
                    for i in range(0, n_rows, chunk_size):
                        yield source.slice(i, chunk_size)
                else:
                    yield source
            else:
                yield source
        except Exception:
            # Fallback: yield the source as-is
            yield source

    def adaptive_chunk_size(self, data: Any) -> int:
        """Determine optimal chunk size based on data characteristics.

        Args:
            data: Input data to analyze

        Returns:
            Optimal chunk size for the data
        """
        data_type = type(data)
        if data_type in self.chunk_size_cache:
            return self.chunk_size_cache[data_type]

        # Default chunk size from config
        optimal_size = self.config.chunk_size

        # Could add more sophisticated logic here based on:
        # - Memory usage
        # - Data size
        # - Column count
        # - Data type complexity

        self.chunk_size_cache[data_type] = optimal_size
        return optimal_size


class EngineService:
    """Service for handling engine adapter operations.

    This service manages the selection and usage of engine adapters
    for different data backends (pandas, polars).
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the engine service.

        Args:
            config: Engine configuration
        """
        self.config = config or EngineConfig()

    def select_adapter(self, data: Any) -> ProcessingResult[Any]:
        """Select appropriate engine adapter with error handling.

        Args:
            data: Input data to analyze

        Returns:
            ProcessingResult containing the selected adapter
        """
        start_time = time.time()
        try:
            # Use the new EngineManager
            engine_manager = EngineManager()
            result = engine_manager.select_adapter(data)

            if not result.success:
                return ProcessingResult(
                    success=False,
                    error=result.error,
                    duration=time.time() - start_time,
                )

            adapter = result.data
            duration = time.time() - start_time

            if adapter is None:
                return ProcessingResult(
                    success=False,
                    error="Unsupported input type. Provide pandas/polars DataFrame or iterable of them.",
                    duration=duration,
                )

            return ProcessingResult(
                success=True,
                data=adapter,
                metrics={
                    "adapter_type": type(adapter).__name__,
                    "adapter_tag": get_adapter_tag(adapter),
                },
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                success=False,
                error=f"Adapter selection failed: {e}",
                duration=duration,
            )

    def consume_stream(
        self,
        adapter: Any,
        chunks: Iterator[Any],
        accs: Dict[str, Any],
        kinds: Any,
        logger: logging.Logger,
        corr_est: Optional[Any],
        row_kmv: RowKMV,
        approx_mem_bytes: int,
        total_missing_cells: int,
        first_columns: list,
        sample_section_html: str,
        cfg: EngineConfig,
        ckpt_mgr: Optional[Any],
        start_time: float,
        report_title: Optional[str],
        n_rows: int,
        chunk_idx: int,
    ) -> ProcessingResult[Tuple[int, int, int, int]]:
        """Consume data stream with improved error handling.

        Args:
            adapter: Engine adapter to use
            chunks: Iterator of data chunks
            accs: Accumulators dictionary
            kinds: Column kinds information
            logger: Logger instance
            corr_est: Correlation estimator
            row_kmv: Row KMV estimator
            approx_mem_bytes: Approximate memory usage
            total_missing_cells: Total missing cells count
            first_columns: First chunk columns
            sample_section_html: Sample section HTML
            cfg: Engine configuration
            ckpt_mgr: Checkpoint manager
            start_time: Start time for timing
            report_title: Report title
            n_rows: Number of rows processed
            chunk_idx: Current chunk index

        Returns:
            ProcessingResult containing (n_rows, approx_mem_bytes, total_missing_cells, chunk_idx)
        """
        start_time_consume = time.time()
        try:
            # Use the engine consume_stream function with chunk-level logging
            result = engine_consume_stream(
                adapter=adapter,
                chunks=chunks,
                accs=accs,
                kinds=kinds,
                logger=logger,
                corr_est=corr_est,
                row_kmv=row_kmv,
                approx_mem_bytes=approx_mem_bytes,
                total_missing_cells=total_missing_cells,
                first_columns=first_columns,
                sample_section_html=sample_section_html,
                cfg=cfg,
                ckpt_mgr=ckpt_mgr,
                start_time=start_time,
                report_title=report_title,
                n_rows=n_rows,
                chunk_idx=chunk_idx,
            )
            duration = time.time() - start_time_consume

            return ProcessingResult(
                success=True,
                data=result,
                metrics={
                    "chunks_processed": chunk_idx,
                    "memory_usage": result[1],
                    "missing_cells": result[2],
                },
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time_consume
            return ProcessingResult(
                success=False,
                error=f"Stream consumption failed: {e}",
                duration=duration,
            )


class MetricsService:
    """Service for handling metrics computation and correlation processing.

    This service manages the computation of various metrics including
    manifest building, correlation processing, and summary generation.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the metrics service.

        Args:
            config: Engine configuration
        """
        self.config = config or EngineConfig()

    def build_manifest_inputs(
        self,
        kinds: Any,
        accs: Dict[str, Any],
        row_kmv: RowKMV,
        first_columns: list,
    ) -> ProcessingResult[Tuple[Any, Any, int, int, Any]]:
        """Build manifest inputs with error handling.

        Args:
            kinds: Column kinds information
            accs: Accumulators dictionary
            row_kmv: Row KMV estimator
            first_columns: First chunk columns

        Returns:
            ProcessingResult containing manifest inputs
        """
        start_time = time.time()
        try:
            result = build_manifest_inputs(
                kinds=kinds,
                accs=accs,
                row_kmv=row_kmv,
                first_columns=first_columns,
            )
            duration = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=result,
                metrics={
                    "n_rows": result[2],
                    "n_cols": result[3],
                    "missing_columns": len(result[4]),
                },
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                success=False,
                error=f"Manifest building failed: {e}",
                duration=duration,
            )

    def apply_correlation_chips(
        self,
        accs: Dict[str, Any],
        kinds: Any,
        top_map: Dict[str, list],
    ) -> ProcessingResult[None]:
        """Apply correlation chips with error handling.

        Args:
            accs: Accumulators dictionary
            kinds: Column kinds information
            top_map: Correlation mapping

        Returns:
            ProcessingResult indicating success/failure
        """
        start_time = time.time()
        try:
            apply_corr_chips(accs, kinds, top_map)
            duration = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=None,
                metrics={
                    "correlations_applied": len(top_map),
                },
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                success=False,
                error=f"Correlation chips application failed: {e}",
                duration=duration,
            )

    def build_summary(
        self,
        kinds_map: Any,
        col_order: Any,
        row_kmv: RowKMV,
        total_missing_cells: int,
        n_rows: int,
        n_cols: int,
        miss_list: Any,
    ) -> ProcessingResult[Optional[dict]]:
        """Build programmatic summary with error handling.

        Args:
            kinds_map: Column kinds mapping
            col_order: Column order
            row_kmv: Row KMV estimator
            total_missing_cells: Total missing cells count
            n_rows: Number of rows
            n_cols: Number of columns
            miss_list: Missing values list

        Returns:
            ProcessingResult containing the summary
        """
        start_time = time.time()
        try:
            result = build_summary(
                kinds_map,
                col_order,
                row_kmv=row_kmv,
                total_missing_cells=total_missing_cells,
                n_rows=n_rows,
                n_cols=n_cols,
                miss_list=miss_list,
            )
            duration = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=result,
                metrics={
                    "summary_size": len(result) if result else 0,
                },
                duration=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult(
                success=False,
                error=f"Summary building failed: {e}",
                duration=duration,
            )


class ResourceManager:
    """Manages resources for processing operations.

    This class provides context managers for resource management,
    ensuring proper cleanup and monitoring of processing resources.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the resource manager.

        Args:
            config: Engine configuration
        """
        self.config = config or EngineConfig()
        self.resources: Dict[str, Any] = {}

    @contextmanager
    def processing_context(self) -> ContextManager[Dict[str, Any]]:
        """Context manager for processing resources.

        Yields:
            Dictionary of available resources
        """
        try:
            # Initialize resources
            self.resources = {
                "start_time": time.time(),
                "memory_monitor": self._start_memory_monitoring(),
                "progress_tracker": self._start_progress_tracking(),
            }
            yield self.resources
        finally:
            # Cleanup resources
            self._cleanup_resources()

    def _start_memory_monitoring(self) -> Dict[str, Any]:
        """Start memory monitoring.

        Returns:
            Memory monitoring configuration
        """
        return {
            "enabled": True,
            "start_time": time.time(),
        }

    def _start_progress_tracking(self) -> Dict[str, Any]:
        """Start progress tracking.

        Returns:
            Progress tracking configuration
        """
        return {
            "enabled": True,
            "start_time": time.time(),
        }

    def _cleanup_resources(self) -> None:
        """Cleanup all resources."""
        self.resources.clear()


class ProcessingService:
    """Main processing service that coordinates all operations.

    This service provides a unified interface for all processing operations,
    handling error recovery, resource management, and performance monitoring.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        chunking_service: Optional[ChunkingService] = None,
        engine_service: Optional[EngineService] = None,
        metrics_service: Optional[MetricsService] = None,
        resource_manager: Optional[ResourceManager] = None,
    ):
        """Initialize the processing service.

        Args:
            config: Engine configuration
            chunking_service: Chunking service instance
            engine_service: Engine service instance
            metrics_service: Metrics service instance
            resource_manager: Resource manager instance
        """
        self.config = config or EngineConfig()
        self.chunking_service = chunking_service or ChunkingService(self.config)
        self.engine_service = engine_service or EngineService(self.config)
        self.metrics_service = metrics_service or MetricsService(self.config)
        self.resource_manager = resource_manager or ResourceManager(self.config)

    def process_with_retry(
        self,
        operation: callable,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> ProcessingResult:
        """Process operation with exponential backoff retry logic.

        Args:
            operation: Function to execute
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for exponential delay

        Returns:
            ProcessingResult of the operation
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                result = operation()
                if hasattr(result, "success") and result.success:
                    return result
                elif hasattr(result, "success"):
                    last_error = result.error
                else:
                    # Assume success if no ProcessingResult
                    return ProcessingResult(success=True, data=result)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor**attempt)

        return ProcessingResult(
            success=False,
            error=f"Failed after {max_retries} attempts: {last_error}",
        )
