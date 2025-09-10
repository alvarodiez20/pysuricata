"""Streaming engine for data processing orchestration.

This module provides the main streaming engine that orchestrates data processing
operations, including adapter selection, chunking, and streaming coordination.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from ..adapters import PandasAdapter, PolarsAdapter
from ..core.exceptions import ComputeError
from ..core.protocols import DataAdapter
from ..core.types import ColumnKinds, ProcessingResult
from ..processing.chunking import AdaptiveChunker, ChunkingStrategy

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


class EngineManager:
    """Manages engine adapters and selection.

    This class manages the available engine adapters and provides
    intelligent selection based on data characteristics.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the engine manager.

        Args:
            logger: Logger for engine operations.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._adapters: Dict[str, DataAdapter] = {}
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Register default adapters."""
        try:
            if pd is not None:
                self._adapters["pandas"] = PandasAdapter()
        except ImportError:
            self.logger.warning("pandas not available, skipping pandas adapter")

        try:
            if pl is not None:
                self._adapters["polars"] = PolarsAdapter()
        except ImportError:
            self.logger.warning("polars not available, skipping polars adapter")

    def select_adapter(self, data: Any) -> ProcessingResult[DataAdapter]:
        """Select appropriate adapter for data.

        Args:
            data: Data to analyze for adapter selection.

        Returns:
            ProcessingResult containing the selected adapter.
        """
        start_time = time.time()

        try:
            # Check for pandas DataFrame
            if isinstance(data, pd.DataFrame):
                if "pandas" in self._adapters:
                    adapter = self._adapters["pandas"]
                    duration = time.time() - start_time
                    return ProcessingResult.success_result(
                        data=adapter,
                        metrics={
                            "adapter_type": "pandas",
                            "selection_reason": "pandas_dataframe",
                        },
                        duration=duration,
                    )

            # Check for polars DataFrame
            if isinstance(data, pl.DataFrame):
                if "polars" in self._adapters:
                    adapter = self._adapters["polars"]
                    duration = time.time() - start_time
                    return ProcessingResult.success_result(
                        data=adapter,
                        metrics={
                            "adapter_type": "polars",
                            "selection_reason": "polars_dataframe",
                        },
                        duration=duration,
                    )

            # Check for iterable of DataFrames
            if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                try:
                    first_item = next(iter(data))
                    return self.select_adapter(first_item)
                except StopIteration:
                    pass

            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Unsupported input type: {type(data)}",
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Adapter selection failed: {str(e)}",
                duration=duration,
            )

    def get_adapter_tag(self, adapter: DataAdapter) -> str:
        """Get tag for adapter.

        Args:
            adapter: Adapter to get tag for.

        Returns:
            Adapter tag string.
        """
        if isinstance(adapter, PandasAdapter):
            return "pandas"
        elif isinstance(adapter, PolarsAdapter):
            return "polars"
        else:
            return adapter.__class__.__name__.lower()

    def register_adapter(self, name: str, adapter: DataAdapter) -> None:
        """Register a custom adapter.

        Args:
            name: Name for the adapter.
            adapter: Adapter instance to register.
        """
        self._adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")

    def get_available_adapters(self) -> Dict[str, str]:
        """Get list of available adapters.

        Returns:
            Dictionary mapping adapter names to their types.
        """
        return {
            name: adapter.__class__.__name__ for name, adapter in self._adapters.items()
        }


class StreamingEngine:
    """Main streaming engine for data processing.

    This class orchestrates the entire data processing pipeline,
    including chunking, streaming, and coordination of operations.
    """

    def __init__(
        self,
        engine_manager: Optional[EngineManager] = None,
        chunker: Optional[AdaptiveChunker] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the streaming engine.

        Args:
            engine_manager: Engine manager for adapter selection.
            chunker: Chunker for data chunking operations.
            logger: Logger for engine operations.
        """
        self.engine_manager = engine_manager or EngineManager(logger)
        self.chunker = chunker or AdaptiveChunker(
            strategy=ChunkingStrategy.ADAPTIVE, logger=logger
        )
        self.logger = logger or logging.getLogger(__name__)

    def process_stream(
        self,
        source: Any,
        config: Any,
        chunk_size: int = 10000,
        force_chunk_in_memory: bool = False,
    ) -> ProcessingResult[Dict[str, Any]]:
        """Process a data stream end-to-end.

        Args:
            source: Data source to process.
            config: Configuration object.
            chunk_size: Size of chunks to create.
            force_chunk_in_memory: Whether to force in-memory chunking.

        Returns:
            ProcessingResult containing processing results.
        """
        start_time = time.time()

        try:
            # Select appropriate adapter
            adapter_result = self.engine_manager.select_adapter(source)
            if not adapter_result.success:
                return ProcessingResult.error_result(
                    f"Adapter selection failed: {adapter_result.error}"
                )

            adapter = adapter_result.data

            # Generate chunks
            chunk_result = self.chunker.chunks_from_source(
                source, chunk_size, force_chunk_in_memory
            )
            if not chunk_result.success:
                return ProcessingResult.error_result(
                    f"Chunking failed: {chunk_result.error}"
                )

            chunks = chunk_result.data

            # Process first chunk to initialize
            first_chunk = next(iter(chunks))
            kinds, accs = adapter.infer_and_build(first_chunk, config)

            # Process remaining chunks
            total_rows = 0
            total_memory = 0
            total_missing = 0

            for chunk in chunks:
                # Update accumulators
                adapter.consume_chunk(chunk, accs, kinds, self.logger)

                # Update statistics
                total_rows += len(chunk) if hasattr(chunk, "__len__") else 0
                total_memory += adapter.estimate_mem(chunk)
                total_missing += adapter.missing_cells(chunk)

            duration = time.time() - start_time

            return ProcessingResult.success_result(
                data={
                    "kinds": kinds,
                    "accs": accs,
                    "total_rows": total_rows,
                    "total_memory": total_memory,
                    "total_missing": total_missing,
                    "adapter_tag": self.engine_manager.get_adapter_tag(adapter),
                },
                metrics={
                    "processing_time": duration,
                    "chunks_processed": chunk_result.metrics.get("chunk_size", 0),
                    "adapter_type": adapter.__class__.__name__,
                },
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Stream processing failed: {str(e)}",
                duration=duration,
            )

    def maybe_corr_estimator(self, kinds: ColumnKinds, config: Any) -> Optional[Any]:
        """Create correlation estimator if needed.

        Args:
            kinds: Column kinds information.
            config: Configuration object.

        Returns:
            Correlation estimator or None.
        """
        try:
            # Import correlation estimator
            from ..corr import StreamingCorr

            # Check if we have enough numeric columns for correlation
            if len(kinds.numeric) < 2:
                return None

            # Check correlation threshold from config
            corr_threshold = getattr(config, "corr_threshold", 0.3)
            if corr_threshold <= 0:
                return None

            return StreamingCorr(kinds.numeric)

        except Exception as e:
            self.logger.warning(f"Failed to create correlation estimator: {e}")
            return None

    def prep_first_chunk(
        self, adapter: DataAdapter, first: Any, config: Any, row_kmv: Any
    ) -> None:
        """Prepare first chunk for processing.

        Args:
            adapter: Data adapter to use.
            first: First data chunk.
            config: Configuration object.
            row_kmv: Row KMV estimator.
        """
        try:
            # Update row KMV with first chunk
            if hasattr(adapter, "update_row_kmv"):
                adapter.update_row_kmv(first, row_kmv)

            # Estimate memory
            mem_estimate = adapter.estimate_mem(first)
            if hasattr(row_kmv, "add_mem"):
                row_kmv.add_mem(mem_estimate)

        except Exception as e:
            self.logger.warning(f"Failed to prepare first chunk: {e}")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information.

        Returns:
            Dictionary with engine information.
        """
        return {
            "available_adapters": self.engine_manager.get_available_adapters(),
            "chunker_strategy": self.chunker.strategy.value,
            "chunker_metrics": self.chunker.get_performance_metrics(),
        }


# Standalone functions for backward compatibility
def get_adapter_tag(adapter: DataAdapter) -> str:
    """Get tag for adapter (standalone function for backward compatibility).

    Args:
        adapter: Adapter to get tag for.

    Returns:
        Adapter tag string.
    """
    if isinstance(adapter, PandasAdapter):
        return "pandas"
    elif isinstance(adapter, PolarsAdapter):
        return "polars"
    else:
        return adapter.__class__.__name__.lower()


def maybe_corr_estimator(kinds: ColumnKinds, config: Any) -> Optional[Any]:
    """Create correlation estimator if needed (standalone function for backward compatibility).

    Args:
        kinds: Column kinds information.
        config: Configuration object.

    Returns:
        Correlation estimator or None.
    """
    try:
        # Import correlation estimator
        from ..analysis.correlation import StreamingCorr

        # Check if we have enough numeric columns for correlation
        if len(kinds.numeric) < 2:
            return None

        # Check correlation threshold from config
        corr_threshold = getattr(config, "corr_threshold", 0.3)
        if corr_threshold <= 0:
            return None

        return StreamingCorr(kinds.numeric)

    except Exception:
        return None


def consume_stream(
    adapter: DataAdapter,
    chunks: Iterator[Any],
    *,
    accs: Dict[str, Any],
    kinds: ColumnKinds,
    logger: Any,
    corr_est: Any,
    row_kmv: Any,
    approx_mem_bytes: int,
    total_missing_cells: int,
    first_columns: List[str],
    sample_section_html: str,
    cfg: Any,
    ckpt_mgr: Any,
    start_time: float,
    report_title: str,
    n_rows: int,
    chunk_idx: int,
) -> Tuple[int, int, int, int]:
    """Legacy consume_stream function for backward compatibility.

    This function provides backward compatibility with the old consume_stream
    interface while using the new StreamingEngine internally.

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
        Tuple of (n_rows, approx_mem_bytes, total_missing_cells, chunk_idx)
    """
    try:
        # Process chunks using the adapter directly
        for chunk in chunks:
            # Update accumulators
            adapter.consume_chunk(chunk, accs, kinds, logger)

            # Update correlation estimator
            if corr_est is not None:
                adapter.update_corr(chunk, corr_est, logger)

            # Update row KMV
            if hasattr(row_kmv, "update_from_pandas") and hasattr(chunk, "values"):
                row_kmv.update_from_pandas(chunk)
            elif hasattr(row_kmv, "update_from_polars") and hasattr(chunk, "to_pandas"):
                row_kmv.update_from_polars(chunk)

            # Update statistics
            n_rows += len(chunk) if hasattr(chunk, "__len__") else 0
            approx_mem_bytes += adapter.estimate_mem(chunk)
            total_missing_cells += adapter.missing_cells(chunk)
            chunk_idx += 1

        return n_rows, approx_mem_bytes, total_missing_cells, chunk_idx

    except Exception as e:
        if logger:
            logger.error(f"Stream consumption failed: {e}")
        raise
