"""Protocol definitions for the compute module.

This module defines the abstract interfaces (protocols) used throughout
the compute system, enabling dependency injection and better testability.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

from .types import ColumnKinds, ProcessingResult


class DataAdapter(Protocol):
    """Protocol for data backend adapters.

    This protocol defines the interface that all data backend adapters
    (pandas, polars, etc.) must implement. It provides a unified interface
    for data processing operations across different backends.
    """

    def infer_and_build(
        self, data: Any, config: Any
    ) -> Tuple[ColumnKinds, Dict[str, Any]]:
        """Infer column types and build accumulators.

        Args:
            data: Input data to analyze.
            config: Configuration object with processing parameters.

        Returns:
            Tuple of (column_kinds, accumulators_dict).
        """
        ...

    def estimate_mem(self, frame: Any) -> int:
        """Estimate memory usage of a data frame.

        Args:
            frame: Data frame to estimate memory for.

        Returns:
            Estimated memory usage in bytes.
        """
        ...

    def missing_cells(self, frame: Any) -> int:
        """Count missing cells in a data frame.

        Args:
            frame: Data frame to count missing cells in.

        Returns:
            Number of missing cells.
        """
        ...

    def consume_chunk(
        self,
        data: Any,
        accs: Dict[str, Any],
        kinds: ColumnKinds,
        logger: Optional[Any] = None,
    ) -> None:
        """Consume a data chunk and update accumulators.

        Args:
            data: Data chunk to process.
            accs: Dictionary of accumulators to update.
            kinds: Column type information.
            logger: Optional logger for progress tracking.
        """
        ...

    def update_corr(
        self, frame: Any, corr_est: Any, logger: Optional[Any] = None
    ) -> None:
        """Update correlation estimator with new data.

        Args:
            frame: Data frame to process.
            corr_est: Correlation estimator to update.
            logger: Optional logger for progress tracking.
        """
        ...

    def sample_section_html(self, first: Any, cfg: Any) -> str:
        """Generate HTML for sample data section.

        Args:
            first: First data chunk for sampling.
            cfg: Configuration object.

        Returns:
            HTML string for sample section.
        """
        ...


class ChunkProcessor(Protocol):
    """Protocol for chunk processing operations.

    This protocol defines the interface for processing data chunks,
    including data conversion and chunk-specific operations.
    """

    def process_chunk(
        self, chunk: Any, accs: Dict[str, Any], kinds: ColumnKinds
    ) -> None:
        """Process a single data chunk.

        Args:
            chunk: Data chunk to process.
            accs: Dictionary of accumulators to update.
            kinds: Column type information.
        """
        ...

    def convert_data(self, series: Any, target_type: str) -> Any:
        """Convert data series to target type.

        Args:
            series: Data series to convert.
            target_type: Target data type.

        Returns:
            Converted data series.
        """
        ...


class TypeInferrer(Protocol):
    """Protocol for type inference operations.

    This protocol defines the interface for inferring column types
    from data, supporting different backends and inference strategies.
    """

    def infer_kinds(self, data: Any) -> ColumnKinds:
        """Infer column types from data.

        Args:
            data: Input data to analyze.

        Returns:
            ColumnKinds object with inferred types.
        """
        ...

    def infer_series_type(self, series: Any) -> str:
        """Infer type of a single data series.

        Args:
            series: Data series to analyze.

        Returns:
            Inferred type string ('numeric', 'categorical', 'datetime', 'boolean').
        """
        ...


class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies.

    This protocol defines the interface for different chunking strategies,
    enabling adaptive chunking based on data characteristics.
    """

    def chunks_from_source(
        self, source: Any, chunk_size: int, force_chunk_in_memory: bool
    ) -> Iterator[Any]:
        """Generate chunks from a data source.

        Args:
            source: Data source to chunk.
            chunk_size: Size of chunks to create.
            force_chunk_in_memory: Whether to force in-memory chunking.

        Yields:
            Data chunks.
        """
        ...

    def adaptive_chunk_size(self, data: Any) -> int:
        """Determine optimal chunk size for data.

        Args:
            data: Data to analyze for optimal chunk size.

        Returns:
            Optimal chunk size.
        """
        ...


class MetricsComputer(Protocol):
    """Protocol for metrics computation.

    This protocol defines the interface for computing various metrics
    and statistics from processed data.
    """

    def build_manifest_inputs(
        self,
        kinds: ColumnKinds,
        accs: Dict[str, Any],
        row_kmv: Any,
        first_columns: Sequence[str],
    ) -> ProcessingResult[
        Tuple[
            Dict[str, Tuple[str, Any]],
            List[str],
            int,
            int,
            List[Tuple[str, float, int]],
        ]
    ]:
        """Build manifest inputs for summary generation.

        Args:
            kinds: Column type information.
            accs: Dictionary of accumulators.
            row_kmv: Row KMV estimator.
            first_columns: First chunk column order.

        Returns:
            ProcessingResult containing manifest inputs.
        """
        ...

    def apply_correlation_chips(
        self,
        accs: Dict[str, Any],
        kinds: ColumnKinds,
        top_map: Dict[str, Any],
    ) -> ProcessingResult[None]:
        """Apply correlation chips to accumulators.

        Args:
            accs: Dictionary of accumulators.
            kinds: Column type information.
            top_map: Correlation mapping.

        Returns:
            ProcessingResult indicating success/failure.
        """
        ...

    def build_summary(
        self,
        kinds_map: Dict[str, Tuple[str, Any]],
        col_order: List[str],
        row_kmv: Any,
        total_missing_cells: int,
        n_rows: int,
        n_cols: int,
        miss_list: List[Tuple[str, float, int]],
    ) -> ProcessingResult[Optional[Dict[str, Any]]]:
        """Build programmatic summary from processed data.

        Args:
            kinds_map: Column kinds mapping.
            col_order: Column order.
            row_kmv: Row KMV estimator.
            total_missing_cells: Total missing cells count.
            n_rows: Number of rows.
            n_cols: Number of columns.
            miss_list: Missing values list.

        Returns:
            ProcessingResult containing the summary.
        """
        ...
