"""Type definitions for the compute module.

This module defines the core data types and structures used throughout
the compute system, providing type safety and clear interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class ColumnKinds:
    """Container for column names grouped by data type.

    This class organizes column names by their inferred data types,
    providing a clean interface for type-based operations.

    Attributes:
        numeric: List of numeric column names.
        categorical: List of categorical column names.
        datetime: List of datetime column names.
        boolean: List of boolean column names.
    """

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return string representation of column kinds."""
        return (
            f"ColumnKinds(num={len(self.numeric)}, "
            f"cat={len(self.categorical)}, "
            f"dt={len(self.datetime)}, "
            f"bool={len(self.boolean)})"
        )

    def total_columns(self) -> int:
        """Get total number of columns.

        Returns:
            Total number of columns across all types.
        """
        return (
            len(self.numeric)
            + len(self.categorical)
            + len(self.datetime)
            + len(self.boolean)
        )

    def get_all_columns(self) -> List[str]:
        """Get all column names in a single list.

        Returns:
            List of all column names.
        """
        return self.numeric + self.categorical + self.datetime + self.boolean


@dataclass
class ProcessingResult(Generic[T]):
    """Type-safe result container for processing operations.

    This class provides a standardized way to return results from
    processing operations, including success status, data, and metadata.

    Attributes:
        success: Whether the operation was successful.
        data: The result data if successful, None otherwise.
        error: Error message if the operation failed, None otherwise.
        metrics: Optional performance metrics dictionary.
        duration: Time taken for the operation in seconds.
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    duration: float = 0.0

    def __post_init__(self) -> None:
        """Initialize metrics dictionary if not provided."""
        if self.metrics is None:
            self.metrics = {}

    @classmethod
    def success_result(
        cls, data: T, metrics: Optional[Dict[str, Any]] = None, duration: float = 0.0
    ) -> ProcessingResult[T]:
        """Create a successful result.

        Args:
            data: Result data.
            metrics: Optional performance metrics.
            duration: Operation duration in seconds.

        Returns:
            ProcessingResult with success=True.
        """
        return cls(success=True, data=data, metrics=metrics or {}, duration=duration)

    @classmethod
    def error_result(cls, error: str, duration: float = 0.0) -> ProcessingResult[T]:
        """Create an error result.

        Args:
            error: Error message.
            duration: Operation duration in seconds.

        Returns:
            ProcessingResult with success=False.
        """
        return cls(success=False, error=error, duration=duration)


@dataclass
class ChunkMetadata:
    """Metadata for a data chunk.

    This class contains metadata about a processed data chunk,
    including size, memory usage, and processing statistics.

    Attributes:
        chunk_size: Number of rows in the chunk.
        memory_bytes: Memory usage of the chunk in bytes.
        missing_cells: Number of missing cells in the chunk.
        processing_time: Time taken to process the chunk in seconds.
        chunk_index: Index of the chunk in the sequence.
    """

    chunk_size: int
    memory_bytes: int
    missing_cells: int
    processing_time: float
    chunk_index: int

    def memory_mb(self) -> float:
        """Get memory usage in megabytes.

        Returns:
            Memory usage in MB.
        """
        return self.memory_bytes / (1024 * 1024)

    def missing_percentage(self) -> float:
        """Get percentage of missing cells.

        Returns:
            Percentage of missing cells (0-100).
        """
        if self.chunk_size == 0:
            return 0.0
        return (self.missing_cells / (self.chunk_size * self.chunk_size)) * 100.0


@dataclass
class InferenceResult:
    """Result of type inference operation.

    This class contains the results of a type inference operation,
    including the inferred types and any warnings or errors.

    Attributes:
        kinds: Inferred column kinds.
        warnings: List of inference warnings.
        errors: List of inference errors.
        confidence: Overall confidence score (0-1).
    """

    kinds: ColumnKinds
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def is_high_confidence(self) -> bool:
        """Check if inference has high confidence.

        Returns:
            True if confidence is above 0.8.
        """
        return self.confidence >= 0.8

    def has_warnings(self) -> bool:
        """Check if there are any warnings.

        Returns:
            True if there are warnings.
        """
        return len(self.warnings) > 0

    def has_errors(self) -> bool:
        """Check if there are any errors.

        Returns:
            True if there are errors.
        """
        return len(self.errors) > 0


@dataclass
class ProcessingConfig:
    """Configuration for processing operations.

    This class contains configuration parameters for various
    processing operations, providing a centralized configuration.

    Attributes:
        chunk_size: Default chunk size for processing.
        max_memory_mb: Maximum memory usage in MB.
        enable_caching: Whether to enable result caching.
        parallel_processing: Whether to enable parallel processing.
        error_threshold: Error threshold for processing.
    """

    chunk_size: int = 10000
    max_memory_mb: int = 1024
    enable_caching: bool = True
    parallel_processing: bool = False
    error_threshold: float = 0.1

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if not 0 <= self.error_threshold <= 1:
            raise ValueError("error_threshold must be between 0 and 1")
