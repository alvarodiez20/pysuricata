"""In-memory chunk adapters.

Exports a unified interface that yields pandas DataFrame chunks from
in-memory inputs (DataFrames or iterables of DataFrames).
"""

from .base import (
    ChunkingConfig,
    ChunkingConfigurationError,
    ChunkingEngine,
    ChunkingError,
    ChunkingMetrics,
    DataFrameType,
    MemoryLimitExceededError,
    UnsupportedDataTypeError,
    chunk_data,  # modern interface
    chunk_data_with_metrics,  # modern interface with metrics
    iter_chunks,  # legacy function for backward compatibility
)

__all__ = [
    # Legacy interface
    "iter_chunks",
    # Modern interfaces
    "chunk_data",
    "chunk_data_with_metrics",
    # Classes and configuration
    "ChunkingEngine",
    "ChunkingConfig",
    "ChunkingMetrics",
    # Exceptions
    "ChunkingError",
    "UnsupportedDataTypeError",
    "ChunkingConfigurationError",
    "MemoryLimitExceededError",
    # Enums
    "DataFrameType",
]
