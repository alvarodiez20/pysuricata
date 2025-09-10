"""Core abstractions and types for the compute module.

This module provides the foundational types, protocols, and exceptions
used throughout the compute system.
"""

from .exceptions import ChunkingError, ComputeError, ConversionError, InferenceError
from .protocols import ChunkProcessor, DataAdapter, TypeInferrer
from .types import (
    ChunkMetadata,
    ColumnKinds,
    InferenceResult,
    ProcessingResult,
)

__all__ = [
    "ComputeError",
    "ChunkingError",
    "InferenceError",
    "ConversionError",
    "DataAdapter",
    "ChunkProcessor",
    "TypeInferrer",
    "ColumnKinds",
    "ProcessingResult",
    "ChunkMetadata",
    "InferenceResult",
]
