"""Compute orchestration and manifest building.

This package orchestrates kind inference, chunk consumption, and
construction of the JSON-safe manifest used by the renderer and API.

The compute module is organized into several focused submodules:

- core: Core abstractions, protocols, and types
- adapters: Data backend adapters (pandas, polars)
- processing: Unified data processing (conversion, chunking, inference)
- analysis: Statistical analysis (correlation, estimators, metrics)
- orchestration: High-level orchestration and coordination
"""

# Core abstractions
# Adapters
from .adapters import (
    BaseAdapter,
    PandasAdapter,
    PolarsAdapter,
)

# Analysis
from .analysis import (
    RowKMV,
    StreamingCorr,
    apply_corr_chips,
    build_kinds_map,
    build_manifest_inputs,
    compute_col_order,
    compute_dataset_shape,
    compute_top_missing,
)

# Orchestration - import only what's needed to avoid circular dependencies
# from .orchestration import (
#     StreamingEngine,
#     EngineManager,
#     ManifestBuilder,
#     ChunkingService,
#     EngineService,
#     MetricsService,
#     ResourceManager,
#     ProcessingService,
# )
# Core abstractions
from .core import (
    ChunkingError,
    ChunkMetadata,
    ChunkProcessor,
    ColumnKinds,
    ComputeError,
    ConversionError,
    DataAdapter,
    InferenceError,
    InferenceResult,
    ProcessingResult,
    TypeInferrer,
)

# Processing
from .processing import (
    AdaptiveChunker,
    ChunkingStrategy,
    ConversionStrategy,
    InferenceStrategy,
    UnifiedConverter,
    UnifiedTypeInferrer,
)

__all__ = [
    # Core
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
    # Adapters
    "BaseAdapter",
    "PandasAdapter",
    "PolarsAdapter",
    # Processing
    "UnifiedConverter",
    "ConversionStrategy",
    "AdaptiveChunker",
    "ChunkingStrategy",
    "UnifiedTypeInferrer",
    "InferenceStrategy",
    # Analysis
    "StreamingCorr",
    "RowKMV",
    "build_kinds_map",
    "compute_top_missing",
    "compute_col_order",
    "compute_dataset_shape",
    "build_manifest_inputs",
    "apply_corr_chips",
    # Orchestration - commented out to avoid circular dependencies
    # "StreamingEngine",
    # "EngineManager",
    # "ManifestBuilder",
    # "ChunkingService",
    # "EngineService",
    # "MetricsService",
    # "ResourceManager",
    # "ProcessingService",
]
