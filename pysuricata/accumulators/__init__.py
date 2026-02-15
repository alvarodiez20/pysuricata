"""High-performance accumulators and streaming sketches optimized for big data.

This package contains production-ready implementations for numeric, categorical, datetime,
and boolean accumulators, as well as reusable sketch algorithms optimized for processing
massive datasets efficiently with minimal memory footprint.

The package provides enterprise-grade accumulator implementations:
- High Performance: Vectorized operations and optimized algorithms
- Scalability: Memory-bounded processing for big data
- Reliability: Comprehensive error handling and validation
- Configurability: Fine-grained control over behavior and performance
"""

# Core sketch algorithms for memory-efficient processing
from .sketches import KMV, MisraGries, ReservoirSampler  # re-export

# Production-grade accumulator system
try:
    from .algorithms import (
        ExtremeTracker,
        MonotonicityDetector,
        OutlierDetector,
        PerformanceMetrics,
        StreamingMoments,
    )
    from .boolean import BooleanAccumulator, BooleanSummary
    from .categorical import CategoricalAccumulator, CategoricalSummary
    from .config import (
        AccumulatorConfig,
        BooleanConfig,
        CategoricalConfig,
        DatetimeConfig,
        NumericConfig,
    )
    from .datetime import DatetimeAccumulator, DatetimeSummary
    from .factory import (
        build_accumulators,
        create_accumulator_config,
        get_accumulator_info,
        validate_accumulator_compatibility,
    )
    from .numeric import DtypeSuggestion, NumericAccumulator, NumericSummary

    # Version information
    __version__ = "2.0.0"
    _MODERN_SYSTEM_AVAILABLE = True

except ImportError:
    # Fallback if new system is not available
    __version__ = "1.0.0"
    _MODERN_SYSTEM_AVAILABLE = False
