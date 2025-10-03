"""Production-grade factory for creating high-performance accumulators optimized for big data.

This module provides a comprehensive factory for creating accumulator instances with
enterprise-grade configuration support, advanced error handling, and optimal performance
characteristics for processing massive datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..compute.core.types import ColumnKinds
from ..config import EngineConfig
from .boolean import BooleanAccumulator
from .categorical import CategoricalAccumulator
from .config import (
    AccumulatorConfig,
    BooleanConfig,
    CategoricalConfig,
    DatetimeConfig,
    NumericConfig,
)
from .datetime import DatetimeAccumulator
from .numeric import NumericAccumulator


def build_accumulators(
    kinds: ColumnKinds,
    cfg: EngineConfig,
    accumulator_config: Optional[AccumulatorConfig] = None,
) -> Dict[str, Any]:
    """Build high-performance accumulator instances optimized for big data processing.

    This function creates accumulator instances for each column based on
    inferred kinds, using either the provided AccumulatorConfig or
    creating one from the legacy EngineConfig with comprehensive validation.

    Args:
        kinds: Column kinds information
        cfg: Legacy engine configuration
        accumulator_config: Optional modern accumulator configuration

    Returns:
        Dictionary mapping column names to accumulator instances

    Raises:
        ValueError: If configuration is invalid
        TypeError: If column kinds are invalid
    """
    # Create or validate accumulator configuration
    if accumulator_config is None:
        accumulator_config = AccumulatorConfig.from_legacy_config(cfg)

    # Validate configuration for production reliability
    accumulator_config.validate()

    accs: Dict[str, Any] = {}

    try:
        # Create numeric accumulators with optimized configuration
        for name in kinds.numeric:
            accs[name] = NumericAccumulator(
                name=name, config=accumulator_config.numeric
            )

        # Create boolean accumulators with efficient processing
        for name in kinds.boolean:
            accs[name] = BooleanAccumulator(
                name=name, config=accumulator_config.boolean
            )

        # Create datetime accumulators with temporal analysis
        for name in kinds.datetime:
            accs[name] = DatetimeAccumulator(
                name=name, config=accumulator_config.datetime
            )

        # Create categorical accumulators with scalable sketch algorithms
        for name in kinds.categorical:
            accs[name] = CategoricalAccumulator(
                name=name, config=accumulator_config.categorical
            )

    except Exception as e:
        raise ValueError(f"Failed to create accumulators: {e}") from e

    return accs


def create_accumulator_config(
    numeric_sample_size: int = 20_000,
    uniques_sketch_size: int = 2_048,
    top_k_size: int = 50,
    enable_performance_tracking: bool = False,
    enable_error_recovery: bool = True,
    max_memory_mb: Optional[int] = None,
) -> AccumulatorConfig:
    """Create a production-grade accumulator configuration with optimized defaults.

    This function provides a convenient way to create AccumulatorConfig instances
    with sensible defaults for big data processing scenarios.

    Args:
        numeric_sample_size: Sample size for numeric statistics
        uniques_sketch_size: Sketch size for unique value estimation
        top_k_size: Maximum number of top-k items to track
        enable_performance_tracking: Enable performance monitoring
        enable_error_recovery: Enable automatic error recovery
        max_memory_mb: Maximum memory usage in MB

    Returns:
        Configured AccumulatorConfig instance
    """
    return AccumulatorConfig(
        numeric=NumericConfig(
            sample_size=numeric_sample_size,
            uniques_sketch_size=uniques_sketch_size,
            top_k_size=top_k_size,
            enable_memory_tracking=enable_performance_tracking,
        ),
        boolean=BooleanConfig(
            enable_ratio_calculation=True,
        ),
        categorical=CategoricalConfig(
            uniques_sketch_size=uniques_sketch_size,
            top_k_size=top_k_size,
        ),
        datetime=DatetimeConfig(
            enable_temporal_patterns=True,
        ),
    )


def get_accumulator_info() -> Dict[str, Any]:
    """Get comprehensive information about available accumulator types.

    Returns:
        Dictionary containing accumulator type information
    """
    return {
        "numeric": {
            "class": "NumericAccumulator",
            "description": "High-performance numeric statistics with streaming algorithms",
            "features": [
                "Streaming moments calculation",
                "Reservoir sampling",
                "Outlier detection",
                "Quantile estimation",
                "Memory-bounded processing",
            ],
        },
        "boolean": {
            "class": "BooleanAccumulator",
            "description": "Efficient boolean value processing with ratio calculations",
            "features": [
                "True/false ratio calculation",
                "Entropy calculation",
                "Vectorized operations",
            ],
        },
        "categorical": {
            "class": "CategoricalAccumulator",
            "description": "Scalable categorical data processing with sketch algorithms",
            "features": [
                "KMV sketch for unique estimation",
                "Misra-Gries for top-k tracking",
                "Diversity metrics",
                "Memory-efficient processing",
            ],
        },
        "datetime": {
            "class": "DatetimeAccumulator",
            "description": "Temporal data analysis with advanced time series features",
            "features": [
                "Temporal statistics",
                "Seasonality detection",
                "Date range analysis",
                "Time zone handling",
            ],
        },
    }


def validate_accumulator_compatibility(
    kinds: ColumnKinds, accumulator_config: AccumulatorConfig
) -> bool:
    """Validate compatibility between column kinds and accumulator configuration.

    Args:
        kinds: Column kinds information
        accumulator_config: Accumulator configuration

    Returns:
        True if compatible, False otherwise

    Raises:
        ValueError: If validation fails
    """
    try:
        # Validate numeric configuration
        if kinds.numeric and accumulator_config.numeric.sample_size <= 0:
            raise ValueError("Numeric sample size must be positive")

        # Validate categorical configuration
        if (
            kinds.categorical
            and accumulator_config.categorical.uniques_sketch_size <= 0
        ):
            raise ValueError("Categorical sketch size must be positive")

        # Validate boolean configuration
        if kinds.boolean and not accumulator_config.boolean.enable_ratio_calculation:
            raise ValueError("Boolean ratio calculation must be enabled")

        # Validate datetime configuration
        if kinds.datetime and not accumulator_config.datetime.enable_temporal_analysis:
            raise ValueError("Datetime temporal analysis must be enabled")

        return True

    except Exception as e:
        raise ValueError(f"Accumulator compatibility validation failed: {e}") from e
