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
from .config import AccumulatorConfig
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
    """Create an optimized AccumulatorConfig for big data processing.

    This is a convenience function for creating accumulator configurations
    with optimal parameter combinations for large-scale data processing.

    Args:
        numeric_sample_size: Sample size for numeric statistics
        uniques_sketch_size: Sketch size for unique counting
        top_k_size: Number of top categories to track
        enable_performance_tracking: Whether to track performance metrics
        enable_error_recovery: Whether to attempt error recovery
        max_memory_mb: Maximum memory usage in MB

    Returns:
        Configured AccumulatorConfig instance optimized for performance

    Raises:
        ValueError: If parameters are invalid
    """
    from .config import (
        BooleanConfig,
        CategoricalConfig,
        DatetimeConfig,
        NumericConfig,
    )

    # Create optimized individual configurations
    numeric_config = NumericConfig(
        sample_size=numeric_sample_size,
        uniques_sketch_size=uniques_sketch_size,
    )

    categorical_config = CategoricalConfig(
        top_k_size=top_k_size,
        uniques_sketch_size=uniques_sketch_size,
    )

    datetime_config = DatetimeConfig(
        sample_size=numeric_sample_size,
        uniques_sketch_size=uniques_sketch_size,
    )

    boolean_config = BooleanConfig()

    # Create master configuration optimized for big data
    config = AccumulatorConfig(
        numeric=numeric_config,
        categorical=categorical_config,
        datetime=datetime_config,
        boolean=boolean_config,
        enable_performance_tracking=enable_performance_tracking,
        enable_error_recovery=enable_error_recovery,
        max_memory_mb=max_memory_mb,
    )

    # Validate configuration for reliability
    config.validate()

    return config


def get_accumulator_info(accs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get comprehensive information about accumulator instances for monitoring.

    This function provides detailed information about accumulator instances,
    including their configuration, memory usage, and performance metrics
    for production monitoring and optimization.

    Args:
        accs: Dictionary of accumulator instances

    Returns:
        Dictionary containing comprehensive information about each accumulator
    """
    info = {}

    for name, acc in accs.items():
        acc_info = {
            "type": type(acc).__name__,
            "name": getattr(acc, "name", name),
            "count": getattr(acc, "count", 0),
            "missing": getattr(acc, "missing", 0),
        }

        # Add configuration information for optimization insights
        if hasattr(acc, "config"):
            acc_info["config"] = {
                "sample_size": getattr(acc.config, "sample_size", None),
                "uniques_sketch_size": getattr(acc.config, "uniques_sketch_size", None),
                "top_k_size": getattr(acc.config, "top_k_size", None),
            }

        # Add performance metrics for production monitoring
        if hasattr(acc, "get_performance_metrics"):
            metrics = acc.get_performance_metrics()
            if metrics:
                acc_info["performance"] = {
                    "update_count": metrics.update_count,
                    "avg_update_time": metrics.avg_update_time,
                    "updates_per_second": metrics.updates_per_second,
                    "memory_usage_bytes": metrics.memory_usage_bytes,
                }

        # Add memory usage for big data optimization
        if hasattr(acc, "_bytes_seen"):
            acc_info["memory_bytes"] = acc._bytes_seen

        info[name] = acc_info

    return info


def validate_accumulator_compatibility(accs: Dict[str, Any]) -> Dict[str, bool]:
    """Validate accumulator compatibility and health for production reliability.

    This function checks if accumulator instances are in a valid state
    and can be safely used for processing large datasets.

    Args:
        accs: Dictionary of accumulator instances

    Returns:
        Dictionary mapping accumulator names to validation results
    """
    results = {}

    for name, acc in accs.items():
        is_valid = True
        issues = []

        # Check basic attributes for reliability
        if not hasattr(acc, "name"):
            is_valid = False
            issues.append("Missing name attribute")

        if not hasattr(acc, "update"):
            is_valid = False
            issues.append("Missing update method")

        if not hasattr(acc, "finalize"):
            is_valid = False
            issues.append("Missing finalize method")

        # Check for data integrity issues
        if hasattr(acc, "count") and acc.count < 0:
            is_valid = False
            issues.append("Negative count")

        if hasattr(acc, "missing") and acc.missing < 0:
            is_valid = False
            issues.append("Negative missing count")

        # Check configuration validity for production reliability
        if hasattr(acc, "config"):
            try:
                if hasattr(acc.config, "validate"):
                    acc.config.validate()
            except Exception as e:
                is_valid = False
                issues.append(f"Invalid configuration: {e}")

        results[name] = {"valid": is_valid, "issues": issues}

    return results


# Backward compatibility function for legacy systems
def build_accumulators_legacy(kinds: ColumnKinds, cfg: EngineConfig) -> Dict[str, Any]:
    """Legacy function for backward compatibility with existing systems.

    This function maintains the original interface while using the new
    high-performance implementation internally.

    Args:
        kinds: Column kinds information
        cfg: Engine configuration

    Returns:
        Dictionary mapping column names to accumulator instances
    """
    return build_accumulators(kinds, cfg, None)
