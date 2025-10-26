"""Configuration system for accumulators.

This module provides comprehensive configuration options for all accumulator types,
enabling fine-tuned control over memory usage, accuracy, and performance characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NumericConfig:
    """Configuration for numeric accumulator behavior.

    Attributes:
        sample_size: Reservoir sample size for quantile estimation. Larger values
            provide better accuracy but use more memory. Default: 20,000
        uniques_sketch_size: KMV sketch size for distinct counting. Larger values
            provide better accuracy but use more memory. Default: 2,048
        top_k_size: Maximum number of top values to track for common values analysis.
            Larger values provide better coverage but use more memory. Default: 20
        enable_monotonicity_detection: Whether to track if values are monotonic.
            Default: True
        enable_outlier_detection: Whether to detect outliers using IQR and MAD.
            Default: True
        max_extremes: Maximum number of extreme values to track. Default: 5
        enable_memory_tracking: Whether to track memory usage. Default: True
        enable_geometric_mean: Whether to compute geometric mean. Default: True
        enable_confidence_intervals: Whether to compute confidence intervals.
            Default: False
        enable_chunk_metadata: Whether to track chunk metadata for visualization.
            When disabled, saves memory by not tracking per-chunk statistics.
            Default: True
        max_chunks: Maximum number of chunks to track metadata for. When exceeded,
            switches to summary mode to prevent unbounded growth. Default: 1000
        outlier_methods: Methods to use for outlier detection. Default: ['iqr', 'mad']
    """

    sample_size: int = 20_000
    uniques_sketch_size: int = 2_048
    top_k_size: int = 50
    enable_monotonicity_detection: bool = True
    enable_outlier_detection: bool = True
    max_extremes: int = 5
    enable_memory_tracking: bool = True
    enable_geometric_mean: bool = True
    enable_confidence_intervals: bool = False
    enable_chunk_metadata: bool = True
    max_chunks: int = 1000
    outlier_methods: list[str] = field(default_factory=lambda: ["iqr", "mad"])

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.uniques_sketch_size <= 0:
            raise ValueError("uniques_sketch_size must be positive")
        if self.top_k_size <= 0:
            raise ValueError("top_k_size must be positive")
        if self.max_extremes <= 0:
            raise ValueError("max_extremes must be positive")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        if not self.outlier_methods:
            raise ValueError("outlier_methods cannot be empty")


@dataclass
class CategoricalConfig:
    """Configuration for categorical accumulator behavior.

    Attributes:
        top_k_size: Maximum number of top categories to track. Default: 50
        uniques_sketch_size: KMV sketch size for distinct counting. Default: 2,048
        enable_case_variants: Whether to track case-insensitive variants.
            Default: True
        enable_trim_variants: Whether to track whitespace-trimmed variants.
            Default: True
        enable_length_stats: Whether to compute string length statistics.
            Default: True
        length_sample_size: Sample size for length statistics. Default: 5,000
        enable_memory_tracking: Whether to track memory usage. Default: True
    """

    top_k_size: int = 50
    uniques_sketch_size: int = 2_048
    enable_case_variants: bool = True
    enable_trim_variants: bool = True
    enable_length_stats: bool = True
    length_sample_size: int = 5_000
    enable_memory_tracking: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.top_k_size <= 0:
            raise ValueError("top_k_size must be positive")
        if self.uniques_sketch_size <= 0:
            raise ValueError("uniques_sketch_size must be positive")
        if self.length_sample_size <= 0:
            raise ValueError("length_sample_size must be positive")


@dataclass
class DatetimeConfig:
    """Configuration for datetime accumulator behavior.

    Attributes:
        sample_size: Reservoir sample size for datetime analysis. Default: 20,000
        uniques_sketch_size: KMV sketch size for distinct counting. Default: 2,048
        enable_monotonicity_detection: Whether to track if timestamps are monotonic.
            Default: True
        enable_temporal_patterns: Whether to compute hour/day/month patterns.
            Default: True
        enable_memory_tracking: Whether to track memory usage. Default: True
        timezone_aware: Whether to handle timezone information. Default: False
    """

    sample_size: int = 20_000
    uniques_sketch_size: int = 2_048
    enable_monotonicity_detection: bool = True
    enable_temporal_patterns: bool = True
    enable_memory_tracking: bool = True
    timezone_aware: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.uniques_sketch_size <= 0:
            raise ValueError("uniques_sketch_size must be positive")


@dataclass
class BooleanConfig:
    """Configuration for boolean accumulator behavior.

    Attributes:
        enable_memory_tracking: Whether to track memory usage. Default: True
        enable_ratio_calculation: Whether to compute true/false ratios. Default: True
    """

    enable_memory_tracking: bool = True
    enable_ratio_calculation: bool = True


@dataclass
class AccumulatorConfig:
    """Master configuration for all accumulator types.

    This configuration object provides fine-grained control over all accumulator
    behaviors while maintaining backward compatibility with existing code.

    Attributes:
        numeric: Configuration for numeric accumulators
        categorical: Configuration for categorical accumulators
        datetime: Configuration for datetime accumulators
        boolean: Configuration for boolean accumulators
        enable_performance_tracking: Whether to track performance metrics
        enable_error_recovery: Whether to attempt error recovery on failures
        max_memory_mb: Maximum memory usage in MB (soft limit). Default: None (unlimited)
    """

    numeric: NumericConfig = field(default_factory=NumericConfig)
    categorical: CategoricalConfig = field(default_factory=CategoricalConfig)
    datetime: DatetimeConfig = field(default_factory=DatetimeConfig)
    boolean: BooleanConfig = field(default_factory=BooleanConfig)
    enable_performance_tracking: bool = False
    enable_error_recovery: bool = True
    max_memory_mb: Optional[int] = None

    @classmethod
    def from_legacy_config(cls, cfg) -> AccumulatorConfig:
        """Create configuration from legacy EngineConfig.

        This method provides backward compatibility with existing configuration
        while enabling new features through the modern configuration system.

        Args:
            cfg: Legacy EngineConfig object

        Returns:
            New AccumulatorConfig with values mapped from legacy config
        """
        return cls(
            numeric=NumericConfig(
                sample_size=getattr(cfg, "numeric_sample_k", 20_000),
                uniques_sketch_size=getattr(cfg, "uniques_k", 2_048),
            ),
            categorical=CategoricalConfig(
                top_k_size=getattr(cfg, "topk_k", 50),
                uniques_sketch_size=getattr(cfg, "uniques_k", 2_048),
            ),
            datetime=DatetimeConfig(
                sample_size=getattr(cfg, "numeric_sample_k", 20_000),
                uniques_sketch_size=getattr(cfg, "uniques_k", 2_048),
            ),
        )

    def validate(self) -> None:
        """Validate the entire configuration."""
        self.numeric.__post_init__()
        self.categorical.__post_init__()
        self.datetime.__post_init__()

        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

    def get_memory_estimate_mb(self) -> float:
        """Estimate total memory usage in MB.

        Returns:
            Estimated memory usage in megabytes
        """
        # Rough estimates based on configuration
        numeric_mb = (
            self.numeric.sample_size * 8 + self.numeric.uniques_sketch_size * 8
        ) / 1_000_000
        categorical_mb = (
            self.categorical.top_k_size * 100 + self.categorical.uniques_sketch_size * 8
        ) / 1_000_000
        datetime_mb = (
            self.datetime.sample_size * 8 + self.datetime.uniques_sketch_size * 8
        ) / 1_000_000
        boolean_mb = 0.001  # Minimal memory usage

        return numeric_mb + categorical_mb + datetime_mb + boolean_mb
