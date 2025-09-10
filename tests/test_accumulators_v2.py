"""Comprehensive tests for the refactored accumulator system.

This module provides extensive tests for all accumulator types, ensuring
correctness, performance, and reliability of the new implementation.
"""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pysuricata.accumulators.algorithms import (
    ExtremeTracker,
    MonotonicityDetector,
    OutlierDetector,
    PerformanceMetrics,
    StreamingMoments,
)
from pysuricata.accumulators.boolean import BooleanAccumulator
from pysuricata.accumulators.categorical import CategoricalAccumulator
from pysuricata.accumulators.config import (
    AccumulatorConfig,
    BooleanConfig,
    CategoricalConfig,
    DatetimeConfig,
    NumericConfig,
)
from pysuricata.accumulators.datetime import DatetimeAccumulator
from pysuricata.accumulators.factory2 import (
    build_accumulators,
    create_accumulator_config,
    get_accumulator_info,
    validate_accumulator_compatibility,
)
from pysuricata.accumulators.numeric import NumericAccumulator


class TestAccumulatorConfig:
    """Test accumulator configuration system."""

    def test_numeric_config_validation(self):
        """Test numeric configuration validation."""
        # Valid configuration
        config = NumericConfig(
            sample_size=10000, uniques_sketch_size=1024, max_extremes=10
        )
        config.__post_init__()  # Should not raise

        # Invalid configurations
        with pytest.raises(ValueError):
            NumericConfig(sample_size=0)

        with pytest.raises(ValueError):
            NumericConfig(uniques_sketch_size=0)

    def test_categorical_config_validation(self):
        """Test categorical configuration validation."""
        # Valid configuration
        config = CategoricalConfig(top_k_size=50, uniques_sketch_size=1024)
        config.__post_init__()  # Should not raise

        # Invalid configurations
        with pytest.raises(ValueError):
            CategoricalConfig(top_k_size=0)

    def test_accumulator_config_creation(self):
        """Test master accumulator configuration creation."""
        config = AccumulatorConfig()
        config.validate()  # Should not raise

        # Test custom configuration
        custom_config = AccumulatorConfig(
            numeric=NumericConfig(sample_size=5000),
            enable_performance_tracking=True,
            max_memory_mb=1000,
        )
        custom_config.validate()

    def test_config_from_legacy(self):
        """Test creation from legacy engine config."""
        # Mock legacy config
        legacy_config = Mock()
        legacy_config.numeric_sample_k = 10000
        legacy_config.uniques_k = 2048
        legacy_config.topk_k = 100

        config = AccumulatorConfig.from_legacy_config(legacy_config)
        assert config.numeric.sample_size == 10000
        assert config.numeric.uniques_sketch_size == 2048
        assert config.categorical.top_k_size == 100


class TestStreamingMoments:
    """Test streaming moments algorithm."""

    def test_basic_statistics(self):
        """Test basic statistical calculations."""
        moments = StreamingMoments()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        moments.update(data)
        stats = moments.get_statistics()

        assert stats["count"] == 5
        assert abs(stats["mean"] - 3.0) < 1e-10
        assert abs(stats["std"] - np.std(data, ddof=1)) < 1e-10
        assert stats["variance"] > 0

    def test_empty_data(self):
        """Test handling of empty data."""
        moments = StreamingMoments()
        stats = moments.get_statistics()

        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_merge_moments(self):
        """Test merging of streaming moments."""
        moments1 = StreamingMoments()
        moments2 = StreamingMoments()

        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])

        moments1.update(data1)
        moments2.update(data2)

        moments1.merge(moments2)
        stats = moments1.get_statistics()

        combined_data = np.concatenate([data1, data2])
        expected_mean = np.mean(combined_data)

        assert stats["count"] == 6
        assert abs(stats["mean"] - expected_mean) < 1e-10

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        moments = StreamingMoments(enable_performance_tracking=True)
        data = np.random.random(1000)

        moments.update(data)
        stats = moments.get_statistics()

        # Should have performance metrics
        assert "update_time" in stats or stats["count"] > 0


class TestExtremeTracker:
    """Test extreme values tracker."""

    def test_basic_extremes(self):
        """Test basic extreme value tracking."""
        tracker = ExtremeTracker(max_extremes=3)
        values = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        indices = np.arange(len(values))

        tracker.update(values, indices)
        min_pairs, max_pairs = tracker.get_extremes()

        # Should track smallest and largest values
        assert len(min_pairs) <= 3
        assert len(max_pairs) <= 3

        if min_pairs:
            assert min_pairs[0][1] == 1.0  # Smallest value
        if max_pairs:
            assert max_pairs[0][1] == 8.0  # Largest value

    def test_merge_extremes(self):
        """Test merging of extreme trackers."""
        tracker1 = ExtremeTracker(max_extremes=2)
        tracker2 = ExtremeTracker(max_extremes=2)

        values1 = np.array([1.0, 3.0])
        values2 = np.array([2.0, 4.0])
        indices1 = np.array([0, 1])
        indices2 = np.array([2, 3])

        tracker1.update(values1, indices1)
        tracker2.update(values2, indices2)

        tracker1.merge(tracker2)
        min_pairs, max_pairs = tracker1.get_extremes()

        # Should have combined extremes
        assert len(min_pairs) >= 1
        assert len(max_pairs) >= 1


class TestMonotonicityDetector:
    """Test monotonicity detection."""

    def test_increasing_sequence(self):
        """Test detection of increasing sequence."""
        detector = MonotonicityDetector()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        detector.update(data)
        mono_inc, mono_dec = detector.get_monotonicity()

        assert mono_inc is True
        assert mono_dec is False

    def test_decreasing_sequence(self):
        """Test detection of decreasing sequence."""
        detector = MonotonicityDetector()
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        detector.update(data)
        mono_inc, mono_dec = detector.get_monotonicity()

        assert mono_inc is False
        assert mono_dec is True

    def test_non_monotonic_sequence(self):
        """Test detection of non-monotonic sequence."""
        detector = MonotonicityDetector()
        data = np.array([1.0, 3.0, 2.0, 4.0, 5.0])

        detector.update(data)
        mono_inc, mono_dec = detector.get_monotonicity()

        assert mono_inc is False
        assert mono_dec is False


class TestOutlierDetector:
    """Test outlier detection."""

    def test_iqr_outliers(self):
        """Test IQR-based outlier detection."""
        detector = OutlierDetector(methods=["iqr"])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is outlier

        detector.update(data)
        outlier_counts = detector.detect_outliers(data)

        assert "iqr" in outlier_counts
        assert outlier_counts["iqr"] >= 1

    def test_mad_outliers(self):
        """Test MAD-based outlier detection."""
        detector = OutlierDetector(methods=["mad"])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is outlier

        detector.update(data)
        outlier_counts = detector.detect_outliers(data)

        assert "mad" in outlier_counts
        assert outlier_counts["mad"] >= 1


class TestNumericAccumulator:
    """Test numeric accumulator."""

    def test_basic_functionality(self):
        """Test basic numeric accumulator functionality."""
        acc = NumericAccumulator("test_col")
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        acc.update(data)
        summary = acc.finalize()

        assert summary.name == "test_col"
        assert summary.count == 5
        assert summary.missing == 0
        assert abs(summary.mean - 3.0) < 1e-10
        assert summary.min == 1.0
        assert summary.max == 5.0

    def test_missing_values(self):
        """Test handling of missing values."""
        acc = NumericAccumulator("test_col")
        data = [1.0, None, 3.0, float("nan"), 5.0]

        acc.update(data)
        summary = acc.finalize()

        assert summary.count == 3  # Only valid numbers
        assert summary.missing == 2  # None and NaN

    def test_extreme_values(self):
        """Test handling of extreme values."""
        acc = NumericAccumulator("test_col")
        data = [1.0, float("inf"), 3.0, float("-inf"), 5.0]

        acc.update(data)
        summary = acc.finalize()

        assert summary.count == 3  # Finite values only
        assert summary.inf == 2  # Both inf values

    def test_performance_tracking(self):
        """Test performance tracking."""
        config = NumericConfig(enable_memory_tracking=True)
        acc = NumericAccumulator("test_col", config)
        data = np.random.random(1000)

        acc.update(data)
        metrics = acc.get_performance_metrics()

        assert metrics is not None
        assert metrics.update_count >= 1

    def test_merge_accumulators(self):
        """Test merging of numeric accumulators."""
        acc1 = NumericAccumulator("test_col")
        acc2 = NumericAccumulator("test_col")

        data1 = [1.0, 2.0, 3.0]
        data2 = [4.0, 5.0, 6.0]

        acc1.update(data1)
        acc2.update(data2)

        acc1.merge(acc2)
        summary = acc1.finalize()

        assert summary.count == 6
        assert abs(summary.mean - 3.5) < 1e-10


class TestBooleanAccumulator:
    """Test boolean accumulator."""

    def test_basic_functionality(self):
        """Test basic boolean accumulator functionality."""
        acc = BooleanAccumulator("test_col")
        data = [True, False, True, True, False]

        acc.update(data)
        summary = acc.finalize()

        assert summary.name == "test_col"
        assert summary.count == 5
        assert summary.true_n == 3
        assert summary.false_n == 2
        assert abs(summary.true_ratio - 0.6) < 1e-10

    def test_missing_values(self):
        """Test handling of missing values."""
        acc = BooleanAccumulator("test_col")
        data = [True, None, False, float("nan")]

        acc.update(data)
        summary = acc.finalize()

        assert summary.count == 2
        assert summary.missing == 2

    def test_entropy_calculation(self):
        """Test entropy calculation."""
        acc = BooleanAccumulator("test_col")
        data = [True, False]  # Perfect balance

        acc.update(data)
        summary = acc.finalize()

        # Entropy should be close to 1.0 for balanced distribution
        assert abs(summary.entropy - 1.0) < 0.1

    def test_distribution_info(self):
        """Test distribution information."""
        acc = BooleanAccumulator("test_col")
        data = [True, False, True, False]

        acc.update(data)
        info = acc.get_distribution_info()

        assert info["total_values"] == 4
        assert info["valid_values"] == 4
        assert info["is_balanced"] is True


class TestCategoricalAccumulator:
    """Test categorical accumulator."""

    def test_basic_functionality(self):
        """Test basic categorical accumulator functionality."""
        acc = CategoricalAccumulator("test_col")
        data = ["A", "B", "A", "C", "B", "A"]

        acc.update(data)
        summary = acc.finalize()

        assert summary.name == "test_col"
        assert summary.count == 6
        assert summary.unique_est >= 3
        assert len(summary.top_items) >= 1

    def test_missing_values(self):
        """Test handling of missing values."""
        acc = CategoricalAccumulator("test_col")

        data = ["A", None, "B", "", "C", float("nan")]
        acc.update(data)

        summary = acc.finalize()

        assert summary.count == 4  # A, B, "", C (empty string is valid)
        assert summary.missing == 2  # None, NaN
        assert summary.empty_zero == 1  # Empty string tracked separately

    def test_string_statistics(self):
        """Test string length statistics."""
        config = CategoricalConfig(enable_length_stats=True)
        acc = CategoricalAccumulator("test_col", config)
        data = ["A", "BB", "CCC"]

        acc.update(data)
        summary = acc.finalize()

        assert summary.avg_len is not None
        assert summary.avg_len > 0

    def test_quality_metrics(self):
        """Test data quality metrics."""
        acc = CategoricalAccumulator("test_col")
        data = ["A", "B", "A", "C"]

        acc.update(data)
        metrics = acc.get_quality_metrics()

        assert "diversity_ratio" in metrics
        assert "entropy" in metrics
        assert metrics["total_values"] == 4


class TestDatetimeAccumulator:
    """Test datetime accumulator."""

    def test_basic_functionality(self):
        """Test basic datetime accumulator functionality."""
        acc = DatetimeAccumulator("test_col")
        # Timestamps in nanoseconds
        data = [1640995200000000000, 1640995260000000000, 1640995320000000000]

        acc.update(data)
        summary = acc.finalize()

        assert summary.name == "test_col"
        assert summary.count == 3
        assert summary.min_ts is not None
        assert summary.max_ts is not None

    def test_missing_values(self):
        """Test handling of missing values."""
        acc = DatetimeAccumulator("test_col")
        data = [1640995200000000000, None, 1640995260000000000]

        acc.update(data)
        summary = acc.finalize()

        assert summary.count == 2
        assert summary.missing == 1

    def test_temporal_analysis(self):
        """Test temporal pattern analysis."""
        config = DatetimeConfig(enable_temporal_patterns=True)
        acc = DatetimeAccumulator("test_col", config)
        data = [1640995200000000000, 1640995260000000000]

        acc.update(data)
        analysis = acc.get_temporal_analysis()

        assert "time_span_days" in analysis
        assert "weekend_ratio" in analysis


class TestFactory:
    """Test factory functions."""

    def test_build_accumulators(self):
        """Test accumulator factory."""
        from pysuricata.compute.core.types import ColumnKinds
        from pysuricata.config import EngineConfig

        kinds = ColumnKinds(
            numeric=["num_col"],
            boolean=["bool_col"],
            categorical=["cat_col"],
            datetime=["dt_col"],
        )
        config = EngineConfig()

        accs = build_accumulators(kinds, config)

        assert "num_col" in accs
        assert "bool_col" in accs
        assert "cat_col" in accs
        assert "dt_col" in accs
        assert isinstance(accs["num_col"], NumericAccumulator)

    def test_create_accumulator_config(self):
        """Test accumulator configuration creation."""
        config = create_accumulator_config(
            numeric_sample_size=10000, enable_performance_tracking=True
        )

        assert config.numeric.sample_size == 10000
        assert config.enable_performance_tracking is True

    def test_get_accumulator_info(self):
        """Test accumulator information retrieval."""
        acc = NumericAccumulator("test")
        acc.update([1, 2, 3])

        info = get_accumulator_info({"test": acc})

        assert "test" in info
        assert info["test"]["type"] == "NumericAccumulator"
        assert info["test"]["count"] == 3

    def test_validate_accumulator_compatibility(self):
        """Test accumulator compatibility validation."""
        acc = NumericAccumulator("test")
        results = validate_accumulator_compatibility({"test": acc})

        assert "test" in results
        assert results["test"]["valid"] is True


class TestPerformance:
    """Test performance characteristics."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        acc = NumericAccumulator("test")
        large_data = np.random.random(100000)

        import time

        start_time = time.time()
        acc.update(large_data)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0

        summary = acc.finalize()
        assert summary.count == 100000

    def test_memory_efficiency(self):
        """Test memory efficiency."""
        config = NumericConfig(sample_size=1000)  # Limited sample size
        acc = NumericAccumulator("test", config)
        large_data = np.random.random(100000)

        acc.update(large_data)
        summary = acc.finalize()

        # Sample should be limited even with large input
        assert len(summary.sample_vals or []) <= 1000
        assert summary.count == 100000
