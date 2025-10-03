"""Tests for automatic boolean detection functionality.

This module tests the new 0/1 → boolean classification feature with comprehensive
test cases covering edge cases, configuration options, and integration scenarios.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
except ImportError:
    pl = None

from pysuricata import profile
from pysuricata.api import ComputeOptions, ProfileConfig
from pysuricata.compute.processing.inference import should_reclassify_numeric_as_boolean


class TestBooleanDetection:
    """Test suite for boolean detection functionality."""

    def test_should_reclassify_numeric_as_boolean_basic(self):
        """Test basic boolean detection with 0/1 values."""
        # Create test data with enough samples
        series = pd.Series([0, 1, 0, 1, 1, 0] * 20, name="is_active")  # 120 samples
        config = ComputeOptions(
            enable_auto_boolean_detection=True,
            boolean_detection_min_samples=100,
            boolean_detection_max_zero_ratio=0.8,
            boolean_detection_require_name_pattern=True,
        )

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is True

    def test_should_reclassify_numeric_as_boolean_disabled(self):
        """Test that boolean detection can be disabled."""
        series = pd.Series([0, 1, 0, 1, 1, 0], name="is_active")
        config = ComputeOptions(enable_auto_boolean_detection=False)

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_insufficient_samples(self):
        """Test boolean detection with insufficient samples."""
        series = pd.Series([0, 1, 0], name="is_active")
        config = ComputeOptions(
            enable_auto_boolean_detection=True, boolean_detection_min_samples=10
        )

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_too_many_zeros(self):
        """Test boolean detection with too many zeros."""
        series = pd.Series([0, 0, 0, 0, 0, 1], name="is_active")
        config = ComputeOptions(
            enable_auto_boolean_detection=True, boolean_detection_max_zero_ratio=0.5
        )

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_no_name_pattern(self):
        """Test boolean detection without boolean-like name patterns."""
        series = pd.Series([0, 1, 0, 1, 1, 0], name="user_id")
        config = ComputeOptions(
            enable_auto_boolean_detection=True,
            boolean_detection_require_name_pattern=True,
        )

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_without_name_pattern_requirement(
        self,
    ):
        """Test boolean detection without requiring name patterns."""
        series = pd.Series([0, 1, 0, 1, 1, 0] * 20, name="user_id")  # 120 samples
        config = ComputeOptions(
            enable_auto_boolean_detection=True,
            boolean_detection_require_name_pattern=False,
        )

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is True

    def test_should_reclassify_numeric_as_boolean_non_binary_values(self):
        """Test boolean detection with non-binary values."""
        series = pd.Series([0, 1, 2, 0, 1], name="is_active")
        config = ComputeOptions(enable_auto_boolean_detection=True)

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_only_zeros(self):
        """Test boolean detection with only zeros."""
        series = pd.Series([0, 0, 0, 0, 0], name="is_active")
        config = ComputeOptions(enable_auto_boolean_detection=True)

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_only_ones(self):
        """Test boolean detection with only ones."""
        series = pd.Series([1, 1, 1, 1, 1], name="is_active")
        config = ComputeOptions(enable_auto_boolean_detection=True)

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is False

    def test_should_reclassify_numeric_as_boolean_with_nulls(self):
        """Test boolean detection with null values."""
        series = pd.Series([0, 1, 0, None, 1, 0] * 20, name="is_active")  # 120 samples
        config = ComputeOptions(enable_auto_boolean_detection=True)

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is True

    def test_should_reclassify_numeric_as_boolean_boolean_name_patterns(self):
        """Test various boolean-like name patterns."""
        boolean_patterns = [
            "is_active",
            "has_permission",
            "can_edit",
            "should_validate",
            "flag_enabled",
            "user_active",
            "feature_enabled",
            "data_valid",
            "process_complete",
            "operation_success",
            "task_failed",
            "error_occurred",
            "warning_shown",
            "status_true",
            "result_false",
            "option_yes",
            "choice_no",
            "toggle_on",
            "switch_off",
        ]

        config = ComputeOptions(
            enable_auto_boolean_detection=True,
            boolean_detection_require_name_pattern=True,
        )

        for pattern in boolean_patterns:
            series = pd.Series([0, 1, 0, 1] * 30, name=pattern)  # 120 samples
            result = should_reclassify_numeric_as_boolean(series, config)
            assert result is True, f"Failed for pattern: {pattern}"

    @pytest.mark.skipif(pl is None, reason="polars not available")
    def test_should_reclassify_numeric_as_boolean_polars(self):
        """Test boolean detection with polars Series."""
        series = pl.Series("is_active", [0, 1, 0, 1, 1, 0] * 20)  # 120 samples
        config = ComputeOptions(enable_auto_boolean_detection=True)

        result = should_reclassify_numeric_as_boolean(series, config)
        assert result is True

    def test_should_reclassify_numeric_as_boolean_error_handling(self):
        """Test error handling in boolean detection."""
        # Test with invalid series
        series = Mock()
        series.dropna.side_effect = Exception("Test error")
        series.name = "test_column"

        config = ComputeOptions(enable_auto_boolean_detection=True)
        logger = Mock()

        result = should_reclassify_numeric_as_boolean(series, config, logger)
        assert result is False
        logger.warning.assert_called_once()


class TestBooleanDetectionIntegration:
    """Integration tests for boolean detection with profile function."""

    def test_profile_with_boolean_detection_pandas(self):
        """Test profile function with boolean detection on pandas DataFrame."""
        df = pd.DataFrame(
            {
                "is_active": [1, 0, 1, 1, 0, 1, 0, 0] * 20,  # 160 samples
                "has_permission": [0, 1, 0, 1, 1, 0, 1, 0] * 20,  # 160 samples
                "user_id": [1, 2, 3, 4, 5, 6, 7, 8] * 20,  # 160 samples
                "rating": [1, 2, 3, 4, 5, 1, 2, 3] * 20,  # 160 samples
            }
        )

        config = ProfileConfig(
            compute=ComputeOptions(
                enable_auto_boolean_detection=True,
                boolean_detection_require_name_pattern=True,
            )
        )

        report = profile(df, config)

        # Check that boolean columns were detected
        assert report.stats["columns"]["is_active"]["type"] == "boolean"
        assert report.stats["columns"]["has_permission"]["type"] == "boolean"
        # user_id has 8 unique values in 160 samples (5% ratio) -> categorical
        assert report.stats["columns"]["user_id"]["type"] == "categorical"
        # rating has 5 unique values in 160 samples -> categorical
        assert report.stats["columns"]["rating"]["type"] == "categorical"

    @pytest.mark.skipif(pl is None, reason="polars not available")
    def test_profile_with_boolean_detection_polars(self):
        """Test profile function with boolean detection on polars DataFrame."""
        df = pl.DataFrame(
            {
                "is_active": [1, 0, 1, 1, 0, 1, 0, 0] * 20,  # 160 samples
                "has_permission": [0, 1, 0, 1, 1, 0, 1, 0] * 20,  # 160 samples
                "user_id": [1, 2, 3, 4, 5, 6, 7, 8] * 20,  # 160 samples
                "rating": [1, 2, 3, 4, 5, 1, 2, 3] * 20,  # 160 samples
            }
        )

        config = ProfileConfig(
            compute=ComputeOptions(
                enable_auto_boolean_detection=True,
                boolean_detection_require_name_pattern=True,
            )
        )

        report = profile(df, config)

        # Check that boolean columns were detected
        assert report.stats["columns"]["is_active"]["type"] == "boolean"
        assert report.stats["columns"]["has_permission"]["type"] == "boolean"
        # user_id has 8 unique values in 160 samples (5% ratio) -> categorical
        assert report.stats["columns"]["user_id"]["type"] == "categorical"
        # rating has 5 unique values in 160 samples -> categorical
        assert report.stats["columns"]["rating"]["type"] == "categorical"

    def test_profile_with_force_column_types(self):
        """Test profile function with forced column types."""
        df = pd.DataFrame(
            {
                "is_active": [1, 0, 1, 1, 0, 1, 0, 0]
                * 20,  # 160 samples - will be auto-detected as boolean
                "user_id": list(range(1, 161)),  # 160 unique values - will stay numeric
                "rating": [1, 2, 3, 4, 5, 1, 2, 3]
                * 20,  # 160 samples - will be categorical
            }
        )

        config = ProfileConfig(
            compute=ComputeOptions(
                enable_auto_boolean_detection=True,
                force_column_types={
                    "user_id": "categorical",  # Force numeric to categorical
                    "rating": "boolean",  # Force categorical to boolean
                },
            )
        )

        report = profile(df, config)

        # Check that forced types were applied
        assert (
            report.stats["columns"]["is_active"]["type"] == "boolean"
        )  # Auto-detected
        assert report.stats["columns"]["user_id"]["type"] == "categorical"  # Forced
        assert report.stats["columns"]["rating"]["type"] == "boolean"  # Forced

    def test_profile_with_boolean_detection_disabled(self):
        """Test profile function with boolean detection disabled."""
        df = pd.DataFrame(
            {
                "is_active": [1, 0, 1, 1, 0, 1, 0, 0] * 20,  # 160 samples
                "has_permission": [0, 1, 0, 1, 1, 0, 1, 0] * 20,  # 160 samples
                "user_id": [1, 2, 3, 4, 5, 6, 7, 8] * 20,  # 160 samples
            }
        )

        config = ProfileConfig(
            compute=ComputeOptions(enable_auto_boolean_detection=False)
        )

        report = profile(df, config)

        # Check that boolean columns were NOT detected (but still reclassified as categorical)
        assert (
            report.stats["columns"]["is_active"]["type"] == "categorical"
        )  # 2 unique values -> categorical
        assert (
            report.stats["columns"]["has_permission"]["type"] == "categorical"
        )  # 2 unique values -> categorical
        assert (
            report.stats["columns"]["user_id"]["type"] == "categorical"
        )  # 8 unique values -> categorical

    def test_profile_with_mixed_data_types(self):
        """Test profile function with mixed data types including edge cases."""
        df = pd.DataFrame(
            {
                "is_active": [1, 0, 1, 1, 0, 1, 0, 0] * 20,  # Should be boolean
                "binary_feature": [0, 1, 0, 1, 0, 1, 0, 1]
                * 20,  # Should be boolean (no name pattern required)
                "user_id": [1, 2, 3, 4, 5, 6, 7, 8] * 20,  # Should stay numeric
                "mostly_zeros": [0, 0, 0, 0, 0, 0, 0, 1]
                * 20,  # Should stay numeric (too many zeros)
                "only_ones": [1, 1, 1, 1, 1, 1, 1, 1]
                * 20,  # Should stay numeric (no zeros)
                "mixed_values": [0, 1, 2, 0, 1, 2, 0, 1]
                * 20,  # Should stay numeric (not binary)
                "small_sample": ([0, 1, 0] * 53)
                + [0],  # Should stay numeric (too small)
            }
        )

        config = ProfileConfig(
            compute=ComputeOptions(
                enable_auto_boolean_detection=True,
                boolean_detection_require_name_pattern=False,
                boolean_detection_min_samples=5,
                boolean_detection_max_zero_ratio=0.8,
            )
        )

        report = profile(df, config)

        # Check classifications
        assert report.stats["columns"]["is_active"]["type"] == "boolean"
        assert report.stats["columns"]["binary_feature"]["type"] == "boolean"
        assert (
            report.stats["columns"]["user_id"]["type"] == "categorical"
        )  # 8 unique values -> categorical
        assert (
            report.stats["columns"]["mostly_zeros"]["type"] == "categorical"
        )  # 2 unique values -> categorical
        assert (
            report.stats["columns"]["only_ones"]["type"] == "categorical"
        )  # 1 unique value -> categorical
        assert (
            report.stats["columns"]["mixed_values"]["type"] == "categorical"
        )  # 3 unique values -> categorical
        assert (
            report.stats["columns"]["small_sample"]["type"] == "boolean"
        )  # 2 unique values (0,1) -> boolean


class TestBooleanDetectionConfiguration:
    """Test configuration options for boolean detection."""

    def test_compute_options_validation(self):
        """Test validation of ComputeOptions parameters."""
        # Test valid configuration
        config = ComputeOptions(
            enable_auto_boolean_detection=True,
            boolean_detection_min_samples=100,
            boolean_detection_max_zero_ratio=0.95,
            boolean_detection_require_name_pattern=True,
            force_column_types={"col1": "boolean", "col2": "numeric"},
        )
        assert config.enable_auto_boolean_detection is True
        assert config.boolean_detection_min_samples == 100
        assert config.boolean_detection_max_zero_ratio == 0.95

    def test_compute_options_validation_errors(self):
        """Test validation errors for invalid ComputeOptions parameters."""
        # Test invalid min_samples
        with pytest.raises(
            ValueError, match="boolean_detection_min_samples must be positive"
        ):
            ComputeOptions(boolean_detection_min_samples=0)

        # Test invalid max_zero_ratio
        with pytest.raises(
            ValueError, match="boolean_detection_max_zero_ratio must be between 0 and 1"
        ):
            ComputeOptions(boolean_detection_max_zero_ratio=1.5)

        # Test invalid column type
        with pytest.raises(ValueError, match="Invalid column type"):
            ComputeOptions(force_column_types={"col1": "invalid_type"})

    def test_force_column_types_validation(self):
        """Test validation of force_column_types parameter."""
        # Test valid types
        valid_types = ["numeric", "categorical", "datetime", "boolean"]
        for valid_type in valid_types:
            config = ComputeOptions(force_column_types={"col1": valid_type})
            assert config.force_column_types["col1"] == valid_type

        # Test invalid type
        with pytest.raises(ValueError):
            ComputeOptions(force_column_types={"col1": "invalid"})


if __name__ == "__main__":
    pytest.main([__file__])
