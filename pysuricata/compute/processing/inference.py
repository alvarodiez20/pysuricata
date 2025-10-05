"""Unified type inference for pandas and polars.

This module provides intelligent type inference capabilities for both
pandas and polars backends, with confidence scoring and fallback strategies.
"""

from __future__ import annotations

import logging
import re
import warnings
from enum import Enum
from typing import Any, Dict, Optional

from ..core.types import ColumnKinds, InferenceResult, ProcessingResult

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


class InferenceStrategy(Enum):
    """Strategy for type inference operations."""

    CONSERVATIVE = "conservative"  # Conservative inference with high confidence
    AGGRESSIVE = "aggressive"  # Aggressive inference with lower confidence
    BALANCED = "balanced"  # Balanced approach
    FAST = "fast"  # Fast inference with minimal analysis


class UnifiedTypeInferrer:
    """Unified type inference for pandas and polars.

    This class provides intelligent type inference capabilities for both
    pandas and polars backends, with confidence scoring and fallback strategies.
    It supports different inference strategies and provides detailed results.

    Attributes:
        strategy: Inference strategy to use.
        logger: Logger for inference operations.
        confidence_threshold: Minimum confidence threshold for inference.
        sample_size: Sample size for inference analysis.
    """

    def __init__(
        self,
        strategy: InferenceStrategy = InferenceStrategy.BALANCED,
        logger: Optional[logging.Logger] = None,
        confidence_threshold: float = 0.8,
        sample_size: int = 10000,
    ):
        """Initialize the unified type inferrer.

        Args:
            strategy: Inference strategy to use.
            logger: Logger for inference operations.
            confidence_threshold: Minimum confidence threshold for inference.
            sample_size: Sample size for inference analysis.
        """
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.sample_size = sample_size

        self._inference_cache: Dict[str, InferenceResult] = {}
        self._type_patterns = self._initialize_type_patterns()

    def infer_kinds(self, data: Any) -> ProcessingResult[ColumnKinds]:
        """Infer column types from data.

        Args:
            data: Input data to analyze.

        Returns:
            ProcessingResult containing the inferred ColumnKinds.
        """
        try:
            if isinstance(data, pd.DataFrame):
                return self._infer_pandas(data)
            elif isinstance(data, pl.DataFrame):
                return self._infer_polars(data)
            else:
                return ProcessingResult.error_result(
                    f"Unsupported data type: {type(data)}"
                )
        except Exception as e:
            return ProcessingResult.error_result(f"Inference failed: {str(e)}")

    def infer_series_type(self, series: Any) -> ProcessingResult[str]:
        """Infer type of a single data series.

        Args:
            series: Data series to analyze.

        Returns:
            ProcessingResult containing the inferred type string.
        """
        try:
            if isinstance(series, pd.Series):
                return self._infer_pandas_series_type(series)
            elif isinstance(series, pl.Series):
                return self._infer_polars_series_type(series)
            else:
                return ProcessingResult.error_result(
                    f"Unsupported series type: {type(series)}"
                )
        except Exception as e:
            return ProcessingResult.error_result(f"Series inference failed: {str(e)}")

    def _infer_pandas(self, df: pd.DataFrame) -> ProcessingResult[ColumnKinds]:
        """Infer types from pandas DataFrame.

        Args:
            df: Pandas DataFrame to analyze.

        Returns:
            ProcessingResult containing the inferred ColumnKinds.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:
            kinds = ColumnKinds()
            warnings_list = []
            errors_list = []
            total_confidence = 0.0
            column_count = 0

            for name, series in df.items():
                try:
                    result = self._infer_pandas_series_type(series)
                    if result.success:
                        kind = result.data
                        getattr(kinds, kind).append(name)
                        total_confidence += (
                            1.0  # Assume full confidence for successful inference
                        )
                    else:
                        # Fallback to categorical for failed inference
                        kinds.categorical.append(name)
                        warnings_list.append(f"Column '{name}': {result.error}")
                        total_confidence += 0.5  # Lower confidence for fallback

                    column_count += 1

                except Exception as e:
                    kinds.categorical.append(name)
                    errors_list.append(f"Column '{name}': {str(e)}")
                    column_count += 1

            # Calculate overall confidence
            overall_confidence = total_confidence / max(column_count, 1)

            return ProcessingResult.success_result(
                data=kinds,
                metrics={
                    "confidence": overall_confidence,
                    "warnings": len(warnings_list),
                    "errors": len(errors_list),
                    "strategy": self.strategy.value,
                },
            )

        except Exception as e:
            return ProcessingResult.error_result(f"Pandas inference failed: {str(e)}")

    def _infer_polars(self, df: pl.DataFrame) -> ProcessingResult[ColumnKinds]:
        """Infer types from polars DataFrame.

        Args:
            df: Polars DataFrame to analyze.

        Returns:
            ProcessingResult containing the inferred ColumnKinds.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            kinds = ColumnKinds()
            warnings_list = []
            errors_list = []
            total_confidence = 0.0
            column_count = 0

            for name in df.columns:
                try:
                    series = df[name]
                    result = self._infer_polars_series_type(series)
                    if result.success:
                        kind = result.data
                        getattr(kinds, kind).append(name)
                        total_confidence += 1.0
                    else:
                        kinds.categorical.append(name)
                        warnings_list.append(f"Column '{name}': {result.error}")
                        total_confidence += 0.5

                    column_count += 1

                except Exception as e:
                    kinds.categorical.append(name)
                    errors_list.append(f"Column '{name}': {str(e)}")
                    column_count += 1

            overall_confidence = total_confidence / max(column_count, 1)

            return ProcessingResult.success_result(
                data=kinds,
                metrics={
                    "confidence": overall_confidence,
                    "warnings": len(warnings_list),
                    "errors": len(errors_list),
                    "strategy": self.strategy.value,
                },
            )

        except Exception as e:
            return ProcessingResult.error_result(f"Polars inference failed: {str(e)}")

    def _infer_pandas_series_type(self, s: pd.Series) -> ProcessingResult[str]:
        """Infer pandas series type with confidence scoring.

        Args:
            s: Pandas series to analyze.

        Returns:
            ProcessingResult containing the inferred type.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:
            dtype_str = str(s.dtype)

            # Fast path for explicit types
            if pd.api.types.is_numeric_dtype(s.dtype):
                return ProcessingResult.success_result("numeric")
            elif pd.api.types.is_bool_dtype(s.dtype):
                return ProcessingResult.success_result("boolean")
            elif pd.api.types.is_datetime64_any_dtype(s.dtype):
                return ProcessingResult.success_result("datetime")

            # Pattern-based inference
            if re.search("int|float|^UInt|^Int|^Float", dtype_str, re.I):
                return ProcessingResult.success_result("numeric")
            elif re.search("bool", dtype_str, re.I):
                return ProcessingResult.success_result("boolean")
            elif re.search("datetime", dtype_str, re.I):
                return ProcessingResult.success_result("datetime")

            # Sample-based inference for object types
            if self.strategy in [
                InferenceStrategy.AGGRESSIVE,
                InferenceStrategy.BALANCED,
            ]:
                return self._sample_based_inference_pandas(s)
            else:
                return ProcessingResult.success_result("categorical")

        except Exception as e:
            return ProcessingResult.error_result(
                f"Pandas series inference failed: {str(e)}"
            )

    def _infer_polars_series_type(self, s: pl.Series) -> ProcessingResult[str]:
        """Infer polars series type with confidence scoring.

        Args:
            s: Polars series to analyze.

        Returns:
            ProcessingResult containing the inferred type.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            dtype = s.dtype

            # Fast path for explicit types
            if dtype in [
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
                pl.UInt64,
                pl.UInt32,
            ]:
                return ProcessingResult.success_result("numeric")
            elif dtype == pl.Boolean:
                return ProcessingResult.success_result("boolean")
            elif dtype in [pl.Datetime, pl.Date]:
                return ProcessingResult.success_result("datetime")

            # For string types, try to infer more specific types
            if dtype == pl.Utf8:
                if self.strategy in [
                    InferenceStrategy.AGGRESSIVE,
                    InferenceStrategy.BALANCED,
                ]:
                    return self._sample_based_inference_polars(s)
                else:
                    return ProcessingResult.success_result("categorical")

            # Default to categorical
            return ProcessingResult.success_result("categorical")

        except Exception as e:
            return ProcessingResult.error_result(
                f"Polars series inference failed: {str(e)}"
            )

    def _sample_based_inference_pandas(self, s: pd.Series) -> ProcessingResult[str]:
        """Perform sample-based inference for pandas series.

        Args:
            s: Pandas series to analyze.

        Returns:
            ProcessingResult containing the inferred type.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:
            # Sample data for analysis
            sample_size = min(self.sample_size, len(s))
            sample = s.head(sample_size)

            # Try datetime conversion
            if self.strategy in [
                InferenceStrategy.AGGRESSIVE,
                InferenceStrategy.BALANCED,
            ]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        ds = pd.to_datetime(
                            sample, errors="coerce", utc=True, format="mixed"
                        )
                        if ds.notna().sum() / len(sample) > 0.8:  # 80% success rate
                            return ProcessingResult.success_result("datetime")
                    except Exception:
                        pass

            # Try numeric conversion
            try:
                ns = pd.to_numeric(sample, errors="coerce")
                if ns.notna().sum() / len(sample) > 0.8:  # 80% success rate
                    return ProcessingResult.success_result("numeric")
            except Exception:
                pass

            # Try boolean conversion
            if self.strategy == InferenceStrategy.AGGRESSIVE:
                try:
                    bs = (
                        sample.astype(str)
                        .str.lower()
                        .isin(["true", "false", "1", "0", "yes", "no"])
                    )
                    if bs.sum() / len(sample) > 0.8:  # 80% success rate
                        return ProcessingResult.success_result("boolean")
                except Exception:
                    pass

            # Default to categorical
            return ProcessingResult.success_result("categorical")

        except Exception as e:
            return ProcessingResult.error_result(
                f"Sample-based inference failed: {str(e)}"
            )

    def _sample_based_inference_polars(self, s: pl.Series) -> ProcessingResult[str]:
        """Perform sample-based inference for polars series.

        Args:
            s: Polars series to analyze.

        Returns:
            ProcessingResult containing the inferred type.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            # Sample data for analysis
            sample_size = min(self.sample_size, s.len())
            sample = s.head(sample_size)

            # Try datetime conversion
            if self.strategy in [
                InferenceStrategy.AGGRESSIVE,
                InferenceStrategy.BALANCED,
            ]:
                try:
                    # First try Date type (for date-only strings like '1914-12-01')
                    ds = sample.cast(pl.Date, strict=False)
                    null_count = ds.null_count()
                    if (
                        sample_size - null_count
                    ) / sample_size > 0.8:  # 80% success rate
                        return ProcessingResult.success_result("datetime")
                except Exception:
                    pass

                try:
                    # Then try Datetime type (for datetime strings with time components)
                    ds = sample.cast(pl.Datetime, strict=False)
                    null_count = ds.null_count()
                    if (
                        sample_size - null_count
                    ) / sample_size > 0.8:  # 80% success rate
                        return ProcessingResult.success_result("datetime")
                except Exception:
                    pass

            # Try numeric conversion
            try:
                ns = sample.cast(pl.Float64, strict=False)
                null_count = ns.null_count()
                if (sample_size - null_count) / sample_size > 0.8:  # 80% success rate
                    return ProcessingResult.success_result("numeric")
            except Exception:
                pass

            # Try boolean conversion
            if self.strategy == InferenceStrategy.AGGRESSIVE:
                try:
                    bs = sample.cast(pl.Boolean, strict=False)
                    null_count = bs.null_count()
                    if (
                        sample_size - null_count
                    ) / sample_size > 0.8:  # 80% success rate
                        return ProcessingResult.success_result("boolean")
                except Exception:
                    pass

            # Default to categorical
            return ProcessingResult.success_result("categorical")

        except Exception as e:
            return ProcessingResult.error_result(
                f"Sample-based inference failed: {str(e)}"
            )

    def _initialize_type_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for type detection.

        Returns:
            Dictionary of compiled regex patterns.
        """
        return {
            "numeric": re.compile(r"int|float|^UInt|^Int|^Float", re.I),
            "boolean": re.compile(r"bool", re.I),
            "datetime": re.compile(r"datetime", re.I),
        }

    def clear_cache(self) -> None:
        """Clear the inference cache."""
        self._inference_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        return {
            "cache_size": len(self._inference_cache),
            "strategy": self.strategy.value,
            "confidence_threshold": self.confidence_threshold,
            "sample_size": self.sample_size,
        }


def should_reclassify_numeric_as_categorical(
    unique_count: int, total_count: int
) -> bool:
    """Determine if a numeric column should be reclassified as categorical.

    This function implements a hybrid approach to reclassify numeric columns
    that have very few unique values as categorical columns. This is useful
    for columns that are technically numeric but represent discrete categories
    (like ratings, grades, or status codes).

    Args:
        unique_count: Number of unique values in the column
        total_count: Total number of values in the column

    Returns:
        True if column should be reclassified as categorical, False otherwise
    """
    if total_count == 0:
        return False

    # Calculate unique ratio
    unique_ratio = unique_count / total_count

    # Reclassify as categorical if:
    # 1. Less than 10 unique values, OR
    # 2. Less than 5% unique ratio (for very large datasets)
    return unique_count < 10 or unique_ratio < 0.05


def should_reclassify_numeric_as_boolean(
    series: Any, config: Any, logger: Optional[Any] = None
) -> bool:
    """Determine if a numeric column should be reclassified as boolean.

    This function implements conservative heuristics to detect numeric columns
    that contain only 0s and 1s and should be treated as boolean columns.
    This improves semantic accuracy and provides more relevant statistics.

    Args:
        series: Numeric series to analyze (pandas.Series or polars.Series)
        config: Configuration object with boolean detection settings
        logger: Optional logger for debugging

    Returns:
        True if column should be reclassified as boolean, False otherwise
    """
    if not config.enable_auto_boolean_detection:
        return False

    try:
        # Get unique values (handle both pandas and polars)
        if hasattr(series, "dropna"):
            # Pandas
            unique_values = set(series.dropna().unique())
            total_count = len(series.dropna())
        else:
            # Polars
            unique_values = set(series.drop_nulls().unique().to_list())
            total_count = series.drop_nulls().len()

        # Must have sufficient samples
        if total_count < config.boolean_detection_min_samples:
            if logger:
                logger.debug(
                    "Boolean detection skipped for '%s': insufficient samples (%d < %d)",
                    getattr(series, "name", "unknown"),
                    total_count,
                    config.boolean_detection_min_samples,
                )
            return False

        # Must contain exactly 0 and 1 (handle numpy types)
        unique_ints = {int(v) for v in unique_values}
        if not unique_ints.issubset({0, 1}):
            return False

        # Must have both values present
        if len(unique_ints) != 2:
            return False

        # Check for reasonable distribution (not mostly zeros)
        if hasattr(series, "dropna"):
            # Pandas
            zero_count = (series == 0).sum()
        else:
            # Polars
            zero_count = (series == 0).sum()

        zero_ratio = zero_count / total_count
        if zero_ratio > config.boolean_detection_max_zero_ratio:
            if logger:
                logger.debug(
                    "Boolean detection skipped for '%s': too many zeros (%.2f > %.2f)",
                    getattr(series, "name", "unknown"),
                    zero_ratio,
                    config.boolean_detection_max_zero_ratio,
                )
            return False

        # Check column name patterns if required
        if config.boolean_detection_require_name_pattern:
            column_name = getattr(series, "name", "").lower()
            boolean_patterns = [
                "is_",
                "has_",
                "can_",
                "should_",
                "flag_",
                "active",
                "enabled",
                "valid",
                "complete",
                "success",
                "failed",
                "error",
                "warning",
                "true",
                "false",
                "yes",
                "no",
                "on",
                "off",
            ]

            if not any(pattern in column_name for pattern in boolean_patterns):
                if logger:
                    logger.debug(
                        "Boolean detection skipped for '%s': no boolean-like name pattern",
                        getattr(series, "name", "unknown"),
                    )
                return False

        return True

    except Exception as e:
        if logger:
            logger.warning(
                "Boolean detection failed for '%s': %s",
                getattr(series, "name", "unknown"),
                str(e),
            )
        return False
