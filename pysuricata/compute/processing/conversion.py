"""Unified data conversion for pandas and polars.

This module provides optimized, zero-copy data conversion capabilities
for both pandas and polars backends, with fallback strategies for
unsupported data types.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np

from ..core.exceptions import ConversionError
from ..core.types import ProcessingResult

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


class ConversionStrategy(Enum):
    """Strategy for data conversion operations."""

    ZERO_COPY = "zero_copy"  # Prefer zero-copy operations
    SAFE = "safe"  # Safe conversion with error handling
    FAST = "fast"  # Fast conversion with minimal checks


class UnifiedConverter:
    """Unified data conversion for pandas and polars.

    This class provides optimized data conversion capabilities for both
    pandas and polars backends, with intelligent fallback strategies
    and zero-copy operations where possible.

    Attributes:
        strategy: Conversion strategy to use.
        logger: Logger for conversion operations.
    """

    def __init__(
        self,
        strategy: ConversionStrategy = ConversionStrategy.ZERO_COPY,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the unified converter.

        Args:
            strategy: Conversion strategy to use.
            logger: Logger for conversion operations.
        """
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self._conversion_cache: dict = {}

    def to_numeric(
        self, series: Any, target_dtype: str = "float64"
    ) -> ProcessingResult[np.ndarray]:
        """Convert series to numeric with zero-copy when possible.

        Args:
            series: Data series to convert.
            target_dtype: Target numpy dtype.

        Returns:
            ProcessingResult containing the converted numeric array.
        """
        try:
            if isinstance(series, pd.Series):
                return self._pandas_to_numeric(series, target_dtype)
            elif isinstance(series, pl.Series):
                return self._polars_to_numeric(series, target_dtype)
            else:
                return ProcessingResult.error_result(
                    f"Unsupported series type: {type(series)}"
                )
        except Exception as e:
            return ProcessingResult.error_result(f"Conversion failed: {str(e)}")

    def to_boolean(self, series: Any) -> ProcessingResult[List[Optional[bool]]]:
        """Convert series to boolean values.

        Args:
            series: Data series to convert.

        Returns:
            ProcessingResult containing the converted boolean list.
        """
        try:
            if isinstance(series, pd.Series):
                return self._pandas_to_boolean(series)
            elif isinstance(series, pl.Series):
                return self._polars_to_boolean(series)
            else:
                return ProcessingResult.error_result(
                    f"Unsupported series type: {type(series)}"
                )
        except Exception as e:
            return ProcessingResult.error_result(f"Boolean conversion failed: {str(e)}")

    def to_datetime_ns(self, series: Any) -> ProcessingResult[List[Optional[int]]]:
        """Convert series to datetime nanoseconds.

        Args:
            series: Data series to convert.

        Returns:
            ProcessingResult containing the converted datetime nanoseconds.
        """
        try:
            if isinstance(series, pd.Series):
                return self._pandas_to_datetime_ns(series)
            elif isinstance(series, pl.Series):
                return self._polars_to_datetime_ns(series)
            else:
                return ProcessingResult.error_result(
                    f"Unsupported series type: {type(series)}"
                )
        except Exception as e:
            return ProcessingResult.error_result(
                f"Datetime conversion failed: {str(e)}"
            )

    def to_categorical_iter(self, series: Any) -> ProcessingResult[Any]:
        """Convert series to categorical iterator.

        Args:
            series: Data series to convert.

        Returns:
            ProcessingResult containing the categorical iterator.
        """
        try:
            if isinstance(series, pd.Series):
                return self._pandas_to_categorical_iter(series)
            elif isinstance(series, pl.Series):
                return self._polars_to_categorical_iter(series)
            else:
                return ProcessingResult.error_result(
                    f"Unsupported series type: {type(series)}"
                )
        except Exception as e:
            return ProcessingResult.error_result(
                f"Categorical conversion failed: {str(e)}"
            )

    def _pandas_to_numeric(
        self, s: pd.Series, target_dtype: str = "float64"
    ) -> ProcessingResult[np.ndarray]:
        """Convert pandas series to numeric with optimization.

        Args:
            s: Pandas series to convert.
            target_dtype: Target numpy dtype.

        Returns:
            ProcessingResult containing the numeric array.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:
            # Fast path for already numeric types
            if pd.api.types.is_numeric_dtype(s.dtype):
                if self.strategy == ConversionStrategy.ZERO_COPY:
                    # Try zero-copy conversion
                    try:
                        return ProcessingResult.success_result(
                            s.to_numpy(dtype=target_dtype, copy=False)
                        )
                    except (ValueError, TypeError):
                        # Fall back to copy if zero-copy fails
                        pass

                return ProcessingResult.success_result(
                    s.to_numpy(dtype=target_dtype, copy=True)
                )

            # Convert non-numeric to numeric
            if self.strategy == ConversionStrategy.SAFE:
                # Safe conversion with error handling
                numeric_series = pd.to_numeric(s, errors="coerce")
            else:
                # Fast conversion
                numeric_series = pd.to_numeric(s, errors="coerce")

            return ProcessingResult.success_result(
                numeric_series.to_numpy(dtype=target_dtype, copy=False)
            )

        except Exception as e:
            return ProcessingResult.error_result(
                f"Pandas numeric conversion failed: {str(e)}"
            )

    def _polars_to_numeric(
        self, s: pl.Series, target_dtype: str = "float64"
    ) -> ProcessingResult[np.ndarray]:
        """Convert polars series to numeric with optimization.

        Args:
            s: Polars series to convert.
            target_dtype: Target numpy dtype.

        Returns:
            ProcessingResult containing the numeric array.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            # Fast path for already numeric types
            if s.dtype in [
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
                pl.UInt64,
                pl.UInt32,
            ]:
                return ProcessingResult.success_result(s.to_numpy())

            # Convert non-numeric to numeric
            if self.strategy == ConversionStrategy.SAFE:
                # Safe conversion with error handling
                numeric_series = s.cast(pl.Float64, strict=False)
            else:
                # Fast conversion
                numeric_series = s.cast(pl.Float64, strict=False)

            return ProcessingResult.success_result(numeric_series.to_numpy())

        except Exception as e:
            return ProcessingResult.error_result(
                f"Polars numeric conversion failed: {str(e)}"
            )

    def _pandas_to_boolean(
        self, s: pd.Series
    ) -> ProcessingResult[List[Optional[bool]]]:
        """Convert pandas series to boolean values.

        Args:
            s: Pandas series to convert.

        Returns:
            ProcessingResult containing the boolean list.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:

            def _coerce(v: Any) -> Optional[bool]:
                if pd.isna(v):
                    return None
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    v_lower = v.lower()
                    if v_lower in ("true", "1", "yes", "on"):
                        return True
                    if v_lower in ("false", "0", "no", "off"):
                        return False
                return None

            result = [_coerce(v) for v in s]
            return ProcessingResult.success_result(result)

        except Exception as e:
            return ProcessingResult.error_result(
                f"Pandas boolean conversion failed: {str(e)}"
            )

    def _polars_to_boolean(
        self, s: pl.Series
    ) -> ProcessingResult[List[Optional[bool]]]:
        """Convert polars series to boolean values.

        Args:
            s: Polars series to convert.

        Returns:
            ProcessingResult containing the boolean list.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            # Fast path for already boolean
            if s.dtype == pl.Boolean:
                return ProcessingResult.success_result(s.to_list())

            # Convert to boolean with null handling
            boolean_series = s.cast(pl.Boolean, strict=False)
            return ProcessingResult.success_result(boolean_series.to_list())

        except Exception as e:
            return ProcessingResult.error_result(
                f"Polars boolean conversion failed: {str(e)}"
            )

    def _pandas_to_datetime_ns(
        self, s: pd.Series
    ) -> ProcessingResult[List[Optional[int]]]:
        """Convert pandas series to datetime nanoseconds.

        Args:
            s: Pandas series to convert.

        Returns:
            ProcessingResult containing the datetime nanoseconds.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:
            if pd.api.types.is_datetime64_any_dtype(s.dtype):
                # Already datetime, convert to nanoseconds
                return ProcessingResult.success_result(
                    [int(v) if pd.notna(v) else None for v in s.astype("int64")]
                )

            # Convert to datetime first
            dt_series = pd.to_datetime(s, errors="coerce", utc=True)
            return ProcessingResult.success_result(
                [int(v) if pd.notna(v) else None for v in dt_series.astype("int64")]
            )

        except Exception as e:
            return ProcessingResult.error_result(
                f"Pandas datetime conversion failed: {str(e)}"
            )

    def _polars_to_datetime_ns(
        self, s: pl.Series
    ) -> ProcessingResult[List[Optional[int]]]:
        """Convert polars series to datetime nanoseconds.

        Args:
            s: Polars series to convert.

        Returns:
            ProcessingResult containing the datetime nanoseconds.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            # Fast path for already datetime
            if s.dtype == pl.Datetime:
                return ProcessingResult.success_result(s.to_list())

            # Convert to datetime first
            dt_series = s.cast(pl.Datetime, strict=False)
            return ProcessingResult.success_result(dt_series.to_list())

        except Exception as e:
            return ProcessingResult.error_result(
                f"Polars datetime conversion failed: {str(e)}"
            )

    def _pandas_to_categorical_iter(self, s: pd.Series) -> ProcessingResult[Any]:
        """Convert pandas series to categorical iterator.

        Args:
            s: Pandas series to convert.

        Returns:
            ProcessingResult containing the categorical iterator.
        """
        if pd is None:
            return ProcessingResult.error_result("pandas not available")

        try:
            # Handle null values and convert to string
            categorical_series = s.astype(str).fillna("")
            return ProcessingResult.success_result(iter(categorical_series))

        except Exception as e:
            return ProcessingResult.error_result(
                f"Pandas categorical conversion failed: {str(e)}"
            )

    def _polars_to_categorical_iter(self, s: pl.Series) -> ProcessingResult[Any]:
        """Convert polars series to categorical iterator.

        Args:
            s: Polars series to convert.

        Returns:
            ProcessingResult containing the categorical iterator.
        """
        if pl is None:
            return ProcessingResult.error_result("polars not available")

        try:
            # Convert to string and handle nulls
            categorical_series = s.cast(pl.Utf8, strict=False).fill_null("")
            return ProcessingResult.success_result(iter(categorical_series))

        except Exception as e:
            return ProcessingResult.error_result(
                f"Polars categorical conversion failed: {str(e)}"
            )

    def clear_cache(self) -> None:
        """Clear the conversion cache."""
        self._conversion_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        return {
            "cache_size": len(self._conversion_cache),
            "strategy": self.strategy.value,
        }
