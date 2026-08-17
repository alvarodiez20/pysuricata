"""Unified type inference for pandas and polars.

This module provides intelligent type inference capabilities for both
pandas and polars backends, with confidence scoring and fallback strategies.
"""

from __future__ import annotations

import logging
import re
import warnings
from enum import Enum
from typing import Any

from ..core.types import ColumnKinds, InferenceResult, ProcessingResult
from .conversion import polars_string_to_datetime

# Date sniffing is a yes/no question, so it does not need a large sample.
_DATE_SNIFF_SAMPLE = 200
# Tried in order, most common first. Each takes pandas' vectorised parser;
# format="mixed" (the previous sole strategy) does not.
_DATE_SNIFF_FORMATS = (
    "ISO8601",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y%m%d",
)


#: Below this year, a "successful" parse cannot produce a usable datetime
#: column: the datetime accumulator's validity window is what `datetime64[ns]`
#: can represent, which starts at 1677-09-21, so every such value would be
#: recorded as missing anyway. Set well below that bound rather than at it, so
#: this stays a filter on parser artifacts and not a new opinion about which
#: historical dates count -- narrowing that window is exactly the mistake the
#: old `-2e18` bound made, and it silently nulled every 19th-century date.
_IMPLAUSIBLE_YEAR = 1000


try:
    import pandas as pd
except ImportError:
    pd = None


def _dated_fraction(parsed: Any) -> float:
    """The fraction of a probe that parsed to something that is really a date.

    `notna().mean()` on its own was the whole test, and it takes the parser's
    word for it. **pandas 3 reads `"T1"` as year 1** -- `T` is the ISO 8601
    time designator, so a bare identifier like a ticket number parses rather
    than failing. On a column of `T0..T680`, `format="mixed"` reports 99.5% of
    the probe as dates under pandas 3 and 34% under pandas 2, which is the
    difference between profiling an identifier column as datetime and as
    categorical.

    Requiring a plausible year keeps the check honest without pinning it to one
    pandas version: a parse landing before `_IMPLAUSIBLE_YEAR` carried no date
    to begin with. Nothing real is excluded -- a column of genuine dates is
    unaffected, and this is a yes/no question about the column's type, not a
    filter on the values, which are parsed for real later.
    """
    if len(parsed) == 0:
        return 0.0
    dated = parsed.notna()
    if not bool(dated.any()):
        return 0.0
    try:
        dated &= parsed.dt.year >= _IMPLAUSIBLE_YEAR
    except (AttributeError, TypeError, ValueError):
        # Not a datetime-like result; `notna` alone is the best available read.
        pass
    return float(dated.mean())


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
        logger: logging.Logger | None = None,
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

        self._inference_cache: dict[str, InferenceResult] = {}
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
            dtype = s.dtype
            dtype_str = str(dtype)

            # Fast path for explicit types
            if pd.api.types.is_bool_dtype(dtype):
                return ProcessingResult.success_result("boolean")
            elif pd.api.types.is_numeric_dtype(dtype):
                return ProcessingResult.success_result("numeric")
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                return ProcessingResult.success_result("datetime")
            elif pd.api.types.is_timedelta64_dtype(dtype):
                # Timedelta is fundamentally numeric (total seconds)
                return ProcessingResult.success_result("numeric")

            # ArrowDtype explicit handling (pyarrow-backed DataFrames)
            if hasattr(dtype, "pyarrow_dtype"):
                import pyarrow as pa

                pa_type = dtype.pyarrow_dtype
                if pa.types.is_boolean(pa_type):
                    return ProcessingResult.success_result("boolean")
                elif (
                    pa.types.is_integer(pa_type)
                    or pa.types.is_floating(pa_type)
                    or pa.types.is_decimal(pa_type)
                ):
                    return ProcessingResult.success_result("numeric")
                elif pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
                    return ProcessingResult.success_result("datetime")
                elif pa.types.is_duration(pa_type):
                    return ProcessingResult.success_result("numeric")
                else:
                    return ProcessingResult.success_result("categorical")

            # Pattern-based inference for remaining string-based dtype detection
            if re.search(r"int|float|^UInt|^Int|^Float", dtype_str, re.I):
                return ProcessingResult.success_result("numeric")
            elif re.search(r"bool", dtype_str, re.I):
                return ProcessingResult.success_result("boolean")
            elif re.search(r"datetime|timedelta", dtype_str, re.I):
                if "timedelta" in dtype_str.lower():
                    return ProcessingResult.success_result("numeric")
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

            # Fast path: use polars' own numeric check (covers all int/uint/float variants)
            if dtype.is_numeric():
                return ProcessingResult.success_result("numeric")
            elif dtype == pl.Boolean:
                return ProcessingResult.success_result("boolean")
            elif dtype in [pl.Datetime, pl.Date]:
                return ProcessingResult.success_result("datetime")
            elif dtype == pl.Duration:
                # Duration is fundamentally numeric (total seconds)
                return ProcessingResult.success_result("numeric")
            elif dtype == pl.Time:
                # Time-of-day mapped to numeric (seconds since midnight)
                return ProcessingResult.success_result("numeric")

            # For string types, try to infer more specific types
            if dtype in (pl.Utf8, pl.String):
                if self.strategy in [
                    InferenceStrategy.AGGRESSIVE,
                    InferenceStrategy.BALANCED,
                ]:
                    return self._sample_based_inference_polars(s)
                else:
                    return ProcessingResult.success_result("categorical")

            # Nested types (Struct, List, Array) — fall through to categorical
            # but log a warning if logger is available
            if (
                dtype in (pl.Struct, pl.List)
                or str(dtype).startswith("list[")
                or str(dtype).startswith("struct")
            ):
                self.logger.debug(
                    "Column '%s' has nested type '%s'; treating as categorical",
                    getattr(s, "name", "?"),
                    dtype,
                )

            # Default to categorical
            return ProcessingResult.success_result("categorical")

        except Exception as e:
            return ProcessingResult.error_result(
                f"Polars series inference failed: {str(e)}"
            )

    def _looks_like_datetime(self, sample: Any) -> bool:
        """Decide whether an object column holds dates, cheaply.

        The obvious implementation -- hand the whole sample to
        ``pd.to_datetime(format="mixed")`` -- was the single most expensive
        thing the profiler did. ``format="mixed"`` disables pandas' vectorised
        parser and falls back to ``dateutil``, parsing one row at a time in
        Python: a single 50,000-row profile spent 20.7% of its runtime there,
        across 166,302 ``get_token`` calls.

        Three changes, in order of effect: probe a couple of hundred rows rather
        than ten thousand (this is a yes/no question, not an estimate); try
        explicit formats first, each of which takes pandas' fast path; and reach
        for ``mixed`` only when every fixed format has failed.
        """
        text = sample.dropna()
        if text.empty:
            return False
        probe = text.head(_DATE_SNIFF_SAMPLE).astype(str)

        for fmt in _DATE_SNIFF_FORMATS:
            try:
                parsed = pd.to_datetime(probe, errors="coerce", utc=True, format=fmt)
            except (ValueError, TypeError):
                continue
            if _dated_fraction(parsed) > 0.8:
                return True

        # Last resort: dateutil, but only over the small probe.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(
                    probe, errors="coerce", utc=True, format="mixed"
                )
            return _dated_fraction(parsed) > 0.8
        except (ValueError, TypeError):
            return False

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

            if len(sample) == 0:
                # Nothing to infer from. Guarding here also avoids the 0/0 that
                # the success-rate checks below would otherwise compute.
                return ProcessingResult.success_result("categorical")

            # Try datetime conversion
            if self.strategy in [
                InferenceStrategy.AGGRESSIVE,
                InferenceStrategy.BALANCED,
            ]:
                if self._looks_like_datetime(sample):
                    return ProcessingResult.success_result("datetime")

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
                # One parse, through the same helper the conversion path uses.
                # This used to be two `cast()` attempts, Date then Datetime,
                # taking whichever looked good -- and the conversion path ran
                # the same two in the same order but kept the *first*. So the
                # two disagreed about which strings are datetimes, and a column
                # this branch typed `datetime` could be converted to nothing,
                # reporting 200 valid timestamps as 100% missing (#214).
                # Sharing the helper is what stops them drifting again.
                ds = (
                    polars_string_to_datetime(sample)
                    if sample.dtype == pl.String
                    else None
                )
                if ds is None and sample.dtype != pl.String:
                    try:
                        ds = sample.cast(pl.Datetime, strict=False)
                    except Exception:
                        ds = None
                if ds is not None:
                    non_null = sample_size - ds.null_count()
                    if sample_size and non_null / sample_size > 0.8:
                        return ProcessingResult.success_result("datetime")

            # Try numeric conversion
            try:
                ns = sample.cast(pl.Float64, strict=False)
                null_count = ns.null_count()
                if (sample_size - null_count) / sample_size > 0.8:  # 80% success rate
                    return ProcessingResult.success_result("numeric")
            except Exception:
                pass

            # Try boolean conversion for string data (AGGRESSIVE only)
            if self.strategy == InferenceStrategy.AGGRESSIVE:
                try:
                    # Convert to lowercase strings and check if they are boolean-like
                    lower_sample = sample.cast(pl.Utf8).str.to_lowercase()
                    bool_mask = lower_sample.is_in(
                        ["true", "false", "1", "0", "yes", "no"]
                    )
                    if bool_mask.sum() / sample_size > 0.8:  # 80% success rate
                        return ProcessingResult.success_result("boolean")
                except Exception:
                    pass

            # Default to categorical
            return ProcessingResult.success_result("categorical")

        except Exception as e:
            return ProcessingResult.error_result(
                f"Sample-based polars inference failed: {str(e)}"
            )

    def _initialize_type_patterns(self) -> dict[str, re.Pattern]:
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

    def get_cache_stats(self) -> dict[str, Any]:
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


# A numeric column is a disguised category when it holds few enough distinct
# values to enumerate, not when it holds few *relative to the row count*. The
# rule used to include a `unique_ratio < 0.05` arm, which made the answer depend
# on how many rows you had: `age` with 67 distinct values is numeric in a
# 1,000-row frame and categorical in a 20,000-row one, though nothing about the
# column changed. Every bounded integer -- age, year, rating, day-of-month, HTTP
# status, state code -- crossed the line purely by growing. For a profiler whose
# pitch is large data, a heuristic that degrades with scale is backwards.
#
# A ceiling is stable under row count, which is the property the ratio lacked.
MAX_CATEGORICAL_LEVELS = 50
# Kept as the private name too: it was written that way and is referenced in
# the tests and in the roadmap.
_MAX_CATEGORICAL_LEVELS = MAX_CATEGORICAL_LEVELS


def should_reclassify_numeric_as_categorical(
    unique_count: int, total_count: int, *, int_like: bool = True
) -> bool:
    """Determine if a numeric column should be reclassified as categorical.

    Reclassifies numeric columns that are really discrete categories -- ratings,
    grades, status codes -- so they get a category card rather than a mean and a
    histogram.

    Args:
        unique_count: Number of unique values in the column.
        total_count: Total number of values in the column.
        int_like: Whether every value is integral. Continuous measurements that
            happen to repeat are still measurements; only whole numbers stand in
            for labels.

    Returns:
        True if column should be reclassified as categorical, False otherwise.
    """
    if total_count == 0:
        return False
    if not int_like:
        return False
    return unique_count <= _MAX_CATEGORICAL_LEVELS


def should_reclassify_numeric_as_boolean(
    series: Any, config: Any, logger: Any | None = None
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
