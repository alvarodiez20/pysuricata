"""Pandas-specific adapter implementation.

This module provides the pandas-specific implementation of the DataAdapter
protocol, optimized for pandas DataFrames and Series.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.types import ColumnKinds
from ..processing.conversion import UnifiedConverter
from ..processing.inference import UnifiedTypeInferrer
from .base import BaseAdapter

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasAdapter(BaseAdapter):
    """Pandas-specific adapter implementation.

    This adapter provides optimized operations for pandas DataFrames and Series,
    leveraging pandas-specific optimizations and features.
    """

    def __init__(
        self,
        converter: UnifiedConverter | None = None,
        inferrer: UnifiedTypeInferrer | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the pandas adapter.

        Args:
            converter: Unified data converter.
            inferrer: Unified type inferrer.
            logger: Logger for adapter operations.
        """
        super().__init__(converter, inferrer, logger)

        if pd is None:
            raise ImportError("pandas is required for PandasAdapter")

    def infer_and_build(
        self, data: Any, config: Any, *, first_chunk_is_whole: bool = True
    ) -> tuple[ColumnKinds, dict[str, Any]]:
        """Infer column types and build accumulators for pandas data.

        Args:
            data: Pandas DataFrame to analyze.
            config: Configuration object with processing parameters.

        Returns:
            Tuple of (column_kinds, accumulators_dict).
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(data)}")

        # Infer column kinds
        result = self.inferrer.infer_kinds(data)
        if not result.success:
            raise RuntimeError(f"Type inference failed: {result.error}")

        kinds = result.data

        # Build accumulators
        accs = {}

        from ...accumulators.factory import build_accumulator, build_accumulators

        # Build accumulators using the factory
        accs = build_accumulators(kinds, config)

        # Set actual dtypes from the DataFrame
        for col_name in data.columns:
            if col_name in accs:
                actual_dtype = str(data[col_name].dtype)
                accs[col_name].set_dtype(actual_dtype)

        # Apply reclassification logic for numeric columns
        from ..processing.inference import (
            should_reclassify_numeric_as_boolean,
            should_reclassify_numeric_as_categorical,
        )

        # First, apply forced column types to all columns
        if config and config.force_column_types:
            for col_name, forced_type in config.force_column_types.items():
                if col_name in data.columns:
                    if self.logger:
                        self.logger.info(
                            "🔧 forcing column '%s' type to '%s' (override)",
                            col_name,
                            forced_type,
                        )

                    # Remove from current category
                    if col_name in kinds.numeric:
                        kinds.numeric.remove(col_name)
                    elif col_name in kinds.categorical:
                        kinds.categorical.remove(col_name)
                    elif col_name in kinds.boolean:
                        kinds.boolean.remove(col_name)
                    elif col_name in kinds.datetime:
                        kinds.datetime.remove(col_name)

                    # Add to forced category. The replacement is built through
                    # the factory so it inherits the run's sketch sizes and
                    # seed, rather than falling back to library defaults.
                    if forced_type in ("boolean", "categorical", "numeric", "datetime"):
                        getattr(kinds, forced_type).append(col_name)
                        accs[col_name] = build_accumulator(
                            forced_type, col_name, config
                        )
                        accs[col_name].set_dtype(str(data[col_name].dtype))

        for col_name in data.columns:
            if col_name in kinds.numeric:
                # Check if this numeric column should be reclassified as boolean
                if config and should_reclassify_numeric_as_boolean(
                    data[col_name], config, self.logger
                ):
                    if self.logger:
                        self.logger.info(
                            "🔄 reclassifying '%s' from numeric to boolean (0/1 values detected)",
                            col_name,
                        )
                    # Move from numeric to boolean
                    kinds.numeric.remove(col_name)
                    kinds.boolean.append(col_name)

                    accs[col_name] = build_accumulator("boolean", col_name, config)
                    accs[col_name].set_dtype(str(data[col_name].dtype))
                elif first_chunk_is_whole:
                    # Reclassification reads the distinct-value ratio of this
                    # chunk. That is only evidence about the column when the
                    # chunk *is* the column: on a stream, an unrepresentative
                    # prefix -- sorted data, a leading run of one value -- would
                    # permanently mislabel it, and nothing revisits the decision.
                    # A 285,000-row column with 244,255 distinct values was
                    # classified categorical because its first 45,000 rows held
                    # nine. Streamed numeric columns therefore stay numeric,
                    # which still reports top values via Misra-Gries.
                    try:
                        column = data[col_name]
                        unique_count = column.nunique()
                        total_count = len(column)
                        # Only whole numbers stand in for labels; a continuous
                        # measurement that repeats is still a measurement.
                        numeric = pd.to_numeric(column, errors="coerce").dropna()
                        int_like = bool(numeric.empty or (numeric % 1 == 0).all())
                        if should_reclassify_numeric_as_categorical(
                            unique_count, total_count, int_like=int_like
                        ):
                            if self.logger:
                                self.logger.info(
                                    "🔄 reclassifying '%s' from numeric to categorical (unique: %d, total: %d)",
                                    col_name,
                                    unique_count,
                                    total_count,
                                )
                            # Move from numeric to categorical
                            kinds.numeric.remove(col_name)
                            kinds.categorical.append(col_name)

                            accs[col_name] = build_accumulator(
                                "categorical", col_name, config
                            )
                            accs[col_name].set_dtype(str(data[col_name].dtype))
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(
                                "Could not check unique values for '%s': %s",
                                col_name,
                                e,
                            )

        return kinds, accs

    def estimate_mem(self, frame: Any) -> int:
        """Estimate memory usage of a pandas DataFrame.

        Args:
            frame: Pandas DataFrame to estimate memory for.

        Returns:
            Estimated memory usage in bytes.
        """
        if not isinstance(frame, pd.DataFrame):
            return 0

        try:
            # Use pandas memory_usage for accurate estimation
            return int(frame.memory_usage(deep=True).sum())
        except Exception:
            # Fallback estimation
            return len(frame) * len(frame.columns) * 8  # Rough estimate

    def missing_cells(self, frame: Any) -> int:
        """Count missing cells in a pandas DataFrame.

        Args:
            frame: Pandas DataFrame to count missing cells in.

        Returns:
            Number of missing cells.
        """
        if not isinstance(frame, pd.DataFrame):
            return 0

        try:
            return int(frame.isnull().sum().sum())
        except Exception:
            return 0

    def consume_chunk(
        self,
        data: Any,
        accs: dict[str, Any],
        kinds: ColumnKinds,
        config: Any | None = None,
        logger: Any | None = None,
        *,
        row_offset: int = 0,
    ) -> None:
        """Consume a pandas DataFrame chunk and update accumulators.

        Args:
            data: Pandas DataFrame chunk to process.
            accs: Dictionary of accumulators to update.
            kinds: Column type information.
            config: Configuration object for type inference and processing.
            logger: Optional logger for progress tracking.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(data)}")

        # Import the pandas-specific consume function
        from ..consume import consume_chunk_pandas

        # Use the existing pandas consume function
        consume_chunk_pandas(data, accs, kinds, config, logger, row_offset=row_offset)

    def update_corr(self, frame: Any, corr_est: Any, logger: Any | None = None) -> None:
        """Update correlation estimator with pandas DataFrame.

        Args:
            frame: Pandas DataFrame to process.
            corr_est: Correlation estimator to update.
            logger: Optional logger for progress tracking.
        """
        if not isinstance(frame, pd.DataFrame):
            return

        try:
            if hasattr(corr_est, "update_from_pandas"):
                corr_est.update_from_pandas(frame)
        except Exception as e:
            if logger:
                logger.warning(f"Correlation update failed: {e}")

    def sample_section_html(self, first: Any, cfg: Any) -> str:
        """Generate HTML for sample data section from pandas DataFrame.

        Args:
            first: First pandas DataFrame chunk for sampling.
            cfg: Configuration object.

        Returns:
            HTML string for sample section.
        """
        if not isinstance(first, pd.DataFrame):
            return ""

        try:
            # Import the render function
            from ...accumulators.factory import seed_for_column
            from ...render.sections import render_sample_section

            # Extract sample_rows from config, default to 10
            sample_rows = getattr(cfg, "sample_rows", 10)
            return render_sample_section(
                first, sample_rows, seed=seed_for_column(cfg, "__sample_section__")
            )
        except Exception:
            return ""

    def _is_compatible_data(self, data: Any) -> bool:
        """Check if data is compatible with pandas adapter.

        Args:
            data: Data to check.

        Returns:
            True if compatible, False otherwise.
        """
        return isinstance(data, pd.DataFrame)

    def get_pandas_info(self) -> dict[str, Any]:
        """Get pandas-specific information.

        Returns:
            Dictionary with pandas-specific information.
        """
        info = self.get_adapter_info()
        info.update(
            {
                "pandas_version": pd.__version__ if pd else None,
                "numpy_version": np.__version__,
            }
        )
        return info

    def update_row_kmv(self, frame: Any, row_kmv: Any) -> None:
        """Update the row-level distinct estimator from a pandas DataFrame.

        Args:
            frame: Pandas DataFrame to process.
            row_kmv: Row KMV estimator to update.
        """
        if not isinstance(frame, pd.DataFrame):
            return

        try:
            if hasattr(row_kmv, "update_from_pandas"):
                row_kmv.update_from_pandas(frame)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to update row KMV: {e}")
