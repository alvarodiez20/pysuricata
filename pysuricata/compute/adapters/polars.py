"""Polars-specific adapter implementation.

This module provides the polars-specific implementation of the DataAdapter
protocol, optimized for polars DataFrames and Series.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..core.types import ColumnKinds
from ..processing.conversion import UnifiedConverter
from ..processing.inference import UnifiedTypeInferrer
from .base import BaseAdapter

try:
    import polars as pl
except ImportError:
    pl = None


class PolarsAdapter(BaseAdapter):
    """Polars-specific adapter implementation.

    This adapter provides optimized operations for polars DataFrames and Series,
    leveraging polars-specific optimizations and features.
    """

    def __init__(
        self,
        converter: Optional[UnifiedConverter] = None,
        inferrer: Optional[UnifiedTypeInferrer] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the polars adapter.

        Args:
            converter: Unified data converter.
            inferrer: Unified type inferrer.
            logger: Logger for adapter operations.
        """
        super().__init__(converter, inferrer, logger)

        if pl is None:
            raise ImportError("polars is required for PolarsAdapter")

    def infer_and_build(
        self, data: Any, config: Any
    ) -> tuple[ColumnKinds, Dict[str, Any]]:
        """Infer column types and build accumulators for polars data.

        Args:
            data: Polars DataFrame to analyze.
            config: Configuration object with processing parameters.

        Returns:
            Tuple of (column_kinds, accumulators_dict).
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError(f"Expected polars.DataFrame, got {type(data)}")

        # Infer column kinds
        result = self.inferrer.infer_kinds(data)
        if not result.success:
            raise RuntimeError(f"Type inference failed: {result.error}")

        kinds = result.data

        # Build accumulators
        accs = {}

        # Import accumulator classes
        from ...accumulators import (
            BooleanAccumulator,
            CategoricalAccumulator,
            DatetimeAccumulator,
            NumericAccumulator,
        )
        from ...accumulators.factory import build_accumulators

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
                            "🔧 forcing column '%s' type to '%s' (override) [pl]",
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

                    # Add to forced category
                    if forced_type == "boolean":
                        kinds.boolean.append(col_name)
                        from ...accumulators import BooleanAccumulator

                        accs[col_name] = BooleanAccumulator(col_name)
                    elif forced_type == "categorical":
                        kinds.categorical.append(col_name)
                        from ...accumulators import CategoricalAccumulator

                        accs[col_name] = CategoricalAccumulator(col_name)
                    elif forced_type == "numeric":
                        kinds.numeric.append(col_name)
                        from ...accumulators import NumericAccumulator

                        accs[col_name] = NumericAccumulator(col_name)
                    elif forced_type == "datetime":
                        kinds.datetime.append(col_name)
                        from ...accumulators import DatetimeAccumulator

                        accs[col_name] = DatetimeAccumulator(col_name)

                    accs[col_name].set_dtype(str(data[col_name].dtype))

        for col_name in data.columns:
            if col_name in kinds.numeric:
                # Check if this numeric column should be reclassified as boolean
                if config and should_reclassify_numeric_as_boolean(
                    data[col_name], config, self.logger
                ):
                    if self.logger:
                        self.logger.info(
                            "🔄 reclassifying '%s' from numeric to boolean (0/1 values detected) [pl]",
                            col_name,
                        )
                    # Move from numeric to boolean
                    kinds.numeric.remove(col_name)
                    kinds.boolean.append(col_name)

                    # Replace accumulator
                    from ...accumulators import BooleanAccumulator

                    accs[col_name] = BooleanAccumulator(col_name)
                    accs[col_name].set_dtype(str(data[col_name].dtype))
                else:
                    # Check if this numeric column should be reclassified as categorical
                    try:
                        unique_count = data[col_name].n_unique()
                        total_count = data.height
                        if should_reclassify_numeric_as_categorical(
                            unique_count, total_count
                        ):
                            if self.logger:
                                self.logger.info(
                                    "🔄 reclassifying '%s' from numeric to categorical (unique: %d, total: %d) [pl]",
                                    col_name,
                                    unique_count,
                                    total_count,
                                )
                            # Move from numeric to categorical
                            kinds.numeric.remove(col_name)
                            kinds.categorical.append(col_name)

                            # Replace accumulator
                            from ...accumulators import CategoricalAccumulator

                            accs[col_name] = CategoricalAccumulator(col_name)
                            accs[col_name].set_dtype(str(data[col_name].dtype))
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(
                                "Could not check unique values for '%s': %s [pl]",
                                col_name,
                                e,
                            )

        return kinds, accs

    def estimate_mem(self, frame: Any) -> int:
        """Estimate memory usage of a polars DataFrame.

        Args:
            frame: Polars DataFrame to estimate memory for.

        Returns:
            Estimated memory usage in bytes.
        """
        if not isinstance(frame, pl.DataFrame):
            return 0

        try:
            # Use polars estimated_size for accurate estimation
            return int(frame.estimated_size())
        except Exception:
            # Fallback estimation
            return frame.height * len(frame.columns) * 8  # Rough estimate

    def missing_cells(self, frame: Any) -> int:
        """Count missing cells in a polars DataFrame.

        Args:
            frame: Polars DataFrame to count missing cells in.

        Returns:
            Number of missing cells.
        """
        if not isinstance(frame, pl.DataFrame):
            return 0

        try:
            # Optimized missing value counting - use sum_horizontal() instead of complex aggregation
            return int(frame.null_count().sum_horizontal().item())
        except Exception:
            try:
                # Fallback to the original method if sum_horizontal() is not available
                return int(frame.null_count().select(pl.sum(pl.all())).to_numpy()[0][0])
            except Exception:
                return 0

    def consume_chunk(
        self,
        data: Any,
        accs: Dict[str, Any],
        kinds: ColumnKinds,
        config: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """Consume a polars DataFrame chunk and update accumulators.

        Args:
            data: Polars DataFrame chunk to process.
            accs: Dictionary of accumulators to update.
            kinds: Column type information.
            config: Configuration object for type inference and processing.
            logger: Optional logger for progress tracking.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError(f"Expected polars.DataFrame, got {type(data)}")

        # Import the polars-specific consume function
        from ..consume_polars import consume_chunk_polars

        # Use the existing polars consume function
        consume_chunk_polars(data, accs, kinds, config, logger)

    def update_corr(
        self, frame: Any, corr_est: Any, logger: Optional[Any] = None
    ) -> None:
        """Update correlation estimator with polars DataFrame.

        Args:
            frame: Polars DataFrame to process.
            corr_est: Correlation estimator to update.
            logger: Optional logger for progress tracking.
        """
        if not isinstance(frame, pl.DataFrame):
            return

        try:
            if hasattr(corr_est, "update_from_polars"):
                corr_est.update_from_polars(frame)
        except Exception as e:
            if logger:
                logger.warning(f"Correlation update failed: {e}")

    def sample_section_html(self, first: Any, cfg: Any) -> str:
        """Generate HTML for sample data section from polars DataFrame.

        Args:
            first: First polars DataFrame chunk for sampling.
            cfg: Configuration object.

        Returns:
            HTML string for sample section.
        """
        if not isinstance(first, pl.DataFrame):
            return ""

        try:
            # Import the polars-specific render function
            from ...render.sections import render_sample_section_polars

            # Extract sample_rows from config, default to 10
            sample_rows = getattr(cfg, "sample_rows", 10)
            return render_sample_section_polars(first, sample_rows)
        except Exception:
            return ""

    def _is_compatible_data(self, data: Any) -> bool:
        """Check if data is compatible with polars adapter.

        Args:
            data: Data to check.

        Returns:
            True if compatible, False otherwise.
        """
        return isinstance(data, pl.DataFrame)

    def get_polars_info(self) -> Dict[str, Any]:
        """Get polars-specific information.

        Returns:
            Dictionary with polars-specific information.
        """
        info = self.get_adapter_info()
        info.update(
            {
                "polars_version": pl.__version__ if pl else None,
                "numpy_version": np.__version__,
            }
        )
        return info

    def update_row_kmv(self, frame: Any, row_kmv: Any) -> None:
        """Update the row-level distinct estimator from a polars DataFrame.

        Args:
            frame: Polars DataFrame to process.
            row_kmv: Row KMV estimator to update.
        """
        if not isinstance(frame, pl.DataFrame):
            return

        try:
            if hasattr(row_kmv, "update_from_polars"):
                row_kmv.update_from_polars(frame)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to update row KMV: {e}")
