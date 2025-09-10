"""Manifest building for compute operations.

This module provides manifest building capabilities for creating
JSON-safe summaries from processed data.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..analysis.metrics import (
    build_kinds_map,
    compute_col_order,
    compute_dataset_shape,
    compute_top_missing,
)
from ..core.types import ProcessingResult

try:
    import pandas as pd
except ImportError:
    pd = None


class ManifestBuilder:
    """Builder for creating data manifests.

    This class provides functionality for building JSON-safe manifests
    from processed data, including dataset summaries and column statistics.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the manifest builder.

        Args:
            logger: Logger for manifest operations.
        """
        self.logger = logger or logging.getLogger(__name__)

    def build_manifest_inputs(
        self,
        kinds: Any,
        accs: Mapping[str, Any],
        row_kmv: Any,
        first_columns: Sequence[str],
    ) -> ProcessingResult[
        Tuple[
            Dict[str, Tuple[str, Any]],
            List[str],
            int,
            int,
            List[Tuple[str, float, int]],
        ]
    ]:
        """Build manifest inputs for summary generation.

        Args:
            kinds: Column kinds information.
            accs: Dictionary of accumulators.
            row_kmv: Row KMV estimator.
            first_columns: First chunk column order.

        Returns:
            ProcessingResult containing manifest inputs.
        """
        start_time = time.time()

        try:
            # Build kinds map
            kinds_map = build_kinds_map(kinds, accs)

            # Compute column order
            col_order = compute_col_order(first_columns, kinds)

            # Compute dataset shape
            n_rows, n_cols = compute_dataset_shape(kinds_map, row_kmv)

            # Compute missing values
            miss_list = compute_top_missing(kinds_map)

            duration = time.time() - start_time

            return ProcessingResult.success_result(
                data=(kinds_map, col_order, n_rows, n_cols, miss_list),
                metrics={
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "missing_columns": len(miss_list),
                },
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Manifest building failed: {str(e)}",
                duration=duration,
            )

    def build_summary(
        self,
        kinds_map: Mapping[str, Tuple[str, Any]],
        col_order: Sequence[str],
        row_kmv: Any,
        total_missing_cells: int,
        n_rows: int,
        n_cols: int,
        miss_list: Sequence[Tuple[str, float, int]],
    ) -> ProcessingResult[Optional[Dict[str, Any]]]:
        """Build programmatic summary from processed data.

        Args:
            kinds_map: Column kinds mapping.
            col_order: Column order.
            row_kmv: Row KMV estimator.
            total_missing_cells: Total missing cells count.
            n_rows: Number of rows.
            n_cols: Number of columns.
            miss_list: Missing values list.

        Returns:
            ProcessingResult containing the summary.
        """
        start_time = time.time()

        try:
            # Import the existing build_summary function
            from ..manifest import build_summary as _build_summary

            # Build the summary
            result = _build_summary(
                kinds_map=kinds_map,
                col_order=col_order,
                row_kmv=row_kmv,
                total_missing_cells=total_missing_cells,
                n_rows=n_rows,
                n_cols=n_cols,
                miss_list=miss_list,
            )

            duration = time.time() - start_time

            return ProcessingResult.success_result(
                data=result,
                metrics={
                    "summary_size": len(result) if result else 0,
                },
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Summary building failed: {str(e)}",
                duration=duration,
            )

    def apply_correlation_chips(
        self,
        accs: Mapping[str, Any],
        kinds: Any,
        top_map: Mapping[str, Any],
    ) -> ProcessingResult[None]:
        """Apply correlation chips to accumulators.

        Args:
            accs: Dictionary of accumulators.
            kinds: Column type information.
            top_map: Correlation mapping.

        Returns:
            ProcessingResult indicating success/failure.
        """
        start_time = time.time()

        try:
            # Import the existing apply_corr_chips function
            from ..analysis.metrics import apply_corr_chips

            # Apply correlation chips
            apply_corr_chips(accs, kinds, top_map)

            duration = time.time() - start_time

            return ProcessingResult.success_result(
                data=None,
                metrics={
                    "correlations_applied": len(top_map),
                },
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Correlation chips application failed: {str(e)}",
                duration=duration,
            )

    def get_manifest_info(self) -> Dict[str, Any]:
        """Get manifest builder information.

        Returns:
            Dictionary with manifest builder information.
        """
        return {
            "builder_type": self.__class__.__name__,
            "capabilities": [
                "manifest_inputs",
                "summary_building",
                "correlation_chips",
            ],
        }
