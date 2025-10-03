"""Analysis module for statistical computations.

This module provides analysis capabilities including correlation analysis,
statistical estimators, and metrics computation.
"""

from ...accumulators.sketches import RowKMV
from .correlation import StreamingCorr
from .metrics import (
    apply_corr_chips,
    build_kinds_map,
    build_manifest_inputs,
    compute_col_order,
    compute_dataset_shape,
    compute_top_missing,
)

__all__ = [
    "StreamingCorr",
    "RowKMV",
    "build_kinds_map",
    "compute_top_missing",
    "compute_col_order",
    "compute_dataset_shape",
    "build_manifest_inputs",
    "apply_corr_chips",
]
