"""Build the JSON-safe manifest from finalized accumulators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..accumulators.protocols import FinalizableAccumulator
from ..render.missing_columns import create_missing_columns_renderer

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def duplicate_fields(row_kmv: Any) -> dict[str, Any]:
    """The duplicate keys of the `summarize()` payload, suppression applied.

    Shared by both payload producers -- this module and `report.py` -- because
    they used to build these keys independently from raw `approx_duplicates()`
    while `render/html.py` applied a resolvability threshold to the same
    numbers. The report printed `< 2,225 · below sketch resolution` and the
    payload returned 1,109 for a frame with no duplicate rows at all, which is
    the wrong way round: the HTML structure is explicitly *not* covered by
    `docs/versioning.md` and the payload is.

    `duplicate_rows_uncertainty` is what makes a zero readable. Zero with an
    uncertainty of zero is "exactly none"; zero with a nonzero uncertainty is
    "nothing resolvable" -- and the bound is not the uncertainty itself. The
    resolvability gate is `DUPLICATE_RESOLUTION_SIGMAS` (3) standard
    deviations (#248), so a suppressed count's ceiling is
    `math.ceil(3 * duplicate_rows_uncertainty)`, the same figure `render/html.py`
    prints. The field stays one sigma rather than exporting the ceiling
    directly, so the multiple can move without bumping `schema_version`;
    `docs/summary-schema.md` states the multiple for consumers who need it.
    """
    if not hasattr(row_kmv, "duplicates"):
        return {
            "duplicate_rows_est": 0,
            "duplicate_rows_pct_est": 0.0,
            "duplicate_rows_uncertainty": 0,
        }
    estimate = row_kmv.duplicates()
    return {
        "duplicate_rows_est": int(estimate.rows),
        "duplicate_rows_pct_est": float(estimate.pct),
        "duplicate_rows_uncertainty": int(estimate.uncertainty),
    }


def build_summary(
    kinds_map: Mapping[str, tuple[str, FinalizableAccumulator]],
    col_order: Sequence[str],
    *,
    row_kmv: Any,
    total_missing_cells: int,
    n_rows: int,
    n_cols: int,
    miss_list: Sequence[tuple[str, float, int]] = (),
) -> Mapping[str, Any]:
    """Construct a minimal, JSON-safe summary manifest.

    Parameters
    - kinds_map: name -> (kind, accumulator)
    - col_order: stable column order to iterate summaries
    - row_kmv: object exposing rows and approx_duplicates()
    - total_missing_cells: total missing across dataset
    - n_rows, n_cols: dataset shape estimates
    - miss_list: optional precomputed top-missing list [(name, pct, count)]
    """
    dataset_summary = {
        "rows_est": int(n_rows),
        "cols": int(n_cols),
        "missing_cells": int(total_missing_cells),
        "missing_cells_pct": (total_missing_cells / max(1, n_rows * n_cols) * 100.0)
        if (n_rows and n_cols)
        else 0.0,
        **duplicate_fields(row_kmv),
        "top_missing": _get_intelligent_top_missing(miss_list, n_rows, n_cols),
    }

    columns_summary: dict[str, dict[str, Any]] = {}
    for name in col_order:
        kind, acc = kinds_map[name]
        if kind == "numeric":
            s = acc.finalize()
            columns_summary[name] = {
                "type": "numeric",
                "count": s.count,
                "missing": s.missing,
                "unique_est": s.unique_est,
                "mean": s.mean,
                "std": s.std,
                "min": s.min,
                "q1": s.q1,
                "median": s.median,
                "q3": s.q3,
                "max": s.max,
                "zeros": s.zeros,
                "negatives": s.negatives,
                "outliers_iqr_est": s.outliers_iqr,
                "approx": bool(s.approx),
                "mem_bytes": s.mem_bytes,
            }
        elif kind == "categorical":
            s = acc.finalize()
            columns_summary[name] = {
                "type": "categorical",
                "count": s.count,
                "missing": s.missing,
                "unique_est": s.unique_est,
                "top_items": s.top_items,
                "approx": bool(s.approx),
                "mem_bytes": s.mem_bytes,
            }
        elif kind == "datetime":
            s = acc.finalize()
            columns_summary[name] = {
                "type": "datetime",
                "count": s.count,
                "missing": s.missing,
                "min_ts": s.min_ts,
                "max_ts": s.max_ts,
                "mem_bytes": s.mem_bytes,
            }
        else:  # boolean
            s = acc.finalize()
            columns_summary[name] = {
                "type": "boolean",
                "count": s.count,
                "missing": s.missing,
                "true": s.true_n,
                "false": s.false_n,
                "mem_bytes": s.mem_bytes,
            }

    return {"dataset": dataset_summary, "columns": columns_summary}


def _get_intelligent_top_missing(
    miss_list: Sequence[tuple[str, float, int]], n_rows: int, n_cols: int
) -> list[dict[str, Any]]:
    """Get top missing columns (max 5) using the analyzer.

    Args:
        miss_list: List of (column_name, missing_pct, missing_count) tuples
        n_rows: Total number of rows in dataset
        n_cols: Total number of columns in dataset

    Returns:
        List of dictionaries with column information for JSON serialization
    """
    if not miss_list:
        return []

    # Use the analyzer to determine what to include (max 5)
    renderer = create_missing_columns_renderer(min_threshold_pct=0.5)
    result = renderer.analyzer.analyze_missing_columns(miss_list, n_cols, n_rows)

    # Return the columns (max 5)
    return [
        {"column": str(col), "pct": float(pct), "count": int(cnt)}
        for col, pct, cnt in result.columns
    ]
