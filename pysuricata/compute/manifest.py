"""Build the JSON-safe manifest from finalized accumulators."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def build_summary(
    kinds_map: Mapping[str, Tuple[str, Any]],
    col_order: Sequence[str],
    *,
    row_kmv: Any,
    total_missing_cells: int,
    n_rows: int,
    n_cols: int,
    miss_list: Sequence[Tuple[str, float, int]] = (),
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
        "missing_cells_pct": (total_missing_cells / max(1, n_rows * n_cols) * 100.0) if (n_rows and n_cols) else 0.0,
        "duplicate_rows_est": int(row_kmv.approx_duplicates()[0]) if hasattr(row_kmv, "approx_duplicates") else 0,
        "duplicate_rows_pct_est": float(row_kmv.approx_duplicates()[1]) if hasattr(row_kmv, "approx_duplicates") else 0.0,
        "top_missing": [
            {"column": str(col), "pct": float(pct), "count": int(cnt)}
            for col, pct, cnt in (list(miss_list)[:5] if miss_list else [])
        ],
    }

    columns_summary: Dict[str, Dict[str, Any]] = {}
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
            }
        elif kind == "datetime":
            s = acc.finalize()
            columns_summary[name] = {
                "type": "datetime",
                "count": s.count,
                "missing": s.missing,
                "min_ts": s.min_ts,
                "max_ts": s.max_ts,
            }
        else:  # boolean
            s = acc.finalize()
            columns_summary[name] = {
                "type": "boolean",
                "count": s.count,
                "missing": s.missing,
                "true": s.true_n,
                "false": s.false_n,
            }

    return {"dataset": dataset_summary, "columns": columns_summary}

