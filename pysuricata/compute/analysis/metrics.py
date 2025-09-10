from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..core.types import ColumnKinds


def build_kinds_map(
    kinds: ColumnKinds, accs: Mapping[str, Any]
) -> Dict[str, Tuple[str, Any]]:
    """Return name -> (kind, accumulator) map for all known columns."""
    return {
        **{name: ("numeric", accs[name]) for name in kinds.numeric if name in accs},
        **{
            name: ("categorical", accs[name])
            for name in kinds.categorical
            if name in accs
        },
        **{name: ("datetime", accs[name]) for name in kinds.datetime if name in accs},
        **{name: ("boolean", accs[name]) for name in kinds.boolean if name in accs},
    }


def compute_top_missing(
    kinds_map: Mapping[str, Tuple[str, Any]],
) -> List[Tuple[str, float, int]]:
    """Compute per-column missing percentage and counts, sorted descending by pct."""
    miss_list: List[Tuple[str, float, int]] = []
    for name, (_kind, acc) in kinds_map.items():
        miss = int(getattr(acc, "missing", 0))
        cnt = int(getattr(acc, "count", 0)) + miss
        pct = (miss / cnt * 100.0) if cnt else 0.0
        miss_list.append((name, pct, miss))
    miss_list.sort(key=lambda t: t[1], reverse=True)
    return miss_list


def compute_col_order(first_columns: Sequence[str], kinds: ColumnKinds) -> List[str]:
    """Prefer the original first chunk order when available; otherwise by kinds."""
    prefer = list(first_columns) if first_columns else []
    valid = set(kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean)
    return [c for c in prefer if c in valid] or (
        kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean
    )


def compute_dataset_shape(
    kinds_map: Mapping[str, Tuple[str, Any]], row_kmv: Any
) -> Tuple[int, int]:
    """Return (n_rows, n_cols) for the dataset used by manifest/reporting.

    Rows are estimated from the row-KMV tracker; columns from the kinds map.
    """
    n_rows = int(getattr(row_kmv, "rows", 0))
    n_cols = int(len(kinds_map))
    return n_rows, n_cols


def build_manifest_inputs(
    *,
    kinds: ColumnKinds,
    accs: Mapping[str, Any],
    row_kmv: Any,
    first_columns: Sequence[str],
) -> Tuple[
    Dict[str, Tuple[str, Any]], List[str], int, int, List[Tuple[str, float, int]]
]:
    """Return the full set of inputs required by build_summary.

    This bundles kinds_map, col_order, (n_rows, n_cols), and miss_list to keep
    the orchestration layer lean and cohesive.
    """
    kinds_map = build_kinds_map(kinds, accs)
    col_order = compute_col_order(first_columns, kinds)
    n_rows, n_cols = compute_dataset_shape(kinds_map, row_kmv)
    miss_list = compute_top_missing(kinds_map)
    return kinds_map, col_order, n_rows, n_cols, miss_list


def apply_corr_chips(
    accs: Mapping[str, Any], kinds: ColumnKinds, top_map: Mapping[str, Any]
) -> None:
    """Attach correlation chip info to correlatable accumulators (numeric).

    Uses duck typing; if an accumulator exposes ``set_corr_top`` it is invoked.
    """
    for name in kinds.numeric:
        acc = accs.get(name)  # type: ignore[attr-defined]
        if acc is None:
            continue
        if hasattr(acc, "set_corr_top"):
            try:
                acc.set_corr_top(top_map.get(name, []))  # type: ignore[call-arg]
            except Exception:
                pass
