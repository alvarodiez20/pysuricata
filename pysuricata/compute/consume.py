"""Chunk consumption and accumulator wiring for pandas chunks."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ..accumulators import (
    BooleanAccumulator,
    CategoricalAccumulator,
    DatetimeAccumulator,
    NumericAccumulator,
)
from .core.types import ColumnKinds
from .processing.inference import UnifiedTypeInferrer


def _estimate_memory_per_row_fast(s: pd.Series) -> float:  # type: ignore[name-defined]
    """Fast memory estimation based on dtype instead of deep profiling.
    
    This avoids the expensive memory_usage(deep=True) which traverses every string.
    
    Args:
        s: pandas Series to estimate memory for
        
    Returns:
        Estimated bytes per row
    """
    dtype = s.dtype
    
    # Fast dtype-based estimation
    if dtype == 'object':
        # For object columns, estimate based on sample
        if len(s) > 0:
            # Sample first 100 values to estimate average string length
            sample_size = min(100, len(s))
            sample = s.head(sample_size)
            # Rough estimate: 8 bytes overhead + average string length
            avg_length = sample.astype(str).str.len().mean()
            return 8 + avg_length
        return 8  # Default for empty series
    elif dtype == 'string':
        # String dtype - estimate based on sample
        if len(s) > 0:
            sample_size = min(100, len(s))
            sample = s.head(sample_size)
            avg_length = sample.str.len().mean()
            return avg_length
        return 8
    else:
        # Numeric/datetime types - use dtype size
        return dtype.itemsize


def _to_numeric_array_pandas(s: pd.Series) -> np.ndarray:  # type: ignore[name-defined]
    """Best-effort fast path to float64 NumPy array with NaN for invalid.

    - If the Series is already numeric (including pandas nullable ints),
      avoid the overhead of `pd.to_numeric` and go straight to NumPy.
    - Otherwise, coerce with `pd.to_numeric(errors='coerce')`.
    """
    try:
        # Fast path for numeric dtypes (exclude booleans)
        if pd is not None:
            from pandas.api import types as pdt  # type: ignore

            dt = getattr(s, "dtype", None)
            if (
                dt is not None
                and not pdt.is_bool_dtype(dt)
                and pdt.is_numeric_dtype(dt)
            ):
                return s.to_numpy(dtype="float64", copy=False)
    except Exception:
        # Fall through to the coercion path on any failure
        pass
    try:
        ns = pd.to_numeric(s, errors="coerce")  # type: ignore[operator]
        return ns.to_numpy(dtype="float64", copy=False)
    except Exception:
        # Last resort: NumPy coercion (may be slower for object dtype)
        return np.asarray(
            getattr(s, "to_numpy", lambda: np.asarray(s))(), dtype="float64"
        )


def _to_bool_array_pandas(s: pd.Series) -> List[Optional[bool]]:  # type: ignore[name-defined]
    if str(s.dtype).startswith("bool"):
        arr = s.astype("boolean").tolist()
        return [None if x is pd.NA else bool(x) for x in arr]

    def _coerce(v: Any) -> Optional[bool]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        vs = str(v).strip().lower()
        if vs in {"true", "1", "t", "yes", "y"}:
            return True
        if vs in {"false", "0", "f", "no", "n"}:
            return False
        return None

    return [_coerce(v) for v in s.tolist()]


def _to_datetime_ns_array_pandas(s: pd.Series) -> List[Optional[int]]:  # type: ignore[name-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            ds = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        except TypeError:
            ds = pd.to_datetime(s, errors="coerce", utc=True)
    vals = ds.astype("int64", copy=False).tolist()
    NAT_INT = -9223372036854775808
    out: List[Optional[int]] = []
    for v in vals:
        out.append(None if v == NAT_INT else int(v))
    return out


def _to_categorical_iter_pandas(s: pd.Series) -> Iterable[Any]:  # type: ignore[name-defined]
    return s.tolist()


def consume_chunk_pandas(
    df: pd.DataFrame,
    accs: Dict[str, Any],
    kinds: ColumnKinds,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:  # type: ignore[name-defined]
    # Initialize memory cache if not present
    if not hasattr(consume_chunk_pandas, '_memory_cache'):
        consume_chunk_pandas._memory_cache = {}
    
    # 1) Create accumulators for columns not seen in the first chunk
    for name in df.columns:
        if name in accs:
            continue
        inferrer = UnifiedTypeInferrer()
        result = inferrer.infer_series_type(df[name])
        if result.success:
            kind = result.data
        else:
            kind = "categorical"  # fallback
        # Get the actual dtype string from the pandas Series
        actual_dtype = str(df[name].dtype)

        if kind == "numeric":
            accs[name] = NumericAccumulator(name)
            accs[name].set_dtype(actual_dtype)
            kinds.numeric.append(name)
        elif kind == "boolean":
            accs[name] = BooleanAccumulator(name)
            accs[name].set_dtype(actual_dtype)
            kinds.boolean.append(name)
        elif kind == "datetime":
            accs[name] = DatetimeAccumulator(name)
            accs[name].set_dtype(actual_dtype)
            kinds.datetime.append(name)
        else:
            accs[name] = CategoricalAccumulator(name)
            accs[name].set_dtype(actual_dtype)
            kinds.categorical.append(name)
        if logger:
            logger.info("➕ discovered new column '%s' inferred as %s", name, kind)

    # 2) Feed accumulators for columns present in this chunk
    for name, acc in accs.items():
        if name not in df.columns:
            if logger:
                logger.debug("column '%s' not present in this chunk; skipping", name)
            continue
        s = df[name]
        
        # Get cached memory usage or calculate and cache it
        if name not in consume_chunk_pandas._memory_cache:
            try:
                # Use fast dtype-based estimation instead of expensive deep profiling
                memory_per_row = _estimate_memory_per_row_fast(s)
                consume_chunk_pandas._memory_cache[name] = memory_per_row
            except Exception:
                consume_chunk_pandas._memory_cache[name] = 0
        
        # Use cached memory estimate
        estimated_memory = int(consume_chunk_pandas._memory_cache[name] * len(s))
        
        if isinstance(acc, NumericAccumulator):
            arr = _to_numeric_array_pandas(s)
            acc.update(arr)
            # Track memory usage using cached estimate
            try:
                acc.add_mem(estimated_memory)
            except Exception:
                pass
            # Track extremes with indices - only every 5 chunks for performance
            # Initialize chunk counter if not exists
            if not hasattr(acc, '_extreme_update_counter'):
                acc._extreme_update_counter = 0
            
            acc._extreme_update_counter += 1
            
            # Only update extremes every 5 chunks to reduce overhead
            if acc._extreme_update_counter % 5 == 0:
                try:
                    finite = np.isfinite(arr)
                    if finite.any():
                        vals = arr[finite]
                        idx = s.index.to_numpy()[finite]
                        if vals.size > 0:
                            k = min(5, vals.size)
                            part_min = np.argpartition(vals, k - 1)[:k]
                            pairs_min = [(idx[i], float(vals[i])) for i in part_min]
                            part_max = np.argpartition(-vals, k - 1)[:k]
                            pairs_max = [(idx[i], float(vals[i])) for i in part_max]
                            acc.update_extremes(pairs_min, pairs_max)
                except Exception:
                    pass
        elif isinstance(acc, BooleanAccumulator):
            arr = _to_bool_array_pandas(s)
            acc.update(arr)
            try:
                acc.add_mem(estimated_memory)
            except Exception:
                pass
        elif isinstance(acc, DatetimeAccumulator):
            arr = _to_datetime_ns_array_pandas(s)
            acc.update(arr)
            try:
                acc.add_mem(estimated_memory)
            except Exception:
                pass
        elif isinstance(acc, CategoricalAccumulator):
            acc.update(_to_categorical_iter_pandas(s))
            # Add memory tracking for categorical columns using cached estimate
            try:
                acc.add_mem(estimated_memory)
            except Exception:
                pass
