"""Chunk consumption and accumulator wiring for pandas chunks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import math
import warnings

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ..accumulators import (
    NumericAccumulator,
    BooleanAccumulator,
    DatetimeAccumulator,
    CategoricalAccumulator,
)
from .infer import ColumnKinds, infer_kind_for_series_pandas


def _to_numeric_array_pandas(s: "pd.Series") -> np.ndarray:  # type: ignore[name-defined]
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
            if dt is not None and not pdt.is_bool_dtype(dt) and pdt.is_numeric_dtype(dt):
                return s.to_numpy(dtype="float64", copy=False)
    except Exception:
        # Fall through to the coercion path on any failure
        pass
    try:
        ns = pd.to_numeric(s, errors="coerce")  # type: ignore[operator]
        return ns.to_numpy(dtype="float64", copy=False)
    except Exception:
        # Last resort: NumPy coercion (may be slower for object dtype)
        return np.asarray(getattr(s, "to_numpy", lambda: np.asarray(s))(), dtype="float64")


def _to_bool_array_pandas(s: "pd.Series") -> List[Optional[bool]]:  # type: ignore[name-defined]
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


def _to_datetime_ns_array_pandas(s: "pd.Series") -> List[Optional[int]]:  # type: ignore[name-defined]
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


def _to_categorical_iter_pandas(s: "pd.Series") -> Iterable[Any]:  # type: ignore[name-defined]
    return s.tolist()


def consume_chunk_pandas(df: "pd.DataFrame", accs: Dict[str, Any], kinds: ColumnKinds, logger: Optional["logging.Logger"] = None) -> None:  # type: ignore[name-defined]
    # 1) Create accumulators for columns not seen in the first chunk
    for name in df.columns:
        if name in accs:
            continue
        kind = infer_kind_for_series_pandas(df[name])
        if kind == "numeric":
            accs[name] = NumericAccumulator(name)
            kinds.numeric.append(name)
        elif kind == "boolean":
            accs[name] = BooleanAccumulator(name)
            kinds.boolean.append(name)
        elif kind == "datetime":
            accs[name] = DatetimeAccumulator(name)
            kinds.datetime.append(name)
        else:
            accs[name] = CategoricalAccumulator(name)
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
        if isinstance(acc, NumericAccumulator):
            arr = _to_numeric_array_pandas(s)
            acc.update(arr)
            # Track extremes with indices for this chunk
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
                acc.add_mem(int(s.memory_usage(deep=True)))
            except Exception:
                pass
        elif isinstance(acc, DatetimeAccumulator):
            arr = _to_datetime_ns_array_pandas(s)
            acc.update(arr)
            try:
                acc.add_mem(int(s.memory_usage(deep=True)))
            except Exception:
                pass
        elif isinstance(acc, CategoricalAccumulator):
            acc.update(_to_categorical_iter_pandas(s))
