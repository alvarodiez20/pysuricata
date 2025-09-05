"""Chunk consumption and accumulator wiring for polars chunks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import warnings

import numpy as np

try:  # optional
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from ..accumulators import (
    NumericAccumulator,
    BooleanAccumulator,
    DatetimeAccumulator,
    CategoricalAccumulator,
)
from .infer import ColumnKinds, infer_kind_for_series_polars


def _to_numeric_array_polars(s: "pl.Series") -> np.ndarray:  # type: ignore[name-defined]
    if pl is None:
        raise RuntimeError("polars not available")
    try:
        # Best-effort numeric casting; keep NaN for invalid
        s2 = s.cast(pl.Float64, strict=False)
        return s2.to_numpy()
    except Exception:
        # Fallback: to_list then numpy coercion
        return np.asarray(s.to_list(), dtype="float64")


def _to_bool_array_polars(s: "pl.Series") -> List[Optional[bool]]:  # type: ignore[name-defined]
    if pl is None:
        raise RuntimeError("polars not available")
    try:
        s2 = s.cast(pl.Boolean, strict=False)
        return [None if v is None else bool(v) for v in s2.to_list()]
    except Exception:
        out: List[Optional[bool]] = []
        for v in s.to_list():
            if v is None:
                out.append(None)
            else:
                vs = str(v).strip().lower()
                if vs in {"true", "1", "t", "yes", "y"}:
                    out.append(True)
                elif vs in {"false", "0", "f", "no", "n"}:
                    out.append(False)
                else:
                    out.append(None)
        return out


def _to_datetime_ns_array_polars(s: "pl.Series") -> List[Optional[int]]:  # type: ignore[name-defined]
    if pl is None:
        raise RuntimeError("polars not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            # Force ns resolution and get integer ns since epoch
            s2 = s.dt.cast_time_unit("ns") if s.dtype == pl.Datetime else s.cast(pl.Datetime("ns"), strict=False)
            # Cast to Int64 to extract raw ns (None for nulls)
            s3 = s2.cast(pl.Int64, strict=False)
            return [None if v is None else int(v) for v in s3.to_list()]
        except Exception:
            return [None] * len(s)


def _to_categorical_iter_polars(s: "pl.Series") -> Iterable[Any]:  # type: ignore[name-defined]
    return s.to_list()


def consume_chunk_polars(df: "pl.DataFrame", accs: Dict[str, Any], kinds: ColumnKinds, logger: Optional["logging.Logger"] = None) -> None:  # type: ignore[name-defined]
    if pl is None:
        raise RuntimeError("polars not available")

    # 1) Create accumulators for columns not seen in the first chunk
    for name in df.columns:
        if name in accs:
            continue
        kind = infer_kind_for_series_polars(df[name])
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
            logger.info("➕ discovered new column '%s' inferred as %s [pl]", name, kind)

    # 2) Feed accumulators for columns present in this chunk
    for name, acc in accs.items():
        if name not in df.columns:
            if logger:
                logger.debug("column '%s' not present in this chunk; skipping [pl]", name)
            continue
        s = df[name]
        if isinstance(acc, NumericAccumulator):
            arr = _to_numeric_array_polars(s)
            acc.update(arr)
            # extremes: approximate via argpartition like pandas path
            try:
                finite = np.isfinite(arr)
                if finite.any():
                    vals = arr[finite]
                    idx = np.arange(len(arr))[finite]
                    if vals.size > 0:
                        k = min(5, vals.size)
                        part_min = np.argpartition(vals, k - 1)[:k]
                        pairs_min = [(int(idx[i]), float(vals[i])) for i in part_min]
                        part_max = np.argpartition(-vals, k - 1)[:k]
                        pairs_max = [(int(idx[i]), float(vals[i])) for i in part_max]
                        acc.update_extremes(pairs_min, pairs_max)
            except Exception:
                pass
        elif isinstance(acc, BooleanAccumulator):
            acc.update(_to_bool_array_polars(s))
            try:
                acc.add_mem(int(df.select(pl.sum(pl.all().map_batches(lambda c: c.estimated_size()))).to_numpy()[0][0]))  # rough
            except Exception:
                pass
        elif isinstance(acc, DatetimeAccumulator):
            acc.update(_to_datetime_ns_array_polars(s))
            try:
                # rough memory accounting
                acc.add_mem(int(df.estimated_size()))
            except Exception:
                pass
        else:  # categorical
            acc.update(_to_categorical_iter_polars(s))

