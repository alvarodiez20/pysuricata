"""Chunk consumption and accumulator wiring for pandas chunks."""

from __future__ import annotations

import logging
import math
import re
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

try:  # optional
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as pd  # noqa: F811

from ..accumulators import (
    BooleanAccumulator,
    CategoricalAccumulator,
    DatetimeAccumulator,
    NumericAccumulator,
)
from .core.types import ColumnKinds
from .processing.inference import UnifiedTypeInferrer

# Per-column memory estimates keyed by (column_name, dtype_string).
# Module-level so it persists across chunks of the same profile() call but
# is naturally scoped to the process; keying by dtype prevents stale reuse
# when the same column name appears with a different dtype in a later call.
_memory_cache: dict[tuple[str, str], float] = {}


def _mean_label_length(lengths: pd.Series) -> float:
    """The mean of a per-value length series, with all-missing meaning zero.

    `Series.mean()` of an all-NA series is NaN, and a NaN estimate propagates
    to `int(estimate * len(s))`, which raises `ValueError: cannot convert float
    NaN to integer` -- the whole profile fails on one all-missing column.

    Reaching that state depends on the pandas version. Under pandas 2,
    `Series([None]).astype(str)` produced the literal string `"None"`, so an
    all-missing object column measured 4 characters per row and the mean was
    never NaN. Under pandas 3 the same call yields NaN, so it always is. The
    guard belongs here either way: a column holding no strings carries no
    string payload, and zero is the honest answer to "how many bytes of text
    per row", where 4 was an artifact of stringifying a missing value.
    """
    mean = lengths.mean()
    return 0.0 if pd.isna(mean) else float(mean)


def _estimate_memory_per_row_fast(s: pd.Series) -> float:
    """Fast memory estimation based on dtype instead of deep profiling.

    This avoids the expensive memory_usage(deep=True) which traverses every string.

    Args:
        s: pandas Series to estimate memory for

    Returns:
        Estimated bytes per row
    """
    dtype = s.dtype

    # Fast dtype-based estimation
    if dtype == "object":
        # For object columns, estimate based on sample
        if len(s) > 0:
            # Sample first 100 values to estimate average string length
            sample_size = min(100, len(s))
            sample = s.head(sample_size)
            # Rough estimate: 8 bytes overhead + average string length
            return 8 + _mean_label_length(sample.astype(str).str.len())
        return 8  # Default for empty series
    elif dtype == "string":
        # String dtype - estimate based on sample
        if len(s) > 0:
            sample_size = min(100, len(s))
            sample = s.head(sample_size)
            return _mean_label_length(sample.str.len())
        return 8
    else:
        # Numeric/datetime types - use dtype size
        return dtype.itemsize


def _to_numeric_array_pandas(s: pd.Series) -> np.ndarray:
    """Best-effort fast path to float64 NumPy array with NaN for invalid.

    - If the Series is already numeric (including pandas nullable ints),
      avoid the overhead of `pd.to_numeric` and go straight to NumPy.
    - Timedelta/Duration dtypes are converted to total seconds.
    - Otherwise, coerce with `pd.to_numeric(errors='coerce')`.
    """
    try:
        if pd is not None:
            from pandas.api import types as pdt  # type: ignore

            dt = getattr(s, "dtype", None)
            if dt is not None:
                # Timedelta → total seconds (float64)
                if pdt.is_timedelta64_dtype(dt):
                    return s.dt.total_seconds().to_numpy(dtype="float64", copy=False)

                # Fast path for numeric dtypes (exclude booleans)
                if not pdt.is_bool_dtype(dt) and pdt.is_numeric_dtype(dt):
                    return s.to_numpy(dtype="float64", copy=False)

                # ArrowDtype duration → total seconds
                if hasattr(dt, "pyarrow_dtype"):
                    try:
                        import pyarrow as pa

                        if pa.types.is_duration(dt.pyarrow_dtype):
                            # Convert arrow duration to pandas timedelta then total_seconds
                            td = s.dt.total_seconds()
                            return td.to_numpy(dtype="float64", copy=False)
                    except Exception:
                        pass
    except Exception:
        # Fall through to the coercion path on any failure
        pass
    try:
        ns = pd.to_numeric(s, errors="coerce")
        return ns.to_numpy(dtype="float64", copy=False)
    except Exception:
        # Last resort: NumPy coercion (may be slower for object dtype)
        return np.asarray(
            getattr(s, "to_numpy", lambda: np.asarray(s))(), dtype="float64"
        )


def _to_bool_array_pandas(s: pd.Series) -> list[bool | None]:
    if str(s.dtype).startswith("bool"):
        arr = s.astype("boolean").tolist()
        return [None if x is pd.NA else bool(x) for x in arr]

    # Vectorized boolean coercion using pandas string operations
    try:
        lower = s.astype(str).str.strip().str.lower()
        true_mask = lower.isin({"true", "1", "t", "yes", "y"})
        false_mask = lower.isin({"false", "0", "f", "no", "n"})

        # Build result array using numpy for speed
        # Default None, set True/False based on masks
        true_np = true_mask.to_numpy()
        false_np = false_mask.to_numpy()
        result: list[bool | None] = [None] * len(s)
        true_indices = np.where(true_np)[0]
        false_indices = np.where(false_np)[0]
        for i in true_indices:
            result[i] = True
        for i in false_indices:
            result[i] = False
        return result
    except Exception:
        # Fallback to per-value coercion for edge cases
        def _coerce(v: Any) -> bool | None:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            vs = str(v).strip().lower()
            if vs in {"true", "1", "t", "yes", "y"}:
                return True
            if vs in {"false", "0", "f", "no", "n"}:
                return False
            return None

        return [_coerce(v) for v in s.tolist()]


def _to_datetime_ns_array_pandas(s: pd.Series) -> np.ndarray:
    """Convert a column to int64 nanoseconds, NaT included as its sentinel.

    Returns the int64 array rather than a list of ``int | None``. Building that
    list meant a ``.tolist()`` plus a Python-level comparison per row purely to
    turn one out-of-range sentinel into ``None`` -- and the accumulator's
    validity window rejects the sentinel anyway, so the boxing bought nothing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            ds = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        except TypeError:
            ds = pd.to_datetime(s, errors="coerce", utc=True)
    # `copy=` is deprecated under pandas 3 (copy-on-write makes it a no-op).
    return _as_int64_nanoseconds(ds.astype("int64").to_numpy(), _datetime_unit(ds))


#: int64 nanoseconds per unit of each resolution pandas can carry a datetime in.
_NS_PER_UNIT = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}

#: What pandas stores for NaT once the column is viewed as int64.
_NAT_INT64 = np.iinfo(np.int64).min


def _datetime_unit(ds: pd.Series) -> str:
    """The resolution a datetime Series is stored at, as a bare unit string.

    `utc=True` gives a `DatetimeTZDtype`, which carries `.unit` directly; the
    string fallback covers a tz-naive `datetime64[us]` reaching here by another
    route. Anything unrecognised is treated as nanoseconds, which is what this
    function assumed unconditionally before.
    """
    unit = getattr(ds.dtype, "unit", None)
    if unit in _NS_PER_UNIT:
        return unit
    match = re.search(r"\[(\w+)", str(ds.dtype))
    return match.group(1) if match and match.group(1) in _NS_PER_UNIT else "ns"


def _as_int64_nanoseconds(values: np.ndarray, unit: str) -> np.ndarray:
    """Rescale int64 datetime values of any resolution to nanoseconds.

    This used to be `astype("int64")` and nothing else, which was correct only
    because pandas 2 stored every datetime at nanosecond resolution. **pandas 3
    defaults to microseconds** -- `pd.date_range(...)` returns
    `datetime64[us]` -- so the same cast silently returned microseconds, and
    every datetime statistic downstream came out a factor of 1,000 wrong while
    still looking entirely plausible. A freshness check read a 2020 timestamp
    as 1970.

    Nanoseconds span roughly 1677-2262. A coarser column can hold dates outside
    that, and those cannot be represented at all, so they become NaT -- the
    sentinel the accumulator's validity window already rejects. Saturating is
    the honest failure: an out-of-range date is missing from the ns view, not
    silently wrapped into a plausible wrong one.
    """
    scale = _NS_PER_UNIT.get(unit, 1)
    if scale == 1:
        return values

    limit = np.iinfo(np.int64).max // scale
    # NaT is int64 min; it is out of range by construction and must stay NaT
    # rather than be scaled into a real date.
    representable = (values != _NAT_INT64) & (np.abs(values) <= limit)
    out = np.full(values.shape, _NAT_INT64, dtype=np.int64)
    np.multiply(values, scale, out=out, where=representable)
    return out


def _to_categorical_iter_pandas(s: pd.Series) -> Iterable[Any]:
    return s.tolist()


def consume_chunk_pandas(
    df: pd.DataFrame,
    accs: dict[str, Any],
    kinds: ColumnKinds,
    config: Any | None = None,
    logger: logging.Logger | None = None,
    *,
    row_offset: int = 0,
) -> None:
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
        cache_key = (name, str(s.dtype))
        if cache_key not in _memory_cache:
            try:
                # Use fast dtype-based estimation instead of expensive deep profiling
                estimate = _estimate_memory_per_row_fast(s)
            except Exception:
                estimate = 0
            # A bytes-per-row estimate is a finite, non-negative number. It used
            # to be neither only in an unreachable-looking corner, and the cost
            # of landing there was the whole profile raising out of `int()` --
            # a memory *estimate* failing a run whose statistics were fine.
            if not math.isfinite(estimate) or estimate < 0:
                estimate = 0
            _memory_cache[cache_key] = estimate

        # Use cached memory estimate
        estimated_memory = int(_memory_cache[cache_key] * len(s))

        # Dispatch on the accumulator's own `kind` rather than on its type. A
        # PyO3 accumulator satisfies no `isinstance` check here, and the order
        # of the old chain was load-bearing without saying so.
        kind = acc.kind
        if kind == "numeric":
            arr = _to_numeric_array_pandas(s)
            # row_offset makes the accumulator's extreme-value indices global.
            acc.update(arr, row_offset=row_offset)
            acc.add_mem(estimated_memory)
        elif kind == "boolean":
            arr = _to_bool_array_pandas(s)
            acc.update(arr)
            acc.add_mem(estimated_memory)
        elif kind == "datetime":
            arr = _to_datetime_ns_array_pandas(s)
            acc.update(arr)
            acc.add_mem(estimated_memory)
        elif kind == "categorical":
            acc.update(_to_categorical_iter_pandas(s))
            acc.add_mem(estimated_memory)
