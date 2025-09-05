"""Kind inference utilities (engine-specific adapters can wrap here)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
import re
import warnings

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
try:  # optional
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

@dataclass
class ColumnKinds:
    """Simple container for column names grouped by kind."""

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return f"ColumnKinds(num={len(self.numeric)}, cat={len(self.categorical)}, dt={len(self.datetime)}, bool={len(self.boolean)})"


def infer_kinds_pandas(df: "pd.DataFrame") -> ColumnKinds:  # type: ignore[name-defined]
    kinds = ColumnKinds()
    for name, s in df.items():
        dt = str(getattr(s, "dtype", "object"))
        if re.search("int|float|^UInt|^Int|^Float", dt, re.I):
            kinds.numeric.append(name)
        elif re.search("bool", dt, re.I):
            kinds.boolean.append(name)
        elif re.search("datetime", dt, re.I):
            kinds.datetime.append(name)
        else:
            # try coercions on a small sample for robustness
            sample = s.head(10_000)
            if pd is not None:
                # datetime?
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        ds = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
                    except TypeError:
                        ds = pd.to_datetime(sample, errors="coerce", utc=True)
                if ds.notna().sum() >= max(10, int(0.7 * len(sample))):
                    kinds.datetime.append(name)
                    continue
                ns = pd.to_numeric(sample, errors="coerce")
                if ns.notna().sum() >= max(10, int(0.7 * len(sample))):
                    kinds.numeric.append(name)
                    continue
            # boolean-like?
            uniq = set(map(str, sample.dropna().unique().tolist()))
            if 1 <= len(uniq) <= 2 and uniq.issubset({"True", "False", "0", "1", "true", "false"}):
                kinds.boolean.append(name)
            else:
                kinds.categorical.append(name)
    return kinds


def infer_kind_for_series_pandas(s: "pd.Series") -> str:  # type: ignore[name-defined]
    dt = str(getattr(s, "dtype", "object"))
    if re.search("int|float|^UInt|^Int|^Float", dt, re.I):
        return "numeric"
    if re.search("bool", dt, re.I):
        return "boolean"
    if re.search("datetime", dt, re.I):
        return "datetime"
    sample = s.head(10_000)
    if pd is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                ds = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
            except TypeError:
                ds = pd.to_datetime(sample, errors="coerce", utc=True)
        if ds.notna().sum() >= max(10, int(0.7 * len(sample))):
            return "datetime"
        ns = pd.to_numeric(sample, errors="coerce")
        if ns.notna().sum() >= max(10, int(0.7 * len(sample))):
            return "numeric"
    uniq = set(map(str, sample.dropna().unique().tolist()))
    if 1 <= len(uniq) <= 2 and uniq.issubset({"True", "False", "0", "1", "true", "false"}):
        return "boolean"
    return "categorical"


def infer_kinds_polars(df: "pl.DataFrame") -> ColumnKinds:  # type: ignore[name-defined]
    kinds = ColumnKinds()
    if pl is None:
        return kinds
    for name in df.columns:
        dt = df.schema.get(name)
        try:
            if pl.datatypes.is_numeric(dt) and not pl.datatypes.is_boolean(dt):
                kinds.numeric.append(name)
            elif pl.datatypes.is_boolean(dt):
                kinds.boolean.append(name)
            elif isinstance(dt, pl.Datetime) or dt == pl.Datetime:
                kinds.datetime.append(name)
            else:
                kinds.categorical.append(name)
        except Exception:
            kinds.categorical.append(name)
    return kinds


def infer_kind_for_series_polars(s: "pl.Series") -> str:  # type: ignore[name-defined]
    if pl is None:
        return "categorical"
    dt = s.dtype
    try:
        if pl.datatypes.is_numeric(dt) and not pl.datatypes.is_boolean(dt):
            return "numeric"
        if pl.datatypes.is_boolean(dt):
            return "boolean"
        if isinstance(dt, pl.Datetime) or dt == pl.Datetime:
            return "datetime"
    except Exception:
        pass
    return "categorical"
