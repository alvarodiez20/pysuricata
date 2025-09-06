from __future__ import annotations

from typing import Optional
import math


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(max(0, n))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024.0


def fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "NaN"
        return f"{x:,.4g}"
    except Exception:
        return str(x)


def fmt_compact(x) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
    except Exception:
        pass
    try:
        return f"{x:.4g}"
    except Exception:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return str(x)

