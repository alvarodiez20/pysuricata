from __future__ import annotations

import math
import numpy as np


def safe_col_id(name: str) -> str:
    return "col_" + "".join(ch if str(ch).isalnum() else "_" for ch in str(name))


def _nice_num(rng: float, do_round: bool = True) -> float:
    if rng <= 0 or not np.isfinite(rng):
        return 1.0
    exp = math.floor(math.log10(rng))
    frac = rng / (10 ** exp)
    if do_round:
        if frac < 1.5:
            nice = 1
        elif frac < 3:
            nice = 2
        elif frac < 7:
            nice = 5
        else:
            nice = 10
    else:
        if frac <= 1:
            nice = 1
        elif frac <= 2:
            nice = 2
        elif frac <= 5:
            nice = 5
        else:
            nice = 10
    return nice * (10 ** exp)


def nice_ticks(vmin: float, vmax: float, n: int = 5):
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        vmax = vmin + 1
    rng = _nice_num(vmax - vmin, do_round=False)
    step = _nice_num(rng / max(1, n - 1), do_round=True)
    nice_min = math.floor(vmin / step) * step
    nice_max = math.ceil(vmax / step) * step
    ticks = []
    t = nice_min
    while t <= nice_max + step * 1e-9 and len(ticks) < 50:
        ticks.append(t)
        t += step
    return ticks, step


def fmt_tick(v: float, step: float) -> str:
    if not np.isfinite(v):
        return ''
    if step >= 1:
        i = int(round(v))
        if abs(i) >= 1000:
            return f"{i:,}"
        return f"{i}"
    if step >= 0.1:
        return f"{v:.1f}"
    if step >= 0.01:
        return f"{v:.2f}"
    try:
        return f"{v:.4g}"
    except Exception:
        return str(v)


def svg_empty(css_class: str, width: int, height: int, aria_label: str = "no data") -> str:
    return f'<svg class="{css_class}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="{aria_label}"></svg>'

