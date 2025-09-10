from __future__ import annotations

"""Small SVG/axis helpers shared by the HTML card renderers.

The utilities here are intentionally lightweight and pure so they can be reused
across numeric and datetime charts. They avoid heavyweight dependencies and
favor predictable output for testing.
"""

import math
import re
from typing import List, Tuple

import numpy as np


def safe_col_id(name: str) -> str:
    """Return a safe DOM id for a column name.

    Non-alphanumeric characters are replaced with underscores and the result is
    prefixed with ``col_`` to ensure it never starts with a digit. Consecutive
    separators are collapsed to a single underscore.

    Args:
        name: Column name as found in the DataFrame.

    Returns:
        A string suitable for use as an HTML id attribute.
    """
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name))
    s = s.strip("_") or "col"
    return f"col_{s}"


def _nice_num(rng: float, do_round: bool = True) -> float:
    """Return a "nice" rounded number roughly equal to ``rng``.

    The function snaps values to the set {1, 2, 5, 10} × 10^k. When
    ``do_round`` is False the function returns a ceiling-like value to avoid
    undershooting the range.

    Args:
        rng: Target range (must be positive and finite to matter).
        do_round: Whether to round to the nearest nice number (True) or select
            a ceiling-like nice number (False).

    Returns:
        A positive "nice" float. Defaults to 1.0 on invalid input.
    """
    if rng <= 0 or not np.isfinite(rng):
        return 1.0
    exp = math.floor(math.log10(rng))
    frac = rng / (10**exp)
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
    return nice * (10**exp)


def nice_ticks(vmin: float, vmax: float, n: int = 5) -> Tuple[List[float], float]:
    """Compute "nice" axis ticks covering [vmin, vmax].

    Produces at most ~50 ticks, evenly spaced using a "nice" step. If the
    interval is degenerate (vmin == vmax), it expands by 1 to avoid zero range.

    Args:
        vmin: Minimum value.
        vmax: Maximum value.
        n: Target number of ticks (>= 1).

    Returns:
        A tuple ``(ticks, step)`` where ``ticks`` is a list of float tick
        locations and ``step`` is the spacing between adjacent ticks.
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        vmax = vmin + 1
    n = max(1, int(n))
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
    """Format a numeric tick value based on the step size.

    Uses thousands separators for large integers and limits decimals for small
    steps. Returns an empty string for non-finite inputs.

    Args:
        v: Tick value.
        step: Step size used to select formatting precision.

    Returns:
        A short, human-friendly string representation of the tick.
    """
    if not np.isfinite(v):
        return ""
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


def _format_pow10_label(exp: int) -> str:
    """Format a 10^exp label as human-friendly text.

    For small positive exponents, returns a plain integer with separators
    (e.g., 1_000). For small negative exponents, returns a decimal string
    (e.g., 0.001). For larger magnitudes, returns scientific notation (1e±k).

    Args:
        exp: Integer exponent for 10**exp.

    Returns:
        str: Label representing 10**exp.
    """
    try:
        if -6 <= exp <= 6:
            if exp >= 0:
                return f"{int(10 ** exp):,}"
            # exp < 0: generate decimal with leading 0.
            frac = 10.0 ** exp
            # choose precision to show at least |exp| decimals
            prec = min(8, max(1, -exp))
            s = f"{frac:.{prec}f}"
            # strip trailing zeros
            s = s.rstrip('0').rstrip('.') if '.' in s else s
            return s
        return f"1e{exp:+d}".replace("+", "")
    except Exception:
        return f"1e{exp}"


def nice_log_ticks_from_log10(log_min: float, log_max: float, max_ticks: int = 6) -> Tuple[List[float], List[str]]:
    """Compute nice log-scale ticks given log10-domain bounds.

    Produces ticks at integer powers of ten (spaced by a step chosen so the
    total count does not exceed ``max_ticks``). Returns both the tick
    positions in log10 space and their human-friendly labels in linear space.

    Args:
        log_min: Minimum bound in log10 space.
        log_max: Maximum bound in log10 space.
        max_ticks: Maximum number of major ticks to generate (>= 1).

    Returns:
        A tuple ``(ticks, labels)`` where ``ticks`` are log10 positions and
        ``labels`` are strings for the corresponding linear values.
    """
    if not (np.isfinite(log_min) and np.isfinite(log_max)):
        return [0.0, 1.0], ["1", "10"]
    if log_max < log_min:
        log_min, log_max = log_max, log_min
    if log_max == log_min:
        log_max = log_min + 1.0
    max_ticks = max(1, int(max_ticks))
    emin = int(math.floor(log_min))
    emax = int(math.ceil(log_max))
    span = max(1, emax - emin)
    step_e = max(1, int(math.ceil(span / max(1, max_ticks - 1))))
    ticks: List[float] = []
    labels: List[str] = []
    e = emin
    while e <= emax and len(ticks) < 50:
        ticks.append(float(e))
        labels.append(_format_pow10_label(e))
        e += step_e
    return ticks, labels


def svg_empty(css_class: str, width: int, height: int, aria_label: str = "no data") -> str:
    """Return a minimal empty SVG placeholder with a viewBox.

    Args:
        css_class: CSS class to apply to the ``<svg>`` element.
        width: Pixel width.
        height: Pixel height.
        aria_label: Accessible label describing the placeholder.

    Returns:
        A string containing an empty SVG element.
    """
    return (
        f'<svg class="{css_class}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" aria-label="{aria_label}"></svg>'
    )
