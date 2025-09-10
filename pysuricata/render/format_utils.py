from __future__ import annotations

import math
from typing import Optional, Union


def human_bytes(n: int) -> str:
    """Format a byte count with binary units (KiB-style thresholds).

    Uses a base of 1024 and always shows one decimal, with thousands
    separators for readability. Negative inputs are clamped to 0.

    Examples:
    - 0 -> "0.0 B"
    - 1023 -> "1,023.0 B"
    - 1024 -> "1.0 KB"
    - 1536 -> "1.5 KB"
    """
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    size = float(max(0, int(n)))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024.0


def fmt_num(x: Optional[Union[int, float]]) -> str:
    """Format a number in a compact, human-friendly way.

    - None -> em dash
    - NaN/Inf -> "NaN" (consistent with tests)
    - Otherwise uses general format with up to 4 significant digits and
      thousands separators where applicable.
    """
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "NaN"
        return f"{x:,.4g}"
    except Exception:
        return str(x)


def fmt_compact(x: object) -> str:
    """Format a value into a short numeric string when possible.

    - None or non-finite floats -> em dash
    - Tries ``format(x, '.4g')`` first; on failure, tries ``float(x)``
      then formats; finally falls back to ``str(x)``.
    """
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
    except Exception:
        # If isinstance checks fail unexpectedly, continue to best-effort
        pass
    try:
        return f"{x:.4g}"
    except Exception:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return str(x)


def fmt_compact_scientific(x: object, threshold: float = 1_000_000) -> str:
    """Format large numbers in scientific notation when they exceed threshold.

    This function provides intelligent formatting for numeric values, using scientific
    notation for large numbers to prevent axis label collisions and improve readability.

    Args:
        x: Value to format (any type that can be converted to float)
        threshold: Use scientific notation for values >= this threshold (default: 1M)

    Returns:
        Formatted string with scientific notation for large numbers, compact format otherwise

    Examples:
        >>> fmt_compact_scientific(1000)
        '1000'
        >>> fmt_compact_scientific(1000000)
        '1.00e+06'
        >>> fmt_compact_scientific(2500000.5)
        '2.50e+06'
        >>> fmt_compact_scientific(None)
        '—'
    """
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
    except Exception:
        pass

    try:
        float_val = float(x)
        if abs(float_val) >= threshold:
            # Use scientific notation for large numbers
            return f"{float_val:.2e}"
        else:
            # Use compact format for smaller numbers
            return f"{float_val:.4g}"
    except Exception:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return str(x)


def human_time(seconds: Optional[Union[int, float]]) -> str:
    """Format a time duration in seconds into a human-readable format.

    Args:
        seconds: Time duration in seconds

    Returns:
        Human-readable time string with appropriate units

    Examples:
        - 0.02 -> "0.02 s"
        - 1.5 -> "1.50 s"
        - 65 -> "1 min 5 s"
        - 3661 -> "1 h 1 min 1 s"
    """
    if seconds is None:
        return "—"

    try:
        s = float(seconds)
        if math.isnan(s) or math.isinf(s) or s < 0:
            return "—"

        # For very small times (< 60s), show precise seconds
        if s < 60:
            return f"{s:.2f} s"

        # For larger times, break down into hours, minutes, seconds
        parts = []

        # Hours
        hours = int(s // 3600)
        if hours > 0:
            parts.append(f"{hours} h")
            s %= 3600

        # Minutes
        minutes = int(s // 60)
        if minutes > 0:
            parts.append(f"{minutes} min")
            s %= 60

        # Seconds (only if non-zero or if it's the only component)
        if s > 0 or not parts:
            if s == int(s):
                parts.append(f"{int(s)} s")
            else:
                parts.append(f"{s:.2f} s")

        return " ".join(parts)

    except Exception:
        return str(seconds)


# Legacy aliases for backward compatibility
_human_bytes = human_bytes
_human_count = lambda x: f"{x:,}" if x is not None else "—"
_human_percent = lambda x: f"{x:.1f}%" if x is not None else "—"
_human_time = human_time
