"""Printing an interval beside the figure it qualifies.

The intervals themselves come from `accumulators/margins.py`. This module only
decides how one is written, and there are two shapes because there are two
kinds of interval:

* `19.99 ± 0.31` when the interval sits roughly evenly around the figure, which
  is the usual case and the form a reader parses without stopping.
* `126,000 (120,541 – 126,000)` when it does not. A key column's distinct count
  is clamped at the row count, so everything it does not know is *below* it,
  and `± 2,730` would claim half of that doubt lies in a region where more
  distinct values than rows would have to exist.

**The margin is a child element, never part of the value's own text.** The
figure a script scrapes out of a stat row is the text up to the first tag, and
`float("19.99 ± 0.31")` raises. Keeping the figure first and the margin in its
own `<span>` means everything that read the cell before still reads it.

An exact figure renders nothing at all. Not `± 0`: zero is a claim about a
sketch, and a column short enough to be counted was never sketched.
"""

from __future__ import annotations

import math
from collections.abc import Callable

#: Below this ratio of the short arm to the long one, `±` would misdescribe the
#: interval and the bounds are printed instead.
_SYMMETRY = 0.5

#: Every interval on the page is at this level. Said once, on hover, rather
#: than in every label -- the caption already carries `(≈)`.
_TITLE = "95% interval"


def format_margin(value: float) -> str:
    """Two significant figures: a margin's third digit is not information."""
    if not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{round(value):,}"
    decimals = max(0, 1 - math.floor(math.log10(magnitude)))
    text = f"{value:.{decimals}f}"
    return text if 1e-4 <= magnitude else f"{value:.1e}"


def margin_html(
    value: float | None,
    lo: float | None,
    hi: float | None,
    fmt: Callable[[float], str] = format_margin,
    half: Callable[[float], str] = format_margin,
) -> str:
    """The interval on `value`, as a fragment to append after it.

    Args:
        value: The published estimate.
        lo: Lower bound of its 95% interval.
        hi: Upper bound.
        fmt: Formatter for the *bounds* when the interval is printed as a
            range.
        half: Formatter for the half-width in the `±` form.

    Returns:
        A `<span class="moe">` with a leading space, or an empty string when
        there is no interval to show -- the figure is exact, or a bound is
        missing or not finite.
    """
    try:
        value, lo, hi = float(value), float(lo), float(hi)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if not all(math.isfinite(v) for v in (value, lo, hi)) or hi <= lo:
        return ""
    below, above = max(0.0, value - lo), max(0.0, hi - value)
    if min(below, above) >= _SYMMETRY * max(below, above):
        body = f"± {half((hi - lo) / 2.0)}"
    else:
        body = f"({fmt(lo)} – {fmt(hi)})"
    return f' <span class="moe" title="{_TITLE}">{body}</span>'


def lower_bound_note(uncertainty: int | None) -> str:
    """The one-sided bound on a table of top-k counts, as a caption under it.

    Args:
        uncertainty: The weight the Misra-Gries counters shed. Zero or missing
            means every count is exact and there is nothing to say.

    Returns:
        A caption paragraph, or an empty string.

    Not a `±`: these counts can only be low, never high. And once per table
    rather than once per row, because it is one number -- every count in the
    table is short by at most the same amount. The payload has published it
    since #328; the report printed the counts beside it with no bound at all.
    """
    bound = int(uncertainty or 0)
    if bound <= 0:
        return ""
    return (
        '<p class="common__caption moe-note">counts are lower bounds · each '
        f"may be short by up to {bound:,}</p>"
    )


def count_margin_html(value: int | None, lo: int | None, hi: int | None) -> str:
    """`margin_html` for whole-number estimates.

    The half-width rounds *up*: `± 3.5` rows is not a quantity of rows, and
    rounding it down would print an interval narrower than the one computed.
    """
    try:
        crosses_zero = float(value) - (float(hi) - float(lo)) / 2.0 < 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if crosses_zero and float(hi) > float(lo):  # type: ignore[arg-type]
        # `791 ± 1,029` describes a count that can be negative. The interval
        # was already cut at zero; `±` would put the missing half back.
        return (
            f' <span class="moe" title="{_TITLE}">'
            f"({round(float(lo)):,} – {round(float(hi)):,})</span>"  # type: ignore[arg-type]
        )
    return margin_html(
        value,
        lo,
        hi,
        fmt=lambda v: f"{round(v):,}",
        half=lambda v: f"{math.ceil(v):,}",
    )
