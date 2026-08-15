"""Recognising a key column, so the report stops averaging one.

A monotonic, fully unique integer column is an identifier. Profiling it as an
ordinary numeric column produces a mean, a standard deviation, a skewness and a
flat uniform histogram, none of which mean anything, plus ``Zeros: 1 (0.0%)``,
which is actively misleading.

Nothing new is computed here. The numeric accumulator already tracks every
signal this needs: monotonicity, the distinct-count estimate, and whether every
value was integral.
"""

from __future__ import annotations

from typing import Any

# The distinct-count estimate is a KMV sketch, so it does not land exactly on
# the row count for a real key -- at k=2048 the standard error is about 2.2%.
# Requiring equality would mean no column above the sketch size ever qualified.
#
# 0.98 was inside that error, which made the test a coin flip on exactly the
# columns it exists for: `np.arange(20_000)`, a perfect key, estimates 19,478 --
# 0.974 of the row count -- and was profiled as a measurement with a mean. A
# threshold has to sit further from 1.0 than the estimator's own error, not
# inside it. Two standard errors, rounded.
_UNIQUENESS_TOLERANCE = 0.95

# Below this, "every value is distinct" is not evidence of anything: a 3-row
# frame of 1, 2, 3 is monotonic, integral and fully distinct, and is not a key.
_MIN_ROWS = 100


def looks_like_identifier(stats: Any) -> bool:
    """Whether a numeric column is really a key rather than a measurement.

    Args:
        stats: A ``NumericStats``-shaped object carrying ``count``, ``missing``,
            ``unique_est``, ``int_like`` and the monotonicity flags.

    Returns:
        True when the column should be presented as an identifier.
    """
    count = int(getattr(stats, "count", 0) or 0)
    if count < _MIN_ROWS:
        return False

    # A key has no gaps in its coverage; a column with nulls is not one.
    if int(getattr(stats, "missing", 0) or 0) > 0:
        return False

    if not bool(getattr(stats, "int_like", False)):
        return False

    # Sorted but repeating is not a key -- that is a grouped measurement, and
    # it is the case most likely to be mistaken for one.
    unique_est = float(getattr(stats, "unique_est", 0) or 0)
    if unique_est < count * _UNIQUENESS_TOLERANCE:
        return False

    mono_inc = bool(getattr(stats, "mono_inc", False))
    mono_dec = bool(getattr(stats, "mono_dec", False))
    return mono_inc or mono_dec


def identifier_facts(stats: Any) -> list[tuple[str, str]]:
    """The questions a key actually raises, in place of moments.

    Args:
        stats: A ``NumericStats``-shaped object.

    Returns:
        Label/value pairs for the identifier card.
    """
    count = int(getattr(stats, "count", 0) or 0)
    # The KMV estimate carries about 2.2% error at k=2048, so on a real key it
    # lands slightly either side of the row count. Reporting "3,064 distinct" in
    # 3,000 rows is arithmetically impossible and reads as a bug; clamp it, the
    # way the row-level duplicate estimate already is.
    unique_est = min(int(getattr(stats, "unique_est", 0) or 0), count)
    minimum = getattr(stats, "min", None)
    maximum = getattr(stats, "max", None)

    duplicates = max(0, count - unique_est)
    facts: list[tuple[str, str]] = [
        ("Rows", f"{count:,}"),
        ("Distinct (≈)", f"{unique_est:,}"),
        ("Duplicates (≈)", f"{duplicates:,}"),
        ("Missing", f"{int(getattr(stats, 'missing', 0) or 0):,}"),
    ]

    if minimum is not None and maximum is not None:
        facts.append(("Range", f"{minimum:,.0f} – {maximum:,.0f}"))
        # A dense key covers its range; a sparse one has gaps worth knowing about.
        span = int(maximum) - int(minimum) + 1
        if span > 0:
            gaps = max(0, span - count)
            facts.append(("Gaps in range", f"{gaps:,}"))

    direction = (
        "ascending"
        if getattr(stats, "mono_inc", False)
        else "descending"
        if getattr(stats, "mono_dec", False)
        else "unordered"
    )
    facts.append(("Order", direction))
    return facts
