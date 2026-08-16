"""The IQR fence, and the marks that crossed it.

Phase 5b.2 of the redesign (#154). An outlier is *defined* by a threshold, so
the threshold is the one graphic that explains the number. The pane this
replaces listed values with no picture of what they crossed, and opened with
roughly 60px announcing ``Low Outliers -- 0 outliers (0.0%)`` above three
severity chips all reading zero.

Two things are worth stating before the code, because both are load-bearing:

**The low side is answerable, not empty.** ``Age``'s lower fence sits at
**-6.7** years. ``min`` is 0.42, so no value in the column is below the fence
and none can be -- that is a fact, derived from two numbers already on
``stats``, and it is worth more than a block of zeroes. :func:`fence_verdict`
branches four ways over it.

**Everything here is one sample.** ``outliers_iqr``, the quartiles and
``sample_vals`` all come from the same reservoir in ``accumulators/numeric.py``,
so the count in the header and the rows in the table cannot disagree. ``min``
and ``max`` are exact, which is what makes the impossibility claim safe: it
rests on the exact extremes rather than on whether the sample happened to catch
the smallest value.

Geometry follows the histogram's rule (#147): the figure is HTML at percentage
offsets, never an SVG with text in it, so every glyph is 11px whatever width
the card gets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

from .card_config import MAD_OUTLIER_THRESHOLD, MAD_SCALE_FACTOR

#: Two marks closer than this, as a percentage of the axis, are drawn as one
#: capsule carrying a count. A mark is ~10px wide on a ~600px axis, so below
#: about this much they overlap into an anonymous blob and stop being countable.
_CLUSTER_PCT = 2.0

#: How many rows the table lists. The rest stay in the count.
_MAX_ROWS = 12

#: Distance bands, in IQRs and in MADs. Kept as they were -- the words appear
#: in two panes now (5b.5 reads them too) and a value that is `high` in one
#: cannot be `moderate` in the other.
_IQR_BANDS = ((3.0, "extreme"), (2.0, "high"))
_MAD_BANDS = ((3.5, "extreme"), (2.5, "high"))


@dataclass(frozen=True)
class Mark:
    """One outlier, or a cluster of them at indistinguishable positions."""

    value: float
    pct: float
    severity: str
    count: int = 1
    title: str = ""


@dataclass(frozen=True)
class Row:
    """One line of the table: a value and what each method makes of it."""

    index: str
    value: float
    iqr: str
    iqr_severity: str
    mad: str
    mad_severity: str


@dataclass(frozen=True)
class Fence:
    """Everything the pane needs, computed once.

    ``lo_possible``/``hi_possible`` are the deterministic half. A fence outside
    the observed range cannot be crossed by any value in the column, and that
    is decided against the exact ``min``/``max`` rather than against the
    sample.
    """

    lo: float
    hi: float
    q1: float
    q3: float
    median: float
    whisker_lo: float
    whisker_hi: float
    #: The axis, which is stretched to reach the fences -- a fence off the end
    #: of the ruler is a fence a reader cannot see.
    domain_lo: float
    domain_hi: float
    #: The column's own extremes. Distinct from the domain, and the sentence
    #: has to quote *these*: "below the minimum of 0.42" is the claim, and
    #: quoting a domain that was widened to the fence would make it circular.
    value_lo: float
    value_hi: float
    n_total: int
    #: Occurrences beyond the fence, which is what `stats.outliers_iqr` counts
    #: and therefore what the card face and the tab badge already say. The rows
    #: below are *distinct values*, so the two figures differ whenever a value
    #: repeats -- hence `n_distinct` alongside it, and the note in the table.
    n_outliers: int
    n_low: int
    n_high: int
    n_distinct: int
    marks: tuple[Mark, ...]
    rows: tuple[Row, ...]
    n_iqr: int
    n_mad: int
    rows_are_partial: bool
    any_index_missing: bool

    @property
    def lo_possible(self) -> bool:
        """Whether any value in the column is below the lower fence.

        Decided against the *exact* minimum rather than the sample, which is
        what makes "no value can cross it" a fact and not an inference from
        whichever values the reservoir happened to keep.
        """
        return self.value_lo < self.lo

    @property
    def hi_possible(self) -> bool:
        return self.value_hi > self.hi

    def pct(self, value: float) -> float:
        """Position on the axis, clamped to it."""
        span = self.domain_hi - self.domain_lo
        if span <= 0:
            return 50.0
        return min(100.0, max(0.0, (value - self.domain_lo) / span * 100.0))


def _band(distance: float, bands: tuple[tuple[float, str], ...]) -> str:
    for threshold, name in bands:
        if distance >= threshold:
            return name
    return "moderate"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_fence(stats, quantiles=None) -> Fence | None:
    """Gather the pane's facts, or ``None`` when no fence can be placed.

    A column with a zero IQR -- every value between Q1 and Q3 identical -- has
    no fence to draw, and a constant column has no axis. Both return ``None``
    rather than a degenerate figure.
    """
    q1, q3, median = stats.q1, stats.q3, stats.median
    if not (_finite(q1) and _finite(q3) and _finite(median)):
        return None

    iqr = float(q3) - float(q1)
    if not (iqr > 0):
        return None

    lo_fence = float(q1) - 1.5 * iqr
    hi_fence = float(q3) + 1.5 * iqr

    sample = [
        float(v) for v in (getattr(stats, "sample_vals", None) or []) if _finite(v)
    ]
    if not sample:
        return None

    value_lo = float(stats.min) if _finite(stats.min) else min(sample)
    value_hi = float(stats.max) if _finite(stats.max) else max(sample)
    value_lo = min(value_lo, min(sample))
    value_hi = max(value_hi, max(sample))
    # The fences are drawn, so the axis has to reach them even when no value
    # does -- a fence off the end of the ruler is a fence a reader cannot see.
    domain_lo = min(value_lo, lo_fence)
    domain_hi = max(value_hi, hi_fence)
    if not (domain_hi > domain_lo):
        return None

    mad = float(stats.mad) if _finite(stats.mad) else 0.0

    def mad_distance(value: float) -> float | None:
        if mad <= 0:
            return None
        return abs(MAD_SCALE_FACTOR * (value - float(median)) / mad)

    # One pass, one dict per value: a value flagged by both methods is one row
    # with two verdicts, which is what replaces the `rowspan` table.
    flagged: dict[float, dict] = {}
    for value in sample:
        by_iqr = value < lo_fence or value > hi_fence
        distance_mad = mad_distance(value)
        by_mad = distance_mad is not None and distance_mad > MAD_OUTLIER_THRESHOLD
        if not (by_iqr or by_mad):
            continue
        entry = flagged.setdefault(
            round(value, 12), {"value": value, "iqr": None, "mad": None}
        )
        if by_iqr:
            gap = (
                (float(q1) - value) / iqr
                if value < lo_fence
                else (value - float(q3)) / iqr
            )
            entry["iqr"] = (gap, _band(gap, _IQR_BANDS))
        if by_mad:
            entry["mad"] = (distance_mad, _band(distance_mad, _MAD_BANDS))

    index_map: dict[float, list] = {}
    for pairs in (
        getattr(stats, "min_items", None) or [],
        getattr(stats, "max_items", None) or [],
    ):
        for idx, val in pairs:
            if _finite(val):
                index_map.setdefault(round(float(val), 12), []).append(idx)

    entries = sorted(flagged.values(), key=lambda e: e["value"])

    n_iqr = sum(1 for e in entries if e["iqr"])
    n_mad = sum(1 for e in entries if e["mad"])

    # The headline counts *occurrences past the fence*, which is exactly what
    # `accumulators/numeric.py` puts in `outliers_iqr` -- so the number here,
    # the number on the card face and the number on the tab badge are one
    # number, computed the same way from the same sample. The rows below are
    # distinct values and are counted separately.
    n_low = sum(1 for v in sample if v < lo_fence)
    n_high = sum(1 for v in sample if v > hi_fence)

    rows: list[Row] = []

    # Most extreme first, which is the order a reader scans for.
    def extremity(entry: dict) -> float:
        gap_iqr = entry["iqr"][0] if entry["iqr"] else 0.0
        gap_mad = (entry["mad"][0] / MAD_OUTLIER_THRESHOLD) if entry["mad"] else 0.0
        return max(gap_iqr, gap_mad)

    any_index_missing = False
    for entry in sorted(entries, key=extremity, reverse=True)[:_MAX_ROWS]:
        key = round(entry["value"], 12)
        idxs = index_map.get(key) or []
        if not idxs:
            any_index_missing = True
        iqr_text = (
            f"{entry['iqr'][1]} · {entry['iqr'][0]:.1f}×" if entry["iqr"] else "—"
        )
        mad_text = (
            f"{entry['mad'][1]} · {entry['mad'][0]:.1f}×" if entry["mad"] else "—"
        )
        rows.append(
            Row(
                index=str(idxs[0]) if idxs else "—",
                value=entry["value"],
                iqr=iqr_text,
                iqr_severity=entry["iqr"][1] if entry["iqr"] else "none",
                mad=mad_text,
                mad_severity=entry["mad"][1] if entry["mad"] else "none",
            )
        )

    fence = Fence(
        lo=lo_fence,
        hi=hi_fence,
        q1=float(q1),
        q3=float(q3),
        median=float(median),
        whisker_lo=_whisker(quantiles, "p1", value_lo),
        whisker_hi=_whisker(quantiles, "p99", value_hi),
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        value_lo=value_lo,
        value_hi=value_hi,
        n_total=int(getattr(stats, "count", 0) or 0),
        n_outliers=n_low + n_high,
        n_low=n_low,
        n_high=n_high,
        n_distinct=len(entries),
        marks=(),
        rows=tuple(rows),
        n_iqr=n_iqr,
        n_mad=n_mad,
        rows_are_partial=len(entries) > _MAX_ROWS,
        any_index_missing=any_index_missing,
    )

    return _with_marks(fence, entries)


def _whisker(quantiles, attr: str, fallback: float) -> float:
    value = getattr(quantiles, attr, None) if quantiles is not None else None
    return float(value) if _finite(value) else fallback


def _with_marks(fence: Fence, entries: list[dict]) -> Fence:
    """Collapse marks that would overlap into one capsule with a count.

    Rule 2(e). Five values inside 0.41 years are narrower than one mark, so
    drawing five marks draws one blob and asserts nothing about how many are
    in it. The capsule says `x5` and its `title` carries the values.
    """
    marks: list[Mark] = []
    for entry in entries:
        pct = fence.pct(entry["value"])
        severity = (entry["iqr"] or entry["mad"])[1]
        if marks and pct - marks[-1].pct < _CLUSTER_PCT:
            previous = marks.pop()
            worse = _worst(previous.severity, severity)
            values = (
                f"{previous.title}, {entry['value']:g}"
                if previous.title
                else (f"{previous.value:g}, {entry['value']:g}")
            )
            marks.append(
                Mark(
                    value=entry["value"],
                    pct=previous.pct,
                    severity=worse,
                    count=previous.count + 1,
                    title=values,
                )
            )
        else:
            marks.append(Mark(value=entry["value"], pct=pct, severity=severity))

    return Fence(**{**fence.__dict__, "marks": tuple(marks)})


_SEVERITY_ORDER = {"moderate": 0, "high": 1, "extreme": 2}


def _worst(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b


def fence_verdict(fence: Fence, fmt) -> str:
    """The sentence that replaces the empty block. Four cases, all reachable.

    1. Nothing crosses either fence.
    2. Only the high side can be crossed -- the ``Age`` case, where the lower
       fence is below the column's minimum.
    3. Only the low side can be crossed.
    4. Both sides are crossable.

    ``fmt`` is the card's number formatter, passed in so the fence reads in the
    same notation as every other figure on the card.
    """
    lo, hi = fmt(fence.lo), fmt(fence.hi)

    if fence.n_outliers == 0:
        return (
            f"No value lies outside the fence at {lo} and {hi}, "
            "so this column has no outliers by the IQR rule."
        )

    if not fence.lo_possible:
        plural = "are" if fence.n_outliers != 1 else "is"
        return (
            f"All {fence.n_outliers:,} {plural} high. The lower fence sits at {lo}, "
            f"below the minimum of {fmt(fence.value_lo)}, so no value can cross it."
        )

    if not fence.hi_possible:
        plural = "are" if fence.n_outliers != 1 else "is"
        return (
            f"All {fence.n_outliers:,} {plural} low. The upper fence sits at {hi}, "
            f"above the maximum of {fmt(fence.value_hi)}, so no value can cross it."
        )

    return (
        f"{fence.n_low:,} below the fence at {lo} and {fence.n_high:,} above it "
        f"at {hi}. Both tails cross."
    )


def method_note(fence: Fence) -> str:
    """Why two columns of verdicts disagree, said once instead of implied.

    The old pane printed two sets of severity chips and left the reader to
    reconcile them. They cannot be reconciled: the fence asks how far a value
    is from the middle *half* of the data, the modified z-score asks how far it
    is from the median in units of typical deviation. Both answers are printed
    because both are true.
    """
    if fence.n_iqr and fence.n_mad and fence.n_iqr != fence.n_mad:
        return (
            f"IQR flags {fence.n_iqr:,}; MAD flags {fence.n_mad:,}. They ask "
            "different questions — distance from the middle half of the data, "
            "against distance from the median in typical deviations — so both "
            "verdicts are printed rather than reconciled."
        )
    if fence.n_iqr and not fence.n_mad:
        return (
            f"IQR flags {fence.n_iqr:,}; MAD flags none. The modified z-score "
            "measures distance from the median in typical deviations, and by "
            "that measure nothing here is far enough."
        )
    if fence.n_mad and not fence.n_iqr:
        return (
            f"MAD flags {fence.n_mad:,}; the IQR fence flags none. The two "
            "methods disagree because they measure different things."
        )
    return (
        f"Both methods flag the same {fence.n_iqr:,}. They ask different "
        "questions and happen to agree on this column."
    )


def render_figure(fence: Fence, name: str, fmt) -> str:
    """The fence, the box, and the marks beyond it.

    HTML at percentage offsets rather than an SVG, for the reason set out in
    ``_07-histogram.css``: a glyph inside a stretched SVG is scaled by the
    ratio of the two axes, and there is no canvas size that is right at both
    284px and 820px.
    """
    label_row: list[str] = []
    track: list[str] = []

    lo_pct, hi_pct = fence.pct(fence.lo), fence.pct(fence.hi)
    q1_pct, q3_pct = fence.pct(fence.q1), fence.pct(fence.q3)

    track.append(
        f'<span class="fence__whisker" style="left:{fence.pct(fence.whisker_lo):.3f}%;'
        f'width:{max(0.0, q1_pct - fence.pct(fence.whisker_lo)):.3f}%"></span>'
    )
    track.append(
        f'<span class="fence__whisker" style="left:{q3_pct:.3f}%;'
        f'width:{max(0.0, fence.pct(fence.whisker_hi) - q3_pct):.3f}%"></span>'
    )
    track.append(
        f'<span class="fence__box" style="left:{q1_pct:.3f}%;'
        f'width:{max(0.0, q3_pct - q1_pct):.3f}%"></span>'
    )

    for crossable, pct, value in (
        (fence.lo_possible, lo_pct, fence.lo),
        (fence.hi_possible, hi_pct, fence.hi),
    ):
        if not crossable:
            continue
        track.append(f'<span class="fence__line" style="left:{pct:.3f}%"></span>')
        anchor = "end" if pct > 50 else "start"
        label_row.append(
            f'<span class="fence__fencelabel" data-anchor="{anchor}" '
            f'style="left:{pct:.3f}%">fence {escape(fmt(value))}</span>'
        )

    track.append(
        f'<span class="fence__median" style="left:{fence.pct(fence.median):.3f}%"></span>'
    )

    for mark in fence.marks:
        title = mark.title or fmt(mark.value)
        track.append(
            f'<span class="fence__mark" data-severity="{mark.severity}" '
            f'title="{escape(title)}" style="left:{mark.pct:.3f}%"></span>'
        )
        if mark.count > 1:
            label_row.append(
                f'<span class="fence__count" style="left:{mark.pct:.3f}%">'
                f"×{mark.count}</span>"
            )

    ticks = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = fence.domain_lo + (fence.domain_hi - fence.domain_lo) * fraction
        anchor = "start" if fraction == 0.0 else ("end" if fraction == 1.0 else "")
        ticks.append(
            f'<span class="fence__tick" data-anchor="{anchor}" '
            f'style="left:{fraction * 100:.3f}%">{escape(fmt(value))}</span>'
        )

    described = (
        f"{name} values with the IQR fence and {fence.n_outliers:,} "
        f"value{'s' if fence.n_outliers != 1 else ''} beyond it"
    )

    return (
        f'<div class="fence" role="img" aria-label="{escape(described)}">'
        f'<div class="fence__labels">{"".join(label_row)}</div>'
        f'<div class="fence__track">{"".join(track)}</div>'
        f'<div class="fence__axis"></div>'
        f'<div class="fence__ticks">{"".join(ticks)}</div>'
        f"</div>"
        f'<ul class="fence__legend">'
        f'<li><span class="key key--box"></span>IQR and P1–P99</li>'
        f'<li><span class="key key--median"></span>median {escape(fmt(fence.median))}</li>'
        f'<li><span class="key key--mark" data-severity="moderate"></span>moderate</li>'
        f'<li><span class="key key--mark" data-severity="extreme"></span>high or extreme</li>'
        f"</ul>"
    )


def render_table(fence: Fence, fmt) -> str:
    """One row per value, both verdicts side by side.

    The `rowspan` this replaces gave a value flagged by both methods two rows
    and a value flagged by one a single row, so the table's shape encoded
    something other than the data.
    """
    head = (
        '<div class="fence-table__head">'
        "<span>Row</span><span>Value</span><span>By IQR</span><span>By MAD</span>"
        "</div>"
    )
    rows = "".join(
        f'<div class="fence-table__row">'
        f'<span class="fence-table__idx">{escape(row.index)}</span>'
        f'<span class="fence-table__val">{escape(fmt(row.value))}</span>'
        f'<span class="fence-table__verdict" data-severity="{row.iqr_severity}">'
        f"{escape(row.iqr)}</span>"
        f'<span class="fence-table__verdict" data-severity="{row.mad_severity}">'
        f"{escape(row.mad)}</span>"
        f"</div>"
        for row in fence.rows
    )

    notes = []
    if fence.rows_are_partial:
        notes.append(
            f"{len(fence.rows)} of {fence.n_outliers:,} shown, the most extreme first."
        )
    if fence.any_index_missing:
        notes.append("A dash means the row index was not tracked.")
    note = (
        f'<p class="fence-table__note">{escape(" ".join(notes))}</p>' if notes else ""
    )

    return f'<div class="fence-table">{head}{rows}</div>{note}'
