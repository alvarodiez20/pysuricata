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

import numpy as np

from .card_config import MAD_OUTLIER_THRESHOLD, MAD_SCALE_FACTOR

#: Two marks closer than this, as a percentage of the axis, are drawn as one
#: capsule carrying a count. A mark is ~10px wide on a ~600px axis, so below
#: about this much they overlap into an anonymous blob and stop being countable.
_CLUSTER_PCT = 2.0

#: How many rows the table lists. The rest stay in the count.
_MAX_ROWS = 12

#: Minimum gap between two *labels* on the axis, as a percentage of it. Marks
#: may sit 2% apart and still be countable as capsules; their labels cannot --
#: `×14` is about 24px and 2% of a 1,099px axis is 22.
_LABEL_MIN_GAP_PCT = 7.0

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
    #: Read only by the quantile strip, where it is the caret sitting above the
    #: band. It lands ~24px from the median inside a dark fill, so the two are
    #: told apart by shape rather than colour -- rule 2 of the token system.
    mean: float
    whisker_lo: float
    whisker_hi: float
    #: The axis: the column's own range, and nothing wider.
    #:
    #: An earlier version stretched it to reach both fences, on the reasoning
    #: that a fence off the end of the ruler cannot be seen. That was solving a
    #: problem that does not exist -- a fence is only *drawn* when a value
    #: crosses it, and a crossed fence is inside the data range by definition.
    #: What the stretch did instead was put `-6.688` at the left end of `Age`,
    #: an age no row holds, presented as where the data starts, and spend 9% of
    #: the width getting there.
    value_lo: float
    value_hi: float
    n_total: int
    #: How many values the fence was actually fitted over: the reservoir, which
    #: is the whole column only below `numeric_sample_size`. `n_outliers` is a
    #: count *within* it, so this is the denominator its percentage takes.
    #: Dividing by `n_total` instead reported a 10% outlier column as 0.2% at a
    #: million rows, the same sample-over-population slip as #327 itself.
    n_sampled: int
    #: Occurrences beyond the fence, counted in the sample above. The card face
    #: and the payload carry the scaled population estimate; this pane stays on
    #: the sample, because the rows below are those very values.
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
    def is_sampled(self) -> bool:
        """Whether the fence saw fewer values than the column holds.

        False when the reservoir held every value, in which case the pane's
        count and the card's figure are the same number and there is nothing to
        reconcile -- so the note stays off rather than stating the obvious.
        """
        return 0 < self.n_sampled < self.n_total

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
        """Position on the axis, clamped to it.

        The clamp matters for the whiskers: `P1` and `P99` come from the same
        sample the axis does, so they land inside it, but a caller passing an
        uncrossable fence would otherwise position it off the figure.
        """
        span = self.value_hi - self.value_lo
        if span <= 0:
            return 50.0
        return min(100.0, max(0.0, (value - self.value_lo) / span * 100.0))


#: How close to an end a label has to be before it anchors there instead of
#: centring on its mark. 2% of a 1,099px axis is 22px, about one label wide.
_ANCHOR_PCT = 2.0


def _anchor(pct: float) -> str:
    """The `data-anchor` for a label at this position, or nothing."""
    if pct <= _ANCHOR_PCT:
        return ' data-anchor="start"'
    if pct >= 100.0 - _ANCHOR_PCT:
        return ' data-anchor="end"'
    return ""


def _band(distance: float, bands: tuple[tuple[float, str], ...]) -> str:
    for threshold, name in bands:
        if distance >= threshold:
            return name
    return "moderate"


#: Returned for a missing reservoir, so the caller's `.size` check is uniform.
_EMPTY = np.empty(0, dtype=np.float64)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _sample_array(reservoir) -> np.ndarray:
    """The reservoir's finite values as a float array.

    `sample_vals` is already a float64 array (#207), so the common path is a
    no-op view. The fallback covers a caller that hands in a plain sequence,
    which the tests and any external consumer of these dataclasses may.
    """
    if reservoir is None:
        return _EMPTY
    try:
        values = np.asarray(reservoir, dtype=float)
    except (TypeError, ValueError):
        values = np.asarray([float(v) for v in reservoir if _finite(v)], dtype=float)
    # No reshape: a boolean mask over an array of any shape already returns a
    # flat one, so the guard that used to sit here could never run.
    return values[np.isfinite(values)]


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

    sample = _sample_array(getattr(stats, "sample_vals", None))
    if sample.size == 0:
        return None
    sample_lo, sample_hi = float(sample.min()), float(sample.max())

    value_lo = float(stats.min) if _finite(stats.min) else sample_lo
    value_hi = float(stats.max) if _finite(stats.max) else sample_hi
    value_lo = min(value_lo, sample_lo)
    value_hi = max(value_hi, sample_hi)
    if not (value_hi > value_lo):
        return None

    mad = float(stats.mad) if _finite(stats.mad) else 0.0

    # Decided over the whole sample at once, then walked only where something
    # was flagged. Element-wise, this was three Python passes over 20,000
    # values per numeric column and the single largest cost in rendering a
    # wide frame -- 7.8s of a 13.3s profile at 60 columns (#207).
    below = sample < lo_fence
    above = sample > hi_fence
    by_iqr_mask = below | above
    if mad > 0:
        mad_distance = np.abs(MAD_SCALE_FACTOR * (sample - float(median)) / mad)
        by_mad_mask = mad_distance > MAD_OUTLIER_THRESHOLD
    else:
        mad_distance = None
        by_mad_mask = np.zeros(sample.shape, dtype=bool)

    # One dict per value: a value flagged by both methods is one row with two
    # verdicts, which is what replaces the `rowspan` table.
    flagged: dict[float, dict] = {}
    for i in np.flatnonzero(by_iqr_mask | by_mad_mask):
        value = float(sample[i])
        entry = flagged.setdefault(
            round(value, 12), {"value": value, "iqr": None, "mad": None}
        )
        if by_iqr_mask[i]:
            gap = (float(q1) - value) / iqr if below[i] else (value - float(q3)) / iqr
            entry["iqr"] = (gap, _band(gap, _IQR_BANDS))
        if by_mad_mask[i]:
            distance = float(mad_distance[i])
            entry["mad"] = (distance, _band(distance, _MAD_BANDS))

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
    n_low = int(below.sum())
    n_high = int(above.sum())

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
        mean=float(stats.mean) if _finite(stats.mean) else float("nan"),
        whisker_lo=_whisker(quantiles, "p1", value_lo),
        whisker_hi=_whisker(quantiles, "p99", value_hi),
        value_lo=value_lo,
        value_hi=value_hi,
        n_total=int(getattr(stats, "count", 0) or 0),
        n_sampled=int(sample.size),
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
    marks = cluster_marks(
        fence,
        [(entry["value"], (entry["iqr"] or entry["mad"])[1]) for entry in entries],
    )
    return Fence(**{**fence.__dict__, "marks": marks})


def cluster_marks(fence: Fence, values: list[tuple[float, str]]) -> tuple[Mark, ...]:
    """Collapse marks that would overlap into one capsule with a count.

    Rule 2(e). Five values inside 0.41 years are narrower than one mark, so
    drawing five marks draws one blob and asserts nothing about how many are
    in it. The capsule says `x5` and its `title` carries the values.

    Shared with the Min/Max pane (5b.5), where the same five lowest values on
    `Age` fall inside 0.41 years and collapse to a single `x5`.
    """
    marks: list[Mark] = []
    for value, severity in sorted(values):
        pct = fence.pct(value)
        if marks and pct - marks[-1].pct < _CLUSTER_PCT:
            previous = marks.pop()
            values_seen = (
                f"{previous.title}, {value:g}"
                if previous.title
                else f"{previous.value:g}, {value:g}"
            )
            marks.append(
                Mark(
                    value=value,
                    pct=previous.pct,
                    severity=_worst(previous.severity, severity),
                    count=previous.count + 1,
                    title=values_seen,
                )
            )
        else:
            marks.append(Mark(value=value, pct=pct, severity=severity))

    return tuple(marks)


#: `inside` is the Min/Max pane's addition and sorts below every warm band: a
#: cluster holding one outlier and four ordinary values is an outlier cluster,
#: never the other way round.
_SEVERITY_ORDER = {"inside": -1, "moderate": 0, "high": 1, "extreme": 2}


def _worst(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b


def classify(fence: Fence, value: float) -> tuple[str, str]:
    """``(severity, phrase)`` for one value against the fence.

    The single source of the severity words, which is why 5b.5 calls it rather
    than re-deriving them: a value that reads `high` in the Outliers pane
    cannot read `moderate` in Min/Max, and no amount of care in two
    implementations guarantees that.
    """
    iqr = fence.q3 - fence.q1
    if iqr <= 0:
        return "inside", "inside the fence"

    if value < fence.lo:
        gap = (fence.q1 - value) / iqr
    elif value > fence.hi:
        gap = (value - fence.q3) / iqr
    else:
        return "inside", "inside the fence"

    band = _band(gap, _IQR_BANDS)
    return band, f"{band} · {gap:.1f}× IQR"


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
        f"IQR flags {fence.n_iqr:,} and MAD flags {fence.n_mad:,}. They ask "
        "different questions and happen to agree on this column."
    )


def render_figure(
    fence: Fence,
    name: str,
    fmt,
    marks: tuple[Mark, ...] | None = None,
    described: str = "",
    legend: str = "",
) -> str:
    """The fence, the box, and whatever marks the caller wants on it.

    Two panes draw this. The Outliers pane marks what crossed the fence; the
    Min/Max pane marks the five lowest and five highest, which is how a reader
    sees that **every one of `Age`'s five maxima is an outlier and not one of
    its five minima is** -- the whole story of that column's tails, and
    invisible in the two tables of bare index-and-value it replaces.

    Sharing the figure is not tidiness. The design requires a reader who has
    opened one pane to be able to read the other, which means the same axis,
    the same box, the same fence position and the same severity colours.

    HTML at percentage offsets rather than an SVG, for the reason set out in
    ``_07-histogram.css``: a glyph inside a stretched SVG is scaled by the
    ratio of the two axes, and there is no canvas size that is right at both
    284px and 820px.
    """
    marks = fence.marks if marks is None else marks
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

    # The fence labels are placed first and never dropped: the fence is what
    # the pane is about, and a count label is a detail the table repeats.
    taken: list[float] = []
    for crossable, pct, value in (
        (fence.lo_possible, lo_pct, fence.lo),
        (fence.hi_possible, hi_pct, fence.hi),
    ):
        if not crossable:
            continue
        track.append(f'<span class="fence__line" style="left:{pct:.3f}%"></span>')
        anchor = _anchor(pct) or (
            ' data-anchor="end"' if pct > 50 else ' data-anchor="start"'
        )
        label_row.append(
            f'<span class="fence__fencelabel"{anchor} '
            f'style="left:{pct:.3f}%">fence {escape(fmt(value))}</span>'
        )
        taken.append(pct)

    track.append(
        f'<span class="fence__median" style="left:{fence.pct(fence.median):.3f}%"></span>'
    )

    for mark in marks:
        title = mark.title or fmt(mark.value)
        track.append(
            f'<span class="fence__mark" data-severity="{mark.severity}" '
            f'title="{escape(title)}" style="left:{mark.pct:.3f}%"></span>'
        )
        # Marks are 2% apart at the closest, which is enough for the capsules
        # and not enough for their labels: `x14` is ~24px, and 2% of a 1,099px
        # axis is 22. `Fare` has 116 outliers in ten clusters and printed ten
        # counts across the tail as one unreadable pile. So a count is printed
        # only where there is room for it, the same drop rule the percentile
        # ladder uses -- and nothing is lost, because the capsule keeps its
        # values in `title` and the table below lists them.
        if mark.count > 1 and all(
            abs(mark.pct - other) >= _LABEL_MIN_GAP_PCT for other in taken
        ):
            # A count centres on its mark, so one at either end of the axis
            # hangs half its width past the figure -- measured at 1.3px on
            # `Fare`, whose lowest five values are all zero and cluster at 0%.
            # The end labels anchor instead, as the ticks and fence labels do.
            label_row.append(
                f'<span class="fence__count"{_anchor(mark.pct)} '
                f'style="left:{mark.pct:.3f}%">×{mark.count}</span>'
            )
            taken.append(mark.pct)

    ticks = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = fence.value_lo + (fence.value_hi - fence.value_lo) * fraction
        anchor = "start" if fraction == 0.0 else ("end" if fraction == 1.0 else "")
        ticks.append(
            f'<span class="fence__tick" data-anchor="{anchor}" '
            f'style="left:{fraction * 100:.3f}%">{escape(fmt(value))}</span>'
        )

    described = described or (
        f"{name} values with the IQR fence and {fence.n_outliers:,} "
        f"value{'s' if fence.n_outliers != 1 else ''} beyond it"
    )

    keys = [
        '<li><span class="key key--box"></span>IQR and P1–P99</li>',
        f'<li><span class="key key--median"></span>median {escape(fmt(fence.median))}</li>',
    ]
    if legend == "tails":
        # The Min/Max pane plots values on both sides of the fence, so it needs
        # the fourth key the Outliers pane has no use for.
        keys.append(
            '<li><span class="key key--mark" data-severity="inside"></span>'
            "inside the fence</li>"
        )
    keys.append(
        '<li><span class="key key--mark" data-severity="moderate"></span>moderate</li>'
    )
    keys.append(
        '<li><span class="key key--mark" data-severity="extreme"></span>'
        "high or extreme</li>"
    )

    return (
        f'<div class="fence" role="img" aria-label="{escape(described)}">'
        f'<div class="fence__labels">{"".join(label_row)}</div>'
        f'<div class="fence__track">{"".join(track)}</div>'
        f'<div class="fence__axis"></div>'
        f'<div class="fence__ticks">{"".join(ticks)}</div>'
        f"</div>"
        f'<ul class="fence__legend">{"".join(keys)}</ul>'
    )


def render_table(fence: Fence, fmt, col_id: str = "") -> str:
    """One row per value, both verdicts side by side.

    The `rowspan` this replaces gave a value flagged by both methods two rows
    and a value flagged by one a single row, so the table's shape encoded
    something other than the data.

    Each value carries `data-col`/`data-value`, which is how the invariance
    fingerprint sees it: the extractor pairs adjacent `<td>`s and `__cap`/
    `__val` divs, and this pane is neither. Tagging the fact in the DOM is the
    durable hook the rest of the report already uses, and it survives any
    restyling of the grid around it.

    The **row index is deliberately not tagged.** It is not a fact about the
    data: where several rows share a value, which index is recorded is decided
    by arrival order, and the harness already drops `min_items`/`max_items`
    indices for exactly that reason -- twelve rows shared a maximum and CI
    recorded a different one from this machine.
    """
    head = (
        '<div class="fence-table__head">'
        "<span>Row</span><span>Value</span><span>By IQR</span><span>By MAD</span>"
        "</div>"
    )
    rows = "".join(
        f'<div class="fence-table__row">'
        f'<span class="fence-table__idx">{escape(row.index)}</span>'
        f'<span class="fence-table__val" data-col="{escape(col_id)}" '
        f'data-value="{row.value:.12g}">{escape(fmt(row.value))}</span>'
        f'<span class="fence-table__verdict" data-severity="{row.iqr_severity}">'
        f"{escape(row.iqr)}</span>"
        f'<span class="fence-table__verdict" data-severity="{row.mad_severity}">'
        f"{escape(row.mad)}</span>"
        f"</div>"
        for row in fence.rows
    )

    notes = []
    if fence.is_sampled:
        # Without this the pane and the card face contradict each other in
        # public: this table counts crossings in the reservoir, the card scales
        # that count to the column (#327), and the two differ by the sampling
        # ratio. Both are right. Saying which is which is what stops a reader
        # concluding one of them is broken.
        notes.append(
            f"Counted in a {fence.n_sampled:,}-value sample of "
            f"{fence.n_total:,}; the card scales this up to the column."
        )
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


# --------------------------------------------------------------------------- #
# the quantile strip (5b.1)
# --------------------------------------------------------------------------- #

#: Two ladder ticks closer than this, as a percentage of the axis, print only
#: the outer one. On `Fare`, P1 through Q3 all land in the first fifth of the
#: axis and every label in that fifth overprints its neighbour. The values stay
#: in the table below, which is where a reader looks for a figure anyway.
_LADDER_MIN_GAP_PCT = 4.0


def render_quantile_strip(fence: Fence, quantiles, fmt) -> str:
    """The nine percentiles as a shape, on the axis the other panes use.

    Phase 5b.1 (#154). They were a column of numbers printed directly under a
    histogram that draws the very shape they describe, and the two could not be
    read against each other. On one axis they become one picture -- and what a
    reader learns from it is not in the table at all: that the middle half of
    `Age` sits in a narrow band well left of centre.

    Three details are requirements rather than choices:

    **The whiskers are two spans terminating at the band edges.** One span
    running P1 to P99 paints *across* the IQR band, which reads as a range that
    contains the box -- a different and weaker claim than the two the box
    actually makes.

    **The median protrudes past both band edges** and the mean sits entirely
    *above* the band as a caret. They land about 24px apart inside a dark fill,
    so they are told apart by shape rather than by colour -- rule 2 of the
    token system, and the reason neither is a coloured line.

    **The ladder drops crowded labels.** See `_LADDER_MIN_GAP_PCT`.
    """
    q1_pct, q3_pct = fence.pct(fence.q1), fence.pct(fence.q3)
    lo_pct = fence.pct(fence.whisker_lo)
    hi_pct = fence.pct(fence.whisker_hi)
    median_pct = fence.pct(fence.median)

    parts = [
        f'<span class="qstrip__whisker" style="left:{lo_pct:.3f}%;'
        f'width:{max(0.0, q1_pct - lo_pct):.3f}%"></span>',
        f'<span class="qstrip__whisker" style="left:{q3_pct:.3f}%;'
        f'width:{max(0.0, hi_pct - q3_pct):.3f}%"></span>',
        f'<span class="qstrip__box" style="left:{q1_pct:.3f}%;'
        f'width:{max(0.0, q3_pct - q1_pct):.3f}%" '
        f'title="Interquartile range {escape(fmt(fence.q1))} to '
        f'{escape(fmt(fence.q3))} — the middle half of the data"></span>',
        f'<span class="qstrip__median" style="left:{median_pct:.3f}%" '
        f'title="Median {escape(fmt(fence.median))}"></span>',
    ]

    mean = fence.mean
    if _finite(mean):
        parts.append(
            f'<span class="qstrip__mean" style="left:{fence.pct(mean):.3f}%" '
            f'title="Mean {escape(fmt(mean))}"></span>'
        )

    ladder = _ladder(fence, quantiles)
    ticks = "".join(
        f'<span class="qstrip__tick" style="left:{pct:.3f}%"></span>'
        f'<span class="qstrip__label" data-row="{row}"{_anchor(pct)} '
        f'style="left:{pct:.3f}%">'
        f'<span class="qstrip__name">{escape(name)}</span>'
        f'<span class="qstrip__figure">{escape(fmt(value))}</span></span>'
        for row, (name, value, pct) in enumerate(ladder)
    )

    keys = [
        f'<li><span class="key key--qbox"></span>IQR {escape(fmt(fence.q1))} – '
        f"{escape(fmt(fence.q3))}</li>",
        '<li><span class="key key--qwhisker"></span>P1 – P99</li>',
        f'<li><span class="key key--qmedian"></span>median '
        f"{escape(fmt(fence.median))}</li>",
    ]
    if _finite(mean):
        keys.append(
            f'<li><span class="key key--qmean"></span>mean {escape(fmt(mean))}</li>'
        )

    return (
        '<div class="qstrip" role="img" aria-label="Distribution shape: '
        f"interquartile range {escape(fmt(fence.q1))} to {escape(fmt(fence.q3))}, "
        f'median {escape(fmt(fence.median))}">'
        f'<div class="qstrip__track">{"".join(parts)}</div>'
        f'<div class="qstrip__axis"></div>'
        f'<div class="qstrip__ladder">{ticks}</div>'
        "</div>"
        f'<ul class="fence__legend">{"".join(keys)}</ul>'
    )


#: Percentiles that never drop, however crowded the axis: the two ends and the
#: middle. The same tiering the histogram's x ticks use, and for the same
#: reason -- a range with no middle says nothing about whether the distribution
#: is centred, and the median is the one figure a reader looks for first.
_LADDER_ANCHORS = frozenset({"P1", "P50", "P99"})


def _ladder(fence: Fence, quantiles) -> list[tuple[str, float, float]]:
    """The percentile ticks that fit, anchors first.

    The first version dropped whichever point crowded its left-hand neighbour,
    walking left to right. On `Fare` -- where P1 through Q3 all land in the
    first fifth of the axis -- that kept `Q3` and dropped **`P50`**, purely by
    arrival order. The median is the one figure a reader looks for first, and
    it was being spent on whichever tick happened to come after it.

    So the anchors are placed first and the rest fill in around them.
    """
    points = [
        ("P1", getattr(quantiles, "p1", None)),
        ("P5", getattr(quantiles, "p5", None)),
        ("P10", getattr(quantiles, "p10", None)),
        ("Q1", fence.q1),
        ("P50", fence.median),
        ("Q3", fence.q3),
        ("P90", getattr(quantiles, "p90", None)),
        ("P95", getattr(quantiles, "p95", None)),
        ("P99", getattr(quantiles, "p99", None)),
    ]
    placed = [
        (name, float(value), fence.pct(float(value)))
        for name, value in points
        if _finite(value)
    ]
    if not placed:
        return []

    kept = [point for point in placed if point[0] in _LADDER_ANCHORS]
    for candidate in placed:
        if candidate in kept:
            continue
        if all(abs(candidate[2] - other[2]) >= _LADDER_MIN_GAP_PCT for other in kept):
            kept.append(candidate)

    # Back into axis order, so the alternating rows alternate along the axis
    # rather than in the order the anchors happened to be added.
    kept.sort(key=lambda point: point[2])
    return kept
