"""The column-type composition, as one 100% stacked bar.

This replaces a 135px donut, and closes #104.

A donut cannot be read to exact proportion -- comparing two arcs is a harder
perceptual task than comparing two lengths against a shared baseline -- and it
stops working below about 200px wide, which is every phone. A stacked bar reads
at any width, reflows for free, and has room to print the count inside each
segment, so the reader does not have to estimate anything.

Two details that are easy to get wrong and are asserted in the tests:

**The widths must sum to exactly 100.** Rounding each share independently
leaves a gap or an overhang at the right edge -- 1/3 three times rounds to 99.9
-- and a bar that does not reach its own end reads as a rendering bug. The
largest-remainder method fixes the total first and distributes the rounding
error to the shares that lost the most to it.

**A type with no columns gets no segment.** A zero-width segment is an artifact
rather than information, and the palest step of the data scale sits close
enough to ``--track`` that a hairline of it reads as a rendering seam. Those
types appear in the legend instead, muted, with their zero -- which is the
thing the reader actually wants to know.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

# Steps of the data scale, assigned by rank rather than by type. Type is not a
# colour -- the legend says which is which. Assigning a hue per type is what
# made olive mean both "categorical" and "passes".
#
# The second element is the text colour a segment may carry, or `None` when it
# may carry none. `--data-3` and `--data-4` are both `None`, for different
# reasons, and both are measured:
#
#   --data-3  #5C7F99  paper on it 4.03, ink on it 3.83 -- neither reaches the
#             4.5:1 text minimum, so it is a fill and never a label background.
#   --data-4  #A8BECD  1.83:1 on the paper. Stack-internal only; it is legal
#             beside another segment and a ghost on its own.
#
# The count still appears in the legend, which is where it goes for a narrow
# segment already -- so this reuses a mechanism rather than adding one.
_STEPS = (
    ("var(--data-1, #2C4A62)", "var(--on-data-1, #FBF9F5)"),
    ("var(--data-2, #3E6280)", "var(--on-data-2, #FBF9F5)"),
    ("var(--data-3, #5C7F99)", None),
    ("var(--data-4, #A8BECD)", "var(--on-data-4, #22201C)"),
)

# Below this share a segment is too narrow to hold its own count without the
# text spilling over its neighbours. The number still appears in the legend.
_MIN_SHARE_FOR_LABEL = 7.0


@dataclass(frozen=True)
class Segment:
    """One type's share of the columns."""

    label: str
    count: int
    percent: float
    fill: str
    #: Text colour legal on this fill, or None when the fill carries no text.
    ink: str | None


def apportion(counts: list[int]) -> list[float]:
    """Percentages that sum to exactly 100, by the largest-remainder method.

    Each share is floored to one decimal, then the leftover tenths are handed
    out to whichever shares lost the most in the flooring. Rounding each share
    on its own instead leaves the bar short or long at the right edge.

    Args:
        counts: Column counts per type. May contain zeros.

    Returns:
        One percentage per count, in the same order, summing to 100.0 when the
        total is positive and to 0.0 when it is not.
    """
    total = sum(counts)
    if total <= 0:
        return [0.0] * len(counts)

    # Work in tenths of a percent so the result is exact in integer arithmetic.
    exact = [count * 1000.0 / total for count in counts]
    floors = [int(value) for value in exact]
    remainder = 1000 - sum(floors)
    order = sorted(
        range(len(counts)),
        key=lambda i: (exact[i] - floors[i], counts[i]),
        reverse=True,
    )
    for position in range(remainder):
        floors[order[position % len(order)]] += 1
    return [value / 10.0 for value in floors]


class CompositionBarRenderer:
    """Renders the column-type composition as a 100% stacked bar."""

    def render(
        self, numeric: int, categorical: int, datetime: int, boolean: int
    ) -> str:
        """Return the bar and its legend.

        Args:
            numeric: Count of numeric columns.
            categorical: Count of categorical columns.
            datetime: Count of datetime columns.
            boolean: Count of boolean columns.

        Returns:
            An HTML fragment. Empty input renders an explicit empty state
            rather than a zero-width bar.
        """
        counts = [
            ("numeric", int(numeric)),
            ("categorical", int(categorical)),
            ("datetime", int(datetime)),
            ("boolean", int(boolean)),
        ]
        total = sum(count for _, count in counts)
        if total <= 0:
            return (
                '<div class="composition is-empty" role="img"'
                ' aria-label="Column types: no columns">'
                '<p class="composition__empty">No columns</p></div>'
            )

        # Descending by size, so the bar reads darkest-first and the eye meets
        # the largest group at the baseline it starts from.
        ranked = sorted(counts, key=lambda pair: (-pair[1], pair[0]))
        percents = apportion([count for _, count in ranked])

        segments: list[Segment] = []
        for index, ((label, count), percent) in enumerate(
            zip(ranked, percents, strict=True)
        ):
            fill, ink = _STEPS[min(index, len(_STEPS) - 1)]
            segments.append(Segment(label, count, percent, fill, ink))

        drawn = [segment for segment in segments if segment.count > 0]
        described = ", ".join(
            f"{segment.label} {segment.count} ({segment.percent:g}%)"
            for segment in drawn
        )
        summary = f"Column types: {described}"

        return (
            f'<div class="composition" role="img" aria-label="{_html.escape(summary)}">'
            f"{self._bar(drawn)}"
            f"{self._legend(segments)}"
            "</div>"
        )

    def _bar(self, drawn: list[Segment]) -> str:
        cells = []
        for segment in drawn:
            # The count goes inside when the segment can hold it. Below that it
            # would overlap its neighbour, and the legend carries it anyway.
            # Two reasons a segment holds no count: it is too narrow, or its
            # fill carries no legal text colour. Both send the number to the
            # legend, which has it either way.
            wide_enough = segment.percent >= _MIN_SHARE_FOR_LABEL
            inner = (
                f'<span class="composition__count">{segment.count:,}</span>'
                if wide_enough and segment.ink is not None
                else ""
            )
            colour = f";color:{segment.ink}" if segment.ink is not None else ""
            cells.append(
                f'<div class="composition__seg" style="width:{segment.percent:g}%;'
                f'background:{segment.fill}{colour}"'
                f' data-type="{segment.label}" data-count="{segment.count}"'
                f' data-percentage="{segment.percent:g}">{inner}</div>'
            )
        return f'<div class="composition__bar">{"".join(cells)}</div>'

    def _legend(self, segments: list[Segment]) -> str:
        items = []
        for segment in segments:
            # A type with no columns keeps its place in the legend, muted, so
            # "no datetime columns" is stated rather than left to be inferred
            # from a bar that simply lacks a colour.
            zero = " is-zero" if segment.count == 0 else ""
            swatch = (
                f'<span class="composition__swatch" style="background:{segment.fill}"></span>'
                if segment.count > 0
                else '<span class="composition__swatch is-zero"></span>'
            )
            items.append(
                f'<li class="composition__item{zero}">{swatch}'
                f'<span class="composition__label">{segment.label}</span>'
                f'<span class="composition__value">{segment.count:,}</span></li>'
            )
        return f'<ul class="composition__legend">{"".join(items)}</ul>'
