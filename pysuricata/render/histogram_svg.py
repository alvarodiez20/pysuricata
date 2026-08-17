"""
SVG-based histogram rendering for PySuricata.

This module provides a lightweight, high-performance histogram implementation
using SVG instead of Canvas/Chart.js. It handles large numbers intelligently
and provides better integration with the existing tooltip system.
"""

from __future__ import annotations

import html as _html
import math
import re
from dataclasses import dataclass

import numpy as np

from .format_utils import fmt_compact_scientific
from .svg_utils import nice_ticks

# Units the column name states outright. Deliberately short, and deliberately
# not clever: this maps a name to a unit only where the name *is* the unit.
#
# The absent branch is the one that matters. A column called `score` has no
# unit, and labelling its axis "SCORE" would be worse than leaving it bare --
# the header already says the column is called score, so the label would add a
# word and no information, while looking like a unit. Anything not listed here
# gets no label at all.
_UNIT_BY_NAME: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"age", "years", "year", "yr", "yrs"}), "YEARS"),
    (frozenset({"month", "months"}), "MONTHS"),
    (frozenset({"day", "days"}), "DAYS"),
    (frozenset({"hour", "hours", "hrs"}), "HOURS"),
    (frozenset({"minute", "minutes", "mins"}), "MINUTES"),
    (frozenset({"second", "seconds", "secs", "duration", "elapsed"}), "SECONDS"),
    (frozenset({"ms", "millis", "milliseconds"}), "MS"),
    (frozenset({"bytes", "size", "nbytes"}), "BYTES"),
    (frozenset({"kb"}), "KB"),
    (frozenset({"mb"}), "MB"),
    (frozenset({"count", "counts", "n", "num", "total", "qty", "quantity"}), "COUNT"),
    (frozenset({"pct", "percent", "percentage"}), "%"),
    (frozenset({"ratio", "rate", "share", "fraction", "proportion"}), "RATIO"),
    (frozenset({"km"}), "KM"),
    (frozenset({"m", "metres", "meters"}), "M"),
    (frozenset({"cm"}), "CM"),
    (frozenset({"kg"}), "KG"),
    (frozenset({"celsius", "degc"}), "°C"),
)

# A trailing `_unit` in the name is a stronger signal than the whole name: an
# `age_years` column names its unit explicitly.
_SPLIT = re.compile(r"[^a-z0-9]+")


def _round_preserving_total(values: np.ndarray, total: int) -> np.ndarray:
    """Round non-negative bin weights to integers summing to ``total``.

    The largest-remainder method: floor everything, then hand the shortfall out
    one unit at a time to the bins with the largest discarded fractions. It is
    the standard way to apportion a whole number across shares, and it has two
    properties this needs and the previous approach had neither of.

    **It cannot produce a negative count.** The previous code rounded each bin
    to nearest and then dumped the *entire* residual into the single bin with
    the largest fractional part. On the Titanic report's `Fare` column at 50
    bins, rounding to nearest overshot by 3 and the chosen bin held 2, so the
    report shipped a bin of **-1** -- a count that cannot exist, drawn as a
    `<rect>` with a negative height that the browser rejects, and printed in
    that bar's tooltip. See #253.

    **And it spreads the correction.** A negative was the visible symptom of
    something wrong in every direction: dumping the whole residual into one bin
    moves rows out of (or into) a single column of the chart. A bin holding 5
    silently displayed 2. Handing out one unit per bin moves each affected bin
    by at most one, which is the smallest change consistent with the total.

    Args:
        values: Non-negative weights, one per bin. Their sum should already be
            ``total`` up to floating-point error.
        total: The row count the bins must sum to.

    Returns:
        Non-negative integers summing exactly to ``total``.
    """
    floors = np.floor(values).astype(int)
    # Clamp defensively: `values` is a sum of non-negative products and cannot
    # be negative, but a caller that broke that would otherwise reintroduce
    # exactly the defect this function exists to remove.
    np.maximum(floors, 0, out=floors)

    shortfall = int(total - floors.sum())
    if shortfall <= 0:
        return floors

    # Largest discarded fraction first; ties go to the earlier bin, which keeps
    # the result a function of the input alone rather than of sort stability.
    order = np.argsort(-(values - floors), kind="stable")
    for index in order[:shortfall]:
        floors[index] += 1
    # More units than bins only happens if `values` summed low by more than one
    # per bin, which float error cannot do -- but if it ever did, the remainder
    # belongs somewhere rather than nowhere.
    remaining = int(total - floors.sum())
    if remaining > 0 and len(floors):
        floors[order[0]] += remaining
    return floors


def derive_x_unit(column_name: str) -> str | None:
    """The unit of a numeric column's x axis, or None when it has none.

    Returns None far more often than not, and that is the intended behaviour --
    an axis with no unit is honest, an axis labelled with a restatement of the
    column name is not.

    Args:
        column_name: The column's name.

    Returns:
        A short uppercase unit, or None when the name does not state one.
    """
    words = [w for w in _SPLIT.split((column_name or "").lower()) if w]
    if not words:
        return None
    # Last word first: `age_years` is years, `years_since_x` is not.
    for word in (words[-1], words[0]):
        for names, unit in _UNIT_BY_NAME:
            if word in names:
                return unit
    return None


@dataclass
class HistogramConfig:
    """Configuration for histogram rendering."""

    width: int = 420
    height: int = 200
    margin_left: int = 60
    margin_right: int = 20
    margin_top: int = 20
    margin_bottom: int = 40

    # Bar styling. The colour is a token, not a literal: the SVG is inline in
    # the report, so the custom property cascades into it and dark mode works
    # without a second copy of the chart. The hex after the comma is the CSS
    # fallback, used only if a renderer will not substitute var() inside a
    # presentation attribute.
    bar_color: str = "var(--data-2, #3E6280)"
    bar_opacity: float = 1.0
    bar_stroke: str = "none"
    bar_stroke_width: float = 0

    # Axis styling
    axis_color: str = "var(--axis, #8F8474)"
    axis_stroke_width: float = 1.0
    tick_length: int = 5

    # Text styling
    # Figures are monospace everywhere in the report; an axis label in a
    # different face than the table beneath it reads as a different kind of
    # number.
    font_family: str = "var(--font-mono)"
    font_size: int = 11
    label_font_size: int = 10
    title_font_size: int = 12

    # Number formatting
    large_number_threshold: float = 1_000_000
    max_label_length: int = 8


@dataclass
class HistogramData:
    """Histogram data structure."""

    counts: np.ndarray
    edges: np.ndarray
    bin_centers: np.ndarray
    total_count: int
    scale: str  # 'lin' or 'log'
    y_max: float
    original_range: tuple[float, float] | None = (
        None  # Original data range for log scale
    )


class SVGHistogramRenderer:
    """Renders histograms as SVG with intelligent number formatting."""

    def __init__(self, config: HistogramConfig | None = None):
        self.config = config or HistogramConfig()

    def render_histogram_from_bins(
        self,
        bin_edges: list[float],
        bin_counts: list[int],
        bins: int,
        scale: str,
        title: str,
        col_id: str,
    ) -> str:
        """Render histogram from pre-computed bin edges and counts.

        This method is used for true distribution histograms where the bin
        edges and counts are already computed from the full dataset.

        Args:
            bin_edges: List of bin edge values
            bin_counts: List of counts per bin
            bins: Number of bins to display (actually used now)
            scale: Scale type ('lin' or 'log')
            title: Chart title
            col_id: Column identifier for tooltips

        Returns:
            SVG string
        """
        if not bin_edges or not bin_counts or len(bin_edges) < 2:
            return self._render_empty_histogram(title)

        # Convert to numpy arrays
        original_edges = np.array(bin_edges)
        original_counts = np.array(bin_counts)

        # Apply log transformation if needed
        if scale == "log":
            # Filter out non-positive values and their corresponding counts
            positive_mask = original_edges > 0
            if not np.any(positive_mask):
                return self._render_empty_histogram(title)

            # Keep only positive edges and their corresponding counts
            positive_edges = original_edges[positive_mask]
            positive_counts = original_counts[
                positive_mask[:-1]
            ]  # Counts are one less than edges

            # Apply log10 transformation to edges
            transformed_edges = np.log10(positive_edges)
            transformed_counts = positive_counts

            # Get the transformed data range
            data_min = transformed_edges[0]
            data_max = transformed_edges[-1]
        else:
            # Use original data for linear scale
            transformed_edges = original_edges
            transformed_counts = original_counts
            data_min = original_edges[0]
            data_max = original_edges[-1]

        # Create new bin edges with the requested number of bins
        if bins <= 1:
            bins = 2  # Minimum 2 bins

        new_edges = np.linspace(data_min, data_max, bins + 1)

        # Redistribute counts to new bins using improved algorithm
        new_counts = np.zeros(bins, dtype=float)  # Use float to avoid precision loss

        for i in range(len(transformed_counts)):
            if transformed_counts[i] > 0:
                # Find which new bins this original bin contributes to
                old_left = transformed_edges[i]
                old_right = transformed_edges[i + 1]

                # Find overlapping new bins
                for j in range(bins):
                    new_left = new_edges[j]
                    new_right = new_edges[j + 1]

                    # Calculate overlap
                    overlap_left = max(old_left, new_left)
                    overlap_right = min(old_right, new_right)

                    if overlap_left < overlap_right:
                        # Calculate proportion of overlap
                        old_width = old_right - old_left
                        overlap_width = overlap_right - overlap_left
                        proportion = overlap_width / old_width if old_width > 0 else 0

                        # Distribute count proportionally (keep as float for now)
                        new_counts[j] += transformed_counts[i] * proportion

        # Convert to integers while preserving total count
        total_original = int(np.sum(transformed_counts))
        total_new = np.sum(new_counts)

        if total_new > 0:
            # Scale to preserve total count
            scale_factor = total_original / total_new
            new_counts = new_counts * scale_factor
            new_counts = _round_preserving_total(new_counts, total_original)
        else:
            new_counts = np.zeros(bins, dtype=int)

        # Calculate new bin centers
        new_bin_centers = (new_edges[:-1] + new_edges[1:]) / 2.0

        # Calculate actual max count
        actual_max = int(np.max(new_counts)) if len(new_counts) > 0 else 0

        # Calculate nice ticks to get the proper y_max for scaling
        # This ensures bars can reach the top tick mark
        y_ticks, _ = nice_ticks(0, actual_max, 5)
        nice_y_max = y_ticks[-1] if y_ticks else actual_max

        # Create histogram data with nice y_max for proper bar scaling
        hist_data = HistogramData(
            counts=new_counts,
            edges=new_edges,
            bin_centers=new_bin_centers,
            total_count=int(np.sum(new_counts)),
            scale=scale,
            y_max=nice_y_max,
        )

        return self._render_figure(hist_data, title, col_id, bins)

    # ------------------------------------------------------------------ #
    # Output: two coordinate systems, deliberately separated
    #
    # Rule 4 of the design system, and the thing every other fix in this
    # phase depends on:
    #
    #   Uniform scale  =>  text size tracks the viewport.
    #   Fixed text     =>  the canvas has to be ~1:1 with its display size.
    #   One static SVG cannot be 1:1 at both 1,099px and 284px.
    #
    # So the SVG holds only what *should* stretch -- bars, gridlines, the two
    # axis rules -- under `preserveAspectRatio="none"`, with every stroke
    # marked `vector-effect="non-scaling-stroke"` so a hairline stays a
    # hairline at any width. Everything with a glyph in it is HTML positioned
    # at percentage offsets, which are scale-independent by construction and
    # render at 11px whatever the chart is doing.
    #
    # The viewBox is 0..100 on both axes, so the numbers inside it *are*
    # percentages: the SVG and the HTML layer are written in the same units,
    # and a bar cannot drift away from its label.
    #
    # It also takes every `<text>` out of the SVG, which is most of what made
    # the histograms 23% of report bytes.
    # ------------------------------------------------------------------ #

    #: The plot's own coordinate space. Not pixels -- see above.
    _SPAN = 100.0

    @staticmethod
    def _n(value: float) -> str:
        """A coordinate, at two decimals, without trailing zeros.

        `x="30.00"` and `x="30"` place a bar in exactly the same spot, and the
        four coordinates on a bar are the most repeated numbers in the report
        -- 50 bars x 6 variants x every numeric column (#206). Two decimals is
        already past what a display can resolve: the viewBox is 0..100, so one
        unit is a percent of the plot, and at 1,100px the third decimal is a
        ten-thousandth of a pixel.
        """
        return f"{round(value, 2):g}"

    #: Which x ticks survive which breakpoint, nine of them.
    #:
    #: Tiering by *importance* rather than by index is the point. Tier 1 is
    #: the two ends -- the range -- plus the midpoint, and never drops however
    #: narrow the card gets. Dropping tier 3 leaves five; dropping tier 2 as
    #: well leaves three.
    #:
    #: The first version made the ends tier 1 and everything else 2 or 3,
    #: which collapsed to *two* labels on a phone: a range with no middle, so
    #: nothing tells you whether the distribution is centred.
    _TICK_TIERS = (1, 3, 2, 3, 1, 3, 2, 3, 1)

    def _render_figure(
        self, hist_data: HistogramData, title: str, col_id: str, bins: int
    ) -> str:
        """The whole chart: a gutter, a stretchable plot, and a caption."""
        safe_title = self.safe_html_escape(title) if title else "data"
        y_ticks: list[float] = []
        if hist_data.y_max:
            y_ticks, _ = nice_ticks(0, hist_data.y_max, 5)

        marks = [
            f'<svg class="hist-svg" viewBox="0 0 {self._SPAN:g} {self._SPAN:g}" '
            f'preserveAspectRatio="none" role="img" '
            f'aria-labelledby="hist-title-{col_id}">',
            f'<title id="hist-title-{col_id}">Histogram for {safe_title}</title>',
            f"<desc>Distribution chart with {max(len(hist_data.edges) - 1, 0)} bins</desc>",
        ]
        marks.extend(self._render_gridlines(hist_data, y_ticks))
        marks.extend(self._render_bars(hist_data, col_id))
        marks.append("</svg>")

        return (
            f'<figure class="hist" data-bins="{bins}">'
            f'<div class="hist__plot">'
            f'<div class="hist__gutter">{self._render_y_labels(hist_data, y_ticks)}'
            f'<span class="hist__unit">ROWS</span></div>'
            f'<div class="hist__area">{"".join(marks)}'
            f"{self._render_x_labels(hist_data)}</div>"
            f"</div>"
            f"{self._render_caption(title, hist_data, bins)}"
            f"</figure>"
        )

    def _render_gridlines(
        self, hist_data: HistogramData, y_ticks: list[float]
    ) -> list[str]:
        """Horizontal rules at each y tick, plus the two axes.

        `vector-effect="non-scaling-stroke"` is what makes this possible. The
        SVG is stretched by a different factor on each axis, so without it a
        1-unit rule would render 11px thick horizontally and 0.28px vertically.
        """
        if not hist_data.y_max:
            return []

        # `vector-effect` is carried by the `.grid` and `.axis` rules rather
        # than repeated here, the same move the bars made (#206).
        parts = []
        for tick in y_ticks:
            y = (1 - tick / hist_data.y_max) * self._SPAN
            parts.append(
                f'<line class="grid" x1="0" y1="{self._n(y)}" '
                f'x2="{self._SPAN:g}" y2="{self._n(y)}"/>'
            )
        parts.append(
            f'<line class="axis" x1="0" y1="{self._SPAN:g}" '
            f'x2="{self._SPAN:g}" y2="{self._SPAN:g}"/>'
        )
        parts.append(f'<line class="axis" x1="0" y1="0" x2="0" y2="{self._SPAN:g}"/>')
        return parts

    def _render_bars(self, hist_data: HistogramData, col_id: str) -> list[str]:
        """Bars, edge to edge, separated by a stroke that does not scale.

        The gap used to be geometry: `bar_w = max(1, bar_width - 1)`, a 1-unit
        gap in viewBox space. Under `preserveAspectRatio="none"` that scales
        with x -- 1.1px at a 1,100px plot, 0.56px at 560px, and **0.28px at
        284px**, where the bars merge into one block. A gap drawn in data units
        is not a gap; it is a gap-shaped fraction of the data.

        So the bars touch, and the separator is a `--paper` stroke marked
        non-scaling, which is 1px at every width by construction.
        """
        if len(hist_data.counts) == 0 or hist_data.y_max == 0:
            return []

        parts = []
        width = self._SPAN / len(hist_data.counts)

        for index, (count, center) in enumerate(
            zip(hist_data.counts, hist_data.bin_centers, strict=False)
        ):
            # Rule 3: a zero count draws nothing. A 1px floor is right for a
            # small non-zero value and wrong for zero -- ten empty months drawn
            # as ten 1px bars assert data that is not there.
            #
            # `<= 0`, not `== 0`. A negative count is not a drawing decision at
            # all: it is a value that cannot exist, and it used to reach here
            # and become `height="-0.33"`, which browsers reject and log. The
            # binning that produced it is fixed above; this is the guard that
            # keeps a value that cannot exist from becoming geometry. See #253.
            if count <= 0:
                continue

            x = index * width
            height = (count / hist_data.y_max) * self._SPAN
            y = self._SPAN - height

            if index < len(hist_data.edges) - 1:
                x0_label = self._format_tick_label_standardized(hist_data.edges[index])
                x1_label = self._format_tick_label_standardized(
                    hist_data.edges[index + 1]
                )
            else:
                x0_label = x1_label = self._format_tick_label_standardized(center)

            pct = (
                (count / hist_data.total_count) * 100.0
                if hist_data.total_count
                else 0.0
            )

            parts.append(
                # Two decimals, not three. The viewBox is 0..100, so a unit is
                # a percent of the plot: at 1,100px the third decimal is a
                # ten-thousandth of a pixel. Six variants x 50 bars x 4
                # coordinates make it the most-repeated number in the report.
                f'<rect class="bar" x="{self._n(x)}" y="{self._n(y)}" '
                f'width="{self._n(width)}" height="{self._n(height)}" '
                # No fill-opacity and no rounded corners: both change the
                # apparent length of a bar, which is the one thing it encodes.
                #
                # `vector-effect` lives in the `.bar` CSS rule, beside the
                # stroke it modifies, rather than being repeated as a 41-byte
                # attribute on every bar (#206).
                f'data-count="{int(count)}" data-pct="{pct:.1f}" '
                f'data-x0="{x0_label}" data-x1="{x1_label}" '
                # `data-col` looks redundant -- the column is on the
                # `.hist-variants` parent, and neither the tooltip handler nor
                # any stylesheet reads it. It is not redundant.
                # `scripts/report_fingerprint.py` scans element by element and
                # takes the scope from the *same* tag, so dropping this turns
                # every `attr::col_age::count` into `attr::::count` and collides
                # the bar counts of every numeric column under one key. That is
                # a weaker invariance guard bought with ~19 bytes a bar, which
                # is the wrong trade (#206).
                f'data-col="{col_id}"/>'
            )

        return parts

    def _render_y_labels(self, hist_data: HistogramData, y_ticks: list[float]) -> str:
        """Count labels in the gutter, at percentage offsets.

        Four glyphs guaranteed (#183), which is what lets the gutter be a fixed
        44px -- so the plot's left edge does not move between columns and bars
        line up down the page.
        """
        if not hist_data.y_max:
            return ""

        out = []
        for tick in y_ticks:
            top = (1 - tick / hist_data.y_max) * 100.0
            label = self._format_tick_label_standardized(tick, is_count=True)
            # The two extreme labels are nudged inward by CSS, or the top one
            # floats above the plot and the `0` hangs below the axis. Which
            # label is which is stated here rather than left to `:first-of-type`
            # and `:last-of-type`: ticks are emitted in *ascending* order, so
            # the first span is the bottom of the plot and the last is the top
            # -- the reverse of what those selectors read as, which is why the
            # nudges were applied to the wrong ends and produced exactly the
            # two defects they exist to prevent.
            edge = ' data-edge="top"' if top <= 0.0 else ""
            if top >= 100.0:
                edge = ' data-edge="bottom"'
            out.append(
                f'<span class="hist__y"{edge} style="top:{top:.3f}%">'
                f"{self.safe_html_escape(label)}</span>"
            )
        return "".join(out)

    def _render_x_labels(self, hist_data: HistogramData) -> str:
        """Nine value labels across the axis, each tagged by importance.

        The renderer cannot know the viewport, so it writes every label it
        would ever want and lets CSS drop tiers at breakpoints: nine become
        five under 760px and three under 440px, with no variants and no JS.
        """
        if len(hist_data.bin_centers) == 0 or len(hist_data.edges) < 2:
            return ""

        low = float(hist_data.edges[0])
        high = float(hist_data.edges[-1])
        if not (math.isfinite(low) and math.isfinite(high)):
            return ""

        count = len(self._TICK_TIERS)
        out = []
        for index, tier in enumerate(self._TICK_TIERS):
            fraction = index / (count - 1)
            value = low + (high - low) * fraction
            if hist_data.scale == "log":
                value = 10**value
            label = self._format_tick_label_standardized(value)
            # The end labels anchor to the plot edge rather than centring on
            # their tick, so a wide value at either end sits inside the chart
            # instead of overhanging it.
            if index == 0:
                anchor = ' data-anchor="start"'
            elif index == count - 1:
                anchor = ' data-anchor="end"'
            else:
                anchor = ""
            out.append(
                f'<span class="hist__tick" data-tier="{tier}"{anchor} '
                f'style="left:{fraction * 100:.3f}%">'
                f"{self.safe_html_escape(label)}</span>"
            )
        return f'<div class="hist__x">{"".join(out)}</div>'

    def _render_caption(self, title: str, hist_data: HistogramData, bins: int) -> str:
        """`years · 25 bins · peak 83 rows at 26–29`.

        The x unit used to sit at the right end of the axis, opposite `ROWS` on
        the left. At 1,100px those two are a hand-span apart and stop reading
        as a pair, so the unit joins the caption -- where it can also carry the
        bin count and the peak. The peak matters more now that the y labels
        abbreviate to four glyphs: this is where the exact figure lives.

        `derive_x_unit` returning None has to read gracefully, so the unit
        clause is omitted rather than replaced by a guess.
        """
        pieces = []
        unit = derive_x_unit(title)
        if unit:
            pieces.append(unit.lower())
        pieces.append(f"{bins} bins")

        if len(hist_data.counts) and int(hist_data.counts.max()) > 0:
            index = int(np.argmax(hist_data.counts))
            peak = int(hist_data.counts[index])
            noun = "row" if peak == 1 else "rows"
            if index < len(hist_data.edges) - 1:
                low = self._format_tick_label_standardized(hist_data.edges[index])
                high = self._format_tick_label_standardized(hist_data.edges[index + 1])
                pieces.append(f"peak {peak:,} {noun} at {low}–{high}")
            else:
                pieces.append(f"peak {peak:,} {noun}")

        text = " · ".join(pieces)
        return (
            f'<figcaption class="hist__caption">'
            f"{self.safe_html_escape(text)}</figcaption>"
        )

    def _render_empty_histogram(self, title: str) -> str:
        """A sentence, in the same figure shape as a real chart.

        It used to be an SVG with `No data` set in the middle of an otherwise
        blank 420x200 canvas, which reads as a chart that failed rather than as
        a column with nothing to draw. Keeping the `<figure>` wrapper means the
        card's layout does not shift between the two cases.
        """
        safe_title = self.safe_html_escape(title) if title else "data"
        return (
            f'<figure class="hist hist--empty">'
            f'<p class="hist__nodata">No values to plot</p>'
            f'<figcaption class="hist__caption">{safe_title}</figcaption>'
            f"</figure>"
        )

    def _format_tick_label_standardized(
        self, value: float, is_count: bool = False
    ) -> str:
        """Format tick labels with intelligent number formatting.

        Args:
            value: The numeric value to format
            is_count: If True, format as integer (for histogram counts).
                     If False, format with appropriate precision (for data ranges).
        """
        # Special case: zero
        if value == 0:
            return "0"

        # A count label is guaranteed four glyphs, not merely encouraged to be
        # short. The y gutter is a fixed 44px -- 27px of 11px mono, a 5px tick
        # and 8px of air -- so that the plot's left edge does not move between
        # columns and bars line up down the page. A five-glyph label either
        # overflows that or forces the gutter to breathe, and a gutter that
        # breathes loses the alignment the fixed one buys.
        #
        # This used to `prefer` short: `12,500` came out as six glyphs and
        # `12.5M` as five. Nobody reads seven significant figures off an axis,
        # and the exact peak is printed in the caption line, so abbreviating
        # costs nothing.
        if is_count:
            return self._format_count_in_four_glyphs(value)

        # Beyond this an axis label stops being readable as a quantity and
        # starts being a ruler: `-2,000,000,000,000,000` is 22 characters, wide
        # enough to collide with its neighbours and too long to take in at a
        # glance. Compact notation is shorter than both that and `-2.0e+15`.
        if abs(value) >= 1e6:
            for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
                if abs(value) >= limit:
                    scaled = value / limit
                    text = (
                        f"{scaled:.0f}"
                        if abs(scaled - round(scaled)) < 0.05
                        else f"{scaled:.1f}"
                    )
                    return f"{text}{suffix}"

        # An x label is a data value, not a count. It gets no glyph budget: it
        # lives in the caption row, which is as wide as the plot.
        if abs(value - round(value)) < 1e-9:
            int_val = int(round(value))
            return f"{int_val:,}" if abs(int_val) >= 1000 else f"{int_val}"

        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            return fmt_compact_scientific(value)
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.1f}"
        return f"{value:.3f}"

    #: Largest first. int64 tops out near 9.2e18, so `E` is the last band a
    #: row count can reach.
    _COUNT_BANDS = (
        (1e18, "E"),
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    )

    def _format_count_in_four_glyphs(self, value: float) -> str:
        """A row count in at most four characters, always.

        The rules that make the bound hold, each of which a plausible-looking
        implementation gets wrong:

        * **No thousands separator under 10,000.** `1,000` is five glyphs;
          `1000` is four.
        * **One decimal only below 10.** `12.5K` is five glyphs, so anything
          that scales to 10 or more rounds to a whole number: 12,700 is `13K`.
          (12,500 is `12K`, not `13K` -- Python rounds halves to even. That is
          fine for an axis label and would not be for a total.)
        * **Promote when rounding overflows the band.** 999,999 scales to
          999.999 in the `K` band and rounds to 1000, which would print
          `1000K`. It belongs in the next band up, as `1.0M`.
        """
        sign = "-" if value < 0 else ""
        magnitude = abs(value)

        if magnitude < 10_000:
            return f"{sign}{int(round(magnitude))}"

        for index, (limit, suffix) in enumerate(self._COUNT_BANDS):
            if magnitude < limit:
                continue
            scaled = magnitude / limit
            if scaled < 10 and abs(scaled - round(scaled)) >= 0.05:
                return f"{sign}{scaled:.1f}{suffix}"
            whole = int(round(scaled))
            if whole >= 1000 and index > 0:
                # Rounding pushed it into the band above.
                bigger_limit, bigger_suffix = self._COUNT_BANDS[index - 1]
                promoted = magnitude / bigger_limit
                if promoted < 10 and abs(promoted - round(promoted)) >= 0.05:
                    return f"{sign}{promoted:.1f}{bigger_suffix}"
                return f"{sign}{int(round(promoted))}{bigger_suffix}"
            return f"{sign}{whole}{suffix}"

        # Unreachable for any finite count: 10,000 is already above the
        # smallest band. Kept so the function is total.
        return f"{sign}{int(round(magnitude))}"

    def safe_html_escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return _html.escape(str(text))
