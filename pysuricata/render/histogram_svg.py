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
from .svg_utils import nice_log_ticks_from_log10, nice_ticks

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

            # Round to integers while preserving total
            new_counts_int = np.round(new_counts).astype(int)

            # Adjust for any rounding errors to preserve exact total
            diff = total_original - np.sum(new_counts_int)
            if diff != 0:
                # Find the bin with the largest fractional part and adjust it
                fractional_parts = new_counts - new_counts_int
                if len(fractional_parts) > 0:
                    max_fractional_idx = np.argmax(fractional_parts)
                    new_counts_int[max_fractional_idx] += int(diff)

            new_counts = new_counts_int
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

        # Calculate dimensions
        inner_width = (
            self.config.width - self.config.margin_left - self.config.margin_right
        )
        inner_height = (
            self.config.height - self.config.margin_top - self.config.margin_bottom
        )

        # Generate SVG
        safe_title = self.safe_html_escape(title) if title else "data"
        svg_parts = [
            f'<svg class="hist-svg" width="{self.config.width}" height="{self.config.height}" '
            f'viewBox="0 0 {self.config.width} {self.config.height}" '
            f'role="img" aria-labelledby="hist-title-{col_id}">',
            f'<title id="hist-title-{col_id}">Histogram for {safe_title}</title>',
            f"<desc>Distribution chart with {len(hist_data.edges) - 1 if len(hist_data.edges) > 0 else 0} bins</desc>",
        ]

        # Add grid
        svg_parts.extend(self._render_grid(hist_data, inner_width, inner_height))

        # Add bars
        svg_parts.extend(
            self._render_bars(hist_data, inner_width, inner_height, col_id)
        )

        # Add axes
        svg_parts.extend(self._render_axes(hist_data, inner_width, inner_height))

        # No in-chart title. The card header already carries the column name,
        # so printing it again inside the plot spent a line of the chart on a
        # word the reader had just read. The <title> element above stays --
        # that is the accessible name, not a caption.
        svg_parts.extend(self._render_unit_labels(title, inner_width))

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _render_unit_labels(self, column_name: str, inner_width: int) -> list[str]:
        """`ROWS` over the y axis, and the x unit at the right end of it.

        Nothing on the old chart said which axis was years and which was rows,
        which is the actual gap -- the column name was printed instead, and the
        header already had that.
        """
        parts = [
            f'<text x="{self.config.margin_left}" y="{self.config.margin_top - 7}" '
            f'text-anchor="start" class="unit-label" '
            f'font-family="{self.config.font_family}" font-size="10" '
            f'letter-spacing="0.12em" fill="{self.config.axis_color}">ROWS</text>'
        ]
        unit = derive_x_unit(column_name)
        if unit:
            y = self.config.margin_top + self._inner_height() + 32
            parts.append(
                f'<text x="{self.config.margin_left + inner_width}" y="{y}" '
                f'text-anchor="end" class="unit-label" '
                f'font-family="{self.config.font_family}" font-size="10" '
                f'letter-spacing="0.12em" fill="{self.config.axis_color}">'
                f"{self.safe_html_escape(unit)}</text>"
            )
        return parts

    def _inner_height(self) -> int:
        return self.config.height - self.config.margin_top - self.config.margin_bottom

    def _render_empty_histogram(self, title: str) -> str:
        """Render an empty histogram when no data is available."""
        safe_title = self.safe_html_escape(title) if title else "data"
        svg_parts = [
            f'<svg class="hist-svg" width="{self.config.width}" height="{self.config.height}" '
            f'viewBox="0 0 {self.config.width} {self.config.height}" '
            f'role="img" aria-labelledby="hist-empty-title">',
            f'<title id="hist-empty-title">Empty histogram for {safe_title}</title>',
            "<desc>No data available for visualization</desc>",
        ]

        # Add "No data" message
        svg_parts.append(
            f'<text x="{self.config.width // 2}" y="{self.config.height // 2}" '
            f'text-anchor="middle" class="hist-empty" '
            f'font-family="{self.config.font_family}" '
            f'font-size="{self.config.font_size}" fill="#999">No data</text>'
        )

        if title:
            svg_parts.append(
                f'<text x="{self.config.width // 2}" y="15" '
                f'text-anchor="middle" class="hist-title" '
                f'font-family="{self.config.font_family}" '
                f'font-size="{self.config.title_font_size}">'
                f"{self.safe_html_escape(title)}</text>"
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _render_grid(
        self, hist_data: HistogramData, inner_width: int, inner_height: int
    ) -> list[str]:
        """Render grid lines."""
        if hist_data.y_max == 0:
            return []

        parts = []

        # Y-axis grid lines
        y_ticks, _ = nice_ticks(0, hist_data.y_max, 5)
        for y_tick in y_ticks[1:-1]:  # Skip first and last
            y_pos = (
                self.config.margin_top + (1 - y_tick / hist_data.y_max) * inner_height
            )
            parts.append(
                f'<line class="grid" x1="{self.config.margin_left}" y1="{y_pos}" '
                f'x2="{self.config.margin_left + inner_width}" y2="{y_pos}"/>'
            )

        return parts

    def _render_bars(
        self, hist_data: HistogramData, inner_width: int, inner_height: int, col_id: str
    ) -> list[str]:
        """Render histogram bars."""
        if len(hist_data.counts) == 0 or hist_data.y_max == 0:
            return []

        parts = []
        bar_width = inner_width / len(hist_data.counts)

        for i, (count, center) in enumerate(
            zip(hist_data.counts, hist_data.bin_centers, strict=False)
        ):
            if count == 0:
                continue

            # Calculate bar dimensions
            x = self.config.margin_left + i * bar_width
            bar_w = max(1, bar_width - 1)  # Small gap between bars
            bar_h = (count / hist_data.y_max) * inner_height

            # Calculate y position (SVG coordinates are top-down)
            y = self.config.margin_top + inner_height - bar_h

            # Calculate bin range for tooltip
            if i < len(hist_data.edges) - 1:
                x0, x1 = hist_data.edges[i], hist_data.edges[i + 1]
                x0_label = self._format_tick_label_standardized(x0)
                x1_label = self._format_tick_label_standardized(x1)
            else:
                x0_label = x1_label = self._format_tick_label_standardized(center)

            # Calculate percentage
            pct = (count / hist_data.total_count) * 100.0

            # Create bar with tooltip data
            parts.append(
                f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                # No stroke, no fill-opacity, no rounded corners: each of
                # those changes the apparent length of a bar, which is the one
                # thing the bar encodes.
                f'fill="{self.config.bar_color}" '
                f'data-count="{int(count)}" data-pct="{pct:.1f}" '
                f'data-x0="{x0_label}" data-x1="{x1_label}" '
                f'data-col="{col_id}"/>'
            )

        return parts

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

        if is_count:
            # For histogram counts (y-axis), always format as integers
            if abs(value) >= 1000:
                return f"{int(value):,}"
            else:
                return f"{int(value)}"
        else:
            # For data values (x-axis)
            # Check if value is effectively an integer
            if abs(value - round(value)) < 1e-9:
                int_val = int(round(value))
                if abs(int_val) >= 1000:
                    return f"{int_val:,}"
                else:
                    return f"{int_val}"

            # Non-integer values
            if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                return fmt_compact_scientific(value)
            elif abs(value) >= 1000:
                return f"{value:,.1f}"
            elif abs(value) >= 1:
                return f"{value:.1f}"
            else:
                return f"{value:.3f}"

    def _render_axes(
        self, hist_data: HistogramData, inner_width: int, inner_height: int
    ) -> list[str]:
        """Render axes and tick labels."""
        parts = []

        # X-axis line
        parts.append(
            f'<line x1="{self.config.margin_left}" y1="{self.config.margin_top + inner_height}" '
            f'x2="{self.config.margin_left + inner_width}" y2="{self.config.margin_top + inner_height}" '
            f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
        )

        # Y-axis line
        parts.append(
            f'<line x1="{self.config.margin_left}" y1="{self.config.margin_top}" '
            f'x2="{self.config.margin_left}" y2="{self.config.margin_top + inner_height}" '
            f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
        )

        # X-axis ticks and labels
        parts.extend(self._render_x_axis_ticks(hist_data, inner_width, inner_height))

        # Y-axis ticks and labels
        parts.extend(self._render_y_axis_ticks(hist_data, inner_height))

        return parts

    def _render_x_axis_ticks(
        self, hist_data: HistogramData, inner_width: int, inner_height: int
    ) -> list[str]:
        """Render X-axis ticks and labels."""
        if len(hist_data.bin_centers) == 0:
            return []

        parts = []

        # Calculate tick positions
        if hist_data.scale == "log":
            # For log scale, use original_range if available for better tick generation
            if hist_data.original_range and hist_data.original_range[0] > 0:
                data_min, data_max = hist_data.original_range
                log_min = math.log10(data_min)
                log_max = math.log10(data_max)
                tick_positions, tick_labels = nice_log_ticks_from_log10(
                    log_min, log_max, 5
                )
                tick_values = [10**pos for pos in tick_positions]
            else:
                # Fallback to bin centers
                positive_centers = hist_data.bin_centers[hist_data.bin_centers > 0]
                if len(positive_centers) > 0:
                    log_min = math.log10(max(1e-10, positive_centers.min()))
                    log_max = math.log10(positive_centers.max())
                    tick_positions, tick_labels = nice_log_ticks_from_log10(
                        log_min, log_max, 5
                    )
                    tick_values = [10**pos for pos in tick_positions]
                else:
                    return []  # No valid data for ticks
        else:
            # For linear scale, use regular ticks
            tick_values, _ = nice_ticks(
                hist_data.bin_centers.min(), hist_data.bin_centers.max(), 5
            )
            tick_labels = [self._format_tick_label_standardized(v) for v in tick_values]

        # Render ticks and labels
        for tick_val, tick_label in zip(tick_values, tick_labels, strict=False):
            # Calculate position - bin_centers are always in linear space
            # (for log scale, they were converted back in _prepare_histogram_data)
            val_min = hist_data.bin_centers.min()
            val_max = hist_data.bin_centers.max()

            # Handle case where all values are the same (constant data)
            if val_max == val_min:
                x_pos = self.config.margin_left + inner_width / 2
            else:
                x_pos = (
                    self.config.margin_left
                    + (tick_val - val_min) / (val_max - val_min) * inner_width
                )

            # Only render if within bounds
            if (
                self.config.margin_left
                <= x_pos
                <= self.config.margin_left + inner_width
            ):
                # Tick line
                parts.append(
                    f'<line x1="{x_pos:.1f}" y1="{self.config.margin_top + inner_height}" '
                    f'x2="{x_pos:.1f}" y2="{self.config.margin_top + inner_height + self.config.tick_length}" '
                    f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
                )

                # The first and last labels are anchored to the plot edge
                # rather than centred on their tick, so a wide value at either
                # end sits inside the chart instead of overhanging it -- which
                # on the left ran under the y-axis labels and on the right ran
                # off the SVG entirely.
                left_edge = self.config.margin_left
                right_edge = self.config.margin_left + inner_width
                if x_pos - left_edge < 18:
                    anchor = "start"
                    label_x = left_edge
                elif right_edge - x_pos < 18:
                    anchor = "end"
                    label_x = right_edge
                else:
                    anchor = "middle"
                    label_x = x_pos

                parts.append(
                    f'<text x="{label_x:.1f}" y="{self.config.margin_top + inner_height + self.config.tick_length + 15}" '
                    f'text-anchor="{anchor}" class="tick-label" '
                    f'font-family="{self.config.font_family}" '
                    f'font-size="{self.config.label_font_size}" '
                    f'fill="{self.config.axis_color}">'
                    f"{self.safe_html_escape(tick_label)}</text>"
                )

        return parts

    def _render_y_axis_ticks(
        self, hist_data: HistogramData, inner_height: int
    ) -> list[str]:
        """Render Y-axis ticks and labels."""
        if hist_data.y_max == 0:
            return []

        parts = []
        y_ticks, _ = nice_ticks(0, hist_data.y_max, 5)

        for y_tick in y_ticks:
            y_pos = (
                self.config.margin_top + (1 - y_tick / hist_data.y_max) * inner_height
            )

            # Tick line
            parts.append(
                f'<line x1="{self.config.margin_left - self.config.tick_length}" y1="{y_pos:.1f}" '
                f'x2="{self.config.margin_left}" y2="{y_pos:.1f}" '
                f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
            )

            # Label
            label = self._format_tick_label_standardized(y_tick, is_count=True)
            parts.append(
                f'<text x="{self.config.margin_left - self.config.tick_length - 5}" y="{y_pos + 4:.1f}" '
                f'text-anchor="end" class="tick-label" '
                f'font-family="{self.config.font_family}" '
                f'font-size="{self.config.label_font_size}" '
                f'fill="{self.config.axis_color}">'
                f"{self.safe_html_escape(label)}</text>"
            )

        return parts

    def safe_html_escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return _html.escape(str(text))
