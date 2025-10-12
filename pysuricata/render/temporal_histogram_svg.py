"""
SVG-based temporal histogram rendering for datetime data.

This module provides specialized histogram renderers for temporal patterns:
hour-of-day, day-of-week, month, and year distributions. All rendering is
done server-side for consistency and performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TemporalHistogramConfig:
    """Configuration for temporal histogram rendering."""

    width: int = 400
    height: int = 180
    margin_left: int = 50
    margin_right: int = 15
    margin_top: int = 15
    margin_bottom: int = 45

    # Bar styling - Temporal Blue theme
    bar_color: str = "#3b82f6"
    bar_opacity: float = 0.8
    bar_stroke: str = "#1d4ed8"
    bar_stroke_width: float = 1.0
    bar_hover_color: str = "#60a5fa"

    # Axis styling
    axis_color: str = "#6b7280"
    axis_stroke_width: float = 1.0
    tick_length: int = 5
    grid_color: str = "#e5e7eb"
    grid_opacity: float = 0.6

    # Text styling
    font_family: str = "system-ui, -apple-system, sans-serif"
    font_size: int = 11
    label_font_size: int = 10
    title_font_size: int = 13

    # Spacing
    bar_padding: float = 0.15  # Padding as fraction of bar width


class TemporalHistogramRenderer:
    """Renders temporal distribution histograms as SVG."""

    def __init__(self, config: Optional[TemporalHistogramConfig] = None):
        """Initialize the temporal histogram renderer.

        Args:
            config: Configuration for rendering. Uses defaults if None.
        """
        self.config = config or TemporalHistogramConfig()

    def render_hour_histogram(
        self,
        counts: List[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Render hour-of-day distribution (0-23 hours).

        Args:
            counts: List of 24 integers representing counts for each hour
            width: Optional custom width
            height: Optional custom height

        Returns:
            SVG string
        """
        if not counts or len(counts) != 24:
            counts = [0] * 24

        labels = [f"{h:02d}:00" for h in range(24)]
        # Show every 3rd hour label to avoid crowding
        visible_labels = [labels[i] if i % 3 == 0 else "" for i in range(24)]

        return self._render_categorical_histogram(
            counts=counts,
            labels=labels,
            visible_labels=visible_labels,
            title="Hour of Day",
            width=width,
            height=height,
        )

    def render_dow_histogram(
        self,
        counts: List[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Render day-of-week distribution (Mon-Sun).

        Args:
            counts: List of 7 integers representing counts for each day (Mon=0)
            width: Optional custom width
            height: Optional custom height

        Returns:
            SVG string
        """
        if not counts or len(counts) != 7:
            counts = [0] * 7

        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        return self._render_categorical_histogram(
            counts=counts,
            labels=labels,
            visible_labels=labels,  # All labels visible for DOW
            title="Day of Week",
            width=width,
            height=height,
        )

    def render_month_histogram(
        self,
        counts: List[int],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Render month distribution (Jan-Dec).

        Args:
            counts: List of 12 integers representing counts for each month
            width: Optional custom width
            height: Optional custom height

        Returns:
            SVG string
        """
        if not counts or len(counts) != 12:
            counts = [0] * 12

        labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        return self._render_categorical_histogram(
            counts=counts,
            labels=labels,
            visible_labels=labels,  # All labels visible for months
            title="Month",
            width=width,
            height=height,
        )

    def render_year_histogram(
        self,
        year_counts: Dict[int, int],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Render year distribution (dynamic years).

        Args:
            year_counts: Dictionary mapping year -> count
            width: Optional custom width
            height: Optional custom height

        Returns:
            SVG string
        """
        if not year_counts:
            return self._render_empty_histogram("Year", width, height)

        # Sort years and extract counts
        sorted_years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in sorted_years]
        labels = [str(year) for year in sorted_years]

        # For many years, show subset of labels
        if len(labels) > 10:
            # Show first, last, and every nth year
            step = max(1, len(labels) // 8)
            visible_labels = [
                labels[i] if (i == 0 or i == len(labels) - 1 or i % step == 0) else ""
                for i in range(len(labels))
            ]
        else:
            visible_labels = labels

        return self._render_categorical_histogram(
            counts=counts,
            labels=labels,
            visible_labels=visible_labels,
            title="Year",
            width=width,
            height=height,
        )

    def _render_categorical_histogram(
        self,
        counts: List[int],
        labels: List[str],
        visible_labels: List[str],
        title: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Render a categorical histogram with bars and labels.

        Args:
            counts: List of counts for each category
            labels: Full labels for tooltips
            visible_labels: Labels to display (may be subset of labels)
            title: Chart title
            width: Optional custom width
            height: Optional custom height

        Returns:
            SVG string
        """
        w = width or self.config.width
        h = height or self.config.height
        ml = self.config.margin_left
        mr = self.config.margin_right
        mt = self.config.margin_top
        mb = self.config.margin_bottom

        inner_w = w - ml - mr
        inner_h = h - mt - mb

        # Calculate stats
        total_count = sum(counts)
        max_count = max(counts) if counts else 1

        if max_count == 0:
            return self._render_empty_histogram(title, width, height)

        n_bars = len(counts)
        bar_width = inner_w / n_bars
        padding = bar_width * self.config.bar_padding

        # Build SVG
        parts = [
            f'<svg class="temporal-bar-chart" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title} Distribution">',
        ]

        # Add defs for gradients and patterns
        parts.append(self._build_defs())

        # Plot area group
        parts.append('<g class="plot-area">')

        # Grid lines (horizontal only for temporal charts)
        y_ticks = self._calculate_nice_ticks(0, max_count, 5)
        for tick in y_ticks:
            if tick == 0:
                continue
            y = mt + inner_h * (1 - tick / max_count)
            parts.append(
                f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml + inner_w}" y2="{y:.1f}" '
                f'stroke="{self.config.grid_color}" stroke-opacity="{self.config.grid_opacity}" />'
            )

        # Bars
        for i, count in enumerate(counts):
            if count == 0:
                continue

            x = ml + i * bar_width + padding
            bar_w = bar_width - 2 * padding
            bar_h = inner_h * (count / max_count)
            y = mt + inner_h - bar_h

            pct = (count / max(1, total_count)) * 100.0
            label = labels[i] if i < len(labels) else ""

            parts.append(
                f'<rect class="temporal-bar bar" '
                f'x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{self.config.bar_color}" '
                f'fill-opacity="{self.config.bar_opacity}" '
                f'stroke="{self.config.bar_stroke}" '
                f'stroke-width="{self.config.bar_stroke_width}" '
                f'data-count="{count}" data-pct="{pct:.1f}" data-label="{label}" '
                f'rx="2" ry="2">'
                f"<title>{label}: {count:,} ({pct:.1f}%)</title>"
                f"</rect>"
            )

        parts.append("</g>")

        # Axes
        axis_y = mt + inner_h
        parts.append(
            f'<line class="axis" x1="{ml}" y1="{axis_y}" x2="{ml + inner_w}" y2="{axis_y}" '
            f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}" />'
        )
        parts.append(
            f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{axis_y}" '
            f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}" />'
        )

        # Y-axis ticks and labels
        for tick in y_ticks:
            y = mt + inner_h * (1 - tick / max_count)
            parts.append(
                f'<line class="tick" x1="{ml - self.config.tick_length}" y1="{y:.1f}" '
                f'x2="{ml}" y2="{y:.1f}" '
                f'stroke="{self.config.axis_color}" stroke-width="1" />'
            )
            parts.append(
                f'<text class="tick-label" x="{ml - 8}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-family="{self.config.font_family}" '
                f'font-size="{self.config.label_font_size}" fill="{self.config.axis_color}">'
                f"{self._format_count(tick)}</text>"
            )

        # X-axis labels
        for i, label in enumerate(visible_labels):
            if not label:
                continue
            x = ml + (i + 0.5) * bar_width
            parts.append(
                f'<text class="tick-label x-tick-label" x="{x:.1f}" y="{axis_y + 20}" '
                f'text-anchor="middle" font-family="{self.config.font_family}" '
                f'font-size="{self.config.label_font_size}" fill="{self.config.axis_color}">'
                f"{label}</text>"
            )

        parts.append("</svg>")
        return "".join(parts)

    def _render_empty_histogram(
        self, title: str, width: Optional[int] = None, height: Optional[int] = None
    ) -> str:
        """Render an empty histogram placeholder.

        Args:
            title: Chart title
            width: Optional custom width
            height: Optional custom height

        Returns:
            SVG string
        """
        w = width or self.config.width
        h = height or self.config.height

        return f'''<svg class="temporal-bar-chart empty" width="{w}" height="{h}" 
            viewBox="0 0 {w} {h}" role="img" aria-label="{title} Distribution (No Data)">
            <text x="{w / 2}" y="{h / 2}" text-anchor="middle" 
                font-family="{self.config.font_family}" 
                font-size="{self.config.font_size}" 
                fill="#9ca3af">No data available</text>
        </svg>'''

    def _build_defs(self) -> str:
        """Build SVG defs section with gradients and filters."""
        return """<defs>
            <filter id="temporal-shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="1" stdDeviation="1" flood-opacity="0.1"/>
            </filter>
        </defs>"""

    def _calculate_nice_ticks(
        self, min_val: float, max_val: float, target_ticks: int = 5
    ) -> List[float]:
        """Calculate nice tick values for an axis.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            target_ticks: Target number of ticks

        Returns:
            List of tick values
        """
        if max_val <= min_val:
            return [0]

        range_val = max_val - min_val
        rough_step = range_val / (target_ticks - 1)

        # Find nice step size
        magnitude = 10 ** np.floor(np.log10(rough_step))
        residual = rough_step / magnitude

        if residual > 5:
            nice_step = 10 * magnitude
        elif residual > 2:
            nice_step = 5 * magnitude
        elif residual > 1:
            nice_step = 2 * magnitude
        else:
            nice_step = magnitude

        # Generate ticks
        ticks = []
        tick = 0
        while tick <= max_val:
            ticks.append(tick)
            tick += nice_step

        return ticks

    def _format_count(self, count: float) -> str:
        """Format count for display.

        Args:
            count: Count value

        Returns:
            Formatted string
        """
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        else:
            return f"{int(count)}"
