"""Modern SVG temporal chart renderer for datetime distributions.

This module provides clean, accessible SVG bar charts for temporal patterns:
hour-of-day, day-of-week, month, and year distributions.
"""

from __future__ import annotations


class TemporalChartRenderer:
    """Renders modern temporal distribution charts as SVG."""

    def __init__(
        self,
        width: int = 400,
        height: int = 160,
        bar_color: str = "#3b82f6",
        bar_hover_color: str = "#60a5fa",
    ):
        """Initialize the temporal chart renderer.

        Args:
            width: Chart width in pixels
            height: Chart height in pixels
            bar_color: Default bar fill color
            bar_hover_color: Bar color on hover
        """
        self.width = width
        self.height = height
        self.bar_color = bar_color
        self.bar_hover_color = bar_hover_color

        # Layout constants
        self.margin_left = 45
        self.margin_right = 15
        self.margin_top = 20
        self.margin_bottom = 40

    def render_hour_chart(self, counts: list[int]) -> str:
        """Render hour-of-day distribution (0-23 hours).

        Args:
            counts: List of 24 integers representing counts for each hour

        Returns:
            SVG string
        """
        if not counts or len(counts) != 24:
            counts = [0] * 24

        labels = [f"{h:02d}:00" for h in range(24)]
        # Show every 3rd hour to avoid crowding
        visible_indices = list(range(0, 24, 3))

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Hour of Day Distribution",
        )

    def render_dow_chart(self, counts: list[int]) -> str:
        """Render day-of-week distribution.

        Args:
            counts: List of 7 integers (Monday=0 to Sunday=6)

        Returns:
            SVG string
        """
        if not counts or len(counts) != 7:
            counts = [0] * 7

        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        visible_indices = list(range(7))  # Show all

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Day of Week Distribution",
        )

    def render_month_chart(self, counts: list[int]) -> str:
        """Render month distribution.

        Args:
            counts: List of 12 integers (January=0 to December=11)

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
        visible_indices = list(range(12))  # Show all

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Month Distribution",
        )

    def render_year_chart(self, year_counts: dict[int, int]) -> str:
        """Render year distribution.

        Args:
            year_counts: Dictionary mapping year -> count

        Returns:
            SVG string
        """
        if not year_counts:
            return self._render_empty_chart("Year Distribution")

        # Sort years and extract counts
        sorted_years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in sorted_years]
        labels = [str(year) for year in sorted_years]

        # Show subset of labels if many years
        if len(labels) > 10:
            step = max(1, len(labels) // 6)
            visible_indices = (
                [0] + list(range(step, len(labels), step)) + [len(labels) - 1]
            )
        else:
            visible_indices = list(range(len(labels)))

        return self._render_bar_chart(
            counts=counts,
            labels=labels,
            visible_indices=visible_indices,
            title="Year Distribution",
        )

    def _render_bar_chart(
        self,
        counts: list[int],
        labels: list[str],
        visible_indices: list[int],
        title: str,
    ) -> str:
        """Render a bar chart with given data.

        Args:
            counts: Data values
            labels: Labels for each bar
            visible_indices: Indices of labels to show
            title: Chart title

        Returns:
            SVG string
        """
        if not counts or max(counts) == 0:
            return self._render_empty_chart(title)

        w = self.width
        h = self.height
        ml = self.margin_left
        mr = self.margin_right
        mt = self.margin_top
        mb = self.margin_bottom

        inner_w = w - ml - mr
        inner_h = h - mt - mb

        max_count = max(counts)
        total_count = sum(counts)
        n_bars = len(counts)
        bar_width = inner_w / n_bars
        bar_padding = bar_width * 0.15

        # Start SVG
        svg_parts = [
            f'<svg class="temporal-chart" width="{w}" height="{h}" ',
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">',
        ]

        # Background grid
        y_ticks = self._calculate_ticks(0, max_count, 5)
        for tick in y_ticks:
            if tick == 0:
                continue
            y = mt + inner_h * (1 - tick / max_count)
            svg_parts.append(
                f'<line class="grid-line" x1="{ml}" y1="{y:.1f}" '
                f'x2="{ml + inner_w}" y2="{y:.1f}" '
                f'stroke="#e5e7eb" stroke-opacity="0.5" />'
            )

        # Bars
        for i, count in enumerate(counts):
            if count == 0:
                continue

            x = ml + i * bar_width + bar_padding
            bw = bar_width - 2 * bar_padding
            bh = inner_h * (count / max_count)
            y = mt + inner_h - bh

            pct = (count / max(1, total_count)) * 100.0
            label = labels[i] if i < len(labels) else ""

            svg_parts.append(
                f'<rect class="bar" '
                f'x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'fill="{self.bar_color}" fill-opacity="0.8" '
                f'stroke="{self.bar_color}" stroke-width="1" '
                f'data-count="{count}" data-pct="{pct:.1f}" data-label="{label}" '
                f'rx="2" ry="2">'
                f"<title>{label}: {count:,} ({pct:.1f}%)</title>"
                f"</rect>"
            )

        # Axes
        axis_y = mt + inner_h
        svg_parts.append(
            f'<line class="axis" x1="{ml}" y1="{axis_y}" x2="{ml + inner_w}" y2="{axis_y}" '
            f'stroke="#6b7280" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{axis_y}" '
            f'stroke="#6b7280" stroke-width="1.5" />'
        )

        # Y-axis labels
        for tick in y_ticks:
            y = mt + inner_h * (1 - tick / max_count)
            svg_parts.append(
                f'<line class="tick" x1="{ml - 5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" '
                f'stroke="#6b7280" stroke-width="1" />'
            )
            svg_parts.append(
                f'<text class="tick-label" x="{ml - 8}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-family="system-ui, sans-serif" '
                f'font-size="11px" fill="#6b7280">'
                f"{self._format_count(tick)}</text>"
            )

        # X-axis labels
        for i in visible_indices:
            if i >= len(labels):
                continue
            x = ml + (i + 0.5) * bar_width
            svg_parts.append(
                f'<text class="x-label" x="{x:.1f}" y="{axis_y + 20}" '
                f'text-anchor="middle" font-family="system-ui, sans-serif" '
                f'font-size="11px" fill="#6b7280">'
                f"{labels[i]}</text>"
            )

        svg_parts.append("</svg>")
        return "".join(svg_parts)

    def _render_empty_chart(self, title: str) -> str:
        """Render an empty chart placeholder.

        Args:
            title: Chart title

        Returns:
            SVG string
        """
        w = self.width
        h = self.height

        return f'''<svg class="temporal-chart empty" width="{w}" height="{h}"
            viewBox="0 0 {w} {h}" role="img" aria-label="{title} (No Data)">
            <text x="{w / 2}" y="{h / 2}" text-anchor="middle"
                font-family="system-ui, sans-serif"
                font-size="14px"
                fill="#9ca3af">No data available</text>
        </svg>'''

    def _calculate_ticks(
        self, min_val: float, max_val: float, target_count: int
    ) -> list[float]:
        """Calculate nice tick values for an axis.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            target_count: Target number of ticks

        Returns:
            List of tick values
        """
        if max_val <= min_val:
            return [0]

        range_val = max_val - min_val
        rough_step = range_val / (target_count - 1)

        # Find nice step size
        magnitude = (
            10 ** int(f"{rough_step:.0e}".split("e")[1]) if rough_step > 0 else 1
        )
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
