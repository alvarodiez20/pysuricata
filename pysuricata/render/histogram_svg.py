"""
SVG-based histogram rendering for PySuricata.

This module provides a lightweight, high-performance histogram implementation
using SVG instead of Canvas/Chart.js. It handles large numbers intelligently
and provides better integration with the existing tooltip system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .format_utils import fmt_compact_scientific
from .svg_utils import nice_log_ticks_from_log10, nice_ticks


@dataclass
class HistogramConfig:
    """Configuration for histogram rendering."""

    width: int = 420
    height: int = 200
    margin_left: int = 60
    margin_right: int = 20
    margin_top: int = 20
    margin_bottom: int = 40

    # Bar styling - Professional Blue
    bar_color: str = "#3b82f6"
    bar_opacity: float = 0.7
    bar_stroke: str = "#1d4ed8"
    bar_stroke_width: float = 0.5

    # Axis styling
    axis_color: str = "#666"
    axis_stroke_width: float = 1.0
    tick_length: int = 5
    grid_color: str = "#eee"
    grid_opacity: float = 0.5

    # Text styling
    font_family: str = "Arial, sans-serif"
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
    original_range: Optional[Tuple[float, float]] = (
        None  # Original data range for log scale
    )


class SVGHistogramRenderer:
    """Renders histograms as SVG with intelligent number formatting."""

    def __init__(self, config: Optional[HistogramConfig] = None):
        self.config = config or HistogramConfig()

    def render_histogram(
        self,
        values: np.ndarray,
        bins: int = 25,
        scale: str = "lin",
        title: str = "",
        col_id: str = "",
    ) -> str:
        """
        Render a histogram as SVG.

        Args:
            values: Array of numeric values
            bins: Number of bins
            scale: Scale type ('lin' or 'log')
            title: Chart title
            col_id: Column ID for tooltips

        Returns:
            SVG string
        """
        if len(values) == 0:
            return self._render_empty_histogram(title)

        # Prepare data
        hist_data = self._prepare_histogram_data(values, bins, scale)

        # Calculate dimensions
        inner_width = (
            self.config.width - self.config.margin_left - self.config.margin_right
        )
        inner_height = (
            self.config.height - self.config.margin_top - self.config.margin_bottom
        )

        # Generate SVG
        svg_parts = [
            f'<svg class="hist-svg" width="{self.config.width}" height="{self.config.height}" '
            f'viewBox="0 0 {self.config.width} {self.config.height}" '
            f'role="img" aria-label="Histogram for {title}">'
        ]

        # Add grid
        svg_parts.extend(self._render_grid(hist_data, inner_width, inner_height))

        # Add bars
        svg_parts.extend(
            self._render_bars(hist_data, inner_width, inner_height, col_id)
        )

        # Add axes
        svg_parts.extend(self._render_axes(hist_data, inner_width, inner_height))

        # Add title
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

    def render_histogram_from_bins(
        self,
        bin_edges: List[float],
        bin_counts: List[int],
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

        # Create histogram data with new bins
        hist_data = HistogramData(
            counts=new_counts,
            edges=new_edges,
            bin_centers=new_bin_centers,
            total_count=int(np.sum(new_counts)),
            scale=scale,
            y_max=int(np.max(new_counts)) if len(new_counts) > 0 else 0,
        )

        # Calculate dimensions
        inner_width = (
            self.config.width - self.config.margin_left - self.config.margin_right
        )
        inner_height = (
            self.config.height - self.config.margin_top - self.config.margin_bottom
        )

        # Generate SVG
        svg_parts = [
            f'<svg class="hist-svg" width="{self.config.width}" height="{self.config.height}" '
            f'viewBox="0 0 {self.config.width} {self.config.height}" '
            f'role="img" aria-label="Histogram for {title}">'
        ]

        # Add grid
        svg_parts.extend(self._render_grid(hist_data, inner_width, inner_height))

        # Add bars
        svg_parts.extend(
            self._render_bars(hist_data, inner_width, inner_height, col_id)
        )

        # Add axes with standardized formatting
        svg_parts.extend(
            self._render_axes_standardized(hist_data, inner_width, inner_height)
        )

        # Add title
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

    def _prepare_histogram_data(
        self, values: np.ndarray, bins: int, scale: str
    ) -> HistogramData:
        """Prepare histogram data for rendering."""

        # Store original range for log scale tick generation
        original_range = None
        if len(values) > 0:
            original_range = (float(values.min()), float(values.max()))

        if scale == "log":
            # Filter out non-positive values for log scale
            positive_values = values[values > 0]
            if len(positive_values) == 0:
                return HistogramData(
                    counts=np.array([]),
                    edges=np.array([]),
                    bin_centers=np.array([]),
                    total_count=0,
                    scale=scale,
                    y_max=0,
                    original_range=original_range,
                )
            transformed_values = np.log10(positive_values)
        else:
            transformed_values = values

        # Create histogram
        counts, edges = np.histogram(transformed_values, bins=bins)

        # Calculate bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        # Convert back to linear space for log scale
        if scale == "log":
            bin_centers = 10**bin_centers
            edges = 10**edges

        return HistogramData(
            counts=counts,
            edges=edges,
            bin_centers=bin_centers,
            total_count=len(values),
            scale=scale,
            y_max=float(counts.max()) if len(counts) > 0 else 0,
            original_range=original_range,
        )

    def _render_empty_histogram(self, title: str) -> str:
        """Render an empty histogram when no data is available."""
        svg_parts = [
            f'<svg class="hist-svg" width="{self.config.width}" height="{self.config.height}" '
            f'viewBox="0 0 {self.config.width} {self.config.height}" '
            f'role="img" aria-label="Empty histogram for {title}">'
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
    ) -> List[str]:
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
                f'<line x1="{self.config.margin_left}" y1="{y_pos}" '
                f'x2="{self.config.margin_left + inner_width}" y2="{y_pos}" '
                f'stroke="{self.config.grid_color}" stroke-opacity="{self.config.grid_opacity}" '
                f'stroke-width="0.5"/>'
            )

        return parts

    def _render_bars(
        self, hist_data: HistogramData, inner_width: int, inner_height: int, col_id: str
    ) -> List[str]:
        """Render histogram bars."""
        if len(hist_data.counts) == 0 or hist_data.y_max == 0:
            return []

        parts = []
        bar_width = inner_width / len(hist_data.counts)

        for i, (count, center) in enumerate(
            zip(hist_data.counts, hist_data.bin_centers)
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
                f'fill="{self.config.bar_color}" fill-opacity="{self.config.bar_opacity}" '
                f'stroke="{self.config.bar_stroke}" stroke-width="{self.config.bar_stroke_width}" '
                f'data-count="{int(count)}" data-pct="{pct:.1f}" '
                f'data-x0="{x0_label}" data-x1="{x1_label}" '
                f'data-col="{col_id}"/>'
            )

        return parts

    def _render_axes_standardized(
        self, hist_data: HistogramData, inner_width: int, inner_height: int
    ) -> List[str]:
        """Render axes with standardized formatting for both linear and log scales."""
        parts = []

        # X-axis line
        parts.append(
            f'<line x1="{self.config.margin_left}" y1="{self.config.height - self.config.margin_bottom}" '
            f'x2="{self.config.width - self.config.margin_right}" y2="{self.config.height - self.config.margin_bottom}" '
            f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
        )

        # Y-axis line
        parts.append(
            f'<line x1="{self.config.margin_left}" y1="{self.config.margin_top}" '
            f'x2="{self.config.margin_left}" y2="{self.config.height - self.config.margin_bottom}" '
            f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
        )

        # Standardized tick generation
        if hist_data.scale == "log":
            # Log scale: use consistent approach
            # For log scale, we need to work with the original data range, not transformed bin centers
            if hasattr(hist_data, "original_range") and hist_data.original_range:
                # Use the original data range for tick generation
                data_min, data_max = hist_data.original_range
                if data_min > 0 and data_max > 0:
                    log_min = math.log10(data_min)
                    log_max = math.log10(data_max)

                    # Generate consistent log ticks
                    tick_positions, tick_labels = nice_log_ticks_from_log10(
                        log_min, log_max, 5
                    )
                    tick_values = [10**pos for pos in tick_positions]
                else:
                    tick_values = []
                    tick_labels = []
            else:
                # Fallback: use positive bin centers if original range not available
                positive_centers = hist_data.bin_centers[hist_data.bin_centers > 0]
                if len(positive_centers) > 0:
                    log_min = math.log10(max(1e-10, positive_centers.min()))
                    log_max = math.log10(positive_centers.max())

                    # Generate consistent log ticks
                    tick_positions, tick_labels = nice_log_ticks_from_log10(
                        log_min, log_max, 5
                    )
                    tick_values = [10**pos for pos in tick_positions]
                else:
                    tick_values = []
                    tick_labels = []
        else:
            # Linear scale: use consistent approach
            if len(hist_data.bin_centers) > 0:
                tick_values, _ = nice_ticks(
                    hist_data.bin_centers.min(), hist_data.bin_centers.max(), 5
                )
                tick_labels = [
                    self._format_tick_label_standardized(v) for v in tick_values
                ]
            else:
                tick_values = []
                tick_labels = []

        # Render X-axis ticks and labels
        for tick_val, tick_label in zip(tick_values, tick_labels):
            # Calculate position
            if hist_data.scale == "log":
                # Map log values to screen coordinates
                log_val = math.log10(max(1e-10, tick_val))

                # Use the same range calculation as for tick generation
                if hasattr(hist_data, "original_range") and hist_data.original_range:
                    data_min, data_max = hist_data.original_range
                    if data_min > 0 and data_max > 0:
                        log_min = math.log10(data_min)
                        log_max = math.log10(data_max)
                    else:
                        # Fallback to bin centers
                        positive_centers = hist_data.bin_centers[
                            hist_data.bin_centers > 0
                        ]
                        if len(positive_centers) > 0:
                            log_min = math.log10(max(1e-10, positive_centers.min()))
                            log_max = math.log10(positive_centers.max())
                        else:
                            log_min = log_max = log_val
                else:
                    # Fallback: use positive bin centers
                    positive_centers = hist_data.bin_centers[hist_data.bin_centers > 0]
                    if len(positive_centers) > 0:
                        log_min = math.log10(max(1e-10, positive_centers.min()))
                        log_max = math.log10(positive_centers.max())
                    else:
                        log_min = log_max = log_val

                # Handle case where all values are the same
                if log_max == log_min:
                    x_pos = self.config.margin_left + inner_width / 2
                else:
                    x_pos = (
                        self.config.margin_left
                        + (log_val - log_min) / (log_max - log_min) * inner_width
                    )
            else:
                # Map linear values to screen coordinates
                val_min = hist_data.bin_centers.min()
                val_max = hist_data.bin_centers.max()

                # Handle case where all values are the same
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
                <= self.config.width - self.config.margin_right
            ):
                # Tick mark
                parts.append(
                    f'<line x1="{x_pos}" y1="{self.config.height - self.config.margin_bottom}" '
                    f'x2="{x_pos}" y2="{self.config.height - self.config.margin_bottom + self.config.tick_length}" '
                    f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
                )

                # Label
                parts.append(
                    f'<text x="{x_pos}" y="{self.config.height - self.config.margin_bottom + 15}" '
                    f'text-anchor="middle" class="tick-label" '
                    f'font-family="{self.config.font_family}" font-size="{self.config.label_font_size}" '
                    f'fill="{self.config.axis_color}">'
                    f"{self.safe_html_escape(tick_label)}</text>"
                )

        # Y-axis ticks and labels
        if hist_data.y_max > 0:
            y_ticks = [
                0,
                hist_data.y_max // 4,
                hist_data.y_max // 2,
                3 * hist_data.y_max // 4,
                hist_data.y_max,
            ]
            y_ticks = [t for t in y_ticks if t >= 0]  # Remove negative ticks

            for tick_val in y_ticks:
                y_pos = (
                    self.config.height
                    - self.config.margin_bottom
                    - (tick_val / hist_data.y_max) * inner_height
                )

                # Tick mark
                parts.append(
                    f'<line x1="{self.config.margin_left - self.config.tick_length}" y1="{y_pos}" '
                    f'x2="{self.config.margin_left}" y2="{y_pos}" '
                    f'stroke="{self.config.axis_color}" stroke-width="{self.config.axis_stroke_width}"/>'
                )

                # Label
                parts.append(
                    f'<text x="{self.config.margin_left - 8}" y="{y_pos + 4}" '
                    f'text-anchor="end" class="tick-label" '
                    f'font-family="{self.config.font_family}" font-size="{self.config.label_font_size}" '
                    f'fill="{self.config.axis_color}">'
                    f"{self.safe_html_escape(self._format_tick_label_standardized(tick_val))}</text>"
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
        if is_count:
            # For histogram counts (y-axis), always format as integers
            if abs(value) >= 1000:
                return f"{int(value):,}"
            else:
                return f"{int(value)}"
        else:
            # For data values (x-axis ranges), use appropriate precision
            if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                return fmt_compact_scientific(value)
            elif abs(value) >= 1000:
                return f"{value:,.0f}"
            elif abs(value) >= 1:
                return f"{value:.1f}"
            else:
                return f"{value:.3f}"

    def _render_axes(
        self, hist_data: HistogramData, inner_width: int, inner_height: int
    ) -> List[str]:
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
    ) -> List[str]:
        """Render X-axis ticks and labels."""
        if len(hist_data.bin_centers) == 0:
            return []

        parts = []

        # Calculate tick positions
        if hist_data.scale == "log":
            # For log scale, use log ticks
            log_min = math.log10(max(1e-10, hist_data.bin_centers.min()))
            log_max = math.log10(hist_data.bin_centers.max())
            tick_positions, tick_labels = nice_log_ticks_from_log10(log_min, log_max, 5)
            tick_values = [10**pos for pos in tick_positions]
        else:
            # For linear scale, use regular ticks
            tick_values, _ = nice_ticks(
                hist_data.bin_centers.min(), hist_data.bin_centers.max(), 5
            )
            tick_labels = [self._format_tick_label_standardized(v) for v in tick_values]

        # Render ticks and labels
        for tick_val, tick_label in zip(tick_values, tick_labels):
            # Calculate position
            if hist_data.scale == "log":
                # Map log values to screen coordinates
                log_val = math.log10(max(1e-10, tick_val))
                log_min = math.log10(max(1e-10, hist_data.bin_centers.min()))
                log_max = math.log10(hist_data.bin_centers.max())

                # Handle case where all values are the same (constant data)
                if log_max == log_min:
                    x_pos = self.config.margin_left + inner_width / 2
                else:
                    x_pos = (
                        self.config.margin_left
                        + (log_val - log_min) / (log_max - log_min) * inner_width
                    )
            else:
                # Map linear values to screen coordinates
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

                # Label
                parts.append(
                    f'<text x="{x_pos:.1f}" y="{self.config.margin_top + inner_height + self.config.tick_length + 15}" '
                    f'text-anchor="middle" class="tick-label" '
                    f'font-family="{self.config.font_family}" '
                    f'font-size="{self.config.label_font_size}" '
                    f'fill="{self.config.axis_color}">'
                    f"{self.safe_html_escape(tick_label)}</text>"
                )

        return parts

    def _render_y_axis_ticks(
        self, hist_data: HistogramData, inner_height: int
    ) -> List[str]:
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
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )


def create_histogram_svg(
    values: np.ndarray,
    bins: int = 25,
    scale: str = "lin",
    title: str = "",
    col_id: str = "",
    config: Optional[HistogramConfig] = None,
) -> str:
    """
    Create a histogram SVG.

    Args:
        values: Array of numeric values
        bins: Number of bins
        scale: Scale type ('lin' or 'log')
        title: Chart title
        col_id: Column ID for tooltips
        config: Optional histogram configuration

    Returns:
        SVG string
    """
    renderer = SVGHistogramRenderer(config)
    return renderer.render_histogram(values, bins, scale, title, col_id)
