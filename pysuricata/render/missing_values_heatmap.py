"""
Cross-column missing values heatmap visualization.

This module provides a dataset-level view showing missing value patterns
across all columns simultaneously, with support for hundreds of columns.
"""

from __future__ import annotations

import html as _html
from typing import Dict, List, Tuple


class MissingValuesHeatmapRenderer:
    """Renders cross-column missing values heatmap for dataset-level analysis."""

    def __init__(self):
        """Initialize the heatmap renderer."""
        self.max_columns_initial = 20  # Show first 20 by default
        self.max_columns_expanded = 100  # Show up to 100 when expanded

    def render_heatmap(
        self,
        per_column_chunk_metadata: Dict[str, List[Tuple[int, int, int]]],
        column_names: List[str],
        total_rows: int,
    ) -> str:
        """Render cross-column missing values heatmap.

        Args:
            per_column_chunk_metadata: Dict mapping column name to chunk metadata
            column_names: List of all column names (in order)
            total_rows: Total number of rows in dataset

        Returns:
            HTML string with heatmap visualization
        """
        if not per_column_chunk_metadata or not column_names:
            return self._render_empty_state()

        # Filter to columns that have missing values
        columns_with_missing = []
        for col_name in column_names:
            if col_name in per_column_chunk_metadata:
                metadata = per_column_chunk_metadata[col_name]
                total_missing = sum(missing for _, _, missing in metadata)
                if total_missing > 0:
                    columns_with_missing.append((col_name, metadata, total_missing))

        if not columns_with_missing:
            return self._render_no_missing_state()

        # Sort by total missing count (descending)
        columns_with_missing.sort(key=lambda x: x[2], reverse=True)

        n_cols = len(columns_with_missing)
        show_expand = n_cols > self.max_columns_initial

        # Build heatmap rows
        initial_rows = self._build_heatmap_rows(
            columns_with_missing[: self.max_columns_initial], total_rows
        )

        if show_expand:
            remaining_rows = self._build_heatmap_rows(
                columns_with_missing[
                    self.max_columns_initial : self.max_columns_expanded
                ],
                total_rows,
            )
            expandable_section = f"""
            <div class="heatmap-expandable" hidden>
                {remaining_rows}
            </div>
            <button class="expand-heatmap-btn btn-soft" onclick="this.previousElementSibling.hidden = !this.previousElementSibling.hidden; this.textContent = this.previousElementSibling.hidden ? 'Show {n_cols - self.max_columns_initial} more columns ↓' : 'Show less ↑';">
                Show {n_cols - self.max_columns_initial} more columns ↓
            </button>
            """
        else:
            expandable_section = ""

        # Build legend
        legend_html = """
        <div class="heatmap-legend">
            <span class="legend-title">Missing % per chunk:</span>
            <span class="legend-item"><span class="color-box hm-none"></span>None (0%)</span>
            <span class="legend-item"><span class="color-box hm-low"></span>Low (0-5%)</span>
            <span class="legend-item"><span class="color-box hm-medium"></span>Medium (5-20%)</span>
            <span class="legend-item"><span class="color-box hm-high"></span>High (20-50%)</span>
            <span class="legend-item"><span class="color-box hm-critical"></span>Critical (50%+)</span>
        </div>
        """

        return f"""
        <div class="missing-values-heatmap">
            <div class="heatmap-header">
                <h4 class="heatmap-title">Cross-Column Missing Values Distribution</h4>
                <div class="heatmap-stats">
                    <span>{n_cols} columns with missing values</span>
                    <span>Showing top {min(n_cols, self.max_columns_initial)} initially</span>
                </div>
            </div>
            {legend_html}
            <div class="heatmap-container">
                <div class="heatmap-rows">
                    {initial_rows}
                    {expandable_section}
                </div>
            </div>
        </div>
        """

    def _build_heatmap_rows(
        self,
        columns_data: List[Tuple[str, List[Tuple[int, int, int]], int]],
        total_rows: int,
    ) -> str:
        """Build heatmap rows for given columns.

        Args:
            columns_data: List of (column_name, chunk_metadata, total_missing) tuples
            total_rows: Total rows in dataset

        Returns:
            HTML string with heatmap rows
        """
        rows_html = ""

        for col_name, chunk_metadata, total_missing in columns_data:
            total_vals = sum(end - start + 1 for start, end, _ in chunk_metadata)
            missing_pct = (total_missing / max(1, total_vals)) * 100.0

            # Build spectrum for this column
            spectrum_html = ""
            for start_row, end_row, missing_count in chunk_metadata:
                chunk_size = end_row - start_row + 1
                chunk_missing_pct = (
                    (missing_count / chunk_size) * 100.0 if chunk_size > 0 else 0.0
                )
                segment_width_pct = (
                    (chunk_size / total_vals) * 100.0 if total_vals > 0 else 0.0
                )

                # Determine color class (5 levels)
                if chunk_missing_pct == 0:
                    color_class = "hm-none"
                elif chunk_missing_pct <= 5:
                    color_class = "hm-low"
                elif chunk_missing_pct <= 20:
                    color_class = "hm-medium"
                elif chunk_missing_pct <= 50:
                    color_class = "hm-high"
                else:
                    color_class = "hm-critical"

                tooltip = f"{col_name} | Rows {start_row:,}-{end_row:,}: {missing_count:,} missing ({chunk_missing_pct:.1f}%)"

                spectrum_html += f"""
                <div class="hm-segment {color_class}"
                     style="width: {segment_width_pct:.2f}%"
                     title="{tooltip}"
                     data-column="{col_name}"
                     data-start="{start_row}"
                     data-end="{end_row}"
                     data-missing="{missing_count}"
                     data-pct="{chunk_missing_pct:.1f}"></div>
                """

            # Truncate long column names for display
            display_name = col_name if len(col_name) <= 25 else col_name[:22] + "..."

            rows_html += f"""
            <div class="heatmap-row">
                <div class="heatmap-label" title="{col_name}">
                    <code>{display_name}</code>
                    <span class="missing-summary">{total_missing:,} ({missing_pct:.1f}%)</span>
                </div>
                <div class="heatmap-spectrum">
                    {spectrum_html}
                </div>
            </div>
            """

        return rows_html

    def _render_empty_state(self) -> str:
        """Render empty state when no data available."""
        return """
        <div class="missing-values-heatmap empty">
            <p class="muted">No chunk metadata available</p>
        </div>
        """

    def _render_no_missing_state(self) -> str:
        """Render state when no columns have missing values."""
        return """
        <div class="missing-values-heatmap no-missing">
            <div class="success-message">
                <span class="icon">✓</span>
                <span class="message">No missing values detected in any column!</span>
            </div>
        </div>
        """


def create_missing_values_heatmap_renderer() -> MissingValuesHeatmapRenderer:
    """Factory function to create a heatmap renderer."""
    return MissingValuesHeatmapRenderer()
