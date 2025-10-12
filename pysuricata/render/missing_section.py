"""Missing Values Section Renderer.

This module provides rendering functionality for the dataset-wide missing values section,
including heatmap, bar chart and spectrum visualizations.
"""

from __future__ import annotations

import html as _html

from .missing_values_heatmap import create_missing_values_heatmap_renderer


class MissingValuesSectionRenderer:
    """Renders the dataset-wide missing values section with heatmap, bar chart and spectrum tabs."""

    def render_section(
        self,
        kinds_map: dict[str, tuple[str, object]],
        accs: dict[str, object],
        n_rows: int,
        n_cols: int,
        total_missing_cells: int,
    ) -> str:
        """Main entry point - returns complete section HTML.

        Args:
            kinds_map: Dictionary mapping column names to (kind, accumulator) tuples
            accs: Dictionary mapping column names to accumulators
            n_rows: Total number of rows in dataset
            n_cols: Total number of columns in dataset
            total_missing_cells: Total number of missing cells across all variables

        Returns:
            Complete HTML string for missing values section
        """
        # Build miss_list from accumulators
        miss_list = []
        for name, (_, acc) in kinds_map.items():
            missing = getattr(acc, "missing", 0)
            count = getattr(acc, "count", 0)
            total = missing + count
            if total > 0:
                pct = (missing / total) * 100
                miss_list.append((name, pct, missing))

        # Sort by missing percentage descending
        miss_list.sort(key=lambda t: t[1], reverse=True)

        # Build heatmap HTML
        heatmap_html = self._build_heatmap(kinds_map, accs, n_rows, miss_list)

        # Create bar chart tab HTML
        bar_chart_html = self._build_bar_chart_tab(miss_list)

        # Create spectrum tab HTML
        spectrum_html = self._build_spectrum_tab(kinds_map, accs, n_rows)

        # Wrap in section with heatmap at top, then tabs and container
        return f"""
        {heatmap_html}

        <div class="missing-values-container">
            <div class="missing-section-tabs">
                <button class="active" data-tab="bar-chart">Bar Chart</button>
                <button data-tab="spectrum">Spectrum</button>
            </div>

            <div class="missing-tab-content active" data-tab="bar-chart">
                {bar_chart_html}
            </div>

            <div class="missing-tab-content" data-tab="spectrum">
                {spectrum_html}
            </div>
        </div>
        """

    def _build_heatmap(
        self,
        kinds_map: dict[str, tuple[str, object]],
        accs: dict[str, object],
        n_rows: int,
        miss_list: list[tuple[str, float, int]],
    ) -> str:
        """Build heatmap visualization for cross-column missing values.

        Args:
            kinds_map: Dictionary mapping column names to (kind, accumulator) tuples
            accs: Dictionary mapping column names to accumulators
            n_rows: Total number of rows in dataset
            miss_list: List of (column_name, missing_pct, missing_count) tuples

        Returns:
            HTML string for heatmap visualization
        """
        # Extract per-column chunk metadata
        per_column_chunk_metadata = {}
        column_names = list(kinds_map.keys())

        for name, (_, acc) in kinds_map.items():
            chunk_metadata = getattr(acc, "chunk_metadata", None)
            if chunk_metadata:
                per_column_chunk_metadata[name] = chunk_metadata
            else:
                # Fallback: create single segment for entire column
                missing = getattr(acc, "missing", 0)
                if missing > 0:
                    per_column_chunk_metadata[name] = [(0, n_rows, missing)]

        # Create and use heatmap renderer
        heatmap_renderer = create_missing_values_heatmap_renderer()
        return heatmap_renderer.render_heatmap(
            per_column_chunk_metadata, column_names, n_rows
        )

    def _build_bar_chart_tab(self, miss_list: list[tuple[str, float, int]]) -> str:
        """Build the bar chart tab showing missing percentages per variable.

        Args:
            miss_list: List of (column_name, missing_pct, missing_count) tuples

        Returns:
            HTML string for bar chart tab
        """
        if not miss_list:
            return """
            <div class="missing-bar-chart">
                <p style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    No missing values found in the dataset.
                </p>
            </div>
            """

        bar_items = []
        for name, pct, count in miss_list:
            severity_class = self._get_severity_class(pct)
            bar_items.append(f"""
            <div class="missing-bar-item">
                <div class="missing-var-name" title="{_html.escape(name)}">
                    {_html.escape(name)}
                </div>
                <div class="missing-bar-visual">
                    <div class="missing-bar-fill {severity_class}" style="width: {min(pct, 100):.1f}%;"></div>
                </div>
                <div class="missing-bar-label">
                    {count:,} ({pct:.1f}%)
                </div>
            </div>
            """)

        return f"""
        <div class="missing-bar-chart">
            {"".join(bar_items)}
        </div>
        """

    def _build_spectrum_tab(
        self,
        kinds_map: dict[str, tuple[str, object]],
        accs: dict[str, object],
        n_rows: int,
    ) -> str:
        """Build the spectrum tab showing per-variable chunk-level missing distributions.

        Args:
            kinds_map: Dictionary mapping column names to (kind, accumulator) tuples
            accs: Dictionary mapping column names to accumulators
            n_rows: Total number of rows in dataset

        Returns:
            HTML string for spectrum tab
        """
        spectrum_items = []

        for name, (_, acc) in kinds_map.items():
            missing = getattr(acc, "missing", 0)
            count = getattr(acc, "count", 0)
            total = missing + count

            if total == 0:
                continue

            pct = (missing / total) * 100
            chunk_metadata = getattr(acc, "chunk_metadata", [])

            if not chunk_metadata:
                # Fallback: create a single segment for the entire column
                spectrum_bar = self._create_spectrum_bar([(0, n_rows, missing)], n_rows)
            else:
                spectrum_bar = self._create_spectrum_bar(chunk_metadata, n_rows)

            spectrum_items.append(f"""
            <div class="missing-spectrum-item">
                <div class="missing-spectrum-header">
                    <span class="missing-var-name" title="{_html.escape(name)}">
                        {_html.escape(name)}
                    </span>
                    <span class="missing-bar-label">
                        {missing:,} ({pct:.1f}%)
                    </span>
                </div>
                {spectrum_bar}
            </div>
            """)

        if not spectrum_items:
            return """
            <div class="missing-spectrum-container">
                <p style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    No missing values found in the dataset.
                </p>
            </div>
            """

        return f"""
        <div class="missing-spectrum-container">
            {"".join(spectrum_items)}
        </div>
        """

    def _create_spectrum_bar(
        self,
        chunk_metadata: list[tuple[int, int, int]],
        n_rows: int,
    ) -> str:
        """Create a spectrum bar visualization for a single variable.

        Args:
            chunk_metadata: List of (start_row, end_row, missing_count) tuples
            n_rows: Total number of rows in dataset

        Returns:
            HTML string for spectrum bar
        """
        if not chunk_metadata or n_rows == 0:
            return '<div class="missing-spectrum-bar"></div>'

        segments = []
        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row
            if chunk_size == 0:
                continue

            missing_pct = (missing_count / chunk_size) * 100
            width_pct = (chunk_size / n_rows) * 100

            severity_class = self._get_severity_class(missing_pct)

            # Create tooltip data
            tooltip_data = f'data-start="{start_row}" data-end="{end_row}" data-missing="{missing_count}" data-total="{chunk_size}" data-pct="{missing_pct:.1f}"'

            segments.append(f"""
            <div class="spectrum-segment {severity_class}"
                 style="width: {width_pct:.2f}%;"
                 {tooltip_data}
                 title="Rows {start_row:,}-{end_row:,}: {missing_count:,} missing ({missing_pct:.1f}%)">
            </div>
            """)

        return f"""
        <div class="missing-spectrum-bar">
            {"".join(segments)}
        </div>
        """

    def _get_severity_class(self, pct: float) -> str:
        """Get CSS class based on missing percentage severity.

        Args:
            pct: Missing percentage

        Returns:
            CSS class name ('low', 'medium', or 'high')
        """
        if pct <= 5:
            return "low"
        elif pct <= 20:
            return "medium"
        else:
            return "high"
