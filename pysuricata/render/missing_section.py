"""Missing Values Section Renderer.

This module provides rendering functionality for the dataset-wide missing values section
with a two-tab compact design: Data Completeness and Missing per Chunk.
"""

from __future__ import annotations

import html as _html


class MissingValuesSectionRenderer:
    """Renders the dataset-wide missing values section with two compact tabs.

    The renderer creates a two-tab interface:
    - Tab 1: Data Completeness - Shows overall missing stats with bars (40px rows)
    - Tab 2: Missing per Chunk - Shows chunk distribution spectrums (35px rows)

    Only columns with missing values are displayed.
    """

    def render_section(
        self,
        kinds_map: dict[str, tuple[str, object]],
        accs: dict[str, object],
        n_rows: int,
        n_cols: int,
        total_missing_cells: int,
    ) -> str:
        """Main entry point - returns complete section HTML with two tabs.

        Args:
            kinds_map: Dictionary mapping column names to (kind, accumulator) tuples
            accs: Dictionary mapping column names to accumulators
            n_rows: Total number of rows in dataset
            n_cols: Total number of columns in dataset
            total_missing_cells: Total number of missing cells across all variables

        Returns:
            Complete HTML string for missing values section
        """
        # Build list of columns with missing values only
        columns_with_missing = []
        for name, (_, acc) in kinds_map.items():
            missing = getattr(acc, "missing", 0)
            if missing > 0:  # Only include columns with missing values
                count = getattr(acc, "count", 0)
                total = missing + count
                pct = (missing / total) * 100 if total > 0 else 0
                chunk_metadata = getattr(acc, "chunk_metadata", None)
                columns_with_missing.append((name, pct, missing, chunk_metadata))

        # Sort by missing percentage descending
        columns_with_missing.sort(key=lambda t: t[1], reverse=True)

        # Count columns with missing values
        n_missing_cols = len(columns_with_missing)

        # Build both tabs
        completeness_tab_html = self._build_completeness_tab(columns_with_missing)
        chunk_tab_html = self._build_chunk_tab(columns_with_missing)

        # Two-tab layout with compact design
        return f"""
        <div class="missing-values-section-redesign">
            <div class="missing-tabs-header">
                <div class="missing-tabs">
                    <button class="active" data-tab="completeness">Data Completeness</button>
                    <button data-tab="chunks">Missing per Chunk</button>
                </div>
                <span class="missing-count-badge">{n_missing_cols} column{"s" if n_missing_cols != 1 else ""} with missing values</span>
            </div>

            <div class="missing-tab-content active" data-tab="completeness">
                {completeness_tab_html}
            </div>

            <div class="missing-tab-content" data-tab="chunks">
                {chunk_tab_html}
            </div>

            <div class="missing-legend">
                <span class="legend-item"><span class="color-box low"></span>Low (0-5%)</span>
                <span class="legend-item"><span class="color-box medium"></span>Medium (5-20%)</span>
                <span class="legend-item"><span class="color-box high"></span>High (20%+)</span>
            </div>
        </div>
        """

    def _build_completeness_tab(
        self,
        columns_with_missing: list[tuple[str, float, int, list | None]],
    ) -> str:
        """Build Data Completeness tab with compact 40px rows.

        Args:
            columns_with_missing: List of (column_name, missing_pct, missing_count, chunk_metadata) tuples

        Returns:
            HTML string for completeness tab content
        """
        if not columns_with_missing:
            return """
            <div class="scrollable-list">
                <div class="no-missing-state">
                    <span class="icon">✓</span>
                    <p>No missing values detected in any column</p>
                </div>
            </div>
            """

        rows = []
        for name, pct, count, _ in columns_with_missing:
            severity_class = self._get_severity_class(pct)
            escaped_name = _html.escape(name)

            rows.append(f"""
            <div class="compact-row">
                <code class="col-name" title="{escaped_name}">{escaped_name}</code>
                <span class="stats">{count:,} <span class="pct">({pct:.1f}%)</span></span>
                <div class="bar">
                    <div class="fill {severity_class}" style="width: {min(pct, 100):.1f}%"></div>
                </div>
            </div>
            """)

        return f"""
        <div class="scrollable-list">
            {"".join(rows)}
        </div>
        """

    def _build_chunk_tab(
        self,
        columns_with_missing: list[tuple[str, float, int, list | None]],
    ) -> str:
        """Build Missing per Chunk tab with ultra-compact 35px rows.

        Args:
            columns_with_missing: List of (column_name, missing_pct, missing_count, chunk_metadata) tuples

        Returns:
            HTML string for chunk tab content
        """
        if not columns_with_missing:
            return """
            <div class="scrollable-list">
                <div class="no-missing-state">
                    <span class="icon">✓</span>
                    <p>No missing values detected in any column</p>
                </div>
            </div>
            """

        rows = []
        for name, _, _, chunk_metadata in columns_with_missing:
            # Truncate long names to 20 characters
            short_name = name[:20] + "..." if len(name) > 20 else name
            escaped_name = _html.escape(short_name)
            escaped_full_name = _html.escape(name)

            # Create chunk spectrum
            spectrum_html = self._create_chunk_spectrum(chunk_metadata, name)

            rows.append(f"""
            <div class="chunk-row">
                <code class="short-name" title="{escaped_full_name}">{escaped_name}</code>
                {spectrum_html}
            </div>
            """)

        return f"""
        <div class="scrollable-list">
            {"".join(rows)}
        </div>
        """

    def _create_chunk_spectrum(
        self,
        chunk_metadata: list[tuple[int, int, int]] | None,
        col_name: str,
    ) -> str:
        """Create chunk spectrum with equal-width segments.

        Args:
            chunk_metadata: List of (start_row, end_row, missing_count) tuples for this column
            col_name: Column name for tooltip

        Returns:
            HTML string for chunk spectrum
        """
        if not chunk_metadata or len(chunk_metadata) == 0:
            # No chunk data: show single neutral segment
            return '<div class="spectrum"><div class="seg unknown" title="No chunk data available"></div></div>'

        segments = []

        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row + 1
            if chunk_size == 0:
                continue

            missing_pct = (missing_count / chunk_size) * 100 if chunk_size > 0 else 0
            severity_class = self._get_severity_class(missing_pct)

            # Create tooltip with chunk details
            tooltip = (
                f"{col_name} | Rows {start_row:,}-{end_row:,}: "
                f"{missing_count:,} missing ({missing_pct:.1f}%)"
            )

            segments.append(f"""
            <div class="seg {severity_class}"
                 title="{_html.escape(tooltip)}"
                 data-start="{start_row}"
                 data-end="{end_row}"
                 data-missing="{missing_count}"
                 data-pct="{missing_pct:.1f}"></div>
            """)

        return f'<div class="spectrum">{"".join(segments)}</div>'

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
