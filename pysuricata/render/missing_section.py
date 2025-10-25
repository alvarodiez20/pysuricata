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
        </div>
        """

    def _build_completeness_tab(
        self,
        columns_with_missing: list[tuple[str, float, int, list | None]],
    ) -> str:
        """Build Data Completeness tab with dual-color bars showing present and missing data.

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
        for name, missing_pct, missing_count, _ in columns_with_missing:
            # Calculate present values from the accumulator data
            # We need to get the count from the original data structure
            # The missing_pct and missing_count are already calculated, so we can derive present
            total = missing_count / (missing_pct / 100) if missing_pct > 0 else 0
            present_count = int(total - missing_count) if total > 0 else 0
            present_pct = 100 - missing_pct

            escaped_name = _html.escape(name)

            # Generate dual-color bar with rich tooltips
            dual_bar_html = self._render_dual_bar(
                present_pct, missing_pct, present_count, missing_count, int(total)
            )

            rows.append(f"""
            <div class="compact-row">
                <code class="col-name missing-col" title="{escaped_name}">{escaped_name}</code>
                {dual_bar_html}
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

    def _calculate_completeness_stats(
        self, count: int, missing: int
    ) -> tuple[int, float, float]:
        """Calculate present count and percentages for completeness display.

        Args:
            count: Number of non-missing values
            missing: Number of missing values

        Returns:
            Tuple of (present_count, present_pct, missing_pct)
        """
        total = count + missing
        present_pct = (count / total * 100) if total > 0 else 0
        missing_pct = (missing / total * 100) if total > 0 else 0
        return count, present_pct, missing_pct

    def _render_dual_bar(
        self,
        present_pct: float,
        missing_pct: float,
        present_count: int,
        missing_count: int,
        total: int,
    ) -> str:
        """Generate dual-color completeness bar HTML with rich tooltips.

        Args:
            present_pct: Percentage of present values
            missing_pct: Percentage of missing values
            present_count: Count of present values
            missing_count: Count of missing values
            total: Total number of values

        Returns:
            HTML string for dual-color bar
        """
        present_tooltip = f"Present: {present_count:,} ({present_pct:.1f}%)"
        missing_tooltip = f"Missing: {missing_count:,} ({missing_pct:.1f}%)"

        return f"""
        <div class="completeness-bar-dual" data-total="{total:,}">
            <div class="bar-fill present"
                 style="width: {present_pct:.1f}%"
                 title="{present_tooltip}"></div>
            <div class="bar-fill missing"
                 style="width: {missing_pct:.1f}%"
                 title="{missing_tooltip}"></div>
        </div>
        """
