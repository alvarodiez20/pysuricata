"""Intelligent missing columns analysis and rendering utilities.

This module provides sophisticated logic for determining how many missing columns
to display in the summary section, with dynamic limits based on dataset size
and smart filtering to show only meaningful missing data.
"""

from __future__ import annotations

import html as _html
from typing import List, Optional, Tuple


class MissingColumnsAnalyzer:
    """Intelligent analyzer for determining missing columns display strategy."""

    # Configuration constants
    MIN_THRESHOLD_PCT = 0.0  # Show all columns with any missing data
    MAX_INITIAL_DISPLAY = 5  # Maximum columns shown initially
    MAX_EXPANDED_DISPLAY = 10  # Maximum columns shown when expanded

    # Dataset size thresholds
    SMALL_DATASET_COLS = 10
    MEDIUM_DATASET_COLS = 50
    LARGE_DATASET_COLS = 200

    def __init__(self, min_threshold_pct: float = MIN_THRESHOLD_PCT):
        """Initialize the analyzer with custom threshold.

        Args:
            min_threshold_pct: Minimum missing percentage to display (default: 0.0%)
        """
        self.min_threshold_pct = min_threshold_pct

    def analyze_missing_columns(
        self, miss_list: List[Tuple[str, float, int]], n_cols: int, n_rows: int
    ) -> MissingColumnsResult:
        """Analyze missing columns and determine display strategy.

        Args:
            miss_list: List of (column_name, missing_pct, missing_count) tuples
            n_cols: Total number of columns in dataset
            n_rows: Total number of rows in dataset

        Returns:
            MissingColumnsResult with display strategy and filtered data
        """
        # Filter out columns with no missing data
        significant_missing = [item for item in miss_list if item[1] > 0.0]

        # Determine dynamic limits based on dataset size
        initial_limit = self._get_initial_display_limit(n_cols, n_rows)
        expanded_limit = self._get_expanded_display_limit(n_cols, n_rows)

        # Determine if we need expandable UI
        needs_expandable = len(significant_missing) > initial_limit

        # Calculate display counts
        initial_display = significant_missing[:initial_limit]
        expanded_display = (
            significant_missing[:expanded_limit]
            if needs_expandable
            else initial_display
        )

        return MissingColumnsResult(
            initial_columns=initial_display,
            expanded_columns=expanded_display,
            needs_expandable=needs_expandable,
            total_significant=len(significant_missing),
            total_insignificant=len(miss_list) - len(significant_missing),
            threshold_used=self.min_threshold_pct,
        )

    def _get_initial_display_limit(self, n_cols: int, n_rows: int) -> int:
        """Determine initial display limit (fixed at 5 for all datasets)."""
        return self.MAX_INITIAL_DISPLAY

    def _get_expanded_display_limit(self, n_cols: int, n_rows: int) -> int:
        """Determine expanded display limit (fixed at 10 for all datasets)."""
        return self.MAX_EXPANDED_DISPLAY


class MissingColumnsResult:
    """Result of missing columns analysis."""

    def __init__(
        self,
        initial_columns: List[Tuple[str, float, int]],
        expanded_columns: List[Tuple[str, float, int]],
        needs_expandable: bool,
        total_significant: int,
        total_insignificant: int,
        threshold_used: float,
    ):
        self.initial_columns = initial_columns
        self.expanded_columns = expanded_columns
        self.needs_expandable = needs_expandable
        self.total_significant = total_significant
        self.total_insignificant = total_insignificant
        self.threshold_used = threshold_used


class MissingColumnsRenderer:
    """Renders missing columns HTML with expandable functionality."""

    def __init__(self, analyzer: Optional[MissingColumnsAnalyzer] = None):
        """Initialize renderer with optional custom analyzer."""
        self.analyzer = analyzer or MissingColumnsAnalyzer()

    def render_missing_columns_html(
        self, miss_list: List[Tuple[str, float, int]], n_cols: int, n_rows: int
    ) -> str:
        """Render complete missing columns HTML with intelligent display strategy.

        Args:
            miss_list: List of (column_name, missing_pct, missing_count) tuples
            n_cols: Total number of columns in dataset
            n_rows: Total number of rows in dataset

        Returns:
            Complete HTML string for missing columns section
        """
        result = self.analyzer.analyze_missing_columns(miss_list, n_cols, n_rows)

        if not result.initial_columns:
            return self._render_no_missing_columns()

        # Render initial display
        initial_html = self._render_columns_list(result.initial_columns, "initial")

        if not result.needs_expandable:
            # No wrapper needed - template provides it
            return initial_html

        # Render expandable version - only the ADDITIONAL columns beyond initial display
        additional_columns = result.expanded_columns[len(result.initial_columns) :]
        expanded_html = self._render_columns_list(additional_columns, "expanded")

        # Calculate how many additional columns we're actually showing
        num_additional = len(additional_columns)

        # Create unique ID for this dataset
        dataset_id = f"missing-cols-{n_cols}-{n_rows}".replace(" ", "")

        # Return just the list items - template provides the <ul> wrapper
        return f'''
            {initial_html}
            <div class="expandable-content" style="display: none;">
                {expanded_html}
            </div>
            <li class="expand-controls">
                <button class="expand-btn" onclick="toggleMissingColumns('{dataset_id}')"
                        data-expanded="false" aria-label="Show more missing columns" data-id="{dataset_id}">
                    <span class="expand-text">Show {num_additional} more...</span>
                    <span class="expand-icon">▼</span>
                </button>
            </li>
        <script>
        function toggleMissingColumns(datasetId) {{
            // Find the button and traverse up to the UL container
            const button = document.querySelector('[data-id="' + datasetId + '"]');
            const container = button.closest('.top-missing');
            const expandable = container.querySelector('.expandable-content');
            const text = container.querySelector('.expand-text');
            const icon = container.querySelector('.expand-icon');

            if (expandable.style.display === 'none') {{
                expandable.style.display = 'block';
                button.setAttribute('data-expanded', 'true');
                text.textContent = 'Show less...';
                icon.textContent = '▲';
            }} else {{
                expandable.style.display = 'none';
                button.setAttribute('data-expanded', 'false');
                text.textContent = 'Show {num_additional} more...';
                icon.textContent = '▼';
            }}
        }}
        </script>
        '''

    def _render_columns_list(
        self, columns: List[Tuple[str, float, int]], css_class: str = ""
    ) -> str:
        """Render a list of missing columns as HTML."""
        if not columns:
            return ""

        html_parts = []
        for col, pct, count in columns:
            severity_class = self._get_severity_class(pct)
            html_parts.append(f'''
            <li class="missing-item {css_class}">
              <div class="missing-info">
                <code class="missing-col" title="{_html.escape(str(col))}">{_html.escape(str(col))}</code>
                <span class="missing-stats">{count:,} ({pct:.1f}%)</span>
              </div>
              <div class="missing-bar"><div class="missing-fill {severity_class}" style="width:{pct:.1f}%;"></div></div>
            </li>
            ''')

        return "".join(html_parts)

    def _get_severity_class(self, pct: float) -> str:
        """Get CSS class based on missing percentage severity."""
        if pct <= 5:
            return "low"
        elif pct <= 20:
            return "medium"
        else:
            return "high"

    def _render_no_missing_columns(self) -> str:
        """Render HTML when no missing columns exist."""
        # Return just the list item - template provides the <ul> wrapper
        return """
            <li class="missing-item">
                <div class="missing-info">
                    <code class="missing-col">No missing data</code>
                    <span class="missing-stats">0 (0.0%)</span>
                </div>
                <div class="missing-bar">
                    <div class="missing-fill low" style="width:0%;"></div>
                </div>
            </li>
        """


def create_missing_columns_renderer(
    min_threshold_pct: float = 0.0,
) -> MissingColumnsRenderer:
    """Factory function to create a configured missing columns renderer.

    Args:
        min_threshold_pct: Minimum missing percentage to display

    Returns:
        Configured MissingColumnsRenderer instance
    """
    analyzer = MissingColumnsAnalyzer(min_threshold_pct)
    return MissingColumnsRenderer(analyzer)
