"""Intelligent missing columns analysis and rendering utilities.

This module provides sophisticated logic for determining how many missing columns
to display in the summary section, with dynamic limits based on dataset size
and smart filtering to show only meaningful missing data.
"""

from __future__ import annotations

import html as _html


class MissingColumnsAnalyzer:
    """Analyzer for determining missing columns to display."""

    # Configuration constants
    MIN_THRESHOLD_PCT = 0.0  # Show all columns with any missing data
    MAX_DISPLAY = 5  # Maximum columns shown (no expand)

    def __init__(self, min_threshold_pct: float = MIN_THRESHOLD_PCT):
        """Initialize the analyzer with custom threshold.

        Args:
            min_threshold_pct: Minimum missing percentage to display (default: 0.0%)
        """
        self.min_threshold_pct = min_threshold_pct

    def analyze_missing_columns(
        self, miss_list: list[tuple[str, float, int]], n_cols: int, n_rows: int
    ) -> MissingColumnsResult:
        """Analyze missing columns and determine what to display.

        Args:
            miss_list: List of (column_name, missing_pct, missing_count) tuples
            n_cols: Total number of columns in dataset
            n_rows: Total number of rows in dataset

        Returns:
            MissingColumnsResult with columns to display (max 5)
        """
        # Filter columns based on threshold
        significant_missing = [
            item for item in miss_list if item[1] > self.min_threshold_pct
        ]

        # Just return top 5, no expandable logic
        display_columns = significant_missing[: self.MAX_DISPLAY]

        return MissingColumnsResult(
            columns=display_columns,
            total_significant=len(significant_missing),
            total_insignificant=len(miss_list) - len(significant_missing),
            threshold_used=self.min_threshold_pct,
        )


class MissingColumnsResult:
    """Result of missing columns analysis."""

    def __init__(
        self,
        columns: list[tuple[str, float, int]],
        total_significant: int,
        total_insignificant: int,
        threshold_used: float,
    ):
        self.columns = columns
        self.total_significant = total_significant
        self.total_insignificant = total_insignificant
        self.threshold_used = threshold_used


class MissingColumnsRenderer:
    """Renders missing columns HTML (max 5 columns)."""

    def __init__(self, analyzer: MissingColumnsAnalyzer | None = None):
        """Initialize renderer with optional custom analyzer."""
        self.analyzer = analyzer or MissingColumnsAnalyzer()

    def render_missing_columns_html(
        self, miss_list: list[tuple[str, float, int]], n_cols: int, n_rows: int
    ) -> str:
        """Render missing columns HTML (max 5 columns).

        Args:
            miss_list: List of (column_name, missing_pct, missing_count) tuples
            n_cols: Total number of columns in dataset
            n_rows: Total number of rows in dataset

        Returns:
            HTML string for missing columns section (list items only)
        """
        result = self.analyzer.analyze_missing_columns(miss_list, n_cols, n_rows)

        if not result.columns:
            return self._render_no_missing_columns()

        # Always return just the initial list (max 5 items)
        return self._render_columns_list(result.columns)

    def _render_columns_list(self, columns: list[tuple[str, float, int]]) -> str:
        """Render a list of missing columns as HTML."""
        if not columns:
            return ""

        html_parts = []
        for col, pct, count in columns:
            severity_class = self._get_severity_class(pct)
            html_parts.append(f'''
            <li class="missing-item">
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
    min_threshold_pct: float = 0.5,
) -> MissingColumnsRenderer:
    """Factory function to create a configured missing columns renderer.

    Args:
        min_threshold_pct: Minimum missing percentage to display (default: 0.5%)

    Returns:
        Configured MissingColumnsRenderer instance
    """
    analyzer = MissingColumnsAnalyzer(min_threshold_pct)
    return MissingColumnsRenderer(analyzer)
