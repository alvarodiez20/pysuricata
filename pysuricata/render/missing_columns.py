"""Intelligent missing columns analysis and rendering utilities.

This module provides sophisticated logic for determining how many missing columns
to display in the summary section, with dynamic limits based on dataset size
and smart filtering to show only meaningful missing data.
"""

from __future__ import annotations

import html as _html


class MissingColumnsAnalyzer:
    """Analyzer for determining missing columns to display."""

    #: The one threshold, and it matches `ProfileConfig`.
    #:
    #: There were three. This constant said 0.0, the factory below said 0.5,
    #: and `config.py` said 0.0 -- and since the render path reads the config,
    #: the factory default had never actually applied to a report. So the same
    #: class filtered differently depending on how it was built, and the one
    #: value a reader could have found by reading the code was the wrong one.
    #:
    #: The value stays 0.0 rather than moving to the 0.5 the factory claimed:
    #: 0.0 is what every shipped report has used, and raising it would quietly
    #: drop columns from people's summaries. `ProfileConfig` is where to change
    #: it.
    MIN_THRESHOLD_PCT = 0.0

    #: How many rows the summary panel has space for.
    MAX_DISPLAY = 5

    def __init__(self, min_threshold_pct: float | None = None):
        """Initialize the analyzer with custom threshold.

        Args:
            min_threshold_pct: Minimum missing percentage to display. Defaults
                to `MIN_THRESHOLD_PCT`.
        """
        self.min_threshold_pct = (
            self.MIN_THRESHOLD_PCT if min_threshold_pct is None else min_threshold_pct
        )

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

        return self._render_columns_list(result.columns) + self._render_remainder(
            result
        )

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

    def _render_remainder(self, result: MissingColumnsResult) -> str:
        """`+ 18 more columns` when the list is cut short.

        `total_significant` was computed, stored on the result, and printed
        nowhere -- so a frame with 23 partially-missing columns showed five and
        read as though that were all of them. A list that truncates in silence
        is worse than a shorter list, because the reader has no way to know
        they are looking at part of the answer.
        """
        hidden = result.total_significant - len(result.columns)
        if hidden <= 0:
            return ""
        noun = "column" if hidden == 1 else "columns"
        # "above 0% missing" is a strange way to say "has any missing at all",
        # and 0 is the shipped default, so it is the phrasing most readers
        # would see.
        qualifier = (
            "with missing values"
            if result.threshold_used <= 0
            else f"above {result.threshold_used:g}% missing"
        )
        return f'<li class="miss-more">+ {hidden:,} more {noun} {qualifier}</li>'

    def _get_severity_class(self, pct: float) -> str:
        """Get CSS class based on missing percentage severity."""
        if pct <= 5:
            return "low"
        elif pct <= 20:
            return "medium"
        else:
            return "high"

    def _render_no_missing_columns(self) -> str:
        """A sentence, not a row.

        This used to render a full list item -- a `<code>` reading
        `No missing data`, a `0 (0.0%)` stat and a zero-width bar -- which is a
        table row impersonating data. An empty state should look like an empty
        state; the zero-width bar in particular is an element drawn to
        represent nothing.
        """
        return '<li class="miss-none">No column has missing values</li>'


def create_missing_columns_renderer(
    min_threshold_pct: float | None = None,
) -> MissingColumnsRenderer:
    """Factory function to create a configured missing columns renderer.

    Args:
        min_threshold_pct: Minimum missing percentage to display. Defaults to
            `MissingColumnsAnalyzer.MIN_THRESHOLD_PCT`, which is the only place
            the value is written down.

    Returns:
        Configured MissingColumnsRenderer instance
    """
    analyzer = MissingColumnsAnalyzer(min_threshold_pct)
    return MissingColumnsRenderer(analyzer)
