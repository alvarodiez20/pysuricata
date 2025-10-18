"""Correlations Section Renderer.

This module provides rendering functionality for the dataset-wide correlations section,
featuring a ranked list of strongest correlations or a matrix view for small datasets.
"""

from __future__ import annotations

import html as _html


class CorrelationsSectionRenderer:
    """Renders the dataset-wide correlations section with list or matrix visualization.

    The renderer creates a view showing significant correlations between numeric columns.
    For small datasets (<15 columns), shows a matrix. For larger datasets, shows a
    ranked list of top correlations similar to the missing values section pattern.
    """

    def render_section(
        self,
        corr_est,
        numeric_columns: list[str],
        threshold: float = 0.5,
    ) -> str:
        """Main entry point - returns complete correlations section HTML.

        Args:
            corr_est: StreamingCorr estimator with full correlation matrix
            numeric_columns: List of numeric column names
            threshold: Minimum absolute correlation value to include (default: 0.5)

        Returns:
            Complete HTML string for correlations section
        """
        if len(numeric_columns) < 2:
            return self._render_no_correlations_state(
                "Correlation analysis requires at least 2 numeric columns"
            )

        # Collect all correlation data from estimator
        all_correlations = self._collect_correlations(corr_est, threshold)

        if not all_correlations:
            return self._render_no_correlations_state(
                f"No significant correlations found (threshold: {threshold:.2f})"
            )

        # Count total correlations
        n_correlations = len(all_correlations)

        # Decide rendering strategy based on number of columns
        if len(numeric_columns) <= 15:
            # Matrix view for small datasets
            correlations_html = self._render_correlation_matrix(
                all_correlations, numeric_columns
            )
        else:
            # List view for large datasets
            correlations_html = self._render_correlations_list(all_correlations)

        # Wrap in section container with header
        return f"""
        <div class="correlations-section-redesign">
            <div class="correlation-section-header">
                <h3 class="correlation-section-title">Correlation Analysis</h3>
                <span class="correlation-count-badge">{n_correlations} significant correlation{"s" if n_correlations != 1 else ""} found</span>
            </div>

            {correlations_html}

            <div class="correlation-legend">
                <span class="legend-item"><span class="color-box very-strong"></span>Very Strong (≥0.9)</span>
                <span class="legend-item"><span class="color-box strong"></span>Strong (0.7-0.9)</span>
                <span class="legend-item"><span class="color-box moderate"></span>Moderate (0.5-0.7)</span>
            </div>
        </div>
        """

    def _collect_correlations(
        self, corr_est, threshold: float
    ) -> list[tuple[str, str, float]]:
        """Extract all correlations from StreamingCorr estimator.

        Args:
            corr_est: StreamingCorr estimator with full correlation matrix
            threshold: Minimum absolute correlation value to include

        Returns:
            List of (col1, col2, correlation_value) tuples, deduplicated and sorted
        """
        if corr_est is None:
            return []

        # Get ALL correlations above threshold (not limited to 10 per column)
        # max_per_col=999 effectively means "no practical limit"
        top_map = corr_est.top_map(threshold=threshold, max_per_col=999)

        # Flatten and deduplicate pairs
        all_correlations = []
        seen_pairs = set()

        for col_name, corr_list in top_map.items():
            for other_col, corr_value in corr_list:
                # Ensure consistent ordering to avoid duplicates
                pair = tuple(sorted([col_name, other_col]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_correlations.append((*pair, corr_value))

        # Sort by absolute correlation strength (strongest first)
        return sorted(all_correlations, key=lambda x: abs(x[2]), reverse=True)

    def _render_correlations_list(
        self, sorted_correlations: list[tuple[str, str, float]]
    ) -> str:
        """Render scrollable list of top correlations (for large datasets).

        Args:
            sorted_correlations: List of (col1, col2, corr_value) sorted by strength

        Returns:
            HTML string for correlations list
        """
        bar_items = []

        # Show top 50 correlations (or all if less)
        display_count = min(50, len(sorted_correlations))

        for i, (col1, col2, corr) in enumerate(sorted_correlations[:display_count]):
            abs_corr = abs(corr)
            strength_class = self._get_strength_class(abs_corr)
            direction = "positive" if corr > 0 else "negative"
            direction_icon = "📈" if corr > 0 else "📉"
            strength_label = self._get_strength_label(abs_corr)

            rank = i + 1
            escaped_col1 = _html.escape(col1)
            escaped_col2 = _html.escape(col2)

            bar_items.append(
                f"""
            <div class="correlation-row">
                <div class="correlation-header">
                    <span class="rank-badge">#{rank}</span>
                    <code class="col-pair" title="{escaped_col1} ↔ {escaped_col2}">
                        <span class="col-name">{escaped_col1}</span>
                        <span class="arrow">↔</span>
                        <span class="col-name">{escaped_col2}</span>
                    </code>
                    <span class="correlation-value {direction}">
                        {corr:+.3f}
                    </span>
                </div>
                <div class="correlation-bar">
                    <div class="bar-fill {strength_class}"
                         style="width: {abs_corr * 100:.1f}%"
                         title="Correlation strength: {abs_corr:.3f}"></div>
                </div>
                <div class="correlation-meta">
                    <span class="strength-label {strength_class}">
                        {strength_label}
                    </span>
                    <span class="direction-indicator">
                        <span class="direction-icon">{direction_icon}</span>
                        <span class="direction-text">{direction.title()}</span>
                    </span>
                </div>
            </div>
            """
            )

        return f"""
        <div class="correlations-container">
            {"".join(bar_items)}
        </div>
        """

    def _render_correlation_matrix(
        self, correlations: list[tuple[str, str, float]], numeric_columns: list[str]
    ) -> str:
        """Render full correlation matrix heatmap (for small datasets).

        Args:
            correlations: List of (col1, col2, corr_value) tuples
            numeric_columns: List of all numeric column names

        Returns:
            HTML string for correlation matrix
        """
        # Build correlation lookup dictionary
        corr_dict = {}
        for col1, col2, corr in correlations:
            corr_dict[(col1, col2)] = corr
            corr_dict[(col2, col1)] = corr  # Symmetric

        # Build matrix HTML
        matrix_html = ['<table class="correlation-matrix">']

        # Header row
        matrix_html.append("<thead><tr><th></th>")
        for col in numeric_columns:
            escaped = _html.escape(col)
            matrix_html.append(f'<th title="{escaped}">{escaped}</th>')
        matrix_html.append("</tr></thead>")

        # Data rows
        matrix_html.append("<tbody>")
        for i, row_col in enumerate(numeric_columns):
            escaped_row = _html.escape(row_col)
            matrix_html.append(f'<tr><th title="{escaped_row}">{escaped_row}</th>')

            for j, col_col in enumerate(numeric_columns):
                if i == j:
                    # Diagonal: correlation with self = 1.0
                    matrix_html.append(
                        '<td class="corr-cell diagonal" data-corr="1.00">1.00</td>'
                    )
                else:
                    # Off-diagonal: look up correlation
                    corr = corr_dict.get((row_col, col_col), 0.0)
                    abs_corr = abs(corr)
                    strength_class = self._get_strength_class(abs_corr)
                    direction_class = "positive" if corr > 0 else "negative"

                    # Only show cell if correlation is significant (≥0.5)
                    if abs_corr >= 0.5:
                        matrix_html.append(
                            f'<td class="corr-cell {strength_class} {direction_class}" '
                            f'data-corr="{corr:.3f}" '
                            f'title="{escaped_row} ↔ {_html.escape(col_col)}: {corr:+.3f}">'
                            f"{corr:+.2f}</td>"
                        )
                    else:
                        # Weak correlation: show as faded
                        matrix_html.append(
                            f'<td class="corr-cell weak" data-corr="{corr:.3f}" '
                            f'title="{escaped_row} ↔ {_html.escape(col_col)}: {corr:+.3f}">'
                            f'<span class="weak-val">{corr:+.2f}</span></td>'
                        )

            matrix_html.append("</tr>")
        matrix_html.append("</tbody>")
        matrix_html.append("</table>")

        return f"""
        <div class="correlation-matrix-container">
            {"".join(matrix_html)}
        </div>
        """

    def _render_no_correlations_state(self, message: str) -> str:
        """Render empty state when no correlations are available.

        Args:
            message: Message to display

        Returns:
            HTML string for empty state
        """
        return f"""
        <div class="correlations-section-redesign">
            <div class="no-correlations-state">
                <span class="icon">📊</span>
                <p>{_html.escape(message)}</p>
            </div>
        </div>
        """

    def _get_strength_class(self, abs_corr: float) -> str:
        """Get CSS class based on correlation strength.

        Args:
            abs_corr: Absolute correlation value

        Returns:
            CSS class name
        """
        if abs_corr >= 0.9:
            return "very-strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        else:
            return "weak"

    def _get_strength_label(self, abs_corr: float) -> str:
        """Get human-readable strength label.

        Args:
            abs_corr: Absolute correlation value

        Returns:
            Strength label string
        """
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        else:
            return "Weak"
