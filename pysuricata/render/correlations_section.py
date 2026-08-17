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
        max_correlations: int = 100,
    ) -> str:
        """Main entry point - returns complete correlations section HTML.

        Args:
            corr_est: StreamingCorr estimator with full correlation matrix
            numeric_columns: List of numeric column names
            threshold: Minimum absolute correlation value to include (default: 0.5)
            max_correlations: Maximum number of correlations to display in list view (default: 100)

        Returns:
            Complete HTML string for correlations section
        """
        if len(numeric_columns) < 2:
            return self._render_too_few_columns_state(numeric_columns)

        # Everything, not only what clears the threshold. The pairs below it
        # were computed either way, and they are the *answer* in the common
        # case: both example reports have no pair above 0.5, and "no
        # significant correlations found" reports that as an absence when it is
        # a finding -- three pairs were checked and all three came back weak.
        every_pair = self._collect_correlations(corr_est, 0.0)
        all_correlations = [c for c in every_pair if abs(c[2]) >= threshold]

        if not all_correlations:
            return self._render_weak_state(every_pair, threshold)

        # Count total correlations
        n_correlations = len(all_correlations)

        # Decide rendering strategy based on number of columns
        # 10, not 15: at fifteen columns the cells are 30px and the labels stop
        # fitting, so the matrix becomes a grid of colours with nothing to say
        # which pair each one is. Two columns is one pair, which the list
        # states in a sentence and the matrix draws as a single cell.
        is_matrix_view = 2 < len(numeric_columns) <= self.MATRIX_MAX_COLUMNS

        if is_matrix_view:
            # Matrix view for small datasets
            # Every pair, not only the strong ones. A cell left blank because
            # its pair fell under the threshold is indistinguishable from one
            # that could not be computed -- and an all-weak row is information.
            correlations_html = self._render_correlation_matrix(
                every_pair, numeric_columns, threshold
            )
        else:
            # List view for large datasets
            correlations_html = self._render_correlations_list(
                all_correlations, max_correlations
            )

        # Build header based on view type
        if is_matrix_view:
            # Matrix view: show traditional header with title
            header_html = f"""
            <div class="correlation-section-header">
                <h3 class="correlation-section-title">Correlation Analysis</h3>
                <span class="correlation-count-badge">{n_correlations} pair{"s" if n_correlations != 1 else ""} above {threshold:.2f}, of {len(every_pair):,} checked</span>
            </div>
            """
            legend_html = ""  # No legend for matrix view
        else:
            # List view: replace title with legend in header
            header_html = f"""
            <div class="correlation-legend-header">
                <span class="correlation-count-badge">{n_correlations} pair{"s" if n_correlations != 1 else ""} above {threshold:.2f}, of {len(every_pair):,} checked</span>
                <div class="correlation-legend">
                    <span class="legend-item"><span class="color-box very-strong"></span>(≥0.9)</span>
                    <span class="legend-item"><span class="color-box strong"></span>(0.7-0.9)</span>
                    <span class="legend-item"><span class="color-box moderate"></span>(0.5-0.7)</span>
                </div>
            </div>
            """
            legend_html = ""  # No separate legend at bottom for list view

        # Wrap in section container with appropriate header
        return f"""
        <div class="correlations-section-redesign">
            {header_html}

            {correlations_html}

            {legend_html}
        </div>
        """

    #: Above this the matrix cells are too small to label.
    MATRIX_MAX_COLUMNS = 10

    #: Below-threshold pairs shown in the weak state. With 40 numeric columns
    #: there are 780 pairs; the point is to show that they were checked, not to
    #: print all of them.
    WEAK_SHOWN = 10

    def _render_weak_state(
        self, every_pair: list[tuple[str, str, float]], threshold: float
    ) -> str:
        """No pair cleared the threshold -- which is a result, not an absence.

        The old copy was an emoji and "No significant correlations found". But
        nothing was missing: the pairs were computed, and every one came back
        weak. Saying how many were checked and how strong the strongest was
        turns a shrug into an answer, and the numbers were already to hand.
        """
        if not every_pair:
            return self._render_nothing_comparable_state()

        checked = len(every_pair)
        strongest = every_pair[0]
        pairs = "pair" if checked == 1 else "pairs"
        verb = "is" if checked == 1 else "are"

        rows = "".join(
            self._render_weak_row(a, b, r) for a, b, r in every_pair[: self.WEAK_SHOWN]
        )
        shown = min(checked, self.WEAK_SHOWN)
        # "top 10 of 780 checked", never "10 pairs" -- the cap is not the count.
        caption = (
            f"Showing the {shown} strongest of {checked:,} checked"
            if checked > shown
            else f"All {checked:,} {pairs}"
        )

        return f"""
        <div class="corr-weak">
            <p class="corr-weak__lede">
                All <strong>{checked:,}</strong> numeric {pairs} {verb} weakly related.
                The strongest is <strong>{abs(strongest[2]):.3f}</strong>, under the
                <strong>{threshold:.2f}</strong> reporting threshold.
            </p>
            <p class="micro-label">{caption}</p>
            <ul class="corr-weak__list">{rows}</ul>
        </div>
        """

    def _render_weak_row(self, col_a: str, col_b: str, corr: float) -> str:
        """One below-threshold pair, on the same diverging bar as the list.

        `--data-3`, not `--data-4`. A quieter step for a weaker pair is the
        right instinct and `--data-4` is the wrong token to spend on it: it is
        **1.83:1 on the paper**, and `_00-tokens.css` records it as
        stack-internal only for that reason. This bar stands alone on the
        paper, so the step below it was invisible in print and close to it on
        screen -- the row rendered as a pair, a gap, and a number.

        Nothing is lost to ambiguity by sharing `--data-3` with the list's
        weakest band: this row only ever renders when *no* pair clears the
        threshold, so the two never appear in the same document.
        """
        return (
            '<li class="corr-weak__row">'
            f'<span class="corr-weak__pair">{self._escape(col_a)} · {self._escape(col_b)}</span>'
            f"{self._diverging_bar(corr, 'var(--data-3)')}"
            f'<span class="corr-weak__value">{corr:+.3f}</span>'
            "</li>"
        )

    def _diverging_bar(self, corr: float, fill: str) -> str:
        """Zero at the centre, negative left, positive right.

        Sign is position, not colour. A red bar for a negative correlation
        reads as *bad*, and a negative correlation is often the interesting
        one. This also survives greyscale and needs no legend.
        """
        magnitude = min(abs(corr), 1.0) * 50.0
        if corr < 0:
            left, width = 50.0 - magnitude, magnitude
        else:
            left, width = 50.0, magnitude
        return (
            '<span class="corr-bar" aria-hidden="true">'
            '<span class="corr-bar__zero"></span>'
            f'<span class="corr-bar__fill" style="left:{left:.2f}%;width:{width:.2f}%;'
            f'background:{fill}"></span>'
            "</span>"
        )

    @staticmethod
    def _escape(text: str) -> str:
        import html as _html

        return _html.escape(str(text))

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
        self, sorted_correlations: list[tuple[str, str, float]], max_display: int = 100
    ) -> str:
        """Render scrollable list of top correlations (for large datasets).

        Args:
            sorted_correlations: List of (col1, col2, corr_value) sorted by strength
            max_display: Maximum number of correlations to display (default: 100)

        Returns:
            HTML string for correlations list
        """
        bar_items = []

        # Show top N correlations (or all if less)
        display_count = min(max_display, len(sorted_correlations))

        for col1, col2, corr in sorted_correlations[:display_count]:
            abs_corr = abs(corr)
            # Three bands, three steps of one blue. Strength is how far the bar
            # runs; the step darkens with it rather than replacing it.
            fill = (
                "var(--data-1)"
                if abs_corr >= 0.9
                else "var(--data-2)"
                if abs_corr >= 0.7
                else "var(--data-3)"
            )
            escaped_col1 = _html.escape(col1)
            escaped_col2 = _html.escape(col2)

            # No rank badge. The list is ordered, so "#1" beside the first row
            # states what its position already says. No direction icon either:
            # the bar's side is the sign.
            bar_items.append(
                f"""
            <div class="correlation-row">
                <div class="col-pair" title="{escaped_col1} ↔ {escaped_col2}">
                    <span class="col-name correlation-col">{escaped_col1}</span>
                    <span class="arrow">↔</span>
                    <span class="col-name correlation-col">{escaped_col2}</span>
                </div>
                {self._diverging_bar(corr, fill)}
                <span class="correlation-value">{corr:+.3f}</span>
            </div>
            """
            )

        return f"""
        <div class="correlations-container">
            <div class="corr-scale micro-label" aria-hidden="true">
                <span>− 1.0</span><span>← 0 →</span><span>+ 1.0</span>
            </div>
            {"".join(bar_items)}
        </div>
        """

    def _render_correlation_matrix(
        self,
        correlations: list[tuple[str, str, float]],
        numeric_columns: list[str],
        threshold: float = 0.5,
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

        # Lower triangle only. The full square prints every pair twice and
        # spends a diagonal saying 1.00 once per column -- half the ink for
        # none of the information. The last column and the first row are
        # dropped with it, since they hold nothing but the mirror.
        matrix_html = ['<table class="correlation-matrix">']

        matrix_html.append("<thead><tr><th></th>")
        for col in numeric_columns[:-1]:
            escaped = _html.escape(col)
            matrix_html.append(f'<th title="{escaped}">{escaped}</th>')
        matrix_html.append("</tr></thead>")

        matrix_html.append("<tbody>")
        for i, row_col in enumerate(numeric_columns[1:], start=1):
            escaped_row = _html.escape(row_col)
            matrix_html.append(f'<tr><th title="{escaped_row}">{escaped_row}</th>')

            for j, col_col in enumerate(numeric_columns[:-1]):
                if j >= i:
                    matrix_html.append('<td class="corr-cell empty"></td>')
                    continue

                corr = corr_dict.get((row_col, col_col))
                if corr is None:
                    # A constant column has no variance and correlates with
                    # nothing. Showing that as 0.00 would claim a measurement
                    # that was never made.
                    matrix_html.append(
                        '<td class="corr-cell none" title="not comparable">·</td>'
                    )
                    continue

                # Below the threshold the cell stays visible and goes quiet.
                # Hiding it would make a weak pair look like an unmeasured one.
                strength = (
                    self._get_strength_class(abs(corr))
                    if abs(corr) >= threshold
                    else "weak"
                )
                escaped_pair = _html.escape(f"{row_col} ↔ {col_col}")
                # The tint is |r|; the sign is the printed number. A red cell
                # for a negative correlation reads as "bad", and a negative
                # correlation is often the interesting one.
                matrix_html.append(
                    f'<td class="corr-cell {strength}" data-corr="{corr:.2f}" '
                    f'title="{escaped_pair}: {corr:+.3f}">{corr:+.2f}</td>'
                )
            matrix_html.append("</tr>")
        matrix_html.append("</tbody></table>")

        return f'<div class="correlation-matrix-container">{"".join(matrix_html)}</div>'

    def _render_too_few_columns_state(self, numeric_columns: list[str]) -> str:
        """Fewer than two numeric columns, so there is no pair to correlate.

        This used to read "Correlation analysis requires at least 2 numeric
        columns", which states the rule and none of the case. The reader
        already knows a correlation needs two things; what they do not know is
        how many this frame has, or which one it is when it has one. Both are
        in hand at this point, and phase 6.1's enriched copy landed only on the
        path where pairs exist and come back weak -- so the two states that
        actually mean *nothing to compare* kept the bare line, and they are the
        ones a small frame hits. See #243.
        """
        count = len(numeric_columns)
        if count == 1:
            only = _html.escape(numeric_columns[0])
            body = (
                f"<p><strong>{only}</strong> is the only numeric column in this "
                "dataset. A correlation describes how two numbers move "
                "together, so one column has nothing to be compared with.</p>"
            )
        else:
            # "No column *is profiled as* numeric", not "this dataset has no
            # numeric columns". The second is a claim about the data and it can
            # be false: a column that never varies is reclassified as
            # categorical, so a frame of two constant floats reaches here and
            # would be told it has no numbers in it. The report's own Summary
            # says 0 numeric for the same frame, so this stays consistent with
            # it while pointing at the classification, which is the thing the
            # reader can actually look into.
            body = (
                "<p><strong>No column in this report is profiled as "
                "numeric</strong>, so there is nothing to correlate.</p>"
                "<p class='micro-label'>Numbers stored as text are profiled as "
                "categorical, and a column that never varies is too — check the "
                "Variables section for how each one was typed.</p>"
            )
        return f"""
        <div class="correlations-section-redesign">
            <div class="no-correlations-state">{body}</div>
        </div>
        """

    def _render_nothing_comparable_state(self) -> str:
        """Two or more numeric columns, and still not one usable coefficient.

        The reachable causes are worth naming rather than summarising as "no
        pairs available", because each one is actionable and the reader cannot
        tell them apart from the sentence alone: a column that never varies has
        no correlation defined with anything, and an estimator that saw too few
        complete rows has nothing to divide by.
        """
        return """
        <div class="correlations-section-redesign">
            <div class="no-correlations-state">
                <p>There are numeric columns here, but <strong>no pair produced
                a usable coefficient</strong>.</p>
                <p class="micro-label">A correlation is undefined when a column
                never varies, and cannot be estimated when too few rows have a
                value in both columns of a pair.</p>
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
