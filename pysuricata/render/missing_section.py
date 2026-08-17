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
                columns_with_missing.append(
                    (name, pct, missing, self._per_chunk_missing(acc))
                )

        # Sort by missing percentage descending
        columns_with_missing.sort(key=lambda t: t[1], reverse=True)

        # Count columns with missing values
        n_missing_cols = len(columns_with_missing)

        # Route on chunk count rather than on a tab. Two tabs over three rows
        # was two clicks for one screen of content -- and with a single chunk
        # the second tab is one full-width block per column, a tab that hides
        # nothing. The same shape of conditional the correlations section uses.
        chunk_count = self._chunk_count(columns_with_missing)

        if not columns_with_missing:
            body = self._render_nothing_missing(n_cols)
        elif chunk_count > 1:
            body = self._render_by_chunk(columns_with_missing, chunk_count)
        else:
            body = self._render_by_column(columns_with_missing)

        complete = max(0, n_cols - n_missing_cols)
        summary = (
            f"{n_missing_cols:,} of {n_cols:,} columns carry missing values"
            if n_missing_cols
            else f"All {n_cols:,} columns are complete"
        )
        footer = (
            f'<p class="miss-complete">{complete:,} of {n_cols:,} columns are complete</p>'
            if n_missing_cols and complete
            else ""
        )

        return f"""
        <div class="missing-values-section-redesign">
            <p class="micro-label">{summary}</p>
            {body}
            {self._render_legend() if columns_with_missing else ""}
            {footer}
        </div>
        """

    @staticmethod
    def _per_chunk_missing(acc: object) -> list[tuple[int, int, int]] | None:
        """This column's missing count per chunk, or None when it is not kept.

        The previous code read ``acc.chunk_metadata``, which no accumulator has
        -- the field lives on the *summary* that ``finalize()`` returns. It was
        therefore always None, and the "Missing per Chunk" tab has been
        rendering without data for as long as it has existed.

        Only the numeric accumulator tracks this. Categorical, datetime and
        boolean keep no per-chunk missing counts, so their rows have no strip
        to draw -- and the dataset-level chunk totals must not be drawn in
        their place, because a strip that is identical beside every column
        claims per-column information it does not have.
        """
        boundaries = getattr(acc, "_chunk_boundaries", None)
        per_chunk = getattr(acc, "_chunk_missing", None)
        if not boundaries or per_chunk is None:
            return None

        out: list[tuple[int, int, int]] = []
        start = 0
        for index, end_cumulative in enumerate(boundaries):
            if index >= len(per_chunk):
                break
            out.append((start, int(end_cumulative) - 1, int(per_chunk[index])))
            start = int(end_cumulative)
        return out or None

    @staticmethod
    def _chunk_count(
        columns_with_missing: list[tuple[str, float, int, list | None]],
    ) -> int:
        """How many chunks the stream was read in, as the accumulators saw it.

        Zero when `enable_chunk_metadata` is off, which has to degrade to the
        single-chunk view rather than draw an empty strip beside every row.
        """
        counts = [len(meta) for _, _, _, meta in columns_with_missing if meta]
        return max(counts) if counts else 1

    @staticmethod
    def _severity(pct: float) -> str:
        """The warm scale, which is the one place data uses it: here the
        encoding *is* severity, so 77% missing should look worse than 0.2%."""
        if pct > 20:
            return "bad"
        if pct >= 5:
            return "warn"
        return "good"

    def _render_legend(self) -> str:
        return (
            '<ul class="miss-legend">'
            '<li><span class="sw good"></span>≤5%</li>'
            '<li><span class="sw warn"></span>5–20%</li>'
            '<li><span class="sw bad"></span>&gt;20%</li>'
            "</ul>"
        )

    def _render_nothing_missing(self, n_cols: int) -> str:
        """One line, not an empty grid."""
        return (
            f'<p class="miss-none">No missing values in any of the '
            f"{n_cols:,} columns.</p>"
        )

    def _render_by_column(
        self, columns_with_missing: list[tuple[str, float, int, list | None]]
    ) -> str:
        """One row per column. Same shape as the summary's missing list, so a
        reader learns it once."""
        rows = []
        for name, pct, count, _ in columns_with_missing:
            rows.append(self._render_row(name, pct, count))
        return f'<ul class="miss-rows">{"".join(rows)}</ul>'

    def _render_row(self, name: str, pct: float, count: int, extra: str = "") -> str:
        severity = self._severity(pct)
        escaped = _html.escape(name)
        return (
            f'<li class="miss-row">'
            f'<code class="miss-row__name" title="{escaped}">{escaped}</code>'
            f'<span class="miss-row__bar"><span class="miss-row__fill {severity}" '
            f'style="width:{min(pct, 100.0):.1f}%"></span></span>'
            f'<span class="miss-row__value {severity}">{count:,} ({pct:.1f}%)</span>'
            f"{extra}</li>"
        )

    #: Beyond this the strip segments go sub-pixel and stop being readable.
    MAX_STRIP_SEGMENTS = 60

    def _render_by_chunk(
        self,
        columns_with_missing: list[tuple[str, float, int, list | None]],
        chunk_count: int,
    ) -> str:
        """Share missing *and* the sequence, as two encodings.

        One bar cannot carry both: at forty chunks a reader takes length for a
        total when it is a sequence. This is also the only place in the report
        that shows a column fine early and empty later, which is a pipeline
        problem rather than a data one.
        """
        rows = []
        for name, pct, count, meta in columns_with_missing:
            rows.append(self._render_row(name, pct, count, self._render_strip(meta)))
        note = (
            f'<p class="miss-strip-note">Each strip is {chunk_count:,} chunks, '
            "left to right</p>"
            if chunk_count <= self.MAX_STRIP_SEGMENTS
            else (
                f'<p class="miss-strip-note">Each strip samples '
                f"{self.MAX_STRIP_SEGMENTS} of {chunk_count:,} chunks, left to right</p>"
            )
        )
        return f'<ul class="miss-rows has-strip">{"".join(rows)}</ul>{note}'

    def _render_strip(self, meta: list | None) -> str:
        """One segment per chunk, coloured by that chunk's severity."""
        if not meta:
            # No rail rather than an empty one: this column kind keeps no
            # per-chunk counts, and a blank rail would read as "no missing
            # values in any chunk" -- a different claim from "not measured".
            return (
                '<span class="miss-strip is-untracked" '
                'title="per-chunk missing counts are not kept for this column"></span>'
            )

        chunks = list(meta)
        step = max(1, len(chunks) // self.MAX_STRIP_SEGMENTS)
        sampled = chunks[::step][: self.MAX_STRIP_SEGMENTS]

        segments = []
        for index, entry in enumerate(sampled):
            try:
                start_row, end_row, missing = entry[0], entry[1], entry[2]
            except (TypeError, IndexError):
                continue
            span = max(1, int(end_row) - int(start_row))
            pct = (int(missing) / span) * 100.0
            severity = self._severity(pct)
            segments.append(
                f'<span class="miss-seg {severity}" data-chunk="{index}" '
                f'data-start="{start_row}" data-end="{end_row}" '
                f'data-missing="{missing}" data-pct="{pct:.1f}" '
                f'title="rows {int(start_row):,}–{int(end_row):,}: '
                f'{int(missing):,} missing ({pct:.1f}%)"></span>'
            )
        return f'<span class="miss-strip">{"".join(segments)}</span>'

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
        <div class="completeness-bar-dual" title="Total: {total:,} values">
            <div class="bar-fill present" style="width: {present_pct:.1f}%"
                 title="{present_tooltip}"></div>
            <div class="bar-fill missing" style="width: {missing_pct:.1f}%"
                 title="{missing_tooltip}"></div>
        </div>
        """
