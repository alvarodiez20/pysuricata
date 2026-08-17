"""Boolean card rendering functionality."""

from .card_base import CardRenderer
from .card_config import DEFAULT_BOOL_CONFIG
from .card_types import BooleanStats, QualityFlags
from .triage import annotate_flags


class BooleanCardRenderer(CardRenderer):
    """Renders boolean data cards."""

    def __init__(self):
        super().__init__()
        self.bool_config = DEFAULT_BOOL_CONFIG

    def render_card(self, stats: BooleanStats) -> str:
        """Render a complete boolean card."""
        col_id = self.safe_col_id(stats.name)
        safe_name = self.safe_html_escape(stats.name)

        # Calculate percentages and quality flags
        total = int(stats.true_n + stats.false_n + stats.missing)
        cnt = int(stats.true_n + stats.false_n)
        miss_pct = (stats.missing / max(1, total)) * 100.0
        miss_cls = "crit" if miss_pct > 20 else ("warn" if miss_pct > 0 else "")

        true_pct_total = (stats.true_n / max(1, total)) * 100.0
        false_pct_total = (stats.false_n / max(1, total)) * 100.0

        quality_flags = self.quality_assessor.assess_boolean_quality(stats)
        quality_flags_html = self._build_quality_flags_html(
            quality_flags, cnt, miss_pct
        )

        # Build components
        stat_row = self._build_stat_row(
            self._left_stats(stats, cnt, miss_cls, miss_pct)
            + self._right_stats(stats, true_pct_total, false_pct_total)
        )

        # Chart (without card container)
        chart_html = self._build_boolean_chart(stats)

        details_html = self._build_details_section(col_id, stats, miss_pct)

        return self._assemble_card(
            col_id,
            safe_name,
            stats,
            quality_flags_html,
            stat_row,
            chart_html,
            details_html,
        )

    def _build_details_section(
        self, col_id: str, stats: BooleanStats, miss_pct: float
    ) -> str:
        """One pane, and only when it has something to say (#193).

        The note further down this file records why boolean had no details
        section: both its panes restated the card face. `Missing Values` was
        the interesting removal -- on numeric and datetime it survives when
        there is more than one chunk, because *where in the read the gaps fall*
        is something the card face cannot show, and boolean accumulators were
        finalized without chunk metadata so the pane had no such fact to offer
        and could not acquire one.

        It can now. `BooleanAccumulator` tracks chunks like the others, so the
        pane earns its tab back under exactly the rule the other kinds already
        use -- and stays absent on the single-chunk reports where it would only
        restate the header.
        """
        # The render-layer `BooleanStats` carries `true_n`/`false_n`, not a
        # `count` -- that is on the accumulator's summary, which is a different
        # type. Present values are the two that are not missing.
        present = int(stats.true_n + stats.false_n)
        total = present + int(stats.missing)
        present_pct = (present / max(1, total)) * 100.0 if total else 0.0
        table = super()._build_missing_values_table(
            present, present_pct, stats.missing, miss_pct, stats, total
        )

        return self._build_tabbed_details(
            col_id,
            [
                (
                    "missing",
                    "Missing Values",
                    f'<div class="sub">{table}</div>',
                    int(getattr(stats, "missing", 0) or 0) > 0
                    and len(getattr(stats, "chunk_metadata", None) or []) > 1,
                ),
            ],
        )

    def _build_quality_flags_html(
        self, flags: QualityFlags, cnt: int, miss_pct: float
    ) -> str:
        """The chips, with the number each one already knows on its face.

        `_quality_flags_markup` builds them; this puts the value on the
        chip and the threshold in a title. Splitting it this way means the
        forty-two places that emit a chip carry on emitting the same
        markup, and the annotation lives in one place rather than being
        repeated at every one of them.
        """
        return annotate_flags(self._quality_flags_markup(flags, cnt, miss_pct))

    def _quality_flags_markup(
        self, flags: QualityFlags, cnt: int, miss_pct: float
    ) -> str:
        """Build quality flags HTML for boolean data."""
        flag_items = []

        if flags.missing:
            severity = "bad" if miss_pct > 20 else "warn"
            threshold = ">20%" if miss_pct > 20 else "≤20%"
            flag_items.append(
                f'<li class="flag {severity}" data-threshold="{threshold}" '
                f'data-value="{miss_pct:.1f}%">Missing</li>'
            )

        if flags.constant:
            flag_items.append('<li class="flag bad">Constant</li>')

        if flags.imbalanced:
            flag_items.append('<li class="flag warn">Imbalanced</li>')

        return (
            f'<ul class="quality-flags">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )

    def _left_stats(
        self, stats: BooleanStats, cnt: int, miss_cls: str, miss_pct: float
    ) -> str:
        """Build left statistics table."""
        unique_vals = int(int(stats.true_n > 0) + int(stats.false_n > 0))

        data = [
            ("Count", f"{cnt:,}", "num"),
            ("Missing", f"{int(stats.missing):,} ({miss_pct:.1f}%)", f"num {miss_cls}"),
            ("Unique", f"{unique_vals}", "num"),
        ]

        return data

    def _right_stats(
        self, stats: BooleanStats, true_pct_total: float, false_pct_total: float
    ) -> str:
        """Build right statistics table."""
        mem_display = self.format_bytes(getattr(stats, "mem_bytes", 0))

        data = [
            ("True", f"{int(stats.true_n):,} ({true_pct_total:.1f}%)", "num"),
            ("False", f"{int(stats.false_n):,} ({false_pct_total:.1f}%)", "num"),
            ("Processed bytes (≈)", mem_display, "num"),
        ]

        return data

    def _build_boolean_chart(self, stats: BooleanStats) -> str:
        """Build boolean chart without card container."""
        svg = self._build_enhanced_boolean_stack_svg(
            int(stats.true_n), int(stats.false_n), int(stats.missing)
        )

        return f"""
        <div class="chart-container">
            {svg}
        </div>
        """

    def _build_enhanced_boolean_stack_svg(
        self,
        true_n: int,
        false_n: int,
        miss: int,
        *,
        width: int = 420,
        height: int | None = None,
    ) -> str:
        """One split bar for a two-valued column.

        The height comes from the config rather than a literal, which is what
        made the 48 there ineffective -- the default argument won and the bar
        stayed a 52px band, chart-sized, for a column with two values.
        """
        if height is None:
            height = self.bool_config.chart_height
        total = max(1, int(true_n + false_n + miss))
        margin = self.bool_config.margin
        inner_w = width - 2 * margin
        seg_h = height - 2 * margin

        w_false = int(inner_w * (false_n / total))
        w_true = int(inner_w * (true_n / total))
        w_miss = max(0, inner_w - w_false - w_true)

        # Flat fills from the data scale, and no gradient defs.
        #
        # `true` was green and `false` was red, which reads as good-versus-bad
        # -- the report passing judgement on someone's data. `Survived` is the
        # column that makes it obvious. Two values of one column get two steps
        # of one hue; the labels say which is which. See #110.
        #
        # The drop-shadow filter goes with them: decoration on a bar whose
        # length is the whole message. The card's layout is redesigned in #117.
        parts = [
            f'<svg class="bool-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        ]

        x = margin

        # False segment with enhanced styling
        if w_false > 0:
            false_pct = (false_n / total) * 100.0
            parts.append(
                f'<rect class="seg false enhanced" x="{x}" y="{margin}" width="{w_false}" height="{seg_h}" '
                f'fill="var(--data-4, #A8BECD)" '
                f'data-count="{false_n:,}" data-percentage="{false_pct:.1f}%" '
                f'data-type="false">'
                f"<title>False: {false_n:,} ({false_pct:.1f}%)</title>"
                f"</rect>"
            )

            # Each label takes the ink paired with the step beneath it. They were
            # all `fill="white"`, which is fine on --data-2 and close to illegible
            # on --data-4: white on #A8BECD is about 1.8:1, against the 4.5:1 a
            # label needs. --on-data-* exists for exactly this pairing, and the
            # token file already states which ink goes with which step.
            if w_false >= self.bool_config.min_segment_width:
                cx = x + w_false / 2
                parts.append(
                    f'<text class="label enhanced" x="{cx:.1f}" y="{margin + seg_h / 2 + 2:.1f}" '
                    f'text-anchor="middle" fill="var(--on-data-4, #22201C)" '
                    f'font-family="var(--font-mono)" font-size="12">'
                    f"false {false_pct:.1f}%"
                    f"</text>"
                )

        x += w_false

        # True segment with enhanced styling
        if w_true > 0:
            true_pct = (true_n / total) * 100.0
            parts.append(
                f'<rect class="seg true enhanced" x="{x}" y="{margin}" width="{w_true}" height="{seg_h}" '
                f'fill="var(--data-2, #3E6280)" '
                f'data-count="{true_n:,}" data-percentage="{true_pct:.1f}%" '
                f'data-type="true">'
                f"<title>True: {true_n:,} ({true_pct:.1f}%)</title>"
                f"</rect>"
            )

            # Add label if segment is wide enough
            if w_true >= self.bool_config.min_segment_width:
                cx = x + w_true / 2
                parts.append(
                    f'<text class="label enhanced" x="{cx:.1f}" y="{margin + seg_h / 2 + 2:.1f}" '
                    f'text-anchor="middle" fill="var(--on-data-2, #FBF9F5)" '
                    f'font-family="var(--font-mono)" font-size="12">'
                    f"true {true_pct:.1f}%"
                    f"</text>"
                )

        x += w_true

        # Missing segment with enhanced styling
        if w_miss > 0:
            miss_pct = (miss / total) * 100.0
            parts.append(
                f'<rect class="seg missing enhanced" x="{x}" y="{margin}" width="{w_miss}" height="{seg_h}" '
                f'fill="var(--track, #EDE6DA)" '
                f'data-count="{miss:,}" data-percentage="{miss_pct:.1f}%" '
                f'data-type="missing">'
                f"<title>Missing: {miss:,} ({miss_pct:.1f}%)</title>"
                f"</rect>"
            )

            # Add label if segment is wide enough
            if w_miss >= self.bool_config.min_segment_width:
                cx = x + w_miss / 2
                parts.append(
                    f'<text class="label enhanced" x="{cx:.1f}" y="{margin + seg_h / 2 + 2:.1f}" '
                    f'text-anchor="middle" fill="var(--ink, #22201C)" '
                    f'font-family="var(--font-mono)" font-size="12">'
                    f"missing {miss_pct:.1f}%"
                    f"</text>"
                )

        parts.append("</svg>")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    # One details pane, and only sometimes (#155 5c.6, then #193)
    #
    # A decision, not an omission, which is why it is written down.
    #
    # A boolean column has two values and two counts. The card face already
    # shows both, as a bar and as a percentage, and nothing is withheld -- so
    # there is no second level of disclosure to offer. What was here:
    #
    #   `Breakdown`      a two-row table under a card showing the same split.
    #   `Missing Values` one fact restated under a header already carrying it.
    #
    # `Breakdown` is gone for good. `Missing Values` was the more interesting
    # removal, and it has come back: on numeric and datetime it survives when
    # there is more than one chunk, because *where in the read the gaps fall*
    # is something the card face cannot show, and this accumulator was
    # finalized without chunk metadata -- so the pane had no such fact to offer
    # and could not acquire one. **#193 gave it one.** `BooleanAccumulator`
    # now tracks chunks like the others, and `_build_details_section` gates the
    # pane on exactly the rule the other three kinds use, so it stays absent on
    # the single-chunk reports where it would only restate the header.
    #
    # The one thing a boolean pane could add that the card cannot is a true
    # rate per chunk -- a flag that is 12% early and 60% late is a pipeline
    # change, and a single 38.4% hides it. That needs the same per-chunk counts
    # #193 is about, and it needs a caveat, because chunks are an artifact of
    # how the file was read: reorder the input and the chart changes.
    # ------------------------------------------------------------------ #

    def _build_dataprep_spectrum_visualization(self, stats: BooleanStats) -> str:
        """Build DataPrep-style spectrum visualization for missing values per chunk.

        This creates a single horizontal bar with segments representing actual processing
        chunks, colored by missing value density (green-yellow-red gradient).

        Args:
            stats: BooleanStats object containing chunk metadata and missing data information

        Returns:
            HTML string for the DataPrep-style spectrum visualization
        """
        # Check if we have chunk metadata
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            # If no chunk metadata, create a simple representation
            return self._build_simple_missing_distribution(stats)

        total_values = stats.true_n + stats.false_n + stats.missing
        if total_values == 0:
            return ""

        # Build the spectrum bar segments
        segments_html = ""
        total_width = 0

        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row + 1
            missing_pct = (
                (missing_count / chunk_size) * 100.0 if chunk_size > 0 else 0.0
            )

            # Calculate segment width as percentage of total
            segment_width_pct = (chunk_size / total_values) * 100.0
            total_width += segment_width_pct

            # Determine color based on missing percentage (DataPrep-style)
            if missing_pct <= 5:
                color_class = "spectrum-low"
            elif missing_pct <= 20:
                color_class = "spectrum-medium"
            else:
                color_class = "spectrum-high"

            segments_html += f"""
            <div class="spectrum-segment {color_class}"
                 style="width: {segment_width_pct:.2f}%"
                 data-start="{start_row}"
                 data-end="{end_row}"
                 data-missing="{missing_count}"
                 data-pct="{missing_pct:.1f}">
            </div>
            """

        # Build summary statistics
        total_chunks = len(chunk_metadata)
        max_missing_pct = max(
            (missing_count / (end_row - start_row + 1)) * 100.0
            for start_row, end_row, missing_count in chunk_metadata
        )
        avg_missing_pct = (
            sum(
                (missing_count / (end_row - start_row + 1)) * 100.0
                for start_row, end_row, missing_count in chunk_metadata
            )
            / total_chunks
        )

        # Determine overall severity
        if max_missing_pct >= 50:
            severity = "critical"
        elif max_missing_pct >= 20:
            severity = "high"
        elif max_missing_pct >= 5:
            severity = "medium"
        else:
            severity = "low"

        return f"""
        <div class="dataprep-spectrum">
            <div class="spectrum-header">
                <span class="spectrum-title">Missing Values Distribution</span>
                <span class="spectrum-stats">
                    {total_chunks} chunks • {max_missing_pct:.1f}% max • {avg_missing_pct:.1f}% avg
                </span>
            </div>
            <div class="spectrum-bar">
                {segments_html}
            </div>
            <div class="spectrum-legend">
                <span class="legend-item spectrum-low">Low (≤5%)</span>
                <span class="legend-item spectrum-medium">Medium (5-20%)</span>
                <span class="legend-item spectrum-high">High (>20%)</span>
            </div>
            <div class="spectrum-summary">
                <span class="severity-indicator {severity}">
                    {severity.title()} missing data severity
                </span>
            </div>
        </div>
        """

    def _build_simple_missing_distribution(self, stats: BooleanStats) -> str:
        """Build a simple missing distribution when no chunk metadata is available."""
        total = stats.true_n + stats.false_n + stats.missing
        if total == 0:
            return ""

        missing_pct = (stats.missing / total) * 100.0 if total > 0 else 0.0

        # Determine severity
        if missing_pct >= 50:
            severity = "critical"
        elif missing_pct >= 20:
            severity = "high"
        elif missing_pct >= 5:
            severity = "medium"
        else:
            severity = "low"

        return f"""
        <div class="dataprep-spectrum">
            <div class="spectrum-header">
                <span class="spectrum-title">Missing Values Distribution</span>
                <span class="spectrum-stats">
                    Single dataset • {missing_pct:.1f}% missing
                </span>
            </div>
            <div class="spectrum-bar">
                <div class="spectrum-segment spectrum-{"high" if missing_pct > 20 else "medium" if missing_pct > 5 else "low"}"
                     style="width: 100%"
                     title="Dataset: {stats.missing:,} missing ({missing_pct:.1f}%)">
                </div>
            </div>
            <div class="spectrum-legend">
                <span class="legend-item spectrum-low">Low (≤5%)</span>
                <span class="legend-item spectrum-medium">Medium (5-20%)</span>
                <span class="legend-item spectrum-high">High (>20%)</span>
            </div>
            <div class="spectrum-summary">
                <span class="severity-indicator {severity}">
                    {severity.title()} missing data severity
                </span>
            </div>
        </div>
        """

    def _assemble_card(
        self,
        col_id: str,
        safe_name: str,
        stats: BooleanStats,
        quality_flags_html: str,
        stat_row: str,
        chart_html: str,
        details_html: str = "",
    ) -> str:
        """Assemble the complete card HTML."""
        docs_url = "https://alvarodiez20.github.io/pysuricata/stats/boolean/"
        info_button = f'''<a href="{docs_url}" target="_blank" rel="noopener noreferrer" class="info-link" title="View documentation for Boolean analysis" aria-label="View Boolean analysis documentation">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path fill="currentColor" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM6.5 5a.75.75 0 0 0 0 1.5h.5v2.5h-.5a.75.75 0 0 0 0 1.5h3a.75.75 0 0 0 0-1.5h-.5V6h-.5A.75.75 0 0 0 8 5.25H6.5zM8 3.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5z"/>
            </svg>
        </a>'''

        return f"""
        <article class="var-card" id="{col_id}">
            <header class="var-card__header">
                <div class="title">
                    <span class="colname">{safe_name}</span>
                    <span class="badge">Boolean</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {quality_flags_html}
                </div>
                {info_button}
            </header>
            <div class="var-card__body">
                <div class="var-chart">{chart_html}</div>
                {stat_row}
            </div>
            {details_html}
        </article>
        """
