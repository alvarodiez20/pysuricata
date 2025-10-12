"""Boolean card rendering functionality."""

from .card_base import CardRenderer, QualityAssessor, TableBuilder
from .card_config import DEFAULT_BOOL_CONFIG
from .card_types import BooleanStats, QualityFlags


def ordinal_number(n):
    """Convert a number to its ordinal form with superscript suffix (1ˢᵗ, 2ⁿᵈ, 3ʳᵈ, 4ᵗʰ, etc.)"""
    # Keep the number normal size, only make the suffix superscript
    number_str = str(n)

    # Add ordinal suffix (only the suffix is superscript)
    if 10 <= n % 100 <= 20:
        suffix = "ᵗʰ"
    else:
        suffix_map = {1: "ˢᵗ", 2: "ⁿᵈ", 3: "ʳᵈ"}
        suffix = suffix_map.get(n % 10, "ᵗʰ")

    return f"{number_str}{suffix}"


class BooleanCardRenderer(CardRenderer):
    """Renders boolean data cards."""

    def __init__(self):
        super().__init__()
        self.quality_assessor = QualityAssessor()
        self.table_builder = TableBuilder()
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
        left_table = self._build_left_table(stats, cnt, miss_cls, miss_pct)
        right_table = self._build_right_table(stats, true_pct_total, false_pct_total)

        # Chart (without card container)
        chart_html = self._build_boolean_chart(stats)

        # Details
        details_html = self._build_details_section(
            col_id, stats, true_pct_total, false_pct_total, miss_pct
        )

        return self._assemble_card(
            col_id,
            safe_name,
            stats,
            quality_flags_html,
            left_table,
            right_table,
            chart_html,
            details_html,
        )

    def _build_quality_flags_html(
        self, flags: QualityFlags, cnt: int, miss_pct: float
    ) -> str:
        """Build quality flags HTML for boolean data."""
        flag_items = []

        if flags.missing:
            severity = "bad" if miss_pct > 20 else "warn"
            flag_items.append(f'<li class="flag {severity}">Missing</li>')

        if flags.constant:
            flag_items.append('<li class="flag bad">Constant</li>')

        if flags.imbalanced:
            flag_items.append('<li class="flag warn">Imbalanced</li>')

        return (
            f'<ul class="quality-flags">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )

    def _build_left_table(
        self, stats: BooleanStats, cnt: int, miss_cls: str, miss_pct: float
    ) -> str:
        """Build left statistics table."""
        unique_vals = int(int(stats.true_n > 0) + int(stats.false_n > 0))

        data = [
            ("Count", f"{cnt:,}", "num"),
            ("Missing", f"{int(stats.missing):,} ({miss_pct:.1f}%)", f"num {miss_cls}"),
            ("Unique", f"{unique_vals}", "num"),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_right_table(
        self, stats: BooleanStats, true_pct_total: float, false_pct_total: float
    ) -> str:
        """Build right statistics table."""
        mem_display = self.format_bytes(getattr(stats, "mem_bytes", 0)) + " (≈)"

        data = [
            ("True", f"{int(stats.true_n):,} ({true_pct_total:.1f}%)", "num"),
            ("False", f"{int(stats.false_n):,} ({false_pct_total:.1f}%)", "num"),
            ("Processed bytes", mem_display, "num"),
        ]

        return self.table_builder.build_key_value_table(data)

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
        height: int = 60,
    ) -> str:
        """Build enhanced boolean stack SVG with improved styling and hover effects."""
        total = max(1, int(true_n + false_n + miss))
        margin = self.bool_config.margin
        inner_w = width - 2 * margin
        seg_h = height - 2 * margin

        w_false = int(inner_w * (false_n / total))
        w_true = int(inner_w * (true_n / total))
        w_miss = max(0, inner_w - w_false - w_true)

        # Enhanced SVG with better styling
        parts = [
            f'<svg class="bool-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '    <linearGradient id="trueGradient" x1="0%" y1="0%" x2="0%" y2="100%">',
            '        <stop offset="0%" style="stop-color:#10b981;stop-opacity:1" />',
            '        <stop offset="100%" style="stop-color:#059669;stop-opacity:1" />',
            "    </linearGradient>",
            '    <linearGradient id="falseGradient" x1="0%" y1="0%" x2="0%" y2="100%">',
            '        <stop offset="0%" style="stop-color:#ef4444;stop-opacity:1" />',
            '        <stop offset="100%" style="stop-color:#dc2626;stop-opacity:1" />',
            "    </linearGradient>",
            '    <linearGradient id="missingGradient" x1="0%" y1="0%" x2="0%" y2="100%">',
            '        <stop offset="0%" style="stop-color:#6b7280;stop-opacity:1" />',
            '        <stop offset="100%" style="stop-color:#4b5563;stop-opacity:1" />',
            "    </linearGradient>",
            '    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
            '        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.1"/>',
            "    </filter>",
            "</defs>",
        ]

        x = margin

        # False segment with enhanced styling
        if w_false > 0:
            false_pct = (false_n / total) * 100.0
            parts.append(
                f'<rect class="seg false enhanced" x="{x}" y="{margin}" width="{w_false}" height="{seg_h}" '
                f'fill="url(#falseGradient)" filter="url(#shadow)" rx="4" ry="4" '
                f'data-count="{false_n:,}" data-percentage="{false_pct:.1f}%" '
                f'data-type="false">'
                f"<title>False: {false_n:,} ({false_pct:.1f}%)</title>"
                f"</rect>"
            )

            # Add label if segment is wide enough
            if w_false >= self.bool_config.min_segment_width:
                cx = x + w_false / 2
                parts.append(
                    f'<text class="label enhanced" x="{cx:.1f}" y="{margin + seg_h / 2 + 2:.1f}" '
                    f'text-anchor="middle" fill="white" font-weight="600" font-size="12">'
                    f"False {false_pct:.1f}%"
                    f"</text>"
                )

        x += w_false

        # True segment with enhanced styling
        if w_true > 0:
            true_pct = (true_n / total) * 100.0
            parts.append(
                f'<rect class="seg true enhanced" x="{x}" y="{margin}" width="{w_true}" height="{seg_h}" '
                f'fill="url(#trueGradient)" filter="url(#shadow)" rx="4" ry="4" '
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
                    f'text-anchor="middle" fill="white" font-weight="600" font-size="12">'
                    f"True {true_pct:.1f}%"
                    f"</text>"
                )

        x += w_true

        # Missing segment with enhanced styling
        if w_miss > 0:
            miss_pct = (miss / total) * 100.0
            parts.append(
                f'<rect class="seg missing enhanced" x="{x}" y="{margin}" width="{w_miss}" height="{seg_h}" '
                f'fill="url(#missingGradient)" filter="url(#shadow)" rx="4" ry="4" '
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
                    f'text-anchor="middle" fill="white" font-weight="600" font-size="12">'
                    f"Missing {miss_pct:.1f}%"
                    f"</text>"
                )

        parts.append("</svg>")
        return "".join(parts)

    def _build_details_section(
        self,
        col_id: str,
        stats: BooleanStats,
        true_pct_total: float,
        false_pct_total: float,
        miss_pct: float,
    ) -> str:
        """Build details section with breakdown and missing values tables."""

        # Build enhanced breakdown table (Common Values style)
        breakdown_table = self._build_common_values_style_table(
            stats, true_pct_total, false_pct_total, miss_pct
        )

        # Build missing values table with distribution
        missing_table = self._build_missing_values_table(stats, miss_pct)

        return f"""
        <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
                <button role="tab" class="active" data-tab="breakdown">Breakdown</button>
                <button role="tab" data-tab="missing">Missing Values</button>
            </nav>
            <div class="tab-panes">
                <section class="tab-pane active" data-tab="breakdown">
                    <div class="sub"><div class="hdr">Value Distribution</div>{breakdown_table}</div>
                </section>
                <section class="tab-pane" data-tab="missing">
                    <div class="sub"><div class="hdr">Missing Values</div>{missing_table}</div>
                </section>
            </div>
        </section>
        """

    def _build_common_values_style_table(
        self,
        stats: BooleanStats,
        true_pct_total: float,
        false_pct_total: float,
        miss_pct: float,
    ) -> str:
        """Build breakdown table in Common Values style similar to numeric cards."""
        int(stats.true_n + stats.false_n + stats.missing)

        # Create value entries with ranking
        values_data = []

        # Add True value
        if stats.true_n > 0:
            values_data.append(("True", stats.true_n, true_pct_total, 1))

        # Add False value
        if stats.false_n > 0:
            values_data.append(("False", stats.false_n, false_pct_total, 2))

        # Add Missing value if present
        if stats.missing > 0:
            values_data.append(("Missing", stats.missing, miss_pct, 3))

        # Sort by count (descending) for ranking
        values_data.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for i, (value, count, pct, _original_rank) in enumerate(values_data):
            # Format value display
            if value == "True":
                formatted_value = "True"
            elif value == "False":
                formatted_value = "False"
            else:
                formatted_value = "Missing"

            rows.append(
                f"<tr class='common-row rank-{i + 1}'>"
                f"<td class='num common-value'>{formatted_value}</td>"
                f"<td class='num common-count'>{int(count):,}</td>"
                f"<td class='num common-pct'>{pct:.1f}%</td>"
                f"<td class='progress-bar'><div class='bar-fill' style='width:{pct:.1f}%'></div></td>"
                f"</tr>"
            )

        body = "".join(rows)
        return (
            '<table class="common-values-table enhanced">'
            "<thead><tr><th>Value</th><th>Count</th><th>Frequency</th><th>Distribution</th></tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
        )

    def _build_missing_values_table(self, stats: BooleanStats, miss_pct: float) -> str:
        """Build simple missing values analysis matching reference HTML."""
        total = int(stats.true_n + stats.false_n + stats.missing)
        present = int(stats.true_n + stats.false_n)
        present_pct = (present / max(1, total)) * 100.0 if total > 0 else 0.0

        # Section 1: Data Completeness
        completeness_html = f"""
        <div class="missing-analysis-header">
            <h4 class="section-title">Data Completeness</h4>
        </div>

        <div class="completeness-container">
            <div class="completeness-stats">
                <span class="stat-item">
                    <span class="stat-label">Present:</span>
                    <span class="stat-value">{present:,} <span class="stat-pct">({present_pct:.1f}%)</span></span>
                </span>
                <span class="stat-item">
                    <span class="stat-label">Missing:</span>
                    <span class="stat-value">{stats.missing:,} <span class="stat-pct">({miss_pct:.1f}%)</span></span>
                </span>
            </div>
            <div class="completeness-bar">
                <div class="bar-fill present" style="width: {present_pct:.1f}%" title="Present: {present_pct:.1f}%"></div>
                <div class="bar-fill missing" style="width: {miss_pct:.1f}%" title="Missing: {miss_pct:.1f}%"></div>
            </div>
        </div>
        """

        # Section 2: Chunk Distribution
        chunk_html = self._build_chunk_distribution_simple(stats)

        return completeness_html + chunk_html

    def _build_chunk_distribution_simple(self, stats: BooleanStats) -> str:
        """Build simple chunk distribution visualization matching reference HTML.

        Args:
            stats: BooleanStats object

        Returns:
            HTML string for chunk distribution
        """
        # Get chunk metadata
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            return ""

        total = int(stats.true_n + stats.false_n + stats.missing)
        if total == 0:
            return ""

        # Build segments
        segments_html = ""
        max_missing_pct = 0.0
        num_chunks = len(chunk_metadata)

        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row + 1
            missing_pct = (
                (missing_count / chunk_size) * 100.0 if chunk_size > 0 else 0.0
            )
            width_pct = (chunk_size / total) * 100.0

            # Track peak
            if missing_pct > max_missing_pct:
                max_missing_pct = missing_pct

            # Determine severity class (3 levels only)
            if missing_pct <= 5:
                severity = "low"
            elif missing_pct <= 20:
                severity = "medium"
            else:
                severity = "high"

            segments_html += f"""
            <div class="chunk-segment {severity}" style="width: {width_pct:.2f}%" title="Rows {start_row:,}-{end_row:,}: {missing_count:,} missing ({missing_pct:.1f}%)"></div>
            """

        return f"""
        <div class="chunk-distribution">
            <h4 class="section-title">Missing Values Distribution</h4>
            <div class="chunk-info">
                <span>{num_chunks} chunks analyzed</span>
                <span>Peak: {max_missing_pct:.1f}%</span>
            </div>
            <div class="chunk-spectrum">
                {segments_html}
            </div>
            <div class="chunk-legend">
                <span class="legend-item"><span class="color-box low"></span>Low (0-5%)</span>
                <span class="legend-item"><span class="color-box medium"></span>Medium (5-20%)</span>
                <span class="legend-item"><span class="color-box high"></span>High (20%+)</span>
            </div>
        </div>
        """

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

            # Create tooltip content
            tooltip_content = (
                f"Rows {start_row:,}-{end_row:,}: "
                f"{missing_count:,} missing ({missing_pct:.1f}%)"
            )

            segments_html += f"""
            <div class="spectrum-segment {color_class}"
                 style="width: {segment_width_pct:.2f}%"
                 title="{tooltip_content}"
                 data-start="{start_row}"
                 data-end="{end_row}"
                 data-missing="{missing_count}"
                 data-missing-pct="{missing_pct:.1f}">
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
            severity_icon = "&#128680;"
        elif max_missing_pct >= 20:
            severity = "high"
            severity_icon = "&#9888;"
        elif max_missing_pct >= 5:
            severity = "medium"
            severity_icon = "&#9889;"
        else:
            severity = "low"
            severity_icon = "&#10004;"

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
                    {severity_icon} {severity.title()} missing data severity
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
            severity_icon = "&#128680;"
        elif missing_pct >= 20:
            severity = "high"
            severity_icon = "&#9888;"
        elif missing_pct >= 5:
            severity = "medium"
            severity_icon = "&#9889;"
        else:
            severity = "low"
            severity_icon = "&#10004;"

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
                    {severity_icon} {severity.title()} missing data severity
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
        left_table: str,
        right_table: str,
        chart_html: str,
        details_html: str,
    ) -> str:
        """Assemble the complete card HTML."""
        return f"""
        <article class="var-card" id="{col_id}">
            <header class="var-card__header">
                <div class="title">
                    <span class="colname">{safe_name}</span>
                    <span class="badge">Boolean</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {quality_flags_html}
                </div>
            </header>
            <div class="var-card__body">
                <div class="triple-row">
                    <div class="box stats-left">{left_table}</div>
                    <div class="box stats-right">{right_table}</div>
                    <div class="box chart">{chart_html}</div>
                </div>
                <div class="card-controls" role="group" aria-label="Column controls">
                    <div class="details-slot">
                        <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
                    </div>
                    <div class="controls-slot"></div>
                </div>
                {details_html}
            </div>
        </article>
        """
