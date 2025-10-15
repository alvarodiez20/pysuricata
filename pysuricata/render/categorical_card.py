"""Categorical card rendering functionality."""

import math
from collections.abc import Sequence

from .card_base import CardRenderer, QualityAssessor, TableBuilder
from .card_config import DEFAULT_CAT_CONFIG, DEFAULT_CHART_DIMS
from .card_types import BarData, CategoricalStats, QualityFlags


class CategoricalCardRenderer(CardRenderer):
    """Renders categorical data cards."""

    def __init__(self):
        super().__init__()
        self.quality_assessor = QualityAssessor()
        self.table_builder = TableBuilder()
        self.cat_config = DEFAULT_CAT_CONFIG
        self.chart_dims = DEFAULT_CHART_DIMS

    def render_card(self, stats: CategoricalStats) -> str:
        """Render a complete categorical card."""
        col_id = self.safe_col_id(stats.name)
        safe_name = self.safe_html_escape(stats.name)

        # Calculate percentages and quality flags
        total = int(getattr(stats, "count", 0) + getattr(stats, "missing", 0))
        miss_pct = (stats.missing / max(1, total)) * 100.0
        miss_cls = "crit" if miss_pct > 20 else ("warn" if miss_pct > 0 else "")

        quality_flags = self.quality_assessor.assess_categorical_quality(stats)
        quality_flags_html = self._build_quality_flags_html(quality_flags, miss_pct)

        # Compute derived stats
        cat_stats = self._compute_categorical_stats(stats)

        # Build components
        approx_badge = self._build_approx_badge(stats.approx)
        left_table = self._build_left_table(stats, miss_cls, miss_pct, cat_stats)
        right_table = self._build_right_table(cat_stats)

        # Chart and details
        items = stats.top_items or []
        topn_list, default_topn = self._get_topn_candidates(items)

        chart_html = self._build_categorical_variants(
            col_id, items, total, topn_list, default_topn
        )
        common_table = self._build_common_values_table(stats)
        norm_tab_btn, norm_tab_pane = self._build_normalization_section(items, stats)
        missing_table = self._build_missing_values_table(stats, miss_pct)

        details_html = self._build_details_section(
            col_id, common_table, norm_tab_btn, norm_tab_pane, missing_table
        )
        controls_html = self._build_controls_section(col_id, topn_list, default_topn)

        return self._assemble_card(
            col_id,
            safe_name,
            stats,
            approx_badge,
            quality_flags_html,
            left_table,
            right_table,
            chart_html,
            details_html,
            controls_html,
        )

    def _compute_categorical_stats(self, stats: CategoricalStats) -> dict:
        """Compute derived stats for categorical data."""
        total = int(getattr(stats, "count", 0))
        items = list(getattr(stats, "top_items", []) or [])
        mode_label, mode_n = items[0] if items else ("—", 0)
        safe_mode_label = self.safe_html_escape(str(mode_label))
        mode_pct = (mode_n / max(1, total)) * 100.0 if total else 0.0

        # Entropy calculation
        if total > 0 and items:
            probs = [c / total for _, c in items]
            entropy = float(-sum(p * math.log2(max(p, 1e-12)) for p in probs))
        else:
            entropy = float("nan")

        # Rare levels analysis
        rare_count = 0
        rare_cov = 0.0
        if total > 0 and items:
            for _, c in items:
                pct = c / total * 100.0
                if pct < 1.0:
                    rare_count += 1
                    rare_cov += pct

        rare_cls = "crit" if rare_cov > 60 else ("warn" if rare_cov >= 30 else "")

        # Top-5 coverage
        top5_cov = 0.0
        if total > 0 and items:
            top5_cov = sum(c for _, c in items[:5]) / total * 100.0

        top5_cls = "good" if top5_cov >= 80 else ("warn" if top5_cov <= 40 else "")

        # Empty strings
        empty_zero = int(getattr(stats, "empty_zero", 0))
        empty_cls = "warn" if empty_zero > 0 else ""

        return {
            "mode_label": mode_label,
            "safe_mode_label": safe_mode_label,
            "mode_n": int(mode_n),
            "mode_pct": float(mode_pct),
            "entropy": float(entropy),
            "rare_count": int(rare_count),
            "rare_cov": float(rare_cov),
            "rare_cls": rare_cls,
            "top5_cov": float(top5_cov),
            "top5_cls": top5_cls,
            "empty_zero": empty_zero,
            "empty_cls": empty_cls,
            "unique_est": int(getattr(stats, "unique_est", 0)),
        }

    def _build_quality_flags_html(self, flags: QualityFlags, miss_pct: float) -> str:
        """Build quality flags HTML for categorical data."""
        flag_items = []

        if flags.high_cardinality:
            flag_items.append('<li class="flag warn">High cardinality</li>')

        if flags.dominant_category:
            flag_items.append('<li class="flag warn">Dominant category</li>')

        if flags.many_rare_levels:
            flag_items.append('<li class="flag warn">Many rare levels</li>')

        if flags.case_variants:
            flag_items.append('<li class="flag">Case variants</li>')

        if flags.trim_variants:
            flag_items.append('<li class="flag">Trim variants</li>')

        if flags.empty_strings:
            flag_items.append('<li class="flag">Empty strings</li>')

        if flags.missing:
            severity = "bad" if miss_pct > 20 else "warn"
            flag_items.append(f'<li class="flag {severity}">Missing</li>')

        return (
            f'<ul class="quality-flags">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )

    def _build_approx_badge(self, approx: bool) -> str:
        """Build approximation badge if needed."""
        return '<span class="badge">approx</span>' if approx else ""

    def _build_left_table(
        self, stats: CategoricalStats, miss_cls: str, miss_pct: float, cat_stats: dict
    ) -> str:
        """Build left statistics table."""
        self.format_bytes(int(getattr(stats, "mem_bytes", 0)))

        data = [
            ("Count", f"{int(getattr(stats, 'count', 0)):,}", "num"),
            (
                "Unique",
                f"{int(getattr(stats, 'unique_est', 0)):,}{' (≈)' if getattr(stats, 'approx', False) else ''}",
                "num",
            ),
            (
                "Missing",
                f"{int(getattr(stats, 'missing', 0)):,} ({miss_pct:.1f}%)",
                f"num {miss_cls}",
            ),
            ("Mode", f"<code>{cat_stats['safe_mode_label']}</code>", None),
            ("Mode %", f"{cat_stats['mode_pct']:.1f}%", "num"),
            (
                "Empty strings",
                f"{int(cat_stats['empty_zero']):,}",
                f"num {cat_stats['empty_cls']}",
            ),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_right_table(self, cat_stats: dict) -> str:
        """Build right statistics table."""
        data = [
            ("Entropy", self.format_number(cat_stats["entropy"]), "num"),
            (
                "Rare levels",
                f"{int(cat_stats['rare_count']):,} ({cat_stats['rare_cov']:.1f}%)",
                f"num {cat_stats['rare_cls']}",
            ),
            (
                "Top 5 coverage",
                f"{cat_stats['top5_cov']:.1f}%",
                f"num {cat_stats['top5_cls']}",
            ),
            (
                "Label length (avg)",
                self.format_number(cat_stats.get("avg_len", float("nan"))),
                "num",
            ),
            ("Length p90", str(cat_stats.get("len_p90", "—")), None),
            (
                "Processed bytes",
                f"{self.format_bytes(int(cat_stats.get('mem_bytes', 0)))} (≈)",
                "num",
            ),
        ]

        return self.table_builder.build_key_value_table(data)

    def _get_topn_candidates(
        self, items: Sequence[tuple[str, int]]
    ) -> tuple[list[int], int]:
        """Get Top-N candidates for categorical display."""
        max_n = max(1, min(15, len(items)))
        candidates = [5, 10, 15, max_n]
        topn_list = sorted({n for n in candidates if 1 <= n <= max_n})
        default_topn = (
            10 if 10 in topn_list else (max(topn_list) if topn_list else max_n)
        )
        return topn_list, default_topn

    def _build_categorical_variants(
        self,
        col_id: str,
        items: Sequence[tuple[str, int]],
        total: int,
        topn_list: list[int],
        default_topn: int,
    ) -> str:
        """Build categorical chart variants."""
        parts = []
        for n in topn_list:
            if len(items) > n:
                keep = max(1, n - 1)
                head = list(items[:keep])
                other = int(sum(c for _, c in items[keep:]))
                data = head + [("Other", other)]
            else:
                data = list(items[:n])

            svg = self._build_categorical_bar_svg(data, total=max(1, int(total)))
            active_class = " active" if n == default_topn else ""
            parts.append(
                f'<div class="cat variant{active_class}" id="{col_id}-cat-top-{n}" data-topn="{n}">{svg}</div>'
            )

        return f"""
        <div class="topn-chart">
            <div class="hist-variants">{"".join(parts)}</div>
        </div>
        """

    def _build_categorical_bar_svg(
        self, items: list[tuple[str, int]], total: int, *, scale: str = "count"
    ) -> str:
        """Build categorical bar chart SVG."""
        if total <= 0 or not items:
            return self.create_empty_svg(
                "cat-svg", self.chart_dims.width, self.chart_dims.height
            )

        bar_data = self._prepare_bar_data(items, total, scale)
        return self._render_bar_svg(bar_data)

    def _prepare_bar_data(
        self, items: list[tuple[str, int]], total: int, scale: str
    ) -> BarData:
        """Prepare bar chart data."""
        labels = [self.safe_html_escape(str(k)) for k, _ in items]
        counts = [int(c) for _, c in items]
        pcts = [(c / total * 100.0) for c in counts]

        if scale == "pct":
            values = pcts
        else:
            values = counts

        return BarData(labels=labels, counts=counts, percentages=pcts, values=values)

    def _render_bar_svg(self, bar_data: BarData) -> str:
        """Render bar chart SVG."""
        width, height = self.chart_dims.width, self.chart_dims.height
        margin_top, margin_bottom = 8, 8
        margin_right = 12

        # Calculate label width
        max_label_len = max((len(label) for label in bar_data.labels), default=0)
        char_w = self.cat_config.char_width
        gutter = max(
            self.cat_config.min_gutter,
            min(self.cat_config.max_gutter, char_w * min(max_label_len, 28) + 16),
        )
        margin_left = max(120, gutter)

        n = len(bar_data.labels)
        iw = width - margin_left - margin_right
        ih = height - margin_top - margin_bottom

        if n <= 0 or iw <= 0 or ih <= 0:
            return self.create_empty_svg("cat-svg", width, height)

        bar_gap = 6
        bar_h = max(4, (ih - bar_gap * (n - 1)) / max(n, 1))

        vmax = max(bar_data.values) or 1.0

        def sx(v: float) -> float:
            return margin_left + (v / vmax) * iw

        parts = [
            f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Top categories">'
        ]

        for i, (label, c, p, val) in enumerate(
            zip(bar_data.labels, bar_data.counts, bar_data.percentages, bar_data.values)
        ):
            y = margin_top + i * (bar_h + bar_gap)
            x0 = margin_left
            x1 = sx(float(val))
            w = max(1.0, x1 - x0)
            short = (
                (label[: self.cat_config.max_label_length] + "…")
                if len(label) > self.cat_config.max_label_length
                else label
            )

            parts.append(
                f'<g class="bar-row">'
                f'<rect class="bar" x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{bar_h:.2f}" rx="2" ry="2">'
                f"<title>{label}\n{c:,} rows ({p:.1f}%)</title>"
                f"</rect>"
                f'<text class="bar-label" x="{margin_left - 6}" y="{y + bar_h / 2 + 3:.2f}" text-anchor="end">{short}</text>'
                f'<text class="bar-value" x="{(x1 - 6 if w >= 56 else x1 + 4):.2f}" y="{y + bar_h / 2 + 3:.2f}" text-anchor="{("end" if w >= 56 else "start")}">{c:,} ({p:.1f}%)</text>'
                f"</g>"
            )

        parts.append("</svg>")
        return "".join(parts)

    def _build_top_values_table(
        self, items: Sequence[tuple[str, int]], count: int, max_rows: int = 15
    ) -> str:
        """Build top values table."""
        rows = []
        total_nonnull = max(1, int(count))
        acc = 0

        for val, c in list(items)[: max_rows - 1]:
            acc += int(c)
            rows.append(
                f"<tr><td><code>{self.safe_html_escape(str(val))}</code></td>"
                f"<td class='num'>{int(c):,}</td>"
                f"<td class='num'>{(int(c) / total_nonnull * 100.0):.1f}%</td></tr>"
            )

        other_n = max(0, total_nonnull - acc)
        if len(items) > (max_rows - 1) or other_n > 0:
            rows.append(
                f"<tr><td><code>Other</code></td>"
                f"<td class='num'>{other_n:,}</td>"
                f"<td class='num'>{(other_n / total_nonnull * 100.0):.1f}%</td></tr>"
            )

        body = "".join(rows) if rows else "<tr><td colspan=3>—</td></tr>"
        return (
            '<table class="kv"><thead><tr><th>Value</th><th>Count</th><th>%</th></tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )

    def _build_normalization_section(
        self, items: Sequence[tuple[str, int]], stats: CategoricalStats
    ) -> tuple[str, str]:
        """Build normalization section if needed."""
        try:
            need_norm = (getattr(stats, "case_variants_est", 0) > 0) or (
                getattr(stats, "trim_variants_est", 0) > 0
            )
            if not (need_norm and items):
                return "", ""

            examples = []
            for val, _ in list(items)[:10]:
                raw = str(val)
                low = raw.lower()
                stp = raw.strip()
                if raw != low or raw != stp:
                    examples.append((raw, low, stp))
                if len(examples) >= 6:
                    break

            rows = (
                "".join(
                    f"<tr><td><code>{self.safe_html_escape(a)}</code></td>"
                    f"<td><code>{self.safe_html_escape(b)}</code></td>"
                    f"<td><code>{self.safe_html_escape(c)}</code></td></tr>"
                    for a, b, c in examples
                )
                or "<tr><td colspan=3>—</td></tr>"
            )

            norm_tbl = (
                '<table class="kv"><thead><tr><th>Original</th><th>lower()</th><th>strip()</th></tr></thead>'
                f"<tbody>{rows}</tbody></table>"
            )

            norm_tab_btn = (
                '<button role="tab" data-tab="normalize">Normalization</button>'
            )
            norm_tab_pane = (
                f'<section class="tab-pane" data-tab="normalize">{norm_tbl}</section>'
            )
            return norm_tab_btn, norm_tab_pane
        except Exception:
            return "", ""

    def _build_details_section(
        self,
        col_id: str,
        common_table: str,
        norm_tab_btn: str,
        norm_tab_pane: str,
        missing_table: str,
    ) -> str:
        """Build details section with tabs."""
        return f"""
        <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
                <button role="tab" class="active" data-tab="common">Common values</button>
                {norm_tab_btn}
                <button role="tab" data-tab="missing">Missing Values</button>
            </nav>
            <div class="tab-panes">
                <section class="tab-pane active" data-tab="common">{common_table}</section>
                {norm_tab_pane}
                <section class="tab-pane" data-tab="missing">
                    <div class="sub"><div class="hdr">Missing Values</div>{missing_table}</div>
                </section>
            </div>
        </section>
        """

    def _build_controls_section(
        self, col_id: str, topn_list: list[int], default_topn: int
    ) -> str:
        """Build controls section."""
        topn_buttons = " ".join(
            f'<button type="button" class="btn-soft{" active" if n == default_topn else ""}" data-topn="{n}">{n}</button>'
            for n in topn_list
        )

        return f"""
        <div class="card-controls" role="group" aria-label="Column controls">
            <div class="details-slot">
                <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
            </div>
            <div class="controls-slot">
                <div class="hist-controls" data-topn="{default_topn}">
                    <div class="center-controls">
                        <span>Top‑N:</span>
                        <div class="bin-group">{topn_buttons}</div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _assemble_card(
        self,
        col_id: str,
        safe_name: str,
        stats: CategoricalStats,
        approx_badge: str,
        quality_flags_html: str,
        left_table: str,
        right_table: str,
        chart_html: str,
        details_html: str,
        controls_html: str,
    ) -> str:
        """Assemble the complete card HTML."""
        return f"""
        <article class="var-card" id="{col_id}">
            <header class="var-card__header">
                <div class="title">
                    <span class="colname">{safe_name}</span>
                    <span class="badge">Categorical</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {approx_badge}
                    {quality_flags_html}
                </div>
            </header>
            <div class="var-card__body">
                <div class="triple-row">
                    <div class="box stats-left">{left_table}</div>
                    <div class="box stats-right">{right_table}</div>
                    <div class="box chart">{chart_html}</div>
                </div>
                {controls_html}
                {details_html}
            </div>
        </article>
        """

    def _build_common_values_table(self, stats: CategoricalStats) -> str:
        """Build common values table with enhanced formatting and functionality.

        This method creates a professional, feature-rich table that provides
        comprehensive insights into the most frequent categorical values in the dataset.

        Args:
            stats: CategoricalStats object containing the data

        Returns:
            HTML string for the enhanced common values table
        """
        try:
            top_items = list(getattr(stats, "top_items", []) or [])
        except Exception:
            top_items = []

        if not top_items:
            return '<div class="muted">No common values to display</div>'

        rows = []
        total_nonnull = max(1, int(getattr(stats, "count", 0)))

        # Take only top 10 values for better display and performance
        top_items = top_items[:10]

        for i, (value, count) in enumerate(top_items):
            pct = (int(count) / total_nonnull) * 100.0 if total_nonnull else 0.0

            # Add ranking indicator for top values
            rank_icon = self._ordinal_number(i + 1)

            # Format categorical value with proper escaping
            formatted_value = self.safe_html_escape(str(value))

            rows.append(
                f"<tr class='common-row rank-{i + 1}'>"
                f"<td class='rank'>{rank_icon}</td>"
                f"<td class='cat common-value'>{formatted_value}</td>"
                f"<td class='num common-count'>{int(count):,}</td>"
                f"<td class='num common-pct'>{pct:.1f}%</td>"
                f"<td class='progress-bar'><div class='bar-fill' style='width:{pct:.1f}%'></div></td>"
                f"</tr>"
            )

        body = "".join(rows)
        return (
            '<table class="common-values-table enhanced">'
            "<thead><tr><th>Rank</th><th>Value</th><th>Count</th><th>Frequency</th><th>Distribution</th></tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
        )

    def _ordinal_number(self, n: int) -> str:
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

    def _build_missing_values_table(
        self, stats: CategoricalStats, miss_pct: float
    ) -> str:
        """Build simple missing values analysis matching reference HTML.

        Args:
            stats: CategoricalStats object containing missing data information
            miss_pct: Pre-calculated missing percentage

        Returns:
            HTML string for the missing values analysis
        """
        # Calculate missing data statistics
        total_values = stats.count + stats.missing
        present_pct = (
            (stats.count / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )

        # Section 1: Data Completeness
        completeness_html = f"""
        <div class="missing-analysis-header">
            <h4 class="section-title">Data Completeness</h4>
        </div>

        <div class="completeness-container">
            <div class="completeness-stats">
                <span class="stat-item">
                    <span class="stat-label">Present:</span>
                    <span class="stat-value">{stats.count:,} <span class="stat-pct">({present_pct:.1f}%)</span></span>
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

    def _build_chunk_distribution_simple(self, stats: CategoricalStats) -> str:
        """Build simple chunk distribution visualization matching reference HTML.

        Args:
            stats: CategoricalStats object

        Returns:
            HTML string for chunk distribution
        """
        # Get chunk metadata
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            return ""

        total_values = stats.count + stats.missing
        if total_values == 0:
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
            width_pct = (chunk_size / total_values) * 100.0

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
            <div class="chunk-segment {severity}"
                 style="width: {width_pct:.2f}%"
                 data-start="{start_row}"
                 data-end="{end_row}"
                 data-missing="{missing_count}"
                 data-total="{chunk_size}"
                 data-pct="{missing_pct:.1f}"></div>
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

    def _build_dataprep_spectrum_visualization(self, stats: CategoricalStats) -> str:
        """Legacy method - no longer used."""
        return ""

    def _get_missing_data_severity(self, missing_pct: float) -> tuple[str, str, str]:
        """Get missing data severity classification with clear thresholds.

        Args:
            missing_pct: Percentage of missing data

        Returns:
            Tuple of (severity_class, label, icon)
        """
        if missing_pct >= 50:
            return "critical", "Critical", "🚨"
        elif missing_pct >= 20:
            return "high", "High", "⚠️"
        elif missing_pct >= 5:
            return "medium", "Medium", "⚡"
        else:
            return "low", "Low", "✅"

    def _build_dataprep_spectrum_visualization(self, stats: CategoricalStats) -> str:
        """Build DataPrep-style spectrum visualization for missing values per chunk.

        This creates a single horizontal bar with segments representing actual processing
        chunks, colored by missing value density (green-yellow-red gradient).

        Args:
            stats: CategoricalStats object containing chunk metadata and missing data information

        Returns:
            HTML string for the DataPrep-style spectrum visualization
        """
        # Check if we have chunk metadata
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            return ""

        total_values = stats.count + stats.missing
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
            severity_icon = "🚨"
        elif max_missing_pct >= 20:
            severity = "high"
            severity_icon = "⚠️"
        elif max_missing_pct >= 5:
            severity = "medium"
            severity_icon = "⚡"
        else:
            severity = "low"
            severity_icon = "✅"

        return f"""
        <div class="dataprep-spectrum">
            <div class="spectrum-header">
                <span class="spectrum-title">Missing Values Distribution</span>
                <span class="spectrum-stats">
                    {total_chunks} chunks • {max_missing_pct:.1f}% max • {avg_missing_pct:.1f}% avg
                </span>
                <span class="spectrum-severity {severity}">
                    {severity_icon} {severity.title()} Missing Data
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
        </div>
        """
