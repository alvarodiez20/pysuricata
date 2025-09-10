"""Categorical card rendering functionality."""

import html as _html
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .card_base import CardRenderer, QualityAssessor, TableBuilder
from .card_config import DEFAULT_CAT_CONFIG, DEFAULT_CHART_DIMS
from .card_types import BarData, CategoricalStats, QualityFlags
from .format_utils import fmt_num as _fmt_num
from .format_utils import human_bytes as _human_bytes
from .svg_utils import svg_empty as _svg_empty


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
        top_values_table = self._build_top_values_table(
            items, int(getattr(stats, "count", 0))
        )
        norm_tab_btn, norm_tab_pane = self._build_normalization_section(items, stats)

        details_html = self._build_details_section(
            col_id, top_values_table, norm_tab_btn, norm_tab_pane
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
        mem_display = self.format_bytes(int(getattr(stats, "mem_bytes", 0)))

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
        self, items: Sequence[Tuple[str, int]]
    ) -> Tuple[List[int], int]:
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
        items: Sequence[Tuple[str, int]],
        total: int,
        topn_list: List[int],
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
        self, items: List[Tuple[str, int]], total: int, *, scale: str = "count"
    ) -> str:
        """Build categorical bar chart SVG."""
        if total <= 0 or not items:
            return self.create_empty_svg(
                "cat-svg", self.chart_dims.width, self.chart_dims.height
            )

        bar_data = self._prepare_bar_data(items, total, scale)
        return self._render_bar_svg(bar_data)

    def _prepare_bar_data(
        self, items: List[Tuple[str, int]], total: int, scale: str
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
        max_label_len = max((len(l) for l in bar_data.labels), default=0)
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
        self, items: Sequence[Tuple[str, int]], count: int, max_rows: int = 15
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
        self, items: Sequence[Tuple[str, int]], stats: CategoricalStats
    ) -> Tuple[str, str]:
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
        self, col_id: str, top_values_table: str, norm_tab_btn: str, norm_tab_pane: str
    ) -> str:
        """Build details section with tabs."""
        return f"""
        <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
                <button role="tab" class="active" data-tab="top">Top values</button>
                {norm_tab_btn}
            </nav>
            <div class="tab-panes">
                <section class="tab-pane active" data-tab="top">{top_values_table}</section>
                {norm_tab_pane}
            </div>
        </section>
        """

    def _build_controls_section(
        self, col_id: str, topn_list: List[int], default_topn: int
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
