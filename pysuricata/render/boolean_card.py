"""Boolean card rendering functionality."""

import html as _html
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .card_base import CardRenderer, QualityAssessor, TableBuilder
from .card_config import DEFAULT_BOOL_CONFIG
from .card_types import BooleanStats, QualityFlags
from .format_utils import human_bytes as _human_bytes
from .svg_utils import svg_empty as _svg_empty


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

        # Chart
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
        mem_display = self.format_bytes(getattr(stats, "mem_bytes", 0)) + " (≈)"
        unique_vals = int(int(stats.true_n > 0) + int(stats.false_n > 0))

        data = [
            ("Count", f"{cnt:,}", "num"),
            ("Missing", f"{int(stats.missing):,} ({miss_pct:.1f}%)", f"num {miss_cls}"),
            ("Unique", f"{unique_vals}", "num"),
            ("Processed bytes", mem_display, "num"),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_right_table(
        self, stats: BooleanStats, true_pct_total: float, false_pct_total: float
    ) -> str:
        """Build right statistics table."""
        data = [
            ("True", f"{int(stats.true_n):,} ({true_pct_total:.1f}%)", "num"),
            ("False", f"{int(stats.false_n):,} ({false_pct_total:.1f}%)", "num"),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_boolean_chart(self, stats: BooleanStats) -> str:
        """Build boolean chart."""
        svg = self._build_boolean_stack_svg(
            int(stats.true_n), int(stats.false_n), int(stats.missing)
        )

        return f"""
        <div class="boolean-chart">
            {svg}
        </div>
        """

    def _build_boolean_stack_svg(
        self,
        true_n: int,
        false_n: int,
        miss: int,
        *,
        width: int = 420,
        height: int = 48,
    ) -> str:
        """Build boolean stack SVG."""
        total = max(1, int(true_n + false_n + miss))
        margin = self.bool_config.margin
        inner_w = width - 2 * margin
        seg_h = height - 2 * margin

        w_false = int(inner_w * (false_n / total))
        w_true = int(inner_w * (true_n / total))
        w_miss = max(0, inner_w - w_false - w_true)

        parts = [
            f'<svg class="bool-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ]

        x = margin

        # False segment
        parts.append(
            f'<rect class="seg false" x="{x}" y="{margin}" width="{w_false}" height="{seg_h}"><title>False: {false_n:,}</title></rect>'
        )
        if w_false >= self.bool_config.min_segment_width:
            cx = x + w_false / 2
            pct = (false_n / total) * 100.0
            parts.append(
                f'<text class="label" x="{cx:.1f}" y="{margin + seg_h / 2:.1f}">False {pct:.1f}%</text>'
            )

        x += w_false

        # True segment
        parts.append(
            f'<rect class="seg true" x="{x}" y="{margin}" width="{w_true}" height="{seg_h}"><title>True: {true_n:,}</title></rect>'
        )
        if w_true >= self.bool_config.min_segment_width:
            cx = x + w_true / 2
            pct = (true_n / total) * 100.0
            parts.append(
                f'<text class="label" x="{cx:.1f}" y="{margin + seg_h / 2:.1f}">True {pct:.1f}%</text>'
            )

        x += w_true

        # Missing segment
        if w_miss:
            parts.append(
                f'<rect class="seg missing" x="{x}" y="{margin}" width="{w_miss}" height="{seg_h}"><title>Missing: {miss:,}</title></rect>'
            )
            if w_miss >= self.bool_config.min_segment_width:
                cx = x + w_miss / 2
                pct = (miss / total) * 100.0
                parts.append(
                    f'<text class="label" x="{cx:.1f}" y="{margin + seg_h / 2:.1f}">Missing {pct:.1f}%</text>'
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
        """Build details section with breakdown table."""
        breakdown_rows = "".join(
            [
                f"<tr><th>True</th><td class='num'>{int(stats.true_n):,}</td><td class='num'>{true_pct_total:.1f}%</td></tr>",
                f"<tr><th>False</th><td class='num'>{int(stats.false_n):,}</td><td class='num'>{false_pct_total:.1f}%</td></tr>",
                f"<tr><th>Missing</th><td class='num'>{int(stats.missing):,}</td><td class='num'>{miss_pct:.1f}%</td></tr>",
            ]
        )

        return f"""
        <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
                <button role="tab" class="active" data-tab="breakdown">Breakdown</button>
            </nav>
            <div class="tab-panes">
                <section class="tab-pane active" data-tab="breakdown">
                    <table class="kv"><thead><tr><th>Value</th><th>Count</th><th>%</th></tr></thead><tbody>{breakdown_rows}</tbody></table>
                </section>
            </div>
        </section>
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
