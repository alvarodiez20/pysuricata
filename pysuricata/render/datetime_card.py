"""DateTime card rendering functionality."""

import html as _html
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .card_base import CardRenderer, QualityAssessor, TableBuilder
from .card_config import DEFAULT_CHART_DIMS, DEFAULT_DT_CONFIG
from .card_types import DateTimeStats, QualityFlags
from .format_utils import human_bytes as _human_bytes
from .svg_utils import nice_ticks as _nice_ticks
from .svg_utils import svg_empty as _svg_empty


class DateTimeCardRenderer(CardRenderer):
    """Renders datetime data cards."""

    def __init__(self):
        super().__init__()
        self.quality_assessor = QualityAssessor()
        self.table_builder = TableBuilder()
        self.dt_config = DEFAULT_DT_CONFIG
        self.chart_dims = DEFAULT_CHART_DIMS

    def render_card(self, stats: DateTimeStats) -> str:
        """Render a complete datetime card."""
        col_id = self.safe_col_id(stats.name)
        safe_name = self.safe_html_escape(stats.name)

        # Calculate percentages and quality flags
        total = int(getattr(stats, "count", 0) + getattr(stats, "missing", 0))
        miss_pct = (stats.missing / max(1, total)) * 100.0
        miss_cls = "crit" if miss_pct > 20 else ("warn" if miss_pct > 0 else "")

        quality_flags = self.quality_assessor.assess_datetime_quality(stats)
        quality_flags_html = self._build_quality_flags_html(quality_flags, miss_pct)

        # Build components
        left_table = self._build_left_table(stats, miss_cls, miss_pct)
        right_table = self._build_right_table(stats)

        # Chart
        chart_html = self._build_timeline_chart(stats)

        # Details
        details_html = self._build_details_section(col_id, stats)

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

    def _build_quality_flags_html(self, flags: QualityFlags, miss_pct: float) -> str:
        """Build quality flags HTML for datetime data."""
        flag_items = []

        if flags.missing:
            severity = "bad" if miss_pct > 20 else "warn"
            flag_items.append(f'<li class="flag {severity}">Missing</li>')

        if flags.monotonic_increasing:
            flag_items.append('<li class="flag good">Monotonic ↑</li>')

        if flags.monotonic_decreasing:
            flag_items.append('<li class="flag good">Monotonic ↓</li>')

        return (
            f'<ul class="quality-flags">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )

    def _build_left_table(
        self, stats: DateTimeStats, miss_cls: str, miss_pct: float
    ) -> str:
        """Build left statistics table."""
        mem_display = self.format_bytes(int(getattr(stats, "mem_bytes", 0))) + " (≈)"

        data = [
            ("Count", f"{int(getattr(stats, 'count', 0)):,}", "num"),
            (
                "Missing",
                f"{int(getattr(stats, 'missing', 0)):,} ({miss_pct:.1f}%)",
                f"num {miss_cls}",
            ),
            ("Min", self._format_timestamp(getattr(stats, "min_ts", None)), None),
            ("Max", self._format_timestamp(getattr(stats, "max_ts", None)), None),
            ("Processed bytes", mem_display, "num"),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_right_table(self, stats: DateTimeStats) -> str:
        """Build right statistics table with sparklines."""
        data = [
            (
                "Hour",
                self._build_sparkline(getattr(stats, "by_hour", []) or []),
                "small",
            ),
            (
                "Day of week",
                self._build_sparkline(getattr(stats, "by_dow", []) or []),
                "small",
            ),
            (
                "Month",
                self._build_sparkline(getattr(stats, "by_month", []) or []),
                "small",
            ),
        ]

        return self.table_builder.build_key_value_table(data)

    def _format_timestamp(self, ts: Optional[int]) -> str:
        """Format a UTC nanoseconds epoch as ISO8601 Z; fallback safely."""
        if ts is None:
            return "—"
        try:
            # Prefer pandas if available for robustness
            if pd is not None:  # type: ignore
                dt = pd.to_datetime(int(ts), utc=True)
                return dt.isoformat()
        except Exception:
            pass
        try:
            from datetime import datetime as _dt

            return _dt.utcfromtimestamp(int(ts) / 1_000_000_000).isoformat() + "Z"
        except Exception:
            return str(ts)

    def _build_sparkline(self, counts: List[int]) -> str:
        """Return an 8-level unicode block sparkline for small arrays."""
        if not counts:
            return ""
        m = max(counts) or 1
        blocks = "▁▂▃▄▅▆▇█"
        return "".join(
            blocks[min(len(blocks) - 1, int(c * (len(blocks) - 1) / m))] for c in counts
        )

    def _build_timeline_chart(self, stats: DateTimeStats) -> str:
        """Build timeline chart."""
        sample = getattr(stats, "sample_ts", None)
        tmin = getattr(stats, "min_ts", None)
        tmax = getattr(stats, "max_ts", None)
        scale_count = getattr(stats, "sample_scale", 1.0)

        svg = self._build_timeline_svg(
            sample,
            tmin,
            tmax,
            bins=self.dt_config.default_bins,
            scale_count=scale_count,
        )

        return f"""
        <div class="timeline-chart">
            {svg}
        </div>
        """

    def _build_timeline_svg(
        self,
        sample: Optional[List[int]],
        tmin: Optional[int],
        tmax: Optional[int],
        *,
        bins: int = 60,
        scale_count: float = 1.0,
    ) -> str:
        """Build timeline SVG from raw ns samples."""
        if not sample or tmin is None or tmax is None:
            return self.create_empty_svg(
                "dt-svg", self.chart_dims.width, self.chart_dims.height
            )

        try:
            a = np.asarray(sample, dtype=np.int64)
            if a.size == 0:
                return self.create_empty_svg(
                    "dt-svg", self.chart_dims.width, self.chart_dims.height
                )

            if tmin == tmax:
                tmax = tmin + 1

            counts, edges = np.histogram(
                a, bins=int(max(10, min(bins, 180))), range=(int(tmin), int(tmax))
            )
            counts = np.maximum(
                0, np.round(counts * max(1.0, float(scale_count)))
            ).astype(int)
            y_max = int(max(1, counts.max()))

            width, height = self.chart_dims.width, self.chart_dims.height
            margin_left, margin_right, margin_top, margin_bottom = 45, 8, 8, 32
            iw = width - margin_left - margin_right
            ih = height - margin_top - margin_bottom

            def sx(x):
                return margin_left + (x - tmin) / (tmax - tmin) * iw

            def sy(y):
                return margin_top + (1 - y / y_max) * ih

            centers = (edges[:-1] + edges[1:]) / 2.0
            pts = " ".join(
                f"{sx(x):.2f},{sy(float(c)):.2f}" for x, c in zip(centers, counts)
            )
            y_ticks, _ = _nice_ticks(0, y_max, 5)

            n_xt = 5
            xt_vals = np.linspace(tmin, tmax, n_xt)
            span_ns = tmax - tmin

            def _format_xtick(v):
                try:
                    if pd is not None:  # type: ignore
                        ts = pd.to_datetime(int(v), utc=True)
                        if span_ns <= self.dt_config.short_span_ns:
                            return ts.strftime("%Y-%m-%d %H:%M")
                        return ts.date().isoformat()
                except Exception:
                    pass
                try:
                    from datetime import datetime as _dt

                    return _dt.utcfromtimestamp(int(v) / 1_000_000_000).strftime(
                        "%Y-%m-%d"
                    )
                except Exception:
                    return str(v)

            parts = [
                f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Timeline">',
                '<g class="plot-area">',
            ]

            # Grid lines
            for yt in y_ticks:
                parts.append(
                    f'<line class="grid" x1="{margin_left}" y1="{sy(yt):.2f}" x2="{margin_left + iw}" y2="{sy(yt):.2f}"></line>'
                )

            # Main line
            parts.append(f'<polyline class="line" points="{pts}"></polyline>')

            # Hotspots for tooltips
            parts.append('<g class="hotspots">')
            for i, c in enumerate(counts):
                if not np.isfinite(c):
                    continue
                x0p = sx(edges[i])
                x1p = sx(edges[i + 1])
                wp = max(1.0, x1p - x0p)
                cp = (edges[i] + edges[i + 1]) / 2.0
                label = _format_xtick(cp)
                title = f"{int(c)} rows&#10;{label}"
                parts.append(
                    f'<rect class="hot" x="{x0p:.2f}" y="{margin_top}" width="{wp:.2f}" height="{ih:.2f}" fill="transparent" pointer-events="all">'
                    f"<title>{title}</title>"
                    f"</rect>"
                )
            parts.append("</g>")
            parts.append("</g>")

            # Axes
            x_axis_y = margin_top + ih
            parts.append(
                f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left + iw}" y2="{x_axis_y}"></line>'
            )
            parts.append(
                f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>'
            )

            # Y ticks
            for yt in y_ticks:
                py = sy(yt)
                parts.append(
                    f'<line class="tick" x1="{margin_left - 4}" y1="{py:.2f}" x2="{margin_left}" y2="{py:.2f}"></line>'
                )
                lab = int(round(yt))
                parts.append(
                    f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3:.2f}" text-anchor="end">{lab}</text>'
                )

            # X ticks
            for xv in xt_vals:
                px = sx(xv)
                parts.append(
                    f'<line class="tick" x1="{px:.2f}" y1="{x_axis_y}" x2="{px:.2f}" y2="{x_axis_y + 4}"></line>'
                )
                parts.append(
                    f'<text class="tick-label" x="{px:.2f}" y="{x_axis_y + 14:.2f}" text-anchor="middle">{_format_xtick(xv)}</text>'
                )

            # Axis titles
            parts.append(
                f'<text class="axis-title x" x="{margin_left + iw / 2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">Time</text>'
            )
            parts.append(
                f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih / 2:.2f}) rotate(-90)" text-anchor="middle">Count</text>'
            )

            parts.append("</svg>")
            return "".join(parts)
        except Exception:
            return self.create_empty_svg(
                "dt-svg", self.chart_dims.width, self.chart_dims.height
            )

    def _build_details_section(self, col_id: str, stats: DateTimeStats) -> str:
        """Build details section with breakdown tables."""
        hours = getattr(stats, "by_hour", []) or []
        dows = getattr(stats, "by_dow", []) or []
        months = getattr(stats, "by_month", []) or []

        # Build tables
        hours_table = self._build_hours_table(hours)
        dows_table = self._build_dows_table(dows)
        months_table = self._build_months_table(months)

        return f"""
        <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
                <button role="tab" class="active" data-tab="breakdown">Breakdown</button>
            </nav>
            <div class="tab-panes">
                <section class="tab-pane active" data-tab="breakdown">
                    <div class="grid-2col">
                        {hours_table}
                        {dows_table}
                    </div>
                    <div class="grid-2col" style="margin-top:8px;">
                        {months_table}
                    </div>
                </section>
            </div>
        </section>
        """

    def _build_hours_table(self, hours: List[int]) -> str:
        """Build hours breakdown table."""
        rows = "".join(
            f'<tr><th>{h:02d}</th><td class="num">{int(c):,}</td></tr>'
            for h, c in enumerate(hours)
        )
        return f'<table class="kv"><thead><tr><th>Hour</th><th>Count</th></tr></thead><tbody>{rows or "<tr><td colspan=2>—</td></tr>"}</tbody></table>'

    def _build_dows_table(self, dows: List[int]) -> str:
        """Build day of week breakdown table."""
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        rows = "".join(
            f'<tr><th>{dow_labels[i]}</th><td class="num">{int(c):,}</td></tr>'
            for i, c in enumerate(dows[:7])
        )
        return f'<table class="kv"><thead><tr><th>Day</th><th>Count</th></tr></thead><tbody>{rows or "<tr><td colspan=2>—</td></tr>"}</tbody></table>'

    def _build_months_table(self, months: List[int]) -> str:
        """Build months breakdown table."""
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        rows = "".join(
            f'<tr><th>{month_labels[i]}</th><td class="num">{int(c):,}</td></tr>'
            for i, c in enumerate(months[:12])
        )
        return f'<table class="kv"><thead><tr><th>Month</th><th>Count</th></tr></thead><tbody>{rows or "<tr><td colspan=2>—</td></tr>"}</tbody></table>'

    def _assemble_card(
        self,
        col_id: str,
        safe_name: str,
        stats: DateTimeStats,
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
                    <span class="badge">Datetime</span>
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
