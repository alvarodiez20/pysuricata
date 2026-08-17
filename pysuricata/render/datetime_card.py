"""DateTime card rendering functionality."""

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .card_base import CardRenderer
from .card_config import DEFAULT_CHART_DIMS, DEFAULT_DT_CONFIG
from .card_types import DateTimeStats, QualityFlags
from .svg_utils import nice_ticks as _nice_ticks
from .temporal_charts import TemporalChartRenderer
from .triage import annotate_flags


class DateTimeCardRenderer(CardRenderer):
    """Renders datetime data cards."""

    def __init__(self):
        super().__init__()
        self.dt_config = DEFAULT_DT_CONFIG
        self.chart_dims = DEFAULT_CHART_DIMS
        self.temporal_renderer = TemporalChartRenderer()

    def _get_chart_dimensions(self) -> tuple[int, int]:
        """Get consistent chart dimensions for datetime timeline.

        Taken from :class:`DateTimeConfig` rather than the shared
        ``ChartDimensions``: the shared 420 units are a chart authored much
        narrower than the column it is drawn into, which scaled the timeline up
        by 2.73 (#217).
        """
        return self.dt_config.chart_width, self.dt_config.chart_height

    def _timeline_margins(
        self, width: int, height: int, y_labels: list[str]
    ) -> tuple[int, int, int, int]:
        """Derive the timeline's margins from the text they have to hold.

        The margins used to be the constants ``45, 35, 25, 42``, sized by eye
        against a 420-unit viewBox. Constants and a viewBox are not independent
        -- a gutter is only "wide enough" relative to the scale the chart ends
        up drawn at -- so widening one without the other is what makes the y
        labels stop fitting. Deriving the gutter from the labels themselves
        removes the coupling: it is correct at 420 units and at 1,100.

        Returns ``(left, right, top, bottom)``.
        """
        cfg = self.dt_config
        widest = max((len(label) for label in y_labels), default=1)
        left = max(
            cfg.min_gutter,
            min(
                cfg.max_gutter,
                widest * cfg.char_width + cfg.tick_len + cfg.label_pad,
            ),
        )
        # The title sits on its own baseline above the plot; the date row sits
        # below the axis, and needs the tick mark plus a line of text.
        top = cfg.title_font + 16
        bottom = cfg.xlabel_font + cfg.tick_len + 23
        right = cfg.margin_right
        # A degenerate width must not produce a negative plot.
        if left + right >= width:
            left, right = cfg.min_gutter, cfg.margin_right
        return left, right, top, bottom

    def render_card(self, stats: DateTimeStats) -> str:
        """Render a complete datetime card."""
        col_id = self.safe_col_id(stats.name)
        safe_name = self.safe_html_escape(stats.name)

        # Calculate percentages and quality flags
        total = int(getattr(stats, "count", 0) + getattr(stats, "missing", 0))
        miss_pct = (stats.missing / max(1, total)) * 100.0
        miss_cls = "crit" if miss_pct > 20 else ("warn" if miss_pct > 0 else "")

        quality_flags = self.quality_assessor.assess_datetime_quality(stats)
        quality_flags_html = self._build_quality_flags_html(
            quality_flags, stats, miss_pct
        )

        # Build components
        stat_row = self._build_stat_row(
            self._left_stats(stats, miss_cls, miss_pct) + self._right_stats(stats)
        )

        # Chart
        chart_html = self._build_timeline_chart(stats)

        # Details
        details_html = self._build_details_section(col_id, stats)

        return self._assemble_card(
            col_id,
            safe_name,
            stats,
            quality_flags_html,
            stat_row,
            chart_html,
            details_html,
        )

    def _build_quality_flags_html(
        self, flags: QualityFlags, stats: DateTimeStats, miss_pct: float
    ) -> str:
        """The chips, with the number each one already knows on its face.

        `_quality_flags_markup` builds them; this puts the value on the
        chip and the threshold in a title. Splitting it this way means the
        forty-two places that emit a chip carry on emitting the same
        markup, and the annotation lives in one place rather than being
        repeated at every one of them.
        """
        return annotate_flags(self._quality_flags_markup(flags, stats, miss_pct))

    def _quality_flags_markup(
        self, flags: QualityFlags, stats: DateTimeStats, miss_pct: float
    ) -> str:
        """Build quality flags HTML for datetime data with enhanced insights."""
        flag_items = []

        if flags.missing:
            severity = "bad" if miss_pct > 20 else "warn"
            threshold = ">20%" if miss_pct > 20 else "≤20%"
            flag_items.append(
                f'<li class="flag {severity}" data-threshold="{threshold}" '
                f'data-value="{miss_pct:.1f}%">Missing</li>'
            )

        if flags.monotonic_increasing:
            flag_items.append('<li class="flag good">Monotonic ↑</li>')

        if flags.monotonic_decreasing:
            flag_items.append('<li class="flag good">Monotonic ↓</li>')

        # Weekend concentrated
        weekend_ratio = getattr(stats, "weekend_ratio", 0.0)
        if weekend_ratio > 0.35:  # >35% on weekends (expected ~28.5%)
            flag_items.append('<li class="flag">Weekend-heavy</li>')

        # Business hours concentrated
        business_hours = getattr(stats, "business_hours_ratio", 0.0)
        if business_hours > 0.5:  # >50% during business hours
            flag_items.append('<li class="flag good">Business hours</li>')

        # Seasonal pattern detected
        seasonal = getattr(stats, "seasonal_pattern", None)
        if seasonal:
            flag_items.append(f'<li class="flag">{seasonal}</li>')

        # High uniqueness (like IDs or log timestamps)
        unique_est = getattr(stats, "unique_est", 0)
        total = stats.count + stats.missing
        if total > 0 and (unique_est / total) > 0.95:
            flag_items.append('<li class="flag warn">High uniqueness</li>')

        # Irregular intervals (high std dev)
        avg_interval = getattr(stats, "avg_interval_seconds", 0.0)
        interval_std = getattr(stats, "interval_std_seconds", 0.0)
        if avg_interval > 0 and (interval_std / avg_interval) > 2.0:
            flag_items.append('<li class="flag warn">Irregular intervals</li>')

        return (
            f'<ul class="quality-flags">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )

    def _left_stats(self, stats: DateTimeStats, miss_cls: str, miss_pct: float) -> str:
        """Build left statistics table."""
        # Format time span
        time_span = getattr(stats, "time_span_days", 0.0)
        if time_span >= 365:
            span_display = f"{time_span / 365:.1f} years"
        elif time_span >= 30:
            span_display = f"{time_span / 30:.1f} months"
        else:
            span_display = f"{time_span:.1f} days"

        # Format avg interval
        avg_interval = getattr(stats, "avg_interval_seconds", 0.0)
        if avg_interval >= 86400:
            interval_display = f"{avg_interval / 86400:.1f} days"
        elif avg_interval >= 3600:
            interval_display = f"{avg_interval / 3600:.1f} hours"
        elif avg_interval >= 60:
            interval_display = f"{avg_interval / 60:.1f} minutes"
        else:
            interval_display = f"{avg_interval:.1f} seconds"

        # Format interval std in human-readable way
        interval_std = getattr(stats, "interval_std_seconds", 0.0)
        if interval_std >= 86400:
            std_display = f"{interval_std / 86400:.1f} days"
        elif interval_std >= 3600:
            std_display = f"{interval_std / 3600:.1f} hours"
        elif interval_std >= 60:
            std_display = f"{interval_std / 60:.1f} minutes"
        else:
            std_display = f"{interval_std:.1f} seconds"

        data = [
            ("Count", f"{int(getattr(stats, 'count', 0)):,}", "num"),
            (
                f"Unique{' (≈)' if getattr(stats, 'approx', True) else ''}",
                f"{int(getattr(stats, 'unique_est', 0)):,}",
                "num",
            ),
            (
                "Missing",
                f"{int(getattr(stats, 'missing', 0)):,} ({miss_pct:.1f}%)",
                f"num {miss_cls}",
            ),
            ("Timezone", "UTC", None),
            ("Time span", span_display, None),
            ("Avg interval", interval_display, None),
            ("Interval std", std_display, None),
        ]

        return data

    def _right_stats(self, stats: DateTimeStats) -> list[tuple[str, str, str | None]]:
        """Build right statistics table with temporal analysis."""
        mem_display = self.format_bytes(int(getattr(stats, "mem_bytes", 0)))

        # Seasonal pattern removed from display

        # Calculate data density (records per day)
        time_span_days = getattr(stats, "time_span_days", 0.0)
        count = getattr(stats, "count", 0)
        if time_span_days > 0:
            density = count / time_span_days
            if density >= 1:
                density_display = f"{density:.1f} records/day"
            elif density > 0:
                density_display = f"{1 / density:.1f} days/record"
            else:
                density_display = "—"
        else:
            density_display = "—"

        data = [
            (
                "Min",
                self._format_timestamp(getattr(stats, "min_ts", None)),
                "timestamp-value",
            ),
            (
                "Max",
                self._format_timestamp(getattr(stats, "max_ts", None)),
                "timestamp-value",
            ),
            ("Weekend %", f"{getattr(stats, 'weekend_ratio', 0.0) * 100:.1f}%", "num"),
            (
                "Business hrs %",
                f"{getattr(stats, 'business_hours_ratio', 0.0) * 100:.1f}%",
                "num",
            ),
            ("Data density", density_display, None),
            ("Processed bytes (≈)", mem_display, "num"),
        ]

        return data

    def _format_timestamp(self, ts: int | None, multiline: bool = True) -> str:
        """Format a UTC nanoseconds epoch as readable datetime string.

        Args:
            ts: Timestamp in nanoseconds
            multiline: If True, format with <br> (date on line 1, time on line 2).
                       If False, single line format.
        """
        if ts is None:
            return "—"
        try:
            # Prefer pandas if available for robustness
            if pd is not None:  # type: ignore
                dt = pd.to_datetime(int(ts), utc=True)
                date_part = dt.strftime("%Y-%m-%d")
                time_part = dt.strftime("%H:%M:%S UTC")
                if multiline:
                    return f"{date_part}<br>{time_part}"
                else:
                    return f"{date_part} {time_part}"
        except Exception:
            pass
        try:
            from datetime import datetime as _dt

            dt = _dt.utcfromtimestamp(int(ts) / 1_000_000_000)
            date_part = dt.strftime("%Y-%m-%d")
            time_part = dt.strftime("%H:%M:%S UTC")
            if multiline:
                return f"{date_part}<br>{time_part}"
            else:
                return f"{date_part} {time_part}"
        except Exception:
            return str(ts)

    def _build_sparkline(self, counts: list[int]) -> str:
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
            stats.name,
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
        sample: list[int] | None,
        tmin: int | None,
        tmax: int | None,
        column_name: str,
        *,
        bins: int = 60,
        scale_count: float = 1.0,
    ) -> str:
        """Build timeline SVG from raw ns samples."""
        if not sample or tmin is None or tmax is None:
            width, height = self._get_chart_dimensions()
            return self.create_empty_svg("dt-svg", width, height)

        try:
            a = np.asarray(sample, dtype=np.int64)
            if a.size == 0:
                width, height = self._get_chart_dimensions()
                return self.create_empty_svg("dt-svg", width, height)

            if tmin == tmax:
                tmax = tmin + 1

            counts, edges = np.histogram(
                a, bins=int(max(10, min(bins, 180))), range=(int(tmin), int(tmax))
            )
            counts = np.maximum(
                0, np.round(counts * max(1.0, float(scale_count)))
            ).astype(int)
            y_max = int(max(1, counts.max()))

            width, height = self._get_chart_dimensions()
            # The y ticks are needed before the margins are: the left gutter is
            # sized to the widest label that goes in it.
            y_ticks, _ = _nice_ticks(0, y_max, 5)
            y_labels = [str(int(round(t))) for t in y_ticks]
            margin_left, margin_right, margin_top, margin_bottom = (
                self._timeline_margins(width, height, y_labels)
            )
            iw = width - margin_left - margin_right
            ih = height - margin_top - margin_bottom

            def sx(x):
                return margin_left + (x - tmin) / (tmax - tmin) * iw

            def sy(y):
                return margin_top + (1 - y / y_max) * ih

            centers = (edges[:-1] + edges[1:]) / 2.0
            pts = " ".join(
                f"{sx(x):.2f},{sy(float(c)):.2f}"
                for x, c in zip(centers, counts, strict=False)
            )
            n_xt = 5
            xt_vals = np.linspace(tmin, tmax, n_xt)
            span_ns = tmax - tmin

            def _format_xtick(v):
                try:
                    if pd is not None:  # type: ignore
                        ts = pd.to_datetime(int(v), utc=True)
                        if span_ns <= self.dt_config.short_span_ns:
                            return ts.strftime("%Y-%m-%d %H:%M")
                        return ts.strftime("%Y-%m-%d")
                except Exception:
                    pass
                try:
                    from datetime import datetime as _dt

                    return _dt.utcfromtimestamp(int(v) / 1_000_000_000).strftime(
                        "%Y-%m-%d"
                    )
                except Exception:
                    return str(v)

            # Sized like `.cat-svg`: intrinsic width and height matching the
            # viewBox, and the stylesheet scales it with `width: 100%; height:
            # auto`. `width="100%" height="100%"` with
            # `preserveAspectRatio="none"` let the box be whatever the column
            # was and stretched the drawing to fit it, which is the mechanism
            # behind the 2.73x inflation in #217 -- and "none" additionally
            # allowed the x and y scales to diverge, so a circle would not have
            # been round.
            parts = [
                f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Timeline">',
            ]

            # Add title with error handling
            try:
                title_text = self.safe_html_escape(column_name)
                parts.append(
                    f'<text x="{width // 2}" y="15" '
                    f'text-anchor="middle" class="hist-title" '
                    f'font-family="system-ui, -apple-system, sans-serif" '
                    f'font-size="12">{title_text}</text>'
                )
            except Exception:
                # Fallback to generic title
                parts.append(
                    f'<text x="{width // 2}" y="15" '
                    f'text-anchor="middle" class="hist-title" '
                    f'font-family="system-ui, -apple-system, sans-serif" '
                    f'font-size="12">Timeline</text>'
                )

            parts.append('<g class="plot-area">')

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
                start_label = _format_xtick(edges[i])
                end_label = _format_xtick(edges[i + 1])
                range_label = (
                    f"{start_label} – {end_label}"
                    if start_label != end_label
                    else start_label
                )
                pct = (c / sum(counts) * 100) if sum(counts) > 0 else 0
                parts.append(
                    f'<rect class="hot" x="{x0p:.2f}" y="{margin_top}" width="{wp:.2f}" height="{ih:.2f}" '
                    f'fill="transparent" pointer-events="all" '
                    f'data-count="{int(c)}" data-pct="{pct:.1f}" data-label="{range_label}">'
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

            # X ticks. The first and last labels are anchored inward rather
            # than centred: a centred date is half its own width past the end
            # of the axis, so the first one ran off the left edge into the y
            # gutter and the last needed a right margin the size of half a
            # timestamp. Anchoring them is what lets `margin_right` be a pad.
            last_xt = len(xt_vals) - 1
            for i, xv in enumerate(xt_vals):
                px = sx(xv)
                anchor = "start" if i == 0 else ("end" if i == last_xt else "middle")
                parts.append(
                    f'<line class="tick" x1="{px:.2f}" y1="{x_axis_y}" x2="{px:.2f}" y2="{x_axis_y + 4}"></line>'
                )
                parts.append(
                    f'<text class="tick-label x-tick-label" x="{px:.2f}" y="{x_axis_y + 16:.2f}" text-anchor="{anchor}" data-edge="{"first" if i == 0 else ("last" if i == last_xt else "")}">{_format_xtick(xv)}</text>'
                )

            # Axis titles removed

            parts.append("</svg>")
            return "".join(parts)
        except Exception:
            width, height = self._get_chart_dimensions()
            return self.create_empty_svg("dt-svg", width, height)

    def _build_temporal_statistics_table(self, stats: DateTimeStats) -> str:
        """Build temporal statistics table with human-readable formatting."""
        # Format time span in human-readable way
        time_span = getattr(stats, "time_span_days", 0.0)
        if time_span >= 365:
            time_span_display = f"{time_span / 365:.1f} years"
        elif time_span >= 30:
            time_span_display = f"{time_span / 30:.1f} months"
        elif time_span >= 7:
            time_span_display = f"{time_span / 7:.1f} weeks"
        else:
            time_span_display = f"{time_span:.1f} days"

        # Format average interval in human-readable way
        avg_interval = getattr(stats, "avg_interval_seconds", 0.0)
        if avg_interval >= 86400:
            interval_display = f"{avg_interval / 86400:.1f} days"
        elif avg_interval >= 3600:
            interval_display = f"{avg_interval / 3600:.1f} hours"
        elif avg_interval >= 60:
            interval_display = f"{avg_interval / 60:.1f} minutes"
        else:
            interval_display = f"{avg_interval:.1f} seconds"

        # Format interval std in human-readable way
        interval_std = getattr(stats, "interval_std_seconds", 0.0)
        if interval_std >= 86400:
            std_display = f"{interval_std / 86400:.1f} days"
        elif interval_std >= 3600:
            std_display = f"{interval_std / 3600:.1f} hours"
        elif interval_std >= 60:
            std_display = f"{interval_std / 60:.1f} minutes"
        else:
            std_display = f"{interval_std:.1f} seconds"

        weekend_ratio = getattr(stats, "weekend_ratio", 0.0)
        business_hours = getattr(stats, "business_hours_ratio", 0.0)
        seasonal = getattr(stats, "seasonal_pattern", None)

        # Table 1: Timestamp details and basic statistics
        timestamp_data = [
            (
                "Min timestamp",
                self._format_timestamp(getattr(stats, "min_ts", None), multiline=False),
                "timestamp-value",
            ),
            (
                "Max timestamp",
                self._format_timestamp(getattr(stats, "max_ts", None), multiline=False),
                "timestamp-value",
            ),
            (
                f"Unique timestamps{' (≈)' if getattr(stats, 'approx', True) else ''}",
                f"{int(getattr(stats, 'unique_est', 0)):,}",
                "num",
            ),
            ("Timezone", "UTC", None),
            ("Time span", time_span_display, None),
            ("Avg interval", interval_display, None),
            ("Interval std dev", std_display, None),
        ]

        # Table 2: Temporal patterns and peaks
        pattern_data = [
            ("Weekend ratio", f"{weekend_ratio * 100:.1f}%", "num"),
            ("Business hours", f"{business_hours * 100:.1f}%", "num"),
            ("Seasonal pattern", seasonal if seasonal else "—", None),
            ("Peak hour", f"{self._get_peak_hour(stats)}", None),
            ("Peak day", f"{self._get_peak_day(stats)}", None),
            ("Peak month", f"{self._get_peak_month(stats)}", None),
            ("Peak year", f"{self._get_peak_year(stats)}", None),
        ]

        # Build both tables
        table1 = self.table_builder.build_key_value_table(timestamp_data)
        table2 = self.table_builder.build_key_value_table(pattern_data)

        return f"""
        <div class="temporal-analysis">
            <div class="temporal-section">
                {table1}
                    </div>
            <div class="temporal-section">
                {table2}
            </div>
        </div>
        """

    def _get_peak_hour(self, stats: DateTimeStats) -> str:
        """Get peak hour from by_hour distribution."""
        by_hour = getattr(stats, "by_hour", []) or []
        if not by_hour or max(by_hour) == 0:
            return "—"
        peak_idx = by_hour.index(max(by_hour))
        return f"{peak_idx:02d}:00 ({by_hour[peak_idx]:,} records)"

    def _get_peak_day(self, stats: DateTimeStats) -> str:
        """Get peak day of week."""
        by_dow = getattr(stats, "by_dow", []) or []
        if not by_dow or max(by_dow) == 0:
            return "—"
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        peak_idx = by_dow.index(max(by_dow))
        return f"{days[peak_idx]} ({by_dow[peak_idx]:,} records)"

    def _get_peak_month(self, stats: DateTimeStats) -> str:
        """Get peak month."""
        by_month = getattr(stats, "by_month", []) or []
        if not by_month or max(by_month) == 0:
            return "—"
        months = [
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
        peak_idx = by_month.index(max(by_month))
        return f"{months[peak_idx]} ({by_month[peak_idx]:,} records)"

    def _get_peak_year(self, stats: DateTimeStats) -> str:
        """Get peak year."""
        by_year = getattr(stats, "by_year", {}) or {}
        if not by_year or max(by_year.values()) == 0:
            return "—"
        peak_year = max(by_year, key=by_year.get)
        return f"{peak_year} ({by_year[peak_year]:,} records)"

    def _build_missing_values_table(self, stats: DateTimeStats) -> str:
        """Build simple missing values analysis."""
        total_values = stats.count + stats.missing
        missing_pct = (
            (stats.missing / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )
        present_pct = (
            (stats.count / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )
        return super()._build_missing_values_table(
            stats.count, present_pct, stats.missing, missing_pct, stats, total_values
        )

    def _get_missing_data_severity(self, missing_pct: float) -> tuple[str, str, str]:
        """Get missing data severity classification."""
        if missing_pct >= 50:
            return "critical", "Critical", ""
        elif missing_pct >= 20:
            return "high", "High", ""
        elif missing_pct >= 5:
            return "medium", "Medium", ""
        else:
            return "low", "Low", ""

    def _build_dataprep_spectrum_visualization(self, stats: DateTimeStats) -> str:
        """Build DataPrep-style spectrum visualization for missing values per chunk."""
        # Check if we have chunk metadata
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            return ""

        total_values = stats.count + stats.missing
        if total_values == 0:
            return ""

        # Build the spectrum bar segments
        segments_html = ""

        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row + 1
            missing_pct = (
                (missing_count / chunk_size) * 100.0 if chunk_size > 0 else 0.0
            )

            # Calculate segment width as percentage of total
            segment_width_pct = (chunk_size / total_values) * 100.0

            # Determine color based on missing percentage
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

            <div class="spectrum-summary">
                <span class="severity-indicator {severity}">
                    {severity.title()} Missing Data
                </span>
                <span class="spectrum-note">
                    Hover over segments to see chunk details
                </span>
            </div>

            <div class="spectrum-legend">
                <div class="legend-item">
                    <span class="legend-color spectrum-low"></span>
                    <span class="legend-label">Low (0-5%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color spectrum-medium"></span>
                    <span class="legend-label">Medium (5-20%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color spectrum-high"></span>
                    <span class="legend-label">High (20%+)</span>
                </div>
            </div>
        </div>
        """

    def _interval_sentence(self, stats: DateTimeStats) -> str:
        """The strongest fact about the column, said first.

        Phase 5c.5 (#155). On a machine-generated series this was a table row
        reading `Interval std dev — 0.0 seconds`, filed alphabetically between
        timezone and weekend ratio. A standard deviation of zero means *every
        gap is identical*: a record every 17 minutes, no gaps at all. That is
        what anyone opens a datetime column to ask, and it was the least
        prominent thing on the card.

        Only the free half of the design's proposal is taken. The mean and the
        standard deviation are already computed; a longest gap and its
        timestamp would need new state kept alongside the interval array, and
        that is a change to the accumulator rather than to this pane.
        """
        average = getattr(stats, "avg_interval_seconds", None)
        deviation = getattr(stats, "interval_std_seconds", None)
        if not isinstance(average, (int, float)) or average <= 0:
            return ""
        if not isinstance(deviation, (int, float)) or deviation < 0:
            return ""

        gap = self._humanise_seconds(average)
        # Exactly zero, not merely small: the claim "every gap is identical" is
        # only true at zero, and a series that is nearly regular is a different
        # and weaker statement.
        if deviation == 0:
            return (
                f"Every gap is identical: one record every {gap}, with no "
                "irregularity at all. That is a generated series rather than "
                "observed events."
            )

        spread = deviation / average
        if spread < 0.1:
            return (
                f"A record every {gap} on average, and the gaps barely vary "
                f"(± {self._humanise_seconds(deviation)}). Close to regular."
            )
        return (
            f"A record every {gap} on average, but the gaps vary widely "
            f"(± {self._humanise_seconds(deviation)}) — this is an event "
            "stream, not a schedule."
        )

    @staticmethod
    def _humanise_seconds(seconds: float) -> str:
        """A duration in the largest unit that keeps it above one."""
        for size, unit in (
            (86400.0, "day"),
            (3600.0, "hour"),
            (60.0, "minute"),
        ):
            if seconds >= size:
                value = seconds / size
                return f"{value:.1f} {unit}{'s' if round(value, 1) != 1.0 else ''}"
        return f"{seconds:.1f} second{'s' if round(seconds, 1) != 1.0 else ''}"

    def _build_temporal_distributions(self, stats: DateTimeStats) -> str:
        """The four small multiples, each saying what it is a picture of.

        Phase 5c.4 (#155). They had an `<h4>` each and nothing else, so a
        211-record hour and a 2,626-record month drew identically -- and the
        peaks that would have resolved it lived in a different tab. Each
        header now carries its own peak, which the card already computed.

        **The year chart is dropped when the span is inside one year.**
        `by_year` is a dict, so a single year renders one bar at full height:
        a chart whose only reading is "all of it". The span is a sentence
        instead.

        The zero-based y axis, the `RECORDS` unit and rule 3 (a zero count
        draws nothing) are already in `temporal_charts.py` and are not touched
        here -- that part of the audit was stale.
        """
        hour_counts = getattr(stats, "by_hour", None) or [0] * 24
        dow_counts = getattr(stats, "by_dow", None) or [0] * 7
        month_counts = getattr(stats, "by_month", None) or [0] * 12
        year_data = getattr(stats, "by_year", None) or {}

        panels = [
            (
                "Hour of day",
                self._get_peak_hour(stats),
                self.temporal_renderer.render_hour_chart(hour_counts),
            ),
            (
                "Day of week",
                self._get_peak_day(stats),
                self.temporal_renderer.render_dow_chart(dow_counts),
            ),
            (
                "Month",
                self._get_peak_month(stats),
                self.temporal_renderer.render_month_chart(month_counts),
            ),
        ]

        populated_years = [year for year, n in year_data.items() if n]
        if len(populated_years) > 1:
            panels.append(
                (
                    "Year",
                    self._get_peak_year(stats),
                    self.temporal_renderer.render_year_chart(year_data),
                )
            )
            year_note = ""
        elif populated_years:
            year_note = (
                f'<p class="temporal__span">Every record falls in '
                f"{populated_years[0]}, so there is no year distribution to "
                "draw.</p>"
            )
        else:
            year_note = ""

        items = "".join(
            f'<div class="temporal-item">'
            f'<h4 class="temporal__head"><span>{title}</span>'
            + (f'<span class="temporal__peak">peak {peak}</span>' if peak else "")
            + f"</h4>{svg}</div>"
            for title, peak, svg in panels
        )

        # Each chart has its own scale, so their heights cannot be compared to
        # one another. A shared scale would fix that and flatten the hour chart
        # to nothing -- a real trade, taken on the readable side, and said out
        # loud rather than left for a reader to discover.
        caption = (
            '<p class="temporal__caption">each chart is scaled to its own '
            "peak, so heights compare within a chart and not between them</p>"
        )
        return f'<div class="temporal-grid">{items}</div>{year_note}{caption}'

    def _build_details_section(self, col_id: str, stats: DateTimeStats) -> str:
        """Details tabs, minus the ones with nothing to say (#154, 5b.4).

        Missing Values rendered on every datetime column, including ones with no
        gaps, where it drew a 100%-present bar and a one-segment strip reading
        0.0%.
        """
        stats_table = self._build_temporal_statistics_table(stats)
        missing_table = self._build_missing_values_table(stats)
        temporal_charts = self._build_temporal_distributions(stats)

        # The regularity sentence leads the pane. It was a table row filed
        # alphabetically, and it is the strongest thing the column knows.
        sentence = self._interval_sentence(stats)
        if sentence:
            stats_table = f'<p class="fence-lede">{sentence}</p>{stats_table}'

        return self._build_tabbed_details(
            col_id,
            [
                (
                    "stats",
                    "Statistics",
                    f'<div class="sub">{stats_table}</div>',
                    bool(stats_table.strip()),
                ),
                (
                    "temporal",
                    "Temporal Distribution",
                    f'<div class="sub">{temporal_charts}</div>',
                    bool(temporal_charts.strip()),
                ),
                (
                    "missing",
                    "Missing Values",
                    f'<div class="sub">{missing_table}</div>',
                    # Same rule as every other card kind: the pane only knows
                    # something the card face does not when there is more than
                    # one chunk -- where in the read the gaps fall.
                    int(getattr(stats, "missing", 0) or 0) > 0
                    and len(getattr(stats, "chunk_metadata", None) or []) > 1,
                ),
            ],
        )

    def _assemble_card(
        self,
        col_id: str,
        safe_name: str,
        stats: DateTimeStats,
        quality_flags_html: str,
        stat_row: str,
        chart_html: str,
        details_html: str,
    ) -> str:
        """Assemble the complete card HTML."""
        docs_url = "https://alvarodiez20.github.io/pysuricata/stats/datetime/"
        info_button = f'''<a href="{docs_url}" target="_blank" rel="noopener noreferrer" class="info-link" title="View documentation for DateTime analysis" aria-label="View DateTime analysis documentation">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path fill="currentColor" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM6.5 5a.75.75 0 0 0 0 1.5h.5v2.5h-.5a.75.75 0 0 0 0 1.5h3a.75.75 0 0 0 0-1.5h-.5V6h-.5A.75.75 0 0 0 8 5.25H6.5zM8 3.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5z"/>
            </svg>
        </a>'''

        return f"""
        <article class="var-card" id="{col_id}">
            <header class="var-card__header">
                <div class="title">
                    <span class="colname">{safe_name}</span>
                    <span class="badge">Datetime</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {quality_flags_html}
                </div>
                {info_button}
            </header>
            <div class="var-card__body">
                <div class="var-chart">{chart_html}</div>
                {stat_row}
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
