"""DateTime card rendering functionality."""

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .card_base import CardRenderer
from .card_config import DEFAULT_CHART_DIMS, DEFAULT_DT_CONFIG
from .card_types import DateTimeStats, QualityFlags
from .flag_reference import (
    BUSINESS_HOURS_FLAG_PCT,
    BUSINESS_HOURS_FLAT_PCT,
    WEEKEND_FLAT_PCT,
    WEEKEND_HEAVY_FLAG_PCT,
    flat_verdict,
)
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

        Its own, not `ChartDimensions` (420x180). That is the numeric card's
        plot size, and the timeline is drawn with `preserveAspectRatio="none"`
        at `width: 100%`, so borrowing it made the viewBox 2.73x smaller than
        the box it was painted into and scaled every label by the same factor.
        See `DateTimeConfig.chart_width`.
        """
        return self.dt_config.chart_width, self.dt_config.chart_height

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

        # Chart, and the two things that now sit beside and above it
        chart_html = self._build_timeline_chart(stats)
        baselines_html = self._build_calendar_baselines(stats)
        lede = self._interval_sentence(stats)
        lede_html = f'<p class="fence-lede dt-lede">{lede}</p>' if lede else ""

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
            lede_html,
            baselines_html,
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

        # Weekend concentrated. The baseline these thresholds are set against
        # used to sit in a comment on this line, which is what #291 was filed
        # about; it is now the constant the panel draws its rule from, so the
        # chip and the rule cannot drift apart. The thresholds themselves are
        # unchanged -- see the two `*_FLAG_PCT` constants for how each relates
        # to its baseline.
        weekend_ratio = getattr(stats, "weekend_ratio", 0.0)
        if weekend_ratio * 100 > WEEKEND_HEAVY_FLAG_PCT:
            flag_items.append('<li class="flag">Weekend-heavy</li>')

        # Business hours concentrated
        business_hours = getattr(stats, "business_hours_ratio", 0.0)
        if business_hours * 100 > BUSINESS_HOURS_FLAG_PCT:
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
        """The identity half of the face: what the column is, and how much.

        `Avg interval` and `Interval std` left this table in #292. They are not
        deleted -- `_build_temporal_statistics_table` still carries both, and
        `_interval_sentence` now leads the card face, where it says what the
        pair means better than the raw pair read.
        """
        # Format time span
        time_span = getattr(stats, "time_span_days", 0.0)
        if time_span >= 365:
            span_display = f"{time_span / 365:.1f} years"
        elif time_span >= 30:
            span_display = f"{time_span / 30:.1f} months"
        else:
            span_display = f"{time_span:.1f} days"

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
            ("Timezone", self._timezone_display(stats), None),
            ("Time span", span_display, None),
        ]

        return data

    def _right_stats(self, stats: DateTimeStats) -> list[tuple[str, str, str | None]]:
        """The range half of the face: where the column starts, ends, and how densely.

        Facts about the column only. `Processed bytes (≈)` moved to the
        Statistics pane in #209 -- see `_build_temporal_statistics_table`.
        `Weekend %` and `Business hrs %` left in #291: bare, neither could be
        judged, so they are drawn against their flat-calendar baseline by
        `_build_calendar_baselines` instead of printed as two more numbers.
        """
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
            # Single-height (#292). Two double-height cells in a four-column
            # grid made every row in the grid taller, including the six that
            # carry one line.
            (
                "Min",
                self._format_timestamp(
                    getattr(stats, "min_ts", None),
                    multiline=False,
                    suffix=self._instant_suffix(stats),
                ),
                "timestamp-value",
            ),
            (
                "Max",
                self._format_timestamp(
                    getattr(stats, "max_ts", None),
                    multiline=False,
                    suffix=self._instant_suffix(stats),
                ),
                "timestamp-value",
            ),
            ("Data density", density_display, None),
        ]

        return data

    #: A column whose dtype carries no zone. `pandas` writes `datetime64[ns]`,
    #: polars `Datetime(time_unit='ns')` -- neither mentions a zone, because
    #: there is not one.
    _NAIVE_LABEL = "— (naive)"

    @staticmethod
    def _timezone_of(stats: DateTimeStats) -> str | None:
        """The column's timezone, or `None` when it genuinely has none (#241).

        The card printed the literal `"UTC"` for every datetime column. A
        `US/Eastern` column was labelled UTC, and so was a naive one -- which
        has no timezone at all. That is the report stating a fact about the
        data that it did not get from the data.

        `source_timezone` is the obvious source and is not sufficient on its
        own: the accumulator stores it only when it is *not* UTC
        (`if tz_part and tz_part != "UTC"`), so `None` means "naive **or**
        UTC" and cannot express the distinction this is about. The dtype
        string can, and is carried on the summary already.
        """
        source = getattr(stats, "source_timezone", None)
        if source:
            return str(source)

        dtype = str(getattr(stats, "dtype_str", "") or "")

        # polars first, and the order is load-bearing. Its dtype reads
        # `Datetime(time_unit='us', time_zone='US/Eastern')` -- which *contains
        # a comma*, so the pandas branch below matches it and returns
        # `time_zone='US/Eastern')` as the zone name. A naive polars column is
        # worse: `time_zone=None` is a non-empty tail, so it reports the string
        # `time_zone=None)` as a timezone instead of saying naive.
        if "time_zone=" in dtype:
            import re as _re

            match = _re.search(r"time_zone=['\"]([^'\"]+)['\"]", dtype)
            return match.group(1) if match else None

        # pandas: `datetime64[ns]` naive, `datetime64[ns, UTC]` aware.
        if "," in dtype:
            tail = dtype.split(",", 1)[1].rstrip(")]").strip()
            return tail or None
        return None

    def _timezone_display(self, stats: DateTimeStats) -> str:
        zone = self._timezone_of(stats)
        return zone if zone else self._NAIVE_LABEL

    def _instant_suffix(self, stats: DateTimeStats) -> str:
        """What to append to a rendered instant.

        The accumulator stores epoch nanoseconds, so rendering them through
        `utc=True` is correct **for a zone-aware column** -- the instant really
        is that moment in UTC, whatever zone it was written in. For a naive
        column there is no instant, only a wall clock, and calling it UTC
        invents the zone. So the suffix is dropped there rather than guessed.
        """
        return " UTC" if self._timezone_of(stats) else ""

    def _format_timestamp(
        self, ts: int | None, multiline: bool = True, suffix: str = " UTC"
    ) -> str:
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
                time_part = dt.strftime("%H:%M:%S") + suffix
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
            time_part = dt.strftime("%H:%M:%S") + suffix
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

    #: The two calendar shares, with what a flat calendar would give each.
    #: Both baselines are arithmetic rather than estimates, and neither costs
    #: the accumulator a new statistic -- the ratios were already computed and
    #: already on the card; #291 only gave them something to be read against.
    _CALENDAR_BASELINES: tuple[tuple[str, str, float, str], ...] = (
        ("Weekend share", "weekend_ratio", WEEKEND_FLAT_PCT, "2 of 7 days"),
        (
            "Business hours",
            "business_hours_ratio",
            BUSINESS_HOURS_FLAT_PCT,
            "8 of 24 hours on 5 of 7 days",
        ),
    )

    def _build_calendar_baselines(self, stats: DateTimeStats) -> str:
        """Draw each calendar share against the flat value that makes it readable.

        `Weekend % 27.0` printed bare is noise wearing the clothes of a
        finding: a flat calendar gives 28.6%, so 27.0 is the *absence* of a
        weekend effect. The bar carries the observed share, the rule carries
        the baseline, and the verdict states the gap in percentage points so
        the reading can be checked against the mark (phase 5e.2, #291).

        The rule is drawn *before* the bar, so a bar that reaches past it
        occludes it rather than crossing it -- rule 2 in `tokens.css`. It also
        protrudes above and below the track onto the paper, which is what keeps
        it visible at 390px after its label is dropped.

        Nothing is drawn for a column with no values. A zero-row frame reaches
        this now that #315 renders one at all, and the ratios finalise to 0.0,
        which the verdict read as `under-represented -28.6pp vs 28.6%` -- a
        confident finding about a column that contains nothing. The whole point
        of this panel is to stop a number being read as a finding when it is
        not one, so it must not become the thing doing that.
        """
        if not int(getattr(stats, "count", 0) or 0):
            return ""

        rows: list[str] = []
        for label, attr, flat_pct, unit_note in self._CALENDAR_BASELINES:
            raw = getattr(stats, attr, None)
            if not isinstance(raw, (int, float)):
                continue
            actual_pct = float(raw) * 100.0
            verdict, tone = flat_verdict(actual_pct, flat_pct)
            title = f"A flat calendar would give {flat_pct:.1f}% \u2014 {unit_note}"
            rows.append(
                f'<div class="cal-base__row">'
                f'<div class="cal-base__head">'
                f'<span class="cal-base__label">{label}</span>'
                f'<span class="cal-base__value vstat__val">{actual_pct:.1f}%</span>'
                f"</div>"
                f'<div class="cal-base__track">'
                f'<span class="cal-base__rule" title="{title}" '
                f'style="left:{min(100.0, flat_pct):.2f}%"></span>'
                f'<span class="cal-base__fill" '
                f'style="width:{min(100.0, max(0.0, actual_pct)):.2f}%"></span>'
                f"</div>"
                f'<span class="cal-base__verdict {tone}">{verdict}</span>'
                f"</div>"
            )
        if not rows:
            return ""
        return (
            '<div class="cal-base">'
            '<span class="cal-base__title vstat__cap">Shape of the calendar</span>'
            + "".join(rows)
            + '<span class="cal-base__note">rule marks the flat-calendar share '
            "\u2014 2 of 7 days, and 8 of 24 hours on 5 of 7 days</span>"
            "</div>"
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

            # A pixel-space coordinate system used to be built here -- margins,
            # an inner box, `sx`/`sy`, bin centres and a `pts` string -- and the
            # only thing it produced was `pts`, which nothing read. The marks
            # below are drawn in `gx`/`gy`, the 0..100 viewBox space the chart
            # actually uses. `render/*` carries a per-file `F841` ignore, so
            # ruff never said so.
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

            # A marks-only SVG in a square viewBox, stretched by CSS, with every
            # label in HTML beside it -- the structure the numeric histogram
            # already uses, and the reason it does.
            #
            # This chart used to draw its labels *inside* an SVG carrying
            # `preserveAspectRatio="none"` at `width: 100%`. Nothing inside such
            # an SVG has a fixed size: the viewBox maps onto whatever box CSS
            # gives it, so an 11px tick label rendered at 37px in a 1,146px
            # column and would render at 5px in a 470px one. There is no viewBox
            # that is right at both widths, which is why the fix is to take the
            # text out of the SVG rather than to pick a better number.
            #
            # Reusing the `hist` classes rather than styling a second chart: the
            # gutter, the tiered x labels, the caption and the axis-label
            # nudges all already exist and are already tested.
            span = self._SPAN
            marks = [
                f'<svg class="hist-svg" viewBox="0 0 {span:g} {span:g}" '
                f'preserveAspectRatio="none" role="img" '
                f'aria-label="Timeline for {self.safe_html_escape(column_name)}">',
                f"<desc>Records over time, {len(counts)} intervals</desc>",
            ]

            def gx(x: float) -> float:
                return (x - tmin) / (tmax - tmin) * span

            def gy(y: float) -> float:
                return (1 - y / y_max) * span

            # `vector-effect="non-scaling-stroke"` throughout: the box is
            # stretched by a different factor on each axis, so without it a
            # 1-unit rule is thick one way and invisible the other.
            for yt in y_ticks:
                marks.append(
                    f'<line class="grid" x1="0" y1="{gy(yt):.3f}" '
                    f'x2="{span:g}" y2="{gy(yt):.3f}" '
                    f'vector-effect="non-scaling-stroke"/>'
                )
            marks.append(
                f'<line class="axis" x1="0" y1="{span:g}" x2="{span:g}" '
                f'y2="{span:g}" vector-effect="non-scaling-stroke"/>'
            )
            marks.append(
                f'<line class="axis" x1="0" y1="0" x2="0" y2="{span:g}" '
                f'vector-effect="non-scaling-stroke"/>'
            )

            # Bars, not a polyline (#293, 5e.4). A line through bucket centres
            # draws a slope between "84 records on 8 Jan" and "83 on 9 Jan",
            # asserting every intermediate value -- and the data holds values
            # only at the buckets. A bucket count is a quantity per interval,
            # which is what a bar means and what a line does not, and the card's
            # own temporal panes and the numeric histogram already draw counts
            # that way. One encoding for counts, across the report.
            #
            # The plan proposed keeping the line above ~180 buckets, where bars
            # would be sub-pixel. That threshold cannot fire: `default_bins` is
            # 60, it is not reachable from `ProfileConfig` or `ComputeOptions`,
            # so `min(bins, 180)` is always 60 and the line branch would be
            # unreachable code with no input that tests it.
            #
            # The sub-pixel risk is real but it is a *viewport* width, not a
            # bucket count -- 60 buckets are 13px each at 1240 and 3.8px at 390.
            # A static report cannot branch on that, and the numeric histogram
            # already ships ~4.5px bars at 390 on the same screen, so this is
            # the width the report has always drawn counts at.
            #
            # Edge to edge with the separator as a non-scaling stroke, exactly
            # as `histogram_svg._render_bars` does and for the reason recorded
            # there: a gap in viewBox units is a fraction of the data, so it
            # shrinks to nothing on a narrow plot.
            for i, c in enumerate(counts):
                # Rule 3: a zero count draws nothing. A 1px floor is right for a
                # small non-zero value and wrong for zero -- an empty month drawn
                # as a 1px bar asserts data that is not there.
                if c <= 0:
                    continue
                x0b = gx(float(edges[i]))
                bar_h = float(c) / y_max * span
                marks.append(
                    # Two decimals: the viewBox is 0..100, so a unit is a
                    # percent of the plot and the third decimal is a
                    # ten-thousandth of a pixel at any width this renders at.
                    f'<rect class="bar" x="{x0b:.2f}" y="{span - bar_h:.2f}" '
                    f'width="{gx(float(edges[i + 1])) - x0b:.2f}" '
                    f'height="{bar_h:.2f}"/>'
                )

            total = int(counts.sum())
            marks.append('<g class="hotspots">')
            for i, c in enumerate(counts):
                x0p = gx(float(edges[i]))
                x1p = gx(float(edges[i + 1]))
                start_label = _format_xtick(edges[i])
                end_label = _format_xtick(edges[i + 1])
                range_label = (
                    f"{start_label} – {end_label}"
                    if start_label != end_label
                    else start_label
                )
                pct = (c / total * 100) if total > 0 else 0.0
                marks.append(
                    f'<rect class="hot" x="{x0p:.3f}" y="0" '
                    f'width="{max(0.001, x1p - x0p):.3f}" height="{span:g}" '
                    f'fill="transparent" pointer-events="all" '
                    f'data-count="{int(c)}" data-pct="{pct:.1f}" '
                    f'data-label="{self.safe_html_escape(range_label)}"/>'
                )
            marks.append("</g>")
            marks.append("</svg>")

            return (
                '<figure class="hist dt-figure">'
                '<div class="hist__plot">'
                f'<div class="hist__gutter">{self._render_count_labels(y_ticks, y_max)}'
                '<span class="hist__unit">ROWS</span></div>'
                f'<div class="hist__area">{"".join(marks)}'
                f"{self._render_time_labels(xt_vals, _format_xtick)}</div>"
                "</div>"
                f"{self._render_timeline_caption(counts, edges, _format_xtick)}"
                "</figure>"
            )
        except Exception:
            width, height = self._get_chart_dimensions()
            return self.create_empty_svg("dt-svg", width, height)

    #: A square viewBox stretched by CSS on both axes, as the histogram uses.
    _SPAN = 100.0

    #: Five time labels, thinning to three under 768px. Dates are ~10 glyphs
    #: where a histogram's numbers are ~6, so nine would collide long before
    #: the histogram's do.
    _TIME_TIERS = (1, 3, 1, 3, 1)

    def _render_count_labels(self, y_ticks, y_max: float) -> str:
        """Count labels in the gutter, positioned as a percentage of the plot.

        `data-edge` marks the two extremes so the stylesheet can nudge them
        inward; without it the top label overhangs the plot and the `0` drops
        into the tick row below.
        """
        if not y_max:
            return ""
        out = []
        for tick in y_ticks:
            top = (1 - tick / y_max) * 100.0
            edge = ""
            if top <= 0.0:
                edge = ' data-edge="top"'
            elif top >= 100.0:
                edge = ' data-edge="bottom"'
            out.append(
                f'<span class="hist__y"{edge} style="top:{top:.3f}%">'
                f"{int(round(float(tick))):,}</span>"
            )
        return "".join(out)

    def _render_time_labels(self, xt_vals, formatter) -> str:
        """Dates across the axis, tagged by importance so CSS can thin them."""
        count = len(self._TIME_TIERS)
        out = []
        for index, tier in enumerate(self._TIME_TIERS):
            fraction = index / (count - 1)
            position = int(round(fraction * (len(xt_vals) - 1)))
            # The end labels anchor to the plot edge rather than centring on
            # their tick, so a wide date at either end sits inside the chart
            # instead of overhanging it.
            if index == 0:
                anchor = ' data-anchor="start"'
            elif index == count - 1:
                anchor = ' data-anchor="end"'
            else:
                anchor = ""
            out.append(
                f'<span class="hist__tick" data-tier="{tier}"{anchor} '
                f'style="left:{fraction * 100:.3f}%">'
                f"{self.safe_html_escape(formatter(xt_vals[position]))}</span>"
            )
        return f'<div class="hist__x">{"".join(out)}</div>'

    def _render_timeline_caption(self, counts, edges, formatter) -> str:
        """`60 intervals · peak 83 rows at 2024-03-01`.

        The same job the histogram's caption does: the y labels round, so the
        exact peak lives here.
        """
        if len(counts) == 0:
            return ""
        peak = int(counts.argmax())
        return (
            '<figcaption class="hist__caption">'
            f"{len(counts):,} intervals · peak {int(counts[peak]):,} rows at "
            f"{self.safe_html_escape(formatter(edges[peak]))}"
            "</figcaption>"
        )

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
                self._format_timestamp(
                    getattr(stats, "min_ts", None),
                    multiline=False,
                    suffix=self._instant_suffix(stats),
                ),
                "timestamp-value",
            ),
            (
                "Max timestamp",
                self._format_timestamp(
                    getattr(stats, "max_ts", None),
                    multiline=False,
                    suffix=self._instant_suffix(stats),
                ),
                "timestamp-value",
            ),
            (
                f"Unique timestamps{' (≈)' if getattr(stats, 'approx', True) else ''}",
                f"{int(getattr(stats, 'unique_est', 0)):,}",
                "num",
            ),
            ("Timezone", self._timezone_display(stats), None),
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
            # UX-21 / #209. Moved out of the card's primary stat row, where it
            # sat under `Data density` among facts about the column and was the
            # only one about the profiler's own bookkeeping.
            (
                "Processed bytes (≈)",
                self.format_bytes(int(getattr(stats, "mem_bytes", 0) or 0)),
                "num",
            ),
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

        # The regularity sentence used to lead this pane. It leads the card
        # face now (#292) -- it is read before the conclusions rather than
        # after them. `Avg interval` and `Interval std dev` stay in the table
        # below, so the pair the sentence interprets is still reachable.

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
        lede_html: str = "",
        baselines_html: str = "",
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
                {lede_html}
                <div class="dt-face">
                    <div class="var-chart">{chart_html}</div>
                    {baselines_html}
                </div>
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
