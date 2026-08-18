"""Numeric card rendering functionality."""

import math
from collections.abc import Sequence

from .card_base import CardRenderer
from .card_config import (
    DEFAULT_CHART_DIMS,
    DEFAULT_HIST_CONFIG,
    DEFAULT_TICK_CONFIG,
)
from .card_types import NumericStats, QualityFlags, QuantileData
from .format_utils import fmt_compact_scientific as _fmt_compact_scientific
from .histogram_svg import SVGHistogramRenderer
from .identifier import identifier_facts, looks_like_identifier
from .outlier_fence import (
    build_fence,
    classify,
    cluster_marks,
    fence_verdict,
    method_note,
    render_figure,
    render_quantile_strip,
    render_table,
)
from .sampling import quantiles_are_sampled
from .triage import annotate_flags


class NumericCardRenderer(CardRenderer):
    """Renders numeric data cards."""

    def __init__(self):
        # Initialize SVG histogram renderer
        self.svg_histogram_renderer = SVGHistogramRenderer()
        super().__init__()
        self.chart_dims = DEFAULT_CHART_DIMS
        self.hist_config = DEFAULT_HIST_CONFIG
        self.tick_config = DEFAULT_TICK_CONFIG

    def render_card(self, stats: NumericStats) -> str:
        """Render a complete numeric card."""
        col_id = self.safe_col_id(stats.name)
        safe_name = self.safe_html_escape(stats.name)

        # Calculate percentages and classes
        percentages = self._calculate_percentages(stats)
        quality_flags = self.quality_assessor.assess_numeric_quality(stats)

        # Build components
        approx_badge = self._build_approx_badge(stats.approx)
        quality_flags_html = self._build_quality_flags_html(
            quality_flags, percentages, stats
        )

        stat_row = self._build_stat_row(
            self._left_stats(stats, percentages) + self._right_stats(stats)
        )

        quantiles = self._compute_quantiles_from_sample(stats.sample_vals or [])
        quant_stats_table = self._build_quant_stats_table(stats, quantiles)

        chart_html = self._build_histogram_variants(col_id, safe_name, stats)

        stats_table = self._build_stats_table(stats)
        common_table = self._build_common_values_table(stats, col_id)
        extremes_table = self._build_extremes_table(stats, quantiles, col_id)
        outliers_pane = self._build_outliers_pane(stats, quantiles, col_id)
        corr_table = self._build_correlation_table(stats)
        missing_table = self._build_missing_values_table(stats)

        stats_quantiles = self._build_statistics_pane(
            stats, quantiles, stats_table, quant_stats_table
        )

        details_html, pane_summary = self._build_details_section(
            col_id,
            stats,
            stats_quantiles,
            common_table,
            extremes_table,
            outliers_pane,
            corr_table,
            missing_table,
        )

        controls_html = self._build_controls_section(
            col_id,
            log_default=bool(getattr(quality_flags, "log_scale_suggested", False)),
            pane_summary=pane_summary,
        )

        return self._assemble_card(
            col_id,
            safe_name,
            stats,
            approx_badge,
            quality_flags_html,
            stat_row,
            chart_html,
            details_html,
            controls_html,
        )

    def _calculate_percentages(self, stats: NumericStats) -> dict:
        """Calculate percentage values for display."""
        total = max(1, stats.count + stats.missing)
        return {
            "miss_pct": (stats.missing / total) * 100.0,
            "zeros_pct": (stats.zeros / max(1, stats.count)) * 100.0
            if stats.count
            else 0.0,
            "neg_pct": (stats.negatives / max(1, stats.count)) * 100.0
            if stats.count
            else 0.0,
            "out_pct": (stats.outliers_iqr / max(1, stats.count)) * 100.0
            if stats.count
            else 0.0,
            "inf_pct": (stats.inf / max(1, stats.count)) * 100.0
            if stats.count
            else 0.0,
        }

    def _build_quality_flags_html(
        self, flags: QualityFlags, percentages: dict, stats: NumericStats
    ) -> str:
        """The chips, with the number each one already knows on its face.

        `_quality_flags_markup` builds them; this puts the value on the
        chip and the threshold in a title. Splitting it this way means the
        forty-two places that emit a chip carry on emitting the same
        markup, and the annotation lives in one place rather than being
        repeated at every one of them.
        """
        return annotate_flags(self._quality_flags_markup(flags, percentages, stats))

    def _quality_flags_markup(
        self, flags: QualityFlags, percentages: dict, stats: NumericStats
    ) -> str:
        """Build quality flags HTML with percentage context and threshold tooltips."""
        flag_items = []

        if flags.missing:
            severity = "bad" if percentages["miss_pct"] > 20 else "warn"
            threshold = ">20%" if percentages["miss_pct"] > 20 else "≤20%"
            flag_items.append(
                f'<li class="flag {severity}" data-threshold="{threshold}" '
                f'data-value="{percentages["miss_pct"]:.1f}%">Missing</li>'
            )

        if flags.infinite:
            flag_items.append(
                f'<li class="flag bad" data-threshold="Any ∞" '
                f'data-value="{stats.inf} values">Has ∞</li>'
            )

        if flags.has_negatives:
            severity = "warn" if percentages["neg_pct"] > 10 else ""
            threshold = ">10%" if percentages["neg_pct"] > 10 else "Present"
            flag_items.append(
                f'<li class="flag {severity}" data-threshold="{threshold}" '
                f'data-value="{percentages["neg_pct"]:.1f}%">Has negatives</li>'
                if severity
                else f'<li class="flag" data-threshold="{threshold}" '
                f'data-value="{percentages["neg_pct"]:.1f}%">Has negatives</li>'
            )

        if flags.zero_inflated:
            severity = "bad" if percentages["zeros_pct"] >= 50.0 else "warn"
            threshold = "≥50%" if percentages["zeros_pct"] >= 50.0 else "<50%"
            flag_items.append(
                f'<li class="flag {severity}" data-threshold="{threshold}" '
                f'data-value="{percentages["zeros_pct"]:.1f}%">Zero‑inflated</li>'
            )

        if flags.positive_only:
            flag_items.append('<li class="flag good">Positive‑only</li>')

        if flags.skewed_right:
            skew_val = getattr(stats, "skew", 0)
            flag_items.append(
                f'<li class="flag warn" data-threshold=">1" data-value="{skew_val:.2f}">Skewed Right</li>'
            )

        if flags.skewed_left:
            skew_val = getattr(stats, "skew", 0)
            flag_items.append(
                f'<li class="flag warn" data-threshold="<-1" data-value="{skew_val:.2f}">Skewed Left</li>'
            )

        if flags.heavy_tailed:
            kurt_val = getattr(stats, "kurtosis", 0)
            flag_items.append(
                f'<li class="flag bad" data-threshold="|kurtosis| > 3" data-value="{kurt_val:.2f}">Heavy‑tailed</li>'
            )

        if flags.approximately_normal:
            jb_val = getattr(stats, "jb_chi2", 0)
            flag_items.append(
                f'<li class="flag good" data-threshold="JB χ² < 5.99" data-value="{jb_val:.2f}">≈ Normal (JB)</li>'
            )

        if flags.discrete:
            # The count, not a ratio: the flag is now an absolute cardinality
            # test, and quoting a ratio would explain it with the number that
            # used to make it wrong.
            flag_items.append(
                f'<li class="flag warn" data-threshold="≤ 50 whole-number levels" '
                f'data-value="~{stats.unique_est:,}">Discrete</li>'
            )

        if flags.heaping:
            heap_pct = getattr(stats, "heap_pct", 0)
            flag_items.append(
                f'<li class="flag" data-threshold="Detected rounding" data-value="{heap_pct:.1f}%">Heaping</li>'
            )

        if flags.bimodal:
            flag_items.append('<li class="flag warn">Possibly bimodal</li>')

        if flags.log_scale_suggested:
            flag_items.append('<li class="flag good">Log‑scale?</li>')

        if flags.constant:
            flag_items.append(
                f'<li class="flag bad" data-threshold="1 unique" data-value="{stats.unique_est}">Constant</li>'
            )

        if flags.quasi_constant:
            top_values = getattr(stats, "top_values", None)
            share = (
                top_values[0][1] / max(1, stats.count)
                if top_values
                else float(stats.unique_est <= 2)
            )
            flag_items.append(
                f'<li class="flag warn" data-threshold="One value covers ≥ 95%" '
                f'data-value="{share:.1%}">Quasi‑constant</li>'
            )

        if flags.many_outliers:
            out_pct = percentages.get("out_pct", 0)
            flag_items.append(
                f'<li class="flag bad" data-threshold=">1%" data-value="{out_pct:.1f}%">Many outliers</li>'
            )

        if flags.some_outliers:
            out_pct = percentages.get("out_pct", 0)
            flag_items.append(
                f'<li class="flag warn" data-threshold="0.3%-1%" data-value="{out_pct:.1f}%">Some outliers</li>'
            )

        if flags.monotonic_increasing:
            flag_items.append('<li class="flag good">Monotonic ↑</li>')

        if flags.monotonic_decreasing:
            flag_items.append('<li class="flag good">Monotonic ↓</li>')

        return (
            f"<ul class='quality-flags'>{''.join(flag_items)}</ul>"
            if flag_items
            else ""
        )

    def _left_stats(
        self, stats: NumericStats, percentages: dict
    ) -> list[tuple[str, str, str]]:
        """The counting half of the stat row: how many, how many missing, how
        many of the kinds of value that need watching."""
        if looks_like_identifier(stats):
            # "Zeros: 1 (0.0%)" on a key is the line UX-2 names as actively
            # misleading: it is true, and it means nothing. Outliers, infinities
            # and negatives are the same. The identifier facts go on the right.
            return [
                ("Count", f"{stats.count:,}", "num"),
                (
                    "Missing",
                    f"{stats.missing:,} ({percentages['miss_pct']:.1f}%)",
                    "num",
                ),
                ("Type", "identifier (key-like)", ""),
            ]

        miss_cls = (
            "crit"
            if percentages["miss_pct"] > 20
            else ("warn" if percentages["miss_pct"] > 0 else "")
        )
        out_cls = (
            "crit"
            if percentages["out_pct"] > 1
            else ("warn" if percentages["out_pct"] > 0.3 else "")
        )
        zeros_cls = "warn" if percentages["zeros_pct"] > 30 else ""
        inf_cls = "crit" if stats.inf else ""
        neg_cls = (
            "warn"
            if 0 < percentages["neg_pct"] <= 10
            else ("crit" if percentages["neg_pct"] > 10 else "")
        )

        data = [
            ("Count", f"{stats.count:,}", "num"),
            (f"Unique{' (≈)' if stats.approx else ''}", f"{stats.unique_est:,}", "num"),
            (
                "Missing",
                f"{stats.missing:,} ({percentages['miss_pct']:.1f}%)",
                f"num {miss_cls}",
            ),
            (
                "Outliers",
                f"{stats.outliers_iqr:,} ({percentages['out_pct']:.1f}%)",
                f"num {out_cls}",
            ),
            (
                "Zeros",
                f"{stats.zeros:,} ({percentages['zeros_pct']:.1f}%)",
                f"num {zeros_cls}",
            ),
            (
                "Infinites",
                f"{stats.inf:,} ({percentages['inf_pct']:.1f}%)",
                f"num {inf_cls}",
            ),
            (
                "Negatives",
                f"{stats.negatives:,} ({percentages['neg_pct']:.1f}%)",
                f"num {neg_cls}",
            ),
        ]

        return data

    def _right_stats(self, stats: NumericStats) -> list[tuple[str, str, str]]:
        """The distribution half: where the values sit.

        Only that. `Processed bytes (≈)` used to close both branches below and
        now lives in the Statistics pane (#209) -- see `_build_stats_table`.
        """
        if looks_like_identifier(stats):
            # A key's mean, median and quartiles are arithmetic on labels. Show
            # what a key actually raises instead: how many, how many distinct,
            # whether the sequence has gaps.
            return [(label, value, "num") for label, value in identifier_facts(stats)]

        # Min, Max and Mean are exact -- the extremes come from every value
        # (#118) and the mean from Welford over the stream -- so they must keep
        # looking different from the three beside them that do not.
        sampled = " (≈)" if quantiles_are_sampled(stats) else ""

        data = [
            ("Min", self.format_number(stats.min), "num"),
            (f"Q1 (P25){sampled}", self.format_number(stats.q1), "num"),
            (f"Median{sampled}", self.format_number(stats.median), "num"),
            ("Mean", self.format_number(stats.mean), "num"),
            (f"Q3 (P75){sampled}", self.format_number(stats.q3), "num"),
            ("Max", self.format_number(stats.max), "num"),
        ]

        return data

    def _compute_quantiles_from_sample(
        self, sample_vals: Sequence[float]
    ) -> QuantileData:
        """Compute quantiles from sample values."""
        if not sample_vals:
            return QuantileData(
                p1=float("nan"),
                p5=float("nan"),
                p10=float("nan"),
                p90=float("nan"),
                p95=float("nan"),
                p99=float("nan"),
            )

        n = len(sample_vals)
        sorted_vals = sorted(sample_vals)

        def _quantile(p: float) -> float:
            i = (n - 1) * p
            lo = int(math.floor(i))
            hi = int(math.ceil(i))
            if lo == hi:
                return float(sorted_vals[int(i)])
            return float(sorted_vals[lo] * (hi - i) + sorted_vals[hi] * (i - lo))

        return QuantileData(
            p1=_quantile(0.01),
            p5=_quantile(0.05),
            p10=_quantile(0.10),
            p90=_quantile(0.90),
            p95=_quantile(0.95),
            p99=_quantile(0.99),
        )

    def _build_quant_stats_table(
        self, stats: NumericStats, quantiles: QuantileData
    ) -> str:
        """Build quantile statistics table."""
        range_val = (
            (stats.max - stats.min)
            if (
                isinstance(stats.max, (int, float))
                and isinstance(stats.min, (int, float))
            )
            else float("nan")
        )

        data = [
            ("Min", self.format_number(stats.min), "num"),
            ("P1 (≈)", self.format_number(quantiles.p1), "num"),
            ("P5 (≈)", self.format_number(quantiles.p5), "num"),
            ("P10 (≈)", self.format_number(quantiles.p10), "num"),
            ("Q1 (P25)", self.format_number(stats.q1), "num"),
            ("Median (P50)", self.format_number(stats.median), "num"),
            ("Q3 (P75)", self.format_number(stats.q3), "num"),
            ("P90 (≈)", self.format_number(quantiles.p90), "num"),
            ("P95 (≈)", self.format_number(quantiles.p95), "num"),
            ("P99 (≈)", self.format_number(quantiles.p99), "num"),
            ("Range", self.format_number(range_val), "num"),
            # `Std Dev` used to be printed here *and* in the statistics table.
            # It is a moment, not an order statistic, so it stays with the
            # moments and this table stops repeating it (#154, 5b.1).
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_histogram_variants(
        self, col_id: str, base_title: str, stats: NumericStats
    ) -> str:
        """Build histogram variants HTML with SVG using true distribution."""
        # Use true distribution histogram data if available
        true_edges = getattr(stats, "true_histogram_edges", None)
        true_counts = getattr(stats, "true_histogram_counts", None)

        if not (
            true_edges and true_counts and len(true_edges) > 1 and len(true_counts) > 0
        ):
            # No true distribution data available
            return f'''
            <div class="hist-chart">
                <div class="hist-variants" data-col="{col_id}">
                    <div class="hist variant active">
                        {self.svg_histogram_renderer._render_empty_histogram("No histogram data")}
                    </div>
                </div>
            </div>
            '''

        # Generate histogram variants with true distribution data
        variants = []
        for bins in self.hist_config.bin_options:
            for scale in ["lin", "log"]:
                # Create title with scale indicator
                title = f"{base_title}{' (log scale)' if scale == 'log' else ''}"

                # Generate SVG histogram with true distribution
                svg_content = self.svg_histogram_renderer.render_histogram_from_bins(
                    bin_edges=true_edges,
                    bin_counts=true_counts,
                    bins=bins,
                    scale=scale,
                    title=title,
                    col_id=col_id,
                    # Where the log axis starts. `getattr` because this is
                    # duck-typed across the accumulator summary and the render
                    # type, and a card built from an older payload has neither.
                    min_positive=getattr(stats, "min_positive", None),
                    non_positive=(
                        int(getattr(stats, "zeros", 0) or 0)
                        + int(getattr(stats, "negatives", 0) or 0)
                    ),
                )

                active_class = " active" if (bins == 25 and scale == "lin") else ""

                variants.append(
                    f'<div class="hist variant{active_class}" id="{col_id}-{scale}-bins-{bins}" data-scale="{scale}" data-bin="{bins}">'
                    f"{svg_content}"
                    f"</div>"
                )

        return f'''
        <div class="hist-chart">
            <div class="hist-variants" data-col="{col_id}">
                {"".join(variants)}
            </div>
        </div>
        '''

    def _build_statistics_pane(
        self,
        stats: NumericStats,
        quantiles: QuantileData,
        stats_table: str,
        quant_stats_table: str,
    ) -> str:
        """The percentiles as a shape, then the tables that hold the figures.

        Phase 5b.1 (#154). Twenty-six key-value rows across two tables, with
        nothing in the layout saying which to read: `Jarque-Bera chi-squared`
        carried the same weight as `Median`, and `Std Dev` was printed twice.

        The strip costs no new statistics. Every number in it was already in
        the two tables below it -- what changes is that they are on an axis, so
        a reader can see the middle half of `Age` sitting in a narrow band well
        left of centre, which no arrangement of a table can show.

        The prose lines spend thresholds the report already holds and never
        showed: `data-threshold="JB chi-squared < 5.99"` has been in the DOM
        all along, so a reader was handed 18.63 with no way to judge it.
        """
        try:
            fence = build_fence(stats, quantiles)
        except Exception:
            fence = None

        tables = f"<div class='stats-quant'>{stats_table}{quant_stats_table}</div>"
        if fence is None:
            return tables

        name = self.safe_html_escape(stats.name)
        header = (
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · statistics</span>'
            '<span class="fence-head__rule"></span>'
            f'<span class="fence-head__count">{fence.n_total:,} values · '
            f"{self.format_number(fence.value_lo)} to "
            f"{self.format_number(fence.value_hi)}</span>"
            "</div>"
        )
        strip = render_quantile_strip(fence, quantiles, self.format_number)
        prose = self._shape_prose(stats)

        return f'<div class="fence-pane">{header}{strip}{prose}</div>{tables}'

    def _shape_prose(self, stats: NumericStats) -> str:
        """Two sentences spending thresholds the report already carries.

        Both numbers are on the card today with nothing to judge them against:
        the Jarque-Bera statistic is printed bare, and the confidence interval
        is printed as two endpoints rather than as a width.
        """
        lines: list[str] = []

        jb = getattr(stats, "jb_chi2", None)
        if isinstance(jb, (int, float)) and math.isfinite(jb):
            # 5.99 is the 95% critical value of chi-squared with 2 d.f., which
            # is the test's own threshold and is already in the DOM as a
            # `data-threshold` nobody renders.
            verdict = (
                "consistent with a normal distribution"
                if jb < 5.99
                else "far enough from normal to reject it"
            )
            lines.append(
                f"Jarque–Bera is {self.format_number(jb)} against a 5.99 "
                f"critical value — {verdict}."
            )

        ci_lo, ci_hi = getattr(stats, "ci_lo", None), getattr(stats, "ci_hi", None)
        mean = getattr(stats, "mean", None)
        if all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in (ci_lo, ci_hi, mean)
        ):
            half = (float(ci_hi) - float(ci_lo)) / 2.0
            lines.append(
                f"The mean carries a ±{self.format_number(half)} 95% interval, "
                f"so {self.format_number(mean)} is well determined at this "
                "sample size."
            )

        if not lines:
            return ""
        return "".join(f'<p class="qstrip__prose">{line}</p>' for line in lines)

    def _build_stats_table(self, stats: NumericStats) -> str:
        """Build detailed statistics table."""
        # IQR and MAD are derived from the same reservoir as the quartiles, so
        # they inherit the same status. Everything else in this table comes
        # from the streaming moments, which see every value.
        sampled = " (≈)" if quantiles_are_sampled(stats) else ""
        data = [
            ("Mean", self.format_number(stats.mean), "num"),
            ("Std Dev", self.format_number(stats.std), "num"),
            ("Variance", self.format_number(stats.variance), "num"),
            ("Std Error", self.format_number(stats.se), "num"),
            ("Coeff. of Var", self.format_number(stats.cv), "num"),
            ("Geometric mean", self.format_number(stats.gmean), "num"),
            (f"IQR{sampled}", self.format_number(stats.iqr), "num"),
            (f"MAD{sampled}", self.format_number(stats.mad), "num"),
            ("Skew", self.format_number(stats.skew), "num"),
            ("Kurtosis", self.format_number(stats.kurtosis), "num"),
            ("Jarque–Bera χ²", self.format_number(stats.jb_chi2), "num"),
            (
                "95% CI (mean)",
                f"[{self.format_number(stats.ci_lo)} – {self.format_number(stats.ci_hi)}]",
                "num",
            ),
            (
                "Granularity",
                f"{self.safe_html_escape(str(stats.gran_step)) if stats.gran_step is not None else '—'} (decimals: {stats.gran_decimals if stats.gran_decimals is not None else '—'})",
                None,
            ),
            ("Heaping %", self.format_number(stats.heap_pct), "num"),
            # UX-21 / #209. This was the last row of the card's right-hand stat
            # table, directly under `Max` -- six facts about the distribution
            # and then one about the profiler's own bookkeeping, in the position
            # of highest attention on the card. It answers a question about
            # PySuricata, not about the data. It is not useless, so it moves
            # here rather than going away.
            (
                "Processed bytes (≈)",
                self.format_bytes(int(getattr(stats, "mem_bytes", 0))),
                "num",
            ),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_common_values_table(self, stats: NumericStats, col_id: str = "") -> str:
        """Ten rows, three columns, and the finding said out loud.

        Phase 5b.3 (#154). Five columns become three: the ordinals
        `1st 2nd 3rd` are decoration on a list that is already ordered, and
        count and percent are one fact about one value rather than two.

        **The bar is scaled to the most common value, not to 100%.** At 3.2%
        of 714 rows every bar was 3% of its track and all ten looked
        identical, which is a ranking drawn so that the ranking cannot be
        seen. Relative scaling hides absolute rarity in exchange, so the
        caption says which scale it is on.
        """
        try:
            top_values = list(getattr(stats, "top_values", None) or [])[:10]
        except Exception:
            top_values = []

        if not top_values:
            return (
                '<p class="fence-none">No value repeats often enough to be '
                "counted among the most common.</p>"
            )

        name = self.safe_html_escape(stats.name)
        total = max(1, int(getattr(stats, "count", 0) or 0))
        top_count = max(int(count) for _, count in top_values) or 1

        header = (
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · common values</span>'
            '<span class="fence-head__rule"></span>'
            "</div>"
        )

        # Nothing repeats, so there is nothing to rank and no tab to render.
        #
        # Drawing it would scale every bar to the top count of 1 and produce
        # ten identical full-width bars -- a ranking drawn over ten values
        # that are all equally common, which is the relative scale at its
        # worst. `PassengerId` is exactly this column. Saying "no value
        # repeats" instead would only restate the card face, where `Unique`
        # already equals the row count, so 5b.4's rule applies: an empty
        # string here removes the tab.
        if top_count == 1:
            return ""

        rows = []
        for value, count in top_values:
            count = int(count)
            pct = count / total * 100.0
            width = count / top_count * 100.0
            if isinstance(value, float) and value.is_integer():
                shown = f"{int(value):,}"
            else:
                shown = _fmt_compact_scientific(value)
            rows.append(
                f'<div class="common__row">'
                # `data-value` is how the invariance fingerprint sees this.
                # The five-column table it replaces was extracted by pairing
                # the ordinal against the value -- so removing the ordinals,
                # which are decoration on an already-ordered list, would have
                # taken the values with them.
                f'<span class="common__value" data-col="{col_id}" '
                f'data-value="{value:.12g}">{self.safe_html_escape(shown)}</span>'
                f'<span class="common__track">'
                f'<span class="common__bar" style="width:{width:.1f}%"></span></span>'
                f'<span class="common__stat" data-col="{col_id}" '
                f'data-count="{count}" data-pct="{pct:.1f}">'
                f"{count:,} · {pct:.1f}%</span>"
                f"</div>"
            )

        finding = self._heaping_finding(stats, top_values)
        lede = f'<p class="fence-lede">{finding}</p>' if finding else ""

        return (
            f'<div class="fence-pane common">{header}{lede}'
            f'<div class="common__rows">{"".join(rows)}</div>'
            '<p class="common__caption">bar is scaled to the most common '
            "value, not to 100%</p>"
            "</div>"
        )

    def _heaping_finding(self, stats: NumericStats, top_values: list) -> str:
        """Two numbers the report already computes and never puts together.

        `Age` stores three decimals, all ten of its most common values are
        whole numbers, and `Heaping %` is 22.27 — each of those was on the
        card somewhere and the reader had to notice the connection unaided.

        `heap_pct` counts values whose last significant digit is 0 or 5, so
        that is what the sentence says. "Heaped on round numbers" is a gloss
        and this is a report.
        """
        parts: list[str] = []

        decimals = getattr(stats, "gran_decimals", None)
        whole = [
            value
            for value, _ in top_values
            if isinstance(value, (int, float)) and float(value).is_integer()
        ]
        if decimals and len(whole) == len(top_values) and len(top_values) > 1:
            plural = "decimal" if decimals == 1 else "decimals"
            parts.append(
                f"All {len(top_values)} are whole numbers, though the column "
                f"stores {decimals} {plural}."
            )

        heap = getattr(stats, "heap_pct", None)
        if isinstance(heap, (int, float)) and math.isfinite(heap) and heap > 0:
            parts.append(f"{heap:.1f}% of values end in a 0 or a 5.")

        return " ".join(parts)

    def _build_extremes_table(
        self,
        stats: NumericStats,
        quantiles: QuantileData | None = None,
        col_id: str = "",
    ) -> str:
        """The two tails, on the axis that says whether either one is unusual.

        Phase 5b.5 (#154). What this replaces was two tables headed `Min
        values` and `Max values`, five rows each of index and value. Ten
        numbers, no context — and a reader could not tell that **every one of
        `Age`'s five maxima is an outlier and not one of its five minima is**,
        which is the whole story of that column's tails and was already
        computable from the fence.

        So the pane plots both tails on the Outliers pane's axis and gives each
        row its position. `classify` is imported rather than reimplemented: a
        value that reads `high` in one pane cannot read `moderate` in the
        other, and two implementations cannot guarantee that however carefully
        they are written.
        """
        lows = [(i, float(v)) for i, v in (getattr(stats, "min_items", None) or [])]
        highs = [(i, float(v)) for i, v in (getattr(stats, "max_items", None) or [])]
        if not lows and not highs:
            return '<p class="fence-none">No extreme values were tracked.</p>'

        try:
            fence = build_fence(stats, quantiles)
        except Exception:
            fence = None

        if fence is None:
            # No fence means no position to report, so this falls back to the
            # bare listing rather than inventing a verdict.
            return self._build_extremes_listing(lows, highs)

        name = self.safe_html_escape(stats.name)
        low_rows = self._tail_rows(fence, lows)
        high_rows = self._tail_rows(fence, highs)

        marks = cluster_marks(
            fence,
            [(value, classify(fence, value)[0]) for _, value in lows + highs],
        )

        header = (
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · extreme values</span>'
            '<span class="fence-head__rule"></span>'
            f'<span class="fence-head__count">{len(lows)} lowest · '
            f"{len(highs)} highest of {fence.n_total:,}</span>"
            "</div>"
        )
        lede = f'<p class="fence-lede">{self._tails_verdict(low_rows, high_rows)}</p>'
        figure = render_figure(
            fence,
            name,
            self.format_number,
            marks=marks,
            described=(
                f"{name} value axis: {len(lows)} lowest and {len(highs)} highest "
                "against the IQR fence"
            ),
            legend="tails",
        )
        body = (
            '<div class="tails-body">'
            f"{self._tail_column('lowest', low_rows, col_id)}"
            f"{self._tail_column('highest', high_rows, col_id)}"
            "</div>"
        )
        return f'<div class="fence-pane">{header}{lede}{figure}{body}</div>'

    def _tail_rows(self, fence, items: list[tuple]) -> list[dict]:
        """One row per tracked value, with ties marked.

        `Age` holds 0.75 twice and 71 twice, and the pane listed them as
        separate rows without comment -- so the same value looked like two
        findings.
        """
        seen: dict[float, int] = {}
        for _, value in items:
            seen[round(value, 12)] = seen.get(round(value, 12), 0) + 1

        rows = []
        for index, value in items:
            severity, phrase = classify(fence, value)
            ties = seen[round(value, 12)]
            rows.append(
                {
                    "index": str(index),
                    "value": value,
                    "severity": severity,
                    "phrase": phrase,
                    "ties": ties,
                }
            )
        return rows

    def _tails_verdict(self, lows: list[dict], highs: list[dict]) -> str:
        """The sentence the two bare tables never said.

        On `Age`: *The low tail is ordinary — all five sit inside the fence.
        Every one of the five highest crosses it.*

        The asymmetry is the finding, so the both-quiet case gets one clause
        rather than two. Saying `all 5 sit inside the fence` twice reads as a
        template that did not notice it was describing the same thing.
        """
        low_beyond = sum(1 for row in lows if row["severity"] != "inside")
        high_beyond = sum(1 for row in highs if row["severity"] != "inside")
        tracked = len(lows) + len(highs)

        if not tracked:
            return "No extreme values were tracked for this column."

        if not (low_beyond or high_beyond):
            return (
                f"Neither tail is unusual: all {tracked} tracked values sit "
                "inside the fence."
            )

        def describe(rows: list[dict], beyond: int, side: str) -> str:
            if beyond == 0:
                return f"all {len(rows)} sit inside the fence"
            if beyond == len(rows):
                return f"every one of the {len(rows)} {side} crosses it"
            return f"{beyond} of {len(rows)} cross it"

        if not lows:
            return f"The high tail: {describe(highs, high_beyond, 'highest')}."
        if not highs:
            return f"The low tail: {describe(lows, low_beyond, 'lowest')}."

        low = describe(lows, low_beyond, "lowest")
        high = describe(highs, high_beyond, "highest")
        opening = "The low tail is ordinary — " if low_beyond == 0 else "The low tail: "
        return f"{opening}{low}. {high[0].upper()}{high[1:]}."

    def _tail_column(self, side: str, rows: list[dict], col_id: str = "") -> str:
        if not rows:
            return ""
        beyond = sum(1 for row in rows if row["severity"] != "inside")
        if beyond == 0:
            summary, tone = "none beyond a fence", "good"
        elif beyond == len(rows):
            summary, tone = f"all {beyond} beyond the IQR fence", "warn"
        else:
            summary, tone = f"{beyond} beyond the IQR fence", "warn"

        lines = "".join(
            f'<div class="tails__row">'
            f'<span class="tails__idx">{self.safe_html_escape(row["index"])}</span>'
            f'<span class="tails__val" data-col="{col_id}" '
            f'data-value="{row["value"]:.12g}">{self.format_number(row["value"])}'
            + (
                f'<span class="tails__tie" title="this value appears '
                f'{row["ties"]} times in the tail">×{row["ties"]}</span>'
                if row["ties"] > 1
                else ""
            )
            + "</span>"
            f'<span class="tails__note" data-severity="{row["severity"]}">'
            f"{row['phrase']}</span>"
            f"</div>"
            for row in rows
        )
        return (
            '<div class="tails__col">'
            f'<div class="tails__head"><span class="tails__side">{len(rows)} {side}</span>'
            f'<span class="tails__summary" data-tone="{tone}">{summary}</span></div>'
            f"{lines}</div>"
        )

    def _build_extremes_listing(self, lows: list[tuple], highs: list[tuple]) -> str:
        """The bare tails, for a column with no fence to place them against."""

        def column(side: str, items: list[tuple]) -> str:
            if not items:
                return ""
            lines = "".join(
                f'<div class="tails__row">'
                f'<span class="tails__idx">{self.safe_html_escape(str(index))}</span>'
                f'<span class="tails__val">{self.format_number(value)}</span>'
                f'<span class="tails__note" data-severity="inside"></span>'
                f"</div>"
                for index, value in items
            )
            return (
                '<div class="tails__col">'
                f'<div class="tails__head"><span class="tails__side">'
                f"{len(items)} {side}</span></div>{lines}</div>"
            )

        return (
            '<div class="fence-pane"><div class="tails-body">'
            f"{column('lowest', lows)}{column('highest', highs)}</div></div>"
        )

    def _build_outliers_pane(
        self,
        stats: NumericStats,
        quantiles: QuantileData | None = None,
        col_id: str = "",
    ) -> str:
        """The fence, the marks that crossed it, and one row per value.

        Phase 5b.2 (#154). What this replaces opened with roughly 60px
        announcing `Low Outliers — 0 outliers (0.0%)` over three severity chips
        all reading zero, said it again for the high side, then listed the
        values in a `rowspan` table with no picture of what they crossed.

        An outlier is *defined* by a threshold, so the threshold is the one
        graphic that explains the number, and it is drawn. The empty low side
        becomes a sentence: `Age`'s lower fence sits below the column's own
        minimum, which is the reason it has no low outliers and is worth more
        than a block of zeroes.

        The arithmetic lives in `render/outlier_fence.py`, because the Min/Max
        pane (5b.5) has to read the same axis and the same severity words -- a
        value that is `high` in one pane cannot be `moderate` in the other, and
        the only way to guarantee that is one implementation.
        """
        try:
            fence = build_fence(stats, quantiles)
        except Exception:
            fence = None

        if fence is None:
            return (
                '<p class="fence-none">No fence can be placed on this column: '
                "the middle half of its values are identical, so the IQR is "
                "zero and the rule has no width to work with.</p>"
            )

        name = self.safe_html_escape(stats.name)
        pct = (fence.n_outliers / fence.n_total * 100.0) if fence.n_total else 0.0

        header = (
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · outliers</span>'
            '<span class="fence-head__rule"></span>'
            f'<span class="fence-head__count" data-col="{col_id}" '
            f'data-count="{fence.n_outliers}" data-pct="{pct:.1f}">'
            f"{fence.n_outliers:,} of {fence.n_total:,} values · {pct:.1f}%</span>"
            "</div>"
        )
        lede = f'<p class="fence-lede">{fence_verdict(fence, self.format_number)}</p>'

        if not fence.rows:
            # Nothing crossed either fence. The sentence above already names
            # both, so a figure with no marks on it would be decoration.
            return f'<div class="fence-pane">{header}{lede}</div>'

        body = (
            '<div class="fence-body">'
            f"{render_table(fence, self.format_number, col_id)}"
            '<div class="fence-methods">'
            '<span class="fence-methods__title">The two methods</span>'
            f'<p class="fence-methods__note">{method_note(fence)}</p>'
            "</div>"
            "</div>"
        )

        figure = render_figure(fence, name, self.format_number)
        return f'<div class="fence-pane">{header}{lede}{figure}{body}</div>'

    #: The pane lists at most this many partners. Beyond it the list stops
    #: informing and starts scrolling -- a 40-column frame would render 39 rows
    #: inside a card.
    _MAX_PARTNERS = 5

    def _build_correlation_table(self, stats: NumericStats) -> str:
        """Every partner this column has, strongest first, capped at five.

        Phase 5b.6 (#154). The pane repeated the section-level empty state
        inside a card: `No significant correlations found`, on a column that
        has partners and simply has no *strong* ones.

        `Age` has exactly two numeric partners in the Titanic frame, so listing
        both is **complete** information in two rows -- nothing is withheld and
        the reader can stop wondering. "Both partners are weak, the stronger is
        Fare at +0.096" is a finding; "no significant correlations" is a shrug
        that leaves a reader unable to tell an uncorrelated column from one the
        threshold happened to hide.

        The bar is the section's own `_diverging_bar` shape, so sign stays
        position and never colour -- a red bar for a negative correlation reads
        as *bad*, and a negative correlation is often the interesting one.
        """
        partners = list(getattr(stats, "corr_top", None) or [])
        if not partners:
            return ""

        partners.sort(key=lambda pair: abs(pair[1]), reverse=True)
        threshold = float(getattr(stats, "corr_threshold", 0.5) or 0.0)

        # `corr_max_per_col` is documented as a maximum, so it still binds --
        # it just cannot push the list past the point where it stops being
        # readable inside a card.
        limit = min(self._MAX_PARTNERS, len(partners))
        shown, hidden = partners[:limit], partners[limit:]

        name = self.safe_html_escape(stats.name)
        strongest, value = shown[0]
        if abs(value) < threshold:
            lede = (
                f"Every partner is weak. The strongest is "
                f"{self.safe_html_escape(str(strongest))} at {value:+.3f}, "
                f"below the {threshold:.2f} threshold."
            )
        else:
            lede = (
                f"The strongest partner is "
                f"{self.safe_html_escape(str(strongest))} at {value:+.3f}."
            )

        rows = "".join(
            f'<div class="corr-partner">'
            f'<span class="corr-partner__name">{self.safe_html_escape(str(other))}</span>'
            f"{self._diverging_bar(corr)}"
            f'<span class="corr-partner__value">{corr:+.3f}</span>'
            f"</div>"
            for other, corr in shown
        )

        # Completeness is the point of this pane, so it says which case it is
        # in. A list that stops at five and a list that *is* the whole set look
        # identical, and only one of them lets a reader stop wondering.
        if hidden:
            note = f"{len(hidden)} more, all below {abs(shown[-1][1]):.2f}."
        elif len(shown) == 1:
            note = "That is this column's only numeric partner, so nothing is withheld."
        else:
            note = (
                f"Those are all {len(shown)} of this column's numeric partners, "
                "so nothing is withheld."
            )
        more = f'<p class="corr-partner__more">{note}</p>'

        return (
            '<div class="fence-pane">'
            '<div class="fence-head">'
            f'<span class="fence-head__title">{name} · correlations</span>'
            '<span class="fence-head__rule"></span>'
            f'<span class="fence-head__count">{len(partners)} numeric '
            f"partner{'s' if len(partners) != 1 else ''}</span>"
            "</div>"
            f'<p class="fence-lede">{lede}</p>'
            f'<div class="corr-partners">{rows}</div>'
            f"{more}"
            "</div>"
        )

    def _diverging_bar(self, corr: float) -> str:
        """Zero at the centre, negative left, positive right.

        Byte-identical in shape to `correlations_section._diverging_bar`, and
        deliberately so: the per-column pane and the section-level list plot
        the same numbers, and a reader who learns to read one must not have to
        relearn the other.
        """
        magnitude = min(abs(corr), 1.0) * 50.0
        left = 50.0 - magnitude if corr < 0 else 50.0
        return (
            '<span class="corr-bar" aria-hidden="true">'
            '<span class="corr-bar__zero"></span>'
            f'<span class="corr-bar__fill" style="left:{left:.2f}%;'
            f'width:{magnitude:.2f}%;background:var(--data-2)"></span>'
            "</span>"
        )

    def _build_missing_values_table(self, stats: NumericStats) -> str:
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

    def _build_dataprep_spectrum_visualization(self, stats: NumericStats) -> str:
        """Build DataPrep-style spectrum visualization for missing values per chunk.

        This creates a single horizontal bar with segments representing actual processing
        chunks, colored by missing value density (green-yellow-red gradient).

        Args:
            stats: NumericStats object containing chunk metadata and missing data information

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
        elif max_missing_pct >= 20:
            severity = "high"
        elif max_missing_pct >= 5:
            severity = "medium"
        else:
            severity = "low"

        return f"""
        <div class="dataprep-spectrum">
            <div class="spectrum-header">
                <span class="spectrum-title">Missing Values per Chunk</span>
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

    def _generate_missing_insights(
        self, chunk_data: list[dict], overall_missing_pct: float
    ) -> dict:
        """Generate insights about missing value patterns.

        Args:
            chunk_data: List of chunk data dictionaries
            overall_missing_pct: Overall missing percentage

        Returns:
            Dictionary containing insights and pattern analysis
        """
        if not chunk_data:
            return {}

        missing_pcts = [chunk["missing_pct"] for chunk in chunk_data]
        max_missing_pct = max(missing_pcts)
        min_missing_pct = min(missing_pcts)
        avg_missing_pct = sum(missing_pcts) / len(missing_pcts)

        # Identify problematic chunks
        high_missing_chunks = [
            chunk for chunk in chunk_data if chunk["missing_pct"] > 20
        ]
        low_missing_chunks = [chunk for chunk in chunk_data if chunk["missing_pct"] < 2]

        # Pattern detection
        patterns = []
        if max_missing_pct - min_missing_pct > 30:
            patterns.append("High variability in missing values across chunks")
        if len(high_missing_chunks) > len(chunk_data) * 0.3:
            patterns.append("Multiple chunks with high missing values")
        if len(low_missing_chunks) > len(chunk_data) * 0.5:
            patterns.append("Most chunks have low missing values")

        # Severity assessment
        if max_missing_pct > 50:
            severity = "critical"
        elif max_missing_pct > 20:
            severity = "high"
        elif max_missing_pct > 5:
            severity = "medium"
        else:
            severity = "low"

        return {
            "overall_missing_pct": overall_missing_pct,
            "max_missing_pct": max_missing_pct,
            "min_missing_pct": min_missing_pct,
            "avg_missing_pct": avg_missing_pct,
            "high_missing_chunks": len(high_missing_chunks),
            "low_missing_chunks": len(low_missing_chunks),
            "patterns": patterns,
            "severity": severity,
            "total_chunks": len(chunk_data),
        }

    def _render_chunk_visualization(
        self, chunk_data: list[dict], insights: dict, stats: NumericStats
    ) -> str:
        """Render the complete chunk visualization.

        Args:
            chunk_data: List of chunk data dictionaries
            insights: Dictionary containing insights and patterns
            stats: NumericStats object

        Returns:
            HTML string for the complete visualization
        """
        if not chunk_data:
            return ""

        # Build chunk bars
        chunk_bars = ""
        max_missing = max(chunk["missing"] for chunk in chunk_data) if chunk_data else 0

        for chunk in chunk_data:
            # Determine severity class
            if chunk["missing_pct"] > 20:
                severity_class = "high"
            elif chunk["missing_pct"] > 5:
                severity_class = "medium"
            else:
                severity_class = "low"

            # Calculate bar width
            bar_width = (
                (chunk["missing"] / max_missing) * 100.0 if max_missing > 0 else 0
            )

            chunk_bars += f"""
            <div class="chunk-bar-item" data-chunk="{chunk["index"]}">
                <div class="chunk-info">
                    <span class="chunk-label">Chunk {chunk["index"]}</span>
                    <span class="chunk-stats">
                        {chunk["missing"]:,} missing ({chunk["missing_pct"]:.1f}%)
                    </span>
                    <span class="chunk-size">Size: {chunk["size"]:,}</span>
                </div>
                <div class="chunk-bar-container">
                    <div class="chunk-bar-fill {severity_class}"
                         style="width: {bar_width:.1f}%"
                         title="Chunk {chunk["index"]}: {chunk["missing"]:,} missing values ({chunk["missing_pct"]:.1f}%)">
                    </div>
                </div>
            </div>"""

        # Build insights section
        insights_html = ""
        if insights.get("patterns"):
            insights_html = f"""
            <div class="chunk-insights">
                <h5>Pattern Analysis</h5>
                <ul class="insights-list">
                    {"".join(f"<li>{pattern}</li>" for pattern in insights["patterns"])}
                </ul>
            </div>"""

        # Build summary statistics
        summary_html = f"""
        <div class="chunk-summary">
            <div class="summary-stats">
                <span class="stat-item">
                    <span class="stat-label">Total Chunks:</span>
                    <span class="stat-value">{insights.get("total_chunks", 0)}</span>
                </span>
                <span class="stat-item">
                    <span class="stat-label">Max Missing:</span>
                    <span class="stat-value">{insights.get("max_missing_pct", 0):.1f}%</span>
                </span>
                <span class="stat-item">
                    <span class="stat-label">Avg Missing:</span>
                    <span class="stat-value">{insights.get("avg_missing_pct", 0):.1f}%</span>
                </span>
                <span class="stat-item severity-{insights.get("severity", "low")}">
                    <span class="stat-label">Severity:</span>
                    <span class="stat-value">{insights.get("severity", "low").title()}</span>
                </span>
            </div>
        </div>"""

        return f"""
        <div class="missing-per-chunk-enhanced">
            <div class="chunk-header">
                <span class="title">Missing Values per Chunk</span>
                <span class="overall-stats">
                    {stats.missing:,} missing ({insights.get("overall_missing_pct", 0):.1f}% overall)
                </span>
            </div>

            <div class="chunk-visualization">
                <div class="chunk-bars">
                    {chunk_bars}
                </div>

                {summary_html}
                {insights_html}
            </div>

            <div class="chunk-legend">
                <div class="legend-item">
                    <span class="legend-color low"></span>
                    <span class="legend-label">Low (0-5%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color medium"></span>
                    <span class="legend-label">Medium (5-20%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color high"></span>
                    <span class="legend-label">High (20%+)</span>
                </div>
            </div>
        </div>
        """

    def _get_missing_data_severity(self, missing_pct: float) -> tuple[str, str, str]:
        """Get missing data severity classification with clear thresholds.

        Args:
            missing_pct: Percentage of missing data

        Returns:
            Tuple of (severity_class, label, icon)
        """
        if missing_pct >= 50:
            return "critical", "Critical", ""
        elif missing_pct >= 20:
            return "high", "High", ""
        elif missing_pct >= 5:
            return "medium", "Medium", ""
        else:
            return "low", "Low", ""

    def _build_quality_indicators(
        self,
        stats: NumericStats,
        missing_pct: float,
        zeros_pct: float,
        inf_pct: float,
        neg_pct: float,
        quality_severity: str,
    ) -> list[dict]:
        """Build quality indicators list with efficient logic.

        Args:
            stats: NumericStats object
            missing_pct: Missing data percentage
            zeros_pct: Zero values percentage
            inf_pct: Infinite values percentage
            neg_pct: Negative values percentage
            quality_severity: Overall quality severity

        Returns:
            List of quality indicator dictionaries
        """
        indicators = []

        # Missing data indicator (always present)
        indicators.append(
            {
                "label": "Missing Data",
                "value": f"{stats.missing:,} ({missing_pct:.1f}%)",
                "severity": quality_severity,
                "description": "Values that are completely absent",
            }
        )

        # Zero values indicator (only if present)
        if stats.zeros > 0:
            zero_severity = (
                "high" if zeros_pct >= 20 else ("medium" if zeros_pct >= 5 else "low")
            )
            indicators.append(
                {
                    "label": "Zero Values",
                    "value": f"{stats.zeros:,} ({zeros_pct:.1f}%)",
                    "severity": zero_severity,
                    "description": "Values equal to zero",
                }
            )

        # Infinite values indicator (only if present)
        if stats.inf > 0:
            indicators.append(
                {
                    "label": "Infinite Values",
                    "value": f"{stats.inf:,} ({inf_pct:.1f}%)",
                    "severity": "critical",
                    "symbol": "∞",
                    "description": "Values that are infinite",
                }
            )

        # Negative values indicator (only if present)
        if stats.negatives > 0:
            neg_severity = "medium" if neg_pct >= 10 else "low"
            indicators.append(
                {
                    "label": "Negative Values",
                    "value": f"{stats.negatives:,} ({neg_pct:.1f}%)",
                    "severity": neg_severity,
                    "description": "Values less than zero",
                }
            )

        return indicators

    def _build_indicators_html(self, quality_indicators: list[dict]) -> str:
        """Build quality indicators HTML with efficient string building.

        Args:
            quality_indicators: List of quality indicator dictionaries

        Returns:
            HTML string for quality indicators
        """
        indicators_html = """
        <div class="quality-indicators">
            <div class="indicators-header">
                <span class="title">Data Quality Indicators</span>
            </div>
            <div class="indicators-grid">
        """

        # Use efficient string building with join
        indicator_items = []
        for indicator in quality_indicators:
            indicator_items.append(f"""
                <div class="indicator-item {indicator["severity"]}">
                    <div class="indicator-icon">{indicator.get("symbol", "")}</div>
                    <div class="indicator-content">
                        <div class="indicator-label">{indicator["label"]}</div>
                        <div class="indicator-value">{indicator["value"]}</div>
                        <div class="indicator-description">{indicator["description"]}</div>
                    </div>
                </div>
            """)

        indicators_html += "".join(indicator_items)
        indicators_html += """
            </div>
        </div>
        """

        return indicators_html

    def _build_recommendations(
        self, stats: NumericStats, missing_pct: float, zeros_pct: float, inf_pct: float
    ) -> list[dict]:
        """Build recommendations list with efficient logic.

        Args:
            stats: NumericStats object
            missing_pct: Missing data percentage
            zeros_pct: Zero values percentage
            inf_pct: Infinite values percentage

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Missing data recommendations
        if missing_pct >= 50:
            recommendations.append(
                {
                    "severity": "critical",
                    "title": "Consider Data Collection Review",
                    "description": "Over 50% missing data suggests fundamental data collection issues",
                }
            )
        elif missing_pct >= 20:
            recommendations.append(
                {
                    "severity": "high",
                    "title": "Investigate Missing Data Patterns",
                    "description": "High missing data rate may indicate systematic issues",
                }
            )
        elif missing_pct >= 5:
            recommendations.append(
                {
                    "severity": "medium",
                    "title": "Monitor Data Quality",
                    "description": "Moderate missing data - consider imputation strategies",
                }
            )
        else:
            recommendations.append(
                {
                    "severity": "low",
                    "title": "Good Data Quality",
                    "description": "Low missing data rate indicates good data collection",
                }
            )

        # Infinite values recommendations
        if stats.inf > 0:
            recommendations.append(
                {
                    "severity": "critical",
                    "title": "Handle Infinite Values",
                    "description": "Infinite values need special handling before analysis",
                }
            )

        # Zero inflation recommendations
        if zeros_pct >= 20:
            recommendations.append(
                {
                    "severity": "medium",
                    "title": "Consider Zero Inflation",
                    "description": "High zero percentage may indicate zero-inflated distribution",
                }
            )

        return recommendations

    def _build_recommendations_html(self, recommendations: list[dict]) -> str:
        """Build recommendations HTML with efficient string building.

        Args:
            recommendations: List of recommendation dictionaries

        Returns:
            HTML string for recommendations
        """
        recommendations_html = """
        <div class="recommendations">
            <div class="recommendations-header">
                <span class="title">Recommendations</span>
            </div>
            <div class="recommendations-list">
        """

        # Use efficient string building with join
        recommendation_items = []
        for rec in recommendations:
            recommendation_items.append(f"""
                <div class="recommendation-item {rec["severity"]}">
                    <div class="recommendation-title">{rec["title"]}</div>
                    <div class="recommendation-description">{rec["description"]}</div>
                </div>
            """)

        recommendations_html += "".join(recommendation_items)
        recommendations_html += """
            </div>
        </div>
        """

        return recommendations_html

    #: What each pane is worth opening for, in the order the tabs appear.
    #: Order is fixed so a tab never moves; it only appears or does not.
    def _pane_counts(self, stats: NumericStats) -> dict[str, str]:
        """The figure beside each tab, and in the closed strip.

        Every one of these is already on `stats`. The pane was rendering them
        and the reader could not see any of it without opening all six.
        """
        counts: dict[str, str] = {}

        top_values = list(getattr(stats, "top_values", None) or [])
        # A column where nothing repeats has no common values, whatever the
        # length of the top-k list -- every entry in it was seen once.
        if top_values and max(int(count) for _, count in top_values) > 1:
            counts["common"] = f"{len(top_values):,}"

        lows = len(getattr(stats, "min_items", None) or [])
        highs = len(getattr(stats, "max_items", None) or [])
        if lows or highs:
            counts["extremes"] = f"{max(lows, highs):,}"

        outliers = int(getattr(stats, "outliers_iqr", 0) or 0)
        if outliers:
            counts["outliers"] = f"{outliers:,}"

        partners = len(getattr(stats, "corr_top", None) or [])
        if partners:
            counts["corr"] = f"{partners:,}"

        missing = int(getattr(stats, "missing", 0) or 0)
        total = missing + int(getattr(stats, "count", 0) or 0)
        if missing and total:
            counts["missing"] = f"{missing / total * 100:.1f}%"

        return counts

    #: How each pane reads in the closed strip. `11 outliers` beside the button
    #: is the reason to open it; its absence is the reason not to.
    _PANE_NOUNS = {
        "stats": "statistics",
        "common": "common values",
        "extremes": "lowest and highest",
        "outliers": "outliers",
        "corr": "correlations",
        "missing": "missing",
    }

    def _build_details_section(
        self,
        col_id: str,
        stats: NumericStats,
        stats_quantiles: str,
        common_table: str,
        extremes_table: str,
        outliers_pane: str,
        corr_table: str,
        missing_table: str,
    ) -> tuple[str, str]:
        """Details tabs in a fixed order, minus the ones with nothing to say.

        Returns the section and the one-line summary that goes beside the
        closed `Details` button.

        Two panes are gated:

        **Correlations** repeated the section-level empty state inside a card.

        **Missing Values** rendered on every column -- including ones with no
        missing values, where it drew a 100%-present bar and a one-segment
        chunk strip reading 0.0%. It is now gated on *missing > 0 and more than
        one chunk*, which is the only condition under which it knows something
        the card face does not: **where in the read the gaps fall**. With one
        chunk it restates a percentage the header already carries, four times
        over. On a single-chunk frame every numeric card loses a tab and
        nothing goes with it.
        """
        chunks = len(getattr(stats, "chunk_metadata", None) or [])
        has_missing = int(getattr(stats, "missing", 0) or 0) > 0

        panes = [
            ("stats", "Statistics", stats_quantiles, bool(stats_quantiles.strip())),
            ("common", "Common values", common_table, bool(common_table.strip())),
            (
                "extremes",
                "Min/Max Values",
                extremes_table,
                bool(extremes_table.strip()),
            ),
            ("outliers", "Outliers", outliers_pane, True),
            (
                "corr",
                "Correlations",
                f'<div class="sub">{corr_table}</div>',
                bool(getattr(stats, "corr_top", None)),
            ),
            (
                "missing",
                "Missing Values",
                f'<div class="sub">{missing_table}</div>',
                has_missing and chunks > 1,
            ),
        ]

        counts = self._pane_counts(stats)
        html = self._build_tabbed_details(col_id, panes, counts)
        return html, self._summarise_panes(panes, counts)

    def _summarise_panes(
        self, panes: list[tuple[str, str, str, bool]], counts: dict[str, str]
    ) -> str:
        """`statistics · 10 common values · 11 outliers · 2 correlations`.

        The tab set is known at render time and was not printed, so `Details`
        promised nothing and a reader had to open every card to learn whether
        opening was worth it.
        """
        parts = []
        for key, _, _, worth in panes:
            if not worth:
                continue
            noun = self._PANE_NOUNS.get(key, key)
            badge = counts.get(key)
            if badge:
                parts.append(f"{badge} {noun}")
            elif key == "outliers":
                # Zero is the informative case for this one, and it is the
                # reason *not* to open the pane -- half of what the strip is
                # for. It reads as a phrase here and would be nonsense as a tab
                # badge, where `Outliers no` was the first attempt.
                parts.append("no outliers")
            else:
                parts.append(noun)
        return " \u00b7 ".join(parts)

    def _build_controls_section(
        self, col_id: str, log_default: bool = False, pane_summary: str = ""
    ) -> str:
        """Build controls section.

        Args:
            col_id: Sanitised column id, used for the details toggle target.
            log_default: Whether the log-scale heuristic fired for this column.
                When it did, the chart opens on a log axis. The card was
                computing the right answer and then drawing the wrong picture:
                a lognormal column labelled *Log-scale?* rendered on a linear
                axis is one bar at the left edge, which teaches the reader that
                the chips are cosmetic.
        """
        bin_buttons = " ".join(
            f'<button type="button" class="btn-soft{" active" if b == 25 else ""}" data-bin="{b}">{b}</button>'
            for b in self.hist_config.bin_options
        )
        scale = "log" if log_default else "lin"
        lin_active = "" if log_default else " active"
        log_active = " active" if log_default else ""

        # What is behind the button. Without it "Details" promises nothing, so
        # a reader opens every card to find out whether opening was worth it.
        summary_html = (
            f'<span class="details-panes">{self.safe_html_escape(pane_summary)}</span>'
            if pane_summary
            else ""
        )

        return f"""
        <div class="card-controls" role="group" aria-label="Numeric controls">
            <div class="details-slot">
                <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
                {summary_html}
            </div>
            <div class="controls-slot">
                <div class="hist-controls" data-scale="{scale}" data-bin="25">
                    <div class="center-controls">
                        <span>Scale:</span>
                        <div class="scale-group">
                            <button type="button" class="btn-soft{lin_active}" data-scale="lin">Linear</button>
                            <button type="button" class="btn-soft{log_active}" data-scale="log">Log</button>
                        </div>
                        <span>Bins:</span>
                        <div class="bin-group">{bin_buttons}</div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _assemble_card(
        self,
        col_id: str,
        safe_name: str,
        stats: NumericStats,
        approx_badge: str,
        quality_flags_html: str,
        stat_row: str,
        chart_html: str,
        details_html: str,
        controls_html: str,
    ) -> str:
        """Assemble the complete card HTML."""
        # A key is not a measurement: saying so in the badge is the whole point
        # of detecting it. Every competing profiler prints the mean of an ID
        # column; being the one that names it instead costs nothing.
        badge = "Identifier" if looks_like_identifier(stats) else "Numeric"
        docs_url = "https://alvarodiez20.github.io/pysuricata/stats/numeric/"
        info_button = f'''<a href="{docs_url}" target="_blank" rel="noopener noreferrer" class="info-link" title="View documentation for Numeric analysis" aria-label="View Numeric analysis documentation">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path fill="currentColor" d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM6.5 5a.75.75 0 0 0 0 1.5h.5v2.5h-.5a.75.75 0 0 0 0 1.5h3a.75.75 0 0 0 0-1.5h-.5V6h-.5A.75.75 0 0 0 8 5.25H6.5zM8 3.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5z"/>
            </svg>
        </a>'''

        return f"""
        <article class="var-card" id="{col_id}">
            <header class="var-card__header">
                <div class="title">
                    <span class="colname" title="{safe_name}">{safe_name}</span>
                    <span class="badge">{badge}</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {approx_badge}
                    {quality_flags_html}
                </div>
                {info_button}
            </header>
            <div class="var-card__body">
                <!-- Restacked (#114). The chart was one third of a row beside
                     two 240px stat tables; full width it gains about 550px,
                     which is what makes 50 bins legible and the log toggle
                     worth having. The stats follow it as one row rather than
                     two columns beside it. -->
                <div class="var-chart">{chart_html}</div>
                {controls_html}
                {stat_row}
                {details_html}
            </div>
        </article>
        """
