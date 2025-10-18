"""Numeric card rendering functionality."""

import math
from collections.abc import Sequence

from .card_base import CardRenderer, QualityAssessor, TableBuilder
from .card_config import (
    DEFAULT_CHART_DIMS,
    DEFAULT_HIST_CONFIG,
    DEFAULT_TICK_CONFIG,
    MAD_OUTLIER_THRESHOLD,
    MAD_SCALE_FACTOR,
)
from .card_types import NumericStats, QualityFlags, QuantileData
from .format_utils import fmt_compact_scientific as _fmt_compact_scientific
from .histogram_svg import SVGHistogramRenderer


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


class NumericCardRenderer(CardRenderer):
    """Renders numeric data cards."""

    def __init__(self):
        # Initialize SVG histogram renderer
        self.svg_histogram_renderer = SVGHistogramRenderer()
        super().__init__()
        self.quality_assessor = QualityAssessor()
        self.table_builder = TableBuilder()
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

        left_table = self._build_left_table(stats, percentages)
        right_table = self._build_right_table(stats)

        quantiles = self._compute_quantiles_from_sample(stats.sample_vals or [])
        quant_stats_table = self._build_quant_stats_table(stats, quantiles)

        chart_html = self._build_histogram_variants(col_id, safe_name, stats)

        stats_table = self._build_stats_table(stats)
        common_table = self._build_common_values_table(stats)
        extremes_table = self._build_extremes_table(stats)
        outliers_low, outliers_high = self._build_outliers_tables(stats)
        corr_table = self._build_correlation_table(stats)
        missing_table = self._build_missing_values_table(stats)

        stats_quantiles = (
            f"<div class='stats-quant'>{stats_table}{quant_stats_table}</div>"
        )

        details_html = self._build_details_section(
            col_id,
            stats_quantiles,
            common_table,
            extremes_table,
            outliers_low,
            outliers_high,
            corr_table,
            missing_table,
        )

        controls_html = self._build_controls_section(col_id)

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

    def _build_approx_badge(self, approx: bool) -> str:
        """Build approximation badge if needed."""
        return '<span class="badge">approx</span>' if approx else ""

    def _build_quality_flags_html(
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
            unique_ratio = getattr(stats, "unique_ratio_approx", 0)
            flag_items.append(
                f'<li class="flag warn" data-threshold="Low unique count" data-value="{unique_ratio:.1%}">Discrete</li>'
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
            unique_ratio = getattr(stats, "unique_ratio_approx", 0)
            flag_items.append(
                f'<li class="flag warn" data-threshold="Very low cardinality" data-value="{unique_ratio:.1%}">Quasi‑constant</li>'
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

    def _build_left_table(self, stats: NumericStats, percentages: dict) -> str:
        """Build left statistics table."""
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

        return self.table_builder.build_key_value_table(data)

    def _build_right_table(self, stats: NumericStats) -> str:
        """Build right statistics table."""
        mem_display = self.format_bytes(int(getattr(stats, "mem_bytes", 0)))

        data = [
            ("Min", self.format_number(stats.min), "num"),
            ("Q1 (P25)", self.format_number(stats.q1), "num"),
            ("Median", self.format_number(stats.median), "num"),
            ("Mean", self.format_number(stats.mean), "num"),
            ("Q3 (P75)", self.format_number(stats.q3), "num"),
            ("Max", self.format_number(stats.max), "num"),
            ("Processed bytes (≈)", mem_display, "num"),
        ]

        return self.table_builder.build_key_value_table(data)

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
            ("Std Dev", self.format_number(stats.std), "num"),
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

    def _build_stats_table(self, stats: NumericStats) -> str:
        """Build detailed statistics table."""
        data = [
            ("Mean", self.format_number(stats.mean), "num"),
            ("Std Dev", self.format_number(stats.std), "num"),
            ("Variance", self.format_number(stats.variance), "num"),
            ("Std Error", self.format_number(stats.se), "num"),
            ("Coeff. of Var", self.format_number(stats.cv), "num"),
            ("Geometric mean", self.format_number(stats.gmean), "num"),
            ("IQR", self.format_number(stats.iqr), "num"),
            ("MAD", self.format_number(stats.mad), "num"),
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
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_common_values_table(self, stats: NumericStats) -> str:
        """Build common values table with enhanced formatting and functionality.

        This method creates a professional, feature-rich table that provides
        comprehensive insights into the most frequent values in the dataset.

        Args:
            stats: NumericStats object containing the data

        Returns:
            HTML string for the enhanced common values table
        """
        try:
            top_values = list(getattr(stats, "top_values", []) or [])
        except Exception:
            top_values = []

        if not top_values:
            return '<div class="muted">No common values to display</div>'

        rows = []
        total_nonnull = max(1, int(getattr(stats, "count", 0)))

        # Take only top 10 values for better display and performance
        top_values = top_values[:10]

        for i, (v, c) in enumerate(top_values):
            pct = (int(c) / total_nonnull) * 100.0 if total_nonnull else 0.0

            # Add ranking indicator for top values
            rank_icon = ordinal_number(i + 1)

            # Format value with appropriate precision and scientific notation for large numbers
            if isinstance(v, float) and v.is_integer():
                formatted_value = f"{int(v):,}"
            else:
                formatted_value = _fmt_compact_scientific(v)

            rows.append(
                f"<tr class='common-row rank-{i + 1}'>"
                f"<td class='rank'>{rank_icon}</td>"
                f"<td class='num common-value'>{formatted_value}</td>"
                f"<td class='num common-count'>{int(c):,}</td>"
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

    def _build_extremes_table(self, stats: NumericStats) -> str:
        """Build extremes table."""

        def _sub(label: str, items: list) -> str:
            if not items:
                return f"<div class='sub'><div class='hdr'>{label}</div><div class='muted'>—</div></div>"
            rows = "".join(
                f"<tr><td>{self.safe_html_escape(str(idx))}</td><td class='num'>{self.format_number(val)}</td></tr>"
                for idx, val in items
            )
            return (
                f"<div class='sub'><div class='hdr'>{label}</div>"
                f"<table class='kv'><thead><tr><th>Index</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table></div>"
            )

        return (
            "<div class='extremes stats-quant'>"
            + _sub("Min values", list(getattr(stats, "min_items", []) or []))
            + _sub("Max values", list(getattr(stats, "max_items", []) or []))
            + "</div>"
        )

    def _build_outliers_tables(self, stats: NumericStats) -> tuple[str, str]:
        """Build outliers tables."""
        try:
            sample_vals = list(getattr(stats, "sample_vals", []) or [])
        except Exception:
            sample_vals = []

        out_tbl_low = out_tbl_high = "<div class='muted'>—</div>"

        try:
            low_list, high_list = self._identify_outliers(stats, sample_vals)
            idx_map = self._build_index_map(stats)

            low_list = sorted(self._deduplicate_outliers(low_list), key=lambda x: x[0])[
                :10
            ]
            high_list = sorted(
                self._deduplicate_outliers(high_list), key=lambda x: -x[0]
            )[:10]

            # Use enhanced outliers table with summary statistics and visual improvements
            out_tbl_low = self._format_enhanced_outliers_table(
                low_list, idx_map, stats, "low"
            )
            out_tbl_high = self._format_enhanced_outliers_table(
                high_list, idx_map, stats, "high"
            )
        except Exception:
            pass

        return out_tbl_low, out_tbl_high

    def _identify_outliers(
        self, stats: NumericStats, sample_vals: list
    ) -> tuple[list, list]:
        """Identify outliers using IQR and MAD methods."""
        low_list = []
        high_list = []

        # IQR method
        if isinstance(stats.q1, (int, float)) and isinstance(stats.q3, (int, float)):
            iqr = stats.q3 - stats.q1
            if iqr and not math.isnan(iqr):
                lo_f, hi_f = stats.q1 - 1.5 * iqr, stats.q3 + 1.5 * iqr
                for v in sample_vals:
                    if not isinstance(v, (int, float)) or not math.isfinite(v):
                        continue
                    if v < lo_f:
                        low_list.append((v, "IQR"))
                    elif v > hi_f:
                        high_list.append((v, "IQR"))

        # MAD method
        if (
            isinstance(stats.mad, (int, float))
            and isinstance(stats.median, (int, float))
            and stats.mad
            and not math.isnan(stats.mad)
            and not math.isnan(stats.median)
        ):
            for v in sample_vals:
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    continue
                mz = abs(MAD_SCALE_FACTOR * (v - stats.median) / stats.mad)
                if mz > MAD_OUTLIER_THRESHOLD:
                    if v < stats.median:
                        low_list.append((v, "MAD"))
                    else:
                        high_list.append((v, "MAD"))

        return low_list, high_list

    def _build_index_map(self, stats: NumericStats) -> dict:
        """Build index mapping for outliers."""
        idx_map = {}
        try:
            for idx, val in list(getattr(stats, "min_items", []) or []) + list(
                getattr(stats, "max_items", []) or []
            ):
                key = round(float(val), 12)
                idx_map.setdefault(key, []).append(idx)
        except Exception:
            pass
        return idx_map

    def _deduplicate_outliers(self, outliers: list) -> list:
        """Group outliers by value, keeping track of all detection methods."""
        value_map = {}
        for v, t in outliers:
            key = round(float(v), 12)
            if key in value_map:
                # Value already seen - add method if different
                existing_val, existing_methods = value_map[key]
                if t not in existing_methods:
                    existing_methods.append(t)
            else:
                value_map[key] = (v, [t])

        # Return list with combined methods
        return [(v, methods) for v, methods in value_map.values()]

    def _get_outlier_severity(
        self, value: float, method: str, stats: NumericStats
    ) -> tuple[str, str]:
        """Calculate and format outlier severity indicator with statistical context.

        Returns:
            Tuple of (severity_text, css_class)
        """
        try:
            if (
                method == "IQR"
                and hasattr(stats, "q1")
                and hasattr(stats, "q3")
                and hasattr(stats, "iqr")
            ):
                # Calculate how many IQRs away from the nearest quartile
                if value < stats.q1:
                    distance = (stats.q1 - value) / stats.iqr if stats.iqr > 0 else 0
                else:
                    distance = (value - stats.q3) / stats.iqr if stats.iqr > 0 else 0

                if distance >= 3.0:
                    return f"Extreme ({distance:.1f}× IQR)", "extreme"
                elif distance >= 2.0:
                    return f"High ({distance:.1f}× IQR)", "high"
                else:
                    return f"Moderate ({distance:.1f}× IQR)", "moderate"

            elif method == "MAD" and hasattr(stats, "median") and hasattr(stats, "mad"):
                # Calculate how many MADs away from median
                distance = abs(value - stats.median) / stats.mad if stats.mad > 0 else 0

                if distance >= 3.5:
                    return f"Extreme ({distance:.1f}× MAD)", "extreme"
                elif distance >= 2.5:
                    return f"High ({distance:.1f}× MAD)", "high"
                else:
                    return f"Moderate ({distance:.1f}× MAD)", "moderate"
            else:
                return "Detected", "moderate"
        except Exception:
            return "Detected", "moderate"

    def _format_outliers_table(
        self, outliers: list, idx_map: dict, stats: NumericStats
    ) -> str:
        """Format outliers into HTML table with enhanced context and severity indicators."""
        if not outliers:
            return "<tr><td colspan=4>—</td></tr>"

        parts = []
        for v, t in outliers:
            key = round(float(v), 12)
            idxs = idx_map.get(key) or []
            idx_disp = self.safe_html_escape(str(idxs[0])) if idxs else "—"

            # Enhanced method labels
            method_label = "Extreme (IQR)" if t == "IQR" else "Extreme (MAD)"

            # Add severity indicator based on method
            severity, severity_class = self._get_outlier_severity(v, t, stats)

            parts.append(
                f"<tr><td>{idx_disp}</td><td class='num'>{self.format_number(v)}</td>"
                f"<td class='method'>{method_label}</td><td class='severity' data-severity='{severity_class}'>{severity}</td></tr>"
            )

        return (
            '<table class="kv"><thead><tr><th>Index</th><th>Value</th><th>Method</th><th>Severity</th></tr></thead><tbody>'
            + "".join(parts)
            + "</tbody></table>"
        )

    def _format_enhanced_outliers_table(
        self, outliers: list, idx_map: dict, stats: NumericStats, direction: str
    ) -> str:
        """Format outliers into enhanced HTML table with visual improvements and summary statistics.

        This method creates a professional, feature-rich table that provides comprehensive
        insights into outliers with summary statistics, severity breakdown, and visual indicators.

        Args:
            outliers: List of (value, method) tuples for outliers
            idx_map: Dictionary mapping values to indices
            stats: NumericStats object containing statistical data
            direction: Direction of outliers ('low' or 'high')

        Returns:
            HTML string for the enhanced outliers table with summary
        """
        if not outliers:
            # Still show summary box even with 0 outliers
            direction_icon = "📉" if direction == "low" else "📈"
            direction_label = "Low Outliers" if direction == "low" else "High Outliers"

            summary_html = f"""
            <div class="outlier-summary">
                <div class="summary-header">
                    <span class="direction-icon">{direction_icon}</span>
                    <span class="direction-label">{direction_label}</span>
                    <span class="outlier-count">0 outliers (0.0%)</span>
                </div>
                <div class="severity-breakdown">
                    <span class="severity-item extreme">Extreme: 0</span>
                    <span class="severity-item high">High: 0</span>
                    <span class="severity-item moderate">Moderate: 0</span>
                </div>
            </div>
            """
            return summary_html

        # Calculate summary statistics
        total_count = getattr(stats, "count", 0)
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / max(1, total_count)) * 100.0

        # Get total outliers from general statistics for context
        total_outliers_iqr = getattr(stats, "outliers_iqr", 0)
        total_outliers_pct = (
            (total_outliers_iqr / max(1, total_count)) * 100.0
            if total_outliers_iqr
            else 0.0
        )

        # Check if we're showing a sample vs full dataset
        is_sample = (
            len(outliers) < total_outliers_iqr if total_outliers_iqr > 0 else False
        )
        sample_note = (
            f" (showing top {outlier_count} of {total_outliers_iqr} total)"
            if is_sample
            else ""
        )

        # Get severity distribution per method
        severity_counts_iqr = {"extreme": 0, "high": 0, "moderate": 0}
        severity_counts_mad = {"extreme": 0, "high": 0, "moderate": 0}
        has_iqr = False
        has_mad = False

        for v, methods in outliers:
            for method in methods:
                _, severity_class = self._get_outlier_severity(v, method, stats)
                if method == "IQR":
                    severity_counts_iqr[severity_class] += 1
                    has_iqr = True
                elif method == "MAD":
                    severity_counts_mad[severity_class] += 1
                    has_mad = True

        # Build summary header
        direction_icon = "📉" if direction == "low" else "📈"
        direction_label = "Low Outliers" if direction == "low" else "High Outliers"

        # Build severity breakdown - show both methods if both are present
        severity_breakdown_html = ""
        if has_iqr and has_mad:
            severity_breakdown_html = f"""
            <div class="severity-breakdown">
                <div class="method-severity-group">
                    <span class="method-label">IQR:</span>
                    <span class="severity-item extreme">Extreme: {severity_counts_iqr["extreme"]}</span>
                    <span class="severity-item high">High: {severity_counts_iqr["high"]}</span>
                    <span class="severity-item moderate">Moderate: {severity_counts_iqr["moderate"]}</span>
                </div>
                <div class="method-severity-group">
                    <span class="method-label">MAD:</span>
                    <span class="severity-item extreme">Extreme: {severity_counts_mad["extreme"]}</span>
                    <span class="severity-item high">High: {severity_counts_mad["high"]}</span>
                    <span class="severity-item moderate">Moderate: {severity_counts_mad["moderate"]}</span>
                </div>
            </div>
            """
        elif has_iqr:
            severity_breakdown_html = f"""
            <div class="severity-breakdown">
                <span class="severity-item extreme">Extreme: {severity_counts_iqr["extreme"]}</span>
                <span class="severity-item high">High: {severity_counts_iqr["high"]}</span>
                <span class="severity-item moderate">Moderate: {severity_counts_iqr["moderate"]}</span>
            </div>
            """
        else:  # has_mad
            severity_breakdown_html = f"""
            <div class="severity-breakdown">
                <span class="severity-item extreme">Extreme: {severity_counts_mad["extreme"]}</span>
                <span class="severity-item high">High: {severity_counts_mad["high"]}</span>
                <span class="severity-item moderate">Moderate: {severity_counts_mad["moderate"]}</span>
            </div>
            """

        summary_html = f"""
        <div class="outlier-summary">
            <div class="summary-header">
                <span class="direction-icon">{direction_icon}</span>
                <span class="direction-label">{direction_label}</span>
                <span class="outlier-count">{outlier_count} outliers ({outlier_pct:.1f}%){sample_note}</span>
            </div>
            {severity_breakdown_html}
            {f'<div class="context-note"><small>💡 This shows the most extreme outliers from a representative sample. The general statistics show all {total_outliers_iqr} outliers ({total_outliers_pct:.1f}%) in the full dataset.</small></div>' if is_sample else ""}
        </div>
        """

        # Build enhanced table rows
        parts = []
        has_missing_indices = False
        for i, (v, methods) in enumerate(outliers):
            key = round(float(v), 12)
            idxs = idx_map.get(key) or []
            idx_disp = self.safe_html_escape(str(idxs[0])) if idxs else "—"
            if not idxs:
                has_missing_indices = True

            # Add ranking for top outliers
            rank_icon = ordinal_number(i + 1)

            # If detected by multiple methods, show each method on a separate row
            for method_idx, method in enumerate(methods):
                # Enhanced method labels
                if method == "IQR":
                    method_label = "IQR Method"
                else:
                    method_label = "MAD Method"

                # Add severity indicator based on method
                severity, severity_class = self._get_outlier_severity(v, method, stats)

                # First method row shows rank, index, value with rowspan
                # Subsequent method rows only show method and severity
                if method_idx == 0:
                    # First row: show all columns with rowspan for rank/index/value
                    rowspan = len(methods)
                    parts.append(
                        f"<tr class='outlier-row rank-{i + 1}'>"
                        f"<td class='rank' rowspan='{rowspan}'>{rank_icon}</td>"
                        f"<td class='index' rowspan='{rowspan}'>{idx_disp}</td>"
                        f"<td class='num outlier-value' rowspan='{rowspan}'>{self.format_number(v)}</td>"
                        f"<td class='method'>{method_label}</td>"
                        f"<td class='severity'><span class='severity-item {severity_class}'>{severity}</span></td>"
                        f"</tr>"
                    )
                else:
                    # Additional method rows: only method and severity
                    parts.append(
                        f"<tr class='outlier-row outlier-row-sub rank-{i + 1}'>"
                        f"<td class='method'>{method_label}</td>"
                        f"<td class='severity'><span class='severity-item {severity_class}'>{severity}</span></td>"
                        f"</tr>"
                    )

        table_html = (
            '<table class="outliers-table enhanced">'
            "<thead><tr><th>Rank</th><th>Index</th><th>Value</th><th>Method</th><th>Severity</th></tr></thead>"
            f"<tbody>{''.join(parts)}</tbody>"
            "</table>"
        )

        # Add note about missing indices if applicable
        index_note = ""
        if has_missing_indices:
            index_note = '<div class="outlier-note"><small>ℹ️ Index shown only for top/bottom extreme values tracked during profiling. Sample-based outliers may not have row indices.</small></div>'

        return summary_html + table_html + index_note

    def _build_correlation_table(self, stats: NumericStats) -> str:
        """Build enhanced correlation table with visual improvements and summary statistics.

        This method creates a professional, feature-rich table that provides comprehensive
        insights into correlations with visual indicators, strength categorization, and context.

        Args:
            stats: NumericStats object containing correlation data

        Returns:
            HTML string for the enhanced correlations table with summary
        """
        corr_data = getattr(stats, "corr_top", []) or []

        if not corr_data:
            threshold = getattr(stats, "corr_threshold", 0.5)
            return f"""
        <div class="correlation-summary">
            <div class="no-correlations">
                <span class="icon">📊</span>
                <span class="message">No significant correlations found</span>
                <small>Correlations below {threshold:.1f} threshold are not shown</small>
            </div>
        </div>
        """

        # Calculate summary statistics
        corr_values = [abs(corr) for _, corr in corr_data]
        sum(corr_values) / len(corr_values) if corr_values else 0

        # Categorize correlations by strength
        strength_counts = {"very_strong": 0, "strong": 0, "moderate": 0, "weak": 0}
        for _, corr in corr_data:
            abs_corr = abs(corr)
            if abs_corr >= 0.9:
                strength_counts["very_strong"] += 1
            elif abs_corr >= 0.7:
                strength_counts["strong"] += 1
            elif abs_corr >= 0.5:
                strength_counts["moderate"] += 1
            else:
                strength_counts["weak"] += 1

        # Build summary header
        summary_html = f"""
        <div class="correlation-summary">
            <div class="summary-header">
                <span class="icon">🔗</span>
                <span class="title">Correlations</span>
                <span class="count">{len(corr_data)} significant correlations</span>
            </div>
            <div class="strength-breakdown">
                <span class="strength-item very-strong">Very Strong: {strength_counts["very_strong"]}</span>
                <span class="strength-item strong">Strong: {strength_counts["strong"]}</span>
                <span class="strength-item moderate">Moderate: {strength_counts["moderate"]}</span>
                <span class="strength-item weak">Weak: {strength_counts["weak"]}</span>
            </div>
        </div>
        """

        # Build enhanced table rows
        parts = []
        for i, (col_name, corr_value) in enumerate(corr_data):
            abs_corr = abs(corr_value)

            # Determine strength and color
            if abs_corr >= 0.9:
                strength = "Very Strong"
                strength_class = "very-strong"
            elif abs_corr >= 0.7:
                strength = "Strong"
                strength_class = "strong"
            elif abs_corr >= 0.5:
                strength = "Moderate"
                strength_class = "moderate"
            else:
                strength = "Weak"
                strength_class = "weak"

            # Direction indicator
            direction = "positive" if corr_value > 0 else "negative"
            direction_icon = "📈" if corr_value > 0 else "📉"

            # Ranking
            rank_icon = ordinal_number(i + 1)

            parts.append(f'''
            <tr class="correlation-row strength-{strength_class}">
                <td class="rank">{rank_icon}</td>
                <td class="column">
                    <code class="missing-col" title="{self.safe_html_escape(col_name)}">{self.safe_html_escape(col_name)}</code>
                </td>
                <td class="correlation-value {direction}">
                    {corr_value:+.3f}
                </td>
                <td class="strength" data-strength="{strength_class}">
                    {strength}
                </td>
                <td class="direction">
                    <span class="direction-icon">{direction_icon}</span>
                    <span class="direction-text">{direction.title()}</span>
                </td>
            </tr>
            ''')

        table_html = f"""
        <table class="correlations-table enhanced">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Column</th>
                    <th>Correlation</th>
                    <th>Strength</th>
                    <th>Direction</th>
                </tr>
            </thead>
            <tbody>
                {"".join(parts)}
            </tbody>
        </table>
        """

        return summary_html + table_html

    def _build_missing_values_table(self, stats: NumericStats) -> str:
        """Build simple missing values analysis matching reference HTML.

        Args:
            stats: NumericStats object containing missing data information

        Returns:
            HTML string for the missing values analysis
        """
        # Calculate missing data statistics
        total_values = stats.count + stats.missing
        missing_pct = (
            (stats.missing / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )
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
                    <span class="stat-value">{stats.missing:,} <span class="stat-pct">({missing_pct:.1f}%)</span></span>
                </span>
            </div>
            <div class="completeness-bar">
                <div class="bar-fill present" style="width: {present_pct:.1f}%" title="Present: {present_pct:.1f}%"></div>
                <div class="bar-fill missing" style="width: {missing_pct:.1f}%" title="Missing: {missing_pct:.1f}%"></div>
            </div>
        </div>
        """

        # Section 2: Chunk Distribution
        chunk_html = self._build_chunk_distribution_simple(stats)

        return completeness_html + chunk_html

    def _build_chunk_distribution_simple(self, stats: NumericStats) -> str:
        """Build simple chunk distribution visualization matching reference HTML.

        Args:
            stats: NumericStats object

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
            <h4 class="section-title">Missing Values per Chunk</h4>
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
                    {severity_icon} {severity.title()} Missing Data
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

    def _simulate_chunk_distribution(self, stats: NumericStats) -> list[dict]:
        """Simulate realistic chunk distribution based on data characteristics.

        Args:
            stats: NumericStats object containing data characteristics

        Returns:
            List of chunk data dictionaries with realistic missing value patterns
        """
        import random

        total_values = stats.count + stats.missing
        missing_pct = (
            (stats.missing / total_values) * 100.0 if total_values > 0 else 0.0
        )

        # Determine number of chunks based on data size
        if total_values < 1000:
            num_chunks = max(2, min(5, total_values // 200))
        elif total_values < 10000:
            num_chunks = max(3, min(8, total_values // 1000))
        else:
            num_chunks = max(5, min(12, total_values // 2000))

        # Calculate base chunk size
        base_chunk_size = total_values // num_chunks
        remaining_values = total_values % num_chunks

        chunks = []
        remaining_missing = stats.missing

        # Create realistic chunk patterns
        for i in range(num_chunks):
            # Vary chunk size slightly for realism
            chunk_size = base_chunk_size + (1 if i < remaining_values else 0)
            chunk_size += random.randint(-chunk_size // 10, chunk_size // 10)
            chunk_size = max(1, chunk_size)

            # Simulate missing value patterns
            if remaining_missing <= 0:
                chunk_missing = 0
            elif i == num_chunks - 1:  # Last chunk gets remaining missing values
                chunk_missing = remaining_missing
            else:
                # Create realistic patterns: some chunks have more missing values
                if missing_pct > 20:  # High missing data - create clusters
                    if random.random() < 0.3:  # 30% chance of high missing chunk
                        chunk_missing = min(remaining_missing, int(chunk_size * 0.6))
                    else:
                        chunk_missing = min(remaining_missing, int(chunk_size * 0.1))
                elif missing_pct > 5:  # Medium missing data - some variation
                    base_missing_pct = missing_pct / 100.0
                    variation = random.uniform(0.5, 1.5)
                    chunk_missing = min(
                        remaining_missing,
                        int(chunk_size * base_missing_pct * variation),
                    )
                else:  # Low missing data - mostly random
                    chunk_missing = min(
                        remaining_missing,
                        random.randint(0, max(1, int(chunk_size * 0.1))),
                    )

            chunk_missing = max(0, min(chunk_missing, remaining_missing))
            remaining_missing -= chunk_missing

            chunk_missing_pct = (
                (chunk_missing / chunk_size) * 100.0 if chunk_size > 0 else 0.0
            )

            chunks.append(
                {
                    "index": i + 1,
                    "size": chunk_size,
                    "missing": chunk_missing,
                    "missing_pct": chunk_missing_pct,
                    "present": chunk_size - chunk_missing,
                }
            )

        return chunks

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
            severity_icon = "🚨"
        elif max_missing_pct > 20:
            severity = "high"
            severity_icon = "⚠️"
        elif max_missing_pct > 5:
            severity = "medium"
            severity_icon = "⚡"
        else:
            severity = "low"
            severity_icon = "✅"

        return {
            "overall_missing_pct": overall_missing_pct,
            "max_missing_pct": max_missing_pct,
            "min_missing_pct": min_missing_pct,
            "avg_missing_pct": avg_missing_pct,
            "high_missing_chunks": len(high_missing_chunks),
            "low_missing_chunks": len(low_missing_chunks),
            "patterns": patterns,
            "severity": severity,
            "severity_icon": severity_icon,
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
                    <span class="stat-value">{insights.get("severity_icon", "✅")} {insights.get("severity", "low").title()}</span>
                </span>
            </div>
        </div>"""

        return f"""
        <div class="missing-per-chunk-enhanced">
            <div class="chunk-header">
                <span class="icon">📊</span>
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
            return "critical", "Critical", "🚨"
        elif missing_pct >= 20:
            return "high", "High", "⚠️"
        elif missing_pct >= 5:
            return "medium", "Medium", "⚡"
        else:
            return "low", "Low", "✅"

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
                "icon": "❓",
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
                    "icon": "0️⃣",
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
                    "icon": "∞",
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
                    "icon": "➖",
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
                <span class="icon">🔍</span>
                <span class="title">Data Quality Indicators</span>
            </div>
            <div class="indicators-grid">
        """

        # Use efficient string building with join
        indicator_items = []
        for indicator in quality_indicators:
            indicator_items.append(f"""
                <div class="indicator-item {indicator["severity"]}">
                    <div class="indicator-icon">{indicator["icon"]}</div>
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
                <span class="icon">💡</span>
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

    def _build_details_section(
        self,
        col_id: str,
        stats_quantiles: str,
        common_table: str,
        extremes_table: str,
        outliers_low: str,
        outliers_high: str,
        corr_table: str,
        missing_table: str,
    ) -> str:
        """Build details section with tabs."""
        return f"""
        <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
                <button role="tab" class="active" data-tab="stats">Statistics</button>
                <button role="tab" data-tab="common">Common values</button>
                <button role="tab" data-tab="extremes">Min/Max Values</button>
                <button role="tab" data-tab="outliers">Outliers</button>
                <button role="tab" data-tab="corr">Correlations</button>
                <button role="tab" data-tab="missing">Missing Values</button>
            </nav>
            <div class="tab-panes">
                <section class="tab-pane active" data-tab="stats">{stats_quantiles}</section>
                <section class="tab-pane" data-tab="common">{common_table}</section>
                <section class="tab-pane" data-tab="extremes">{extremes_table}</section>
                <section class="tab-pane" data-tab="outliers">
                    <div class="stats-quant">
                        <div class="sub">{outliers_low}</div>
                        <div class="sub">{outliers_high}</div>
                    </div>
                </section>
                <section class="tab-pane" data-tab="corr">
                    <div class="sub">{corr_table}</div>
                </section>
                <section class="tab-pane" data-tab="missing">
                    <div class="sub">{missing_table}</div>
                </section>
            </div>
        </section>
        """

    def _build_controls_section(self, col_id: str) -> str:
        """Build controls section."""
        bin_buttons = " ".join(
            f'<button type="button" class="btn-soft{" active" if b == 25 else ""}" data-bin="{b}">{b}</button>'
            for b in self.hist_config.bin_options
        )

        return f"""
        <div class="card-controls" role="group" aria-label="Numeric controls">
            <div class="details-slot">
                <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
            </div>
            <div class="controls-slot">
                <div class="hist-controls" data-scale="lin" data-bin="25">
                    <div class="center-controls">
                        <span>Scale:</span>
                        <div class="scale-group">
                            <button type="button" class="btn-soft active" data-scale="lin">Linear</button>
                            <button type="button" class="btn-soft" data-scale="log">Log</button>
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
        left_table: str,
        right_table: str,
        chart_html: str,
        details_html: str,
        controls_html: str,
    ) -> str:
        """Assemble the complete card HTML."""
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
                    <span class="badge">Numeric</span>
                    <span class="dtype chip">{stats.dtype_str}</span>
                    {approx_badge}
                    {quality_flags_html}
                </div>
                {info_button}
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
