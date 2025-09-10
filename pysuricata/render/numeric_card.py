"""Numeric card rendering functionality."""

import html as _html
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .card_base import (
    CardRenderer,
    QualityAssessor,
    TableBuilder,
    compute_x_ticks_and_labels,
    format_hist_bin_labels,
)
from .card_config import (
    DEFAULT_CHART_DIMS,
    DEFAULT_HIST_CONFIG,
    DEFAULT_TICK_CONFIG,
    EPSILON,
    MAD_OUTLIER_THRESHOLD,
    MAD_SCALE_FACTOR,
)
from .card_types import HistogramData, NumericStats, QualityFlags, QuantileData
from .format_utils import fmt_compact as _fmt_compact
from .format_utils import fmt_compact_scientific as _fmt_compact_scientific
from .format_utils import fmt_num as _fmt_num
from .format_utils import human_bytes as _human_bytes
from .svg_utils import _format_pow10_label as _fmt_pow10_label
from .svg_utils import fmt_tick as _fmt_tick
from .svg_utils import nice_ticks as _nice_ticks
from .svg_utils import svg_empty as _svg_empty


class NumericCardRenderer(CardRenderer):
    """Renders numeric data cards."""

    def __init__(self):
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
        quality_flags_html = self._build_quality_flags_html(quality_flags, percentages)

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

    def _build_quality_flags_html(self, flags: QualityFlags, percentages: dict) -> str:
        """Build quality flags HTML with percentage context."""
        flag_items = []

        if flags.missing:
            severity = "bad" if percentages["miss_pct"] > 20 else "warn"
            flag_items.append(f'<li class="flag {severity}">Missing</li>')

        if flags.infinite:
            flag_items.append('<li class="flag bad">Has ∞</li>')

        if flags.has_negatives:
            severity = "warn" if percentages["neg_pct"] > 10 else ""
            flag_items.append(
                f'<li class="flag {severity}">Has negatives</li>'
                if severity
                else '<li class="flag">Has negatives</li>'
            )

        if flags.zero_inflated:
            severity = "bad" if percentages["zeros_pct"] >= 50.0 else "warn"
            flag_items.append(f'<li class="flag {severity}">Zero‑inflated</li>')

        if flags.positive_only:
            flag_items.append('<li class="flag good">Positive‑only</li>')

        if flags.skewed_right:
            flag_items.append('<li class="flag warn">Skewed Right</li>')

        if flags.skewed_left:
            flag_items.append('<li class="flag warn">Skewed Left</li>')

        if flags.heavy_tailed:
            flag_items.append('<li class="flag bad">Heavy‑tailed</li>')

        if flags.approximately_normal:
            flag_items.append('<li class="flag good">≈ Normal (JB)</li>')

        if flags.discrete:
            flag_items.append('<li class="flag warn">Discrete</li>')

        if flags.heaping:
            flag_items.append('<li class="flag">Heaping</li>')

        if flags.bimodal:
            flag_items.append('<li class="flag warn">Possibly bimodal</li>')

        if flags.log_scale_suggested:
            flag_items.append('<li class="flag good">Log‑scale?</li>')

        if flags.constant:
            flag_items.append('<li class="flag bad">Constant</li>')

        if flags.quasi_constant:
            flag_items.append('<li class="flag warn">Quasi‑constant</li>')

        if flags.many_outliers:
            flag_items.append('<li class="flag bad">Many outliers</li>')

        if flags.some_outliers:
            flag_items.append('<li class="flag warn">Some outliers</li>')

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
            ("Unique", f"{stats.unique_est:,}{' (≈)' if stats.approx else ''}", "num"),
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
            ("Processed bytes", f"{mem_display} (≈)", "num"),
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
            ("P1", f"{self.format_number(quantiles.p1)} (≈)", "num"),
            ("P5", f"{self.format_number(quantiles.p5)} (≈)", "num"),
            ("P10", f"{self.format_number(quantiles.p10)} (≈)", "num"),
            ("Q1 (P25)", self.format_number(stats.q1), "num"),
            ("Median (P50)", self.format_number(stats.median), "num"),
            ("Q3 (P75)", self.format_number(stats.q3), "num"),
            ("P90", f"{self.format_number(quantiles.p90)} (≈)", "num"),
            ("P95", f"{self.format_number(quantiles.p95)} (≈)", "num"),
            ("P99", f"{self.format_number(quantiles.p99)} (≈)", "num"),
            ("Range", self.format_number(range_val), "num"),
            ("Std Dev", self.format_number(stats.std), "num"),
        ]

        return self.table_builder.build_key_value_table(data)

    def _build_histogram_variants(
        self, col_id: str, base_title: str, stats: NumericStats
    ) -> str:
        """Build histogram variants HTML."""
        values = stats.sample_vals or []
        scale_count = getattr(stats, "sample_scale", 1.0)

        # Generate all histogram variants
        variants = []
        for bins in self.hist_config.bin_options:
            for scale in ["lin", "log"]:
                hist_svg = self._build_histogram_svg(
                    base_title, values, bins=bins, scale=scale, scale_count=scale_count
                )
                active_class = " active" if (bins == 25 and scale == "lin") else ""
                variants.append(
                    f'<div class="hist variant{active_class}" id="{col_id}-{scale}-bins-{bins}" '
                    f'data-scale="{scale}" data-bin="{bins}">{hist_svg}</div>'
                )

        return f"""
        <div class="hist-chart">
            <div class="hist-variants" data-col="{col_id}">
                {"".join(variants)}
            </div>
        </div>
        """

    def _build_histogram_svg(
        self,
        base_title: str,
        vals: Sequence[float],
        *,
        bins: int = 25,
        scale: str = "lin",
        scale_count: float = 1.0,
        x_min_override: Optional[float] = None,
        x_max_override: Optional[float] = None,
    ) -> str:
        """Build histogram SVG from values."""
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return self.create_empty_svg(
                "hist-svg", self.chart_dims.width, self.chart_dims.height
            )

        if scale == "log":
            arr = arr[arr > 0]
            if arr.size == 0:
                return self.create_empty_svg(
                    "hist-svg", self.chart_dims.width, self.chart_dims.height
                )
            arr = np.log10(arr)

        x_min, x_max = float(np.min(arr)), float(np.max(arr))
        if x_min_override is not None and np.isfinite(x_min_override):
            # For log scale, override values need to be converted to log10 space
            if scale == "log":
                x_min = float(np.log10(x_min_override)) if x_min_override > 0 else x_min
            else:
                x_min = float(x_min_override)
        if x_max_override is not None and np.isfinite(x_max_override):
            # For log scale, override values need to be converted to log10 space
            if scale == "log":
                x_max = float(np.log10(x_max_override)) if x_max_override > 0 else x_max
            else:
                x_max = float(x_max_override)
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        # Create histogram data
        hist_data = self._create_histogram_data(arr, bins, x_min, x_max, scale_count)

        # Build SVG
        return self._build_histogram_svg_content(
            base_title, hist_data, x_min, x_max, scale
        )

    def _create_histogram_data(
        self, arr: np.ndarray, bins: int, x_min: float, x_max: float, scale_count: float
    ) -> HistogramData:
        """Create histogram data structure with proper count handling.

        This method intelligently handles count scaling to ensure accurate representation
        of data distribution while maintaining performance for large datasets.

        Args:
            arr: Input array of numeric values
            bins: Number of histogram bins
            x_min: Minimum value for histogram range
            x_max: Maximum value for histogram range
            scale_count: Scaling factor for counts (used for sampling)

        Returns:
            HistogramData object with properly scaled counts
        """
        counts, edges = np.histogram(arr, bins=int(bins), range=(x_min, x_max))

        # Only apply scaling if it's significantly different from 1.0
        # This prevents unnecessary scaling when working with full datasets
        if abs(scale_count - 1.0) > 0.01:
            counts_scaled = np.maximum(
                0, np.round(counts * max(1.0, float(scale_count)))
            ).astype(int)
        else:
            # Use actual counts when scale is close to 1.0
            counts_scaled = counts.astype(int)

        y_max = int(max(1, counts_scaled.max()))
        total_n = int(counts_scaled.sum()) if counts_scaled.size else 0

        return HistogramData(
            counts=counts,
            edges=edges,
            scaled_counts=counts_scaled,
            y_max=y_max,
            total_n=total_n,
        )

    def _build_histogram_svg_content(
        self,
        base_title: str,
        hist_data: HistogramData,
        x_min: float,
        x_max: float,
        scale: str,
    ) -> str:
        """Build the actual SVG content for histogram."""
        # Calculate dimensions
        iw = (
            self.chart_dims.width
            - self.chart_dims.margin_left
            - self.chart_dims.margin_right
        )
        ih = (
            self.chart_dims.height
            - self.chart_dims.margin_top
            - self.chart_dims.margin_bottom
        )

        def sx(x):
            return self.chart_dims.margin_left + (x - x_min) / (x_max - x_min) * iw

        def sy(y):
            return self.chart_dims.margin_top + (1 - y / hist_data.y_max) * ih

        # X and Y ticks
        x_ticks, x_step, x_tick_labels = compute_x_ticks_and_labels(x_min, x_max, scale)
        y_ticks, y_step = _nice_ticks(0, hist_data.y_max, self.tick_config.y_max_ticks)

        # Ensure top value is represented
        if y_ticks and (y_ticks[-1] < hist_data.y_max - EPSILON):
            y_ticks.append(float(hist_data.y_max))

        parts = [
            f'<svg class="hist-svg" width="{self.chart_dims.width}" height="{self.chart_dims.height}" '
            f'viewBox="0 0 {self.chart_dims.width} {self.chart_dims.height}" role="img" aria-label="Histogram">',
            '<g class="plot-area">',
        ]

        # Draw bars
        for i, c in enumerate(hist_data.scaled_counts):
            x0 = hist_data.edges[i]
            x1 = hist_data.edges[i + 1]
            x = sx(x0)
            w = max(1.0, sx(x1) - sx(x0) - 1.0)
            y = sy(int(c))
            h = (self.chart_dims.margin_top + ih) - y
            pct = (c / hist_data.total_n * 100.0) if hist_data.total_n else 0.0

            # Tooltip/domain labels
            x0_disp, x1_disp = format_hist_bin_labels(x0, x1, scale)
            parts.append(
                f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="1" ry="1" '
                f'data-count="{int(c)}" data-pct="{pct:.1f}" data-x0="{x0_disp}" data-x1="{x1_disp}" '
                f'title="{int(c)} rows ({pct:.1f}%) - [{x0_disp} – {x1_disp}]">'
                f"</rect>"
            )

        parts.append("</g>")

        # Draw axes and ticks
        self._add_axes_and_ticks(
            parts, x_ticks, x_tick_labels, y_ticks, x_min, x_max, scale, sx, sy, iw, ih
        )

        # Add axis titles
        self._add_axis_titles(parts, base_title, scale, iw, ih)

        parts.append("</svg>")
        return "".join(parts)

    def _add_axes_and_ticks(
        self,
        parts,
        x_ticks,
        x_tick_labels,
        y_ticks,
        x_min,
        x_max,
        scale,
        sx,
        sy,
        iw,
        ih,
    ):
        """Add axes and tick marks to SVG."""
        x_axis_y = self.chart_dims.margin_top + ih

        # X and Y axes
        parts.append(
            f'<line class="axis" x1="{self.chart_dims.margin_left}" y1="{x_axis_y}" '
            f'x2="{self.chart_dims.margin_left + iw}" y2="{x_axis_y}"></line>'
        )
        parts.append(
            f'<line class="axis" x1="{self.chart_dims.margin_left}" y1="{self.chart_dims.margin_top}" '
            f'x2="{self.chart_dims.margin_left}" y2="{x_axis_y}"></line>'
        )

        # X ticks
        if scale == "log":
            self._add_log_scale_ticks(
                parts, x_ticks, x_tick_labels, x_min, x_max, sx, x_axis_y
            )
        else:
            self._add_linear_scale_ticks(parts, x_ticks, x_min, x_max, sx, x_axis_y)

        # Y ticks
        for yt in y_ticks:
            py = sy(yt)
            parts.append(
                f'<line class="tick" x1="{self.chart_dims.margin_left - 4}" y1="{py}" '
                f'x2="{self.chart_dims.margin_left}" y2="{py}"></line>'
            )
            parts.append(
                f'<text class="tick-label yt" x="{self.chart_dims.margin_left - 6}" y="{py + 3}" '
                f'text-anchor="end" transform="rotate(-30 {self.chart_dims.margin_left - 6} {py + 3})">'
                f"{_fmt_tick(yt, 1.0)}</text>"
            )

    def _add_log_scale_ticks(
        self, parts, x_ticks, x_tick_labels, x_min, x_max, sx, x_axis_y
    ):
        """Add log scale tick marks with improved visibility and reliability.

        This method generates tick marks for logarithmic scales, ensuring proper
        visibility and handling edge cases gracefully.

        Args:
            parts: List to append SVG elements to
            x_ticks: List of major tick positions
            x_tick_labels: List of tick labels (optional)
            x_min: Minimum value on the axis
            x_max: Maximum value on the axis
            sx: Scale function for x-coordinates
            x_axis_y: Y-coordinate of the x-axis
        """
        labs = x_tick_labels or []

        # Ensure we have ticks to display
        if not x_ticks:
            return

        for i, xt in enumerate(x_ticks):
            px = sx(xt)
            # Enhanced major tick visibility for log scale
            parts.append(
                f'<line class="tick-major" x1="{px}" y1="{x_axis_y}" x2="{px}" y2="{x_axis_y + 6}" '
                f'stroke="currentColor"></line>'
            )

            # Generate label with robust fallback
            try:
                lab = (
                    labs[i]
                    if i < len(labs) and labs[i]
                    else _fmt_pow10_label(int(round(xt)))
                )
            except Exception:
                lab = _fmt_pow10_label(int(round(xt)))

            # Add label with improved positioning
            parts.append(
                f'<text class="tick-label xt" x="{px}" y="{x_axis_y + 16}" text-anchor="middle" '
                f'transform="rotate(-30 {px} {x_axis_y + 16})">{lab}</text>'
            )

    def _add_linear_scale_ticks(self, parts, x_ticks, x_min, x_max, sx, x_axis_y):
        """Add linear scale tick marks.

        This method generates major tick marks for linear scales with clean,
        simple appearance.

        Args:
            parts: List to append SVG elements to
            x_ticks: List of major tick positions
            x_min: Minimum value on the axis
            x_max: Maximum value on the axis
            sx: Scale function for x-coordinates
            x_axis_y: Y-coordinate of the x-axis
        """
        # Add major tick marks
        for i, xt in enumerate(x_ticks):
            px = sx(xt)
            parts.append(
                f'<line class="tick-major" x1="{px}" y1="{x_axis_y}" x2="{px}" y2="{x_axis_y + 6}" '
                f'stroke="currentColor"></line>'
            )
            # Add label for all ticks - use scientific notation for large numbers
            label = _fmt_compact_scientific(xt)
            parts.append(
                f'<text class="tick-label xt" x="{px}" y="{x_axis_y + 16}" text-anchor="middle" '
                f'transform="rotate(-30 {px} {x_axis_y + 16})">{label}</text>'
            )

    def _add_axis_titles(self, parts, base_title, scale, iw, ih):
        """Add axis titles to SVG."""
        x_axis_y = self.chart_dims.margin_top + ih

        if scale == "log":
            parts.append(
                f'<text class="axis-title x" x="{self.chart_dims.margin_left + iw / 2:.2f}" '
                f'y="{x_axis_y + 28}" text-anchor="middle">'
                f'log<tspan baseline-shift="sub" font-size="8">10</tspan>({base_title})'
                f"</text>"
            )
        else:
            parts.append(
                f'<text class="axis-title x" x="{self.chart_dims.margin_left + iw / 2:.2f}" '
                f'y="{x_axis_y + 28}" text-anchor="middle">{base_title}</text>'
            )

        parts.append(
            f'<text class="axis-title y" transform="translate({self.chart_dims.margin_left - 36},'
            f'{self.chart_dims.margin_top + ih / 2:.2f}) rotate(-90)" text-anchor="middle">Count</text>'
        )

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
            rank_icon = (
                "🥇"
                if i == 0
                else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i + 1}."))
            )

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

    def _build_outliers_tables(self, stats: NumericStats) -> Tuple[str, str]:
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
    ) -> Tuple[list, list]:
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
        """Remove duplicate outliers."""
        seen = set()
        result = []
        for v, t in outliers:
            k = (round(float(v), 12), t)
            if k in seen:
                continue
            seen.add(k)
            result.append((v, t))
        return result

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
            return '<div class="muted">No outliers detected</div>'

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

        # Get severity distribution
        severity_counts = {"extreme": 0, "high": 0, "moderate": 0}
        for v, t in outliers:
            _, severity_class = self._get_outlier_severity(v, t, stats)
            severity_counts[severity_class] += 1

        # Build summary header
        direction_icon = "📉" if direction == "low" else "📈"
        direction_label = "Low Outliers" if direction == "low" else "High Outliers"

        summary_html = f"""
        <div class="outlier-summary">
            <div class="summary-header">
                <span class="direction-icon">{direction_icon}</span>
                <span class="direction-label">{direction_label}</span>
                <span class="outlier-count">{outlier_count} outliers ({outlier_pct:.1f}%){sample_note}</span>
            </div>
            <div class="severity-breakdown">
                <span class="severity-item extreme">Extreme: {severity_counts["extreme"]}</span>
                <span class="severity-item high">High: {severity_counts["high"]}</span>
                <span class="severity-item moderate">Moderate: {severity_counts["moderate"]}</span>
            </div>
            {f'<div class="context-note"><small>💡 This shows the most extreme outliers from a representative sample. The general statistics show all {total_outliers_iqr} outliers ({total_outliers_pct:.1f}%) in the full dataset.</small></div>' if is_sample else ""}
        </div>
        """

        # Build enhanced table rows
        parts = []
        for i, (v, t) in enumerate(outliers):
            key = round(float(v), 12)
            idxs = idx_map.get(key) or []
            idx_disp = self.safe_html_escape(str(idxs[0])) if idxs else "—"

            # Enhanced method labels
            method_label = "IQR Method" if t == "IQR" else "MAD Method"

            # Add severity indicator based on method
            severity, severity_class = self._get_outlier_severity(v, t, stats)

            # Add ranking for top outliers
            rank_icon = (
                "🥇"
                if i == 0
                else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i + 1}."))
            )

            # Calculate how extreme this outlier is as a percentage
            if hasattr(stats, "mean") and hasattr(stats, "std") and stats.std > 0:
                z_score = abs(v - stats.mean) / stats.std
                extreme_pct = min(99.9, (1 - (1 / (1 + z_score))) * 100)
            else:
                extreme_pct = 50.0  # Default fallback

            parts.append(
                f"<tr class='outlier-row rank-{i + 1}'>"
                f"<td class='rank'>{rank_icon}</td>"
                f"<td class='index'>{idx_disp}</td>"
                f"<td class='num outlier-value'>{self.format_number(v)}</td>"
                f"<td class='method'>{method_label}</td>"
                f"<td class='severity' data-severity='{severity_class}'>{severity}</td>"
                f"<td class='progress-bar'><div class='bar-fill' style='width:{extreme_pct:.1f}%'></div></td>"
                f"</tr>"
            )

        table_html = (
            '<table class="outliers-table enhanced">'
            "<thead><tr><th>Rank</th><th>Index</th><th>Value</th><th>Method</th><th>Severity</th><th>Extremity</th></tr></thead>"
            f"<tbody>{''.join(parts)}</tbody>"
            "</table>"
        )

        return summary_html + table_html

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
            return """
        <div class="correlation-summary">
            <div class="no-correlations">
                <span class="icon">📊</span>
                <span class="message">No significant correlations found</span>
                <small>Correlations below 0.3 threshold are not shown</small>
            </div>
        </div>
        """

        # Calculate summary statistics
        corr_values = [abs(corr) for _, corr in corr_data]
        avg_strength = sum(corr_values) / len(corr_values) if corr_values else 0

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
            rank_icon = (
                "🥇"
                if i == 0
                else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i + 1}."))
            )

            # Progress bar width (0-100%)
            bar_width = min(100, abs_corr * 100)

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
                <td class="progress-bar">
                    <div class="bar-fill" style="width:{bar_width:.1f}%"></div>
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
                    <th>Visual</th>
                </tr>
            </thead>
            <tbody>
                {"".join(parts)}
            </tbody>
        </table>
        """

        return summary_html + table_html

    def _build_missing_values_table(self, stats: NumericStats) -> str:
        """Build comprehensive missing values analysis table with visual elements.

        This method creates a professional, feature-rich analysis of missing data
        including summary statistics, visual indicators, and data quality insights.
        Optimized for performance on large datasets with efficient calculations.

        Args:
            stats: NumericStats object containing missing data information

        Returns:
            HTML string for the enhanced missing values analysis
        """
        # Calculate missing data statistics with safe division
        total_values = stats.count + stats.missing
        missing_pct = (
            (stats.missing / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )
        present_pct = (
            (stats.count / max(1, total_values)) * 100.0 if total_values > 0 else 0.0
        )

        # Calculate other data quality metrics with safe division
        zeros_pct = (
            (stats.zeros / max(1, stats.count)) * 100.0 if stats.count > 0 else 0.0
        )
        inf_pct = (stats.inf / max(1, stats.count)) * 100.0 if stats.count > 0 else 0.0
        neg_pct = (
            (stats.negatives / max(1, stats.count)) * 100.0 if stats.count > 0 else 0.0
        )

        # Determine data quality severity with clear thresholds
        quality_severity, quality_label, quality_icon = self._get_missing_data_severity(
            missing_pct
        )

        # Build summary header with performance-optimized string formatting
        summary_html = f"""
        <div class="missing-summary">
            <div class="summary-header">
                <span class="icon">📊</span>
                <span class="title">Missing Values Analysis</span>
                <span class="quality-indicator {quality_severity}">
                    {quality_icon} {quality_label} Missing Data
                </span>
            </div>
            <div class="data-overview">
                <div class="overview-item">
                    <span class="label">Total Values</span>
                    <span class="value">{total_values:,}</span>
                </div>
                <div class="overview-item present">
                    <span class="label">Present</span>
                    <span class="value">{stats.count:,}</span>
                    <span class="percentage">({present_pct:.1f}%)</span>
                </div>
                <div class="overview-item missing">
                    <span class="label">Missing</span>
                    <span class="value">{stats.missing:,}</span>
                    <span class="percentage">({missing_pct:.1f}%)</span>
                </div>
            </div>
        </div>
        """

        # Build visual progress bars with efficient string formatting
        progress_html = f"""
        <div class="missing-visualization">
            <div class="progress-container">
                <div class="progress-bar-container">
                    <div class="progress-label">Data Completeness</div>
                    <div class="progress-bar">
                        <div class="progress-fill present" style="width: {present_pct:.1f}%"></div>
                        <div class="progress-fill missing" style="width: {missing_pct:.1f}%"></div>
                    </div>
                    <div class="progress-legend">
                        <span class="legend-item present">Present: {present_pct:.1f}%</span>
                        <span class="legend-item missing">Missing: {missing_pct:.1f}%</span>
                    </div>
                </div>
            </div>
        </div>
        """

        # Build data quality indicators with efficient list comprehension
        quality_indicators = self._build_quality_indicators(
            stats, missing_pct, zeros_pct, inf_pct, neg_pct, quality_severity
        )

        # Build quality indicators HTML with optimized string building
        indicators_html = self._build_indicators_html(quality_indicators)

        # Build recommendations with efficient logic
        recommendations = self._build_recommendations(
            stats, missing_pct, zeros_pct, inf_pct
        )

        # Build recommendations HTML
        recommendations_html = self._build_recommendations_html(recommendations)

        return summary_html + progress_html + indicators_html + recommendations_html

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
                        <div class="sub"><div class="hdr">Low outliers</div>{outliers_low}</div>
                        <div class="sub"><div class="hdr">High outliers</div>{outliers_high}</div>
                    </div>
                </section>
                <section class="tab-pane" data-tab="corr">
                    <div class="sub"><div class="hdr">Correlations</div>{corr_table}</div>
                </section>
                <section class="tab-pane" data-tab="missing">
                    <div class="sub"><div class="hdr">Missing Values</div>{missing_table}</div>
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
