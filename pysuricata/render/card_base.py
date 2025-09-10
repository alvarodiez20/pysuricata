"""Base functionality for card rendering."""

import html as _html
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .card_config import (
    DEFAULT_CSS_CLASSES,
    DEFAULT_QUALITY_THRESHOLDS,
    EPSILON,
    MAD_OUTLIER_THRESHOLD,
    MAD_SCALE_FACTOR,
)
from .card_types import QualityFlags
from .format_utils import fmt_compact as _fmt_compact
from .format_utils import fmt_compact_scientific as _fmt_compact_scientific
from .format_utils import fmt_num as _fmt_num
from .format_utils import human_bytes as _human_bytes
from .svg_utils import _format_pow10_label as _fmt_pow10_label
from .svg_utils import fmt_tick as _fmt_tick
from .svg_utils import nice_log_ticks_from_log10 as _nice_log_ticks_from_log10
from .svg_utils import nice_ticks as _nice_ticks
from .svg_utils import safe_col_id as _safe_col_id
from .svg_utils import svg_empty as _svg_empty


class CardRenderer:
    """Base class for card rendering functionality."""

    def __init__(self):
        self.css = DEFAULT_CSS_CLASSES
        self.thresholds = DEFAULT_QUALITY_THRESHOLDS

    def safe_html_escape(self, text: str) -> str:
        """Safely escape HTML content."""
        return _html.escape(str(text))

    def safe_col_id(self, name: str) -> str:
        """Generate safe column ID for HTML."""
        return _safe_col_id(name)

    def format_number(self, value: Union[int, float]) -> str:
        """Format number for display."""
        return _fmt_num(value)

    def format_compact(self, value: Union[int, float]) -> str:
        """Format number in compact notation."""
        return _fmt_compact(value)

    def format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human-readable format."""
        return _human_bytes(bytes_count)

    def create_empty_svg(self, svg_class: str, width: int, height: int) -> str:
        """Create empty SVG placeholder."""
        return _svg_empty(svg_class, width, height)


class QualityAssessor:
    """Assesses data quality and generates flags."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS

    def assess_numeric_quality(self, stats) -> QualityFlags:
        """Assess quality for numeric data."""
        flags = QualityFlags()

        # Calculate percentages
        total = max(1, stats.count + stats.missing)
        miss_pct = (stats.missing / total) * 100.0
        zeros_pct = (stats.zeros / max(1, stats.count)) * 100.0 if stats.count else 0.0
        neg_pct = (
            (stats.negatives / max(1, stats.count)) * 100.0 if stats.count else 0.0
        )
        out_pct = (
            (stats.outliers_iqr / max(1, stats.count)) * 100.0 if stats.count else 0.0
        )
        inf_pct = (stats.inf / max(1, stats.count)) * 100.0 if stats.count else 0.0

        # Missing data
        flags.missing = miss_pct > self.thresholds.missing_warn_pct

        # Infinite values
        flags.infinite = stats.inf > 0

        # Negative values
        flags.has_negatives = neg_pct > 0

        # Zero inflation
        flags.zero_inflated = zeros_pct >= self.thresholds.zero_warn_pct

        # Positive only
        if (
            isinstance(stats.min, (int, float))
            and math.isfinite(stats.min)
            and stats.min > 0
        ):
            flags.positive_only = True

        # Skewness
        if isinstance(stats.skew, float) and math.isfinite(stats.skew):
            flags.skewed_right = stats.skew >= self.thresholds.skew_threshold
            flags.skewed_left = stats.skew <= -self.thresholds.skew_threshold

        # Kurtosis
        if isinstance(stats.kurtosis, float) and math.isfinite(stats.kurtosis):
            flags.heavy_tailed = (
                abs(stats.kurtosis) >= self.thresholds.kurtosis_threshold
            )

        # Jarque-Bera test
        if isinstance(stats.jb_chi2, float) and math.isfinite(stats.jb_chi2):
            flags.approximately_normal = stats.jb_chi2 <= self.thresholds.jb_threshold

        # Discrete detection
        if stats.int_like:
            unique_ratio = getattr(stats, "unique_ratio_approx", None)
            if unique_ratio is not None and not math.isnan(unique_ratio):
                flags.discrete = unique_ratio <= self.thresholds.unique_ratio_threshold
            elif stats.unique_est <= max(1, min(50, int(0.05 * max(1, stats.count)))):
                flags.discrete = True

        # Heaping
        if isinstance(stats.heap_pct, float) and math.isfinite(stats.heap_pct):
            flags.heaping = stats.heap_pct >= self.thresholds.heaping_threshold

        # Bimodal
        flags.bimodal = getattr(stats, "bimodal", False)

        # Log scale suggestion
        if flags.positive_only and flags.skewed_right:
            flags.log_scale_suggested = True

        # Constant/quasi-constant
        uniq_est = max(0, int(stats.unique_est))
        total_nonnull = max(1, int(stats.count))
        unique_ratio = (uniq_est / total_nonnull) if total_nonnull else 0.0

        if uniq_est == 1:
            flags.constant = True
        elif unique_ratio <= self.thresholds.quasi_constant_threshold or uniq_est <= 2:
            flags.quasi_constant = True

        # Outliers
        if out_pct > self.thresholds.outlier_crit_pct:
            flags.many_outliers = True
        elif out_pct > self.thresholds.outlier_warn_pct:
            flags.some_outliers = True

        # Monotonicity
        if total_nonnull > 1:
            flags.monotonic_increasing = stats.mono_inc
            flags.monotonic_decreasing = stats.mono_dec

        return flags

    def assess_categorical_quality(self, stats) -> QualityFlags:
        """Assess quality for categorical data."""
        flags = QualityFlags()

        # Calculate percentages
        total = max(1, stats.count + stats.missing)
        miss_pct = (stats.missing / total) * 100.0

        # Missing data
        flags.missing = miss_pct > self.thresholds.missing_warn_pct

        # High cardinality
        if stats.unique_est > max(
            200, int(self.thresholds.high_cardinality_threshold * max(1, stats.count))
        ):
            flags.high_cardinality = True

        # Dominant category
        if stats.top_items:
            mode_count = stats.top_items[0][1] if stats.top_items else 0
            if mode_count >= int(
                self.thresholds.dominant_category_threshold * max(1, stats.count)
            ):
                flags.dominant_category = True

        # Case and trim variants
        flags.case_variants = stats.case_variants_est > 0
        flags.trim_variants = stats.trim_variants_est > 0

        # Empty strings
        flags.empty_strings = stats.empty_zero > 0

        return flags

    def assess_boolean_quality(self, stats) -> QualityFlags:
        """Assess quality for boolean data."""
        flags = QualityFlags()

        # Calculate percentages
        total = max(1, stats.true_n + stats.false_n + stats.missing)
        miss_pct = (stats.missing / total) * 100.0
        cnt = stats.true_n + stats.false_n

        # Missing data
        flags.missing = miss_pct > self.thresholds.missing_warn_pct

        # Constant
        if cnt > 0 and (stats.true_n == 0 or stats.false_n == 0):
            flags.constant = True

        # Imbalanced
        if cnt > 0:
            p = (stats.true_n / max(1, cnt)) if cnt else 0.0
            flags.imbalanced = p <= self.thresholds.imbalance_threshold or p >= (
                1 - self.thresholds.imbalance_threshold
            )

        return flags

    def assess_datetime_quality(self, stats) -> QualityFlags:
        """Assess quality for datetime data."""
        flags = QualityFlags()

        # Calculate percentages
        total = max(1, stats.count + stats.missing)
        miss_pct = (stats.missing / total) * 100.0

        # Missing data
        flags.missing = miss_pct > self.thresholds.missing_warn_pct

        # Monotonicity
        if stats.count > 1:
            flags.monotonic_increasing = getattr(stats, "mono_inc", False)
            flags.monotonic_decreasing = getattr(stats, "mono_dec", False)

        return flags


class TableBuilder:
    """Builds HTML tables for card display."""

    def __init__(self, css_classes=None):
        self.css = css_classes or DEFAULT_CSS_CLASSES

    def build_key_value_table(self, data: List[Tuple[str, str, Optional[str]]]) -> str:
        """Build a key-value table.

        Args:
            data: List of (key, value, css_class) tuples
        """
        rows = []
        for key, value, css_class in data:
            class_attr = f' class="{css_class}"' if css_class else ""
            rows.append(f"<tr><th>{key}</th><td{class_attr}>{value}</td></tr>")

        return (
            f'<table class="{self.css.kv_table}"><tbody>{"".join(rows)}</tbody></table>'
        )

    def build_quality_flags_html(self, flags: QualityFlags) -> str:
        """Build quality flags HTML."""
        flag_items = []

        # Numeric flags
        if flags.missing:
            severity = (
                "bad" if hasattr(self, "_miss_pct") and self._miss_pct > 20 else "warn"
            )
            flag_items.append(f'<li class="{self.css.flag} {severity}">Missing</li>')

        if flags.infinite:
            flag_items.append(f'<li class="{self.css.flag} bad">Has ∞</li>')

        if flags.has_negatives:
            flag_items.append(f'<li class="{self.css.flag}">Has negatives</li>')

        if flags.zero_inflated:
            flag_items.append(f'<li class="{self.css.flag} warn">Zero‑inflated</li>')

        if flags.positive_only:
            flag_items.append(f'<li class="{self.css.flag} good">Positive‑only</li>')

        if flags.skewed_right:
            flag_items.append(f'<li class="{self.css.flag} warn">Skewed Right</li>')

        if flags.skewed_left:
            flag_items.append(f'<li class="{self.css.flag} warn">Skewed Left</li>')

        if flags.heavy_tailed:
            flag_items.append(f'<li class="{self.css.flag} bad">Heavy‑tailed</li>')

        if flags.approximately_normal:
            flag_items.append(f'<li class="{self.css.flag} good">≈ Normal (JB)</li>')

        if flags.discrete:
            flag_items.append(f'<li class="{self.css.flag} warn">Discrete</li>')

        if flags.heaping:
            flag_items.append(f'<li class="{self.css.flag}">Heaping</li>')

        if flags.bimodal:
            flag_items.append(f'<li class="{self.css.flag} warn">Possibly bimodal</li>')

        if flags.log_scale_suggested:
            flag_items.append(f'<li class="{self.css.flag} good">Log‑scale?</li>')

        if flags.constant:
            flag_items.append(f'<li class="{self.css.flag} bad">Constant</li>')

        if flags.quasi_constant:
            flag_items.append(f'<li class="{self.css.flag} warn">Quasi‑constant</li>')

        if flags.many_outliers:
            flag_items.append(f'<li class="{self.css.flag} bad">Many outliers</li>')

        if flags.some_outliers:
            flag_items.append(f'<li class="{self.css.flag} warn">Some outliers</li>')

        if flags.monotonic_increasing:
            flag_items.append(f'<li class="{self.css.flag} good">Monotonic ↑</li>')

        if flags.monotonic_decreasing:
            flag_items.append(f'<li class="{self.css.flag} good">Monotonic ↓</li>')

        # Categorical flags
        if flags.high_cardinality:
            flag_items.append(f'<li class="{self.css.flag} warn">High cardinality</li>')

        if flags.dominant_category:
            flag_items.append(
                f'<li class="{self.css.flag} warn">Dominant category</li>'
            )

        if flags.many_rare_levels:
            flag_items.append(f'<li class="{self.css.flag} warn">Many rare levels</li>')

        if flags.case_variants:
            flag_items.append(f'<li class="{self.css.flag}">Case variants</li>')

        if flags.trim_variants:
            flag_items.append(f'<li class="{self.css.flag}">Trim variants</li>')

        if flags.empty_strings:
            flag_items.append(f'<li class="{self.css.flag}">Empty strings</li>')

        # Boolean flags
        if flags.imbalanced:
            flag_items.append(f'<li class="{self.css.flag} warn">Imbalanced</li>')

        return (
            f'<ul class="{self.css.quality_flags}">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )


def format_hist_bin_labels(x0: float, x1: float, scale: str) -> Tuple[str, str]:
    """Return compact labels for a histogram bin range with scientific notation for large numbers."""
    if scale == "log":
        try:
            return _fmt_compact_scientific(10**x0), _fmt_compact_scientific(10**x1)
        except Exception:
            pass
    return _fmt_compact_scientific(x0), _fmt_compact_scientific(x1)


def compute_x_ticks_and_labels(x_min: float, x_max: float, scale: str):
    """Compute x ticks and labels depending on axis scale with improved tick count.

    This function generates appropriate tick marks for both linear and logarithmic scales,
    ensuring sufficient tick density for good readability while avoiding overcrowding.

    Args:
        x_min: Minimum value on the axis
        x_max: Maximum value on the axis
        scale: Scale type ('linear' or 'log')

    Returns:
        Tuple of (tick_positions, step_size, tick_labels)
    """
    if scale == "log":
        # Increase from 8 to 10 for better tick density on log scale
        ticks_all, labels_all = _nice_log_ticks_from_log10(x_min, x_max, 10)
        x_ticks = [
            x for x in ticks_all if x >= x_min - EPSILON and x <= x_max + EPSILON
        ]

        # Ensure at least boundary ticks even if range is within a single decade
        if not x_ticks:
            e0 = int(math.floor(x_min))
            e1 = int(math.ceil(x_max))
            x_ticks = [e0] if e0 == e1 else [e0, e1]
            labels_all = [_fmt_pow10_label(t) for t in x_ticks]
            return x_ticks, 1.0, labels_all

        # Ensure minimum of 3 ticks for better readability
        if len(x_ticks) < 3 and len(ticks_all) >= 3:
            # Take more ticks if available
            x_ticks = ticks_all[: min(3, len(ticks_all))]

        lbl_map = {t: lbl for t, lbl in zip(ticks_all, labels_all)}
        return (
            x_ticks,
            1.0,
            [lbl_map.get(t, _fmt_pow10_label(int(round(t)))) for t in x_ticks],
        )

    # Linear scale - increase from 6 to 8 for more ticks
    x_ticks, x_step = _nice_ticks(x_min, x_max, 8)
    xt = [x for x in x_ticks if x >= x_min - EPSILON and x <= x_max + EPSILON]
    if not xt or abs(xt[0] - x_min) > EPSILON:
        xt = [x_min] + [x for x in xt if x > x_min]
    return xt, x_step, None
