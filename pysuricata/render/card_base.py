"""Base functionality for card rendering."""

import html as _html
import math

from ..compute.processing.inference import MAX_CATEGORICAL_LEVELS
from .card_config import (
    DEFAULT_CSS_CLASSES,
    DEFAULT_QUALITY_THRESHOLDS,
    EPSILON,
)
from .card_types import QualityFlags
from .format_utils import fmt_compact as _fmt_compact
from .format_utils import fmt_compact_scientific as _fmt_compact_scientific
from .format_utils import fmt_num as _fmt_num
from .format_utils import human_bytes as _human_bytes
from .svg_utils import _format_pow10_label as _fmt_pow10_label
from .svg_utils import nice_log_ticks_from_log10 as _nice_log_ticks_from_log10
from .svg_utils import nice_ticks as _nice_ticks
from .svg_utils import safe_col_id as _safe_col_id
from .svg_utils import svg_empty as _svg_empty


def _where_the_gaps_fall(chunk_metadata) -> str:
    """Say, in words, where a column's missing values concentrate.

    A chunk strip exists to reveal that gaps are not evenly spread. It shows
    where they fall; this says it, so the finding survives a phone, a PDF and
    a reader who does not hover (#294).

    The claim is the smallest number of chunks holding at least half the
    missing values -- the same quantity the strip encodes, read out.

    That alone is not yet a finding: on an even spread, half the data holds
    half the gaps by definition, and which chunks get named is decided by how
    ties happen to sort. So the share is compared against the share of *rows*
    those same chunks hold -- what they would carry if the gaps were spread
    evenly -- and it only speaks when it is at least half again as concentrated
    as that. The comparison has to be against rows and not against the chunk
    count, because the last chunk of a file is usually a short one: two chunks
    of 50,000 and 10,000 rows holding 10,000 gaps each is an even split by
    chunk and a threefold concentration by data.

    Otherwise it says the gaps are spread, which is true and more useful than
    a ranking of noise.
    """
    counts = [missing for _, _, missing in chunk_metadata]
    sizes = [end - start + 1 for start, end, _ in chunk_metadata]
    total = sum(counts)
    total_rows = sum(sizes)
    if total <= 0 or total_rows <= 0:
        return ""

    # Ranked by gap *rate*, not by raw count. Two chunks holding 10,000gaps
    # each rank equally by count, and a tie then resolves to whichever came
    # first -- which on a file whose last chunk is short is the wrong one, and
    # names the chunk where the gaps are thinnest.
    order = sorted(
        range(len(counts)),
        key=lambda i: (counts[i] / sizes[i] if sizes[i] else 0.0, counts[i]),
        reverse=True,
    )
    running = 0
    holders: list[int] = []
    for i in order:
        holders.append(i)
        running += counts[i]
        if running * 2 >= total:
            break

    n_chunks = len(counts)
    share = running / total * 100.0
    rows_share = sum(sizes[i] for i in holders) / total_rows * 100.0
    if share < 1.5 * rows_share:
        return f"The {total:,} missing values are spread across all {n_chunks} chunks."

    holders.sort()
    k = len(holders)
    if holders == list(range(n_chunks - k, n_chunks)):
        where = "The last chunk holds" if k == 1 else f"The last {k} chunks hold"
    elif holders == list(range(k)):
        where = "The first chunk holds" if k == 1 else f"The first {k} chunks hold"
    elif k == 1:
        where = f"Chunk {holders[0] + 1} of {n_chunks} holds"
    else:
        where = f"{k} of the {n_chunks} chunks hold"

    return f"{where} {share:.0f}% of the {total:,} missing values."


class CardRenderer:
    """Base class for card rendering functionality."""

    def __init__(self):
        self.css = DEFAULT_CSS_CLASSES
        self.thresholds = DEFAULT_QUALITY_THRESHOLDS
        self.quality_assessor = QualityAssessor()
        self.table_builder = TableBuilder()

    def safe_html_escape(self, text: str) -> str:
        """Safely escape HTML content."""
        return _html.escape(str(text))

    def safe_col_id(self, name: str) -> str:
        """Generate safe column ID for HTML."""
        return _safe_col_id(name)

    def format_number(self, value: int | float) -> str:
        """Format number for display."""
        return _fmt_num(value)

    def format_compact(self, value: int | float) -> str:
        """Format number in compact notation."""
        return _fmt_compact(value)

    def format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human-readable format."""
        return _human_bytes(bytes_count)

    def create_empty_svg(self, svg_class: str, width: int, height: int) -> str:
        """Create empty SVG placeholder."""
        return _svg_empty(svg_class, width, height)

    def _build_approx_badge(self, approx: bool) -> str:
        """Return an 'approx' badge if values are approximate, else empty string."""
        return '<span class="badge">approx</span>' if approx else ""

    def _build_tabbed_details(
        self,
        col_id: str,
        panes: "list[tuple[str, str, str, bool]]",
        counts: "dict[str, str] | None" = None,
    ) -> str:
        """A tabbed details section, rendering only the tabs that have something.

        Args:
            col_id: Sanitised column id.
            panes: ordered ``(key, label, html, worth_showing)``. Order is fixed
                so a tab never moves; it only appears or does not.

        A tab that repeats the card face, or reports a zero, costs a click to
        learn nothing. The Missing Values pane rendered on every column,
        including ones with no missing values, as a 100%-present bar and a
        one-segment strip reading 0.0%.

        The first surviving pane is the active one, so a card whose Statistics
        tab is dropped still opens on something.

        `counts` maps a pane key to the figure that pane is worth opening for
        -- `11` beside Outliers. A reader picking a tab should not have to
        guess which one holds the thing they came for.
        """
        kept = [(key, label, html) for key, label, html, worth in panes if worth]
        if not kept:
            return ""

        # The active marker is built outside the f-string. Nesting the same
        # quote character, or a backslash escape, inside one is Python 3.12+
        # syntax and this package supports 3.10 -- ruff's py310 target caught it
        # where the local interpreter, being newer, ran it happily. Second 3.10
        # slip of the day; the first reached CI.
        active_tab = ' class="active"'
        counts = counts or {}

        def tab(index: int, key: str, label: str) -> str:
            # The label lives in its own span so the active underline can go on
            # the text rather than on the 44px tap box. On the box the rule
            # paints ~29px below the word and reads as a second hairline
            # floating under the strip.
            badge = counts.get(key)
            inner = f'<span class="tab__label">{label}'
            if badge:
                inner += f' <span class="tab__count">{badge}</span>'
            inner += "</span>"
            return (
                f'<button role="tab"{active_tab if index == 0 else ""} '
                f'data-tab="{key}">{inner}</button>'
            )

        tabs = "".join(tab(i, key, label) for i, (key, label, _) in enumerate(kept))
        bodies = "".join(
            f'<section class="tab-pane{" active" if i == 0 else ""}" '
            f'data-tab="{key}">{html}</section>'
            for i, (key, _, html) in enumerate(kept)
        )
        return (
            f'<section id="{col_id}-details" class="details-section" hidden>'
            f'<nav class="tabs" role="tablist" aria-label="More details">{tabs}</nav>'
            f'<div class="tab-panes">{bodies}</div>'
            "</section>"
        )

    def _build_stat_row(self, rows: list[tuple[str, str, str | None]]) -> str:
        """One full-width stat row in place of two narrow tables.

        The tables were 240px each beside a squeezed chart. As a row they take
        the card's full width, which is what lets the chart have the rest.

        `minmax(0, 1fr)` rather than `1fr`: a grid track's default minimum is
        its content, so one long value -- `-1.2345678e+18` is the case that does
        it -- widens its column and pushes the others out of alignment instead
        of wrapping inside its own cell.

        Lives here rather than on one renderer because all four card kinds need
        the same row. #114 restacked the numeric card and left the other three
        on `.triple-row`, so a report mixing column types showed two different
        card architectures side by side -- which is more jarring than either
        one alone.
        """
        cells = []
        for label, value, cls in rows:
            tone = ""
            for level in ("crit", "warn"):
                if level in (cls or ""):
                    tone = f" is-{level}"
                    break
            cells.append(
                f'<div class="vstat{tone}">'
                f'<div class="vstat__cap">{label}</div>'
                f'<div class="vstat__val">{value}</div>'
                "</div>"
            )
        return f'<div class="vstat-row">{"".join(cells)}</div>'

    def _build_chunk_distribution_simple(self, stats, total_values: int) -> str:
        """Build chunk-level missing values distribution bar.

        Args:
            stats: Any stats object that may carry a ``chunk_metadata`` attribute.
            total_values: Pre-computed total row count (present + missing).
        """
        chunk_metadata = getattr(stats, "chunk_metadata", None)
        if not chunk_metadata:
            return ""
        if total_values == 0:
            return ""

        segments_html = ""
        max_missing_pct = 0.0
        num_chunks = len(chunk_metadata)

        for start_row, end_row, missing_count in chunk_metadata:
            chunk_size = end_row - start_row + 1
            missing_pct = (
                (missing_count / chunk_size) * 100.0 if chunk_size > 0 else 0.0
            )
            width_pct = (chunk_size / total_values) * 100.0

            if missing_pct > max_missing_pct:
                max_missing_pct = missing_pct

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
            <h4 class="section-title">Missing values per chunk</h4>
            <p class="chunk-finding">{_where_the_gaps_fall(chunk_metadata)}</p>
            <div class="chunk-info">
                <span>{num_chunks} chunks analyzed</span>
                <span>Peak: {max_missing_pct:.1f}%</span>
            </div>
            <div class="chunk-spectrum">
                {segments_html}
            </div>
        </div>
        """

    def _build_missing_values_table(
        self,
        present_count: int,
        present_pct: float,
        missing_count: int,
        missing_pct: float,
        stats,
        total_values: int,
    ) -> str:
        """Build data completeness section (shared across all card types).

        Args:
            present_count: Number of non-missing rows.
            present_pct: Pre-computed present percentage (0-100).
            missing_count: Number of missing rows.
            missing_pct: Pre-computed missing percentage (0-100).
            stats: Any stats object (used for chunk_metadata access).
            total_values: Total row count; passed to chunk distribution.
        """
        completeness_html = f"""
        <div class="missing-analysis-header">
            <h4 class="section-title">Data Completeness</h4>
        </div>

        <div class="completeness-container">
            <div class="completeness-stats">
                <span class="stat-item">
                    <span class="stat-label">Present:</span>
                    <span class="stat-value">{present_count:,} <span class="stat-pct">({present_pct:.1f}%)</span></span>
                </span>
                <span class="stat-item">
                    <span class="stat-label">Missing:</span>
                    <span class="stat-value">{missing_count:,} <span class="stat-pct">({missing_pct:.1f}%)</span></span>
                </span>
            </div>
            <div class="completeness-bar">
                <div class="bar-fill present" style="width: {present_pct:.1f}%" title="Present: {present_pct:.1f}%"></div>
                <div class="bar-fill missing" style="width: {missing_pct:.1f}%" title="Missing: {missing_pct:.1f}%"></div>
            </div>
        </div>
        """
        chunk_html = self._build_chunk_distribution_simple(stats, total_values)
        return completeness_html + chunk_html


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

        # Discrete: few enough distinct whole numbers that the values read as
        # labels rather than as measurements. The ceiling is the type
        # classifier's own, imported rather than repeated, so the flag and the
        # classification cannot disagree -- and, being an absolute count, it
        # does not change with the row count the way the old unique *ratio*
        # did.
        if stats.int_like and 0 < int(stats.unique_est) <= MAX_CATEGORICAL_LEVELS:
            flags.discrete = True

        # Heaping
        if isinstance(stats.heap_pct, float) and math.isfinite(stats.heap_pct):
            flags.heaping = stats.heap_pct >= self.thresholds.heaping_threshold

        # Bimodal
        flags.bimodal = getattr(stats, "bimodal", False)

        # Log scale suggestion
        if flags.positive_only and flags.skewed_right:
            flags.log_scale_suggested = True

        # Constant / quasi-constant.
        #
        # This used to fire when `unique_est / count` fell below 2%, which makes
        # the flag a function of the row count rather than of the column: `age`,
        # 68 distinct integers between 18 and 85, is unflagged at 1,000 rows and
        # "Quasi-constant" at 20,000. That is the same unique-ratio reasoning the
        # type classifier dropped in #84, left behind in the flag layer -- and
        # since #86 put these chips in a triage block at the top of the report,
        # the false alarm became the first thing a reader sees.
        #
        # Quasi-constant is a claim about *concentration*, not cardinality:
        # almost every row holds the same value. Misra-Gries counts are lower
        # bounds, so a share computed from them can understate dominance but
        # never invent it, which is the right direction for a warning.
        uniq_est = max(0, int(stats.unique_est))
        total_nonnull = max(1, int(stats.count))

        if uniq_est == 1:
            flags.constant = True
        elif uniq_est <= 2:
            flags.quasi_constant = True
        else:
            top_values = getattr(stats, "top_values", None)
            if top_values:
                share = top_values[0][1] / total_nonnull
                flags.quasi_constant = share >= self.thresholds.dominant_value_share

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

        # Case and trim variants: flag only when lowercasing/stripping *reduces* the
        # unique count, meaning there are genuine case or whitespace variants.
        # A zero estimate means the feature is disabled — treat as no variants.
        flags.case_variants = (
            stats.case_variants_est > 0 and stats.unique_est > stats.case_variants_est
        )
        flags.trim_variants = (
            stats.trim_variants_est > 0 and stats.unique_est > stats.trim_variants_est
        )

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

    def build_key_value_table(self, data: list[tuple[str, str, str | None]]) -> str:
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
            # The accumulator counts `value == "" or value == "0"`, so the
            # label has to say both.
            flag_items.append(f'<li class="{self.css.flag}">Empty or zero</li>')

        # Boolean flags
        if flags.imbalanced:
            flag_items.append(f'<li class="{self.css.flag} warn">Imbalanced</li>')

        return (
            f'<ul class="{self.css.quality_flags}">{"".join(flag_items)}</ul>'
            if flag_items
            else ""
        )


def format_hist_bin_labels(x0: float, x1: float, scale: str) -> tuple[str, str]:
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

        lbl_map = {t: lbl for t, lbl in zip(ticks_all, labels_all, strict=False)}
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
