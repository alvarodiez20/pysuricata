"""Configuration constants for card rendering."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ChartDimensions:
    """Standard chart dimensions."""

    width: int = 420
    height: int = 160
    margin_left: int = 45
    margin_right: int = 8
    margin_top: int = 12
    margin_bottom: int = 36


@dataclass(frozen=True)
class HistogramConfig:
    """Histogram configuration."""

    default_bins: int = 25
    bin_options: Tuple[int, ...] = (10, 25, 50)
    max_bins: int = 200
    min_bins: int = 10
    sample_cap: int = 200_000


@dataclass(frozen=True)
class CategoricalConfig:
    """Categorical chart configuration."""

    default_topn: int = 10
    topn_options: Tuple[int, ...] = (5, 10, 15)
    max_display_rows: int = 15
    max_label_length: int = 24
    char_width: int = 7
    min_gutter: int = 60
    max_gutter: int = 180


@dataclass(frozen=True)
class DateTimeConfig:
    """DateTime chart configuration."""

    default_bins: int = 60
    min_bins: int = 10
    max_bins: int = 180
    max_xticks: int = 5
    short_span_ns: int = 3 * 24 * 3600 * 1_000_000_000  # 3 days in nanoseconds


@dataclass(frozen=True)
class BooleanConfig:
    """Boolean chart configuration."""

    chart_height: int = 48
    margin: int = 4
    min_segment_width: int = 44


@dataclass(frozen=True)
class QualityThresholds:
    """Quality assessment thresholds."""

    # Missing data thresholds
    missing_warn_pct: float = 0.0
    missing_crit_pct: float = 20.0

    # Outlier thresholds
    outlier_warn_pct: float = 0.3
    outlier_crit_pct: float = 1.0

    # Zero inflation thresholds
    zero_warn_pct: float = 30.0
    zero_crit_pct: float = 50.0

    # Negative value thresholds
    neg_warn_pct: float = 10.0

    # Skewness thresholds
    skew_threshold: float = 1.0

    # Kurtosis threshold
    kurtosis_threshold: float = 3.0

    # Jarque-Bera test threshold
    jb_threshold: float = 5.99

    # Heaping threshold
    heaping_threshold: float = 30.0

    # Unique ratio thresholds
    unique_ratio_threshold: float = 0.05
    quasi_constant_threshold: float = 0.02

    # Categorical thresholds
    high_cardinality_threshold: float = 0.5
    dominant_category_threshold: float = 0.7
    rare_coverage_warn: float = 30.0
    rare_coverage_crit: float = 60.0
    top5_coverage_good: float = 80.0
    top5_coverage_warn: float = 40.0

    # Boolean thresholds
    imbalance_threshold: float = 0.05


@dataclass(frozen=True)
class SparklineConfig:
    """Sparkline configuration."""

    blocks: str = "▁▂▃▄▅▆▇█"
    levels: int = 8


@dataclass(frozen=True)
class TickConfig:
    """Tick and grid configuration."""

    max_ticks: int = 8
    y_max_ticks: int = 5
    minor_subdivisions: int = 4
    tick_length: int = 4
    minor_tick_length: int = 2
    label_rotation: int = -30


@dataclass(frozen=True)
class CSSClasses:
    """CSS class names used in rendering."""

    # Card structure
    var_card: str = "var-card"
    var_card_header: str = "var-card__header"
    var_card_body: str = "var-card__body"
    title: str = "title"
    colname: str = "colname"
    badge: str = "badge"
    dtype_chip: str = "dtype chip"

    # Layout
    triple_row: str = "triple-row"
    box: str = "box"
    stats_left: str = "stats-left"
    stats_right: str = "stats-right"
    chart: str = "chart"

    # Controls
    card_controls: str = "card-controls"
    details_slot: str = "details-slot"
    controls_slot: str = "controls-slot"
    details_toggle: str = "details-toggle"
    btn_soft: str = "btn-soft"
    active: str = "active"

    # Charts
    hist_chart: str = "hist-chart"
    hist_variants: str = "hist-variants"
    hist_variant: str = "hist variant"
    topn_chart: str = "topn-chart"
    cat_variant: str = "cat variant"

    # Quality flags
    quality_flags: str = "quality-flags"
    flag: str = "flag"
    flag_bad: str = "bad"
    flag_warn: str = "warn"
    flag_good: str = "good"

    # Tables
    kv_table: str = "kv"
    num_cell: str = "num"
    small_cell: str = "small"

    # Details
    details_section: str = "details-section"
    tabs: str = "tabs"
    tab_panes: str = "tab-panes"
    tab_pane: str = "tab-pane"

    # SVG elements
    svg: str = "svg"
    plot_area: str = "plot-area"
    bar: str = "bar"
    bar_row: str = "bar-row"
    bar_label: str = "bar-label"
    bar_value: str = "bar-value"
    axis: str = "axis"
    tick: str = "tick"
    tick_label: str = "tick-label"
    axis_title: str = "axis-title"
    grid: str = "grid"
    line: str = "line"
    hotspots: str = "hotspots"
    hot: str = "hot"
    seg: str = "seg"
    label: str = "label"


# Default configurations
DEFAULT_CHART_DIMS = ChartDimensions()
DEFAULT_HIST_CONFIG = HistogramConfig()
DEFAULT_CAT_CONFIG = CategoricalConfig()
DEFAULT_DT_CONFIG = DateTimeConfig()
DEFAULT_BOOL_CONFIG = BooleanConfig()
DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()
DEFAULT_SPARKLINE_CONFIG = SparklineConfig()
DEFAULT_TICK_CONFIG = TickConfig()
DEFAULT_CSS_CLASSES = CSSClasses()

# Common constants
EPSILON = 1e-9
NANOSECONDS_PER_SECOND = 1_000_000_000
MAD_SCALE_FACTOR = 0.67448975
MAD_OUTLIER_THRESHOLD = 3.5
