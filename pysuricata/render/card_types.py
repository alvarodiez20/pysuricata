"""Type definitions for card rendering."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np


@dataclass
class NumericStats:
    """Statistics for numeric columns."""

    name: str
    dtype_str: str
    count: int
    missing: int
    unique_est: int
    approx: bool
    min: int | float
    max: int | float
    mean: int | float
    median: int | float
    std: int | float
    variance: int | float
    se: int | float
    cv: int | float
    gmean: int | float
    q1: int | float
    q3: int | float
    iqr: int | float
    mad: int | float
    skew: int | float
    kurtosis: int | float
    jb_chi2: int | float
    ci_lo: int | float
    ci_hi: int | float
    gran_step: int | float | None
    gran_decimals: int | None
    heap_pct: int | float
    zeros: int
    negatives: int
    inf: int
    outliers_iqr: int
    int_like: bool
    unique_ratio_approx: float | None
    mono_inc: bool
    mono_dec: bool
    bimodal: bool
    mem_bytes: int
    sample_vals: Sequence[float] | None
    sample_scale: float
    top_values: Sequence[tuple[Any, int]] | None
    min_items: Sequence[tuple[Any, int | float]] | None
    max_items: Sequence[tuple[Any, int | float]] | None
    corr_top: Sequence[tuple[str, float]] | None
    chunk_metadata: (
        Sequence[tuple[int, int, int]] | None
    )  # (start_row, end_row, missing_count)
    corr_threshold: float = 0.5  # Threshold used for correlation filtering
    # The smallest strictly positive value, for the log histogram's left edge
    # (#258). None when the column has none, which is when there is no log
    # variant worth drawing at all.
    min_positive: float | None = None


@dataclass
class CategoricalStats:
    """Statistics for categorical columns."""

    name: str
    dtype_str: str
    count: int
    missing: int
    unique_est: int
    approx: bool
    mem_bytes: int
    top_items: Sequence[tuple[str, int]] | None
    empty_zero: int
    case_variants_est: int
    trim_variants_est: int
    #: `(start_row, end_row, missing_in_chunk)` per chunk (#193).
    #:
    #: The renderer is duck-typed over this type and the accumulator's summary,
    #: so both have to carry it. Without it here the unit-test type diverges
    #: from the runtime one and the Missing Values gate reads `None` -- which
    #: does not tighten the rule, it hides the pane permanently.
    chunk_metadata: list[tuple[int, int, int]] | None = None
    #: Levels seen exactly once, out of an exactly counted total (#297).
    #:
    #: Both `None` together when the column had more levels than the counter's
    #: capacity -- unknown, never zero. Duck-typed against the accumulator's
    #: summary like `chunk_metadata` above, so both types have to carry it.
    singleton_levels: int | None = None
    exact_levels: int | None = None


@dataclass
class DateTimeStats:
    """Statistics for datetime columns."""

    name: str
    dtype_str: str
    count: int
    missing: int
    mem_bytes: int
    min_ts: int | None
    max_ts: int | None
    mono_inc: bool
    mono_dec: bool
    sample_ts: list[int] | None
    sample_scale: float
    by_hour: list[int] | None
    by_dow: list[int] | None
    by_month: list[int] | None
    by_year: dict[int, int] | None
    # Temporal analysis fields
    unique_est: int = 0
    approx: bool = True  # unique_est uses KMV sketch, always approximate
    time_span_days: float = 0.0
    avg_interval_seconds: float = 0.0
    interval_std_seconds: float = 0.0
    weekend_ratio: float = 0.0
    business_hours_ratio: float = 0.0
    seasonal_pattern: str | None = None
    chunk_metadata: Sequence[tuple[int, int, int]] | None = None


@dataclass
class BooleanStats:
    """Statistics for boolean columns."""

    name: str
    dtype_str: str
    true_n: int
    false_n: int
    missing: int
    mem_bytes: int
    #: `(start_row, end_row, missing_in_chunk)` per chunk (#193).
    #:
    #: The renderer is duck-typed over this type and the accumulator's summary,
    #: so both have to carry it. Without it here the unit-test type diverges
    #: from the runtime one and the Missing Values gate reads `None` -- which
    #: does not tighten the rule, it hides the pane permanently.
    chunk_metadata: list[tuple[int, int, int]] | None = None


@dataclass
class QualityFlags:
    """Quality assessment flags."""

    missing: bool = False
    infinite: bool = False
    has_negatives: bool = False
    zero_inflated: bool = False
    positive_only: bool = False
    skewed_right: bool = False
    skewed_left: bool = False
    heavy_tailed: bool = False
    approximately_normal: bool = False
    discrete: bool = False
    heaping: bool = False
    bimodal: bool = False
    log_scale_suggested: bool = False
    constant: bool = False
    quasi_constant: bool = False
    many_outliers: bool = False
    some_outliers: bool = False
    monotonic_increasing: bool = False
    monotonic_decreasing: bool = False
    high_cardinality: bool = False
    dominant_category: bool = False
    many_rare_levels: bool = False
    case_variants: bool = False
    trim_variants: bool = False
    empty_strings: bool = False
    imbalanced: bool = False


@dataclass
class ChartMargins:
    """Chart margin configuration."""

    left: int
    right: int
    top: int
    bottom: int


@dataclass
class TickInfo:
    """Tick mark information."""

    positions: list[float]
    labels: list[str] | None
    step: float


@dataclass
class HistogramData:
    """Histogram data structure."""

    counts: np.ndarray
    edges: np.ndarray
    scaled_counts: np.ndarray
    y_max: int
    total_n: int


@dataclass
class BarData:
    """Bar chart data structure."""

    labels: list[str]
    counts: list[int]
    percentages: list[float]
    values: list[float]
    #: Where an evenly-split column would put every bar, in the same units as
    #: ``values`` -- so the renderer can place it with the same scale function
    #: as the bars and cannot put the rule on a different axis from the data
    #: it is read against (phase 5f.2, #296).
    #:
    #: ``None`` when there is nothing to compare to: one level, or a level
    #: count the sketch never resolved.
    even_split_value: float | None = None
    #: The same mark as a share of the column, for the tooltip and for
    #: `data-even-pct`. Kept beside the value rather than derived back out of
    #: it: the value is in the chart's units and the round trip through `vmax`
    #: is exactly the kind of arithmetic that goes wrong silently.
    even_split_share: float | None = None


@dataclass
class QuantileData:
    """Quantile data structure."""

    p1: float
    p5: float
    p10: float
    p90: float
    p95: float
    p99: float


# Type aliases for better readability
ColumnStats = Union[NumericStats, CategoricalStats, DateTimeStats, BooleanStats]
ValueCount = tuple[str, int]
IndexValue = tuple[Any, int | float]
CorrelationPair = tuple[str, float]
