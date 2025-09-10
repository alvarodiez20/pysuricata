"""Type definitions for card rendering."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    min: Union[int, float]
    max: Union[int, float]
    mean: Union[int, float]
    median: Union[int, float]
    std: Union[int, float]
    variance: Union[int, float]
    se: Union[int, float]
    cv: Union[int, float]
    gmean: Union[int, float]
    q1: Union[int, float]
    q3: Union[int, float]
    iqr: Union[int, float]
    mad: Union[int, float]
    skew: Union[int, float]
    kurtosis: Union[int, float]
    jb_chi2: Union[int, float]
    ci_lo: Union[int, float]
    ci_hi: Union[int, float]
    gran_step: Optional[Union[int, float]]
    gran_decimals: Optional[int]
    heap_pct: Union[int, float]
    zeros: int
    negatives: int
    inf: int
    outliers_iqr: int
    int_like: bool
    unique_ratio_approx: Optional[float]
    mono_inc: bool
    mono_dec: bool
    bimodal: bool
    mem_bytes: int
    sample_vals: Optional[Sequence[float]]
    sample_scale: float
    top_values: Optional[Sequence[Tuple[Any, int]]]
    min_items: Optional[Sequence[Tuple[Any, Union[int, float]]]]
    max_items: Optional[Sequence[Tuple[Any, Union[int, float]]]]
    corr_top: Optional[Sequence[Tuple[str, float]]]


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
    top_items: Optional[Sequence[Tuple[str, int]]]
    empty_zero: int
    case_variants_est: int
    trim_variants_est: int


@dataclass
class DateTimeStats:
    """Statistics for datetime columns."""

    name: str
    dtype_str: str
    count: int
    missing: int
    mem_bytes: int
    min_ts: Optional[int]
    max_ts: Optional[int]
    mono_inc: bool
    mono_dec: bool
    sample_ts: Optional[List[int]]
    sample_scale: float
    by_hour: Optional[List[int]]
    by_dow: Optional[List[int]]
    by_month: Optional[List[int]]


@dataclass
class BooleanStats:
    """Statistics for boolean columns."""

    name: str
    dtype_str: str
    true_n: int
    false_n: int
    missing: int
    mem_bytes: int


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

    positions: List[float]
    labels: Optional[List[str]]
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

    labels: List[str]
    counts: List[int]
    percentages: List[float]
    values: List[float]


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
ValueCount = Tuple[str, int]
IndexValue = Tuple[Any, Union[int, float]]
CorrelationPair = Tuple[str, float]
