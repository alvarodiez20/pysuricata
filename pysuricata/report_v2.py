"""pysuricata.report_v2

Out-of-core (streaming) EDA report generator, designed to:
- Work on datasets that do not fit into memory by consuming chunks.
- Be backend-agnostic at the orchestration layer (pandas today; easy to add polars/arrow).
- Keep computations testable via small, pure accumulator classes.
- Produce a lightweight self-contained HTML report (no heavy deps).

Usage (programmatic):

    from pysuricata.report_v2 import generate_report
    html = generate_report("/path/to/big.csv", chunk_size=250_000)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)

This module intentionally avoids adding heavyweight dependencies. It will use:
- pandas for CSV chunking if available
- pyarrow for parquet batch iteration if available

Later, a polars chunk iterator can be plugged in with minimal code (see TODOs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import warnings
import math
import os
import re
import hashlib
import random
import statistics
from collections import Counter, defaultdict
import html as _html
from datetime import datetime
import logging
import pickle
import gzip
import glob
import heapq

import numpy as np

# === Template & assets helpers (reuse from classic report) ===
import time
from typing import Optional, List
from dataclasses import dataclass, field
from .utils import (
    load_css,
    load_template,
    embed_favicon,
    embed_image,
    load_script,
)

# Resolve pysuricata version (matches classic report)
try:
    try:
        from importlib.metadata import version as _pkg_version  # Python 3.8+
        from importlib.metadata import PackageNotFoundError as _PkgNotFound
    except Exception:  # pragma: no cover
        from importlib_metadata import version as _pkg_version  # type: ignore
        from importlib_metadata import PackageNotFoundError as _PkgNotFound  # type: ignore
except Exception:  # last resort
    _pkg_version = None
    class _PkgNotFound(Exception):
        pass

def _resolve_pysuricata_version() -> str:
    if _pkg_version is not None:
        try:
            return _pkg_version("pysuricata")
        except _PkgNotFound:
            pass
        except Exception:
            pass
    try:
        from . import __version__  # type: ignore
        if isinstance(__version__, str) and __version__:
            return __version__
    except Exception:
        pass
    return os.getenv("PYSURICATA_VERSION", "dev")



def _coerce_to_dataframe(data, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Coerce supported inputs to a pandas DataFrame.
    - If `data` is a DataFrame, return as-is.
    - If `data` is a 2D NumPy array, use provided `columns` or auto-generate names.
    - Otherwise raise a TypeError.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise TypeError("NumPy input must be 2D (rows, cols)")
        n_rows, n_cols = data.shape
        if columns is not None:
            if len(columns) != n_cols:
                raise ValueError("Length of `columns` must match array width")
            col_names = list(columns)
        else:
            col_names = [f"col_{i}" for i in range(n_cols)]
        return pd.DataFrame(data, columns=col_names)
    raise TypeError("`data` must be a pandas DataFrame or a 2D NumPy array")


# --- Section Timer for logging ---
class _SectionTimer:
    """Context manager to log start/end (and duration) of a named section."""
    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label
        self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("▶ %s...", self.label)
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc_type is not None:
            # Log stack trace at ERROR level
            self.logger.exception("✖ %s failed after %.2fs", self.label, dt)
        else:
            self.logger.info("✓ %s done (%.2fs)", self.label, dt)

try:  # optional; used for chunking and type coercion
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore

try:  # optional; used for parquet batch iteration
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional
    pa = None  # type: ignore
    pq = None  # type: ignore

# =============================
# Helpers: hashing & sampling
# =============================

def _u64(x: bytes) -> int:
    """Return a 64-bit unsigned integer hash from bytes using SHA1 (fast enough, no deps)."""
    # Take first 8 bytes of sha1 digest
    return int.from_bytes(hashlib.sha1(x).digest()[:8], "big", signed=False)


class KMV:
    """K-Minimum Values distinct counter (approximate uniques) without extra deps.

    Keep the k smallest 64-bit hashes of the observed values. If fewer than k items
    have been seen, |S| is exact uniques. Otherwise, estimate uniques as (k-1)/t,
    where t is the kth smallest hash normalized to (0,1].
    """

    __slots__ = ("k", "_values")

    def __init__(self, k: int = 2048) -> None:
        self.k = int(k)
        self._values: List[int] = []  # store as integers in [0, 2^64)

    def add(self, v: Any) -> None:
        if v is None:
            v = b"__NULL__"
        elif isinstance(v, bytes):
            pass
        else:
            v = str(v).encode("utf-8", "ignore")
        h = _u64(v)
        if len(self._values) < self.k:
            self._values.append(h)
            if len(self._values) == self.k:
                self._values.sort()
        else:
            # maintain k-smallest set (max-heap simulation via last element after sort)
            if h < self._values[-1]:
                # insert in sorted order (k is small)
                # bisect manually to avoid import
                lo, hi = 0, self.k - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if self._values[mid] < h:
                        lo = mid + 1
                    else:
                        hi = mid
                self._values.insert(lo, h)
                # trim to size
                del self._values[self.k]

    @property
    def is_exact(self) -> bool:
        return len(self._values) < self.k

    def estimate(self) -> int:
        n = len(self._values)
        if n == 0:
            return 0
        if n < self.k:
            # exact
            return n
        # normalize kth smallest to (0,1]
        kth = self._values[-1]
        t = (kth + 1) / 2**64
        if t <= 0:
            return n
        return max(n, int(round((self.k - 1) / t)))


class ReservoirSampler:
    """Reservoir sampler for numeric/datetime values to approximate quantiles/histograms."""

    __slots__ = ("k", "_buf", "_seen")

    def __init__(self, k: int = 20_000) -> None:
        self.k = int(k)
        self._buf: List[float] = []
        self._seen: int = 0

    def add_many(self, arr: Sequence[float]) -> None:
        for x in arr:
            self.add(float(x))

    def add(self, x: float) -> None:
        self._seen += 1
        if len(self._buf) < self.k:
            self._buf.append(x)
        else:
            j = random.randint(1, self._seen)
            if j <= self.k:
                self._buf[j - 1] = x

    def values(self) -> List[float]:
        return self._buf


class MisraGries:
    """Heavy hitters (top-K) with deterministic memory.

    Maintains up to k counters. Good for approximate top categories.
    """

    __slots__ = ("k", "counters")

    def __init__(self, k: int = 50) -> None:
        self.k = int(k)
        self.counters: Dict[Any, int] = {}

    def add(self, x: Any, w: int = 1) -> None:
        if x in self.counters:
            self.counters[x] += w
            return
        if len(self.counters) < self.k:
            self.counters[x] = w
            return
        # decrement all
        to_del = []
        for key in list(self.counters.keys()):
            self.counters[key] -= w
            if self.counters[key] <= 0:
                to_del.append(key)
        for key in to_del:
            del self.counters[key]

    def items(self) -> List[Tuple[Any, int]]:
        # items are approximate; a second pass could refine if needed
        return sorted(self.counters.items(), key=lambda kv: (-kv[1], str(kv[0])[:64]))


# =============================
# Accumulators per dtype
# =============================

@dataclass
class NumericSummary:
    name: str
    count: int
    missing: int
    unique_est: int
    mean: float
    std: float
    variance: float
    se: float
    cv: float
    gmean: float
    min: float
    q1: float
    median: float
    q3: float
    iqr: float
    mad: float
    skew: float
    kurtosis: float
    jb_chi2: float
    max: float
    zeros: int
    negatives: int
    outliers_iqr: int
    approx: bool
    inf: int
    # v2 extras (approximate)
    int_like: bool = False
    unique_ratio_approx: float = float("nan")
    hist_counts: Optional[List[int]] = None
    top_values: List[Tuple[float, int]] = field(default_factory=list)
    # carry a small reservoir sample to render histograms with axes (lin/log)
    sample_vals: Optional[List[float]] = None
    # quality/extras
    heap_pct: float = float("nan")
    gran_decimals: Optional[int] = None
    gran_step: Optional[float] = None
    bimodal: bool = False
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    # alignment extras
    mem_bytes: int = 0
    mono_inc: bool = False
    mono_dec: bool = False
    dtype_str: str = "numeric"
    corr_top: List[Tuple[str, float]] = field(default_factory=list)
    sample_scale: float = 1.0
    # extremes with indices (global index labels where available)
    min_items: List[Tuple[Any, float]] = field(default_factory=list)
    max_items: List[Tuple[Any, float]] = field(default_factory=list)


class NumericAccumulator:
    """Streaming numeric stats + sampling for quantiles and histogram."""

    def __init__(self, name: str, sample_k: int = 20_000, uniques_k: int = 2048) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self.zeros = 0
        self.negatives = 0
        self._min = math.inf
        self._max = -math.inf
        self._mean = 0.0
        self._m2 = 0.0  # for variance (Welford)
        self._m3 = 0.0
        self._m4 = 0.0
        self._log_sum_pos = 0.0
        self._pos_count = 0
        self._sample = ReservoirSampler(sample_k)
        self._uniques = KMV(uniques_k)
        self.inf = 0
        # track whether all observed values are integer-like (within tiny epsilon)
        self._int_like_all = True
        # streaming memory and monotonicity tracking
        self._bytes_seen = 0
        self._last_val: Optional[float] = None
        self._mono_inc = True
        self._mono_dec = True
        self._dtype_str: str = "numeric"
        self._corr_top: List[Tuple[str, float]] = []
        # Track extremes with indices across chunks (exact mins/maxs)
        self._min_pairs: List[Tuple[Any, float]] = []  # (idx, value)
        self._max_pairs: List[Tuple[Any, float]] = []

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "numeric"

    def set_corr_top(self, items: List[Tuple[str, float]]) -> None:
        self._corr_top = list(items or [])

    def update_extremes(self, pairs_min: List[Tuple[Any, float]], pairs_max: List[Tuple[Any, float]]) -> None:
        # Merge chunk minima
        for idx, v in pairs_min:
            self._min_pairs.append((idx, float(v)))
        # Keep only 5 smallest
        self._min_pairs.sort(key=lambda t: t[1])
        if len(self._min_pairs) > 5:
            del self._min_pairs[5:]
        # Merge chunk maxima
        for idx, v in pairs_max:
            self._max_pairs.append((idx, float(v)))
        # Keep only 5 largest
        self._max_pairs.sort(key=lambda t: -t[1])
        if len(self._max_pairs) > 5:
            del self._max_pairs[5:]
    
    def _combine_moments(self, n2: int, mean2: float, M2_2: float, M3_2: float, M4_2: float) -> None:
        """Merge another group's (n2, mean2, M2_2, M3_2, M4_2) into this accumulator."""
        if n2 <= 0:
            return
        n1 = self.count
        if n1 == 0:
            self.count = int(n2)
            self._mean = float(mean2)
            self._m2 = float(M2_2)
            self._m3 = float(M3_2)
            self._m4 = float(M4_2)
            return

        delta = mean2 - self._mean
        n = n1 + n2
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2

        M2 = self._m2 + M2_2 + delta2 * (n1 * n2 / n)
        M3 = (
            self._m3 + M3_2
            + delta3 * (n1 * n2 * (n1 - n2) / (n * n))
            + 3.0 * delta * (n1 * M2_2 - n2 * self._m2) / n
        )
        M4 = (
            self._m4 + M4_2
            + delta4 * (n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n ** 3))
            + 6.0 * delta2 * (n1 * n1 * M2_2 + n2 * n2 * self._m2) / (n * n)
            + 4.0 * delta * (n1 * M3_2 - n2 * self._m3) / n
        )

        self.count = int(n)
        self._mean = self._mean + (n2 / n) * delta
        self._m2 = float(M2)
        self._m3 = float(M3)
        self._m4 = float(M4)

    def update(self, arr: Sequence[Optional[float]]) -> None:
        """Update accumulator with a chunk of values.

        Preferred path: if `arr` is (or can be) a NumPy float array, compute
        per-chunk stats vectorially and merge them via `_combine_moments`.
        Fallback path: element-wise update (legacy behavior).
        """
        # --- Fast vectorized path (NumPy array of floats with NaN/±inf) ---
        try:
            a = arr if isinstance(arr, np.ndarray) else np.asarray(arr, dtype=float)
        except Exception:
            a = None  # triggers fallback

        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating):
            finite_mask = np.isfinite(a)  # False for NaN and ±inf
            if finite_mask.size == 0:
                return
            nan_mask = np.isnan(a)
            inf_mask = np.isinf(a)
            self.missing += int(nan_mask.sum())
            self.inf += int(inf_mask.sum())

            arr_f = a[finite_mask]
            if arr_f.size == 0:
                return

            # memory accounting (approximate: bytes of finite values processed)
            try:
                self._bytes_seen += int(arr_f.nbytes)
            except Exception:
                self._bytes_seen += int(arr_f.size * 8)

            # Basic counts and extremas on finite values
            n2 = int(arr_f.size)
            min2 = float(np.min(arr_f))
            max2 = float(np.max(arr_f))
            zeros2 = int((arr_f == 0.0).sum())
            neg2 = int((arr_f < 0.0).sum())

            # Geometric mean components (strictly positive)
            pos_mask = arr_f > 0.0
            pos_count2 = int(pos_mask.sum())
            log_sum_pos2 = float(np.log(arr_f[pos_mask]).sum()) if pos_count2 else 0.0

            # Integer-likeness tracking for this chunk
            int_like_chunk = bool(np.all(np.abs(arr_f - np.rint(arr_f)) < 1e-12))
            if self._int_like_all and not int_like_chunk:
                self._int_like_all = False

            # Monotonicity tracking against last value and within chunk (non-strict)
            try:
                if self._last_val is not None and arr_f.size > 0:
                    if arr_f[0] < self._last_val:
                        self._mono_inc = False
                    if arr_f[0] > self._last_val:
                        self._mono_dec = False
                if arr_f.size >= 2:
                    if np.any(arr_f[1:] < arr_f[:-1]):
                        self._mono_inc = False
                    if np.any(arr_f[1:] > arr_f[:-1]):
                        self._mono_dec = False
                if arr_f.size > 0:
                    self._last_val = float(arr_f[-1])
            except Exception:
                pass

            # Reservoir sample & uniques
            try:
                self._sample.add_many(arr_f.tolist())
            except Exception:
                for x in arr_f:
                    self._sample.add(float(x))
            for x in arr_f:
                self._uniques.add(float(x))

            # Per-chunk central moments (about mean)
            mean2 = float(arr_f.mean())
            d = arr_f - mean2
            M2_2 = float(np.sum(d * d))
            M3_2 = float(np.sum(d * d * d))
            M4_2 = float(np.sum(d * d * d * d))

            # Merge moments and update counters
            self._combine_moments(n2, mean2, M2_2, M3_2, M4_2)

            if min2 < self._min:
                self._min = min2
            if max2 > self._max:
                self._max = max2
            self.zeros += zeros2
            self.negatives += neg2
            self._log_sum_pos += log_sum_pos2
            self._pos_count += pos_count2
            return

    # --- Fallback: element-wise update (original logic) ---
        for v in arr:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                self.missing += 1
                continue
            x = float(v)
            if math.isinf(x):
                self.inf += 1
                continue
            # memory accounting (approximate)
            self._bytes_seen += 8
            # integer-likeness tracking
            if self._int_like_all:
                if not (math.isfinite(x) and abs(x - round(x)) < 1e-12):
                    self._int_like_all = False
            if x == 0.0:
                self.zeros += 1
            if x < 0.0:
                self.negatives += 1
            # monotonicity tracking
            if self._last_val is not None:
                if x < self._last_val:
                    self._mono_inc = False
                if x > self._last_val:
                    self._mono_dec = False
            self._last_val = x
            self._min = x if x < self._min else self._min
            self._max = x if x > self._max else self._max
            # Online update for mean, M2, M3, M4 (Pebay 2008)
            n1 = self.count
            n = n1 + 1
            delta = x - self._mean
            delta_n = delta / n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * n1
            m1_old = self._mean
            m2_old = self._m2
            m3_old = self._m3
            m4_old = self._m4
            m4 = m4_old + term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2_old - 4 * delta_n * m3_old
            m3 = m3_old + term1 * delta_n * (n - 2) - 3 * delta_n * m2_old
            m2 = m2_old + term1
            m1 = m1_old + delta_n
            self._mean = m1
            self._m2 = m2
            self._m3 = m3
            self._m4 = m4
            self.count = n
            if x > 0.0:
                self._log_sum_pos += math.log(x)
                self._pos_count += 1
            self._sample.add(x)
            self._uniques.add(x)

    def finalize(self) -> NumericSummary:
        if self.count > 1:
            var = self._m2 / (self.count - 1)
        else:
            var = float("nan")
        std = math.sqrt(var) if var >= 0 and not math.isnan(var) else float("nan")
        svals = sorted(self._sample.values())
        # v2 extras: unique ratio, histogram, top-5
        sample_n = len(svals)
        sample_unique = len(set(svals)) if svals else 0
        unique_ratio_approx = (sample_unique / sample_n) if sample_n else float("nan")
        def _q(p: float) -> float:
            if not svals:
                return float("nan")
            i = (len(svals) - 1) * p
            lo = math.floor(i)
            hi = math.ceil(i)
            if lo == hi:
                return float(svals[int(i)])
            return float(svals[lo] * (hi - i) + svals[hi] * (i - lo))
        q1 = _q(0.25)
        med = _q(0.5)
        q3 = _q(0.75)
        iqr = (q3 - q1) if all(map(math.isfinite, [q1, q3])) else float("nan")
        # MAD from sample (|x - median| median)
        if svals and math.isfinite(med):
            mad = float(np.median([abs(v - med) for v in svals]))
        else:
            mad = float("nan")
        # Histogram preview from sample (25 bins, linear scale)
        if svals:
            try:
                counts, _edges = np.histogram(svals, bins=25)
                hist_counts = [int(c) for c in counts.tolist()]
            except Exception:
                hist_counts = None
        else:
            hist_counts = None
        # Outliers via IQR on sample (scaled by sampling rate)
        if math.isfinite(iqr):
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            lo, hi = float("nan"), float("nan")
        outliers_iqr = 0
        if all(map(math.isfinite, [lo, hi])):
            if sample_n:
                out_s = sum(1 for x in svals if x < lo or x > hi)
                outliers_iqr = int(round(out_s * (self.count / sample_n)))
        unique_est = self._uniques.estimate()
        approx = not self._uniques.is_exact
        # Standard error & coefficient of variation
        se = (std / math.sqrt(self.count)) if (self.count > 1 and not math.isnan(std)) else float("nan")
        cv = (std / self._mean) if (self.count and not math.isnan(std) and self._mean != 0) else float("nan")
        # Geometric mean (only valid for strictly positive data)
        if self._min > 0 and self.count > 0 and self._pos_count == self.count:
            gmean = math.exp(self._log_sum_pos / self.count)
        else:
            gmean = float("nan")
        # Skewness, kurtosis (excess), and Jarque–Bera
        if self.count > 2 and self._m2 > 0 and math.isfinite(self._m2):
            skew = (math.sqrt(self.count) * self._m3) / (self._m2 ** 1.5)
        else:
            skew = float("nan")
        if self.count > 3 and self._m2 > 0 and math.isfinite(self._m2):
            kurt = (self.count * self._m4) / (self._m2 * self._m2) - 3.0
        else:
            kurt = float("nan")
        if all(map(lambda v: isinstance(v, (int, float)) and not math.isnan(v), [self.count, skew, kurt])) and self.count > 3:
            jb = float(self.count / 6.0 * (skew * skew + 0.25 * (kurt * kurt)))
        else:
            jb = float("nan")
        # Approximate Top-5 for discrete integer-like series
        top_values: List[Tuple[float, int]] = []
        try:
            # Heuristic: treat as discrete if all seen are int-like and small-ish cardinality
            is_discrete = bool(self._int_like_all and ((unique_est <= max(1, min(50, int(0.05 * max(1, self.count))))) or (sample_unique and (sample_unique <= 50 and unique_ratio_approx <= 0.05))))
            if is_discrete and svals:
                scale = (self.count / sample_n) if sample_n else 1.0
                ctr = Counter(svals)
                top = ctr.most_common(5)
                top_values = [(float(round(v)), int(round(c * scale))) for v, c in top]
        except Exception:
            top_values = []

        # 95% CI for mean
        if isinstance(se, float) and math.isfinite(se):
            ci_lo = float(self._mean - 1.96 * se)
            ci_hi = float(self._mean + 1.96 * se)
        else:
            ci_lo = float("nan"); ci_hi = float("nan")

        # Heaping / round-number bias (% at round values) — sample-based
        heap_pct = float("nan")
        try:
            if svals:
                if self._int_like_all:
                    vv = [int(round(v)) for v in svals]
                    if vv:
                        hits = sum(1 for v in vv if (v % 10 == 0) or (v % 5 == 0))
                        heap_pct = hits / len(vv) * 100.0
                else:
                    frac = [abs(v - round(v)) for v in svals]
                    # also catch .5 rounding
                    half = [abs((v - math.floor(v)) - 0.5) for v in svals]
                    tol = 0.002
                    hits = sum(1 for a,b in zip(frac, half) if (a <= tol) or (b <= tol))
                    heap_pct = hits / len(svals) * 100.0
        except Exception:
            pass

        # Granularity (decimals) and step size (approx)
        gran_decimals = None
        gran_step = None
        try:
            if svals:
                if not self._int_like_all:
                    # estimate modal count of decimals from string repr
                    decs = []
                    for x in svals[: min(50_000, len(svals))]:
                        xs = str(x)
                        if "." in xs:
                            decs.append(len(xs.split(".")[-1]))
                        else:
                            decs.append(0)
                    if decs:
                        # mode
                        counts = Counter(decs)
                        gran_decimals = int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])
                # step ≈ median of positive diffs between unique sorted values (rounded if we have decimals)
                arr_g = np.unique(np.asarray(svals, dtype=float))
                if gran_decimals is not None and gran_decimals >= 0:
                    arr_g = np.round(arr_g, int(min(12, gran_decimals)))
                diffs = np.diff(np.sort(arr_g))
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size > 0:
                    gran_step = float(np.median(diffs))
        except Exception:
            pass

        # Bimodality hint via valley detection on binned counts (sample-based)
        bimodal = False
        try:
            if svals and len(svals) >= 10:
                nb = min(25, max(10, int(math.sqrt(len(svals)))))
                c_bi, _e_bi = np.histogram(svals, bins=nb)
                if c_bi.size >= 3:
                    valley = (c_bi[1:-1] < c_bi[:-2]) & (c_bi[1:-1] < c_bi[2:])
                    bimodal = bool((valley & (c_bi[1:-1] <= 0.8 * c_bi.max())).sum() >= 1)
        except Exception:
            bimodal = False

        sample_scale = (self.count / sample_n) if sample_n else 1.0
        return NumericSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            unique_est=unique_est,
            mean=self._mean if self.count else float("nan"),
            std=std,
            variance=var,
            se=se,
            cv=cv,
            gmean=gmean,
            min=self._min if self.count else float("nan"),
            q1=q1,
            median=med,
            q3=q3,
            iqr=iqr,
            mad=mad,
            skew=skew,
            kurtosis=kurt,
            jb_chi2=jb,
            max=self._max if self.count else float("nan"),
            zeros=self.zeros,
            negatives=self.negatives,
            outliers_iqr=outliers_iqr,
            approx=approx,
            inf=self.inf,
            int_like=bool(self._int_like_all),
            unique_ratio_approx=unique_ratio_approx,
            hist_counts=hist_counts,
            top_values=top_values,
            sample_vals=svals,
            heap_pct=heap_pct,
            gran_decimals=gran_decimals,
            gran_step=gran_step,
            bimodal=bimodal,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            mem_bytes=int(self._bytes_seen),
            mono_inc=bool(self._mono_inc),
            mono_dec=bool(self._mono_dec),
            dtype_str=self._dtype_str,
            corr_top=list(self._corr_top),
            sample_scale=float(sample_scale),
            min_items=list(self._min_pairs),
            max_items=list(self._max_pairs),
        )


@dataclass
class BooleanSummary:
    name: str
    count: int
    missing: int
    true_n: int
    false_n: int
    mem_bytes: int = 0
    dtype_str: str = "boolean"


class BooleanAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self.true_n = 0
        self.false_n = 0
        self._dtype_str: str = "boolean"
        self._mem_bytes: int = 0

    def update(self, arr: Sequence[Any]) -> None:
        for v in arr:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                self.missing += 1
                continue
            if bool(v):
                self.true_n += 1
            else:
                self.false_n += 1
            self.count += 1

    def finalize(self) -> BooleanSummary:
        # Approximate memory: ~1 byte per boolean cell (very rough; marked as ≈ in UI)
        mem_bytes = int(self._mem_bytes if self._mem_bytes > 0 else (self.count + self.missing))
        return BooleanSummary(self.name, self.count, self.missing, self.true_n, self.false_n, mem_bytes, self._dtype_str)

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "boolean"

    def add_mem(self, n: int) -> None:
        try:
            self._mem_bytes += int(n)
        except Exception:
            pass


@dataclass
class CategoricalSummary:
    name: str
    count: int
    missing: int
    unique_est: int
    top_items: List[Tuple[str, int]]
    approx: bool
    # extras for alignment
    mem_bytes: int = 0
    avg_len: Optional[float] = None
    len_p90: Optional[int] = None
    empty_zero: int = 0
    case_variants_est: int = 0
    trim_variants_est: int = 0
    dtype_str: str = "categorical"


class CategoricalAccumulator:
    def __init__(self, name: str, topk_k: int = 50, uniques_k: int = 2048) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self._uniques = KMV(uniques_k)
        self._uniques_lower = KMV(uniques_k)
        self._uniques_strip = KMV(uniques_k)
        self.topk = MisraGries(topk_k)
        self._len_sum = 0
        self._len_n = 0
        self._len_sample = ReservoirSampler(5000)
        self._empty_zero = 0
        self._bytes_seen = 0
        self._dtype_str: str = "categorical"

    def update(self, arr: Sequence[Any]) -> None:
        for v in arr:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                self.missing += 1
                continue
            self.count += 1
            s = str(v)
            self.topk.add(s, 1)
            self._uniques.add(s)
            try:
                self._uniques_lower.add(s.lower())
                self._uniques_strip.add(s.strip())
            except Exception:
                pass
            L = len(s)
            self._len_sum += L
            self._len_n += 1
            self._len_sample.add(float(L))
            if L == 0:
                self._empty_zero += 1
            self._bytes_seen += L  # approximate

    def finalize(self) -> CategoricalSummary:
        items = self.topk.items()[:10]
        # cast keys to str
        top_items = [(str(k), int(c)) for k, c in items]
        return CategoricalSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            unique_est=self._uniques.estimate(),
            top_items=top_items,
            approx=not self._uniques.is_exact,
            mem_bytes=int(self._bytes_seen),
            avg_len=(self._len_sum / self._len_n) if self._len_n else None,
            len_p90=(int(np.quantile(self._len_sample.values(), 0.90)) if self._len_sample.values() else None),
            empty_zero=int(self._empty_zero),
            case_variants_est=max(0, int(self._uniques.estimate() - self._uniques_lower.estimate())),
            trim_variants_est=max(0, int(self._uniques.estimate() - self._uniques_strip.estimate())),
            dtype_str=self._dtype_str,
        )

    @property
    def avg_len(self) -> Optional[float]:
        return (self._len_sum / self._len_n) if self._len_n else None

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "categorical"


@dataclass
class DatetimeSummary:
    name: str
    count: int
    missing: int
    min_ts: Optional[int]
    max_ts: Optional[int]
    by_hour: List[int]  # 24 counts
    by_dow: List[int]   # 7 counts, Monday=0
    by_month: List[int] # 12 counts, Jan=1 index => store 12-length
    # v2 additions
    dtype_str: str = "datetime"
    mono_inc: bool = False
    mono_dec: bool = False
    mem_bytes: int = 0
    sample_ts: Optional[List[int]] = None
    sample_scale: float = 1.0


class DatetimeAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self._min_ts: Optional[int] = None
        self._max_ts: Optional[int] = None
        self.by_hour = [0] * 24
        self.by_dow = [0] * 7
        self.by_month = [0] * 12
        self._uniques = KMV(2048)
        self._dtype_str: str = "datetime"
        self._mem_bytes: int = 0
        self._sample = ReservoirSampler(20_000)
        self._last_ts: Optional[int] = None
        self._mono_inc = True
        self._mono_dec = True

    def update(self, arr_ns: Sequence[Optional[int]]) -> None:
        # arr_ns: integer nanoseconds since epoch; None/NaN for missing
        for v in arr_ns:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                self.missing += 1
                continue
            ts = int(v)
            self.count += 1
            if self._min_ts is None or ts < self._min_ts:
                self._min_ts = ts
            if self._max_ts is None or ts > self._max_ts:
                self._max_ts = ts
            self._uniques.add(ts)
            # sample and monotonicity
            self._sample.add(float(ts))
            if self._last_ts is not None:
                if ts < self._last_ts:
                    self._mono_inc = False
                if ts > self._last_ts:
                    self._mono_dec = False
            self._last_ts = ts
            # breakdowns (UTC)
            try:
                dt = datetime.utcfromtimestamp(ts / 1_000_000_000)
                self.by_hour[dt.hour] += 1
                self.by_dow[dt.weekday()] += 1
                self.by_month[dt.month - 1] += 1
            except Exception:
                # ignore ill-formed
                pass

    def finalize(self) -> DatetimeSummary:
        svals = [int(x) for x in self._sample.values()]
        sample_scale = (self.count / len(svals)) if svals else 1.0
        return DatetimeSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            min_ts=self._min_ts,
            max_ts=self._max_ts,
            by_hour=self.by_hour,
            by_dow=self.by_dow,
            by_month=self.by_month,
            dtype_str=self._dtype_str,
            mono_inc=bool(self._mono_inc),
            mono_dec=bool(self._mono_dec),
            mem_bytes=int(self._mem_bytes),
            sample_ts=svals,
            sample_scale=float(sample_scale),
        )

    @property
    def unique_est(self) -> int:
        return self._uniques.estimate()

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "datetime"

    def add_mem(self, n: int) -> None:
        try:
            self._mem_bytes += int(n)
        except Exception:
            pass


# =============================
# Chunk iterators (sources)
# =============================

FrameLike = Any


def _iter_chunks_from_path(path: str, *, chunk_size: int = 200_000, engine: str = "auto",
                           csv_kwargs: Optional[Mapping[str, Any]] = None,
                           parquet_columns: Optional[Sequence[str]] = None) -> Iterator[FrameLike]:
    """Yield chunked frames from a file path.

    Supports CSV via pandas; Parquet via pyarrow. Keeps memory bounded by chunk_size/batch_size.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv", ".txt"}:
        if pd is None:
            raise RuntimeError("pandas is required for CSV chunking but is not installed")
        kwargs = dict(
            chunksize=int(chunk_size),
            low_memory=True,
        )
        if csv_kwargs:
            kwargs.update(csv_kwargs)
        # heuristic: delimiter by extension
        if ext == ".tsv":
            kwargs.setdefault("sep", "\t")
        yield from pd.read_csv(path, **kwargs)  # type: ignore[misc]
        return
    if ext in {".parquet", ".pq"}:
        if pq is None:
            raise RuntimeError("pyarrow is required for Parquet streaming but is not installed")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk_size, columns=list(parquet_columns) if parquet_columns else None):
            # Convert to pandas DataFrame to reuse normalization functions. This keeps memory bounded.
            yield batch.to_pandas(types_mapper=None)
        return
    raise ValueError(f"Unsupported file extension: {ext}")


# =============================
# Normalization per chunk (backend mini-adapter)
# =============================

@dataclass
class ColumnKinds:
    numeric: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)


def _infer_kinds_pandas(df: "pd.DataFrame") -> ColumnKinds:  # type: ignore[name-defined]
    kinds = ColumnKinds()
    for name, s in df.items():
        dt = str(getattr(s, "dtype", "object"))
        if re.search("int|float|^UInt|^Int|^Float", dt, re.I):
            kinds.numeric.append(name)
        elif re.search("bool", dt, re.I):
            kinds.boolean.append(name)
        elif re.search("datetime", dt, re.I):
            kinds.datetime.append(name)
        else:
            # try coercions on a small sample for robustness
            sample = s.head(10_000)
            if pd is not None:
                # datetime?
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        ds = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
                    except TypeError:
                        ds = pd.to_datetime(sample, errors="coerce", utc=True)
                if ds.notna().sum() >= max(10, int(0.7 * len(sample))):
                    kinds.datetime.append(name)
                    continue
                ns = pd.to_numeric(sample, errors="coerce")
                if ns.notna().sum() >= max(10, int(0.7 * len(sample))):
                    kinds.numeric.append(name)
                    continue
            # boolean-like?
            uniq = set(map(str, sample.dropna().unique().tolist()))
            if 1 <= len(uniq) <= 2 and uniq.issubset({"True", "False", "0", "1", "true", "false"}):
                kinds.boolean.append(name)
            else:
                kinds.categorical.append(name)
    return kinds


def _to_numeric_array_pandas(s: "pd.Series") -> np.ndarray:  # type: ignore[name-defined]
    ns = pd.to_numeric(s, errors="coerce")
    # Return as float64 NumPy array with NaN for invalids to enable vectorized stats.
    return ns.to_numpy(dtype="float64", copy=False)


def _to_bool_array_pandas(s: "pd.Series") -> List[Optional[bool]]:  # type: ignore[name-defined]
    # Try best-effort conversion
    if str(s.dtype).startswith("bool"):
        arr = s.astype("boolean").tolist()
        return [None if x is pd.NA else bool(x) for x in arr]
    # strings like "true"/"false"/"1"/"0"
    def _coerce(v: Any) -> Optional[bool]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        vs = str(v).strip().lower()
        if vs in {"true", "1", "t", "yes", "y"}:
            return True
        if vs in {"false", "0", "f", "no", "n"}:
            return False
        return None
    return [_coerce(v) for v in s.tolist()]


def _to_datetime_ns_array_pandas(s: "pd.Series") -> List[Optional[int]]:  # type: ignore[name-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            ds = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        except TypeError:
            ds = pd.to_datetime(s, errors="coerce", utc=True)
    # Convert NaT to None using int64 representation
    vals = ds.astype("int64", copy=False).tolist()
    NAT_INT = -9223372036854775808
    out: List[Optional[int]] = []
    for v in vals:
        out.append(None if v == NAT_INT else int(v))
    return out


def _to_categorical_iter_pandas(s: "pd.Series") -> Iterable[Any]:  # type: ignore[name-defined]
    return s.tolist()


# === ID helper used by classic template cards ===

def _safe_col_id(name: str) -> str:
    return "col_" + "".join(ch if str(ch).isalnum() else "_" for ch in str(name))



def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NaN"
    try:
        return f"{x:,.4g}"
    except Exception:
        return str(x)

# Human‑readable bytes (align with classic report)
def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(max(0, n))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024.0


# Tiny spark histogram SVG for numeric preview
def _spark_hist_svg(counts: Optional[List[int]], width: int = 180, height: int = 48, pad: int = 2) -> str:
    if not counts:
        return '<svg class="spark spark-hist" width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg>'.format(w=width, h=height)
    max_c = max(counts) or 1
    bar_w = (width - 2 * pad) / len(counts)
    parts = [f'<svg class="spark spark-hist" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="distribution">']
    for i, c in enumerate(counts):
        h = 0 if max_c == 0 else (c / max_c) * (height - 2 * pad)
        x = pad + i * bar_w
        y = height - pad - h
        parts.append(f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-1,1):.2f}" height="{h:.2f}" rx="1" ry="1"></rect>')
    parts.append("</svg>")
    return "".join(parts)



# === Histogram with axes (classic-compatible) helpers ===

def _nice_num(rng: float, do_round: bool = True) -> float:
    import math
    if rng <= 0 or not np.isfinite(rng):
        return 1.0
    exp = math.floor(math.log10(rng))
    frac = rng / (10 ** exp)
    if do_round:
        if frac < 1.5:
            nice = 1
        elif frac < 3:
            nice = 2
        elif frac < 7:
            nice = 5
        else:
            nice = 10
    else:
        if frac <= 1:
            nice = 1
        elif frac <= 2:
            nice = 2
        elif frac <= 5:
            nice = 5
        else:
            nice = 10
    return nice * (10 ** exp)


def _nice_ticks(vmin: float, vmax: float, n: int = 5):
    import math
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        vmax = vmin + 1
    rng = _nice_num(vmax - vmin, do_round=False)
    step = _nice_num(rng / max(1, n - 1), do_round=True)
    nice_min = math.floor(vmin / step) * step
    nice_max = math.ceil(vmax / step) * step
    ticks = []
    t = nice_min
    while t <= nice_max + step * 1e-9 and len(ticks) < 50:
        ticks.append(t)
        t += step
    return ticks, step


def _fmt_tick(v: float, step: float) -> str:
    if not np.isfinite(v):
        return ''
    # integers with thousands separators when step is coarse enough
    if step >= 1:
        i = int(round(v))
        if abs(i) >= 1000:
            return f"{i:,}"
        return f"{i}"
    if step >= 0.1:
        return f"{v:.1f}"
    if step >= 0.01:
        return f"{v:.2f}"
    try:
        return f"{v:.4g}"
    except Exception:
        return str(v)


def _svg_empty(css_class: str, width: int, height: int, aria_label: str = "no data") -> str:
    return f'<svg class="{css_class}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="{aria_label}"></svg>'


def _fmt_compact(x) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
    except Exception:
        pass
    try:
        return f"{x:.4g}"
    except Exception:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return str(x)


def _build_hist_svg_from_vals(
    base_title: str,
    vals: Sequence[float],
    *,
    bins: int = 25,
    width: int = 420,
    height: int = 160,
    margin_left: int = 45,
    margin_bottom: int = 36,
    margin_top: int = 8,
    margin_right: int = 8,
    scale: str = "lin",
    scale_count: float = 1.0,
    x_min_override: Optional[float] = None,
    x_max_override: Optional[float] = None,
) -> str:
    # Prepare values (drop NaN/inf, optional log10+ for 'log')
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _svg_empty("hist-svg", width, height)
    if scale == "log":
        arr = arr[arr > 0]
        if arr.size == 0:
            return _svg_empty("hist-svg", width, height)
        arr = np.log10(arr)

    x_min, x_max = float(np.min(arr)), float(np.max(arr))
    if x_min_override is not None and np.isfinite(x_min_override):
        x_min = float(x_min_override)
    if x_max_override is not None and np.isfinite(x_max_override):
        x_max = float(x_max_override)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    counts, edges = np.histogram(arr, bins=int(bins), range=(x_min, x_max))
    counts_scaled = np.maximum(0, np.round(counts * max(1.0, float(scale_count)))).astype(int)
    y_max = int(max(1, counts_scaled.max()))
    total_n = int(counts_scaled.sum()) if counts_scaled.size else 0

    iw = width - margin_left - margin_right
    ih = height - margin_top - margin_bottom

    def sx(x):
        return margin_left + (x - x_min) / (x_max - x_min) * iw

    def sy(y):
        return margin_top + (1 - y / y_max) * ih

    x_ticks, x_step = _nice_ticks(x_min, x_max, 6)
    # Ensure first x tick sits exactly at the y-axis (x_min) and no tick before
    xt = [x for x in x_ticks if x >= x_min - 1e-9 and x <= x_max + 1e-9]
    if not xt or abs(xt[0] - x_min) > 1e-9:
        xt = [x_min] + [x for x in xt if x > x_min]
    x_ticks = xt
    y_ticks, y_step = _nice_ticks(0, y_max, 5)

    parts = [
        f'<svg class="hist-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Histogram">',
        '<g class="plot-area">'
    ]

    for i, c in enumerate(counts_scaled):
        x0 = edges[i]
        x1 = edges[i + 1]
        x = sx(x0)
        w = max(1.0, sx(x1) - sx(x0) - 1.0)
        y = sy(int(c))
        h = (margin_top + ih) - y
        pct = (c / total_n * 100.0) if total_n else 0.0
        parts.append(
            f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="1" ry="1" '
            f'data-count="{int(c)}" data-pct="{pct:.1f}" data-x0="{_fmt_compact(x0)}" data-x1="{_fmt_compact(x1)}">'
            f'<title>{int(c)} rows ({pct:.1f}%)&#10;[{_fmt_compact(x0)} – {_fmt_compact(x1)}]</title>'
            f'</rect>'
        )
    parts.append('</g>')

    x_axis_y = margin_top + ih
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left + iw}" y2="{x_axis_y}"></line>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')

    for xt in x_ticks:
        px = sx(xt)
        parts.append(f'<line class="tick" x1="{px}" y1="{x_axis_y}" x2="{px}" y2="{x_axis_y + 4}"></line>')
        parts.append(f'<text class="tick-label" x="{px}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_tick(xt, x_step)}</text>')
    for yt in y_ticks:
        py = sy(yt)
        parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py}" x2="{margin_left}" y2="{py}"></line>')
        parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3}" text-anchor="end">{_fmt_tick(yt, y_step)}</text>')

    x_title = f"log10({base_title})" if scale == "log" else base_title
    parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">{x_title}</text>')
    parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">Count</text>')

    parts.append('</svg>')
    return ''.join(parts)


# === Public builder helpers (parity with report.py) ===
def build_hist_svg_with_axes(
    s: "pd.Series",  # type: ignore[name-defined]
    bins: int = 20,
    width: int = 420,
    height: int = 160,
    margin_left: int = 45,
    margin_bottom: int = 36,
    margin_top: int = 8,
    margin_right: int = 8,
    sample_cap: Optional[int] = 200_000,
    scale: str = "lin",
    auto_bins: bool = True,
) -> str:
    """Public wrapper for numeric histogram with axes (API-compatible with report.py).

    Accepts a pandas Series, coerces to numeric, optionally samples, and renders an SVG.
    """
    try:
        ss = s.dropna()
        if ss.empty:
            return _svg_empty("hist-svg", width, height)
        vals = pd.to_numeric(ss, errors="coerce").dropna().to_numpy()  # type: ignore[name-defined]
        if vals.size == 0:
            return _svg_empty("hist-svg", width, height)
        if scale == "log":
            vals = vals[vals > 0]
            if vals.size == 0:
                return _svg_empty("hist-svg", width, height)
        # Optional sampling cap
        if sample_cap is not None and vals.size > int(sample_cap):
            try:
                rng = np.random.default_rng(0)
                idx = rng.choice(vals.size, size=int(sample_cap), replace=False)
                vals = vals[idx]
            except Exception:
                vals = vals[: int(sample_cap)]
        # Freedman–Diaconis bins
        if auto_bins and vals.size > 1:
            try:
                q1 = float(np.quantile(vals, 0.25))
                q3 = float(np.quantile(vals, 0.75))
                iqr_local = q3 - q1
                if iqr_local > 0:
                    h = 2.0 * iqr_local * (vals.size ** (-1.0 / 3.0))
                    if h > 0:
                        fd_bins = int(np.clip(np.ceil((float(np.max(vals)) - float(np.min(vals))) / h), 10, 200))
                        bins = fd_bins
            except Exception:
                pass
        name = str(getattr(s, "name", "Value"))
        return _build_hist_svg_from_vals(
            name, vals, bins=int(bins), width=width, height=height,
            margin_left=margin_left, margin_bottom=margin_bottom,
            margin_top=margin_top, margin_right=margin_right,
            scale=scale,
        )
    except Exception:
        return _svg_empty("hist-svg", width, height)


def build_cat_bar_svg(
    s: "pd.Series",  # type: ignore[name-defined]
    top: int = 10,
    width: int = 420,
    height: int = 160,
    margin_left: int = 120,
    margin_right: int = 12,
    margin_top: int = 8,
    margin_bottom: int = 8,
    scale: str = "count",
    include_other: bool = True,
) -> str:
    """Public wrapper for categorical Top-N bar chart (API-compatible with report.py)."""
    try:
        s2 = s.astype(str).dropna()
        if s2.empty:
            return _svg_empty("cat-svg", width, height)
        vc = s2.value_counts()
        if vc.empty:
            return _svg_empty("cat-svg", width, height)
        items: List[Tuple[str, int]] = []
        if int(top) > 0:
            head = vc.head(int(top))
            items = [(str(k), int(v)) for k, v in head.items()]
            if include_other and len(vc) > int(top):
                other = int(vc.iloc[int(top):].sum())
                items.append(("Other", other))
        else:
            items = [(str(k), int(v)) for k, v in vc.items()]
        total = int(vc.sum())
        return _build_cat_bar_svg_from_items(items, total=total, scale=scale)
    except Exception:
        return _svg_empty("cat-svg", width, height)


def build_dt_line_svg(
    ts: "pd.Series",  # type: ignore[name-defined]
    bins: int = 60,
    width: int = 420,
    height: int = 160,
    margin_left: int = 45,
    margin_right: int = 8,
    margin_top: int = 8,
    margin_bottom: int = 32,
) -> str:
    """Public wrapper for the datetime timeline SVG (API-compatible with report.py)."""
    try:
        ts2 = ts.dropna()
        if ts2.empty:
            return _svg_empty("dt-svg", width, height)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ds = pd.to_datetime(ts2, errors="coerce", utc=True)  # type: ignore[name-defined]
        vals = ds.dropna().astype("int64").to_numpy()
        if vals.size == 0:
            return _svg_empty("dt-svg", width, height)
        tmin = int(np.min(vals))
        tmax = int(np.max(vals))
        if tmin == tmax:
            tmax = tmin + 1
        bins = int(max(10, min(bins, max(10, min(vals.size, 180)))))
        counts, edges = np.histogram(vals, bins=bins, range=(tmin, tmax))
        y_max = int(max(1, counts.max()))
        iw = width - margin_left - margin_right
        ih = height - margin_top - margin_bottom
        def sx(x):
            return margin_left + (x - tmin) / (tmax - tmin) * iw
        def sy(y):
            return margin_top + (1 - y / y_max) * ih
        centers = (edges[:-1] + edges[1:]) / 2.0
        pts = " ".join(f"{sx(x):.2f},{sy(float(c)):.2f}" for x, c in zip(centers, counts))
        y_ticks, _ = _nice_ticks(0, y_max, 5)
        n_xt = 5
        xt_vals = np.linspace(tmin, tmax, n_xt)
        span_ns = tmax - tmin
        def _fmt_xt(v):
            try:
                tsv = pd.to_datetime(int(v))  # type: ignore[name-defined]
                if span_ns <= 3 * 24 * 3600 * 1e9:
                    return tsv.strftime('%Y-%m-%d %H:%M')
                return tsv.date().isoformat()
            except Exception:
                return str(v)
        parts = [
            f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Timeline">',
            '<g class="plot-area">'
        ]
        for yt in y_ticks:
            parts.append(f'<line class="grid" x1="{margin_left}" y1="{sy(yt):.2f}" x2="{margin_left + iw}" y2="{sy(yt):.2f}"></line>')
        parts.append(f'<polyline class="line" points="{pts}"></polyline>')
        parts.append('<g class="hotspots">')
        for i, c in enumerate(counts):
            if not np.isfinite(c):
                continue
            x0p = sx(edges[i])
            x1p = sx(edges[i+1])
            wp = max(1.0, x1p - x0p)
            cp = (edges[i] + edges[i+1]) / 2.0
            label = _fmt_xt(cp)
            title = f"{int(c)} rows&#10;{label}"
            parts.append(
                f'<rect class="hot" x="{x0p:.2f}" y="{margin_top}" width="{wp:.2f}" height="{ih:.2f}" fill="transparent" pointer-events="all">'
                f'<title>{title}</title>'
                f'</rect>'
            )
        parts.append('</g>')
        parts.append('</g>')
        x_axis_y = margin_top + ih
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left+iw}" y2="{x_axis_y}"></line>')
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')
        for yt in y_ticks:
            py = sy(yt)
            parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py:.2f}" x2="{margin_left}" y2="{py:.2f}"></line>')
            lab = int(round(yt))
            parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3:.2f}" text-anchor="end">{lab}</text>')
        for xv in xt_vals:
            px = sx(xv)
            parts.append(f'<line class="tick" x1="{px:.2f}" y1="{x_axis_y}" x2="{px:.2f}" y2="{x_axis_y + 4}"></line>')
            parts.append(f'<text class="tick-label" x="{px:.2f}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_xt(xv)}</text>')
        parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">Time</text>')
        parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">Count</text>')
        parts.append('</svg>')
        return ''.join(parts)
    except Exception:
        return _svg_empty("dt-svg", width, height)


def _build_cat_bar_svg_from_items(
    items: List[Tuple[str, int]],
    total: int,
    *,
    width: int = 420,
    height: int = 160,
    margin_top: int = 8,
    margin_bottom: int = 8,
    margin_left_min: int = 120,
    margin_right: int = 12,
    scale: str = "count",
) -> str:
    if total <= 0 or not items:
        return _svg_empty("cat-svg", width, height)
    labels = [_html.escape(str(k)) for k, _ in items]
    counts = [int(c) for _, c in items]
    pcts = [(c / total * 100.0) for c in counts]

    max_label_len = max((len(l) for l in labels), default=0)
    char_w = 7
    gutter = max(60, min(180, char_w * min(max_label_len, 28) + 16))
    mleft = max(margin_left_min, gutter)

    n = len(labels)
    iw = width - mleft - margin_right
    ih = height - margin_top - margin_bottom
    if n <= 0 or iw <= 0 or ih <= 0:
        return _svg_empty("cat-svg", width, height)
    bar_gap = 6
    bar_h = max(4, (ih - bar_gap * (n - 1)) / max(n, 1))

    if scale == "pct":
        vmax = max(pcts) or 1.0
        values = pcts
    else:
        vmax = float(max(counts)) or 1.0
        values = counts

    def sx(v: float) -> float:
        return mleft + (v / vmax) * iw

    parts = [f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Top categories">']
    for i, (label, c, p, val) in enumerate(zip(labels, counts, pcts, values)):
        y = margin_top + i * (bar_h + bar_gap)
        x0 = mleft
        x1 = sx(float(val))
        w = max(1.0, x1 - x0)
        short = (label[:24] + "…") if len(label) > 24 else label
        parts.append(
            f'<g class="bar-row">'
            f'<rect class="bar" x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{bar_h:.2f}" rx="2" ry="2">'
            f'<title>{label}\n{c:,} rows ({p:.1f}%)</title>'
            f'</rect>'
            f'<text class="bar-label" x="{mleft-6}" y="{y + bar_h/2 + 3:.2f}" text-anchor="end">{short}</text>'
            f"<text class=\"bar-value\" x=\"{(x1 - 6 if w >= 56 else x1 + 4):.2f}\" y=\"{y + bar_h/2 + 3:.2f}\" text-anchor=\"{('end' if w >= 56 else 'start')}\">{c:,} ({p:.1f}%)</text>"
            f'</g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_numeric_card(s: NumericSummary) -> str:
    col_id = _safe_col_id(s.name)
    safe_name = _html.escape(str(s.name))
    # severity classes
    miss_pct = (s.missing / max(1, s.count + s.missing)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    zeros_pct = (s.zeros / max(1, s.count)) * 100.0 if s.count else 0.0
    neg_pct = (s.negatives / max(1, s.count)) * 100.0 if s.count else 0.0
    out_pct = (s.outliers_iqr / max(1, s.count)) * 100.0 if s.count else 0.0
    zeros_cls = 'warn' if zeros_pct > 30 else ''
    neg_cls = 'warn' if 0 < neg_pct <= 10 else ('crit' if neg_pct > 10 else '')
    out_cls = 'crit' if out_pct > 1 else ('warn' if out_pct > 0.3 else '')
    inf_cls = 'crit' if s.inf else ''
    approx_badge = '<span class="badge">approx</span>' if s.approx else ''

    # Discrete heuristic (approximate)
    discrete = False
    try:
        if s.int_like:
            if (s.unique_est <= max(1, min(50, int(0.05 * max(1, s.count))))) or (isinstance(s.unique_ratio_approx, float) and not math.isnan(s.unique_ratio_approx) and s.unique_ratio_approx <= 0.05):
                discrete = True
    except Exception:
        discrete = False

    # Quality flags (classic‑style)
    flag_items = []
    if miss_pct > 0:
        flag_items.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    if s.inf:
        flag_items.append('<li class="flag bad">Has ∞</li>')
    if neg_pct > 0:
        cls = 'warn' if neg_pct > 10 else ''
        flag_items.append(f'<li class="flag {cls}">Has negatives</li>' if cls else '<li class="flag">Has negatives</li>')
    if zeros_pct >= 50.0:
        flag_items.append('<li class="flag bad">Zero‑inflated</li>')
    elif zeros_pct >= 30.0:
        flag_items.append('<li class="flag warn">Zero‑inflated</li>')
    if isinstance(s.min, (int, float)) and math.isfinite(s.min) and s.min > 0:
        flag_items.append('<li class="flag good">Positive‑only</li>')
    if isinstance(s.skew, float) and math.isfinite(s.skew):
        if s.skew >= 1:
            flag_items.append('<li class="flag warn">Skewed Right</li>')
        elif s.skew <= -1:
            flag_items.append('<li class="flag warn">Skewed Left</li>')
    if isinstance(s.kurtosis, float) and math.isfinite(s.kurtosis) and abs(s.kurtosis) >= 3:
        flag_items.append('<li class="flag bad">Heavy‑tailed</li>')
    if isinstance(s.jb_chi2, float) and math.isfinite(s.jb_chi2) and s.jb_chi2 <= 5.99:
        flag_items.append('<li class="flag good">≈ Normal (JB)</li>')
    if discrete:
        flag_items.append('<li class="flag warn">Discrete</li>')
    # Extras: bimodality, heaping, log-scale hint
    if isinstance(s.heap_pct, float) and math.isfinite(s.heap_pct) and s.heap_pct >= 30.0:
        flag_items.append('<li class="flag">Heaping</li>')
    if getattr(s, 'bimodal', False):
        flag_items.append('<li class="flag warn">Possibly bimodal</li>')
    if (isinstance(s.min, (int,float)) and math.isfinite(s.min) and s.min > 0) and (isinstance(s.skew, float) and math.isfinite(s.skew) and s.skew >= 1):
        flag_items.append('<li class="flag good">Log‑scale?</li>')
    # Alignment extras: constant/quasi‑constant/outlier flags/monotonic
    try:
        uniq_est = max(0, int(s.unique_est))
        total_nonnull = max(1, int(s.count))
        unique_ratio = (uniq_est / total_nonnull) if total_nonnull else 0.0
        if uniq_est == 1:
            flag_items.append('<li class="flag bad">Constant</li>')
        elif unique_ratio <= 0.02 or uniq_est <= 2:
            flag_items.append('<li class="flag warn">Quasi‑constant</li>')
        if out_pct > 1.0:
            flag_items.append('<li class="flag bad">Many outliers</li>')
        elif out_pct > 0.3:
            flag_items.append('<li class="flag warn">Some outliers</li>')
        if total_nonnull > 1 and s.mono_inc:
            flag_items.append('<li class="flag good">Monotonic ↑</li>')
        elif total_nonnull > 1 and s.mono_dec:
            flag_items.append('<li class="flag good">Monotonic ↓</li>')
    except Exception:
        pass
    quality_flags_html = f"<ul class='quality-flags'>{''.join(flag_items)}</ul>" if flag_items else ""

    mem_display = _human_bytes(int(getattr(s, 'mem_bytes', 0)))
    inf_pct = (s.inf / max(1, s.count)) * 100.0 if s.count else 0.0
    left_tbl = f"""
    <table class="kv"><tbody>
      <tr><th>Count</th><td class="num">{s.count:,}</td></tr>
      <tr><th>Unique</th><td class="num">{s.unique_est:,}{' (≈)' if s.approx else ''}</td></tr>
      <tr><th>Missing</th><td class="num {miss_cls}">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Outliers</th><td class="num {out_cls}">{s.outliers_iqr:,} ({out_pct:.1f}%)</td></tr>
      <tr><th>Zeros</th><td class="num {zeros_cls}">{s.zeros:,} ({zeros_pct:.1f}%)</td></tr>
      <tr><th>Infinites</th><td class="num {inf_cls}">{s.inf:,} ({inf_pct:.1f}%)</td></tr>
      <tr><th>Negatives</th><td class="num {neg_cls}">{s.negatives:,} ({neg_pct:.1f}%)</td></tr>
    </tbody></table>
    """

    right_tbl = f"""
    <table class="kv"><tbody>
      <tr><th>Min</th><td class="num">{_fmt_num(s.min)}</td></tr>
      <tr><th>Median</th><td class="num">{_fmt_num(s.median)}</td></tr>
      <tr><th>Mean</th><td class="num">{_fmt_num(s.mean)}</td></tr>
      <tr><th>Max</th><td class="num">{_fmt_num(s.max)}</td></tr>
      <tr><th>Q1</th><td class="num">{_fmt_num(s.q1)}</td></tr>
      <tr><th>Q3</th><td class="num">{_fmt_num(s.q3)}</td></tr>
      <tr><th>Processed bytes</th><td class="num">{mem_display} (≈)</td></tr>
    </tbody></table>
    """

    # Quantile grid (sample-based percentiles)
    svals = getattr(s, "sample_vals", None) or []
    def _q_from_sample(p: float) -> float:
        if not svals:
            return float("nan")
        i = (len(svals) - 1) * p
        lo = math.floor(i); hi = math.ceil(i)
        if lo == hi:
            return float(svals[int(i)])
        return float(svals[lo] * (hi - i) + svals[hi] * (i - lo))
    p1 = _q_from_sample(0.01)
    p5 = _q_from_sample(0.05)
    p10 = _q_from_sample(0.10)
    p90 = _q_from_sample(0.90)
    p95 = _q_from_sample(0.95)
    p99 = _q_from_sample(0.99)
    range_val = (s.max - s.min) if (isinstance(s.max, (int,float)) and isinstance(s.min, (int,float))) else float('nan')

    quant_stats_table = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Min</th><td class=\"num\">{_fmt_num(s.min)}</td></tr>
      <tr><th>P1</th><td class=\"num\">{_fmt_num(p1)}</td></tr>
      <tr><th>P5</th><td class=\"num\">{_fmt_num(p5)}</td></tr>
      <tr><th>P10</th><td class=\"num\">{_fmt_num(p10)}</td></tr>
      <tr><th>Q1 (P25)</th><td class=\"num\">{_fmt_num(s.q1)}</td></tr>
      <tr><th>Median (P50)</th><td class=\"num\">{_fmt_num(s.median)}</td></tr>
      <tr><th>Q3 (P75)</th><td class=\"num\">{_fmt_num(s.q3)}</td></tr>
      <tr><th>P90</th><td class=\"num\">{_fmt_num(p90)}</td></tr>
      <tr><th>P95</th><td class=\"num\">{_fmt_num(p95)}</td></tr>
      <tr><th>P99</th><td class=\"num\">{_fmt_num(p99)}</td></tr>
      <tr><th>Max</th><td class=\"num\">{_fmt_num(s.max)}</td></tr>
      <tr><th>Range</th><td class=\"num\">{_fmt_num(range_val)}</td></tr>
      <tr><th>IQR</th><td class=\"num\">{_fmt_num(s.iqr)}</td></tr>
    </tbody></table>
    """

    details_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Mean</th><td class=\"num\">{_fmt_num(s.mean)}</td></tr>
      <tr><th>Std</th><td class=\"num\">{_fmt_num(s.std)}</td></tr>
      <tr><th>Variance</th><td class=\"num\">{_fmt_num(s.variance)}</td></tr>
      <tr><th>SE (mean)</th><td class=\"num\">{_fmt_num(s.se)}</td></tr>
      <tr><th>95% CI (mean)</th><td class=\"num\">{_fmt_num(s.ci_lo)} – {_fmt_num(s.ci_hi)}</td></tr>
      <tr><th>Coeff. Variation</th><td class=\"num\">{_fmt_num(s.cv)}</td></tr>
      <tr><th>Geo-Mean</th><td class=\"num\">{_fmt_num(s.gmean)}</td></tr>
      <tr><th>MAD</th><td class=\"num\">{_fmt_num(s.mad)}</td></tr>
      <tr><th>Skewness</th><td class=\"num\">{_fmt_num(s.skew)}</td></tr>
      <tr><th>Kurtosis (excess)</th><td class=\"num\">{_fmt_num(s.kurtosis)}</td></tr>
      <tr><th>Jarque–Bera χ²</th><td class=\"num\">{_fmt_num(s.jb_chi2)}</td></tr>
      <tr><th>Heaping at round values</th><td class=\"num\">{_fmt_num(s.heap_pct)}%</td></tr>
      <tr><th>Granularity (decimals)</th><td class=\"num\">{s.gran_decimals if s.gran_decimals is not None else '—'}</td></tr>
      <tr><th>Step ≈</th><td class=\"num\">{_fmt_num(s.gran_step)}</td></tr>
      <tr><th>Bimodality hint</th><td>{'Yes' if s.bimodal else 'No'}</td></tr>
    </tbody></table>
    """

    # Extremes with row indices (from tracked minima/maxima across chunks)
    extremes_section = ""
    try:
        mins_pairs = getattr(s, 'min_items', [])
        maxs_pairs = getattr(s, 'max_items', [])
        if mins_pairs:
            rows_min = ''.join(f'<tr><th><code>{_html.escape(str(i))}</code></th><td class="num">{_fmt_num(v)}</td></tr>' for i, v in mins_pairs)
            min_table = f'<section class="subtable"><h4 class="small muted">Top 5 minima</h4><table class="kv"><tbody>{rows_min}</tbody></table></section>'
        else:
            min_table = ''
        if maxs_pairs:
            rows_max = ''.join(f'<tr><th><code>{_html.escape(str(i))}</code></th><td class="num">{_fmt_num(v)}</td></tr>' for i, v in maxs_pairs)
            max_table = f'<section class="subtable"><h4 class="small muted">Top 5 maxima</h4><table class="kv"><tbody>{rows_max}</tbody></table></section>'
        else:
            max_table = ''
        out_table = ''
        try:
            if math.isfinite(s.iqr):
                lo_b = s.q1 - 1.5 * s.iqr
                hi_b = s.q3 + 1.5 * s.iqr
                cand = mins_pairs + maxs_pairs
                outs = [(i, v, (lo_b - v) if v < lo_b else (v - hi_b)) for (i, v) in cand if (v < lo_b or v > hi_b)]
                outs.sort(key=lambda t: t[2], reverse=True)
                outs = outs[:5]
                if outs:
                    rows_out = ''.join(f'<tr><th><code>{_html.escape(str(i))}</code></th><td class="num">{_fmt_num(v)}</td></tr>' for i, v, _d in outs)
                    out_table = f'<section class="subtable"><h4 class="small muted">Top outliers</h4><table class="kv"><tbody>{rows_out}</tbody></table></section>'
        except Exception:
            pass
        extremes_section = min_table + max_table + out_table
    except Exception:
        extremes_section = ""

    # Build classic histogram variants (from the reservoir sample)
    sample_vals = getattr(s, "sample_vals", None) or []
    has_sample = isinstance(sample_vals, (list, tuple)) and len(sample_vals) > 0

    if has_sample:
        scale_factor = float(s.sample_scale) if hasattr(s, 'sample_scale') else 1.0
        lin10 = _build_hist_svg_from_vals(s.name, sample_vals, bins=10, scale="lin", scale_count=scale_factor, x_min_override=s.min, x_max_override=s.max)
        lin25 = _build_hist_svg_from_vals(s.name, sample_vals, bins=25, scale="lin", scale_count=scale_factor, x_min_override=s.min, x_max_override=s.max)
        lin50 = _build_hist_svg_from_vals(s.name, sample_vals, bins=50, scale="lin", scale_count=scale_factor, x_min_override=s.min, x_max_override=s.max)
    else:
        empty_svg = _svg_empty("hist-svg", 420, 160)
        lin10 = lin25 = lin50 = empty_svg

    positive_only = bool(isinstance(s.min, (int, float)) and math.isfinite(s.min) and s.min > 0)
    if positive_only and has_sample:
        scale_factor = float(s.sample_scale) if hasattr(s, 'sample_scale') else 1.0
        log10_svg = _build_hist_svg_from_vals(s.name, sample_vals, bins=10, scale="log", scale_count=scale_factor, x_min_override=(math.log10(s.min) if isinstance(s.min,(int,float)) and s.min>0 else None), x_max_override=(math.log10(s.max) if isinstance(s.max,(int,float)) and s.max>0 else None))
        log25_svg = _build_hist_svg_from_vals(s.name, sample_vals, bins=25, scale="log", scale_count=scale_factor, x_min_override=(math.log10(s.min) if isinstance(s.min,(int,float)) and s.min>0 else None), x_max_override=(math.log10(s.max) if isinstance(s.max,(int,float)) and s.max>0 else None))
        log50_svg = _build_hist_svg_from_vals(s.name, sample_vals, bins=50, scale="log", scale_count=scale_factor, x_min_override=(math.log10(s.min) if isinstance(s.min,(int,float)) and s.min>0 else None), x_max_override=(math.log10(s.max) if isinstance(s.max,(int,float)) and s.max>0 else None))
        log_variants = "".join([
            f'<div id="{col_id}-log-bins-10" class="hist variant" style="display:none">{log10_svg}</div>',
            f'<div id="{col_id}-log-bins-25" class="hist variant" style="display:none">{log25_svg}</div>',
            f'<div id="{col_id}-log-bins-50" class="hist variant" style="display:none">{log50_svg}</div>',
        ])
        scale_group = '<div class="scale-group"><button type="button" class="btn-soft btn-scale active" data-scale="lin">Linear</button><button type="button" class="btn-soft btn-scale" data-scale="log">Log</button></div>'
    else:
        log_variants = ""
        scale_group = ""

    hist_variants_html = f'''
      <div class="hist-variants">
        <div id="{col_id}-lin-bins-10" class="hist variant" style="display:none">{lin10}</div>
        <div id="{col_id}-lin-bins-25" class="hist variant">{lin25}</div>
        <div id="{col_id}-lin-bins-50" class="hist variant" style="display:none">{lin50}</div>
        {log_variants}
      </div>
    '''

    top5_section = ""
    try:
        if getattr(s, "top_values", None):
            rows = ''.join(f'<tr><th>{_fmt_num(v)}</th><td class="num">{c:,} ({(c/max(1,s.count)*100):.1f}%)</td></tr>' for v, c in s.top_values)
            top5_section = f'<div class="box" style="margin-top:8px;"><h4 class="small muted">Top 5 values</h4><table class="kv"><tbody>{rows}</tbody></table></div>'
    except Exception:
        top5_section = ""

    controls_html = f'''
      <div class="card-controls" role="group" aria-label="Column controls">
        <div class="details-slot">
          <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
        </div>
        <div class="controls-slot">
          <div class="hist-controls" data-col="{col_id}" data-scale="lin" data-bin="25" role="group" aria-label="Histogram controls">
            <div class="center-controls">
              {scale_group}
              <div class="bin-group">
                <button type="button" class="btn-soft btn-bins" data-bin="10">10</button>
                <button type="button" class="btn-soft btn-bins active" data-bin="25">25</button>
                <button type="button" class="btn-soft btn-bins" data-bin="50">50</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    '''

    details_section = f'''
      <section id="{col_id}-details" class="details-section" hidden>
        <nav class="tabs" role="tablist" aria-label="More details">
          <button role="tab" class="active" data-tab="stats">Statistics</button>
          <button role="tab" data-tab="values">Common values</button>
        </nav>
        <div class="tab-panes">
          <section class="tab-pane active" data-tab="stats">
            <div class="grid-2col">{quant_stats_table}{details_tbl}</div>
          </section>
          <section class="tab-pane" data-tab="values">
            {top5_section or '<p class="muted small">No common values available.</p>'}
          </section>
          <section class="tab-pane" data-tab="extremes">
            {extremes_section or '<p class="muted small">No extreme values found.</p>'}
          </section>
        </div>
      </section>
    '''

    # Correlation chips (optional, if provided in summary)
    corr_section_html = ""
    try:
        chips = []
        for other, r in (s.corr_top or [])[:3]:
            cls = 'good' if abs(r) >= 0.8 else 'warn'
            sign = '+' if r >= 0 else '−'
            chips.append(f'<li class="flag {cls}"><code>{_html.escape(str(other))}</code> {sign}{abs(r):.2f}</li>')
        if chips:
            corr_section_html = f'<section class="subtable"><h4 class="small muted">Top correlations</h4><ul class="quality-flags">{"".join(chips)}</ul></section>'
    except Exception:
        corr_section_html = ""

    return f"""
    <article class="var-card" id="{col_id}"> 
      <header class="var-card__header">
        <div class="title"><span class="colname" title="{safe_name}">{safe_name}</span>
          <span class="badge">Numeric</span>
          <span class="dtype chip">{s.dtype_str}</span>
          {approx_badge}
          {quality_flags_html}
        </div>
      </header>
      <div class="var-card__body">
        <div class="triple-row">
          <div class="box stats-left">{left_tbl}</div>
          <div class="box stats-right">{right_tbl}</div>
          <div class="box chart">
            {hist_variants_html}
          </div>
        </div>
        {controls_html}
        {details_section}
        {corr_section_html}
      </div>
    </article>
    """


def _render_bool_card(s: BooleanSummary) -> str:
    col_id = _safe_col_id(s.name)
    total = int(s.true_n + s.false_n + s.missing)
    cnt = int(s.true_n + s.false_n)
    miss_pct = (s.missing / max(1, total)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    true_pct_total = (s.true_n / max(1, total)) * 100.0
    false_pct_total = (s.false_n / max(1, total)) * 100.0

    mem_display = _human_bytes(getattr(s, 'mem_bytes', 0)) + ' (≈)'

    # Flags
    flags = []
    if miss_pct > 0:
        flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    if cnt > 0 and (s.true_n == 0 or s.false_n == 0):
        flags.append('<li class="flag bad">Constant</li>')
    if cnt > 0:
        p = s.true_n / cnt
        if p <= 0.05 or p >= 0.95:
            flags.append('<li class="flag warn">Imbalanced</li>')
    quality_flags_html = f"<ul class='quality-flags'>{''.join(flags)}</ul>" if flags else ""

    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{cnt:,}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Unique</th><td class=\"num\">{(int(s.true_n>0)+int(s.false_n>0))}</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display}</td></tr>
    </tbody></table>
    """
    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>True</th><td class=\"num\">{s.true_n:,} ({true_pct_total:.1f}%)</td></tr>
      <tr><th>False</th><td class=\"num\">{s.false_n:,} ({false_pct_total:.1f}%)</td></tr>
    </tbody></table>
    """

    # Stacked boolean bar (false | true | missing)
    def _bool_stack_svg(true_n: int, false_n: int, miss: int,
                        width: int = 420, height: int = 48, margin: int = 4) -> str:
        total = max(1, int(true_n + false_n + miss))
        inner_w = width - 2 * margin
        seg_h = height - 2 * margin
        fw = (false_n / total) * inner_w
        tw = (true_n  / total) * inner_w
        mw = (miss    / total) * inner_w
        parts = [f'<svg class="bool-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Boolean distribution">']
        x = float(margin)
        def seg(css: str, n: int, w: float):
            nonlocal x
            if w <= 0 or n <= 0:
                return
            pct = (n / total * 100.0)
            parts.append(
                f'<rect class="seg {css}" x="{x:.2f}" y="{margin}" width="{w:.2f}" height="{seg_h:.2f}" rx="2" ry="2">'
                f'<title>{n:,} {css.title()} ({pct:.1f}%)</title>'
                f'</rect>'
            )
            label = f"{pct:.1f}%" if w >= 28 else ''
            if w >= 80:
                label = f"{n:,} ({pct:.1f}%)"
            if label:
                cx = x + w / 2.0
                cy = margin + seg_h / 2.0 + 4
                parts.append(f'<text class="seg-label" x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600;">{label}</text>')
            x += w
        seg('false', false_n, fw)
        seg('true',  true_n,  tw)
        seg('missing', miss,   mw)
        parts.append('</svg>')
        return ''.join(parts)

    chart_html = _bool_stack_svg(s.true_n, s.false_n, s.missing)

    # Suggestions (approx)
    suggestions = []
    if cnt > 0 and (s.true_n == 0 or s.false_n == 0):
        suggestions.append("Constant boolean → consider dropping (no variance).")
    if miss_pct > 0:
        suggestions.append("Missing values present → consider explicit missing indicator.")
    if cnt > 0:
        p = s.true_n / cnt
        if p <= 0.05 or p >= 0.95:
            suggestions.append("Severely imbalanced → consider class weighting.")
    suggestions_html = "<ul class=\"suggestions\">" + "".join(f"<li>{x}</li>" for x in suggestions) + "</ul>" if suggestions else "<p class=\"muted small\">No specific suggestions.</p>"

    return f"""
    <article class=\"var-card\" id=\"{col_id}\"> 
      <header class=\"var-card__header\"><div class=\"title\"><span class=\"colname\">{_html.escape(str(s.name))}</span>
        <span class=\"badge\">Boolean</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {quality_flags_html}
      </div></header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}</div>
          <div class=\"box chart\">{chart_html}</div>
        </div>
        <div class=\"card-controls\" role=\"group\" aria-label=\"Column controls\">
          <div class=\"details-slot\">
            <button type=\"button\" class=\"details-toggle btn-soft\" aria-controls=\"{col_id}-details\" aria-expanded=\"false\">Details</button>
          </div>
          <div class=\"controls-slot\"></div>
        </div>
        <section id=\"{col_id}-details\" class=\"details-section\" hidden>
          <nav class=\"tabs\" role=\"tablist\" aria-label=\"More details\">
            <button role=\"tab\" class=\"active\" data-tab=\"stats\">Statistics</button>
            <button role=\"tab\" data-tab=\"suggestions\">Suggestions</button>
          </nav>
          <div class=\"tab-panes\">
            <section class=\"tab-pane active\" data-tab=\"stats\"><div class=\"grid-2col\">{left_tbl}{right_tbl}</div></section>
            <section class=\"tab-pane\" data-tab=\"suggestions\">{suggestions_html}</section>
          </div>
        </section>
      </div>
    </article>
    """


def _render_cat_card(s: CategoricalSummary) -> str:
    col_id = _safe_col_id(s.name)
    safe_name = _html.escape(str(s.name))
    total = s.count + s.missing
    miss_pct = (s.missing / max(1, total)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    approx_badge = '<span class="badge">approx</span>' if s.approx else ''

    # Mode and basic coverage metrics from top_items (approx)
    mode_label, mode_n = (s.top_items[0] if s.top_items else ("—", 0))
    safe_mode_label = _html.escape(str(mode_label))
    mode_pct = (mode_n / max(1, s.count)) * 100.0 if s.count else 0.0
    # Entropy from top_items distribution (approx)
    if s.count > 0 and s.top_items:
        probs = [c / s.count for _, c in s.top_items]
        entropy = float(-sum(p * math.log2(max(p, 1e-12)) for p in probs))
    else:
        entropy = float('nan')
    # Rare levels and coverage (<1% of non-null)
    rare_count = 0
    rare_cov = 0.0
    if s.count > 0:
        for _, c in s.top_items:
            pct = c / s.count * 100.0
            if pct < 1.0:
                rare_count += 1
                rare_cov += pct
    rare_cls = 'crit' if rare_cov > 60 else ('warn' if rare_cov >= 30 else '')
    # Top 5 coverage
    top5_cov = 0.0
    if s.count > 0 and s.top_items:
        top5_cov = sum(c for _, c in s.top_items[:5]) / s.count * 100.0
    top5_cls = 'good' if top5_cov >= 80 else ('warn' if top5_cov <= 40 else '')
    empty_cls = 'warn' if s.empty_zero > 0 else ''

    # Flags (align with v1; approximate where needed)
    flags = []
    if s.unique_est > max(200, int(0.5 * max(1, s.count))):
        flags.append('<li class="flag warn">High cardinality</li>')
    if mode_n >= int(0.7 * max(1, s.count)) and s.count:
        flags.append('<li class="flag warn">Dominant category</li>')
    if rare_cov >= 30.0:
        flags.append('<li class="flag warn">Many rare levels</li>')
    if s.case_variants_est > 0:
        flags.append('<li class="flag">Case variants</li>')
    if s.trim_variants_est > 0:
        flags.append('<li class="flag">Trim variants</li>')
    if s.empty_zero > 0:
        flags.append('<li class="flag">Empty strings</li>')
    if miss_pct > 0:
        flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    quality_flags_html = f"<ul class=\"quality-flags\">{''.join(flags)}</ul>" if flags else ""

    mem_display = _human_bytes(int(getattr(s, 'mem_bytes', 0)))
    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{s.count:,}</td></tr>
      <tr><th>Unique</th><td class=\"num\">{s.unique_est:,}{' (≈)' if s.approx else ''}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Mode</th><td><code>{safe_mode_label}</code></td></tr>
      <tr><th>Mode %</th><td class=\"num\">{mode_pct:.1f}%</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display} (≈)</td></tr>
    </tbody></table>
    """

    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Entropy</th><td class=\"num\">{_fmt_num(entropy)}</td></tr>
      <tr><th>Rare levels</th><td class=\"num {rare_cls}\">{rare_count:,} ({rare_cov:.1f}%)</td></tr>
      <tr><th>Top 5 coverage</th><td class=\"num {top5_cls}\">{top5_cov:.1f}%</td></tr>
      <tr><th>Label length (avg)</th><td class=\"num\">{_fmt_num(s.avg_len)}</td></tr>
      <tr><th>Length p90</th><td class=\"num\">{s.len_p90 if s.len_p90 is not None else '—'}</td></tr>
      <tr><th>Empty strings</th><td class=\"num {empty_cls}\">{s.empty_zero:,}</td></tr>
    </tbody></table>
    """

    # Levels table (Top 20, approximate)
    rows = []
    for label, c in (s.top_items[:20] if s.top_items else []):
        pct = c / max(1, s.count + s.missing) * 100.0
        rows.append(f'<tr><td><code>{_html.escape(str(label))}</code></td><td class="num">{c:,}</td><td class="num">{pct:.1f}%</td></tr>')
    levels_table_html = f'<table class="kv"><thead><tr><th>Level</th><th>Count</th><th>%</th></tr></thead><tbody>{"".join(rows) or "<tr><td colspan=3>—</td></tr>"}</tbody></table>'

    # Build Top-N bar chart variants
    items = s.top_items or []
    maxN = max(1, min(15, len(items)))
    candidates = [5, 10, 15, maxN]
    topn_list = sorted({n for n in candidates if 1 <= n <= maxN})
    default_topn = 10 if 10 in topn_list else (max(topn_list) if topn_list else maxN)

    variants_html_parts = []
    for n in topn_list:
        if len(items) > n:
            keep = max(1, n - 1)
            head = items[:keep]
            other = sum(c for _, c in items[keep:])
            data = head + [("Other", other)]
        else:
            data = items[:n]
        svg = _build_cat_bar_svg_from_items(data, total=max(1, s.count + s.missing), scale="count")
        style = "" if n == default_topn else "display:none"
        variants_html_parts.append(f'<div id="{col_id}-cat-top-{n}" class="cat variant" style="{style}">{svg}</div>')
    variants_html = "".join(variants_html_parts)
    bin_buttons = "".join(
        f'<button type="button" class="btn-soft btn-bins{(" active" if n == default_topn else "" )}" data-topn="{n}">Top {n}</button>'
        for n in topn_list
    )

    card_html = f"""
    <article class=\"var-card\" id=\"{col_id}\">
      <header class=\"var-card__header\"> 
        <div class=\"title\"><span class=\"colname\" title=\"{safe_name}\">{safe_name}</span>
        <span class=\"badge\">Categorical</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {approx_badge}
        {quality_flags_html}
        </div>
      </header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}</div>
          <div class=\"box chart\"> <div class=\"hist-variants\">{variants_html}</div> </div>
        </div>
        <div class=\"card-controls\" role=\"group\" aria-label=\"Categorical controls\">
          <div class=\"details-slot\">
            <button type=\"button\" class=\"details-toggle btn-soft\" aria-controls=\"{col_id}-details\" aria-expanded=\"false\">Details</button>
          </div>
          <div class=\"controls-slot\">
            <div class=\"hist-controls\" data-col=\"{col_id}\" data-topn=\"{default_topn}\" role=\"group\" aria-label=\"Categorical controls\">
              <div class=\"center-controls\"><div class=\"bin-group\">{bin_buttons}</div></div>
            </div>
          </div>
        </div>
        <section id=\"{col_id}-details\" class=\"details-section\" hidden>
          <nav class=\"tabs\" role=\"tablist\" aria-label=\"More details\">
            <button role=\"tab\" class=\"active\" data-tab=\"levels\">Levels</button>
            <button role=\"tab\" data-tab=\"quality\">Quality</button>
          </nav>
          <div class=\"tab-panes\">
            <section class=\"tab-pane active\" data-tab=\"levels\">{levels_table_html}</section>
            <section class=\"tab-pane\" data-tab=\"quality\">
              <table class=\"kv\"><tbody>
                <tr><th>Case variants (groups)</th><td class=\"num\">{s.case_variants_est:,}</td></tr>
                <tr><th>Trim variants (groups)</th><td class=\"num\">{s.trim_variants_est:,}</td></tr>
                <tr><th>Empty strings</th><td class=\"num\">{s.empty_zero:,}</td></tr>
                <tr><th>Unique ratio</th><td class=\"num\">{(s.unique_est/max(1,s.count)):.2f}</td></tr>
              </tbody></table>
            </section>
          </div>
        </section>
      </div>
    </article>
    """
    return card_html


def _render_dt_card(s: DatetimeSummary) -> str:
    col_id = _safe_col_id(s.name)
    safe_name = _html.escape(str(s.name))
    miss_pct = (s.missing / max(1, s.count + s.missing)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    flags = []
    if miss_pct > 0:
        flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    if s.count > 1 and s.mono_inc:
        flags.append('<li class="flag good">Monotonic ↑</li>')
    if s.count > 1 and s.mono_dec:
        flags.append('<li class="flag good">Monotonic ↓</li>')
    quality_flags_html = f"<ul class='quality-flags'>{''.join(flags)}</ul>" if flags else ""
    def _fmt_ts(ts: Optional[int]) -> str:
        if ts is None:
            return '—'
        try:
            return datetime.utcfromtimestamp(ts / 1_000_000_000).isoformat() + 'Z'
        except Exception:
            return str(ts)
    mem_display = _human_bytes(getattr(s, 'mem_bytes', 0)) + ' (≈)'
    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{s.count:,}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Min</th><td>{_fmt_ts(s.min_ts)}</td></tr>
      <tr><th>Max</th><td>{_fmt_ts(s.max_ts)}</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display}</td></tr>
    </tbody></table>
    """
    # simple ascii sparks for hour/dow/month
    def spark(counts: List[int]) -> str:
        if not counts:
            return ''
        m = max(counts) or 1
        blocks = '▁▂▃▄▅▆▇█'
        levels = [blocks[min(len(blocks)-1, int(c * (len(blocks)-1) / m))] for c in counts]
        return ''.join(levels)
    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Hour</th><td class=\"small\">{spark(s.by_hour)}</td></tr>
      <tr><th>Day of week</th><td class=\"small\">{spark(s.by_dow)}</td></tr>
      <tr><th>Month</th><td class=\"small\">{spark(s.by_month)}</td></tr>
    </tbody></table>
    """
    # Build timeline SVG from sample (scaled to approximate counts)
    def _dt_line_svg_from_sample(sample: Optional[List[int]], tmin: Optional[int], tmax: Optional[int], bins: int = 60, scale_count: float = 1.0) -> str:
        if not sample or tmin is None or tmax is None:
            return _svg_empty("dt-svg", 420, 160)
        a = np.asarray(sample, dtype=np.int64)
        if a.size == 0:
            return _svg_empty("dt-svg", 420, 160)
        if tmin == tmax:
            tmax = tmin + 1
        counts, edges = np.histogram(a, bins=int(max(10, min(bins, 180))), range=(int(tmin), int(tmax)))
        counts = np.maximum(0, np.round(counts * max(1.0, float(scale_count)))).astype(int)
        y_max = int(max(1, counts.max()))
        width, height = 420, 160
        margin_left, margin_right, margin_top, margin_bottom = 45, 8, 8, 32
        iw = width - margin_left - margin_right
        ih = height - margin_top - margin_bottom
        def sx(x):
            return margin_left + (x - tmin) / (tmax - tmin) * iw
        def sy(y):
            return margin_top + (1 - y / y_max) * ih
        centers = (edges[:-1] + edges[1:]) / 2.0
        pts = " ".join(f"{sx(x):.2f},{sy(float(c)):.2f}" for x, c in zip(centers, counts))
        y_ticks, _ = _nice_ticks(0, y_max, 5)
        n_xt = 5
        xt_vals = np.linspace(tmin, tmax, n_xt)
        span_ns = tmax - tmin
        def _fmt_xt(v):
            try:
                ts = pd.to_datetime(int(v))
                if span_ns <= 3 * 24 * 3600 * 1e9:
                    return ts.strftime('%Y-%m-%d %H:%M')
                return ts.date().isoformat()
            except Exception:
                return str(v)
        parts = [
            f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Timeline">',
            '<g class="plot-area">'
        ]
        for yt in y_ticks:
            parts.append(f'<line class="grid" x1="{margin_left}" y1="{sy(yt):.2f}" x2="{margin_left + iw}" y2="{sy(yt):.2f}"></line>')
        parts.append(f'<polyline class="line" points="{pts}"></polyline>')
        parts.append('<g class="hotspots">')
        for i, c in enumerate(counts):
            if not np.isfinite(c):
                continue
            x0p = sx(edges[i])
            x1p = sx(edges[i+1])
            wp = max(1.0, x1p - x0p)
            cp = (edges[i] + edges[i+1]) / 2.0
            label = _fmt_xt(cp)
            title = f"{int(c)} rows&#10;{label}"
            parts.append(
                f'<rect class="hot" x="{x0p:.2f}" y="{margin_top}" width="{wp:.2f}" height="{ih:.2f}" fill="transparent" pointer-events="all">'
                f'<title>{title}</title>'
                f'</rect>'
            )
        parts.append('</g>')
        parts.append('</g>')
        x_axis_y = margin_top + ih
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left+iw}" y2="{x_axis_y}"></line>')
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')
        for yt in y_ticks:
            py = sy(yt)
            parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py:.2f}" x2="{margin_left}" y2="{py:.2f}"></line>')
            lab = int(round(yt))
            parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3:.2f}" text-anchor="end">{lab}</text>')
        for xv in xt_vals:
            px = sx(xv)
            parts.append(f'<line class="tick" x1="{px:.2f}" y1="{x_axis_y}" x2="{px:.2f}" y2="{x_axis_y + 4}"></line>')
            parts.append(f'<text class="tick-label" x="{px:.2f}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_xt(xv)}</text>')
        parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">Time</text>')
        parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">Count</text>')
        parts.append('</svg>')
        return ''.join(parts)

    chart_html = _dt_line_svg_from_sample(s.sample_ts, s.min_ts, s.max_ts, bins=60, scale_count=getattr(s, 'sample_scale', 1.0))

    # Details: full breakdown tables
    hours_tbl = ''.join(f'<tr><th>{h:02d}</th><td class="num">{c:,}</td></tr>' for h, c in enumerate(s.by_hour))
    dows = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    dows_tbl = ''.join(f'<tr><th>{dows[i]}</th><td class="num">{c:,}</td></tr>' for i, c in enumerate(s.by_dow))
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_tbl = ''.join(f'<tr><th>{months[i]}</th><td class="num">{c:,}</td></tr>' for i, c in enumerate(s.by_month))

    details_html = f'''
      <section id="{col_id}-details" class="details-section" hidden>
        <nav class="tabs" role="tablist" aria-label="More details">
          <button role="tab" class="active" data-tab="breakdown">Breakdown</button>
        </nav>
        <div class="tab-panes">
          <section class="tab-pane active" data-tab="breakdown">
            <div class="grid-2col">
              <table class="kv"><thead><tr><th>Hour</th><th>Count</th></tr></thead><tbody>{hours_tbl}</tbody></table>
              <table class="kv"><thead><tr><th>Day</th><th>Count</th></tr></thead><tbody>{dows_tbl}</tbody></table>
            </div>
            <div class="grid-2col" style="margin-top:8px;">
              <table class="kv"><thead><tr><th>Month</th><th>Count</th></tr></thead><tbody>{months_tbl}</tbody></table>
            </div>
          </section>
        </div>
      </section>
    '''

    return f"""
    <article class=\"var-card\" id=\"{col_id}\"> 
      <header class=\"var-card__header\"><div class=\"title\"><span class=\"colname\">{safe_name}</span>
        <span class=\"badge\">Datetime</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {quality_flags_html}
      </div></header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}</div>
          <div class=\"box chart\">{chart_html}</div>
        </div>
        <div class=\"card-controls\" role=\"group\" aria-label=\"Column controls\">
          <div class=\"details-slot\">
            <button type=\"button\" class=\"details-toggle btn-soft\" aria-controls=\"{col_id}-details\" aria-expanded=\"false\">Details</button>
          </div>
          <div class=\"controls-slot\"></div>
        </div>
        {details_html}
      </div>
    </article>
    """


@dataclass
class ReportConfig:
    title: str = "PySuricata EDA Report (streaming)"
    chunk_size: int = 200_000
    numeric_sample_k: int = 20_000
    uniques_k: int = 2048
    topk_k: int = 50
    engine: str = "auto"  # reserved for future (e.g., force polars)
    csv_kwargs: Optional[Mapping[str, Any]] = None
    parquet_columns: Optional[Sequence[str]] = None
    # Logging
    logger: Optional[logging.Logger] = None
    log_level: int = logging.INFO
    log_every_n_chunks: int = 1  # set >1 to reduce verbosity on huge runs
    include_sample: bool = True
    sample_rows: int = 10
    # Correlations (optional, lightweight)
    compute_correlations: bool = True
    corr_threshold: float = 0.6
    corr_max_cols: int = 50
    corr_max_per_col: int = 2

    # Checkpointing
    checkpoint_every_n_chunks: int = 0          # 0 disables
    checkpoint_dir: Optional[str] = None        # default: dirname(output_file) or CWD
    checkpoint_prefix: str = "pysuricata_ckpt"
    checkpoint_write_html: bool = False         # also dump partial HTML next to pickle
    checkpoint_max_to_keep: int = 3             # rotate old checkpoints


@dataclass
class _State:
    kinds: ColumnKinds
    accs: Dict[str, Any]  # name -> accumulator


def _build_accumulators(kinds: ColumnKinds, cfg: ReportConfig) -> Dict[str, Any]:
    accs: Dict[str, Any] = {}
    for name in kinds.numeric:
        accs[name] = NumericAccumulator(name, sample_k=cfg.numeric_sample_k, uniques_k=cfg.uniques_k)
    for name in kinds.boolean:
        accs[name] = BooleanAccumulator(name)
    for name in kinds.datetime:
        accs[name] = DatetimeAccumulator(name)
    for name in kinds.categorical:
        accs[name] = CategoricalAccumulator(name, topk_k=cfg.topk_k, uniques_k=cfg.uniques_k)
    return accs


# === Checkpointing helpers ===

class _CheckpointManager:
    def __init__(self, directory: str, prefix: str = "pysuricata_ckpt", keep: int = 3, write_html: bool = False) -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.prefix = prefix
        self.keep = max(1, int(keep))
        self.write_html = write_html

    def _glob(self, ext: str) -> List[str]:
        return sorted(glob.glob(os.path.join(self.directory, f"{self.prefix}_chunk*.{ext}")))

    def _path_for(self, chunk_idx: int, ext: str) -> str:
        return os.path.join(self.directory, f"{self.prefix}_chunk{chunk_idx:06d}.{ext}")

    def rotate(self) -> None:
        # Keep only the newest `keep` checkpoint files (pkl.gz). Remove older siblings (.html too).
        pkls = self._glob("pkl.gz")
        if len(pkls) <= self.keep:
            return
        to_remove = pkls[: len(pkls) - self.keep]
        for p in to_remove:
            try:
                os.remove(p)
            except Exception:
                pass
            # remove matching html if present
            html_p = p.replace(".pkl.gz", ".html")
            try:
                if os.path.exists(html_p):
                    os.remove(html_p)
            except Exception:
                pass

    def save(self, chunk_idx: int, state: Mapping[str, Any], html: Optional[str] = None) -> Tuple[str, Optional[str]]:
        pkl_path = self._path_for(chunk_idx, "pkl.gz")
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        html_path = None
        if self.write_html and html is not None:
            html_path = self._path_for(chunk_idx, "html")
            with open(html_path, "w", encoding="utf-8") as hf:
                hf.write(html)
        self.rotate()
        return pkl_path, html_path


def _make_state_snapshot(
    *, kinds: ColumnKinds, accs: Dict[str, Any], row_kmv: "_RowKMV",
    total_missing_cells: int, approx_mem_bytes: int, chunk_idx: int,
    first_columns: List[str], sample_section_html: str, cfg: ReportConfig
) -> Dict[str, Any]:
    # NOTE: Keep the snapshot pickle‑friendly. Avoid putting whole DataFrames here.
    return {
        "version": 1,
        "timestamp": time.time(),
        "chunk_idx": int(chunk_idx),
        "first_columns": list(first_columns),
        "sample_section_html": sample_section_html,
        "kinds": kinds,
        "accs": accs,
        "row_kmv": row_kmv,
        "total_missing_cells": int(total_missing_cells),
        "approx_mem_bytes": int(approx_mem_bytes),
        "config": {
            "title": cfg.title,
            "chunk_size": cfg.chunk_size,
            "numeric_sample_k": cfg.numeric_sample_k,
            "uniques_k": cfg.uniques_k,
            "topk_k": cfg.topk_k,
            "compute_correlations": cfg.compute_correlations,
            "corr_threshold": cfg.corr_threshold,
        },
    }


def _render_html_snapshot(
    *, kinds: ColumnKinds, accs: Dict[str, Any], first_columns: List[str],
    row_kmv: "_RowKMV", total_missing_cells: int, approx_mem_bytes: int,
    start_time: float, cfg: ReportConfig, report_title: Optional[str],
    sample_section_html: str
) -> str:
    # ---- Build kinds map
    kinds_map = {
        **{name: ("numeric", accs[name]) for name in kinds.numeric},
        **{name: ("categorical", accs[name]) for name in kinds.categorical},
        **{name: ("datetime", accs[name]) for name in kinds.datetime},
        **{name: ("boolean", accs[name]) for name in kinds.boolean},
    }

    # ---- Top missing
    miss_list: List[Tuple[str, float, int]] = []
    for name, (kind, acc) in kinds_map.items():
        miss = getattr(acc, "missing", 0)
        cnt = getattr(acc, "count", 0) + miss
        pct = (miss / cnt * 100.0) if cnt else 0.0
        miss_list.append((name, pct, miss))
    miss_list.sort(key=lambda t: t[1], reverse=True)
    top_missing_list = ""
    for col, pct, count in miss_list[:5]:
        severity_class = "low" if pct <= 5 else ("medium" if pct <= 20 else "high")
        top_missing_list += f"""
        <li class="missing-item">
          <div class="missing-info">
            <code class="missing-col" title="{_html.escape(str(col))}">{_html.escape(str(col))}</code>
            <span class="missing-stats">{count:,} ({pct:.1f}%)</span>
          </div>
          <div class="missing-bar"><div class="missing-fill {severity_class}" style="width:{pct:.1f}%;"></div></div>
        </li>
        """
    if not top_missing_list:
        top_missing_list = """
        <li class="missing-item"><div class="missing-info"><code class="missing-col">None</code><span class="missing-stats">0 (0.0%)</span></div><div class="missing-bar"><div class="missing-fill low" style="width:0%;"></div></div></li>
        """

    # ---- Quick counts
    n_rows = int(getattr(row_kmv, "rows", 0))
    n_cols = len(kinds_map)
    total_cells = n_rows * n_cols
    missing_overall = f"{total_missing_cells:,} ({(total_missing_cells/max(1,total_cells)*100):.1f}%)"
    dup_rows, dup_pct = row_kmv.approx_duplicates()
    duplicates_overall = f"{dup_rows:,} ({dup_pct:.1f}%)"

    # ---- Other quick metrics
    constant_cols = 0
    high_card_cols = 0
    for name, (kind, acc) in kinds_map.items():
        if kind in ("numeric", "categorical"):
            u = acc._uniques.estimate() if hasattr(acc, "_uniques") else getattr(acc, "unique_est", 0)
        elif kind == "datetime":
            u = acc.unique_est
        else:
            present = (acc.true_n > 0) + (acc.false_n > 0)
            u = int(present)
        total = getattr(acc, "count", 0) + getattr(acc, "missing", 0)
        if u <= 1:
            constant_cols += 1
        if kind == "categorical" and n_rows:
            if (u / n_rows) > 0.5:
                high_card_cols += 1

    # ---- Date range
    if kinds.datetime:
        mins, maxs = [], []
        for name in kinds.datetime:
            acc = accs[name]
            if acc._min_ts is not None: mins.append(acc._min_ts)
            if acc._max_ts is not None: maxs.append(acc._max_ts)
        if mins and maxs:
            date_min = datetime.utcfromtimestamp(min(mins) / 1_000_000_000).isoformat() + "Z"
            date_max = datetime.utcfromtimestamp(max(maxs) / 1_000_000_000).isoformat() + "Z"
        else:
            date_min = date_max = "—"
    else:
        date_min = date_max = "—"

    text_cols = len(kinds.categorical)
    avg_text_len_vals = [acc.avg_len for name, (k, acc) in kinds_map.items() if k == "categorical" and acc.avg_len is not None]
    avg_text_len = f"{(sum(avg_text_len_vals)/len(avg_text_len_vals)):.1f}" if avg_text_len_vals else "—"

    # ---- Variable cards (preserve first chunk order if available)
    col_order = [c for c in list(first_columns) if c in kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean] or (kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean)
    all_cards_list: List[str] = []
    for name in col_order:
        acc = accs[name]
        if name in kinds.numeric:
            all_cards_list.append(_render_numeric_card(acc.finalize()))
        elif name in kinds.categorical:
            all_cards_list.append(_render_cat_card(acc.finalize()))
        elif name in kinds.datetime:
            all_cards_list.append(_render_dt_card(acc.finalize()))
        elif name in kinds.boolean:
            all_cards_list.append(_render_bool_card(acc.finalize()))
    variables_section_html = f"""
        <section id="vars">  
          <span id="numeric-vars" class="anchor-alias"></span>
          <h2 class="section-title">Variables</h2>
          <p class="muted small">Analyzing {len(kinds.numeric)+len(kinds.categorical)+len(kinds.datetime)+len(kinds.boolean)} variables ({len(kinds.numeric)} numeric, {len(kinds.categorical)} categorical, {len(kinds.datetime)} datetime, {len(kinds.boolean)} boolean).</p>
          <div class="cards-grid">{''.join(all_cards_list)}</div>
        </section>
    """
    sections_html = (sample_section_html or "") + variables_section_html

    # ---- Load template + assets (same as classic)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(module_dir, "static")
    template_dir = os.path.join(module_dir, "templates")
    template_path = os.path.join(template_dir, "report_template.html")
    template = load_template(template_path)
    css_path = os.path.join(static_dir, "css", "style.css")
    css_tag = load_css(css_path)
    script_path = os.path.join(static_dir, "js", "functionality.js")
    script_content = load_script(script_path)
    logo_light_path = os.path.join(static_dir, "images", "logo_suricata_transparent.png")
    logo_dark_path  = os.path.join(static_dir, "images", "logo_suricata_transparent_dark_mode.png")
    logo_light_img = embed_image(logo_light_path, element_id="logo-light", alt_text="Logo", mime_type="image/png")
    logo_dark_img  = embed_image(logo_dark_path,  element_id="logo-dark",  alt_text="Logo (dark)", mime_type="image/png")
    logo_html = f'<span id="logo">{logo_light_img}{logo_dark_img}</span>'
    favicon_path = os.path.join(static_dir, "images", "favicon.ico")
    favicon_tag = embed_favicon(favicon_path)

    end_time = time.time()
    duration_seconds = end_time - start_time
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pysuricata_version = _resolve_pysuricata_version()
    repo_url = "https://github.com/alvarodiez20/pysuricata"

    html = template.format(
      favicon=favicon_tag,
      css=css_tag,
      script=script_content,
      logo=logo_html,
      report_title= report_title or cfg.title,
      report_date=report_date,
      pysuricata_version=pysuricata_version,
      report_duration=f"{duration_seconds:.2f}",
      repo_url=repo_url,
      n_rows=f"{n_rows:,}",
      n_cols=f"{n_cols:,}",
      memory_usage=_human_bytes(approx_mem_bytes) if approx_mem_bytes else "—",
      missing_overall=missing_overall,
      duplicates_overall=duplicates_overall,
      numeric_cols=len(kinds.numeric),
      categorical_cols=len(kinds.categorical),
      datetime_cols=len(kinds.datetime),
      bool_cols=len(kinds.boolean),
      top_missing_list=top_missing_list,
      n_unique_cols=f"{n_cols:,}",
      constant_cols=f"{constant_cols:,}",
      high_card_cols=f"{high_card_cols:,}",
      date_min=date_min,
      date_max=date_max,
      text_cols=f"{text_cols:,}",
      avg_text_len=avg_text_len,
      dataset_sample_section=sections_html,
    )
    return html


# Public API


# === Formatting helpers used by classic template ===

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024.0

def _memory_usage_bytes(s):
    try:
        return int(s.memory_usage(index=False, deep=True))
    except TypeError:
        try:
            return int(s.memory_usage(deep=True))
        except Exception:
            return 0
    except Exception:
        return 0

def _fmt_duration(seconds: float) -> str:
    try:
        s = int(round(seconds))
    except Exception:
        return f"{seconds:.2f}s"
    parts = []
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


# === Sample section rendering helper ===
def _render_sample_section(df: "pd.DataFrame", sample_rows: int = 10) -> str:  # type: ignore[name-defined]
    """Build the collapsible Sample section using a random sample from the first chunk.
    Mirrors the styling/structure from the classic report (right‑aligned numerics, row numbers).
    """
    try:
        n = min(int(sample_rows), len(df.index))
        sample_df = df.sample(n=n) if n > 0 else df.head(0)
        # Original positional row numbers within this chunk
        row_pos = pd.Index(df.index).get_indexer(sample_df.index)
        sample_df = sample_df.copy()
        sample_df.insert(0, "", row_pos)

        # Right-align numeric columns via a span
        try:
            num_cols = sample_df.columns.intersection(df.select_dtypes(include=[np.number]).columns)
            for c in num_cols:
                sample_df[c] = sample_df[c].map(lambda v: f'<span class="num">{v}</span>' if pd.notna(v) else "")
        except Exception:
            pass

        sample_html_table = sample_df.to_html(classes="sample-table", index=False, escape=False)
    except Exception:
        sample_html_table = "<em>Unable to render sample preview.</em>"
        n = 0

    return f"""
    <section id="sample" class="collapsible-card">
      <span id="dataset-sample" class="anchor-alias"></span>
      <details class="card">
        <summary>
          <span>Sample</span>
          <span class="chev" aria-hidden="true">▸</span>
        </summary>
        <div class="card-content">
          <div class="sample-scroll">{sample_html_table}</div>
          <p class="muted small">Showing {n} randomly sampled rows from the first chunk.</p>
        </div>
      </details>
    </section>
    """


# Light row-wise distinct estimator for duplicates
class _RowKMV:
    def __init__(self, k: int = 8192) -> None:
        self.kmv = KMV(k)
        self.rows = 0
    def update_from_pandas(self, df: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        if pd is None:
            return
        try:
            # fast row-hash: xor column hashes (uint64) to produce a row signature
            h = None
            for c in df.columns:
                hc = pd.util.hash_pandas_object(df[c], index=False).to_numpy(dtype="uint64", copy=False)
                h = hc if h is None else (h ^ hc)
            if h is None:
                return
            self.rows += int(len(h))
            # feed as bytes
            for v in h:
                self.kmv.add(int(v))
        except Exception:
            # conservative fallback: sample a few stringified rows
            n = min(2000, len(df))
            sample = df.head(n).astype(str).agg("|".join, axis=1)
            for s in sample:
                self.kmv.add(s)
            self.rows += n
    def approx_duplicates(self) -> Tuple[int, float]:
        uniq = self.kmv.estimate()
        d = max(0, self.rows - uniq)
        pct = (d / self.rows * 100.0) if self.rows else 0.0
        return d, pct


# =============================
# Lightweight streaming correlations (pairwise sums)
# =============================
class _StreamingCorr:
    def __init__(self, columns: Sequence[str]):
        self.cols = list(columns)
        self.pairs: Dict[Tuple[str, str], Dict[str, float]] = {}

    def update_from_pandas(self, df: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        if pd is None:
            return
        use_cols = [c for c in self.cols if c in df.columns]
        if len(use_cols) < 2:
            return
        arrs: Dict[str, np.ndarray] = {}
        for c in use_cols:
            try:
                a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64", copy=False)
            except Exception:
                a = np.asarray(df[c].to_numpy(), dtype=float)
            arrs[c] = a
        for i in range(len(use_cols)):
            ci = use_cols[i]
            xi = arrs[ci]
            for j in range(i + 1, len(use_cols)):
                cj = use_cols[j]
                yj = arrs[cj]
                m = np.isfinite(xi) & np.isfinite(yj)
                if not m.any():
                    continue
                x = xi[m]; y = yj[m]
                n = float(x.size)
                sx = float(np.sum(x)); sy = float(np.sum(y))
                sx2 = float(np.sum(x * x)); sy2 = float(np.sum(y * y))
                sxy = float(np.sum(x * y))
                key = (ci, cj)
                if key not in self.pairs:
                    self.pairs[key] = {"n": 0.0, "sx": 0.0, "sy": 0.0, "sx2": 0.0, "sy2": 0.0, "sxy": 0.0}
                st = self.pairs[key]
                st["n"] += n
                st["sx"] += sx; st["sy"] += sy
                st["sx2"] += sx2; st["sy2"] += sy2
                st["sxy"] += sxy

    def top_map(self, *, threshold: float = 0.6, max_per_col: int = 2) -> Dict[str, List[Tuple[str, float]]]:
        res: Dict[str, List[Tuple[str, float]]] = {c: [] for c in self.cols}
        for (ci, cj), st in self.pairs.items():
            n = st["n"]
            if n <= 1:
                continue
            num = n * st["sxy"] - st["sx"] * st["sy"]
            denx = n * st["sx2"] - st["sx"] ** 2
            deny = n * st["sy2"] - st["sy"] ** 2
            if denx <= 0 or deny <= 0:
                continue
            r = float(num / math.sqrt(denx * deny))
            if not math.isfinite(r):
                continue
            if abs(r) >= threshold:
                res[ci].append((cj, r))
                res[cj].append((ci, r))
        for c in list(res.keys()):
            res[c].sort(key=lambda t: -abs(t[1]))
            if max_per_col and len(res[c]) > max_per_col:
                res[c] = res[c][:max_per_col]
        return res


def generate_report(
    source: Union[str, "pd.DataFrame"],  # type: ignore[name-defined]
    *,
    config: Optional[ReportConfig] = None,
    output_file: Optional[str] = None,
    report_title: Optional[str] = None,
    return_summary: bool = False,
) -> str:
    """Generate an HTML EDA report from a big source (streaming, out-of-core).

    Parameters
    ----------
    source : str | pandas.DataFrame
        Either a file path (CSV/Parquet) or an in-memory pandas DataFrame.
        (Later: a chunk iterator can be supported by overloading.)
    config : ReportConfig, optional
        Tuning knobs for chunk sizes and approximations.
    output_file : str, optional
        If provided, write the HTML report to this file.
    report_title : str, optional
        Custom title for the report.

    Returns
    -------
    html : str
        A self-contained HTML snippet with the report.
    When return_summary=True, returns (html, summary: dict) with dataset- and per-column stats.
    """
    cfg = config or ReportConfig()
    start_time = time.time()

    # Configure logger
    logger = cfg.logger or logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(cfg.log_level)
    logger.info("Starting report generation: source=%s", source if isinstance(source, str) else f"DataFrame{getattr(source, 'shape', '')}")
    logger.info("chunk_size=%d, uniques_k=%d, numeric_sample_k=%d, topk_k=%d", cfg.chunk_size, cfg.uniques_k, cfg.numeric_sample_k, cfg.topk_k)

    # Build chunk iterator
    with _SectionTimer(logger, "Build chunk iterator"):
        if isinstance(source, str):
            chunks = _iter_chunks_from_path(
                source,
                chunk_size=cfg.chunk_size,
                engine=cfg.engine,
                csv_kwargs=cfg.csv_kwargs,
                parquet_columns=cfg.parquet_columns,
            )
        else:
            if pd is None:
                raise RuntimeError("pandas is required for DataFrame input")
            def _one() -> Iterator["pd.DataFrame"]:  # type: ignore[name-defined]
                yield source  # type: ignore[misc]
            chunks = _one()

    with _SectionTimer(logger, "Read first chunk"):
        try:
            first = next(chunks)
        except StopIteration:
            logger.warning("Empty source; nothing to report")
            return _render_empty_html(cfg.title)
    row_kmv = _RowKMV()
    total_missing_cells = 0
    approx_mem_bytes = 0
    sample_section_html = ""
    if pd is not None and isinstance(first, pd.DataFrame):
        try:
            approx_mem_bytes = int(first.memory_usage(deep=True).sum())
        except Exception:
            approx_mem_bytes = 0
        row_kmv.update_from_pandas(first)
        total_missing_cells += int(first.isna().sum().sum())
        # Render sample section if enabled
        sample_section_html = ""
        if cfg.include_sample and pd is not None and isinstance(first, pd.DataFrame):
            sample_section_html = _render_sample_section(first, cfg.sample_rows)
        first_columns = list(first.columns)
    if pd is not None and isinstance(first, pd.DataFrame):
        with _SectionTimer(logger, "Infer kinds & build accumulators"):
            kinds = _infer_kinds_pandas(first)
            accs = _build_accumulators(kinds, cfg)
            # Set dtype chip from first chunk's dtype
            try:
                dtypes_map = {c: str(first[c].dtype) for c in first.columns}
                for name in kinds.numeric:
                    if name in accs and isinstance(accs[name], NumericAccumulator):
                        accs[name].set_dtype(dtypes_map.get(name, "numeric"))
                for name in kinds.categorical:
                    if name in accs and isinstance(accs[name], CategoricalAccumulator):
                        accs[name].set_dtype(dtypes_map.get(name, "category"))
                for name in kinds.boolean:
                    if name in accs and isinstance(accs[name], BooleanAccumulator):
                        accs[name].set_dtype(dtypes_map.get(name, "boolean"))
                for name in kinds.datetime:
                    if name in accs and isinstance(accs[name], DatetimeAccumulator):
                        accs[name].set_dtype(dtypes_map.get(name, "datetime64[ns]"))
            except Exception:
                pass
        # Optional streaming correlations
        corr_est = None
        if cfg.compute_correlations and len(kinds.numeric) > 1 and len(kinds.numeric) <= cfg.corr_max_cols:
            corr_est = _StreamingCorr(kinds.numeric)
        with _SectionTimer(logger, "Consume first chunk"):
            _consume_chunk_pandas(first, accs, kinds, logger)
            if corr_est is not None:
                try:
                    corr_est.update_from_pandas(first)
                except Exception:
                    logger.exception("Correlation update failed on first chunk")
        logger.info("kinds: %d numeric, %d categorical, %d datetime, %d boolean", len(kinds.numeric), len(kinds.categorical), len(kinds.datetime), len(kinds.boolean))
        n_rows = len(first)
        n_cols = len(first.columns)
        chunk_idx = 1
        # Checkpoint manager (optional)
        ckpt_mgr = None
        if cfg.checkpoint_every_n_chunks and cfg.checkpoint_every_n_chunks > 0:
            # Decide directory
            base_dir = cfg.checkpoint_dir or (os.path.dirname(output_file) if output_file else os.getcwd())
            ckpt_mgr = _CheckpointManager(base_dir, prefix=cfg.checkpoint_prefix, keep=cfg.checkpoint_max_to_keep, write_html=cfg.checkpoint_write_html)
        for ch in chunks:
            chunk_idx += 1
            if (chunk_idx - 1) % max(1, cfg.log_every_n_chunks) == 0:
                logger.info("processing chunk %d: %d rows", chunk_idx, len(ch))
            _consume_chunk_pandas(ch, accs, kinds, logger)
            try:
                approx_mem_bytes += int(ch.memory_usage(deep=True).sum())
            except Exception:
                pass
            if corr_est is not None:
                try:
                    corr_est.update_from_pandas(ch)
                except Exception:
                    logger.exception("Correlation update failed on chunk %d", chunk_idx)
            n_rows += len(ch)
            row_kmv.update_from_pandas(ch)
            total_missing_cells += int(ch.isna().sum().sum())
            # --- Checkpointing ---
            if ckpt_mgr and (chunk_idx % cfg.checkpoint_every_n_chunks == 0):
                try:
                    snapshot = _make_state_snapshot(
                        kinds=kinds,
                        accs=accs,
                        row_kmv=row_kmv,
                        total_missing_cells=total_missing_cells,
                        approx_mem_bytes=approx_mem_bytes,
                        chunk_idx=chunk_idx,
                        first_columns=first_columns,
                        sample_section_html=sample_section_html,
                        cfg=cfg,
                    )
                    html_ckpt = None
                    if cfg.checkpoint_write_html:
                        html_ckpt = _render_html_snapshot(
                            kinds=kinds,
                            accs=accs,
                            first_columns=first_columns,
                            row_kmv=row_kmv,
                            total_missing_cells=total_missing_cells,
                            approx_mem_bytes=approx_mem_bytes,
                            start_time=start_time,
                            cfg=cfg,
                            report_title=report_title,
                            sample_section_html=sample_section_html,
                        )
                    pkl_path, html_path = ckpt_mgr.save(chunk_idx, snapshot, html=html_ckpt)
                    logger.info("Checkpoint saved at %s%s", pkl_path, f" and {html_path}" if html_path else "")
                except Exception:
                    logger.exception("Failed to write checkpoint at chunk %d", chunk_idx)
    else:
        raise TypeError("Currently only pandas DataFrame chunks are supported. Provide a CSV/Parquet path or pandas DataFrame.")

    # Build per-column accumulator map grouped by kind for metrics
    kinds_map = {
        **{name: ("numeric", accs[name]) for name in kinds.numeric},
        **{name: ("categorical", accs[name]) for name in kinds.categorical},
        **{name: ("datetime", accs[name]) for name in kinds.datetime},
        **{name: ("boolean", accs[name]) for name in kinds.boolean},
    }

    # Finalize correlation chips and attach to numeric accumulators
    if 'corr_est' in locals() and corr_est is not None:
        top_map = corr_est.top_map(threshold=cfg.corr_threshold, max_per_col=cfg.corr_max_per_col)
        for name in kinds.numeric:
            if name in accs and isinstance(accs[name], NumericAccumulator):
                accs[name].set_corr_top(top_map.get(name, []))

    # Top-missing list and summary metrics
    with _SectionTimer(logger, "Compute top-missing, duplicates & quick metrics"):
        pass  # marker for timing; actual work happens in the following lines
    miss_list: List[Tuple[str, float, int]] = []  # (name, pct, count)
    for name, (kind, acc) in kinds_map.items():
        miss = getattr(acc, "missing", 0)
        cnt = getattr(acc, "count", 0) + miss
        pct = (miss / cnt * 100.0) if cnt else 0.0
        miss_list.append((name, pct, miss))
    miss_list.sort(key=lambda t: t[1], reverse=True)
    top_missing_list = ""
    for col, pct, count in miss_list[:5]:
        severity_class = "low" if pct <= 5 else ("medium" if pct <= 20 else "high")
        top_missing_list += f"""
        <li class=\"missing-item\"> 
          <div class=\"missing-info\"> 
            <code class=\"missing-col\" title=\"{_html.escape(str(col))}\">{_html.escape(str(col))}</code>
            <span class=\"missing-stats\">{count:,} ({pct:.1f}%)</span>
          </div>
          <div class=\"missing-bar\"><div class=\"missing-fill {severity_class}\" style=\"width:{pct:.1f}%;\"></div></div>
        </li>
        """
    if not top_missing_list:
        top_missing_list = """
        <li class=\"missing-item\"><div class=\"missing-info\"><code class=\"missing-col\">None</code><span class=\"missing-stats\">0 (0.0%)</span></div><div class=\"missing-bar\"><div class=\"missing-fill low\" style=\"width:0%;\"></div></div></li>
        """
    logger.info("top-missing columns: %s", ", ".join([c for c,_,_ in miss_list[:5]]) or "(none)")

    # Quick counts
    n_rows = int(getattr(row_kmv, "rows", 0))
    n_cols = len(kinds_map)
    total_cells = n_rows * n_cols
    missing_overall = f"{total_missing_cells:,} ({(total_missing_cells/max(1,total_cells)*100):.1f}%)"
    dup_rows, dup_pct = row_kmv.approx_duplicates()
    duplicates_overall = f"{dup_rows:,} ({dup_pct:.1f}%)"

    # n_unique_cols (classic used nunique().count() which is # of columns). Keep same.
    n_unique_cols = n_cols

    # constant/high-card metrics
    constant_cols = 0
    high_card_cols = 0
    likely_id_cols: List[str] = []
    for name, (kind, acc) in kinds_map.items():
        if kind in ("numeric", "categorical"):
            u = acc._uniques.estimate() if hasattr(acc, "_uniques") else getattr(acc, "unique_est", 0)
        elif kind == "datetime":
            u = acc.unique_est
        else:  # boolean
            # boolean unique among {True, False} (ignore missing for ID heuristic)
            present = (acc.true_n > 0) + (acc.false_n > 0)
            u = int(present)
        total = getattr(acc, "count", 0) + getattr(acc, "missing", 0)
        if u <= 1:
            constant_cols += 1
        if kind == "categorical" and n_rows:
            if (u / n_rows) > 0.5:
                high_card_cols += 1
        if total == n_rows and getattr(acc, "missing", 0) == 0 and n_rows:
            if u >= int(0.98 * n_rows):
                likely_id_cols.append(name)
    if len(likely_id_cols) > 3:
        likely_id_cols = likely_id_cols[:3] + ["..."]
    likely_id_cols_str = ", ".join(likely_id_cols) if likely_id_cols else "—"

    # Date range across datetime columns
    if kinds.datetime:
        mins, maxs = [], []
        for name in kinds.datetime:
            acc = accs[name]
            if acc._min_ts is not None: mins.append(acc._min_ts)
            if acc._max_ts is not None: maxs.append(acc._max_ts)
        if mins and maxs:
            date_min = datetime.utcfromtimestamp(min(mins) / 1_000_000_000).isoformat() + "Z"
            date_max = datetime.utcfromtimestamp(max(maxs) / 1_000_000_000).isoformat() + "Z"
        else:
            date_min = date_max = "—"
    else:
        date_min = date_max = "—"

    # Text columns / avg length (approx from categorical accs)
    text_cols = len(kinds.categorical)
    avg_text_len_vals = [acc.avg_len for name, (k, acc) in kinds_map.items() if k == "categorical" and acc.avg_len is not None]
    avg_text_len = f"{(sum(avg_text_len_vals)/len(avg_text_len_vals)):.1f}" if avg_text_len_vals else "—"

    # Build Variables section using existing CSS classes, preserving original column order
    col_order = []
    if pd is not None and isinstance(first, pd.DataFrame):
        col_order = [c for c in list(first.columns) if c in kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean]
    else:
        col_order = kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean

    all_cards_list: List[str] = []
    for name in col_order:
        acc = accs[name]
        if name in kinds.numeric:
            all_cards_list.append(_render_numeric_card(acc.finalize()))
        elif name in kinds.categorical:
            all_cards_list.append(_render_cat_card(acc.finalize()))
        elif name in kinds.datetime:
            all_cards_list.append(_render_dt_card(acc.finalize()))
        elif name in kinds.boolean:
            all_cards_list.append(_render_bool_card(acc.finalize()))

    with _SectionTimer(logger, "Render Variables section"):
        variables_section_html = f"""
            <section id=\"vars\">  
              <span id=\"numeric-vars\" class=\"anchor-alias\"></span>
              <h2 class=\"section-title\">Variables</h2>
              <p class=\"muted small\">Analyzing {len(kinds.numeric)+len(kinds.categorical)+len(kinds.datetime)+len(kinds.boolean)} variables ({len(kinds.numeric)} numeric, {len(kinds.categorical)} categorical, {len(kinds.datetime)} datetime, {len(kinds.boolean)} boolean).</p>
              <div class=\"cards-grid\">{''.join(all_cards_list)}</div>
            </section>
        """
    logger.info("rendered %d variable cards", len(all_cards_list))
    sections_html = (sample_section_html or "") + variables_section_html

    with _SectionTimer(logger, "Render final HTML"):
        html = _render_html_snapshot(
            kinds=kinds,
            accs=accs,
            first_columns=first_columns,
            row_kmv=row_kmv,
            total_missing_cells=total_missing_cells,
            approx_mem_bytes=approx_mem_bytes,
            start_time=start_time,
            cfg=cfg,
            report_title=report_title,
            sample_section_html=sample_section_html,
        )

    # Optional minimal JSON-like summary for programmatic use
    summary_obj = None
    try:
        # dataset-level
        dataset_summary = {
            "rows_est": int(getattr(row_kmv, "rows", 0)),
            "cols": int(len(kinds_map)),
            "missing_cells": int(total_missing_cells),
            "missing_cells_pct": (total_missing_cells / max(1, n_rows * n_cols) * 100.0) if (n_rows and n_cols) else 0.0,
            "duplicate_rows_est": int(row_kmv.approx_duplicates()[0]),
            "duplicate_rows_pct_est": float(row_kmv.approx_duplicates()[1]),
            "top_missing": [
                {"column": str(col), "pct": float(pct), "count": int(cnt)}
                for col, pct, cnt in (miss_list[:5] if 'miss_list' in locals() else [])
            ],
        }
        # per-column summaries
        columns_summary: Dict[str, Dict[str, Any]] = {}
        for name in col_order:
            kind, acc = kinds_map[name]
            if kind == "numeric":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "numeric",
                    "count": s.count,
                    "missing": s.missing,
                    "unique_est": s.unique_est,
                    "mean": s.mean,
                    "std": s.std,
                    "min": s.min,
                    "q1": s.q1,
                    "median": s.median,
                    "q3": s.q3,
                    "max": s.max,
                    "zeros": s.zeros,
                    "negatives": s.negatives,
                    "outliers_iqr_est": s.outliers_iqr,
                    "approx": bool(s.approx),
                }
            elif kind == "categorical":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "categorical",
                    "count": s.count,
                    "missing": s.missing,
                    "unique_est": s.unique_est,
                    "top_items": s.top_items,
                    "approx": bool(s.approx),
                }
            elif kind == "datetime":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "datetime",
                    "count": s.count,
                    "missing": s.missing,
                    "min_ts": s.min_ts,
                    "max_ts": s.max_ts,
                }
            else:  # boolean
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "boolean",
                    "count": s.count,
                    "missing": s.missing,
                    "true": s.true_n,
                    "false": s.false_n,
                }
        summary_obj = {"dataset": dataset_summary, "columns": columns_summary}
    except Exception:
        summary_obj = None

    if output_file:
        with _SectionTimer(logger, f"Write HTML to {output_file}"):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html)
        logger.info("report written: %s (%s)", output_file, _human_bytes(len(html.encode('utf-8'))))
    logger.info("Report generation complete in %.2fs", time.time() - start_time)
    if return_summary:
        return html, (summary_obj or {})  # type: ignore[return-value]
    return html


def _render_empty_html(title: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title></head>
    <body><div class=\"container\"><h1>{title}</h1><p>Empty source.</p></div></body></html>
    """


# =============================
# Chunk consumption for pandas
# =============================
def _infer_kind_for_series_pandas(s: "pd.Series") -> str:  # type: ignore[name-defined]
    dt = str(getattr(s, "dtype", "object"))
    if re.search("int|float|^UInt|^Int|^Float", dt, re.I):
        return "numeric"
    if re.search("bool", dt, re.I):
        return "boolean"
    if re.search("datetime", dt, re.I):
        return "datetime"
    sample = s.head(10_000)
    if pd is not None:
        ds = pd.to_datetime(sample, errors="coerce", utc=True)
        if ds.notna().sum() >= max(10, int(0.7 * len(sample))):
            return "datetime"
        ns = pd.to_numeric(sample, errors="coerce")
        if ns.notna().sum() >= max(10, int(0.7 * len(sample))):
            return "numeric"
    uniq = set(map(str, sample.dropna().unique().tolist()))
    if 1 <= len(uniq) <= 2 and uniq.issubset({"True", "False", "0", "1", "true", "false"}):
        return "boolean"
    return "categorical"


def _consume_chunk_pandas(df: "pd.DataFrame", accs: Dict[str, Any], kinds: ColumnKinds, logger: Optional[logging.Logger] = None) -> None:  # type: ignore[name-defined]
    # 1) Create accumulators for columns not seen in the first chunk
    for name in df.columns:
        if name in accs:
            continue
        kind = _infer_kind_for_series_pandas(df[name])
        if kind == "numeric":
            accs[name] = NumericAccumulator(name)
            kinds.numeric.append(name)
        elif kind == "boolean":
            accs[name] = BooleanAccumulator(name)
            kinds.boolean.append(name)
        elif kind == "datetime":
            accs[name] = DatetimeAccumulator(name)
            kinds.datetime.append(name)
        else:
            accs[name] = CategoricalAccumulator(name)
            kinds.categorical.append(name)
        if logger:
            logger.info("➕ discovered new column '%s' inferred as %s", name, kind)

    # 2) Feed accumulators for columns present in this chunk
    for name, acc in accs.items():
        if name not in df.columns:
            # Column absent in this chunk (ragged files / column pruning)
            if logger:
                logger.debug("column '%s' not present in this chunk; skipping", name)
            continue
        s = df[name]
        if isinstance(acc, NumericAccumulator):
            arr = _to_numeric_array_pandas(s)
            acc.update(arr)
            # Track extremes with indices for this chunk
            try:
                finite = np.isfinite(arr)
                if finite.any():
                    vals = arr[finite]
                    idx = s.index.to_numpy()[finite]
                    # smallest 5
                    if vals.size > 0:
                        k = min(5, vals.size)
                        part_min = np.argpartition(vals, k - 1)[:k]
                        pairs_min = [(idx[i], float(vals[i])) for i in part_min]
                        # largest 5
                        part_max = np.argpartition(-vals, k - 1)[:k]
                        pairs_max = [(idx[i], float(vals[i])) for i in part_max]
                        acc.update_extremes(pairs_min, pairs_max)
            except Exception:
                pass
        elif isinstance(acc, BooleanAccumulator):
            arr = _to_bool_array_pandas(s)
            acc.update(arr)
            try:
                acc.add_mem(int(s.memory_usage(deep=True)))
            except Exception:
                pass
        elif isinstance(acc, DatetimeAccumulator):
            arr = _to_datetime_ns_array_pandas(s)
            acc.update(arr)
            try:
                acc.add_mem(int(s.memory_usage(deep=True)))
            except Exception:
                pass
        elif isinstance(acc, CategoricalAccumulator):
            acc.update(_to_categorical_iter_pandas(s))
