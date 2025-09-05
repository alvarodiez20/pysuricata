from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple
import math
from collections import Counter

import numpy as np

from .sketches import ReservoirSampler, KMV


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
            self._bytes_seen += 8
            if x == 0.0:
                self.zeros += 1
            if x < 0.0:
                self.negatives += 1
            if abs(x - round(x)) > 1e-12:
                self._int_like_all = False
            # monotonicity tracking
            if self._last_val is None:
                self._last_val = x
            else:
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

        # Heap detection heuristic (sample-based)
        heap_pct = float("nan")
        try:
            if svals and len(svals) >= 25:
                # Look for mass at .0 or .5 boundaries
                if self._int_like_all:
                    # proportion at exact integers
                    ints = sum(1 for v in svals if abs(v - round(v)) <= 1e-12)
                    heap_pct = ints / len(svals) * 100.0
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

