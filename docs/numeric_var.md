---
title: Numerical Variable Analysis
description: How pysuricata profiles and summarizes numerical (continuous and discrete) columns at scale. Includes math, algorithms, configuration, and implementation notes.
icon: material/calculator-variant
---

> **TL;DR**: This page defines *exact* and *approximate* statistics for numerical columns, the incremental formulas we use (Welford/Pébay), how we build histograms and quantiles (exact, KLL, t‑digest), outlier rules (z‑score, IQR, MAD), and the guarantees, complexity, and configuration knobs.

!!! tip "Audience"
    Designed for users who want to **understand and trust** the numbers in the HTML report as well as contributors who need to **modify** the accumulator implementations.

---

## Scope

`pysuricata` treats a **numerical variable** as any column with a machine type among `{int8, int16, int32, int64, float32, float64, decimal}` (nullable). Values may include `NaN`, `±Inf`, and missing markers.

We report **global** statistics and (optionally) **windowed** ones (e.g., per‑file chunk, per time window in a stream). All stats are computed **incrementally** via *stateful accumulators* so that we can process datasets larger than RAM.

---

## Summary cards (what you’ll see in the report)

- Count (non‑null), missing %, distinct (exact/approx), min, max
- Central tendency: mean, median
- Dispersion: variance, standard deviation, IQR, MAD
- Shape: skewness, excess kurtosis
- Quantiles (configurable set) and histogram preview
- Outlier flags (IQR fences and modified z‑score)

> All numbers are derived from the algorithms below and are reproducible when re‑run with the same config and data.

---

## Mathematical definitions

### Notation
Let \(x_1, x_2, \ldots, x_n\) be the non‑missing observations for a column.

- **Mean**: \(\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i\)
- **Unbiased sample variance**: \(s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i-\bar{x})^2\)
- **Std. deviation**: \(s = \sqrt{s^2}\)
- **Median**: any \(m\) such that half the mass is \(\le m\) and half \(\ge m\)
- **Quantile** \(Q(p)\): infimum \(q\) with CDF \(\ge p\)
- **IQR**: \(Q(0.75) - Q(0.25)\)
- **MAD**: \(\operatorname{median}(|x_i - \operatorname{median}(x)|)\)
- **Skewness (g₁)**:
  \[
  g_1 = \frac{n}{(n-1)(n-2)} \cdot \frac{\frac{1}{n}\sum (x_i-\bar{x})^3}{s^3}
  \]
- **Excess kurtosis (g₂)**:
  \[
  g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \cdot \frac{\frac{1}{n}\sum (x_i-\bar{x})^4}{s^4} - \frac{3(n-1)^2}{(n-2)(n-3)}
  \]

### Outlier rules
- **IQR fences**: an observation is an outlier if \(x < Q(0.25) - 1.5\,\mathrm{IQR}\) or \(x > Q(0.75) + 1.5\,\mathrm{IQR}\). 3.0 is a common “extreme” fence.
- **Modified z‑score (robust)**: \(M_i = 0.6745\,\frac{x_i - \operatorname{median}(x)}{\operatorname{MAD}}\). Flag if \(|M_i| > 3.5\).
- **Z‑score (classical)**: \(Z_i = \frac{x_i - \bar{x}}{s}\). Sensitive to non‑normality and outliers.

---

## Incremental & mergeable estimation (how we compute at scale)

We maintain *running* central moments using **Welford’s method** with **Pébay’s parallel merge formulas** so we can:

- update statistics in **O(1)** per observation
- **merge** partial results from multiple chunks/threads/nodes exactly (up to FP rounding)

### Online update for (n, mean, M2, M3, M4)
For a new value \(x\) and previous state \((n, \mu, M_2, M_3, M_4)\):

\[
\begin{aligned}
 n' &= n + 1 \\
 \delta &= x - \mu \\
 \delta_n &= \frac{\delta}{n'} \\
 \delta_n^2 &= \delta_n^2 \\
 \mu' &= \mu + \delta_n \\
 M_4' &= M_4 + \delta(\delta^3 \frac{n(n-1)}{n'^3} + 6\,\delta_n M_2' - 4\,\delta_n^2 M_3) \\
 M_3' &= M_3 + \delta(\delta^2 \frac{n(n-1)}{n'^2} - 3\,\delta_n M_2) \\
 M_2' &= M_2 + \delta(\delta - \delta_n)
\end{aligned}
\]

Then:
\(s^2 = M_2'/(n'-1)\), and shape statistics are computed from \(M_2', M_3', M_4'\) as in the definitions above.

> **Numerical stability**: this form avoids catastrophic cancellation found in two‑pass formulas.

### Parallel/Chunked merge (Pébay)
Given two partial states \(A=(n_a, \mu_a, M_{2a}, M_{3a}, M_{4a})\) and \(B=(n_b, \mu_b, M_{2b}, M_{3b}, M_{4b})\), define \(\delta = \mu_b - \mu_a\), \(n = n_a + n_b\). Then (abridged):

\[
\begin{aligned}
\mu &= \mu_a + \delta \cdot \frac{n_b}{n} \\
M_2 &= M_{2a} + M_{2b} + \delta^2 \cdot \frac{n_a n_b}{n} \\
M_3 &= M_{3a} + M_{3b} + \delta^3 \cdot \frac{n_a n_b (n_a - n_b)}{n^2} + 3\delta \cdot \frac{n_a M_{2b} - n_b M_{2a}}{n} \\
M_4 &= M_{4a} + M_{4b} + \delta^4 \cdot \frac{n_a n_b (n_a^2 - n_a n_b + n_b^2)}{n^3} \\
&\quad + 6\delta^2 \cdot \frac{n_a n_b}{n^2}(n_a M_{2b} + n_b M_{2a}) + 4\delta \cdot \frac{n_a M_{3b} - n_b M_{3a}}{n}
\end{aligned}
\]

These enable exact merges of chunked/parallel passes.

??? info "References"
    - Welford’s online mean/variance — Wikipedia: <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>
    - Pébay, *Formulas for Robust, One‑Pass Parallel Computation of Covariances and Statistical Moments*, Sandia Report SAND2008‑6212: <https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf>

---

## Quantiles & histograms

=== "Exact"
    - **Quantiles**: maintain a reservoir sample or, for in‑memory small columns, sort all values and index by \(\lceil p(n+1) \rceil\) with linear interpolation.
    - **Histogram**: fixed binning with computed bin width. Default bin count uses **Freedman–Diaconis**:
      \[ h = 2\,\frac{\mathrm{IQR}}{n^{1/3}} \quad\Rightarrow\quad k = \left\lceil \frac{\max - \min}{h} \right\rceil. \]
    - Pros: exact; Cons: may be expensive for very large \(n\).

=== "Approximate (default for large data)"
    - **KLL sketch** for quantiles — sublinear memory with provable error bounds.
    - **t‑digest** (optional) — excellent tail accuracy for percentiles like P99.
    - **Streaming histogram** — fold approximate quantiles into dynamic bins.

    **References:**
    - KLL: *Optimal Quantile Approximation in Streams* — <https://arxiv.org/abs/1603.05346>
    - t‑digest (Dunning & Ertl) — <https://arxiv.org/abs/1902.04023>
    - Freedman–Diaconis rule — <https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule>

---

## Distinct counts (optional)

For numerical columns with many repeated values (e.g., codes, rounded measures) we can report **distinct** and **top‑k**. For large domains we offer:

- **HyperLogLog** for approximate distinct with small, fixed memory.
- **Space‑Saving** for top‑k heavy hitters.

**References:** HLL (Flajolet et al.) — <https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>, Space‑Saving — <https://www.cs.ucdavis.edu/~minle/paper/summary.pdf>

---

## Missing, NaN, and Inf handling

- **Missing**: values recognized as nulls are excluded from *moment* calculations; we still report `missing_count` and `missing_pct`.
- **NaN**: treated as missing.
- **±Inf**: excluded from moments but counted under a dedicated `infinite_count` and surfaced in warnings.
- **Type coercion**: strings that parse to numbers are counted only if parsing is enabled.

!!! warning "Edge cases"
    All‑missing columns, or columns with \(n<2\), will have undefined variance/shape; we report `null` and document why.

---

## Computation guarantees & complexity

- **Time**: \(O(n)\) total, \(O(1)\) amortized per value for moments; sketches are \(O(\log(1/\varepsilon))\) per update.
- **Space**: \(O(1)\) for moments; sketches use tens to thousands of bytes depending on accuracy.
- **Determinism**: merges are associative up to IEEE‑754 rounding; we use stable orders where feasible.

---

## Configuration

=== "Python API"
    ```python
    from pysuricata import profile

    report = profile(
        df,
        numeric={
            "quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
            "hist": {"strategy": "fd", "max_bins": 256},
            "sketch": {"quantiles": "kll", "tdigest": {"delta": 1000}},
            "distinct": {"method": "hll", "precision": 14},
            "outliers": {"method": "iqr|mad|zscore", "fence": 1.5},
            "nan_policy": "exclude",
        }
    )
    ```

=== "CLI"
    ```bash
    pysuricata report data.parquet \
      --quantiles 0.01 0.05 0.5 0.95 0.99 \
      --hist fd --hist-max-bins 256 \
      --qsketch kll --tdigest-delta 1000 \
      --distinct hll --hll-precision 14 \
      --outliers mad --fence 3.5
    ```

=== "Env vars"
    ```bash
    export PS_NUMERIC_Q=0.01,0.05,0.5,0.95,0.99
    export PS_HIST_STRATEGY=fd
    export PS_HIST_MAX_BINS=256
    export PS_QSKETCH=kll
    ```

> **Note:** Exact option names may evolve. See the Python docstrings for authoritative signatures.

---

## API surface (accumulators)

```python
class NumericAccumulator:
    """Stateful, mergeable accumulator for one numerical column."""

    def update(self, values):
        """Update running moments and sketches from an array/Series."""

    def merge(self, other):
        """Parallel merge (Pébay)."""

    def finalize(self):
        """Return a dataclass with count, missing, mean, s, skew, kurtosis, min, max, quantiles, histogram, outliers."""
```

!!! example "Minimal usage"
    ```python
    import pandas as pd
    from pysuricata.numeric import NumericAccumulator

    acc = NumericAccumulator()
    for chunk in pd.read_csv("data.csv", chunksize=200_000):
        acc.update(chunk["amount"].to_numpy())
    summary = acc.finalize()
    ```

---

## Validation

- Cross‑check against `numpy`, `scipy.stats`, and `pandas` on small datasets.
- Property‑based tests: invariants under concatenation (merge == single pass), scaling/translation laws.
- Randomized FP tests to catch catastrophic cancellation.

---

## See also

- **Skewness and kurtosis** — <https://en.wikipedia.org/wiki/Skewness>, <https://en.wikipedia.org/wiki/Kurtosis>
- **MAD & robust stats** — <https://en.wikipedia.org/wiki/Median_absolute_deviation>
- **IQR & Tukey fences** — <https://en.wikipedia.org/wiki/Interquartile_range>

---

## Changelog

- *v0.1*: initial draft of numerical analysis module and documentation.
