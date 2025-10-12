---
title: Numeric Variable Analysis
description: Complete mathematical formulas, algorithms, and implementation details for numerical statistics in pysuricata
---

# Numeric Variable Analysis

This page provides comprehensive technical documentation for how pysuricata profiles and summarizes numerical (continuous and discrete) columns at scale using proven streaming algorithms with mathematical guarantees.

!!! tip "Audience"
    Designed for users who want to **understand and trust** the numbers in the HTML report, as well as contributors who need to **modify** or **extend** the accumulator implementations.

## Overview

PySuricata treats a **numerical variable** as any column with machine type among `{int8, int16, int32, int64, float32, float64, decimal}` (nullable). Values may include `NaN`, `±Inf`, and missing markers, all handled appropriately.

All statistics are computed **incrementally** via *stateful accumulators* using single-pass streaming algorithms, enabling processing of datasets larger than available RAM.

## Summary Statistics Provided

For each numeric column, the report includes:

- **Count**: non-null values, missing percentage
- **Central tendency**: mean, median
- **Dispersion**: variance, standard deviation, IQR, MAD, coefficient of variation
- **Shape**: skewness, excess kurtosis, bimodality hints
- **Range**: min, max, range
- **Quantiles**: configurable set (default: 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
- **Outliers**: IQR fences, z-score, MAD-based detection
- **Distribution**: histogram with adaptive binning
- **Special values**: zeros, negatives, infinities
- **Extremes**: min/max values with row indices
- **Confidence intervals**: 95% CI for mean
- **Correlations**: top correlations with other numeric columns (optional)

## Mathematical Definitions

### Notation

Let \(x_1, x_2, \ldots, x_n\) be the non-missing, finite observations for a column.

### Central Tendency

**Mean (arithmetic average):**

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

**Median:**

The value \(m\) such that at least half the observations are \(\le m\) and at least half are \(\ge m\). For even \(n\), typically the average of the two middle values.

\[
\text{median} = 
\begin{cases}
x_{(n+1)/2} & \text{if } n \text{ odd} \\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ even}
\end{cases}
\]

where \(x_{(k)}\) denotes the \(k\)-th order statistic (sorted values).

### Dispersion

**Sample variance (unbiased):**

\[
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i-\bar{x})^2
\]

**Standard deviation:**

\[
s = \sqrt{s^2}
\]

**Coefficient of variation (CV):**

\[
\text{CV} = \frac{s}{|\bar{x}|} \times 100\%
\]

Measures relative variability; useful for comparing dispersion across variables with different scales.

**Interquartile range (IQR):**

\[
\text{IQR} = Q_{0.75} - Q_{0.25}
\]

Robust measure of spread, resistant to outliers.

**Median absolute deviation (MAD):**

\[
\text{MAD} = \text{median}(|x_i - \text{median}(x)|)
\]

Highly robust measure of variability; often preferred over standard deviation for non-normal data.

### Shape Statistics

**Skewness (Fisher-Pearson, \(g_1\)):**

Measures asymmetry of the distribution.

\[
g_1 = \frac{n}{(n-1)(n-2)} \cdot \frac{m_3}{s^3}
\]

where \(m_3 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3\) is the third central moment.

**Interpretation:**
- \(g_1 = 0\): symmetric
- \(g_1 > 0\): right-skewed (long right tail)
- \(g_1 < 0\): left-skewed (long left tail)

**Excess kurtosis (\(g_2\)):**

Measures tail heaviness relative to normal distribution.

\[
g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \cdot \frac{m_4}{s^4} - \frac{3(n-1)^2}{(n-2)(n-3)}
\]

where \(m_4 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^4\) is the fourth central moment.

**Interpretation:**
- \(g_2 = 0\): normal (mesokurtic)
- \(g_2 > 0\): heavy tails (leptokurtic)
- \(g_2 < 0\): light tails (platykurtic)

### Quantiles

**Quantile function \(Q(p)\):**

For probability \(p \in [0,1]\), the \(p\)-quantile is:

\[
Q(p) = \inf\{x : F(x) \ge p\}
\]

where \(F(x) = \mathbb{P}(X \le x)\) is the cumulative distribution function.

**Common quantiles:**
- \(Q(0.25)\): first quartile (Q1)
- \(Q(0.50)\): median (Q2)
- \(Q(0.75)\): third quartile (Q3)

### Confidence Intervals

**95% confidence interval for mean:**

Assuming approximate normality (or large \(n\) by CLT):

\[
\text{CI}_{0.95}(\bar{x}) = \bar{x} \pm t_{n-1,0.975} \cdot \frac{s}{\sqrt{n}}
\]

where \(t_{n-1,0.975}\) is the 97.5th percentile of Student's t-distribution with \(n-1\) degrees of freedom.

For large \(n\), \(t_{n-1,0.975} \approx 1.96\).

### Outlier Detection

**IQR fences (Tukey's method):**

Lower fence:

\[
L = Q_{0.25} - 1.5 \cdot \text{IQR}
\]

Upper fence:

\[
U = Q_{0.75} + 1.5 \cdot \text{IQR}
\]

Values outside \([L, U]\) are flagged as outliers. For "extreme" outliers, use multiplier 3.0 instead of 1.5.

**Modified z-score (robust):**

\[
M_i = 0.6745 \cdot \frac{x_i - \text{median}(x)}{\text{MAD}}
\]

Flag if \(|M_i| > 3.5\). The constant 0.6745 makes MAD consistent with standard deviation under normality.

**Classical z-score:**

\[
Z_i = \frac{x_i - \bar{x}}{s}
\]

Flag if \(|Z_i| > 3\). Sensitive to outliers themselves (not robust).

## Streaming Algorithms

### Welford's Online Algorithm

For computing mean and variance in a **single pass** with **O(1) memory** and **numerical stability**.

**Initialization:**

\[
n = 0, \quad \mu = 0, \quad M_2 = 0
\]

**Update step:** for each new value \(x\):

\[
\begin{aligned}
n &\leftarrow n + 1 \\
\delta &= x - \mu \\
\mu &\leftarrow \mu + \frac{\delta}{n} \\
\delta_2 &= x - \mu \\
M_2 &\leftarrow M_2 + \delta \cdot \delta_2
\end{aligned}
\]

**Finalize:**

\[
\text{mean} = \mu, \quad \text{variance} = \frac{M_2}{n-1}
\]

**Properties:**
- **Numerical stability**: avoids catastrophic cancellation in \(\sum x_i^2 - n\bar{x}^2\)
- **Exact**: produces same result as two-pass method (up to FP rounding)
- **Online**: updates in O(1) time per value

**Reference:** Welford, B.P. (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", *Technometrics*, 4(3): 419–420.

### Higher Moments (Skewness, Kurtosis)

Extending Welford to track \(M_3\) and \(M_4\):

**Update step:**

\[
\begin{aligned}
n &\leftarrow n + 1 \\
\delta &= x - \mu \\
\delta_n &= \frac{\delta}{n} \\
\delta_n^2 &= \delta_n^2 \\
\mu &\leftarrow \mu + \delta_n \\
M_4 &\leftarrow M_4 + \delta \left(\delta^3 \frac{n(n-1)}{n^3} + 6\delta_n M_2 - 4\delta_n M_3\right) \\
M_3 &\leftarrow M_3 + \delta \left(\delta^2 \frac{n(n-1)}{n^2} - 3\delta_n M_2\right) \\
M_2 &\leftarrow M_2 + \delta(\delta - \delta_n)
\end{aligned}
\]

**Finalize:**

\[
\begin{aligned}
s^2 &= \frac{M_2}{n-1} \\
g_1 &= \frac{n}{(n-1)(n-2)} \cdot \frac{M_3/n}{(s^2)^{3/2}} \\
g_2 &= \frac{n(n+1)}{(n-1)(n-2)(n-3)} \cdot \frac{M_4/n}{(s^2)^2} - \frac{3(n-1)^2}{(n-2)(n-3)}
\end{aligned}
\]

### Pébay's Parallel Merge Formulas

To combine results from **multiple chunks** or **parallel threads**, Pébay's formulas enable **exact merging** of moments.

Given two partial states:
- State A: \((n_a, \mu_a, M_{2a}, M_{3a}, M_{4a})\)
- State B: \((n_b, \mu_b, M_{2b}, M_{3b}, M_{4b})\)

Define:

\[
\delta = \mu_b - \mu_a, \quad n = n_a + n_b
\]

**Merged state:**

\[
\begin{aligned}
\mu &= \mu_a + \delta \cdot \frac{n_b}{n} \\
M_2 &= M_{2a} + M_{2b} + \delta^2 \cdot \frac{n_a n_b}{n} \\
M_3 &= M_{3a} + M_{3b} + \delta^3 \cdot \frac{n_a n_b (n_a - n_b)}{n^2} + 3\delta \cdot \frac{n_a M_{2b} - n_b M_{2a}}{n} \\
M_4 &= M_{4a} + M_{4b} + \delta^4 \cdot \frac{n_a n_b (n_a^2 - n_a n_b + n_b^2)}{n^3} \\
&\quad + 6\delta^2 \cdot \frac{n_a n_b}{n^2}(n_a M_{2b} + n_b M_{2a}) + 4\delta \cdot \frac{n_a M_{3b} - n_b M_{3a}}{n}
\end{aligned}
\]

**Properties:**
- **Associative**: order of merging doesn't matter (up to FP rounding)
- **Exact**: same result as single-pass over concatenated data
- **Parallelizable**: enables multi-core and distributed computation

**Reference:** Pébay, P. (2008), "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments", Sandia Report SAND2008-6212.

## Quantile Estimation

### Exact Quantiles (Reservoir Sampling)

For **exact quantiles**, maintain a **reservoir sample** of size \(k\) (default 20,000):

1. First \(k\) values: keep all
2. For value \(i > k\): include with probability \(k/i\), replacing random existing sample

**Uniform guarantee:** Every subset of size \(k\) has equal probability.

**Quantile computation:** Sort sample and compute using linear interpolation:

\[
Q(p) \approx x_{(\lceil p \cdot k \rceil)}
\]

**Pros:** Exact for small datasets, unbiased estimator  
**Cons:** \(O(k)\) memory, \(O(k \log k)\) sort cost

### Approximate Quantiles (KLL Sketch)

For massive datasets, use **KLL sketch** (Karnin, Lang, Liberty):

**Properties:**
- **Space:** \(O(\frac{1}{\epsilon} \log \log \frac{1}{\delta})\)
- **Error bound:** \(\epsilon\)-approximate with probability \(1-\delta\)
- **Mergeable:** Combine sketches from multiple streams

**Example:** With \(\epsilon=0.01\), quantiles accurate to ±1 percentile using ~1 KB memory.

**Reference:** Karnin, Z., Lang, K., Liberty, E. (2016), "Optimal Quantile Approximation in Streams", arXiv:1603.05346.

### T-Digest (Alternative)

**T-digest** (Dunning & Ertl) provides excellent **tail accuracy**:

**Properties:**
- Better accuracy for extreme quantiles (P99, P99.9)
- Adaptive compression
- Mergeable

**Reference:** Dunning, T., Ertl, O. (2019), "Computing Extremely Accurate Quantiles Using t-Digests", arXiv:1902.04023.

## Histogram Construction

### Freedman-Diaconis Rule

Optimal bin width for histograms:

\[
h = 2 \cdot \frac{\text{IQR}}{n^{1/3}}
\]

Number of bins:

\[
k = \left\lceil \frac{\max - \min}{h} \right\rceil
\]

**Rationale:** Balances bias and variance; works well for wide variety of distributions.

### Sturges' Rule (Alternative)

\[
k = \lceil \log_2 n \rceil + 1
\]

Simpler but may undersmooth for large \(n\).

### Scott's Rule (Alternative)

\[
h = 3.5 \cdot \frac{s}{n^{1/3}}
\]

Assumes normal distribution; similar to Freedman-Diaconis.

## Distinct Count Estimation

For numeric columns with many repeated values (e.g., categorical-like integers):

### KMV (K-Minimum Values)

Maintain the \(k\) smallest hash values from column.

**Estimator:**

\[
\hat{n}_{\text{distinct}} = \frac{k-1}{x_k}
\]

where \(x_k\) is the \(k\)-th smallest hash (normalized to [0,1]).

**Error bound:**

\[
\text{Relative error} \approx \frac{1}{\sqrt{k}}
\]

**Example:** \(k=2048\) → ~2.2% error (95% confidence)

**Properties:**
- **Mergeable**: Union of two KMV sketches
- **Space**: \(O(k)\) per column
- **Update**: \(O(\log k)\) per value

### HyperLogLog (Alternative)

**Space:** \(O(\epsilon^{-2})\) for relative error \(\epsilon\)  
**Error:** Typical 2% with 1.5 KB  
**Standard:** Redis, BigQuery, many production systems

## Missing, NaN, and Inf Handling

- **Missing (NULL):** Excluded from moment calculations; counted separately
- **NaN (Not-a-Number):** Treated as missing
- **±Inf:** Excluded from moments; counted under `inf_count` and surfaced in warnings
- **Type coercion:** Strings parsing to numbers counted only if parsing enabled

!!! warning "Edge Cases"
    - All-missing columns: statistics undefined (reported as `null`)
    - \(n < 2\): variance/shape undefined
    - \(n < 4\): kurtosis undefined

## Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Moments** | \(O(n)\) total, \(O(1)\) per value | \(O(1)\) | Welford/Pébay |
| **Reservoir sampling** | \(O(n)\) total, \(O(1)\) amortized | \(O(k)\) | \(k\) = sample size |
| **Quantiles (exact)** | \(O(k \log k)\) | \(O(k)\) | Sort sample |
| **KLL sketch** | \(O(n \log \log n)\) | \(O(\epsilon^{-1} \log \log n)\) | Approximate |
| **KMV distinct** | \(O(n \log k)\) | \(O(k)\) | Heap operations |
| **Histogram** | \(O(n + k)\) | \(O(b)\) | \(b\) bins |

## Configuration

Control numeric analysis via `ReportConfig`:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Sample size for quantiles/histograms
config.compute.numeric_sample_size = 20_000  # Default

# Sketch sizes
config.compute.uniques_sketch_size = 2_048  # KMV (default)
config.compute.top_k_size = 50  # Top values (if tracking)

# Quantiles to compute
# (Not yet configurable, default: [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

# Histogram bins
# (Automatic via Freedman-Diaconis, max 256)

# Outlier detection method
# (Always computed: IQR, z-score, MAD)

# Random seed for reproducibility
config.compute.random_seed = 42

report = profile(df, config=config)
```

## Implementation Details

### NumericAccumulator Class

```python
class NumericAccumulator:
    def __init__(self, name: str, config: NumericConfig):
        self.name = name
        self.count = 0
        self.missing = 0
        self.zeros = 0
        self.negatives = 0
        self.inf = 0
        
        # Streaming moments
        self._moments = StreamingMoments()
        
        # Reservoir sample for quantiles
        self._sample = ReservoirSampler(config.sample_size)
        
        # KMV for distinct count
        self._uniques = KMV(config.uniques_sketch_size)
        
        # Min/max tracking
        self._extremes = ExtremeTracker()
    
    def update(self, values: np.ndarray):
        """Update with chunk of values"""
        # Filter out missing/NaN/Inf
        # Update moments
        # Update sample
        # Update KMV
        # Track extremes
        pass
    
    def finalize(self) -> NumericSummary:
        """Compute final statistics"""
        # Compute mean, variance, skewness, kurtosis from moments
        # Compute quantiles from sample
        # Estimate distinct count from KMV
        # Build histogram
        # Detect outliers
        return NumericSummary(...)
```

## Validation

PySuricata validates numeric algorithms against reference implementations:

- **NumPy/SciPy**: Cross-check mean, variance, skewness, kurtosis on small datasets
- **Property-based tests**: Invariants under concatenation (merge = single pass)
- **Scaling laws**: \(\text{Var}(aX) = a^2 \text{Var}(X)\)
- **Translation laws**: \(\text{Mean}(X+c) = \text{Mean}(X) + c\)
- **Numerical stability**: Test with extreme values, large cancellations

## Examples

### Basic Usage

```python
import pandas as pd
from pysuricata import profile

df = pd.DataFrame({"amount": [10, 20, 30, None, 50]})
report = profile(df)
report.save_html("report.html")
```

### Streaming Large Dataset

```python
from pysuricata import profile, ReportConfig

def read_chunks():
    for i in range(100):
        yield pd.read_parquet(f"data/part-{i}.parquet")

config = ReportConfig()
config.compute.numeric_sample_size = 50_000
config.compute.random_seed = 42

report = profile(read_chunks(), config=config)
```

### Access Statistics Programmatically

```python
from pysuricata import summarize

stats = summarize(df)
amount_stats = stats["columns"]["amount"]

print(f"Mean: {amount_stats['mean']}")
print(f"Std: {amount_stats['std']}")
print(f"Skewness: {amount_stats['skewness']}")
```

## References

1. **Welford, B.P. (1962)**, "Note on a Method for Calculating Corrected Sums of Squares and Products", *Technometrics*, 4(3): 419–420.

2. **Pébay, P. (2008)**, "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments", Sandia Report SAND2008-6212. [PDF](https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf)

3. **Karnin, Z., Lang, K., Liberty, E. (2016)**, "Optimal Quantile Approximation in Streams", *IEEE FOCS*. [arXiv:1603.05346](https://arxiv.org/abs/1603.05346)

4. **Dunning, T., Ertl, O. (2019)**, "Computing Extremely Accurate Quantiles Using t-Digests", [arXiv:1902.04023](https://arxiv.org/abs/1902.04023)

5. **Tukey, J.W. (1977)**, *Exploratory Data Analysis*, Addison-Wesley.

6. **Freedman, D., Diaconis, P. (1981)**, "On the histogram as a density estimator", *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 57: 453–476.

7. **Wikipedia: Algorithms for calculating variance** - [Link](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)

8. **Wikipedia: Skewness** - [Link](https://en.wikipedia.org/wiki/Skewness)

9. **Wikipedia: Kurtosis** - [Link](https://en.wikipedia.org/wiki/Kurtosis)

10. **Wikipedia: Median absolute deviation** - [Link](https://en.wikipedia.org/wiki/Median_absolute_deviation)

## See Also

- [Categorical Analysis](categorical.md) - String/categorical variables
- [DateTime Analysis](datetime.md) - Temporal variables
- [Streaming Algorithms](../algorithms/streaming.md) - Welford/Pébay deep dive
- [Sketch Algorithms](../algorithms/sketches.md) - KMV, HyperLogLog, KLL
- [Configuration Guide](../configuration.md) - All parameters

---

*Last updated: 2025-10-12*




