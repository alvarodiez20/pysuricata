---
title: Correlation Analysis
description: Streaming correlation computation with mathematical formulas and implementation details
---

# Correlation Analysis

PySuricata computes **pairwise correlations** between numeric columns using **streaming algorithms** that operate in bounded memory.

!!! info "Examples on this page assume a DataFrame named `df`"

    Every snippet below that does not build its own frame expects one already in
    scope. Paste this first to follow along:

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(5_000),
            "amount": rng.lognormal(3, 1, 5_000),
            "country": rng.choice(["ES", "FR", "DE"], 5_000),
            "signed_up": pd.date_range("2024-01-01", periods=5_000, freq="17min"),
            "active": rng.random(5_000) > 0.3,
        }
    )
    ```

## Overview

Correlation analysis reveals linear relationships between numeric variables, helping identify:
- Redundant features (highly correlated)
- Related measurements (positively/negatively correlated)
- Independent variables (near-zero correlation)

### Key Features

- **Streaming computation**: O(p²) space for p numeric columns
- **Single-pass algorithm**: No need to store full data
- **Exact Pearson correlation**: Not approximate
- **Configurable threshold**: Only report significant correlations
- **Per-column top-k**: Show most correlated pairs

## Pearson Correlation Coefficient

### Definition

For two numeric variables X and Y with observations \((x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\):

\[
r_{XY} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
\]

where \(\bar{x}\) and \(\bar{y}\) are the means.

### Alternative Formula

\[
r_{XY} = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{\sqrt{n\sum x_i^2 - (\sum x_i)^2} \sqrt{n\sum y_i^2 - (\sum y_i)^2}}
\]

This form enables **streaming computation** by maintaining sufficient statistics.

### Properties

- **Range**: \(r \in [-1, 1]\)
- **Interpretation**:
  - \(r = 1\): perfect positive linear relationship
  - \(r = -1\): perfect negative linear relationship
  - \(r = 0\): no linear relationship
  - \(0 < r < 1\): positive correlation
  - \(-1 < r < 0\): negative correlation

### Strength Guidelines

| \(|r|\) Range | Strength |
|---------------|----------|
| 0.0 - 0.2 | Very weak |
| 0.2 - 0.4 | Weak |
| 0.4 - 0.6 | Moderate |
| 0.6 - 0.8 | Strong |
| 0.8 - 1.0 | Very strong |

## Streaming Algorithm

### Sufficient Statistics

To compute \(r_{XY}\) without storing all data, maintain:

\[
\begin{aligned}
n &= \text{count of pairs} \\
S_x &= \sum x_i \\
S_y &= \sum y_i \\
S_{xx} &= \sum x_i^2 \\
S_{yy} &= \sum y_i^2 \\
S_{xy} &= \sum x_i y_i
\end{aligned}
\]

### Update Step

For each new pair \((x, y)\):

\[
\begin{aligned}
n &\leftarrow n + 1 \\
S_x &\leftarrow S_x + x \\
S_y &\leftarrow S_y + y \\
S_{xx} &\leftarrow S_{xx} + x^2 \\
S_{yy} &\leftarrow S_{yy} + y^2 \\
S_{xy} &\leftarrow S_{xy} + xy
\end{aligned}
\]

### Finalize

Compute correlation:

\[
r = \frac{nS_{xy} - S_x S_y}{\sqrt{nS_{xx} - S_x^2} \sqrt{nS_{yy} - S_y^2}}
\]

### Missing Values

Only pairs with **both values present** are included:

```python
mask = ~(isnan(x) | isnan(y) | isinf(x) | isinf(y))
x_valid = x[mask]
y_valid = y[mask]
# Update with valid pairs only
```

## Why There Are No P-Values

The t-test for \(H_0: \rho = 0\),

\[
t = r \sqrt{\frac{n-2}{1-r^2}}, \qquad t \sim t_{n-2},
\]

is cheap to compute from numbers PySuricata already has, and it is not reported.
Two reasons.

**At profiling scale it is always significant.** With \(n = 10^6\), an
\(|r|\) of 0.002 clears \(p < 0.05\). A p-value column would say "yes" beside
every pair, which is not information. What a reader needs is the *magnitude*,
which is what `corr_threshold` filters on.

**The multiplicity is severe and unfixable here.** Testing every pair means
\(m = \binom{p}{2}\) tests — 1,225 for 50 columns, so a Bonferroni-adjusted
\(\alpha\) of \(0.05/1225 \approx 4 \times 10^{-5}\). Correcting honestly
makes the test useless; not correcting makes it misleading. Neither is worth
putting on a card.

Correlations are reported as what they are: a magnitude, above a threshold you
set, with the pair count visible so you can see how many comparisons produced
it.

## Implementation

`StreamingCorr` lives in `pysuricata/compute/analysis/correlation.py`. It holds
**sufficient statistics**, not the data: per column \(\sum x\) and
\(\sum x^2\), and per pair \(\sum xy\) and a joint count. Everything above
is recovered from those at finalize time.

That is what makes it streamable and mergeable — adding two sets of sums gives
the same answer as one pass over the concatenation, so chunked results equal
unchunked ones here as everywhere else.

Two details worth knowing:

- **Missing values are handled pairwise.** A pair's count is the number of rows
  where *both* columns are finite, so two columns with different missingness
  still get a correct \(r\) rather than one computed against a mismatched
  \(n\).
- **The finite masks are computed once per column** and reused across every pair
  it appears in, rather than recomputed per pair. On a wide frame that is most
  of the saving.

## Complexity

### Space Complexity

For \(p\) numeric columns:
- Number of pairs: \(m = \binom{p}{2} = \frac{p(p-1)}{2} = O(p^2)\)
- Space per pair: O(1) (6 floating-point values)
- **Total space**: O(p²)

**Example:**
- 10 columns → 45 pairs → ~2 KB
- 50 columns → 1,225 pairs → ~50 KB
- 100 columns → 4,950 pairs → ~200 KB

### Time Complexity

Per chunk with \(n\) rows and \(p\) columns:
- Iterate over \(O(p^2)\) pairs
- For each pair: \(O(n)\) to compute valid mask and sums
- **Total per chunk**: O(n p²)

For dataset with \(N\) total rows:
- **Total time**: O(N p²)

### When to Disable

For large \(p\) (many columns), correlation computation can be expensive:

- \(p > 100\): Consider disabling or using sampling
- \(p > 500\): Strongly recommend disabling

Configuration:

```python
from pysuricata import ProfileConfig
config = ProfileConfig()
config.compute.compute_correlations = False  # Disable
```

## Configuration

Control correlation analysis via `ProfileConfig`:

```python
from pysuricata import profile, ProfileConfig, ComputeOptions

# Using the public API
config = ProfileConfig(compute=ComputeOptions(
    compute_correlations=True,  # Default
    corr_threshold=0.5,  # Default (only |r| >= 0.5)
    corr_max_cols=50,  # Default (skip if > 50 cols)
    corr_max_per_col=10  # Default (top 10 per column)
))

report = profile(df, config=config)
```

## Interpretation

### High Positive Correlation (r > 0.8)

**Interpretation:** Variables move together strongly.

**Examples:**
- Height and weight (r ≈ 0.7-0.8)
- Temperature in °F and °C (r = 1.0, exact conversion)
- Revenue and profit (r ≈ 0.8-0.9)

**Actionable insights:**
- Potential redundancy (consider removing one feature)
- Useful for imputation (predict one from other)
- Check for derived features (one computed from other)

### High Negative Correlation (r < -0.8)

**Interpretation:** Variables move in opposite directions.

**Examples:**
- Latitude and temperature (r ≈ -0.5 to -0.7)
- Altitude and air pressure (r ≈ -0.9)
- Discount and profit margin (r ≈ -0.6)

**Actionable insights:**
- Substitutes or inverse relationships
- Consider composite features (sum, ratio)

### Low Correlation (|r| < 0.2)

**Interpretation:** Little to no linear relationship.

**Note:** Variables may still have **nonlinear** relationships (e.g., quadratic, exponential).

**Actionable insights:**
- Independent features (good for model diversity)
- May need nonlinear analysis (polynomial features, interactions)

## Limitations

### Linear Relationships Only

Pearson correlation measures **linear** association only.

**Example:** Quadratic relationship \(y = x^2\)
- Correlation: \(r \approx 0\) (if x spans negative and positive)
- But strong nonlinear relationship exists

**Solutions:**
- Use Spearman rank correlation (monotonic relationships)
- Plot scatter plots
- Use mutual information (any dependency)

### Sensitive to Outliers

Single extreme value can dominate correlation.

**Solutions:**
- Use Spearman instead (rank-based, robust)
- Remove outliers before computing
- Use robust correlation measures (MAD-based)

### Correlation ≠ Causation

High correlation does **not** imply causation.

**Example:** Ice cream sales and drowning deaths (r ≈ 0.9)
- Spurious correlation (confounded by temperature/summer)

## Alternatives

### Spearman Rank Correlation

Measures **monotonic** (not necessarily linear) relationships.

\[
\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2 - 1)}
\]

where \(d_i\) is the rank difference for observation \(i\).

**Advantages:**
- Captures monotonic nonlinear relationships
- Robust to outliers
- No distribution assumptions

**Disadvantages:**
- Requires sorting (more expensive)
- Not streamable (needs ranks)

!!! note "Not computed"
    Spearman needs ranks, and ranks need the whole column — a streaming pass
    cannot produce them in bounded memory. It could be approximated from the
    reservoir samples, at an error nobody has characterised, which is exactly
    the kind of unlabelled estimate this project avoids. Pearson on the full
    column beats Spearman on a sample.

### Kendall Tau

Another rank-based correlation measure.

**Advantages:**
- More robust than Spearman
- Better for small samples

**Disadvantages:**
- Even more expensive to compute (O(n log n) or O(n²))

### Mutual Information

Measures **any dependency** (linear or nonlinear).

\[
MI(X, Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
\]

**Advantages:**
- Detects any relationship
- Information-theoretic

**Disadvantages:**
- Requires binning (continuous → discrete)
- Harder to interpret than correlation

## Examples

### Basic Usage

```python
import pandas as pd
from pysuricata import profile, ProfileConfig

df = pd.DataFrame({
    "x": range(100),
    "y": [2*i + 1 for i in range(100)],  # y = 2x + 1
    "z": [100 - i for i in range(100)]    # z = 100 - x
})

config = ProfileConfig()
config.compute.compute_correlations = True
config.compute.corr_threshold = 0.5

report = profile(df, config=config)
# Expect: r(x,y) ≈ 1.0, r(x,z) ≈ -1.0, r(y,z) ≈ -1.0
```

### Access Correlations Programmatically

```python
import numpy as np
import pandas as pd

from pysuricata import summarize

rng = np.random.default_rng(0)
x = rng.standard_normal(1_000)
df = pd.DataFrame({"x": x, "y": 2 * x + rng.standard_normal(1_000) * 0.3})

stats = summarize(df)

x_stats = stats["columns"]["x"]
correlations = x_stats.get("corr_top", [])

for col, r in correlations:
    print(f"x vs {col}: r = {r:.3f}")
```

### High-Dimensional Data

```python
import numpy as np
import pandas as pd

from pysuricata import ProfileConfig, profile

rng = np.random.default_rng(0)
wide_df = pd.DataFrame({f"c{i}": rng.standard_normal(1_000) for i in range(200)})

# Many columns: disable correlations
config = ProfileConfig()
config.compute.compute_correlations = False  # Too expensive

report = profile(wide_df, config=config)
```

## References

1. **Pearson, K. (1895)**, "Notes on regression and inheritance in the case of two parents", *Proceedings of the Royal Society of London*, 58: 240–242.

2. **Rodgers, J.L., Nicewander, W.A. (1988)**, "Thirteen Ways to Look at the Correlation Coefficient", *The American Statistician*, 42(1): 59–66.

3. **Spearman, C. (1904)**, "The proof and measurement of association between two things", *American Journal of Psychology*, 15: 72–101.

4. **Benjamini, Y., Hochberg, Y. (1995)**, "Controlling the false discovery rate: a practical and powerful approach to multiple testing", *Journal of the Royal Statistical Society B*, 57(1): 289–300.

5. **Wikipedia: Pearson correlation coefficient** - [Link](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

6. **Wikipedia: Spearman's rank correlation** - [Link](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

## See Also

- [Numeric Analysis](../stats/numeric.md) - Univariate numeric statistics
- [Streaming Algorithms](../algorithms/streaming.md) - Streaming computation techniques
- [Configuration Guide](../configuration.md) - All parameters
