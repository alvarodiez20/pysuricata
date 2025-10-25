---
title: Correlation Analysis
description: Streaming correlation computation with mathematical formulas and implementation details
---

# Correlation Analysis

PySuricata computes **pairwise correlations** between numeric columns using **streaming algorithms** that operate in bounded memory.

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

## Statistical Significance

### t-test for Correlation

Test \(H_0: \rho = 0\) (no correlation in population).

**Test statistic:**

\[
t = r \sqrt{\frac{n-2}{1-r^2}}
\]

Under \(H_0\), \(t \sim t_{n-2}\) (Student's t-distribution with \(n-2\) degrees of freedom).

**P-value:**

\[
p = 2 \cdot P(T_{n-2} > |t|)
\]

**Reject \(H_0\) if** \(p < \alpha\) (e.g., \(\alpha = 0.05\)).

!!! note "Not implemented in current version"
    Significance tests are planned for future release. Current version reports raw correlations.

### Multiple Testing Correction

When testing \(m = \binom{p}{2}\) pairs, use **Bonferroni correction**:

\[
\alpha_{\text{adj}} = \frac{\alpha}{m}
\]

Or **False Discovery Rate (FDR)** control via Benjamini-Hochberg procedure.

**Example:** 50 columns → 1,225 pairs
- Bonferroni: \(\alpha_{\text{adj}} = 0.05/1225 \approx 0.00004\)
- Very conservative

## Implementation

### StreamingCorr Class

```python
class StreamingCorr:
    def __init__(self, columns: List[str]):
        self.cols = columns
        self.pairs = {}  # (col1, col2) -> {n, sx, sy, sxx, syy, sxy}

    def update(self, df: pd.DataFrame):
        """Update with chunk of data"""
        for i, col1 in enumerate(self.cols):
            for j in range(i+1, len(self.cols)):
                col2 = self.cols[j]

                # Extract values
                x = df[col1].to_numpy()
                y = df[col2].to_numpy()

                # Filter valid pairs
                mask = np.isfinite(x) & np.isfinite(y)
                x_valid = x[mask]
                y_valid = y[mask]

                if len(x_valid) == 0:
                    continue

                # Update sufficient statistics
                key = (col1, col2)
                if key not in self.pairs:
                    self.pairs[key] = {
                        'n': 0, 'sx': 0, 'sy': 0,
                        'sxx': 0, 'syy': 0, 'sxy': 0
                    }

                stats = self.pairs[key]
                stats['n'] += len(x_valid)
                stats['sx'] += float(np.sum(x_valid))
                stats['sy'] += float(np.sum(y_valid))
                stats['sxx'] += float(np.sum(x_valid ** 2))
                stats['syy'] += float(np.sum(y_valid ** 2))
                stats['sxy'] += float(np.sum(x_valid * y_valid))

    def finalize(self, threshold: float = 0.0) -> Dict:
        """Compute final correlations"""
        results = {}

        for (col1, col2), stats in self.pairs.items():
            n = stats['n']
            if n < 2:
                continue

            # Compute correlation
            num = n * stats['sxy'] - stats['sx'] * stats['sy']
            den1 = n * stats['sxx'] - stats['sx'] ** 2
            den2 = n * stats['syy'] - stats['sy'] ** 2

            if den1 <= 0 or den2 <= 0:
                continue

            r = num / (math.sqrt(den1) * math.sqrt(den2))

            # Filter by threshold
            if abs(r) >= threshold:
                results[(col1, col2)] = r

        return results
```

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
config = ReportConfig()
config.compute.compute_correlations = False  # Disable
```

## Configuration

Control correlation analysis via `ReportConfig`:

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

!!! note "Not implemented in current version"
    Spearman correlation is planned for future release.

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
from pysuricata import profile, ReportConfig

df = pd.DataFrame({
    "x": range(100),
    "y": [2*i + 1 for i in range(100)],  # y = 2x + 1
    "z": [100 - i for i in range(100)]    # z = 100 - x
})

config = ReportConfig()
config.compute.compute_correlations = True
config.compute.corr_threshold = 0.5

report = profile(df, config=config)
# Expect: r(x,y) ≈ 1.0, r(x,z) ≈ -1.0, r(y,z) ≈ -1.0
```

### Access Correlations Programmatically

```python
from pysuricata import summarize

stats = summarize(df)

x_stats = stats["columns"]["x"]
correlations = x_stats.get("corr_top", [])

for col, r in correlations:
    print(f"x vs {col}: r = {r:.3f}")
```

### High-Dimensional Data

```python
# Many columns: disable correlations
config = ReportConfig()
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

---

*Last updated: 2025-10-12*
