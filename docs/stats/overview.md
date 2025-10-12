---
title: Statistical Methods Overview
description: High-level overview of all statistical analysis methods in PySuricata
---

# Statistical Methods Overview

PySuricata analyzes **four variable types** with specialized algorithms for each.

## Analysis by Variable Type

### Numeric Variables

**Exact statistics** using Welford/Pébay streaming algorithms:
- Mean, variance, standard deviation
- Skewness, kurtosis
- Min, max, range

**Approximate statistics** using probabilistic data structures:
- Quantiles (reservoir sampling)
- Distinct count (KMV sketch)
- Histograms (adaptive binning)

**Key formulas:**
\[
\bar{x} = \frac{1}{n}\sum x_i, \quad s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2
\]

**→ [Full Documentation](numeric.md)**

### Categorical Variables

**Analysis includes:**
- Top-k values (Misra-Gries algorithm)
- Distinct count (KMV sketch)
- Entropy and Gini impurity
- String statistics

**Key formulas:**
\[
H(X) = -\sum p(x) \log_2 p(x), \quad \text{Gini}(X) = 1 - \sum p(x)^2
\]

**→ [Full Documentation](categorical.md)**

### DateTime Variables

**Temporal analysis:**
- Hour, day-of-week, month distributions
- Monotonicity detection
- Timeline visualizations

**Key formulas:**
\[
\Delta t = \max(t) - \min(t), \quad M = \frac{n_{\text{increasing}}}{n-1}
\]

**→ [Full Documentation](datetime.md)**

### Boolean Variables

**Binary analysis:**
- True/false counts and ratios
- Entropy calculation
- Imbalance detection

**Key formulas:**
\[
H = -p \log_2(p) - (1-p)\log_2(1-p)
\]

**→ [Full Documentation](boolean.md)**

## Advanced Analytics

### Correlations

Streaming Pearson correlation between numeric columns.

\[
r_{XY} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}
\]

**→ [Full Documentation](../analytics/correlations.md)**

### Missing Values

Intelligent missing data analysis with chunk-level tracking.

\[
MR = \frac{n_{\text{missing}}}{n_{\text{total}}}
\]

**→ [Full Documentation](../analytics/missing-values.md)**

## Algorithms

### Streaming Statistics

- **Welford's algorithm**: Online mean/variance
- **Pébay's formulas**: Parallel merging

**→ [Full Documentation](../algorithms/streaming.md)**

### Sketch Algorithms

- **KMV**: Distinct count estimation
- **Misra-Gries**: Top-k heavy hitters
- **Reservoir sampling**: Uniform sampling

**→ [Full Documentation](../algorithms/sketches.md)**

## Guarantees

| Method | Type | Error |
|--------|------|-------|
| Mean, variance | Exact | Machine precision |
| Skewness, kurtosis | Exact | Machine precision |
| Distinct (KMV) | Approximate | ~2% (k=2048) |
| Top-k (Misra-Gries) | Guarantee | All freq > n/k found |
| Quantiles (reservoir) | Exact | From uniform sample |

## See Also

- [Numeric Analysis](numeric.md) - Complete numeric documentation
- [Categorical Analysis](categorical.md) - Categorical methods
- [DateTime Analysis](datetime.md) - Temporal analysis
- [Boolean Analysis](boolean.md) - Binary variables
- [Streaming Algorithms](../algorithms/streaming.md) - Algorithm details
- [Sketch Algorithms](../algorithms/sketches.md) - Probabilistic structures

---

*Last updated: 2025-10-12*




