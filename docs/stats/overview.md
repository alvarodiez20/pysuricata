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

$$
\bar{x} = \frac{1}{n}\sum x_i, \quad s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2
$$

**→ [Full Documentation](numeric.md)**

### Categorical Variables

**Analysis includes:**

- Top-k values (Misra-Gries algorithm)
- Distinct count (KMV sketch)
- Entropy and Gini impurity
- String statistics

**Key formulas:**

$$
H(X) = -\sum p(x) \log_2 p(x), \quad \text{Gini}(X) = 1 - \sum p(x)^2
$$

**→ [Full Documentation](categorical.md)**

### DateTime Variables

**Temporal analysis:**

- Hour, day-of-week, month distributions
- Monotonicity detection
- Time span and sampling rate

**Key formulas:**

$$
M = \frac{n_{\uparrow}}{n - 1}, \quad r = \frac{n}{\Delta t}
$$

**→ [Full Documentation](datetime.md)**

### Boolean Variables

**Binary analysis:**

- True/False proportions
- Entropy and balance metrics
- Imbalance detection

**Key formulas:**

$$
H = -p \log_2(p) - (1-p) \log_2(1-p)
$$

**→ [Full Documentation](boolean.md)**

## Advanced Analytics

### Correlations

Streaming Pearson correlation between numeric columns, using pairwise co-moment tracking. Correlations above a configurable threshold are reported.

$$
r_{xy} = \frac{\text{Cov}(X, Y)}{s_X \cdot s_Y}
$$

### Missing Values

Per-column and dataset-wide missing value analysis:

- Missing count and percentage per column
- Top missing columns visualization
- Missing pattern detection

## Algorithms

All statistics use **single-pass streaming algorithms** with bounded memory:

| Algorithm | Used for | Space |
|-----------|----------|-------|
| Welford/Pébay | Mean, variance, skewness, kurtosis | O(1) |
| Reservoir sampling | Quantiles, histograms | O(s) |
| KMV sketch | Distinct count | O(k) |
| Misra-Gries | Top-k frequent values | O(k) |

## Guarantees

- **Exact:** moments (mean, variance, skewness, kurtosis), min/max, counts
- **Approximate:** quantiles (within ±1 percentile), distinct count (~2.2% error with default k=2048)
- **Deterministic:** set `random_seed` for reproducible reservoir sampling

## See Also

- [Streaming Algorithms](../algorithms/streaming.md) — Algorithm details
- [Sketch Algorithms](../algorithms/sketches.md) — Probabilistic structures
