---
title: Boolean Variable Analysis
description: Complete mathematical formulas for boolean/binary variable analysis in pysuricata
---

# Boolean Variable Analysis

Comprehensive documentation for analyzing boolean (True/False) variables in PySuricata with information-theoretic measures.

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

PySuricata treats **boolean variables** as columns with two distinct values (True/False, 1/0, Yes/No). Analysis focuses on balance, information content, and missing patterns.

### Key Features

- **True/False counts** and ratios
- **Entropy** (information content, in bits)
- **Missing value handling**

## Summary Statistics Provided

Exactly what `summarize()` publishes for a boolean column, and nothing else:

| key | meaning |
|---|---|
| `count` | non-missing values |
| `missing` | missing values |
| `true`, `false` | counts |
| `true_ratio`, `false_ratio` | \(p = n_{\text{true}} / n\), and \(1 - p\) |
| `entropy` | Shannon entropy, in bits |
| `mem_bytes`, `dtype` | bookkeeping |

The derived quantities below — imbalance ratio, balance score, information
content per value — are **not** in the payload. They are one line of arithmetic
from `true_ratio`, and they are documented here because reading a boolean column
means reasoning about them, not because PySuricata computes them for you.

## Mathematical Definitions

### Basic Counts

Let the boolean column have:
- \(n_{\text{true}}\) = count of True values
- \(n_{\text{false}}\) = count of False values
- \(n_{\text{missing}}\) = count of missing/null values
- \(n = n_{\text{true}} + n_{\text{false}}\) = non-null count
- \(n_{\text{total}} = n + n_{\text{missing}}\) = total observations

### Probability

**Probability of True:**

\[
p = \frac{n_{\text{true}}}{n}
\]

**Probability of False:**

\[
q = 1 - p = \frac{n_{\text{false}}}{n}
\]

### True/False Ratio

\[
R = \frac{n_{\text{true}}}{n_{\text{false}}}
\]

**Interpretation:**
- \(R = 1\): perfectly balanced (50/50)
- \(R > 1\): more True than False
- \(R < 1\): more False than True
- \(R \to \infty\): nearly all True
- \(R \to 0\): nearly all False

### Imbalance Ratio { data-derived }

!!! note "Derived, not published"
    Not a key in the payload. Compute it from `true_ratio` if you want it.

Measures deviation from balanced distribution:

\[
I = \frac{|n_{\text{true}} - n_{\text{false}}|}{n} = |2p - 1|
\]

**Properties:**
- \(I = 0\): perfectly balanced (\(p = 0.5\))
- \(I = 1\): completely imbalanced (\(p = 0\) or \(p = 1\))
- Range: \([0, 1]\)

**Interpretation:**
- \(I < 0.2\): well balanced (40/60 to 60/40)
- \(0.2 \le I < 0.6\): moderately imbalanced
- \(I \ge 0.6\): severely imbalanced
- \(I > 0.9\): nearly constant

### Balance Score { data-derived }

!!! note "Derived, not published"
    Not a key in the payload. Compute it from `true_ratio` if you want it.

Alternative measure of balance:

\[
B = 1 - |0.5 - p|
\]

**Properties:**
- \(B = 1\): perfectly balanced (\(p = 0.5\))
- \(B = 0.5\): completely imbalanced (\(p = 0\) or \(p = 1\))
- Range: \([0.5, 1]\)

**Interpretation:**
- \(B > 0.9\): well balanced
- \(0.7 < B \le 0.9\): moderately balanced
- \(B \le 0.7\): imbalanced

### Shannon Entropy

Measures the **information content** or **uncertainty** in the boolean distribution:

\[
H = -p \log_2(p) - (1-p) \log_2(1-p)
\]

By convention, \(0 \log_2(0) = 0\).

**Properties:**
- \(H = 0\) bits if \(p = 0\) or \(p = 1\) (no uncertainty, deterministic)
- \(H = 1\) bit if \(p = 0.5\) (maximum uncertainty, uniformly random)
- Range: \([0, 1]\) bits

**Interpretation:**
- \(H < 0.5\): low information content, predictable
- \(H \approx 1.0\): high information content, unpredictable
- \(H = 1.0\): fair coin flip

**Entropy vs. Probability:**

| \(p\) | \(H\) (bits) | Interpretation |
|-------|--------------|----------------|
| 0.0 | 0.00 | No information (constant False) |
| 0.1 | 0.47 | Low entropy, mostly False |
| 0.5 | 1.00 | Maximum entropy, balanced |
| 0.9 | 0.47 | Low entropy, mostly True |
| 1.0 | 0.00 | No information (constant True) |

### Information Content per True Value { data-derived }

!!! note "Derived, not published"
    Not a key in the payload.

Average information conveyed by each True observation:

\[
IC_{\text{true}} = -\log_2(p) \text{ bits}
\]

**Example:**
- \(p = 0.5\): \(IC = 1\) bit (unsurprising)
- \(p = 0.1\): \(IC = 3.32\) bits (rare event, informative)
- \(p = 0.01\): \(IC = 6.64\) bits (very rare, very informative)

**Use case:** In imbalanced classification, rare class has higher information content.

### Information Content per False Value { data-derived }

!!! note "Derived, not published"
    Not a key in the payload.

\[
IC_{\text{false}} = -\log_2(1 - p) \text{ bits}
\]

## Why There Is No Balance Test

A binomial test of \(H_0: p = 0.5\) is the obvious next step, and PySuricata
does not run one. Deliberately:

\[
Z = \frac{\hat{p} - 0.5}{\sqrt{0.25 / n}}
\]

At profiling scale that statistic is not informative. With \(n = 10^6\), a true
rate of 0.501 — a difference nobody cares about — gives \(Z = 2\) and a
"significant" result. The test answers *is this exactly 0.5*, which is never the
question; the question is *is this far enough from 0.5 to matter*, and that is
`true_ratio` against a threshold you choose.

Running it anyway would put a p-value on every boolean card that is essentially
a function of the row count. If you want the test, you have \(n\) and
\(\hat{p}\) in the payload.

## Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Count True/False** | \(O(n)\) | \(O(1)\) | Single pass |
| **Entropy** | \(O(1)\) | \(O(1)\) | From counts |
| **All metrics** | \(O(n)\) | \(O(1)\) | Single pass |

Boolean analysis is extremely efficient: O(1) space, O(n) time.

## Configuration

Control boolean analysis via `ProfileConfig`:

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()

# Boolean-specific config
# (Currently no boolean-specific parameters)

report = profile(df, config=config)
```

## Implementation Details

`BooleanAccumulator` lives in `pysuricata/accumulators/boolean.py` and
`BooleanSummary` beside it. Rather than a sketch that can drift, the shape:

- It subclasses `PicklableAccumulator`, so a long run can be checkpointed.
- `update()` takes a **numpy array**, never a frame or a Series — the adapter
  has already converted the column. That is what makes the accumulator testable
  in isolation and mergeable across machines.
- State is four integers: `count`, `missing`, `true_n`, `false_n`. Everything on
  the summary is derived from those at `finalize()`, which is why the whole
  column kind is \(O(1)\) in space.
- `merge()` adds the four counters. Exact, order-independent, and the reason
  chunked results equal unchunked ones.
- `chunk_metadata` carries `(start_row, end_row, missing_in_chunk)` per chunk,
  which is what the Missing Values pane is drawn from.

## Examples

### Basic Usage

```python
import pandas as pd
from pysuricata import profile

df = pd.DataFrame({
    "is_active": [True, False, True, True, None, False]
})

report = profile(df)
report.save_html("report.html")
```

### Imbalanced Boolean

```python
import pandas as pd

from pysuricata import profile

# Highly imbalanced (10% True)
df = pd.DataFrame({
    "is_fraud": [False] * 90 + [True] * 10
})

report = profile(df)
# Will show low entropy and a true_ratio near 0.1
```

### Access Statistics

```python
import pandas as pd

from pysuricata import summarize

df = pd.DataFrame({"is_active": [True, True, False, True, False, True]})
stats = summarize(df)
active_stats = stats["columns"]["is_active"]

print(f"True count:  {active_stats['true']}")
print(f"False count: {active_stats['false']}")
print(f"True %:      {active_stats['true'] / active_stats['count']:.1%}")
print(f"Missing:     {active_stats['missing']}")
```

## Interpreting Results

### Well-Balanced (p ≈ 0.5)

- `true_ratio` between 0.4 and 0.6
- `entropy` ≈ 1.0 bit

**Implications:**
- High information content
- Good for binary classification (no class imbalance)
- Unpredictable values

**Example:** Fair coin flip, A/B test with even split.

### Imbalanced (p << 0.5 or p >> 0.5)

- `true_ratio` below 0.2 or above 0.8
- `entropy` < 0.5 bits

**Implications:**
- Low information content
- May need rebalancing for ML
- Predictable values

**Example:** Fraud detection (1% positive), rare disease (0.1% positive).

### Nearly Constant (p < 0.01 or p > 0.99)

- `true_ratio` below 0.01 or above 0.99
- `entropy` < 0.1 bits

**Implications:**
- Almost no information
- Consider removing column
- May indicate data quality issue

**Example:** "is_deleted" flag in active records table (all False).

## Use in Machine Learning

### Class Imbalance

For binary classification with boolean target:

**Balanced** (\(0.4 < p < 0.6\)):
- Standard algorithms work well
- Use accuracy as metric

**Moderately imbalanced** (\(0.1 < p < 0.4\) or \(0.6 < p < 0.9\)):
- Consider class weights
- Use F1-score, AUC-ROC
- Try SMOTE for oversampling

**Severely imbalanced** (\(p < 0.1\) or \(p > 0.9\)):
- Must use rebalancing techniques
- Precision-recall curve essential
- Consider anomaly detection instead

### Entropy as Feature Quality

High entropy boolean features (\(H \approx 1\)):
- Good discriminative power
- Worth including in model

Low entropy boolean features (\(H < 0.5\)):
- Low information content
- May not help model
- Consider interaction terms

## Special Cases

### All True or All False

- `entropy` = 0 (no information)
- `true_ratio` is 1.0 or 0.0

**Recommendation:** Remove column (constant value).

### All Missing

- No non-null values
- Statistics undefined

**Recommendation:** Remove column or investigate data source.

### Three-Valued Boolean

Columns with True, False, and many NULLs:

**Interpretation:** May be ternary (True/False/Unknown) rather than binary.

**Recommendation:**
- Report missing percentage
- Consider as categorical instead
- Imputation may not be appropriate

## References

1. **Shannon, C.E. (1948)**, "A Mathematical Theory of Communication", *Bell System Technical Journal*, 27: 379–423.

2. **Cover, T.M., Thomas, J.A. (2006)**, *Elements of Information Theory*, 2nd ed., Wiley.

3. **Chawla, N.V. et al. (2002)**, "SMOTE: Synthetic Minority Over-sampling Technique", *JAIR*, 16: 321–357.

4. **He, H., Garcia, E.A. (2009)**, "Learning from Imbalanced Data", *IEEE TKDE*, 21(9): 1263–1284.

5. **Wikipedia: Entropy (information theory)** - [Link](https://en.wikipedia.org/wiki/Entropy_(information_theory))

6. **Wikipedia: Binary classification** - [Link](https://en.wikipedia.org/wiki/Binary_classification)

## See Also

- [Categorical Analysis](categorical.md) - For multi-class variables
- [Data Quality](../analytics/quality.md) - Quality metrics
- [Configuration Guide](../configuration.md) - All parameters
