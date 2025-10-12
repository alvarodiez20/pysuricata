---
title: Boolean Variable Analysis
description: Complete mathematical formulas for boolean/binary variable analysis in pysuricata
---

# Boolean Variable Analysis

Comprehensive documentation for analyzing boolean (True/False) variables in PySuricata with information-theoretic measures.

## Overview

PySuricata treats **boolean variables** as columns with two distinct values (True/False, 1/0, Yes/No). Analysis focuses on balance, information content, and missing patterns.

### Key Features

- **True/False counts** with percentages
- **Balance ratio** (distribution symmetry)
- **Entropy** (information content)
- **Information per value** (bits)
- **Imbalance detection** (skewed distributions)
- **Missing value handling**

## Summary Statistics Provided

For each boolean column:

- **Count**: total non-null values
- **True count**: number of True values
- **False count**: number of False values
- **Missing count**: number of missing/null values
- **True percentage**: \(p = n_{\text{true}} / n\)
- **False percentage**: \(1 - p\)
- **Missing percentage**: \(n_{\text{missing}} / n_{\text{total}}\)
- **Entropy**: Shannon entropy in bits
- **Balance score**: measure of distribution symmetry
- **Imbalance ratio**: deviation from 50/50 split

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

### Imbalance Ratio

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

### Balance Score

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

### Information Content per True Value

Average information conveyed by each True observation:

\[
IC_{\text{true}} = -\log_2(p) \text{ bits}
\]

**Example:**
- \(p = 0.5\): \(IC = 1\) bit (unsurprising)
- \(p = 0.1\): \(IC = 3.32\) bits (rare event, informative)
- \(p = 0.01\): \(IC = 6.64\) bits (very rare, very informative)

**Use case:** In imbalanced classification, rare class has higher information content.

### Information Content per False Value

\[
IC_{\text{false}} = -\log_2(1 - p) \text{ bits}
\]

## Statistical Tests

### Binomial Test for Balance

Test if \(p = 0.5\) (balanced distribution).

**Null hypothesis:** \(H_0: p = 0.5\)

**Test statistic:**

\[
Z = \frac{\hat{p} - 0.5}{\sqrt{0.5 \cdot 0.5 / n}}
\]

Under \(H_0\) and large \(n\), \(Z \sim N(0, 1)\).

**P-value (two-tailed):**

\[
\text{p-value} = 2 \cdot \Phi(-|Z|)
\]

where \(\Phi\) is the standard normal CDF.

**Interpretation:**
- Small p-value (< 0.05): reject balance hypothesis (distribution is skewed)
- Large p-value: consistent with balanced distribution

!!! note "Not implemented in current version"
    Statistical tests are planned for future release.

## Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Count True/False** | \(O(n)\) | \(O(1)\) | Single pass |
| **Entropy** | \(O(1)\) | \(O(1)\) | From counts |
| **All metrics** | \(O(n)\) | \(O(1)\) | Single pass |

Boolean analysis is extremely efficient: O(1) space, O(n) time.

## Configuration

Control boolean analysis via `ReportConfig`:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Boolean-specific config
# (Currently no boolean-specific parameters)

report = profile(df, config=config)
```

## Implementation Details

### BooleanAccumulator Class

```python
class BooleanAccumulator:
    def __init__(self, name: str, config: BooleanConfig):
        self.name = name
        self.count = 0
        self.missing = 0
        self.true_count = 0
        self.false_count = 0
    
    def update(self, values: pd.Series):
        """Update with chunk of values"""
        # Filter out missing
        # Count True values
        # Count False values
        pass
    
    def finalize(self) -> BooleanSummary:
        """Compute final statistics"""
        # Compute percentages
        # Compute entropy
        # Compute balance scores
        # Detect imbalance
        return BooleanSummary(
            count=self.count,
            missing=self.missing,
            true_count=self.true_count,
            false_count=self.false_count,
            true_pct=self.true_count / max(1, self.count),
            entropy=self._compute_entropy(),
            balance_score=self._compute_balance(),
            imbalance_ratio=self._compute_imbalance(),
        )
    
    def _compute_entropy(self) -> float:
        if self.count == 0:
            return 0.0
        p = self.true_count / self.count
        if p == 0.0 or p == 1.0:
            return 0.0
        return -(p * math.log2(p) + (1-p) * math.log2(1-p))
    
    def _compute_balance(self) -> float:
        if self.count == 0:
            return 0.0
        p = self.true_count / self.count
        return 1.0 - abs(0.5 - p)
    
    def _compute_imbalance(self) -> float:
        if self.count == 0:
            return 0.0
        return abs(self.true_count - self.false_count) / self.count
```

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
# Highly imbalanced (10% True)
df = pd.DataFrame({
    "is_fraud": [False] * 90 + [True] * 10
})

report = profile(df)
# Will show low entropy, high imbalance
```

### Access Statistics

```python
from pysuricata import summarize

stats = summarize(df)
active_stats = stats["columns"]["is_active"]

print(f"True count: {active_stats['true_count']}")
print(f"True %: {active_stats['true_pct']:.1%}")
print(f"Entropy: {active_stats['entropy']:.2f} bits")
print(f"Balance: {active_stats['balance_score']:.2f}")
```

## Interpreting Results

### Well-Balanced (p ≈ 0.5)

- **Entropy** ≈ 1.0 bit
- **Balance score** > 0.9
- **Imbalance ratio** < 0.2

**Implications:**
- High information content
- Good for binary classification (no class imbalance)
- Unpredictable values

**Example:** Fair coin flip, A/B test with even split.

### Imbalanced (p << 0.5 or p >> 0.5)

- **Entropy** < 0.5 bits
- **Balance score** < 0.7
- **Imbalance ratio** > 0.6

**Implications:**
- Low information content
- May need rebalancing for ML
- Predictable values

**Example:** Fraud detection (1% positive), rare disease (0.1% positive).

### Nearly Constant (p < 0.01 or p > 0.99)

- **Entropy** < 0.1 bits
- **Balance score** ≈ 0.5
- **Imbalance ratio** > 0.98

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

- Entropy = 0 (no information)
- Balance score = 0.5 (worst)
- Imbalance ratio = 1.0 (complete)

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

---

*Last updated: 2025-10-12*




