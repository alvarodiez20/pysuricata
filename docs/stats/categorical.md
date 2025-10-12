---
title: Categorical Variable Analysis
description: Complete mathematical formulas and algorithms for categorical data analysis in pysuricata
---

# Categorical Variable Analysis

This page provides comprehensive documentation for how PySuricata analyzes categorical (string, object) variables using scalable streaming algorithms with mathematical guarantees.

## Overview

PySuricata treats a **categorical variable** as any column with string-like values, objects, or low-cardinality integers. Analysis focuses on frequency distributions, diversity metrics, and string characteristics.

### Key Features

- **Top-k values** with frequencies (Misra-Gries algorithm)
- **Distinct count** estimation (KMV sketch)
- **Diversity metrics** (entropy, Gini impurity, concentration)
- **String statistics** (length distribution, empty strings)
- **Variant detection** (case-insensitive, trimmed)
- **Memory-efficient** streaming algorithms

## Summary Statistics Provided

For each categorical column:

- **Count**: non-null values, missing percentage
- **Distinct**: unique value count (exact or approximate)
- **Top values**: most frequent values with counts and percentages
- **Entropy**: Shannon entropy (information content)
- **Gini impurity**: concentration measure
- **Diversity ratio**: uniqueness measure
- **Most common ratio**: dominance of top value
- **String length**: mean, p90, distribution
- **Special values**: empty strings, whitespace-only
- **Variants**: case-insensitive and trimmed unique counts

## Mathematical Definitions

### Frequency and Probability

Let \(x_1, x_2, \ldots, x_n\) be the non-missing categorical values.

**Frequency of value \(v\):**

\[
f(v) = |\{i : x_i = v\}|
\]

**Relative frequency (empirical probability):**

\[
p(v) = \frac{f(v)}{n}
\]

**Distinct count:**

\[
d = |\{v : f(v) > 0\}|
\]

### Shannon Entropy

Measures the **information content** or **uncertainty** in the distribution:

\[
H(X) = -\sum_{v} p(v) \log_2 p(v)
\]

where the sum is over all distinct values \(v\) with \(p(v) > 0\).

**Properties:**
- \(H(X) = 0\) if one value has \(p=1\) (no uncertainty)
- \(H(X) = \log_2 d\) if all values equally likely (maximum entropy)
- Units: **bits** of information

**Interpretation:**
- Low entropy (< 1 bit): highly concentrated, predictable
- Medium entropy (1-3 bits): moderate diversity
- High entropy (> 3 bits): high diversity, hard to predict

**Example:**
- Uniform distribution over 8 values: \(H = \log_2 8 = 3\) bits
- One value 90%, others 10%/9: \(H \approx 0.57\) bits

### Gini Impurity

Measures the **probability of misclassification** if labels were assigned randomly according to the distribution:

\[
\text{Gini}(X) = 1 - \sum_{v} p(v)^2
\]

**Properties:**
- \(\text{Gini}(X) = 0\) if one value (no impurity)
- \(\text{Gini}(X) = 1 - 1/d\) if uniform over \(d\) values
- Range: \([0, 1)\)

**Interpretation:**
- Low Gini (< 0.2): concentrated distribution
- Medium Gini (0.2-0.6): moderate spread
- High Gini (> 0.6): high diversity

**Use in ML:** Decision trees use Gini impurity for splitting criteria.

### Diversity Ratio

Simple measure of uniqueness:

\[
D = \frac{d}{n}
\]

where \(d\) = distinct count, \(n\) = total count.

**Interpretation:**
- \(D \to 0\): low diversity (many repeats)
- \(D \to 1\): high diversity (mostly unique)

**Special cases:**
- \(D = 1\): all values unique (e.g., primary keys)
- \(D = 1/n\): all values identical

### Concentration Ratio

Fraction of observations in the top \(k\) values:

\[
CR_k = \frac{\sum_{i=1}^{k} f(v_i)}{n}
\]

where \(v_1, v_2, \ldots\) are values sorted by frequency (descending).

**Example:** \(CR_5 = 0.80\) means top 5 values account for 80% of data.

**Interpretation:**
- High \(CR_k\): distribution dominated by few values
- Low \(CR_k\): distribution spread across many values

### Most Common Ratio

Dominance of the single most frequent value:

\[
MCR = \frac{f(v_{\max})}{n} = p(v_{\max})
\]

**Interpretation:**
- MCR > 0.9: highly dominant category (nearly constant)
- MCR < 0.1: no dominant category (high diversity)

## Streaming Algorithms

### Misra-Gries Algorithm for Top-K

Finds the \(k\) most frequent items in a stream using **O(k) space** with **frequency guarantees**.

**Algorithm:**

1. **Initialize:** empty dictionary \(M\) (max size \(k\))
2. **For each value \(v\):**
   - If \(v \in M\): increment \(M[v]\)
   - Else if \(|M| < k\): add \(M[v] = 1\)
   - Else: decrement all counts in \(M\); remove zeros
3. **Output:** items in \(M\) with estimated counts

**Guarantee:** For any value \(v\) with true frequency \(f(v)\):
- If \(f(v) > n/k\), then \(v\) is in output
- Estimated frequency within \(n/k\) of true frequency

**Space complexity:** \(O(k)\)  
**Update complexity:** \(O(k)\) worst-case, \(O(1)\) amortized

**Mergeable:** Yes (sum counters from multiple streams)

**Example:** \(k=50\), \(n=1,000,000\)
- Guaranteed to find all items with frequency > 20,000
- Frequency estimates within ±20,000

**Reference:** Misra, J., Gries, D. (1982), "Finding repeated elements", *Science of Computer Programming*, 2(2): 143–152.

### KMV Sketch for Distinct Count

Estimates cardinality using \(k\) minimum hash values.

**Algorithm:**

1. **Initialize:** empty set \(S\) (max size \(k\))
2. **For each value \(v\):**
   - Compute hash \(h(v) \in [0,1]\)
   - If \(|S| < k\) or \(h(v) < \max(S)\):
     - Add \(h(v)\) to \(S\)
     - If \(|S| > k\): remove \(\max(S)\)
3. **Estimate:**

\[
\hat{d} = \frac{k-1}{x_k}
\]

where \(x_k = \max(S)\) is the \(k\)-th smallest hash.

**Error bound:**

\[
\text{Relative error} \approx \frac{1}{\sqrt{k}}
\]

**Space:** \(O(k)\)  
**Update:** \(O(\log k)\) (heap operations)  
**Mergeable:** Yes (union of sets)

**Example:** \(k=2048\)
- Error: ~2.2% (95% confidence)
- Space: ~16 KB (assuming 64-bit hashes)

**Reference:** Bar-Yossef, Z. et al. (2002), "Counting Distinct Elements in a Data Stream", *RANDOM*.

### Space-Saving Algorithm (Alternative)

Maintains top-k with guaranteed error bounds:

**Error bound:** Estimated frequency within \(\epsilon n\) where \(\epsilon = 1/k\)

**Advantage over Misra-Gries:** Tighter worst-case bounds, better for skewed distributions.

**Reference:** Metwally, A., Agrawal, D., El Abbadi, A. (2005), "Efficient Computation of Frequent and Top-k Elements in Data Streams", *ICDT*.

## String Analysis

### Length Statistics

For string values, track:

**Mean length:**

\[
\bar{L} = \frac{1}{n} \sum_{i=1}^{n} |x_i|
\]

where \(|x_i|\) is the character count of string \(x_i\).

**P90 length:** 90th percentile of lengths (via reservoir sampling)

**Length distribution:** Histogram of string lengths

**Use cases:**
- Detect outliers (abnormally long strings)
- Validate constraints (max length)
- Estimate storage requirements

### Empty Strings

Count of strings that are:
- Empty: `""`
- Whitespace-only: match `/^\s*$/`
- NULL vs. empty distinction

**Formula:**

\[
n_{\text{empty}} = |\{i : x_i = "" \text{ or } x_i \text{ matches } /^\s*$/\}|
\]

### Case Variants

Estimate distinct count after case normalization:

\[
d_{\text{lower}} = |\{v.\text{lower}() : v \in \text{values}\}|
\]

**Interpretation:**
- \(d_{\text{lower}} < d\): case variants present (e.g., "USA", "usa")
- \(d_{\text{lower}} = d\): no case variants

### Trim Variants

Estimate distinct count after removing leading/trailing whitespace:

\[
d_{\text{trim}} = |\{v.\text{strip}() : v \in \text{values}\}|
\]

**Interpretation:**
- \(d_{\text{trim}} < d\): whitespace variants present
- \(d_{\text{trim}} = d\): no trim variants

## Chi-Square Uniformity Test

Test if the distribution is uniform (all categories equally likely).

**Null hypothesis:** \(p(v_1) = p(v_2) = \cdots = p(v_d) = 1/d\)

**Test statistic:**

\[
\chi^2 = \sum_{i=1}^{d} \frac{(f(v_i) - E)^2}{E}
\]

where \(E = n/d\) is the expected frequency under uniformity.

**Distribution under \(H_0\):** \(\chi^2_{d-1}\) (chi-square with \(d-1\) degrees of freedom)

**P-value:** \(P(\chi^2_{d-1} > \chi^2_{\text{obs}})\)

**Interpretation:**
- Small p-value (< 0.05): reject uniformity (distribution is skewed)
- Large p-value: consistent with uniform distribution

!!! note "Not implemented in current version"
    Chi-square test is planned for future release.

## Cardinality Categories

Classify categorical variables by distinct count:

| Category | Distinct Count | Examples |
|----------|----------------|----------|
| **Boolean-like** | 2-3 | Yes/No, True/False/Unknown |
| **Low cardinality** | 4-20 | Status codes, categories |
| **Medium cardinality** | 21-100 | US states, countries |
| **High cardinality** | 101-10,000 | Zip codes, product IDs |
| **Very high cardinality** | > 10,000 | User IDs, URLs, emails |

**Recommended actions:**
- **Low cardinality:** Show all values in report
- **High cardinality:** Show top-k only, estimate distinct
- **Very high cardinality:** Consider as identifier (unique key)

## Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Misra-Gries** | \(O(nk)\) worst, \(O(n)\) amortized | \(O(k)\) | Top-k values |
| **KMV distinct** | \(O(n \log k)\) | \(O(k)\) | Distinct count |
| **Entropy** | \(O(n + d)\) | \(O(d)\) | From frequency table |
| **String lengths** | \(O(n)\) | \(O(k)\) | Reservoir sample |
| **Exact distinct** | \(O(n)\) | \(O(d)\) | Hash set |

## Configuration

Control categorical analysis via `ReportConfig`:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Top-k size (Misra-Gries)
config.compute.top_k_size = 50  # Default

# Distinct count sketch size (KMV)
config.compute.uniques_sketch_size = 2_048  # Default

# String length sample size
# (Not separately configurable, uses numeric_sample_size)

# Enable/disable case variants
# (Always enabled, no toggle)

# Enable/disable trim variants
# (Always enabled, no toggle)

report = profile(df, config=config)
```

## Implementation Details

### CategoricalAccumulator Class

```python
class CategoricalAccumulator:
    def __init__(self, name: str, config: CategoricalConfig):
        self.name = name
        self.count = 0
        self.missing = 0
        
        # Top-k values
        self._topk = MisraGries(config.top_k_size)
        
        # Distinct count
        self._uniques = KMV(config.uniques_sketch_size)
        self._uniques_lower = KMV(config.uniques_sketch_size)  # Case-insensitive
        self._uniques_strip = KMV(config.uniques_sketch_size)  # Trimmed
        
        # String lengths
        self._len_sum = 0
        self._len_n = 0
        self._len_sample = ReservoirSampler(config.length_sample_size)
        
        # Special values
        self._empty_count = 0
    
    def update(self, values: pd.Series):
        """Update with chunk of values"""
        # Filter out missing
        # Update top-k
        # Update distinct sketches (original, lower, strip)
        # Track string lengths
        # Count empty strings
        pass
    
    def finalize(self) -> CategoricalSummary:
        """Compute final statistics"""
        # Get top values from Misra-Gries
        # Estimate distinct from KMV
        # Compute entropy and Gini
        # Compute string length stats
        return CategoricalSummary(...)
```

## Validation

PySuricata validates categorical algorithms:

- **Exact vs approximate**: Compare KMV estimate to exact count (small datasets)
- **Top-k correctness**: Verify all items with \(f > n/k\) are found
- **Entropy bounds**: Check \(0 \le H(X) \le \log_2 d\)
- **Gini bounds**: Check \(0 \le \text{Gini}(X) < 1\)
- **Mergeability**: Verify merge = concatenate (for Misra-Gries, KMV)

## Examples

### Basic Usage

```python
import pandas as pd
from pysuricata import profile

df = pd.DataFrame({
    "country": ["USA", "UK", "USA", "DE", None, "USA", "FR"]
})

report = profile(df)
report.save_html("report.html")
```

### High-Cardinality Column

```python
# Column with 10,000 unique values
df = pd.DataFrame({
    "user_id": [f"user_{i}" for i in range(100_000)]
})

config = ReportConfig()
config.compute.top_k_size = 100  # Show top 100
config.compute.uniques_sketch_size = 4_096  # More accurate distinct

report = profile(df, config=config)
```

### Access Statistics

```python
from pysuricata import summarize

stats = summarize(df)
country_stats = stats["columns"]["country"]

print(f"Distinct: {country_stats['distinct']}")
print(f"Top value: {country_stats['top_values'][0]}")
print(f"Entropy: {country_stats['entropy']:.2f} bits")
print(f"Gini: {country_stats['gini']:.3f}")
```

## Interpreting Results

### High Entropy

\(H(X) > 5\) bits suggests:
- Many distinct values (> 32)
- Fairly uniform distribution
- High information content
- Possibly high cardinality (consider as identifier)

### Low Entropy

\(H(X) < 1\) bit suggests:
- Few distinct values (< 4 effective)
- Skewed distribution (one value dominates)
- Low information content
- Consider as low-cardinality categorical

### High Gini

\(\text{Gini} > 0.7\) suggests:
- Values well-distributed
- No single dominant category
- Good for stratification

### Low Gini

\(\text{Gini} < 0.2\) suggests:
- One or few values dominate
- Imbalanced distribution
- Consider as nearly constant column

## Special Cases

### All Unique (Primary Key)

- Distinct count \(d = n\)
- Entropy \(H = \log_2 n\) (maximum)
- Diversity ratio \(D = 1.0\)
- Top-k meaningless (all have count 1)

**Recommendation:** Flag as identifier, exclude from analysis.

### Nearly Constant

- Distinct count \(d = 2\) with \(p_1 > 0.99\)
- Entropy \(H < 0.1\) bits
- Gini \(< 0.02\)

**Recommendation:** Consider removing (low variance).

### Many Empty Strings

- Empty count > 10% of non-null

**Possible data quality issue:** Missing values encoded as empty strings.

## References

1. **Misra, J., Gries, D. (1982)**, "Finding repeated elements", *Science of Computer Programming*, 2(2): 143–152.

2. **Bar-Yossef, Z. et al. (2002)**, "Counting Distinct Elements in a Data Stream", *RANDOM*.

3. **Metwally, A., Agrawal, D., El Abbadi, A. (2005)**, "Efficient Computation of Frequent and Top-k Elements in Data Streams", *ICDT*.

4. **Shannon, C.E. (1948)**, "A Mathematical Theory of Communication", *Bell System Technical Journal*, 27: 379–423.

5. **Breiman, L. et al. (1984)**, *Classification and Regression Trees*, Wadsworth.

6. **Wikipedia: Entropy (information theory)** - [Link](https://en.wikipedia.org/wiki/Entropy_(information_theory))

7. **Wikipedia: Decision tree learning** - [Link](https://en.wikipedia.org/wiki/Decision_tree_learning)

## See Also

- [Numeric Analysis](numeric.md) - Numeric variables
- [Sketch Algorithms](../algorithms/sketches.md) - KMV, Misra-Gries deep dive
- [Data Quality](../analytics/quality.md) - Quality metrics
- [Configuration Guide](../configuration.md) - All parameters

---

*Last updated: 2025-10-12*




