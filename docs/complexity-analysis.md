# Complexity Analysis

## Overview

PySuricata processes data in a **single pass** using streaming algorithms. All statistics are computed incrementally from small, fixed-size state — memory usage depends on configuration parameters, not dataset size.

**Key property:** For a dataset with *n* rows and *p* columns, PySuricata uses O(p · k) memory where *k* is the sketch/sample size (constant), making total memory independent of *n*.

### Summary

| Component | Time per value | Space | Dominant parameter |
|-----------|---------------|-------|--------------------|
| Streaming moments | O(1) | O(1) | — |
| Reservoir sampling | O(1) | O(s) | `numeric_sample_size` (default: 20,000) |
| KMV distinct count | O(log k) | O(k) | `uniques_sketch_size` (default: 2,048) |
| Misra-Gries top-k | O(1) amortized | O(k) | `top_k_size` (default: 50) |
| Extreme tracking | O(log k) | O(k) | `max_extremes` (default: 5) |
| Correlations | O(p) per value | O(p²) | Number of numeric columns |

**Total per numeric column:** O(s + k) ≈ 22 KB with defaults

**Total per categorical column:** O(k) ≈ 18 KB with defaults

**DateTime and Boolean columns:** O(1)

---

## Algorithms

### Streaming Moments (Welford/Pébay)

**Purpose:** Exact mean, variance, skewness, kurtosis in a single pass.

- **Update:** O(1) per value — maintains running sums of powers of deviations
- **Merge:** O(1) — Pébay's formulas combine two partial states exactly
- **Finalize:** O(1)
- **Space:** O(1) — six floating-point accumulators (n, μ, M₂, M₃, M₄)

This is numerically stable — it avoids the catastrophic cancellation of naïve Σx² − nμ² formulas.

**Reference:** Welford (1962), Pébay (2008)

### Reservoir Sampling

**Purpose:** Maintain a uniform random sample for quantile estimation and histograms.

- **Update:** O(1) per value — include with probability s/i, replacing a random element
- **Merge:** O(s) — weighted combination of two reservoirs
- **Finalize:** O(s log s) — sort for quantile computation
- **Space:** O(s) where s = `numeric_sample_size`

Every subset of size s from the stream has equal probability of being the sample.

**Reference:** Vitter (1985)

### KMV (K-Minimum Values) Sketch

**Purpose:** Approximate distinct count estimation.

- **Update:** O(log k) — hash value, maintain min-heap of k smallest hashes
- **Merge:** O(k log k) — union two heaps, keep k smallest
- **Estimate:** d̂ = (k − 1) / x_k where x_k is the k-th smallest hash
- **Space:** O(k) where k = `uniques_sketch_size`
- **Error:** ~1/√k relative error — with k=2048, approximately 2.2%

The implementation transitions from exact counting (for low-cardinality columns) to approximate estimation when the number of distinct values exceeds a threshold.

**Reference:** Bar-Yossef et al. (2002)

### Misra-Gries (Top-K Frequent Values)

**Purpose:** Find the most frequent values in a stream.

- **Update:** O(1) amortized — increment counter or decrement all
- **Merge:** O(k) — sum counters from two states
- **Finalize:** O(k log k) — sort by frequency
- **Space:** O(k) where k = `top_k_size`
- **Guarantee:** Any value with true frequency > n/k is included in the output

Frequency estimates are within ±n/k of the true count.

**Reference:** Misra & Gries (1982)

### Extreme Tracking

**Purpose:** Track the minimum and maximum values with their row indices.

- **Update:** O(log k) per value — bounded heap operations
- **Merge:** O(k log k) — merge and truncate heaps
- **Space:** O(k) where k = `max_extremes` (default: 5)

Uses a min-heap for minimums and a negated max-heap for maximums.

### Correlations

**Purpose:** Streaming Pearson correlation between all pairs of numeric columns.

- **Update:** O(p) per value — update co-moment for each column pair
- **Finalize:** O(p²) — compute correlations from co-moments
- **Space:** O(p²) — stores pairwise co-moments

This is the most expensive component. For datasets with many numeric columns, disable with `compute_correlations = False` or increase `corr_threshold`.

---

## Accumulator Breakdown

### NumericAccumulator

| Component | Space |
|-----------|-------|
| StreamingMoments | O(1) |
| ReservoirSampler | O(s) |
| KMV sketch | O(k) |
| ExtremeTracker | O(k) |
| MisraGries | O(k) |
| Histogram bins | O(b) |
| **Total** | **O(s + k + b)** |

With defaults: s=20,000, k=2,048, b=25

### CategoricalAccumulator

| Component | Space |
|-----------|-------|
| KMV sketch (original) | O(k) |
| KMV sketch (lowercase) | O(k) |
| KMV sketch (trimmed) | O(k) |
| MisraGries | O(k) |
| String length tracking | O(1) |
| **Total** | **O(k)** |

### DatetimeAccumulator

Fixed-size counters: hour (24), weekday (7), month (12), plus min/max tracking.

**Total:** O(1)

### BooleanAccumulator

Two counters (true, false) plus missing count.

**Total:** O(1)

---

## Configuration Impact

| Parameter | Effect on memory | Effect on accuracy |
|-----------|-----------------|-------------------|
| `chunk_size` ↑ | More rows in memory per iteration | No effect on final accuracy |
| `numeric_sample_size` ↑ | Larger reservoir per numeric column | Better quantile estimates |
| `uniques_sketch_size` ↑ | Larger KMV sketch per column | Better distinct count accuracy |
| `top_k_size` ↑ | More frequent values tracked | More comprehensive top-k |
| `compute_correlations` off | Saves O(p²) | No correlations reported |

---

## Performance Optimizations (v0.0.14)

Several hot paths were vectorized to reduce per-value Python overhead:

| Optimization | Change |
|---|---|
| KMV/MisraGries updates | Per-value `add()` → batch `add_many()` with pre-counted values |
| Missing/Inf counting | Python loop → vectorized `np.isnan()` / `np.isinf()` |
| Row hashing (duplicates) | Per-row tuple + `hash()` → vectorized polynomial hash via numpy |
| Boolean coercion | Per-value `str().lower()` → pandas vectorized `str.strip().str.lower()` |
| Correlation finite masks | Recomputed per pair → pre-computed once per column, reused |

## References

1. Welford, B.P. (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", *Technometrics*, 4(3): 419–420
2. Pébay, P. (2008), "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments", Sandia Report SAND2008-6212
3. Bar-Yossef, Z. et al. (2002), "Counting Distinct Elements in a Data Stream", *RANDOM*
4. Misra, J. & Gries, D. (1982), "Finding repeated elements", *Science of Computer Programming*, 2(2): 143–152
5. Vitter, J.S. (1985), "Random Sampling with a Reservoir", *ACM TOMS*, 11(1): 37–57
