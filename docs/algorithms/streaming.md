---
title: Streaming Statistics Algorithms  
description: Complete derivations of Welford's and Pébay's algorithms for online computation of statistical moments
---

# Streaming Statistics Algorithms

Deep dive into the **streaming algorithms** that power PySuricata's memory-efficient statistics computation.

## Overview

Streaming algorithms compute statistics in **single-pass**, **constant-memory** mode, enabling analysis of datasets larger than RAM.

### Key Algorithms

- **Welford's algorithm**: Online mean and variance
- **Pébay's formulas**: Parallel merging of moments
- **Higher moments**: Skewness and kurtosis extension
- **Numerical stability**: Avoiding catastrophic cancellation

## Welford's Online Algorithm

### The Problem

**Naive variance** formula:

\[
s^2 = \frac{1}{n-1}\left(\sum x_i^2 - \frac{(\sum x_i)^2}{n}\right)
\]

**Issues:**
- Requires two passes (one for \(\sum x_i\), one for \(\sum x_i^2\))
- Catastrophic cancellation if \(\sum x_i^2 \approx (\sum x_i)^2 / n\)
- Poor numerical stability

### Welford's Solution

**State variables:**
- \(n\): count
- \(\mu\): running mean
- \(M_2\): sum of squared deviations from current mean

**Update formulas:**

\[
\begin{aligned}
n &\leftarrow n + 1 \\
\delta &= x - \mu \\
\mu &\leftarrow \mu + \frac{\delta}{n} \\
\delta_2 &= x - \mu_{\text{new}} \\
M_2 &\leftarrow M_2 + \delta \cdot \delta_2
\end{aligned}
\]

**Finalize:**

\[
\text{variance} = \frac{M_2}{n-1}
\]

### Derivation

Starting from the definition:

\[
M_2^{(n)} = \sum_{i=1}^{n} (x_i - \mu^{(n)})^2
\]

After adding \(x_{n+1}\):

\[
M_2^{(n+1)} = \sum_{i=1}^{n+1} (x_i - \mu^{(n+1)})^2
\]

Expand using the mean update:

\[
\mu^{(n+1)} = \mu^{(n)} + \frac{x_{n+1} - \mu^{(n)}}{n+1} = \mu^{(n)} + \frac{\delta}{n+1}
\]

After algebraic manipulation (see Welford 1962):

\[
M_2^{(n+1)} = M_2^{(n)} + \delta \cdot (x_{n+1} - \mu^{(n+1)})
\]

**Key insight:** Update uses both old and new mean, providing numerical stability.

### Pseudocode

```python
def welford_update(n, mean, M2, x):
    """Update running moments with new value x"""
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    delta2 = x - mean_new
    M2_new = M2 + delta * delta2
    return n_new, mean_new, M2_new

def welford_finalize(n, mean, M2):
    """Compute final statistics"""
    if n < 2:
        return mean, None
    variance = M2 / (n - 1)
    return mean, variance
```

### Properties

1. **Single-pass**: Only one scan through data
2. **Constant memory**: O(1) space (3 numbers)
3. **Numerically stable**: No catastrophic cancellation
4. **Exact**: Same result as two-pass (up to FP rounding)
5. **Online**: Can process streaming data

## Pébay's Parallel Merge

### The Problem

How to **combine** partial results from multiple chunks/threads?

Given:
- State A: \((n_a, \mu_a, M_{2a})\)
- State B: \((n_b, \mu_b, M_{2b})\)

Want: Combined state \((n, \mu, M_2)\) equivalent to processing all data together.

### Pébay's Solution

**Combined state:**

\[
\begin{aligned}
n &= n_a + n_b \\
\delta &= \mu_b - \mu_a \\
\mu &= \mu_a + \delta \cdot \frac{n_b}{n} \\
M_2 &= M_{2a} + M_{2b} + \delta^2 \cdot \frac{n_a n_b}{n}
\end{aligned}
\]

### Derivation

The combined mean is the weighted average:

\[
\mu = \frac{n_a \mu_a + n_b \mu_b}{n_a + n_b} = \mu_a + \delta \cdot \frac{n_b}{n}
\]

For \(M_2\), use the identity:

\[
M_2 = \sum (x_i - \mu)^2 = \sum (x_i - \mu_a)^2 - n(\mu - \mu_a)^2
\]

Applying to both groups and summing:

\[
\begin{aligned}
M_2 &= M_{2a} + n_a(\mu_a - \mu)^2 + M_{2b} + n_b(\mu_b - \mu)^2 \\
&= M_{2a} + M_{2b} + n_a\left(-\delta \frac{n_b}{n}\right)^2 + n_b\left(\delta \frac{n_a}{n}\right)^2 \\
&= M_{2a} + M_{2b} + \delta^2 \frac{n_a n_b (n_b + n_a)}{n^2} \\
&= M_{2a} + M_{2b} + \delta^2 \frac{n_a n_b}{n}
\end{aligned}
\]

### Pseudocode

```python
def pebay_merge(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    """Merge two partial states"""
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    M2 = M2_a + M2_b + delta**2 * n_a * n_b / n
    
    return n, mean, M2
```

### Properties

1. **Associative**: Order of merging doesn't matter
2. **Commutative**: A ∪ B = B ∪ A
3. **Exact**: Same result as single-pass over concatenated data
4. **Parallel**: Enables multi-threading, distributed computation
5. **Fast**: O(1) time to merge two states

## Higher Moments Extension

### Third and Fourth Moments

**State variables:**
- \(n\), \(\mu\), \(M_2\), \(M_3\), \(M_4\)

Where:
- \(M_3 = \sum (x_i - \mu)^3\)
- \(M_4 = \sum (x_i - \mu)^4\)

### Online Update

```python
def moments_update(n, mean, M2, M3, M4, x):
    """Update all four moments"""
    n_new = n + 1
    delta = x - mean
    delta_n = delta / n_new
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * n
    
    mean_new = mean + delta_n
    M4_new = M4 + term1 * delta_n2 * (n_new*n_new - 3*n_new + 3) + 6*delta_n2*M2 - 4*delta_n*M3
    M3_new = M3 + term1 * delta_n * (n_new - 2) - 3*delta_n*M2
    M2_new = M2 + term1
    
    return n_new, mean_new, M2_new, M3_new, M4_new
```

### Pébay Merge for Higher Moments

```python
def pebay_merge_moments(n_a, mean_a, M2_a, M3_a, M4_a,
                        n_b, mean_b, M2_b, M3_b, M4_b):
    """Merge higher moments"""
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    
    delta = mean_b - mean_a
    delta2 = delta * delta
    delta3 = delta2 * delta
    delta4 = delta3 * delta
    
    mean = mean_a + delta * n_b / n
    
    M2 = M2_a + M2_b + delta2 * n_a * n_b / n
    
    M3 = M3_a + M3_b + \
         delta3 * n_a * n_b * (n_a - n_b) / (n * n) + \
         3 * delta * (n_a * M2_b - n_b * M2_a) / n
    
    M4 = M4_a + M4_b + \
         delta4 * n_a * n_b * (n_a*n_a - n_a*n_b + n_b*n_b) / (n * n * n) + \
         6 * delta2 * (n_a*n_a * M2_b + n_b*n_b * M2_a) / (n * n) + \
         4 * delta * (n_a * M3_b - n_b * M3_a) / n
    
    return n, mean, M2, M3, M4
```

### Computing Skewness and Kurtosis

```python
def compute_shape(n, M2, M3, M4):
    """Compute skewness and excess kurtosis"""
    if n < 3:
        return None, None
    
    variance = M2 / (n - 1)
    if variance == 0:
        return None, None
    
    # Skewness (g1)
    g1 = (n / ((n-1) * (n-2))) * (M3 / n) / (variance ** 1.5)
    
    if n < 4:
        return g1, None
    
    # Excess kurtosis (g2)
    g2 = ((n * (n+1)) / ((n-1) * (n-2) * (n-3))) * (M4 / n) / (variance ** 2) - \
         (3 * (n-1) ** 2) / ((n-2) * (n-3))
    
    return g1, g2
```

## Numerical Stability Analysis

### Why Naive Formula Fails

Consider \(x_i \approx 10^9 + \epsilon_i\) where \(|\epsilon_i| \ll 10^9\).

**Naive formula:**

\[
\sum x_i^2 \approx n \cdot 10^{18}, \quad \left(\sum x_i\right)^2 / n \approx n \cdot 10^{18}
\]

Subtraction loses precision (catastrophic cancellation).

**Welford's formula:**

Works with deviations \(x_i - \mu\), which are \(O(\epsilon)\), avoiding large intermediate values.

### Condition Number

For variance computation, Welford's algorithm has **condition number** \(\kappa \approx 1\), while naive formula has \(\kappa \approx \frac{\bar{x}^2}{\sigma^2}\) (can be huge).

## Parallelization

### MapReduce Pattern

```python
# Map phase: compute partial moments per chunk
def map_chunk(chunk):
    n, mean, M2, M3, M4 = 0, 0.0, 0.0, 0.0, 0.0
    for x in chunk:
        n, mean, M2, M3, M4 = moments_update(n, mean, M2, M3, M4, x)
    return n, mean, M2, M3, M4

# Reduce phase: merge all partial results
def reduce_moments(states):
    result = (0, 0.0, 0.0, 0.0, 0.0)
    for state in states:
        result = pebay_merge_moments(*result, *state)
    return result

# Usage
partial_states = [map_chunk(chunk) for chunk in chunks]
final_state = reduce_moments(partial_states)
```

### Multi-threading

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_moments(data, n_threads=4):
    chunks = np.array_split(data, n_threads)
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        states = list(executor.map(map_chunk, chunks))
    
    return reduce_moments(states)
```

## Implementation in PySuricata

### StreamingMoments Class

```python
class StreamingMoments:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0
    
    def update(self, values: np.ndarray):
        """Update with array of values"""
        for x in values:
            if not np.isfinite(x):
                continue
            self.n, self.mean, self.M2, self.M3, self.M4 = \
                moments_update(self.n, self.mean, self.M2, self.M3, self.M4, x)
    
    def merge(self, other: 'StreamingMoments'):
        """Merge with another moments object"""
        self.n, self.mean, self.M2, self.M3, self.M4 = \
            pebay_merge_moments(
                self.n, self.mean, self.M2, self.M3, self.M4,
                other.n, other.mean, other.M2, other.M3, other.M4
            )
    
    def finalize(self):
        """Compute final statistics"""
        if self.n < 2:
            return {"mean": self.mean, "variance": None, "skewness": None, "kurtosis": None}
        
        variance = self.M2 / (self.n - 1)
        std = math.sqrt(variance)
        skewness, kurtosis = compute_shape(self.n, self.M2, self.M3, self.M4)
        
        return {
            "count": self.n,
            "mean": self.mean,
            "variance": variance,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
```

## Validation

### Test Properties

```python
def test_welford_equivalence():
    """Verify Welford = two-pass"""
    data = np.random.randn(10000)
    
    # Welford
    n, mean, M2 = 0, 0.0, 0.0
    for x in data:
        n, mean, M2 = welford_update(n, mean, M2, x)
    var_welford = M2 / (n - 1)
    
    # Two-pass
    mean_twopass = np.mean(data)
    var_twopass = np.var(data, ddof=1)
    
    assert np.isclose(mean, mean_twopass)
    assert np.isclose(var_welford, var_twopass)

def test_pebay_merge():
    """Verify merge = concatenate"""
    data_a = np.random.randn(5000)
    data_b = np.random.randn(3000)
    
    # Separate
    state_a = compute_moments(data_a)
    state_b = compute_moments(data_b)
    merged = pebay_merge_moments(*state_a, *state_b)
    
    # Combined
    data_combined = np.concatenate([data_a, data_b])
    combined = compute_moments(data_combined)
    
    assert np.allclose(merged, combined)
```

## References

1. **Welford, B.P. (1962)**, "Note on a Method for Calculating Corrected Sums of Squares and Products", *Technometrics*, 4(3): 419–420.

2. **Pébay, P. (2008)**, "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments", Sandia Report SAND2008-6212.

3. **Chan, T.F., Golub, G.H., LeVeque, R.J. (1983)**, "Algorithms for Computing the Sample Variance: Analysis and Recommendations", *The American Statistician*, 37(3): 242–247.

4. **West, D.H.D. (1979)**, "Updating Mean and Variance Estimates: An Improved Method", *Communications of the ACM*, 22(9): 532–535.

5. **Wikipedia: Algorithms for calculating variance** - [Link](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)

## See Also

- [Numeric Analysis](../stats/numeric.md) - Application of these algorithms
- [Sketch Algorithms](sketches.md) - Other streaming algorithms
- [Performance Tips](../performance.md) - Optimization strategies

---

*Last updated: 2025-10-12*




