---
title: Sketch Algorithms
description: Probabilistic data structures for streaming analytics - KMV, Misra-Gries, HyperLogLog
---

# Sketch Algorithms

**Sketch algorithms** (probabilistic data structures) enable approximate analytics on massive streams with bounded memory and mathematical error guarantees.

## Overview

Sketches trade perfect accuracy for:
- **Constant memory**: Independent of dataset size
- **Fast updates**: O(1) or O(log k) per element
- **Mergeability**: Combine sketches from parallel streams
- **Mathematical guarantees**: Provable error bounds

## K-Minimum Values (KMV)

### Purpose
Estimate distinct count (cardinality) of a set.

### Algorithm

1. Hash each element to [0,1]: \(h(x) \sim \text{Uniform}(0,1)\)
2. Keep the \(k\) smallest hash values
3. Estimate: \(\hat{d} = \frac{k-1}{x_k}\) where \(x_k\) is the \(k\)-th smallest hash

### Error Bound

\[
\text{Relative error} \approx \frac{1}{\sqrt{k}}
\]

**Example**: \(k=2048\) → ~2.2% error at 95% confidence

### Implementation

```python
import heapq, hashlib

class KMV:
    def __init__(self, k):
        self.k = k
        self.heap = []  # Max-heap of k smallest hashes
    
    def add(self, value):
        h = self._hash(value)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, -h)  # Negative for max-heap
        elif h < -self.heap[0]:
            heapq.heapreplace(self.heap, -h)
    
    def estimate(self):
        if len(self.heap) == 0:
            return 0
        if len(self.heap) < self.k:
            return len(self.heap)
        kth_min = -self.heap[0]
        return (self.k - 1) / kth_min if kth_min > 0 else float('inf')
    
    def _hash(self, value):
        return int(hashlib.md5(str(value).encode()).hexdigest(), 16) / (2**128)
```

### Mergeability

Union of two KMV sketches: merge heaps and keep k smallest.

```python
def merge(kmv1, kmv2):
    combined = KMV(kmv1.k)
    for h in kmv1.heap + kmv2.heap:
        combined.heap.append(h)
    combined.heap = heapq.nlargest(combined.k, combined.heap)
    heapq.heapify(combined.heap)
    return combined
```

## Misra-Gries Algorithm

### Purpose
Find top-k most frequent items (heavy hitters).

### Algorithm

1. Maintain dictionary of ≤k items with counts
2. For each new item:
   - If in dictionary: increment count
   - Else if dictionary not full: add with count 1
   - Else: decrement all counts, remove zeros
3. Output: items remaining in dictionary

### Guarantee

For any item with true frequency \(f > n/k\):
- Guaranteed to appear in output
- Estimated frequency within \(\pm n/k\) of true frequency

### Implementation

```python
class MisraGries:
    def __init__(self, k):
        self.k = k
        self.counts = {}
    
    def add(self, item):
        if item in self.counts:
            self.counts[item] += 1
        elif len(self.counts) < self.k:
            self.counts[item] = 1
        else:
            # Decrement all counts
            to_remove = []
            for key in self.counts:
                self.counts[key] -= 1
                if self.counts[key] == 0:
                    to_remove.append(key)
            for key in to_remove:
                del self.counts[key]
    
    def top_k(self):
        return sorted(self.counts.items(), key=lambda x: -x[1])
```

### Complexity

- Space: O(k)
- Update: O(k) worst-case, O(1) amortized
- Output: O(k log k) for sorting

### Mergeability

Sum counts from multiple Misra-Gries structures.

## HyperLogLog (HLL)

### Purpose
Estimate distinct count with very low memory.

### Key Idea
Count leading zeros in binary representation of hashes.

**Intuition**: If max leading zeros = \(m\), roughly \(2^m\) distinct elements seen.

### Algorithm

1. Hash elements to binary strings
2. Split hash into \(2^b\) buckets (first \(b\) bits)
3. Track max leading zeros per bucket
4. Combine with harmonic mean

### Estimator

\[
\hat{d} = \alpha_m \cdot m^2 \cdot \left(\sum_{j=1}^{m} 2^{-M_j}\right)^{-1}
\]

where:
- \(m = 2^b\) = number of buckets
- \(M_j\) = max leading zeros in bucket \(j\)
- \(\alpha_m\) = bias correction constant

### Error

\[
\text{Relative standard error} \approx \frac{1.04}{\sqrt{m}}
\]

**Example**: \(m=1024\) (10 KB) → ~3% error

### Properties

- **Space**: \(O(m \log \log n)\) bits
- **Mergeable**: Element-wise max of bucket values
- **Production-ready**: Used in Redis, BigQuery

!!! note "Not implemented in current version"
    PySuricata uses KMV instead of HLL. HLL may be added in future for even lower memory.

## Reservoir Sampling

### Purpose
Maintain uniform random sample of fixed size \(k\) from stream.

### Algorithm (Algorithm R)

```python
import random

class ReservoirSampler:
    def __init__(self, k):
        self.k = k
        self.reservoir = []
        self.n = 0
    
    def add(self, item):
        self.n += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.reservoir[j] = item
    
    def get_sample(self):
        return self.reservoir.copy()
```

### Guarantee

Every element has **exactly probability** \(k/n\) of being in the sample.

**Proof sketch**:
- Element \(i\) enters reservoir with probability \(\min(1, k/i)\)
- Survives subsequent updates with probability \(\prod_{j=i+1}^{n} (1 - 1/j)\)
- Total: \(k/n\)

### Complexity

- Space: O(k)
- Update: O(1)
- Uniform guarantee: Exact

### Use Cases

- Quantile estimation (sort sample)
- Histogram construction
- Outlier detection

## Bloom Filters

### Purpose
Test set membership with false positives, no false negatives.

### Algorithm

1. Initialize bit array of size \(m\) to 0
2. Use \(k\) hash functions
3. Add: set bits at \(h_1(x), h_2(x), \ldots, h_k(x)\) to 1
4. Query: check if all \(k\) bits are 1

### False Positive Rate

\[
P_{\text{fp}} \approx \left(1 - e^{-kn/m}\right)^k
\]

### Optimal Parameters

For desired \(P_{\text{fp}}\) and \(n\) elements:

\[
\begin{aligned}
m &= -\frac{n \ln P_{\text{fp}}}{(\ln 2)^2} \\
k &= \frac{m}{n} \ln 2
\end{aligned}
\]

**Example**: \(n=10^6\), \(P_{\text{fp}}=0.01\) → \(m \approx 9.6\) Mb, \(k=7\)

!!! note "Not implemented in current version"
    Bloom filters are not currently used in PySuricata but may be added for duplicate detection optimization.

## Count-Min Sketch

### Purpose
Estimate frequencies of items in stream.

### Algorithm

1. Initialize \(d \times w\) matrix of counters to 0
2. Use \(d\) hash functions mapping to \([0, w)\)
3. Add item: increment counters at \(M[i, h_i(x)]\) for \(i=1,\ldots,d\)
4. Estimate frequency: \(\hat{f}(x) = \min_i M[i, h_i(x)]\)

### Error Bound

With probability \(1-\delta\):

\[
\hat{f}(x) \le f(x) + \epsilon n
\]

where \(w = \lceil e / \epsilon \rceil\) and \(d = \lceil \ln(1/\delta) \rceil\).

### Comparison to Misra-Gries

- **Count-Min**: Point queries, any item
- **Misra-Gries**: Top-k queries, only frequent items

!!! note "Not implemented in current version"
    PySuricata uses Misra-Gries for top-k. Count-Min Sketch may be added for full frequency queries.

## Choosing Algorithms

| Need | Algorithm | Memory | Error |
|------|-----------|--------|-------|
| Distinct count | KMV | O(k) | \(1/\sqrt{k}\) |
| Distinct count (min memory) | HyperLogLog | O(m log log n) | \(1/\sqrt{m}\) |
| Top-k items | Misra-Gries | O(k) | \(n/k\) frequency |
| Top-k items (point queries) | Count-Min | O(dw) | \(\epsilon n\) |
| Uniform sample | Reservoir | O(k) | Exact |
| Membership test | Bloom filter | O(m) | \(P_{\text{fp}}\) |

## Implementation in PySuricata

### NumericAccumulator

```python
self._uniques = KMV(config.uniques_sketch_size)  # Distinct count
self._sample = ReservoirSampler(config.sample_size)  # Quantiles
```

### CategoricalAccumulator

```python
self._topk = MisraGries(config.top_k_size)  # Top values
self._uniques = KMV(config.uniques_sketch_size)  # Distinct count
```

## References

1. **Bar-Yossef, Z. et al. (2002)**, "Counting Distinct Elements in a Data Stream", *RANDOM*.

2. **Misra, J., Gries, D. (1982)**, "Finding repeated elements", *Science of Computer Programming*, 2(2): 143–152.

3. **Flajolet, P. et al. (2007)**, "HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm", *DMTCS*, AH: 137–156.

4. **Vitter, J.S. (1985)**, "Random Sampling with a Reservoir", *ACM TOMS*, 11(1): 37–57.

5. **Bloom, B.H. (1970)**, "Space/time trade-offs in hash coding with allowable errors", *CACM*, 13(7): 422–426.

6. **Cormode, G., Muthukrishnan, S. (2005)**, "An Improved Data Stream Summary: The Count-Min Sketch and its Applications", *Journal of Algorithms*, 55(1): 58–75.

## See Also

- [Streaming Algorithms](streaming.md) - Welford/Pébay moments
- [Numeric Analysis](../stats/numeric.md) - Using KMV and reservoir
- [Categorical Analysis](../stats/categorical.md) - Using Misra-Gries

---

*Last updated: 2025-10-12*




