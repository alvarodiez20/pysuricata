---
title: Reservoir Sampling
description: Uniform random sampling from data streams
---

# Reservoir Sampling

Maintain uniform random sample of fixed size \(k\) from stream of unknown length.

## Algorithm R (Vitter)

```python
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
```

## Guarantee

Every element has **exactly** probability \(k/n\) of being in sample.

## Proof Sketch

Element \(i\) enters reservoir with probability \(\min(1, k/i)\).

It survives subsequent updates:

\[
P(\text{survive}) = \prod_{j=i+1}^{n} \left(1 - \frac{1}{j}\right) = \frac{i}{n}
\]

Total probability:

\[
P(\text{in sample}) = \frac{k}{i} \cdot \frac{i}{n} = \frac{k}{n}
\]

## Complexity

- **Space**: O(k)
- **Time per element**: O(1) amortized
- **Uniform guarantee**: Exact

## Use Cases

- Quantile estimation (sort sample)
- Histogram construction
- Representative sampling

## See Also

- [Sketch Algorithms](sketches.md) - Other streaming algorithms
- [Numeric Analysis](../stats/numeric.md) - Using reservoir for quantiles

---

*Last updated: 2025-10-12*




