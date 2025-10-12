---
title: Duplicate Detection
description: Row duplicate detection using hash-based algorithms
---

# Duplicate Detection

PySuricata estimates duplicate rows using memory-efficient hash-based algorithms.

## Duplicate Rate

For dataset with \(n\) total rows and \(d\) distinct rows:

\[
DR = \frac{n - d}{n} = 1 - \frac{d}{n}
\]

## Detection Method

Uses KMV sketch on row hashes for approximate distinct count:

```python
# Conceptual algorithm
for row in dataset:
    row_hash = hash(tuple(row))
    kmv.add(row_hash)

n_distinct = kmv.estimate()
duplicate_rate = (n_total - n_distinct) / n_total
```

## Exact vs Approximate

**Exact** (for small datasets):
```python
exact_duplicates = df.duplicated().sum()
dup_pct = (exact_duplicates / len(df)) * 100
```

**Approximate** (PySuricata for large datasets):
- Uses KMV sketch
- ~2% error with default settings
- Constant memory

## See Also

- [Sketch Algorithms](../algorithms/sketches.md) - KMV details
- [Data Quality](quality.md) - Quality metrics

---

*Last updated: 2025-10-12*




