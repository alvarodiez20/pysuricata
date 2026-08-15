---
title: Duplicate Detection
description: Row duplicate detection using hash-based algorithms
---

# Duplicate Detection

PySuricata estimates duplicate rows using memory-efficient hash-based algorithms.

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
