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

## Reading the Number

The duplicate count is `rows − distinct`. `rows` is exact and `distinct` is a
sketch estimate, so **the whole absolute error of the distinct estimate lands on
the duplicate count** — a quantity that is usually far smaller. On 200,000 rows
with 2,000 genuine duplicates, a 1% error on the distinct count is 2,000 rows:
inside spec for the sketch, and 100% wrong about the answer.

So the payload publishes two keys, and they are meant to be read together:

| key | meaning |
|---|---|
| `duplicate_rows_est` | the estimate, **suppressed to `0`** when it does not exceed its own uncertainty |
| `duplicate_rows_uncertainty` | one standard deviation on that count, in rows; `0` when the count is exact |

An `est` of `0` beside a large `uncertainty` means *we cannot tell*, not *there
are none*. Full table in
[the `summarize()` schema](../summary-schema.md#reading-the-duplicate-count).

## Getting an Exact Count

If the frame fits in memory and you need certainty rather than a bound, pandas
will tell you exactly — at the cost of holding every row:

```python
exact_duplicates = df.duplicated().sum()
dup_pct = (exact_duplicates / len(df)) * 100
```

The sketch exists for the case where that is not available: constant memory,
whatever the row count.

## See Also

- [Sketch Algorithms](../algorithms/sketches.md) - KMV details
- [Data Quality](quality.md) - Quality metrics
