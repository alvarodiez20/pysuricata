---
title: Data Quality Metrics
description: Data quality assessment and metrics in PySuricata
---

# Data Quality Metrics

PySuricata computes several data quality metrics automatically.

## Dataset-Level Metrics

### Missing Cells Percentage

\[
\text{Missing\%} = \frac{\sum_{\text{cols}} n_{\text{missing}}}{\text{rows} \times \text{cols}} \times 100
\]

**Thresholds:**
- < 5%: Good quality
- 5-20%: Moderate issues
- > 20%: Significant problems

### Duplicate Rows (Approximate)

\[
\text{Dup\%} = \left(1 - \frac{n_{\text{distinct}}}{n_{\text{total}}}\right) \times 100
\]

### Constant Columns

Columns with single unique value (zero variance).

### Highly Correlated Pairs

Pairs with |r| > 0.95 may indicate redundancy.

## Column-Level Metrics

### Completeness

\[
\text{Completeness} = \frac{n_{\text{present}}}{n_{\text{total}}} \times 100
\]

### Cardinality

- Very low (< 10): Consider as categorical
- Very high (> 0.9n): Consider as identifier

### Outliers

Percentage of values outside acceptable ranges.

## Quality Checks in CI/CD

```python
from pysuricata import summarize

def check_quality(df):
    stats = summarize(df)
    
    # Assertions
    assert stats["dataset"]["missing_cells_pct"] < 5.0
    assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
    
    for col, col_stats in stats["columns"].items():
        if "unique" in col.lower():
            # Expect high cardinality for ID columns
            assert col_stats["distinct"] == col_stats["count"]
```

## See Also

- [Missing Values](missing-values.md) - Missing data analysis
- [Duplicates](duplicates.md) - Duplicate detection

---

*Last updated: 2025-10-12*




