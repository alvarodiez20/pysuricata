---
title: Missing Values Analysis
description: Intelligent missing data detection, visualization, and analysis with adaptive display
---

# Missing Values Analysis

Comprehensive guide to PySuricata's intelligent missing values analysis with adaptive display and chunk-level distribution tracking.

## Overview

Missing data is ubiquitous in real-world datasets. PySuricata provides:

- **Intelligent display**: Adaptive limits based on dataset size
- **Chunk-level tracking**: See missing data distribution across chunks
- **Pattern detection**: Identify systematic missingness
- **Smart filtering**: Show only significant missing columns
- **Expandable UI**: Progressive disclosure for many columns

## Missing Data Mechanisms

### MAR, MCAR, MNAR

**Missing Completely At Random (MCAR)**:
- Missingness independent of observed/unobserved data
- \(P(\text{missing} | X, Y) = P(\text{missing})\)
- **Example**: Sensor randomly fails

**Missing At Random (MAR)**:
- Missingness depends on observed data only
- \(P(\text{missing} | X, Y_{\text{obs}}) = P(\text{missing} | X)\)
- **Example**: Older patients skip optional questions

**Missing Not At Random (MNAR)**:
- Missingness depends on unobserved values
- **Example**: High earners don't report income

!!! note "Detection not automated"
    Determining mechanism requires domain knowledge. PySuricata shows patterns to help investigation.

## Mathematical Definitions

### Missing Rate

For column with \(n_{\text{total}}\) observations:

\[
MR = \frac{n_{\text{missing}}}{n_{\text{total}}}
\]

### Missing Pattern Entropy

For \(k\) different missing patterns (combinations of missing columns):

\[
H_{\text{pattern}} = -\sum_{i=1}^{k} p_i \log_2 p_i
\]

where \(p_i\) is the proportion of rows with pattern \(i\).

**High entropy**: Many different patterns (complex missingness)  
**Low entropy**: Few patterns (systematic missingness)

!!! note "Not implemented"
    Pattern entropy computation is planned for future release.

## Intelligent Display System

### Dynamic Limits

Limits adapt to dataset size:

| Dataset Size | Initial Display | Expanded Display |
|--------------|-----------------|------------------|
| ≤10 columns | All | All |
| 11-50 columns | 10 | 25 |
| 51-200 columns | 12 | 25 |
| >200 columns | 15 | 25 |

### Smart Filtering

**Threshold**: Only show columns with >\(t\)% missing (default \(t=0.5\)%)

**Rationale**: Columns with <0.5% missing are usually not concerning.

### Expandable UI

For datasets with many missing columns:
1. **Initial view**: Show top \(n\) columns
2. **Expand button**: Reveal up to 25 total
3. **Smooth animation**: JavaScript-powered transition

## Chunk-Level Distribution

Track missing data **per chunk** to identify:
- Temporal patterns (early vs. late data)
- Batch patterns (certain files have more missing)
- System issues (outages, collection failures)

### Visualization

Horizontal bar showing missing percentage per chunk:

```
Chunk 1  ████░░░░░░  40%
Chunk 2  ██░░░░░░░░  20%
Chunk 3  ░░░░░░░░░░   0%
Chunk 4  ███████░░░  70%
```

Reveals chunk 4 has data quality issue.

## Configuration

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Missing columns display threshold (default 0.5%)
# (Not yet configurable in current version)

# Maximum initial display (default: dynamic based on dataset size)
# (Not yet configurable in current version)

report = profile(df, config=config)
```

## Implementation

### MissingColumnsAnalyzer

```python
class MissingColumnsAnalyzer:
    MIN_THRESHOLD_PCT = 0.5
    MAX_INITIAL_DISPLAY = 8
    MAX_EXPANDED_DISPLAY = 25
    
    def analyze_missing_columns(self, miss_list, n_cols, n_rows):
        """Analyze and filter missing columns"""
        # Filter significant missing
        significant = [
            item for item in miss_list 
            if item[1] >= self.MIN_THRESHOLD_PCT
        ]
        
        # Determine limits
        initial_limit = self._get_initial_display_limit(n_cols, n_rows)
        expanded_limit = self._get_expanded_display_limit(n_cols, n_rows)
        
        # Build result
        return MissingColumnsResult(
            initial_columns=significant[:initial_limit],
            expanded_columns=significant[:expanded_limit],
            needs_expandable=len(significant) > initial_limit,
            total_significant=len(significant),
            total_insignificant=len(miss_list) - len(significant)
        )
```

## Interpreting Results

### High Missing Percentage (>50%)

**Possible causes:**
- Optional field (by design)
- Data collection issue
- Recent column (added midway)
- Rare event (e.g., "error_message" only on errors)

**Actions:**
- Verify if intentional
- Consider imputation or exclusion
- Check data pipeline

### Systematic Patterns

Multiple columns missing together:

**Possible causes:**
- Related optional section (e.g., address fields)
- Batch import failure
- Survey skip logic

**Actions:**
- Analyze co-occurrence
- Check data source
- Document business logic

### Increasing Over Time

More missing in later chunks:

**Possible causes:**
- Degrading data quality
- System malfunction
- Intentional change

**Actions:**
- Investigate recent changes
- Alert data engineering team

## Little's MCAR Test

Statistical test for MCAR assumption.

**Null hypothesis**: Data is MCAR

**Test statistic**: Compare means of subgroups defined by missing patterns

**P-value interpretation**:
- Large p-value: Consistent with MCAR
- Small p-value: Reject MCAR (MAR or MNAR)

!!! note "Not implemented"
    Little's test is planned for future release. Currently, users must perform external analysis.

**Reference**: Little, R.J.A. (1988), "A Test of Missing Completely at Random for Multivariate Data with Missing Values", *JASA*, 83(404): 1198–1202.

## Imputation Considerations

### Mean/Median Imputation

\[
x_{\text{imputed}} = \begin{cases}
x & \text{if observed} \\
\bar{x} & \text{if missing}
\end{cases}
\]

**Pros**: Simple, fast  
**Cons**: Reduces variance, distorts correlations

### Multiple Imputation

Generate \(m\) complete datasets with different imputations, analyze separately, combine results.

**Pros**: Preserves uncertainty  
**Cons**: Complex, computationally expensive

### Model-Based

Use ML model to predict missing values from other columns.

**Pros**: Can capture complex relationships  
**Cons**: Requires training, may introduce bias

!!! warning "PySuricata does not impute"
    PySuricata is a profiling tool, not a preprocessing tool. Imputation should be done separately based on domain knowledge.

## Best Practices

1. **Document missingness**: Record why data is missing
2. **Distinguish NULL types**: NULL vs. empty string vs. "N/A"
3. **Set thresholds**: Define acceptable missing percentages
4. **Monitor trends**: Track missing rates over time
5. **Investigate patterns**: Look for systematic missingness

## Examples

### Basic Usage

```python
from pysuricata import profile

# Dataset with missing values
df = pd.DataFrame({
    "age": [25, 30, None, 45, 50],
    "income": [50000, None, None, 80000, 90000],
    "city": ["NYC", "LA", None, "Chicago", None]
})

report = profile(df)
# Report shows missing percentages and patterns
```

### Access Missing Statistics

```python
from pysuricata import summarize

stats = summarize(df)
print(f"Missing cells: {stats['dataset']['missing_cells_pct']:.1f}%")

for col, col_stats in stats["columns"].items():
    missing_pct = col_stats.get("missing_pct", 0)
    if missing_pct > 10:
        print(f"{col}: {missing_pct:.1f}% missing")
```

## References

1. **Little, R.J.A., Rubin, D.B. (2019)**, *Statistical Analysis with Missing Data*, 3rd ed., Wiley.

2. **Rubin, D.B. (1976)**, "Inference and Missing Data", *Biometrika*, 63(3): 581–592.

3. **Schafer, J.L., Graham, J.W. (2002)**, "Missing Data: Our View of the State of the Art", *Psychological Methods*, 7(2): 147–177.

4. **Wikipedia: Missing data** - [Link](https://en.wikipedia.org/wiki/Missing_data)

## See Also

- [Data Quality](quality.md) - Overall quality metrics
- [Numeric Analysis](../stats/numeric.md) - Handling missing in numeric columns
- [Configuration](../configuration.md) - Display settings

---

*Last updated: 2025-10-12*




