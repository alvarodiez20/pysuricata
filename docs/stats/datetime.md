---
title: DateTime Variable Analysis
description: Temporal analysis with mathematical formulas for datetime variables in pysuricata
---

# DateTime Variable Analysis

Comprehensive documentation for temporal data analysis in PySuricata, including time distributions, seasonality detection, and gap analysis.

## Overview

PySuricata treats **datetime variables** as columns with temporal types (datetime64, timestamp). Analysis focuses on temporal patterns, distributions, and data quality.

### Key Features

- **Temporal range**: min/max timestamps, time span
- **Distribution analysis**: hour, day-of-week, month patterns
- **Monotonicity detection**: sorted sequences
- **Gap analysis**: missing time periods
- **Timeline visualization**: temporal coverage
- **Timezone handling**: UTC normalization

## Summary Statistics Provided

For each datetime column:

- **Count**: non-null timestamps, missing percentage
- **Range**: minimum and maximum timestamps
- **Span**: total time covered (in days, hours, etc.)
- **Hour distribution**: counts by hour (0-23)
- **Day-of-week distribution**: counts by weekday (Mon-Sun)
- **Month distribution**: counts by month (Jan-Dec)
- **Monotonicity**: increasing/decreasing/mixed
- **Timeline chart**: visual temporal distribution

## Mathematical Definitions

### Temporal Measures

Let \(t_1, t_2, \ldots, t_n\) be the non-missing timestamp values (in seconds since epoch or similar).

**Time span:**

\[
\Delta t = \max(t) - \min(t)
\]

Typically reported in days, hours, or appropriate units.

**Sampling rate (average):**

\[
r = \frac{n}{\Delta t}
\]

Average observations per unit time (e.g., rows per day).

**Time density:**

\[
\rho = \frac{n}{t_{\max} - t_{\min}}
\]

Similar to sampling rate; measures temporal concentration.

### Monotonicity Coefficient

Measures how sorted the timestamps are.

**Strictly increasing pairs:**

\[
n_{\uparrow} = |\{i : t_i < t_{i+1}\}|
\]

**Monotonicity coefficient:**

\[
M = \frac{n_{\uparrow}}{n - 1}
\]

**Interpretation:**
- \(M = 1\): strictly increasing (perfectly sorted)
- \(M = 0\): strictly decreasing (reverse sorted)
- \(M \approx 0.5\): random order

**Use cases:**
- Detect time-sorted data (logs, time series)
- Identify reverse chronological order
- Flag shuffled temporal data

### Temporal Entropy

Distribution entropy over time bins:

\[
H_{\text{time}} = -\sum_{b \in \text{bins}} p_b \log_2 p_b
\]

where \(p_b\) is the proportion of timestamps in bin \(b\).

**High entropy**: events spread uniformly over time  
**Low entropy**: events concentrated in specific periods

### Seasonality Detection

Detect periodic patterns using **Fourier analysis** or **autocorrelation**.

**Autocorrelation at lag \(\tau\):**

\[
\rho(\tau) = \frac{\text{Cov}(X_t, X_{t+\tau})}{\text{Var}(X_t)}
\]

For count time series \(X_t\) (observations per time unit).

**Significant autocorrelation** at lag \(\tau\) suggests periodicity with period \(\tau\).

**Common periods:**
- Daily: \(\tau = 1\) day
- Weekly: \(\tau = 7\) days
- Monthly: \(\tau \approx 30\) days
- Yearly: \(\tau = 365\) days

!!! note "Not fully implemented"
    Seasonality detection via autocorrelation is planned for future release. Current version shows hour/day/month distributions which reveal patterns manually.

## Temporal Distributions

### Hour Distribution

Count of observations by hour of day (0-23):

\[
n_h = |\{i : \text{hour}(t_i) = h\}| \quad \text{for } h \in \{0, 1, \ldots, 23\}
\]

**Use cases:**
- Detect business hours (9am-5pm peaks)
- Identify batch job times (off-hours spikes)
- Analyze user activity patterns

**Visualization:** Bar chart showing hourly counts.

### Day-of-Week Distribution

Count by day (Monday=0, Sunday=6):

\[
n_d = |\{i : \text{weekday}(t_i) = d\}| \quad \text{for } d \in \{0, 1, \ldots, 6\}
\]

**Use cases:**
- Detect weekday vs. weekend patterns
- Identify business day data
- Analyze periodic behavior

**Visualization:** Bar chart showing daily counts.

### Month Distribution

Count by month (Jan=1, Dec=12):

\[
n_m = |\{i : \text{month}(t_i) = m\}| \quad \text{for } m \in \{1, 2, \ldots, 12\}
\]

**Use cases:**
- Detect seasonal effects
- Identify fiscal quarters
- Analyze annual patterns

**Visualization:** Bar chart showing monthly counts.

### Timeline Histogram

Temporal histogram showing observation density over time:

1. Divide time range into \(k\) bins
2. Count observations in each bin
3. Display as histogram

**Bin width:** \(w = \Delta t / k\)

**Reveals:**
- Gaps in data collection
- Burst periods (high activity)
- Data quality issues (missing periods)

## Gap Analysis

Detect missing time periods in temporal data.

**Expected interval:**

\[
\Delta_{\text{exp}} = \text{median}(\{t_{i+1} - t_i : i = 1, \ldots, n-1\})
\]

**Gap threshold:**

\[
\theta = c \cdot \Delta_{\text{exp}}
\]

where \(c > 1\) (e.g., \(c = 2\) or \(c = 5\)).

**Gaps:**

\[
G = \{(t_i, t_{i+1}) : t_{i+1} - t_i > \theta\}
\]

**Gap statistics:**
- Number of gaps: \(|G|\)
- Total missing time: \(\sum_{(t_i, t_{i+1}) \in G} (t_{i+1} - t_i - \theta)\)
- Longest gap: \(\max_{(t_i, t_{i+1}) \in G} (t_{i+1} - t_i)\)

!!! note "Not implemented in current version"
    Gap analysis is planned for future release.

## Timezone Handling

All timestamps are normalized to **UTC** for analysis:

\[
t_{\text{UTC}} = t_{\text{local}} - \text{offset}
\]

**Rationale:**
- Consistent comparisons across time zones
- Avoids DST complications
- Standard for distributed systems

**Reported in UI:** Original timezone if available, UTC for calculations.

## Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Min/max** | \(O(n)\) | \(O(1)\) | Single pass |
| **Hour/day/month counts** | \(O(n)\) | \(O(1)\) | Fixed-size arrays (24, 7, 12) |
| **Monotonicity** | \(O(n)\) | \(O(1)\) | Compare adjacent pairs |
| **Timeline histogram** | \(O(n)\) | \(O(k)\) | \(k\) bins |
| **Gap detection** | \(O(n \log n)\) | \(O(n)\) | Sorting required |

## Configuration

Control datetime analysis via `ReportConfig`:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Timeline histogram bins
# (Not separately configurable, uses default 50)

# Gap detection threshold
# (Not yet implemented)

report = profile(df, config=config)
```

## Implementation Details

### DatetimeAccumulator Class

```python
class DatetimeAccumulator:
    def __init__(self, name: str, config: DatetimeConfig):
        self.name = name
        self.count = 0
        self.missing = 0
        
        # Range tracking
        self.min_ts = None
        self.max_ts = None
        
        # Distribution counters
        self.hour_counts = [0] * 24
        self.weekday_counts = [0] * 7
        self.month_counts = [0] * 12
        
        # Monotonicity tracking
        self.prev_ts = None
        self.monotonic_inc = 0
        self.monotonic_dec = 0
    
    def update(self, values: pd.Series):
        """Update with chunk of timestamps"""
        # Convert to UTC
        # Update min/max
        # Count by hour/day/month
        # Track monotonicity
        pass
    
    def finalize(self) -> DatetimeSummary:
        """Compute final statistics"""
        # Compute span
        # Compute monotonicity coefficient
        # Format distributions
        # Build timeline
        return DatetimeSummary(...)
```

## Examples

### Basic Usage

```python
import pandas as pd
from pysuricata import profile

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=1000, freq="H")
})

report = profile(df)
report.save_html("report.html")
```

### Time Series Data

```python
# Stock prices
df = pd.read_csv("stocks.csv", parse_dates=["date"])

report = profile(df)
# Analyze temporal patterns
```

### Access Statistics

```python
from pysuricata import summarize

stats = summarize(df)
ts_stats = stats["columns"]["timestamp"]

print(f"Min: {ts_stats['min']}")
print(f"Max: {ts_stats['max']}")
print(f"Span: {ts_stats['max'] - ts_stats['min']}")
print(f"Hour distribution: {ts_stats['hour_distribution']}")
```

## Interpreting Results

### Monotonically Increasing

\(M = 1.0\): Timestamps are sorted (common for logs, time series).

**Implications:**
- Data collected in chronological order
- Suitable for time series analysis
- May enable optimizations (binary search)

### Random Order

\(M \approx 0.5\): Timestamps shuffled or unordered.

**Implications:**
- Data may need sorting for analysis
- Not a true time series
- Consider sorting before visualization

### Hourly Patterns

Peak in business hours (9am-5pm):
- Typical for user activity data
- Web traffic, transactions, etc.

Flat distribution:
- Automated data collection (24/7 sensors)
- No human activity pattern

### Weekly Patterns

Weekday peaks, weekend lows:
- Business activity data
- Employee-generated events

Uniform distribution:
- 24/7 operations
- Automated systems

### Monthly Patterns

Seasonal variations:
- Retail sales (holiday spikes)
- Weather data (summer/winter)

Uniform distribution:
- No seasonal effect
- Steady-state process

## Special Cases

### All Same Timestamp

- Distinct count = 1
- Span = 0
- Monotonicity undefined

**Possible issue:** Snapshot data, not time series.

### Large Gaps

Long periods without data:
- Data collection interruptions
- System downtime
- Seasonal business (e.g., ski resorts)

**Recommendation:** Investigate gaps, document known outages.

### Future Timestamps

Timestamps > current time:
- Data quality issue
- Incorrect timezone
- System clock skew

**Recommendation:** Flag as data quality problem.

## References

1. **Box, G.E.P., Jenkins, G.M., Reinsel, G.C. (2015)**, *Time Series Analysis: Forecasting and Control*, Wiley.

2. **Brockwell, P.J., Davis, R.A. (2016)**, *Introduction to Time Series and Forecasting*, Springer.

3. **Cleveland, R.B. et al. (1990)**, "STL: A Seasonal-Trend Decomposition Procedure Based on Loess", *Journal of Official Statistics*, 6(1): 3–73.

4. **Wikipedia: Autocorrelation** - [Link](https://en.wikipedia.org/wiki/Autocorrelation)

5. **Wikipedia: Seasonality** - [Link](https://en.wikipedia.org/wiki/Seasonality)

## See Also

- [Numeric Analysis](numeric.md) - For temporal metrics as numbers
- [Data Quality](../analytics/quality.md) - Quality checks
- [Configuration Guide](../configuration.md) - All parameters


