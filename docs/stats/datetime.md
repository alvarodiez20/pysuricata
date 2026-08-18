---
title: DateTime Variable Analysis
description: Temporal analysis with mathematical formulas for datetime variables in pysuricata
---

# DateTime Variable Analysis

Comprehensive documentation for temporal data analysis in PySuricata, including time distributions, seasonality detection, and gap analysis.

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

!!! note "Derived, not published"
    Not a key in the payload. The bin counts it needs are: `by_hour`, `by_dow`,
    `by_month` and `by_year` are all published, so this is one line over
    whichever binning you care about.

Distribution entropy over time bins:

\[
H_{\text{time}} = -\sum_{b \in \text{bins}} p_b \log_2 p_b
\]

where \(p_b\) is the proportion of timestamps in bin \(b\).

**High entropy**: events spread uniformly over time  
**Low entropy**: events concentrated in specific periods

### Seasonality

`seasonal_pattern` is published, and it is derived from the binned
distributions — the hour, day-of-week, month and year counts — not from an
autocorrelation.

Autocorrelation at lag \(\tau\),

\[
\rho(\tau) = \frac{\text{Cov}(X_t, X_{t+\tau})}{\text{Var}(X_t)}
\]

would be the stronger method, and it is not a streaming one: it needs the count
series \(X_t\) held in full at a resolution nobody can pick in advance. The
binned distributions are what a single bounded-memory pass can honestly offer,
and they reveal a daily, weekly or yearly shape by inspection. Anything finer
belongs to a tool that gets to hold the series.

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

## Intervals, and Why There Is No Gap List

Two interval statistics are published:

| key | meaning |
|---|---|
| `avg_interval_seconds` | mean gap between consecutive timestamps |
| `interval_std_seconds` | its standard deviation |

Together they answer *is this regular*: a nightly extract has a std near zero
against an average near 86,400, and one that has been missing days does not.

A **list of gaps** is not published. Finding
\(G = \{(t_i, t_{i+1}) : t_{i+1} - t_i > \theta\}\) needs consecutive
timestamps in order, and a stream offers neither — chunks arrive in whatever
order the source yields them, and the column need not be sorted. Reporting gaps
from a sample would produce a list that changes between runs, which is worse
than no list.

`mono_inc` and `mono_dec` tell you whether the column *is* ordered, which is the
precondition for the question. If it is, and you need the gaps, you have the
column.

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

Control datetime analysis via `ProfileConfig`:

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()

# Timeline histogram bins
# (Not separately configurable, uses default 50)

# Gap detection threshold
# (Not yet implemented)

report = profile(df, config=config)
```

## Implementation Details

`DatetimeAccumulator` lives in `pysuricata/accumulators/datetime.py`. The shape:

- `update()` takes a **numpy array** of epoch-nanosecond integers, never a frame
  or a Series.
- Bounded state: min and max timestamps, fixed-width count arrays for hour,
  day-of-week, month and year, a monotonicity tracker, and a running interval
  accumulator. None of it grows with the number of rows.
- `merge()` takes element-wise sums of the count arrays and the extremes of the
  bounds — exact, and order-independent.
- Timestamps are normalised to UTC for the arithmetic; the original zone is
  reported as `source_timezone`.

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
from pysuricata import profile
# Stock prices
df = pd.read_csv("stocks.csv", parse_dates=["date"])

report = profile(df)
# Analyze temporal patterns
```

### Access Statistics

```python
import pandas as pd

from pysuricata import summarize

df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=500, freq="17min")})
stats = summarize(df)
ts_stats = stats["columns"]["timestamp"]

# min_ts and max_ts are nanoseconds since the epoch, UTC.
print(f"Min:  {pd.Timestamp(ts_stats['min_ts'])}")
print(f"Max:  {pd.Timestamp(ts_stats['max_ts'])}")
print(f"Span: {pd.Timedelta(ts_stats['max_ts'] - ts_stats['min_ts'])}")
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
