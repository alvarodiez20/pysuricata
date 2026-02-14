---
title: Advanced Features
description: Advanced usage patterns and power user tips for PySuricata
---

# Advanced Features

Advanced techniques for power users.

## Custom Markdown Descriptions

Add rich descriptions to reports:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()
config.render.description = """
# Q4 2024 Analysis

**Dataset**: Customer transactions  
**Period**: Oct-Dec 2024  
**Source**: production.transactions

## Key Findings

- Revenue up 15% YoY
- Average transaction: $87.50
- Peak hour: 2pm EST
"""

report = profile(df, config=config)
```

## Streaming from Multiple Sources

Combine data from multiple sources:

```python
def multi_source_generator():
    # Source 1: CSV files
    for i in range(10):
        yield pd.read_csv(f"batch_{i}.csv")
    
    # Source 2: Parquet files
    for i in range(5):
        yield pd.read_parquet(f"archive_{i}.parquet")
    
    # Source 3: Database
    for chunk in pd.read_sql("SELECT * FROM logs", conn, chunksize=100_000):
        yield chunk

report = profile(multi_source_generator())
```

## Parallel Processing with Dask

```python
import dask.dataframe as dd
from pysuricata import profile

# Load with Dask
ddf = dd.read_csv("large_*.csv")

# Convert to generator
def dask_generator():
    for partition in ddf.partitions:
        yield partition.compute()

report = profile(dask_generator())
```

## Custom Sampling Strategy

```python
# Sample every Nth row for very large datasets
def sampled_generator(n=10):
    for chunk in pd.read_csv("huge.csv", chunksize=100_000):
        yield chunk[::n]  # Every 10th row

report = profile(sampled_generator())
```

## Merging Accumulator States (Distributed)

```python
from pysuricata.accumulators import NumericAccumulator

# Worker 1
acc1 = NumericAccumulator("amount")
acc1.update(data_partition_1)

# Worker 2
acc2 = NumericAccumulator("amount")
acc2.update(data_partition_2)

# Merge on coordinator
acc1.merge(acc2)
final_stats = acc1.finalize()
```

## Conditional Profiling

Profile only rows meeting criteria:

```python
def filtered_generator():
    for chunk in pd.read_csv("data.csv", chunksize=100_000):
        # Only active users
        yield chunk[chunk["status"] == "active"]

report = profile(filtered_generator())
```

## See Also

- [Configuration](configuration.md) - All parameters
- [Performance Tips](performance.md) - Optimization
- [Examples](examples.md) - More use cases

