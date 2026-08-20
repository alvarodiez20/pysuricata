---
title: Advanced Features
description: Advanced usage patterns and power user tips for PySuricata
---

# Advanced Features

Advanced techniques for power users.

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

## Custom Markdown Descriptions

Add rich descriptions to reports:

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()
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

## Streaming a Source Directly

Before hand-rolling a generator, check whether the input is one PySuricata
already streams. A path, an Arrow table or reader, and a DuckDB relation are all
read a batch at a time without being materialised:

```python
import duckdb

from pysuricata import profile, summarize

profile("events.parquet")

con = duckdb.connect()
relation = con.sql('''
    SELECT o.*, c.segment
    FROM 'orders/*.parquet' o
    JOIN 'customers.parquet' c USING (customer_id)
    WHERE o.created_at > '2026-01-01'
''')
summarize(relation)
```

That last one is a query that has not run yet — a filtered join across several
Parquet files, profiled without any of it existing as a frame. See
[Arrow, Parquet and DuckDB](data-sources.md).

## Streaming from Multiple Sources

For anything the built-in readers do not cover, combine sources yourself:

```python
from pysuricata import profile
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
from pysuricata import profile
# Sample every Nth row for very large datasets
def sampled_generator(n=10):
    for chunk in pd.read_csv("huge.csv", chunksize=100_000):
        yield chunk[::n]  # Every 10th row

report = profile(sampled_generator())
```

## Merging Accumulator States (Distributed)

```python
import numpy as np

from pysuricata.accumulators import NumericAccumulator

rng = np.random.default_rng(0)
data_partition_1 = rng.lognormal(3, 1, 50_000)
data_partition_2 = rng.lognormal(3, 1, 50_000)

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

This is exact, not an approximation of a single-machine run. The moments merge
by Pébay's formulas and the sketches by construction, so **the merged result
equals the result of one pass over the concatenation** —
`benchmarks/accuracy.py` asserts it. It is also why the order the partitions
arrive in does not matter.

## Conditional Profiling

Profile only rows meeting criteria:

```python
from pysuricata import profile
def filtered_generator():
    for chunk in pd.read_csv("data.csv", chunksize=100_000):
        # Only active users
        yield chunk[chunk["status"] == "active"]

report = profile(filtered_generator())
```

## Seeing a Report Before the Run Finishes

For multi-hour pipelines analyzing massive datasets (i.e. over 100M rows), set
`progress_report=N` to render a report every `N` chunks. The state is also
serialized to disk each time, so an interrupted job can be resumed or
inspected.

One option, one intent: *show me something before it finishes*. It supersedes
`checkpoint.every_n_chunks` and `checkpoint.write_html`, which are deprecated
and removed in 1.0.0. `checkpoint.dir`, `checkpoint.prefix` and
`checkpoint.max_to_keep` still say where the files go, what they are called and
how many are kept.

Turning it on does not change the result. A partial report finalizes the
accumulators at a chunk boundary, and finalizing leaves the reservoir's random
state untouched — `benchmarks/accuracy.py` asserts that the payload is
identical with partial reports on and off.

```python
import numpy as np
import pandas as pd

from pysuricata import ProfileConfig, profile

rng = np.random.default_rng(0)

config = ProfileConfig()
# A report roughly every million rows at the default 50,000-row chunk. Leave
# chunk_size alone -- raising it costs memory and time both, and how often you
# want to see something is what you are actually setting here.
config.compute.progress_report = 20
config.compute.checkpoint.dir = "/tmp/pysuricata/nightly"



def multi_source_generator():
    """Yield one chunk per source file, never holding more than one."""
    for i in range(4):
        yield pd.DataFrame({"amount": rng.lognormal(3, 1, 200_000)})


report = profile(multi_source_generator(), config=config)
```

## See Also

- [Arrow, Parquet and DuckDB](data-sources.md) - Sources that stream themselves
- [Configuration](configuration.md) - All parameters
- [Performance Tips](performance.md) - Optimization
- [Examples](examples.md) - More use cases
