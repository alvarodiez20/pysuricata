# PySuricata

Generate clean, self-contained EDA reports for pandas/polars DataFrames and large in-memory chunked iterables.


!!! tip
    Works great on small and medium datasets; for very large datasets, sample first.

## Features
- Summary stats, missingness, duplicates
- Numeric, categorical, datetime, and boolean cards with inline SVG charts
- Out-of-core streaming for in-memory DataFrame chunks (low peak memory)
- Approximate distinct counts and heavy hitters for large columns
- Streaming correlations for numeric columns
- Self-contained HTML export (inline CSS/JS/images)

## Quick links
- [Installation](install.md)
- [Usage](usage.md)
- [API](api.md)
- [Architecture](architecture.md)

## Quick start

```python
import pandas as pd
from pysuricata import profile, ReportConfig

df = pd.read_csv("/path/to/data.csv")
rep = profile(df, config=ReportConfig())
rep.save_html("report.html")

# Or stream in-memory chunks you create
def chunk_iter():
    for i in range(10):
        yield pd.read_csv(f"/data/part-{i}.csv")

rep = profile((ch for ch in chunk_iter()), config=ReportConfig())
rep.save_html("report.html")
```

## How it works

- Reads input in chunks (pandas DataFrames) and feeds type-specific accumulators.
- Numeric accumulators maintain Welford/Pébay moments, a reservoir sample, and KMV distinct.
- Categorical accumulators use Misra–Gries for top-k and KMV for distinct.
- Datetime accumulators count by hour/day/month and keep min/max.
- A lightweight streaming correlation estimator tracks Pearson r for numeric pairs.
- The template renders a self-contained HTML with precise duration (e.g., 0.02s) and processed bytes (≈).
- Deterministic visuals via `ReportConfig.compute.random_seed`.

See also: Numeric analysis details in numeric_var.md.
