---
title: Basic Usage
description: How to generate reports and access statistics with PySuricata
---

# Basic Usage

## Generating an HTML Report

The simplest way to use PySuricata is to generate an HTML report from a DataFrame:

```python
import pandas as pd
from pysuricata import profile

df = pd.read_csv("data.csv")
report = profile(df)
report.save_html("report.html")
```

Open `report.html` in any browser. The file is self-contained — no external assets needed.

## Using Polars

PySuricata works natively with polars DataFrames and LazyFrames. Install polars support with:

```bash
pip install pysuricata[polars]
```

Then use it the same way:

```python
import polars as pl
from pysuricata import profile

# Eager DataFrame
df = pl.read_csv("data.csv")
report = profile(df)
report.save_html("report.html")

# LazyFrame — PySuricata collects it in chunks internally
lf = pl.scan_csv("large_file.csv")
report = profile(lf)
report.save_html("report.html")
```

## Streaming Large Datasets

For datasets that don't fit in memory, pass a generator yielding DataFrame chunks:

=== "Pandas"

    ```python
    import pandas as pd
    from pysuricata import profile

    def read_in_chunks():
        for i in range(10):
            yield pd.read_csv(f"data/part-{i}.csv")

    report = profile(read_in_chunks())
    report.save_html("report.html")
    ```

=== "Pandas chunked reader"

    ```python
    import pandas as pd
    from pysuricata import profile

    # pandas read_csv has a built-in chunksize parameter
    chunks = pd.read_csv("large_file.csv", chunksize=200_000)
    report = profile(chunks)
    report.save_html("report.html")
    ```

=== "Polars"

    ```python
    import polars as pl
    from pysuricata import profile

    df = pl.read_parquet("large_file.parquet")

    # Manually slice into chunks
    step = 200_000
    chunks = (df.slice(i, min(step, df.height - i)) for i in range(0, df.height, step))

    report = profile(chunks)
    report.save_html("report.html")
    ```

Each chunk is processed and discarded, so memory stays bounded regardless of total dataset size.

## Getting Statistics Without HTML

Use `summarize()` to get a dictionary of statistics without generating an HTML report:

```python
from pysuricata import summarize

stats = summarize(df)

# Dataset-level statistics
print(stats["dataset"])
# {'rows_est': 891, 'cols': 12, 'missing_cells_pct': 8.7, ...}

# Per-column statistics
print(stats["columns"]["age"])
# {'mean': 29.7, 'std': 14.5, 'min': 0.42, 'max': 80.0, ...}
```

This is useful for CI/CD quality checks:

```python
stats = summarize(df)
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

## Saving Stats as JSON

```python
report = profile(df)
report.save_json("stats.json")
```

## Reproducible Reports

Set `random_seed` to make histogram sampling deterministic across runs:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()
config.compute.random_seed = 42

report = profile(df, config=config)
# Same report every time with the same data
```

## End-to-End Example

A complete example covering all four column types:

```python
import pandas as pd
from pysuricata import profile, ReportConfig

df = pd.DataFrame({
    "amount": [1.0, 2.5, None, 4.0, 5.5],
    "country": ["US", "US", "DE", None, "FR"],
    "ts": pd.to_datetime(["2021-01-01", "2021-01-02", None, "2021-01-04", "2021-01-05"]),
    "flag": [True, False, True, None, False],
})

config = ReportConfig()
config.compute.random_seed = 0

report = profile(df, config=config)
report.save_html("report.html")
```

This generates a report with:

- **amount** analyzed as numeric (mean, std, histogram, outliers)
- **country** analyzed as categorical (top values, distinct count, entropy)
- **ts** analyzed as datetime (range, day-of-week distribution)
- **flag** analyzed as boolean (true/false ratio, balance score)

## See Also

- [Configuration Guide](configuration.md) — All available options
- [Advanced Features](advanced.md) — Streaming from multiple sources, distributed processing
- [Examples Gallery](examples.md) — More real-world use cases
