---
title: Basic Usage
description: How to generate reports and access statistics with PySuricata
---

# Basic Usage

## Generating an HTML Report

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

The simplest way to use PySuricata is to generate an HTML report from a DataFrame:

```python
import pandas as pd
from pysuricata import profile

df = pd.read_csv("data.csv")
report = profile(df)
report.save_html("report.html")
```

Open `report.html` in any browser. The file is self-contained — no external assets needed.

## Profiling a File Without Loading It

`profile()` takes a path as readily as a frame, and reads it a batch at a time:

```python
from pysuricata import profile

report = profile("events.parquet")
report.save_html("report.html")
```

CSV, Parquet, JSON and Arrow IPC (`.arrow`, `.feather`, `.ipc`) all work, as do
an Arrow table or reader and a DuckDB relation. The file never exists as one
frame, which is where the memory saving comes from — 307 MB against 581 MB on a
180 MB Parquet file. See [Arrow, Parquet and DuckDB](data-sources.md).

## Setting Options

Most calls need nothing. When one does, the seven most-reached-for settings are
keywords:

```python
from pysuricata import profile

report = profile(df, seed=42, correlations=False, title="Q4 2024")
```

Or take a preset, and adjust from there:

```python
from pysuricata import profile

report = profile(df, preset="fast")           # or "thorough"
report = profile(df, preset="fast", sample=20_000)
```

A `ProfileConfig` is the escape hatch for everything else — see
[Configuration](configuration.md).

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
from pysuricata import summarize
stats = summarize(df)
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

The payload's keys are a versioned contract — see
[the `summarize()` schema](summary-schema.md). If you want a build gate with an
exit code rather than an assertion, `pysuricata check` does the same single pass
from a shell; see [Gating CI on drift](data-checks.md).

## Comparing Two Datasets

`compare(a, b)` reports what moved — schema, dataset and per column — as a
description rather than a verdict:

```python
import numpy as np
import pandas as pd

from pysuricata import compare

rng = np.random.default_rng(0)
before = pd.DataFrame({"amount": rng.lognormal(3, 1.0, 5_000)})
after = pd.DataFrame({"amount": rng.lognormal(3.4, 1.0, 5_000)})

diff = compare(before, after)
print(diff.columns["amount"].median_shift_sigma)
```

See [Comparing two datasets](comparing.md).

## Saving Stats as JSON

```python
from pysuricata import profile
report = profile(df)
report.save_json("stats.json")
```

## Reproducible Reports

They already are. `random_seed` defaults to `0`, so the same data produces the
same report — re-running is a no-op rather than a set of sampling wobbles.

Pass `seed=` to pin a different sample:

```python
from pysuricata import profile

report = profile(df, seed=42)
```

`seed=None` gives a fresh sample each run, if that is what you want.

## End-to-End Example

A complete example covering all four column types:

```python
import pandas as pd
from pysuricata import profile

df = pd.DataFrame({
    "amount": [1.0, 2.5, None, 4.0, 5.5],
    "country": ["US", "US", "DE", None, "FR"],
    "ts": pd.to_datetime(["2021-01-01", "2021-01-02", None, "2021-01-04", "2021-01-05"]),
    "flag": [True, False, True, None, False],
})

report = profile(df, title="Four column kinds")
report.save_html("report.html")
```

This generates a report with:

- **amount** analyzed as numeric (mean, std, histogram, outliers)
- **country** analyzed as categorical (top values, distinct count, entropy)
- **ts** analyzed as datetime (range, day-of-week distribution)
- **flag** analyzed as boolean (true/false counts and ratios, entropy)

## See Also

- [Configuration Guide](configuration.md) — All available options
- [Advanced Features](advanced.md) — Streaming from multiple sources, distributed processing
- [Examples Gallery](examples.md) — More real-world use cases
- [Command Line](cli.md) — the same operations from a shell
