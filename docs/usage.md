# Usage

## Basic

```python
import pandas as pd
from pysuricata import profile

df = pd.read_csv("data.csv")
rep = profile(df)
rep.save_html("report.html")
```

### Also save stats as JSON

```python
from pysuricata import profile

rep = profile(df)
rep.save_json("report.json")
```

## Streaming large in-memory data

```python
from pysuricata import profile, ReportConfig
import pandas as pd

cfg = ReportConfig()

# From an iterable/generator yielding pandas DataFrame chunks
def chunk_iter():
    for i in range(10):
        yield pd.read_csv(f"/data/part-{i}.csv")  # you pre-chunk externally

rep = profile((ch for ch in chunk_iter()), config=cfg)
rep.save_html("report.html")
```

## Streaming polars

```python
import polars as pl
from pysuricata import profile

df = pl.read_parquet("/data/big.parquet")
rep = profile(df)  # eager or LazyFrame supported
rep.save_html("report.html")
```

### Streaming with polars iterables and LazyFrame

Keep Polars end‑to‑end. The engine consumes either Pandas or Polars chunks.

Iterable of Polars DataFrames:

```python
import polars as pl
from pysuricata import profile, ReportConfig

df = pl.DataFrame({
    "a": list(range(100_000)),
    "b": [float(i) if i % 5 else None for i in range(100_000)],
})

step = 20_000
chunks = (df.slice(i, min(step, df.height - i)) for i in range(0, df.height, step))
rep = profile(chunks, config=ReportConfig())
rep.save_html("polars_iterable_report.html")
```

Polars LazyFrame (windowed collect under the hood):

```python
import polars as pl
from pysuricata import profile, ReportConfig, ComputeOptions

lf = (
    pl.LazyFrame({
        "x": list(range(200_000)),
        "y": [float(i) if i % 7 else None for i in range(200_000)],
        "z": ["a" if i % 2 else "b" for i in range(200_000)],
    })
    .with_columns(pl.col("x") * 2)
)

cfg = ReportConfig(compute=ComputeOptions(chunk_size=50_000))
rep = profile(lf, config=cfg)
rep.save_html("polars_lazy_report.html")
```

## Deterministic visuals (reproducible sampling)

Use `random_seed` to make histogram sampling deterministic across runs.

```python
from pysuricata import profile, ReportConfig

cfg = ReportConfig()
cfg.compute.random_seed = 42
rep = profile(df, config=cfg)
```

## Programmatic summary

Ask for a compact JSON-like dictionary of stats:

```python
from pysuricata import summarize
summary = summarize(df)
print(summary["dataset"])           # rows_est, cols, missing_cells, duplicates, top-missing
print(summary["columns"]["amount"]) # per-column stats by type
```

### Processed bytes and timing

The report displays:
- Processed bytes (≈): total bytes handled across chunks (not peak RSS)
- Precise generation time in seconds (e.g., 0.02s)

## End-to-end minimal example

```python
import pandas as pd
from pysuricata import profile, ReportConfig

df = pd.DataFrame(
    {
        "amount": [1.0, 2.5, None, 4.0, 5.5],
        "country": ["US", "US", "DE", None, "FR"],
        "ts": pd.to_datetime(["2021-01-01", "2021-01-02", None, "2021-01-04", "2021-01-05"]),
        "flag": [True, False, True, None, False],
    }
)

cfg = ReportConfig()
cfg.compute.random_seed = 0
rep = profile(df, config=cfg)
rep.save_html("report.html")
```
