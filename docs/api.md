# High-level API

The unified API exposes two entry points that cover most workflows:

- `profile(data, config=None) -> Report`: compute + HTML
- `summarize(data, config=None) -> Mapping[str, Any]`: stats-only

Import from the package root:

```python
from pysuricata import profile, summarize, ReportConfig
```

## Inputs

- In-memory `pandas.DataFrame`
- Iterable/generator yielding pandas or polars `DataFrame` chunks
- `polars.DataFrame` or `LazyFrame` (processed natively; no pandas conversion)

## Report object

```python
from pysuricata import profile, ReportConfig

rep = profile(df, config=ReportConfig())
rep.save_html("report.html")
rep.save_json("report.json")

# In notebooks, the report displays inline
rep
```

## Quick render

```python
rep = profile(df)
rep.save_html("report.html")
```

## Stats-only (CI/data-quality)

```python
stats = summarize(df)  # compute-only fast path (skips HTML)

# Example: assert no column has > 10% missing
bad = [
    (name, col["missing"]) for name, col in stats["columns"].items()
    if col.get("missing", 0) / max(1, col.get("count", 0)) > 0.10
]
assert not bad, f"Columns too missing: {bad}"
```

## Configuration

The top-level `ReportConfig` wraps compute and render options:

```python
from pysuricata import ReportConfig

cfg = ReportConfig(
    compute=ReportConfig.compute.__class__(
        chunk_size=250_000,
        columns=["a", "b", "c"],
        numeric_sample_size=50_000,
        max_uniques=4096,
        top_k=100,
        engine="auto",
    )
)

rep = profile(df, config=cfg)
```

### Load and chunk outside

You can read data with any library and either pass a single DataFrame or an iterable of DataFrames you manage:

```python
import pandas as pd
from pysuricata import profile

def chunk_iter():
    for i in range(10):
        yield pd.read_parquet(f"/data/part-{i}.parquet")

rep = profile((ch for ch in chunk_iter()))
```

## Common use cases

- Small DataFrame (in-memory):
  ```python
  import pandas as pd
  from pysuricata import profile
  df = pd.DataFrame({"x": [1,2,3], "y": ["a","b","a"]})
  rep = profile(df)
  ```

- Large dataset (streaming in-memory):
  ```python
  from pysuricata import ReportConfig, profile
  cfg = ReportConfig()
  cfg.compute.chunk_size = 250_000
  rep = profile((ch for ch in chunk_iter()), config=cfg)
  rep.save_html("report.html")
  ```

- Column selection:
  ```python
  from pysuricata import ReportConfig, summarize
  cfg = ReportConfig()
  cfg.compute.columns = ["id", "amount", "ts"]
  stats = summarize(df[["id", "amount", "ts"]])
  ```

- CI check: enforce low duplicates and missingness:
  ```python
  stats = summarize(df)
  ds = stats["dataset"]
  assert ds["duplicate_rows_pct_est"] < 1.0
  assert ds["missing_cells_pct"] < 5.0
  ```

## Notes and limits

- Current engine consumes pandas or polars DataFrames (or iterables of them). Polars eager/LazyFrames are processed natively without conversion.
- `render` options are placeholders; the v2 template already renders a self-contained light theme.
 
