# High-Level API

Two entry points cover most workflows:

| Function | Returns | Use case |
|----------|---------|----------|
| `profile(data, config)` | `Report` | HTML report + statistics |
| `summarize(data, config)` | `dict` | Statistics only (no HTML) |

```python
from pysuricata import profile, summarize, ReportConfig
```

## Inputs

`profile()` and `summarize()` accept:

- `pandas.DataFrame`
- `polars.DataFrame` or `polars.LazyFrame` (requires `pysuricata[polars]`)
- `Iterable[pandas.DataFrame]` — a generator yielding chunks

## Report Object

```python
report = profile(df)
report.save_html("report.html")   # Self-contained HTML file
report.save_json("stats.json")    # Statistics as JSON

# Jupyter: displays inline automatically
report
```

## Stats-Only Path

`summarize()` skips HTML rendering — useful for CI/CD checks:

```python
stats = summarize(df)

# Dataset-level
print(stats["dataset"]["rows_est"])
print(stats["dataset"]["missing_cells_pct"])

# Per-column
print(stats["columns"]["age"]["mean"])

# Quality gate
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

## Configuration

All options live in `ReportConfig`:

```python
cfg = ReportConfig()

# Chunking
cfg.compute.chunk_size = 250_000        # rows per chunk (default: 200_000)

# Sampling
cfg.compute.numeric_sample_size = 50_000  # reservoir size (default: 20_000)
cfg.compute.random_seed = 42              # deterministic sampling

# Sketch parameters
cfg.compute.uniques_sketch_size = 2_048  # KMV sketch size (default: 2_048)
cfg.compute.top_k_size = 50             # Misra-Gries k (default: 50)

# Correlations
cfg.compute.compute_correlations = True
cfg.compute.corr_threshold = 0.5        # minimum |r| to report

# Column selection
cfg.compute.columns = ["col_a", "col_b"]  # profile only these

# Render
cfg.render.title = "My Report"

report = profile(df, config=cfg)
```

## Streaming Usage

Pass a generator to process data larger than RAM:

```python
import pandas as pd
from pysuricata import profile

def chunks():
    for path in sorted(Path("data/").glob("*.parquet")):
        yield pd.read_parquet(path)

report = profile(chunks())
report.save_html("report.html")
```

## Determinism

Set `random_seed` to make reservoir sampling reproducible:

```python
cfg = ReportConfig()
cfg.compute.random_seed = 42
# Same data + same seed = identical report
```

## See Also

- [Configuration Guide](configuration.md) — full parameter reference
- [Basic Usage](usage.md) — more examples
- [Performance Tips](performance.md) — tuning for large datasets
