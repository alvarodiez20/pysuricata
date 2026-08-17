# High-Level API

Two entry points cover most workflows:

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

| Function | Returns | Use case |
|----------|---------|----------|
| `profile(data, config)` | `Report` | HTML report + statistics |
| `summarize(data, config)` | `dict` | Statistics only (no HTML) |

```python
from pysuricata import profile, summarize, ProfileConfig
```

## Inputs

`profile()` and `summarize()` accept:

- `pandas.DataFrame`
- `polars.DataFrame` or `polars.LazyFrame` (requires `pysuricata[polars]`)
- `Iterable[pandas.DataFrame]` — a generator yielding chunks

## Report Object

```python
from pysuricata import profile
report = profile(df)
report.save_html("report.html")   # Self-contained HTML file
report.save_json("stats.json")    # Statistics as JSON

# Jupyter: displays inline automatically
report
```

## Stats-Only Path

`summarize()` skips HTML rendering — useful for CI/CD checks:

```python
import numpy as np
import pandas as pd

from pysuricata import summarize

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "customer_id": range(1_000),
        "age": rng.integers(18, 80, 1_000),
        "city": rng.choice(["NY", "LA", "SF"], 1_000),
    }
)
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

All options live in `ProfileConfig`:

```python
from pysuricata import ProfileConfig, profile
cfg = ProfileConfig()

# Chunking
cfg.compute.chunk_size = 250_000        # rows per chunk (default: 50_000)

# Sampling
cfg.compute.numeric_sample_size = 50_000  # reservoir size (default: 20_000)
cfg.compute.random_seed = 42              # deterministic sampling

# Sketch parameters
cfg.compute.max_uniques = 2_048  # KMV sketch size (default: 2_048)
cfg.compute.top_k = 50             # Misra-Gries k (default: 50)

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
from pysuricata import ProfileConfig
cfg = ProfileConfig()
cfg.compute.random_seed = 42
# Same data + same seed = identical report
```

## See Also

- [Configuration Guide](configuration.md) — full parameter reference
- [Basic Usage](usage.md) — more examples
- [Performance Tips](performance.md) — tuning for large datasets
