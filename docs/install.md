# Installation

## Requirements

**Python 3.10 or newer.** 3.10 through 3.14 are tested on every push.

The only runtime dependencies are **pandas**, **numpy** and **markdown**. That
is deliberate: no plotting library, no templating engine, no web framework — the
report is hand-rolled SVG inlined into one HTML file.

## From PyPI

```bash
# using uv (recommended)
uv add pysuricata

# or using pip
pip install pysuricata
```

## Extras

| extra | pulls in | needed for |
|---|---|---|
| `polars` | `polars>=1.34` | profiling `polars.DataFrame` / `LazyFrame` |
| `system` | `psutil` | only the memory-measurement recipes in [Performance](performance.md) — nothing in the library imports it |

```bash
uv add "pysuricata[polars]"
# or: pip install "pysuricata[polars]"
```

**`pyarrow` and `duckdb` are not extras and not dependencies.** Reading Parquet
or Arrow needs `pyarrow` installed however you like, and says so if it is
missing; the DuckDB reader is duck-typed on the relation's batch-reader method,
so nothing imports it. See
[Arrow, Parquet and DuckDB](data-sources.md).

## Verify

```pycon
>>> import pandas as pd
>>> from pysuricata import profile
>>> df = pd.DataFrame({"x": [1, 2, 3]})
>>> profile(df).html[:15]
'<!DOCTYPE html>'
```

Installing also puts a `pysuricata` command on your path:

```bash
pysuricata --version
```

See the [CLI reference](cli.md).

## Next

- [Quick Start](quickstart.md) — your first report
- [Why PySuricata?](why-pysuricata.md) — the design decisions behind it
