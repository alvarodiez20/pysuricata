# High-Level API

Two entry points cover most workflows, and a third for comparing two datasets.

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
| `profile(data, ...)` | `Report` | HTML report + statistics |
| `summarize(data, ...)` | `dict` | Statistics only, no HTML rendered |
| `compare(a, b, ...)` | `Comparison` | Every delta between two datasets |

```python
from pysuricata import compare, profile, summarize
```

For signatures and every parameter, see the [generated reference](reference.md).

## Inputs

`profile()` and `summarize()` accept:

| input | note |
|---|---|
| `pandas.DataFrame` | |
| `polars.DataFrame` or `polars.LazyFrame` | requires `pysuricata[polars]` |
| a path — `str` or `os.PathLike` | `.csv`, `.parquet`, `.json`, `.arrow`, `.feather`, `.ipc`, `.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods` |
| an Arrow table or reader | or anything exporting `__arrow_c_stream__` |
| a DuckDB relation | a query that has not run yet |
| `Iterable[DataFrame]` | a generator yielding pandas or polars chunks |

The middle three are read a **batch at a time** and never materialised as one
frame. `pyarrow` is needed for Parquet and Arrow, and says so if it is missing;
DuckDB is duck-typed on the relation's batch reader, so nothing imports it. See
[Arrow, Parquet and DuckDB](data-sources.md).

A spreadsheet is the one path format that cannot stream — no engine puts a
`chunksize` on `read_excel`, so a workbook is always built whole before
anything sees a row. `python-calamine` is tried first (one dependency across
all five spreadsheet formats, and what the [browser demo](https://pysuricata.pages.dev)
already uses), falling back to pandas' own per-format engine — openpyxl,
xlrd, pyxlsb, odfpy — when calamine is not installed. Only the first sheet is
read; a workbook that keeps its data on a later sheet needs
`pd.read_excel(path, sheet_name=...)` and the resulting frame passed in
directly.

Anything else raises `UnsupportedDataError`, whose message lists the forms above.

## Configuring a call

Three forms, cheapest first. `configuration.md` has the full
[side-by-side](configuration.md#three-ways-to-configure-in-order-of-effort);
in short:

```python
from pysuricata import ProfileConfig, profile

# 1. keyword options -- the common case
profile(df, seed=42, correlations=False, title="Q4 2024")

# 2. a preset -- one word for an intent
profile(df, preset="fast")          # or "thorough"

# 3. a ProfileConfig -- the escape hatch
my_config = ProfileConfig()
my_config.compute.max_uniques = 8_192
profile(df, config=my_config)
```

The seven keywords are `chunk_size`, `columns`, `sample`, `correlations`,
`seed`, `progress` and `title`. Anything else raises `ConfigurationError`, and
the message names the keyword you were probably reaching for.

Precedence runs defaults → preset → keyword options. `config=` **cannot** be
combined with either of the other two: it takes everything, and a caller who
built one means it.

## Report Object

```python
from pysuricata import profile

report = profile(df)

report.html                        # the document, as a string
report.stats                       # the same mapping summarize() returns

report.save_html("report.html")    # self-contained HTML
report.save_json("stats.json")     # statistics as JSON
report.save("out.html")            # dispatches on the extension

report                             # Jupyter: displays inline
report.show(height="800px")        # or explicitly, at a size
```

`save()` picks HTML or JSON from the suffix, so a variable holding a path does
not need a branch around it.

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

The payload is versioned and its keys are a contract — see
[the `summarize()` schema](summary-schema.md). For a gate with an exit code
rather than an assertion, see [`pysuricata check`](cli.md#check).

## Comparing Two Datasets

```python
import numpy as np
import pandas as pd

from pysuricata import compare

rng = np.random.default_rng(0)
last_month = pd.DataFrame({"amount": rng.lognormal(3, 1.0, 5_000)})
this_month = pd.DataFrame({"amount": rng.lognormal(3.4, 1.0, 5_000)})

diff = compare(last_month, this_month)

diff.schema.added                          # columns that appeared
diff.columns["amount"].median_shift_sigma  # in baseline standard deviations
diff.to_dict()                             # JSON-safe
```

Both sides accept anything `profile()` does, or a `summarize()` payload you
already have, which is not re-profiled. See
[Comparing two datasets](comparing.md).

## Errors

Everything the API raises deliberately descends from `PySuricataError`, so one
`except` catches the lot:

| exception | also a | raised when |
|---|---|---|
| `PySuricataError` | `Exception` | the base — catch this to catch all three |
| `UnsupportedDataError` | `TypeError` | `data` is not one of the accepted forms |
| `ConfigurationError` | `ValueError` | an option, preset or config value is not valid |

Because each one also subclasses the builtin it replaces, existing
`except TypeError` / `except ValueError` handlers keep working.

```python
from pysuricata import ConfigurationError, profile

try:
    profile(df, chunksize=50_000)   # not a keyword -- note the missing underscore
except ConfigurationError as e:
    print(e)   # names `chunk_size`
```

## Streaming Usage

Pass a path to stream a file, or a generator to produce the chunks yourself:

```python
from pathlib import Path

import pandas as pd

from pysuricata import profile

profile("events.parquet")           # streamed, a batch at a time


def chunks():
    for path in sorted(Path("data/").glob("*.parquet")):
        yield pd.read_parquet(path)


report = profile(chunks())
report.save_html("report.html")
```

Each chunk is folded into the accumulators and discarded, so memory stays
bounded in rows. It is not bounded in *columns* — see
[#207](https://github.com/alvarodiez20/pysuricata/issues/207).

## Determinism

Reports are reproducible by default: `random_seed` is `0`, not `None`, so the
same data produces the same report and re-running is a no-op.

```python
from pysuricata import profile

profile(df, seed=42)     # a different, still-fixed sample
profile(df, seed=None)   # a fresh sample each run
```

## Progress

Long runs can report where they are. Always to **stderr**, never stdout, so a
piped `summarize` stays parseable.

```python
from pysuricata import profile

profile("events.parquet", progress="auto")  # only when stderr is a terminal
profile("events.parquet", progress=True)    # always
```

`progress` also accepts a callable, taking chunks, rows and elapsed time.

## See Also

- [Generated reference](reference.md) — signatures, straight from the source
- [Configuration Guide](configuration.md) — full parameter reference
- [Basic Usage](usage.md) — more examples
- [CLI reference](cli.md) — the same three operations from a shell
- [Performance Tips](performance.md) — tuning for large datasets
