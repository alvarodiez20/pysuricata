---
title: Frequently Asked Questions
description: Common questions and answers about PySuricata
---

# Frequently Asked Questions

## General

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

### What is PySuricata?

PySuricata is a Python library for exploratory data analysis that generates self-contained HTML reports. It uses streaming algorithms to process data in chunks, keeping memory bounded regardless of dataset size.

### Is PySuricata production-ready?

PySuricata is actively maintained with CI/CD, test coverage tracked via Codecov, and regular releases on PyPI. That said, evaluate it against your own requirements — it's still a young project.

## Installation

### How do I install PySuricata?

```bash
pip install pysuricata
```

With polars support:

```bash
pip install pysuricata[polars]
```

### What are the dependencies?

**Required:** pandas, markdown, numpy (on Python ≥3.13)

**Optional:** polars (`pip install pysuricata[polars]`), psutil (`pip install
pysuricata[system]`, only needed for the memory-measurement recipes in
[Performance](performance.md) — nothing in the library imports it)

PySuricata requires Python 3.10+.

### Why is my installation failing?

Common issues:

1. **Python version** — PySuricata requires Python 3.10+:
   ```bash
   python --version
   ```

2. **Conflicting packages** — Try a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install pysuricata
   ```

## Usage

### How do I generate a report?

```python
from pysuricata import profile

report = profile(df)
report.save_html("report.html")
```

### Can I profile only specific columns?

Yes:

```python
from pysuricata import profile

report = profile(df, columns=["id", "amount", "country"])
```

### How do I make reports reproducible?

They already are. `random_seed` defaults to `0`, so the same data produces the
same report — re-running is a no-op rather than a set of sampling wobbles.

Pass `seed=` to pin a different sample, or `seed=None` for a fresh one each run:

```python
from pysuricata import profile

report = profile(df, seed=42)
```

### Can I get statistics without generating HTML?

Yes, use `summarize()`:

```python
import pandas as pd

from pysuricata import summarize

df = pd.DataFrame({"my_column": [1.0, 2.0, 3.0, None]})
stats = summarize(df)
print(stats["dataset"])
print(stats["columns"]["my_column"])
```

## Performance

### How much memory does PySuricata use?

Memory usage depends on configuration, not dataset size. The main factors are:

- **chunk_size** — rows held in memory per iteration (default: 50,000)
- **numeric_sample_size** — reservoir sample size per numeric column (default: 20,000)
- **max_uniques** — KMV sketch size per column (default: 2,048)

Processing a 10 GB dataset uses roughly the same memory as processing a 100 MB
one — **in rows**.

In columns it is not bounded: state is per column, at roughly 1.2 MB each, so a
20,000 x 600 frame costs 797 MB against 52 MB for a 1,000,000 x 14 one on more
cells. That is
a known limit, tracked in
[#207](https://github.com/alvarodiez20/pysuricata/issues/207), and worth knowing
before you point it at a very wide frame.

### My report is slow. How can I speed it up?

Start with the preset:

```python
from pysuricata import profile

report = profile(df, preset="fast")
```

That turns down the sample size, the sketches and top-k, and skips the
\(O(p^2)\) correlation step. If you only need the numbers, `summarize()` skips
rendering altogether.

Do **not** raise `chunk_size` to go faster. The sketch merges are superlinear in
batch size, so a bigger batch costs more memory *and* more time. See
[Performance Tips](performance.md).

### Can PySuricata handle very large datasets?

Yes. The cheapest way is to hand over the path and let it stream:

```python
from pysuricata import profile

profile("events.parquet")
```

Or pass a generator, when the chunks are yours to produce:

```python
from pysuricata import profile
def read_large_dataset():
    for file in large_files:
        yield pd.read_parquet(file)

report = profile(read_large_dataset())
```

Memory stays bounded because only one chunk is in memory at a time.

### Why are correlations slow?

Correlation computation is O(p²) where p is the number of numeric columns. For datasets with many numeric columns, either disable correlations or increase `corr_threshold` to reduce the number reported.

## Technical

### Are the statistics exact or approximate?

**Exact:**

- Mean, variance, skewness, kurtosis (Welford/Pébay algorithms)
- Min, max, count

**Approximate:**

- Distinct count — KMV sketch, ~2.2% error with default k=2048
- Top-k values — Misra-Gries, guaranteed to find all items with frequency > n/k
- Quantiles — computed from a reservoir sample

### What algorithms does PySuricata use?

| Algorithm | Purpose | Reference |
|-----------|---------|-----------|
| Welford/Pébay | Exact streaming moments | Welford (1962), Pébay (2008) |
| KMV sketch | Distinct count estimation | Bar-Yossef et al. (2002) |
| Misra-Gries | Top-k frequent values | Misra & Gries (1982) |
| Reservoir sampling | Uniform random sample | Vitter (1985) |

See [Statistical Methods](stats/overview.md) for details.

### Does PySuricata support distributed computing?

Accumulators are mergeable — you can process data on separate machines and combine results. However, PySuricata doesn't include built-in distribution; you'd need to use an external framework.

## Data

### Does PySuricata modify my data?

No. PySuricata only reads data, never modifies it.

### What data formats are supported?

`profile()` and `summarize()` take more than a frame:

| input | note |
|---|---|
| `pandas.DataFrame` | |
| `polars.DataFrame` / `LazyFrame` | needs `pysuricata[polars]` |
| a **file path** | `.csv`, `.parquet`, `.json`, `.arrow`, `.feather`, `.ipc`, `.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods` |
| an **Arrow** table or reader | or anything exporting `__arrow_c_stream__` |
| a **DuckDB relation** | a query that has not run yet |
| an iterable of frames | your own chunks, from anywhere |

The middle two are read a **batch at a time** and never exist as one frame, so
handing over the path costs less than loading it yourself — 307 MB against
581 MB on a 180 MB Parquet file:

```python
profile("events.parquet")            # streamed
profile(pd.read_parquet("events.parquet"))  # loaded first
```

A spreadsheet is the exception among file paths: no engine puts a `chunksize`
on `read_excel`, so a workbook is always built whole first, and only the
first sheet is read. `python-calamine` covers all five spreadsheet formats
with one dependency; without it, pandas falls back to openpyxl, xlrd, pyxlsb
or odfpy depending on the extension.

Anything else — a SQL cursor, a Dask partition — goes through pandas or
polars first. See [Arrow, Parquet and DuckDB](data-sources.md).

### Is there a command line?

Yes, three subcommands:

```bash
pysuricata profile data.csv --output report.html
pysuricata summarize data.csv --output stats.json
pysuricata check data.parquet --baseline baseline.json
```

`check` is the one worth knowing about: it compares a dataset against a stored
baseline and **exits non-zero** when a threshold is crossed, so the same single
pass runs in a notebook and in CI. See [the CLI reference](cli.md) and
[Gating CI on drift](data-checks.md).

### How do I see what changed between two datasets?

`compare(a, b)` reports every delta — schema, dataset and per column — as a
description rather than a verdict:

```python
import numpy as np
import pandas as pd

from pysuricata import compare

rng = np.random.default_rng(0)
last_month = pd.DataFrame({"amount": rng.lognormal(3, 1.0, 2_000)})
this_month = pd.DataFrame({"amount": rng.lognormal(3.4, 1.0, 2_000)})

diff = compare(last_month, this_month)
diff.columns["amount"].median_shift_sigma
```

See [Comparing two datasets](comparing.md).

### How does PySuricata handle missing values?

Missing values are excluded from statistical calculations (mean, variance, etc.) and reported separately with count and percentage per column.

## Reports

### Why is my HTML report large?

Report size grows with the number of columns. Each column adds a variable card with statistics and an SVG chart. To reduce size, profile fewer columns or reduce `top_k`.

### Can I display reports in Jupyter?

```python
from pysuricata import profile
report = profile(df)
report  # Auto-displays inline

# Or with custom height
report.display_in_notebook(height="800px")
```

### Can I export to PDF?

Not built-in. You can print the HTML report to PDF from your browser, or use a tool like `wkhtmltopdf`.

## Getting Help

- [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- [GitHub Issues](https://github.com/alvarodiez20/pysuricata/issues)

Still have questions? Ask in [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions).
