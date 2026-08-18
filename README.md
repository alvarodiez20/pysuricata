# PySuricata

[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![Python versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/alvarodiez20/pysuricata/branch/main/graph/badge.svg)](https://codecov.io/gh/alvarodiez20/pysuricata)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://alvarodiez20.github.io/pysuricata/)
[![Downloads](https://static.pepy.tech/badge/pysuricata)](https://pepy.tech/project/pysuricata)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-alvarodiez20-blue?logo=linkedin)](https://www.linkedin.com/in/alvarodiez20/)

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/pysuricata/static/images/logo_suricata_transparent.png" alt="PySuricata Logo" width="300">

  <h3>Exploratory Data Analysis for Python, Built on Streaming Algorithms</h3>

  <p><strong>One pass over your data. A self-contained HTML report, a versioned JSON payload, or a CI gate: all from the same pass.</strong></p>

  <p>
    <a href="https://pysuricata.pages.dev"><strong>Live Demo</strong></a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/">Documentation</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/examples/">Examples</a>
  </p>
</div>

---

## See it before you install it

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/docs/assets/report-screenshot.png" alt="A PySuricata report: the dataset summary, the columns flagged as needing a look with the threshold each one crossed, and a numeric column card with its histogram and bin controls" width="900">
</div>

- **[Run it in your browser →](https://pysuricata.pages.dev)**: drop a CSV, Parquet file or Excel workbook and get the real report back. The profiler is compiled to WebAssembly and runs in the page, so **nothing is uploaded**.
- **[Open a finished report →](https://alvarodiez20.github.io/pysuricata/assets/example_report.html)**: two years of hourly bike rentals, as PySuricata renders it — all four card kinds, and correlations worth ranking.

## Quick Start

```bash
uv add pysuricata      # or: pip install pysuricata
```

```python
import pandas as pd
from pysuricata import profile

df = pd.read_csv("bike_sharing.csv")
profile(df).save_html("report.html")
```

That is the whole API for the common case. Optional extras:

```bash
uv add "pysuricata[polars]"   # polars.DataFrame and LazyFrame
uv add "pysuricata[system]"   # psutil-backed memory reporting
```

## Why PySuricata

**It reads your data once.**
Data is processed in chunks using streaming algorithms, so memory usage stays bounded **in the number of rows**: a million rows costs no more than twenty thousand. It is *not* bounded in the number of columns, which is a real limit rather than a footnote. See [Where the bound stops holding](#where-the-bound-stops-holding).

**It is not only a report.** The same pass gives three outputs: `profile()` for the HTML, `summarize()` for a versioned JSON payload with no markup in the way, and `pysuricata check` for a CI gate that exits non-zero when a threshold is crossed. Most profilers give you the first and stop.

**Arrow is the boundary, not pandas.** Anything exporting the Arrow C stream interface (`__arrow_c_stream__`) is profiled without materialising it, whatever library produced it. Arrow IPC is what R, Julia and Rust write, so a file from another runtime is read directly.

**Approximations say so.** Quantiles, distinct counts and duplicate estimates come from sketches. The report labels them and carries their error bound rather than printing an estimate as an exact integer.

**One file, no assets.** A report is a single HTML file with inline CSS, JS and SVG. It opens from a mail attachment on a machine with no network.

### Everything else

- **Streaming architecture**: data is processed in configurable chunks, keeping memory bounded in rows, though not in columns (see above). Useful for datasets with more rows than fit in RAM.
- **Pandas and Polars**: works natively with `pandas.DataFrame`, `polars.DataFrame` and `polars.LazyFrame`, plus Parquet files, Arrow IPC files (`.arrow`, `.feather`, `.ipc`), DuckDB relations and Arrow batches.
- **Configurable**: control chunk size, sample size, correlations and more with keyword options, a `preset=`, or a `ProfileConfig`.
- **Reproducible**: seeded random sampling produces deterministic results across runs.
- **Typed**: ships `py.typed`; `summarize()` returns a payload carrying a `schema_version`.
- **CLI tool**: `profile`, `summarize` and `check` from the command line.

## What's in a Report

Each column is analyzed based on its type:

- **Numeric**: mean, variance, skewness, kurtosis, quantiles, histogram, outlier detection (IQR, MAD, z-score), correlations
- **Categorical**: top values, distinct count, entropy, Gini impurity, string length statistics
- **DateTime**: temporal range, hour/day/month distributions, monotonicity detection
- **Boolean**: true/false counts and ratios, entropy

Plus dataset-level metrics: row/column counts, memory usage, missing value percentages, and duplicate row estimates.

---

The examples below assume a `df` in scope. The Quick Start frame works, or anything of your own:

<!-- docs-check:setup -->
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "age": rng.normal(30, 12, 800).round(1),
        "fare": rng.gamma(2, 20, 800).round(2),
        "sex": rng.choice(["male", "female"], 800),
        "booked": pd.date_range("2024-01-01", periods=800, freq="h"),
    }
)
```

## Statistics Only (No HTML)

Use `summarize()` for CI/CD quality checks. The payload carries a `schema_version` and is treated as a contract:

```python
from pysuricata import summarize

stats = summarize(df)

assert stats["schema_version"] == 2
assert stats["dataset"]["missing_cells_pct"] < 5.0

# Gate on the upper bound, not the point estimate. Below the sketch's own
# resolution `duplicate_rows_est` is suppressed to 0 -- a frame with no
# duplicates and one whose duplicates are merely unresolved both read 0,
# and a gate reading either alone would pass the second case by accident.
# `duplicate_rows_hi` is the same bound the HTML report prints either way,
# so this fails closed instead.
rows = stats["dataset"]["rows_est"]
duplicate_pct_hi = stats["dataset"]["duplicate_rows_hi"] / rows * 100 if rows else 0.0
assert duplicate_pct_hi < 1.0

print(f"Mean age: {stats['columns']['age']['mean']:.1f}")
```

## Streaming Large Datasets

Process datasets larger than RAM by passing a generator:

```python
import pandas as pd
from pysuricata import profile

def read_in_chunks():
    for i in range(100):
        yield pd.read_parquet(f"data/part-{i}.parquet")

report = profile(read_in_chunks())
report.save_html("large_report.html")
```

A Parquet path, an Arrow IPC file, a DuckDB relation or an Arrow source needs no generator at all. Hand it over and it is read a batch at a time, without ever existing as one frame:

```python
import duckdb
from pysuricata import profile

report = profile("data/events.parquet")

# Written by arrow::write_ipc_file() in R, Arrow.write() in Julia, or the
# arrow crate in Rust. The framing is read from the file, not its extension.
report = profile("data/events.arrow")

# A relation is a query that has not run yet, so a filtered join across
# several files is profiled without any of it being landed.
relation = duckdb.connect("warehouse.db").sql("SELECT * FROM events")
report = profile(relation)
```

Measured on a 4,000,000 × 6 frame written as a 180 MB Parquet file, above a 118 MB bare-import floor: **307 MB** for `profile(path)` against **581 MB** for `profile(pd.read_parquet(path))`.

The readers behind that (`stream_parquet`, `stream_ipc`, `stream_arrow` and `stream_duckdb`) are exported from `pysuricata.sources` for when you want the batches rather than a profile.

### Where the bound stops holding

Bounded memory is a claim about **rows**, not about columns. Each column keeps its own
sketches for the whole run and gets its own card in the report, so both memory and report
size grow linearly with the width of the frame.

Measured with `python -m benchmarks.columns` at 20,000 rows, taking the slope from 100
columns to 600: **1.2 MB of resident memory and 59 KB of report per column**. A
20,000 x 600 frame costs 797 MB and emits a 35 MB report, while a 1,000,000 x 14 frame
holds 1.2x the cells for 52 MB. Wide frames are the axis the streaming design does not
yet cover, tracked in [#207](https://github.com/alvarodiez20/pysuricata/issues/207).

## Comparing Two Datasets

`compare()` runs both through the same single pass and reports what moved:

```python
from pysuricata import compare

last_week, this_week = df.iloc[:400], df.iloc[400:]
diff = compare(last_week, this_week)

diff.schema.added                       # columns that appeared
diff.columns["fare"].median_shift_sigma # in baseline standard deviations
diff.to_dict()                          # JSON-safe, three sections
```

It reports every delta, whether or not it crosses a threshold, because it is a description and not a verdict. `pysuricata check` is the same arithmetic with a threshold and an exit code.

## Configuration

Pass keyword options for the common cases:

```python
from pysuricata import profile

report = profile(
    df,
    chunk_size=250_000,   # default 50_000
    sample=20_000,
    seed=42,
    correlations=True,
    title="My Analysis",
)
```

Or start from a preset, `"fast"` or `"thorough"`:

```python
from pysuricata import profile

report = profile(df, preset="fast")
```

For everything else, build a `ProfileConfig`. Keyword options and `config=` are mutually exclusive:

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()
config.compute.chunk_size = 250_000
config.compute.random_seed = 42
config.compute.corr_threshold = 0.5
config.render.title = "My Analysis"

report = profile(df, config=config)
```

See the [Configuration Guide](https://alvarodiez20.github.io/pysuricata/configuration/) for all options.

## CLI

```bash
# Generate an HTML report
pysuricata profile data.csv --output report.html

# Get JSON statistics
pysuricata summarize data.csv

# Compare against a stored baseline; exit non-zero when a threshold is crossed
pysuricata check data.csv --write-baseline baseline.json
pysuricata check data.csv --baseline baseline.json --max-missing-pct 5
```

`check` exits `0` on pass, `1` when a threshold is crossed, and `2` when the check could not run, so it drops into CI without a wrapper.

### GitHub Action

The same gate, as a step instead of a script:

```yaml
- uses: alvarodiez20/pysuricata@v1
  with:
    file: data.csv
    baseline: baseline.json
    max-missing-pct: 5
```

Every `pysuricata check` flag is an input; see [`action.yml`](action.yml).

## How It Works

PySuricata uses well-known streaming algorithms from the academic literature:

| Algorithm | Purpose | Time | Space |
|-----------|---------|------|-------|
| **Welford/Pébay** | Exact mean, variance, skewness, kurtosis | O(1) per value | O(1) |
| **KMV sketch** | Distinct count estimation (~2.2% error) | O(log k) per value | O(k) |
| **Misra-Gries** | Top-k frequent values | O(1) amortized | O(k) |
| **Reservoir sampling** | Uniform random sample for quantiles | O(1) per value | O(s) |

*k = `top_k` (default 50) for Misra-Gries; `max_uniques` (default 2048) sizes the KMV distinct-count sketch; s = sample size (`numeric_sample_size`, default 20 000)*

KMV's relative standard error is `1/sqrt(k - 2)`, which is where the ~2.2% comes from. Approximate values are labelled approximate in the report and carry their error bound rather than being printed as exact integers.

All statistics are computed in a **single pass** over the data.

## Documentation

- [Quick Start](https://alvarodiez20.github.io/pysuricata/quickstart/)
- [User Guide](https://alvarodiez20.github.io/pysuricata/usage/)
- [Configuration](https://alvarodiez20.github.io/pysuricata/configuration/)
- [Arrow, Parquet and DuckDB](https://alvarodiez20.github.io/pysuricata/data-sources/)
- [Command Line](https://alvarodiez20.github.io/pysuricata/cli/)
- [Gating CI on drift](https://alvarodiez20.github.io/pysuricata/data-checks/)
- [Comparing two datasets](https://alvarodiez20.github.io/pysuricata/comparing/)
- [API Reference](https://alvarodiez20.github.io/pysuricata/api/) · [generated reference](https://alvarodiez20.github.io/pysuricata/reference/)
- [The `summarize()` schema](https://alvarodiez20.github.io/pysuricata/summary-schema/)
- [Statistical Methods](https://alvarodiez20.github.io/pysuricata/stats/overview/)
- [Examples](https://alvarodiez20.github.io/pysuricata/examples/)

## Contributing

Contributions are welcome. See the [Contributing Guide](https://alvarodiez20.github.io/pysuricata/contributing/).

```bash
git clone https://github.com/alvarodiez20/pysuricata.git
cd pysuricata
uv sync --dev
uv run pytest
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built using algorithms from:

- Welford, B.P. (1962): streaming moments
- Pébay, P. (2008): parallel merging of moments
- Bar-Yossef, Z. et al. (2002): KMV distinct count estimation
- Misra, J. & Gries, D. (1982): streaming heavy hitters

Named after **suricatas (meerkats)**: small, vigilant animals that work cooperatively and thrive in harsh environments with limited resources.
