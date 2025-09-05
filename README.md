# `pysuricata`
[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/pysuricata/static/images/logo_suricata_transparent.png" alt="pysuricata Logo" width="300">
</div>



A lightweight Python library to generate self-contained HTML reports for exploratory data analysis (EDA).

📖 [Read the documentation](https://alvarodiez20.github.io/pysuricata/)


## Installation

Install `pysuricata` directly from PyPI:

```bash
pip install pysuricata
```

## Why use pysuricata?
- **Instant reports**: Generate clean, self-contained HTML reports directly from pandas DataFrames.
- **Out-of-core option (v2)**: Consume in-memory DataFrame chunks and profile datasets larger than RAM.
- **No heavy deps**: Minimal runtime dependencies (pandas/pyarrow optional depending on source).
- **Rich insights**: Summaries for numeric, categorical, datetime columns, missing values, duplicates, correlations, and sample rows.
- **Portable**: Reports are standalone HTML (with inline CSS/JS/images) that can be easily shared.
- **Customizable**: Title, sample display, and output path can be tailored to your needs.

## Quick Example (classic, in-memory DataFrame)

The following example demonstrates how to generate an EDA report using the Iris dataset with Pandas:

```python
import pandas as pd
from pysuricata import profile

# Load the Iris dataset directly using Pandas
iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(iris_url)

# Build the report and save to a file
rep = profile(iris_df)
rep.save_html("iris_report.html")
```

## Streaming report (low memory)

For large datasets, stream in-memory DataFrame chunks you control.

```python
from pysuricata import profile, ReportConfig
import pandas as pd

def chunk_iter():
    for i in range(10):
        yield pd.read_csv(f"part-{i}.csv")  # You manage chunking externally

rep = profile((ch for ch in chunk_iter()), config=ReportConfig())
rep.save_html("report.html")

# Optional: stats-only
from pysuricata import summarize
stats = summarize(iris_df)
```

Highlights:

- Streams data in chunks, low peak memory.
- Shows processed bytes (≈) and precise generation time (e.g., 0.02s).
- Approximate distinct (KMV), heavy hitters (Misra–Gries), quantiles/histograms via reservoir sampling.
- Numeric extras: 95% CI for mean, coefficient of variation, heaping %, granularity hints, bimodality.
- Categorical extras: case/trim variants, empty strings, length stats.
- Datetime details: per-hour, day-of-week, and month breakdown tables + timeline chart.
- Correlation chips (streaming) for numeric columns.
- Hardened HTML escaping for column names and labels.

