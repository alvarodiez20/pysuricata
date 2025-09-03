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
- **Out-of-core option (v2)**: Stream CSV/Parquet in chunks and profile datasets larger than RAM.
- **No heavy deps**: Minimal runtime dependencies (pandas/pyarrow optional depending on source).
- **Rich insights**: Summaries for numeric, categorical, datetime columns, missing values, duplicates, correlations, and sample rows.
- **Portable**: Reports are standalone HTML (with inline CSS/JS/images) that can be easily shared.
- **Customizable**: Title, sample display, and output path can be tailored to your needs.

## Quick Example (classic, in-memory DataFrame)

The following example demonstrates how to generate an EDA report using the Iris dataset with Pandas:


```python
import pandas as pd
import pysuricata
from IPython.display import HTML

# Load the Iris dataset directly using Pandas
iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(iris_url)

# Generate the HTML EDA report and save it to a file
html_report = pysuricata.generate_report(iris_df, output_file="iris_report.html")

# Display the report in a Jupyter Notebook
HTML(html_report)
```

## Out-of-core streaming report (v2)

For large CSV/Parquet files, use the streaming generator in `report_v2`.

```python
from pysuricata.report_v2 import generate_report, ReportConfig

# From file path (CSV/Parquet)
html = generate_report(
    
    source="/path/to/big.parquet",  # or .csv
    config=ReportConfig(chunk_size=250_000, compute_correlations=True),
    output_file="report_big.html",
)

# Or from a DataFrame (single chunk)
import pandas as pd
df = pd.read_csv("data.csv")
html = generate_report(df)

# Optional: get a programmatic JSON-like summary too
html, summary = generate_report("/path/to/big.csv", return_summary=True)
```

Highlights in v2:
- Streams data in chunks, low peak memory.
- Shows processed bytes (≈) and precise generation time (e.g., 0.02s).
- Approximate distinct (KMV), heavy hitters (Misra–Gries), quantiles/histograms via reservoir sampling.
- Numeric extras: 95% CI for mean, coefficient of variation, heaping %, granularity hints, bimodality.
- Categorical extras: case/trim variants, empty strings, length stats.
- Datetime details: per-hour, day-of-week, and month breakdown tables + timeline chart.
- Correlation chips (streaming) for numeric columns.
- Hardened HTML escaping for column names and labels.

## What’s New

- Out-of-core `report_v2` with CSV/Parquet chunking (pandas/pyarrow backends).
- Processed bytes displayed in Summary and per-variable cards.
- Precise duration in header (e.g., “0.02s”).
- Removed “Likely ID” flag to reduce false positives.
- Datetime Details section with human-readable breakdown tables.
- Numeric extremes now show row IDs (tracked across chunks).
- Optional `(html, summary)` return for programmatic consumption.
