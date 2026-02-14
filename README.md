# PySuricata

[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![Python versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/alvarodiez20/pysuricata/branch/main/graph/badge.svg)](https://codecov.io/gh/alvarodiez20/pysuricata)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://alvarodiez20.github.io/pysuricata/)
[![Downloads](https://static.pepy.tech/badge/pysuricata)](https://pepy.tech/project/pysuricata)

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/pysuricata/static/images/logo_suricata_transparent.png" alt="PySuricata Logo" width="300">
  
  <h3>Exploratory Data Analysis for Python, Built on Streaming Algorithms</h3>
  
  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/">Documentation</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/examples/">Examples</a>
  </p>
</div>

---

## What It Does

PySuricata generates **self-contained HTML reports** from pandas or polars DataFrames. Reports include per-column statistics, histograms, correlation chips, missing value analysis, and outlier detection.

Data is processed in chunks using streaming algorithms, so memory usage stays bounded regardless of dataset size.

## Quick Start

### Installation

```bash
pip install pysuricata
```

With polars support:

```bash
pip install pysuricata[polars]
```

### Generate a Report

```python
import pandas as pd
from pysuricata import profile

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

report = profile(df)
report.save_html("titanic_report.html")
```

### Example Report

This is a real report generated from the Titanic dataset (891 rows × 12 columns):

<div align="center">
  <a href="https://alvarodiez20.github.io/pysuricata/assets/titanic_report.html">
    <strong>▶ View the live interactive Titanic report →</strong>
  </a>
</div>

## Features

- **Streaming architecture** — Data is processed in configurable chunks, keeping memory bounded. Useful for datasets that don't fit in RAM.
- **Pandas and Polars** — Works natively with `pandas.DataFrame`, `polars.DataFrame`, and `polars.LazyFrame`.
- **Self-contained HTML** — Single file with inline CSS, JS, and SVG charts. No external assets needed.
- **Configurable** — Control chunk sizes, sample sizes, sketch parameters, and correlation thresholds via `ReportConfig`.
- **Reproducible** — Seeded random sampling produces deterministic results across runs.
- **CLI tool** — Profile datasets from the command line.

## How It Works

PySuricata uses well-known streaming algorithms from the academic literature:

| Algorithm | Purpose | Complexity |
|-----------|---------|------------|
| **Welford/Pébay** | Exact mean, variance, skewness, kurtosis | O(1) per value, mergeable |
| **KMV sketch** | Distinct count estimation | O(log k) per value, ~2.2% error |
| **Misra-Gries** | Top-k frequent values | O(1) amortized, guaranteed |
| **Reservoir sampling** | Uniform random sample for quantiles | O(1) per value, exact probability |

All statistics are computed in a **single pass** over the data.

## What's in a Report

Each column is analyzed based on its type:

- **Numeric** — Mean, variance, skewness, kurtosis, quantiles, histogram, outlier detection (IQR, MAD, z-score), correlations
- **Categorical** — Top values, distinct count, entropy, Gini impurity, string length statistics
- **DateTime** — Temporal range, hour/day/month distributions, monotonicity detection
- **Boolean** — True/false ratios, entropy, balance score

Plus dataset-level metrics: row/column counts, memory usage, missing value percentages, and duplicate row estimates.

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

## Statistics Only (No HTML)

Use `summarize()` for CI/CD quality checks:

```python
from pysuricata import summarize

stats = summarize(df)

assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0

print(f"Mean age: {stats['columns']['age']['mean']:.1f}")
```

## Configuration

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()
config.compute.chunk_size = 250_000
config.compute.random_seed = 42
config.compute.compute_correlations = True
config.compute.corr_threshold = 0.5
config.render.title = "My Analysis"

report = profile(df, config=config)
```

See the [Configuration Guide](https://alvarodiez20.github.io/pysuricata/configuration/) for all options.

## CLI

```bash
# Generate HTML report
pysuricata profile data.csv --output report.html

# Get JSON statistics
pysuricata summarize data.csv
```

## Documentation

- [Quick Start](https://alvarodiez20.github.io/pysuricata/quickstart/)
- [User Guide](https://alvarodiez20.github.io/pysuricata/usage/)
- [Configuration](https://alvarodiez20.github.io/pysuricata/configuration/)
- [API Reference](https://alvarodiez20.github.io/pysuricata/api/)
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

- Welford, B.P. (1962) — Streaming moments
- Pébay, P. (2008) — Parallel merging of moments
- Bar-Yossef, Z. et al. (2002) — KMV distinct count estimation
- Misra, J. & Gries, D. (1982) — Streaming heavy hitters

Named after **suricatas (meerkats)** — small, vigilant animals that work cooperatively and thrive in harsh environments with limited resources.
