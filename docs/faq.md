---
title: Frequently Asked Questions
description: Common questions and answers about PySuricata
---

# Frequently Asked Questions

## General

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

**Required:** pandas, markdown, psutil, numpy (on Python ≥3.13)

**Optional:** polars (install with `pip install pysuricata[polars]`)

PySuricata requires Python 3.9+.

### Why is my installation failing?

Common issues:

1. **Python version** — PySuricata requires Python 3.9+:
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
from pysuricata import profile, ReportConfig

config = ReportConfig()
config.compute.columns = ["col1", "col2", "col3"]

report = profile(df, config=config)
```

### How do I make reports reproducible?

Set a random seed:

```python
config = ReportConfig()
config.compute.random_seed = 42

report = profile(df, config=config)
```

### Can I get statistics without generating HTML?

Yes, use `summarize()`:

```python
from pysuricata import summarize

stats = summarize(df)
print(stats["dataset"])
print(stats["columns"]["my_column"])
```

## Performance

### How much memory does PySuricata use?

Memory usage depends on configuration, not dataset size. The main factors are:

- **chunk_size** — rows held in memory per iteration (default: 200,000)
- **numeric_sample_size** — reservoir sample size per numeric column (default: 20,000)
- **uniques_sketch_size** — KMV sketch size per column (default: 2,048)

Processing a 10 GB dataset uses roughly the same memory as processing a 100 MB one.

### My report is slow. How can I speed it up?

Three quick changes:

```python
config = ReportConfig()
config.compute.compute_correlations = False    # Skip O(p²) correlation step
config.compute.numeric_sample_size = 10_000    # Smaller reservoir sample
config.compute.chunk_size = 500_000            # Fewer iterations
```

See [Performance Tips](performance.md) for more strategies.

### Can PySuricata handle very large datasets?

Yes, by passing a generator:

```python
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

Anything that can be loaded into pandas or polars: CSV, Parquet, JSON, Excel, SQL databases. Load into a DataFrame first, then pass it to `profile()`.

### How does PySuricata handle missing values?

Missing values are excluded from statistical calculations (mean, variance, etc.) and reported separately with count and percentage per column.

## Reports

### Why is my HTML report large?

Report size grows with the number of columns. Each column adds a variable card with statistics and an SVG chart. To reduce size, profile fewer columns or reduce `top_k_size`.

### Can I display reports in Jupyter?

```python
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
