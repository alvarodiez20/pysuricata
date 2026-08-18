# Quick Start

Get started with PySuricata in less than 5 minutes!

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

## Installation

Install pysuricata from PyPI:

=== "uv (recommended)"
    ```bash
    uv add pysuricata
    ```

=== "pip"
    ```bash
    pip install pysuricata
    ```

For optional polars support:

=== "uv (recommended)"
    ```bash
    uv add pysuricata[polars]
    ```

=== "pip"
    ```bash
    pip install pysuricata[polars]
    ```

## Command Line Usage

The fastest way to profile a dataset — no script needed:

```bash
# Generate an HTML report
pysuricata profile data.csv --output report.html

# Get JSON statistics (no HTML)
pysuricata summarize data.csv --output stats.json

# Gate a build on drift, with an exit code
pysuricata check data.parquet --baseline baseline.json
```

Every option is in the [CLI reference](cli.md); `check` in particular has a
guide of its own in [Gating CI on drift](data-checks.md).

## Your First Report

### Step 1: Import and Load Data

=== "Pandas"
    ```python
    import pandas as pd
    from pysuricata import profile

    # Load a dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    ```

=== "Polars"
    ```python
    import polars as pl
    from pysuricata import profile

    # Load a dataset
    df = pl.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    ```

=== "From URL"
    ```python
    import pandas as pd
    from pysuricata import profile

    # Load directly from URL
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    ```

### Step 2: Generate Report

```python
from pysuricata import profile
# Create the profile report
report = profile(df)

# Save to HTML
report.save_html("iris_report.html")
```

That's it! Open `iris_report.html` in your browser to see a comprehensive analysis.

You did not have to configure anything, and you will not have to for most
frames. When you do, the seven most-reached-for settings are keywords and there
are two presets:

```python
from pysuricata import profile

report = profile(df, preset="fast", title="Iris")
```

See [Configuration](configuration.md).

### Skip the load

`profile()` takes a path too, and reads it a batch at a time rather than
building a frame first:

```python
from pysuricata import profile

report = profile("data.parquet")
```

## Understanding Your Report

The generated report contains several sections:

### 1. Dataset Overview
- Number of rows and columns
- Memory usage (approximate)
- Missing values summary
- Duplicate rows estimate
- Processing time

### 2. Variables Section

For each variable, you'll see:

#### Numeric Variables
- Count, missing percentage
- Mean, median, standard deviation
- Min, max, range
- Quantiles (Q1, Q2, Q3)
- Skewness and kurtosis
- Histogram visualization

#### Categorical Variables
- Count, missing percentage
- Number of unique values
- Top values with frequencies
- Diversity metrics

#### DateTime Variables
- Temporal range (min/max)
- Distribution by hour, day of week, month
- Timeline visualization

#### Boolean Variables
- True/False counts
- Balance ratio
- Missing percentage

### 3. Correlations (for numeric columns)
- Top correlations for each numeric variable
- Correlation strength indicators

## Working with Your Data

### Save as JSON (for programmatic access)

```python
import pandas as pd

from pysuricata import summarize

df = pd.DataFrame({"sepal_length": [5.1, 4.9, 6.2], "species": ["a", "a", "b"]})

# Generate statistics only -- no HTML is rendered
stats = summarize(df)
print(stats["dataset"])  # Dataset-level metrics
print(stats["columns"]["sepal_length"])  # Per-column stats
```

Or, from a report you already built:

```python
report.save_json("iris_stats.json")
```

### Display in Jupyter Notebook

```python
# In a Jupyter notebook
from pysuricata import profile

report = profile(df)
report  # Automatically displays inline
```

For better display in notebooks:

```python
report.display_in_notebook(height="800px")
```

## Common Use Cases

### Quick Data Quality Check

```python
from pysuricata import summarize

stats = summarize(df)

# Check data quality metrics
print(f"Missing cells: {stats['dataset']['missing_cells_pct']:.2f}%")
print(f"Duplicate rows: {stats['dataset']['duplicate_rows_pct_est']:.2f}%")

# Assert quality requirements
assert stats['dataset']['missing_cells_pct'] < 5.0, "Too many missing values"
```

### Profile Specific Columns

```python
from pysuricata import profile, ProfileConfig

# Select specific columns
config = ProfileConfig()
config.compute.columns = ["sepal_length", "sepal_width", "species"]

report = profile(df, config=config)
```

### Reproducible Reports

```python
from pysuricata import profile, ProfileConfig

# Set random seed for deterministic sampling
config = ProfileConfig()
config.compute.random_seed = 42

report = profile(df, config=config)
# Same report every time!
```

### Process Large Files in Chunks

```python
import pandas as pd
from pysuricata import profile

# Read and process in chunks
def read_chunks():
    for chunk in pd.read_csv("large_file.csv", chunksize=100_000):
        yield chunk

report = profile(read_chunks())
report.save_html("large_file_report.html")
```

## Configuration Basics

PySuricata is highly configurable. Here are some common settings:

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()

# Adjust chunk size to trade memory against chunk boundaries -- not for speed
config.compute.chunk_size = 50_000  # Default

# Control sample sizes
config.compute.numeric_sample_size = 20_000  # For quantiles/histograms
config.compute.max_uniques = 2_048            # For distinct counts
config.compute.top_k = 50                     # For top values

# Enable/disable correlations
config.compute.compute_correlations = True
config.compute.corr_threshold = 0.5

# Deterministic sampling (already the default -- this pins a different sample)
config.compute.random_seed = 42

# Generate report
report = profile(df, config=config)
```

## Performance Tips

### For Large Datasets (> 1 GB)

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()
config.compute.chunk_size = 100_000  # near the top of the useful range
config.compute.numeric_sample_size = 10_000  # Smaller samples
config.compute.compute_correlations = False  # Skip if not needed

report = profile(df, config=config)
```

### For Memory-Constrained Environments

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.chunk_size = 50_000  # Smaller chunks
config.compute.numeric_sample_size = 5_000  # Smaller samples
config.compute.max_uniques = 1_024  # Smaller sketches

report = profile(df, config=config)
```

### For Speed

```python
from pysuricata import profile

report = profile(df, preset="fast")
```

## Next Steps

Now that you've created your first report, explore:

- **[Basic Usage](usage.md)** - Detailed usage patterns
- **[Configuration](configuration.md)** - All configuration options
- **[Performance Tips](performance.md)** - Optimize for your use case
- **[Examples Gallery](examples.md)** - More real-world examples
- **[Statistical Methods](stats/overview.md)** - Understand the algorithms
- **[Command Line](cli.md)** - `profile`, `summarize` and `check` from a shell
- **[Gating CI on drift](data-checks.md)** - fail a build when the data moves

## Troubleshooting

### Report is too large
Size grows with the number of **columns**, not rows — each column contributes a
card with its own SVG charts.

- Profile fewer columns: `profile(df, columns=["col1", "col2"])`
- Reduce `top_k`, which sets how many bars a categorical card draws
- Skip correlations: `profile(df, correlations=False)`
- Drop the sample table: `config.render.include_sample = False`

### Out of memory
- Hand over the path rather than a loaded frame, so the file streams
- Reduce `chunk_size`
- Reduce `numeric_sample_size` and `max_uniques`
- Memory is bounded in rows but grows about 3 MB per column, so a very wide
  frame is the case to watch
  ([#207](https://github.com/alvarodiez20/pysuricata/issues/207))

### Report takes too long
- `profile(df, preset="fast")`
- Hand over the path instead of a loaded frame, so the file streams
- Use `summarize()` if you do not need the HTML
- Do **not** raise `chunk_size` — it costs memory and time both

### Want more decimal places
```python
# Not currently configurable, but stats JSON has full precision
report.save_json("stats.json")
```

## Get Help

- 📖 [Full Documentation](index.md)
- 💬 [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- 🐛 [Report Issues](https://github.com/alvarodiez20/pysuricata/issues)
- 📧 [Contact Maintainer](mailto:alvarodiez20@gmail.com)

---

Ready for more advanced features? Check out the [Advanced Guide](advanced.md).
