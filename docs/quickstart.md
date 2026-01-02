# Quick Start

Get started with PySuricata in less than 5 minutes!

## Installation

Install pysuricata from PyPI:

```bash
pip install pysuricata
```

For polars support (optional):

```bash
pip install pysuricata polars
```

## Command Line Usage

The fastest way to profile a dataset:

```bash
# Generate an HTML report
pysuricata profile data.csv --output report.html

# Get JSON statistics (no HTML)
pysuricata summarize data.csv --output stats.json

# With options
pysuricata profile data.csv -o report.html --seed 42 --no-correlations
```

See `pysuricata --help` for all options.

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
# Create the profile report
report = profile(df)

# Save to HTML
report.save_html("iris_report.html")
```

That's it! Open `iris_report.html` in your browser to see a comprehensive analysis.

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
# Generate statistics only
from pysuricata import summarize

stats = summarize(df)
print(stats["dataset"])  # Dataset-level metrics
print(stats["columns"]["sepal_length"])  # Per-column stats

# Or save from report
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
from pysuricata import profile, ReportConfig

# Select specific columns
config = ReportConfig()
config.compute.columns = ["sepal_length", "sepal_width", "species"]

report = profile(df, config=config)
```

### Reproducible Reports

```python
from pysuricata import profile, ReportConfig

# Set random seed for deterministic sampling
config = ReportConfig()
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
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Adjust chunk size for memory management
config.compute.chunk_size = 200_000  # Default

# Control sample sizes
config.compute.numeric_sample_size = 20_000  # For quantiles/histograms
config.compute.uniques_sketch_size = 2_048   # For distinct counts
config.compute.top_k_size = 50               # For top values

# Enable/disable correlations
config.compute.compute_correlations = True
config.compute.corr_threshold = 0.5

# Deterministic sampling
config.compute.random_seed = 42

# Generate report
report = profile(df, config=config)
```

## Performance Tips

### For Large Datasets (> 1 GB)

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()
config.compute.chunk_size = 250_000  # Larger chunks
config.compute.numeric_sample_size = 10_000  # Smaller samples
config.compute.compute_correlations = False  # Skip if not needed

report = profile(df, config=config)
```

### For Memory-Constrained Environments

```python
config = ReportConfig()
config.compute.chunk_size = 50_000  # Smaller chunks
config.compute.numeric_sample_size = 5_000  # Smaller samples
config.compute.uniques_sketch_size = 1_024  # Smaller sketches

report = profile(df, config=config)
```

### For Speed

```python
config = ReportConfig()
config.compute.compute_correlations = False  # Skip correlations
config.compute.top_k_size = 20  # Fewer top values

report = profile(df, config=config)
```

## Next Steps

Now that you've created your first report, explore:

- **[Basic Usage](usage.md)** - Detailed usage patterns
- **[Configuration](configuration.md)** - All configuration options
- **[Performance Tips](performance.md)** - Optimize for your use case
- **[Examples Gallery](examples.md)** - More real-world examples
- **[Statistical Methods](stats/overview.md)** - Understand the algorithms

## Troubleshooting

### Report is too large
- Reduce `numeric_sample_size`
- Skip correlations: `config.compute.compute_correlations = False`
- Profile fewer columns: `config.compute.columns = ["col1", "col2"]`

### Out of memory
- Reduce `chunk_size`
- Reduce all sample sizes
- Process columns separately

### Report takes too long
- Increase `chunk_size` (if memory allows)
- Disable correlations
- Reduce `top_k_size`

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




