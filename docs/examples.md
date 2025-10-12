---
title: Examples Gallery
description: Comprehensive examples and use cases for PySuricata
---

# Examples Gallery

Real-world examples showing how to use PySuricata in various scenarios.

## Small Dataset (Iris)

Classic machine learning dataset with 150 rows × 5 columns.

```python
import pandas as pd
from pysuricata import profile

# Load Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Generate report
report = profile(df)
report.save_html("iris_report.html")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
# Output: Rows: 150, Columns: 5
```

**Expected output:**
- 4 numeric variables (sepal/petal dimensions)
- 1 categorical variable (species)
- No missing values
- Strong correlations between dimensions

## Medium Dataset (Titanic)

Popular dataset with mixed types and missing values.

```python
import pandas as pd
from pysuricata import profile

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Generate report
report = profile(df)
report.save_html("titanic_report.html")
```

**Features:**
- 891 rows × 12 columns
- Numeric: age, fare, siblings/spouses
- Categorical: name, ticket, cabin, embarked
- Boolean: survived
- Missing values in age (~20%), cabin (~77%)

## Large Dataset (Streaming)

Process multi-GB dataset in bounded memory.

```python
import pandas as pd
from pysuricata import profile, ReportConfig

def read_large_dataset():
    """Generator yielding chunks"""
    for i in range(100):
        yield pd.read_parquet(f"data/part-{i}.parquet")

# Configure for large data
config = ReportConfig()
config.compute.chunk_size = 250_000
config.compute.numeric_sample_size = 50_000
config.compute.random_seed = 42

# Profile
report = profile(read_large_dataset(), config=config)
report.save_html("large_dataset_report.html")
```

## Wide Dataset (Many Columns)

Handle datasets with hundreds of columns.

```python
import pandas as pd
import numpy as np
from pysuricata import profile, ReportConfig

# Create wide dataset
n_rows, n_cols = 10_000, 500
df = pd.DataFrame(
    np.random.randn(n_rows, n_cols),
    columns=[f"feature_{i}" for i in range(n_cols)]
)

# Disable correlations (too expensive for 500 columns)
config = ReportConfig()
config.compute.compute_correlations = False

report = profile(df, config=config)
report.save_html("wide_dataset_report.html")
```

**Note:** For \(p > 100\) columns, correlation computation is O(p²) and may be slow.

## Time Series Data

Analyze temporal patterns in datetime columns.

```python
import pandas as pd
import numpy as np
from pysuricata import profile

# Generate time series
dates = pd.date_range("2023-01-01", periods=10_000, freq="H")
df = pd.DataFrame({
    "timestamp": dates,
    "value": np.random.randn(10_000).cumsum(),
    "category": np.random.choice(["A", "B", "C"], 10_000)
})

report = profile(df)
report.save_html("timeseries_report.html")
```

**Analysis includes:**
- Hour-of-day distribution
- Day-of-week pattern
- Month distribution
- Monotonicity detection

## High Missing Values

Dataset with significant missing data.

```python
import pandas as pd
import numpy as np
from pysuricata import profile

# Create dataset with missing values
df = pd.DataFrame({
    "always_present": range(1000),
    "mostly_present": [i if i % 10 != 0 else None for i in range(1000)],  # 10% missing
    "half_missing": [i if i % 2 == 0 else None for i in range(1000)],     # 50% missing
    "mostly_missing": [i if i % 10 == 0 else None for i in range(1000)],  # 90% missing
})

report = profile(df)
report.save_html("missing_values_report.html")
```

**Report highlights:**
- Missing percentage per column
- Chunk-level distribution
- Data completeness visualizations

## All Categorical

Text-heavy dataset (e.g., customer feedback).

```python
import pandas as pd
from pysuricata import profile

df = pd.DataFrame({
    "customer_id": [f"CUST_{i:05d}" for i in range(10_000)],
    "product": np.random.choice(["Product A", "Product B", "Product C"], 10_000),
    "rating": np.random.choice(["Poor", "Fair", "Good", "Excellent"], 10_000),
    "feedback": [f"Comment {i}" for i in range(10_000)],
})

report = profile(df)
report.save_html("categorical_report.html")
```

**Analysis includes:**
- Top values and frequencies
- Distinct counts (KMV sketch)
- String length statistics
- Entropy and Gini metrics

## Polars DataFrame

Use polars instead of pandas.

```python
import polars as pl
from pysuricata import profile

# Create polars DataFrame
df = pl.DataFrame({
    "id": range(1000),
    "value": [float(i) for i in range(1000)],
    "category": ["A"] * 500 + ["B"] * 500
})

# Profile works natively with polars
report = profile(df)
report.save_html("polars_report.html")
```

## Polars LazyFrame

Streaming evaluation with polars.

```python
import polars as pl
from pysuricata import profile, ReportConfig

# Create lazy frame
lf = pl.scan_csv("large_file.csv").filter(pl.col("value") > 0)

# Configure chunk size
config = ReportConfig()
config.compute.chunk_size = 50_000

# Profile lazily evaluated data
report = profile(lf, config=config)
report.save_html("polars_lazy_report.html")
```

## Jupyter Notebook Integration

Display reports inline in notebooks.

```python
import pandas as pd
from pysuricata import profile

df = pd.read_csv("data.csv")
report = profile(df)

# Display inline (automatic)
report

# Or with custom size
report.display_in_notebook(height="800px")
```

## Programmatic Access (Stats Only)

Use for data quality checks without HTML.

```python
from pysuricata import summarize

# Get statistics only (faster than full report)
stats = summarize(df)

# Check data quality
print(f"Rows: {stats['dataset']['rows']}")
print(f"Missing cells: {stats['dataset']['missing_cells_pct']:.1f}%")
print(f"Duplicate rows: {stats['dataset']['duplicate_rows_pct_est']:.1f}%")

# Per-column stats
for col, col_stats in stats["columns"].items():
    if col_stats.get("missing_pct", 0) > 10:
        print(f"{col}: {col_stats['missing_pct']:.1f}% missing")
```

## CI/CD Data Quality Gates

Enforce quality thresholds in pipelines.

```python
from pysuricata import summarize

def validate_data_quality(df):
    """Validate data quality, raise if fails"""
    stats = summarize(df)
    
    # Check missing data
    missing_pct = stats["dataset"]["missing_cells_pct"]
    assert missing_pct < 5.0, f"Too many missing values: {missing_pct:.1f}%"
    
    # Check duplicates
    dup_pct = stats["dataset"]["duplicate_rows_pct_est"]
    assert dup_pct < 1.0, f"Too many duplicates: {dup_pct:.1f}%"
    
    # Check specific columns
    for col in ["customer_id", "transaction_id"]:
        col_stats = stats["columns"][col]
        assert col_stats["distinct"] == col_stats["count"], \
            f"{col} has duplicates"
    
    print("✓ Data quality checks passed")

# Use in pipeline
validate_data_quality(df)
```

## Custom Column Selection

Profile only specific columns.

```python
from pysuricata import profile, ReportConfig

# Large dataset, only analyze key columns
config = ReportConfig()
config.compute.columns = ["user_id", "purchase_amount", "timestamp"]

report = profile(df, config=config)
report.save_html("key_columns_report.html")
```

## Reproducible Reports

Generate identical reports across runs.

```python
from pysuricata import profile, ReportConfig
from datetime import datetime

# Set random seed
config = ReportConfig()
config.compute.random_seed = 42

# Add metadata
config.render.title = "Weekly Data Report"
config.render.description = f"""
Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset: production.customers
Version: 1.2.3
"""

report = profile(df, config=config)
report.save_html(f"report_{datetime.now().strftime('%Y%m%d')}.html")
```

## Memory-Constrained Environment

Profile on device with limited RAM.

```python
from pysuricata import profile, ReportConfig

# Optimize for low memory
config = ReportConfig()
config.compute.chunk_size = 10_000  # Small chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.uniques_sketch_size = 1_024  # Small sketches
config.compute.top_k_size = 20  # Few top values
config.compute.compute_correlations = False  # Skip correlations

report = profile(df, config=config)
report.save_html("low_memory_report.html")
```

## Export Statistics as JSON

Save stats for external processing.

```python
from pysuricata import profile

report = profile(df)

# Save HTML
report.save_html("report.html")

# Save JSON
report.save_json("report.json")

# Or load JSON for custom analysis
import json
with open("report.json") as f:
    stats = json.load(f)
    
# Custom visualization
import matplotlib.pyplot as plt
missing = {col: s["missing_pct"] for col, s in stats["columns"].items()}
plt.bar(missing.keys(), missing.values())
plt.title("Missing Values by Column")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("missing_chart.png")
```

## Combine Multiple Datasets

Compare multiple datasets (manual).

```python
from pysuricata import summarize

# Profile multiple datasets
stats_train = summarize(df_train)
stats_test = summarize(df_test)

# Compare key metrics
print("Train vs Test Comparison:")
print(f"Rows: {stats_train['dataset']['rows']} vs {stats_test['dataset']['rows']}")
print(f"Missing: {stats_train['dataset']['missing_cells_pct']:.1f}% vs {stats_test['dataset']['missing_cells_pct']:.1f}%")

# Column-level comparison
for col in df_train.columns:
    train_mean = stats_train["columns"][col].get("mean")
    test_mean = stats_test["columns"][col].get("mean")
    if train_mean and test_mean:
        print(f"{col} mean: {train_mean:.2f} vs {test_mean:.2f}")
```

## Next Steps

- Explore [Configuration](configuration.md) for all options
- See [Performance Tips](performance.md) for optimization
- Check [Advanced Features](advanced.md) for power user tips

---

*Last updated: 2025-10-12*




