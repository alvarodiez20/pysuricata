---
title: Examples Gallery
description: Comprehensive examples and use cases for PySuricata
---

# Examples Gallery

<figure class="ps-figure" markdown="0">
  <iframe src="../assets/diagrams/figures.html?only=report-card" title="An annotated numeric column card" loading="lazy"></iframe>
</figure>

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
from pysuricata import profile, ProfileConfig

def read_large_dataset():
    """Generator yielding chunks"""
    for i in range(100):
        yield pd.read_parquet(f"data/part-{i}.parquet")

# Configure for large data
config = ProfileConfig()
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
from pysuricata import profile, ProfileConfig

# Create wide dataset
n_rows, n_cols = 10_000, 500
df = pd.DataFrame(
    np.random.randn(n_rows, n_cols),
    columns=[f"feature_{i}" for i in range(n_cols)]
)

# Disable correlations (too expensive for 500 columns)
config = ProfileConfig()
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
import numpy as np
import pandas as pd

from pysuricata import profile

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "customer_id": [f"CUST_{i:05d}" for i in range(10_000)],
    "product": rng.choice(["Product A", "Product B", "Product C"], 10_000),
    "rating": rng.choice(["Poor", "Fair", "Good", "Excellent"], 10_000),
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
from pysuricata import profile, ProfileConfig

# Create lazy frame
lf = pl.scan_csv("large_file.csv").filter(pl.col("value") > 0)

# Configure chunk size
config = ProfileConfig()
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
print(f"Rows: {stats['dataset']['rows_est']}")
print(f"Missing cells: {stats['dataset']['missing_cells_pct']:.1f}%")
print(f"Duplicate rows: {stats['dataset']['duplicate_rows_pct_est']:.1f}%")

# Per-column stats
for col, col_stats in stats["columns"].items():
    if col_stats.get("missing_pct", 0) > 10:
        print(f"{col}: {col_stats['missing_pct']:.1f}% missing")
```

## CI/CD Data Quality Gates

!!! tip "There is a built-in for this"

    `pysuricata check` does the same single pass from a shell, with a stored
    baseline, a thresholds file and an exit code:

    ```bash
    pysuricata check data.parquet --baseline baseline.json --max-missing-pct 5
    ```

    See [Gating CI on drift](data-checks.md) and the
    [CLI reference](cli.md#check). The recipe below is the arithmetic, for when
    the gate lives inside your own code.

Enforce quality thresholds in pipelines.

```python
import pandas as pd

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

    # Check specific columns. unique_est is a KMV estimate, so compare with a
    # tolerance rather than for equality -- see Sketch Algorithms for the bound.
    for col in ["customer_id", "transaction_id"]:
        col_stats = stats["columns"][col]
        assert col_stats["unique_est"] >= col_stats["count"] * 0.97, (
            f"{col} looks like it has duplicates"
        )

    print("✓ Data quality checks passed")


# Use in pipeline
df = pd.DataFrame(
    {
        "customer_id": range(1_000),
        "transaction_id": range(1_000),
        "amount": [float(i) for i in range(1_000)],
    }
)
validate_data_quality(df)
```

## Custom Column Selection

Profile only specific columns.

```python
from pysuricata import profile, ProfileConfig

# Large dataset, only analyze key columns
config = ProfileConfig()
config.compute.columns = ["user_id", "purchase_amount", "timestamp"]

report = profile(df, config=config)
report.save_html("key_columns_report.html")
```

## Reproducible Reports

Generate identical reports across runs.

```python
from pysuricata import profile, ProfileConfig
from datetime import datetime

# Set random seed
config = ProfileConfig()
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
from pysuricata import profile, ProfileConfig

# Optimize for low memory
config = ProfileConfig()
config.compute.chunk_size = 10_000  # Small chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.max_uniques = 1_024  # Small sketches
config.compute.top_k = 20  # Few top values
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

!!! tip "There is a built-in for this"

    `compare(a, b)` reports every delta — schema, dataset and per column —
    normalised into baseline standard deviations, with the sketch-based ones
    marked approximate:

    ```python
    import numpy as np
    import pandas as pd

    from pysuricata import compare

    rng = np.random.default_rng(0)
    df_train = pd.DataFrame({"amount": rng.lognormal(3, 1, 5_000)})
    df_test = pd.DataFrame({"amount": rng.lognormal(3.1, 1, 2_000)})

    diff = compare(df_train, df_test)
    diff.columns["amount"].median_shift_sigma
    ```

    See [Comparing two datasets](comparing.md). The recipe below is the manual
    version, for when you want two payloads side by side.

Compare multiple datasets (manual).

```python
import numpy as np
import pandas as pd

from pysuricata import summarize

rng = np.random.default_rng(0)
df_train = pd.DataFrame({"amount": rng.lognormal(3, 1, 5_000)})
df_test = pd.DataFrame({"amount": rng.lognormal(3.1, 1, 2_000)})

# Profile multiple datasets
stats_train = summarize(df_train)
stats_test = summarize(df_test)

# Compare key metrics
print("Train vs Test Comparison:")
print(f"Rows: {stats_train['dataset']['rows_est']} vs {stats_test['dataset']['rows_est']}")
print(f"Missing: {stats_train['dataset']['missing_cells_pct']:.1f}% vs {stats_test['dataset']['missing_cells_pct']:.1f}%")

# Column-level comparison
for col in df_train.columns:
    train_mean = stats_train["columns"][col].get("mean")
    test_mean = stats_test["columns"][col].get("mean")
    if train_mean and test_mean:
        print(f"{col} mean: {train_mean:.2f} vs {test_mean:.2f}")
```

## Profile a File Without Loading It

`profile()` takes a path and reads it a batch at a time — the file never exists
as one frame.

```python
from pysuricata import profile

report = profile("events.parquet")
report.save_html("events.html")
```

CSV, Parquet, JSON and Arrow IPC all work. A DuckDB relation works too, and it
is a query that has not run yet, so a filtered join across several Parquet files
can be profiled without any of it being landed:

```python
import duckdb

from pysuricata import summarize

con = duckdb.connect()
relation = con.sql("SELECT * FROM 'events/*.parquet' WHERE amount > 0")
summarize(relation)
```

See [Arrow, Parquet and DuckDB](data-sources.md).

## Start From a Preset

One word for an intent, rather than working out which knobs to turn.

```python
import numpy as np
import pandas as pd

from pysuricata import profile

rng = np.random.default_rng(0)
df = pd.DataFrame({"amount": rng.lognormal(3, 1, 5_000)})

fast = profile(df, preset="fast")          # small samples, no correlations
careful = profile(df, preset="thorough")   # large samples, every correlation
```

Keyword options layer on top and win, so `profile(df, preset="fast",
correlations=True)` is a fast profile with the correlation step put back.

## From the Command Line

```bash
pysuricata profile data.csv --output report.html
pysuricata summarize data.csv | jq .dataset
pysuricata check data.parquet --baseline baseline.json --require-fresh
```

Progress goes to stderr, so the middle one stays parseable without `--quiet`.
Every option is in the [CLI reference](cli.md).

## Next Steps

- Explore [Configuration](configuration.md) for all options
- See [Performance Tips](performance.md) for optimization
- Check [Advanced Features](advanced.md) for power user tips
