# Why PySuricata?

PySuricata is a Python library for generating HTML data profiling reports from pandas or polars DataFrames. Its main design choice is a **streaming architecture**: data is processed in chunks, so memory usage stays bounded regardless of dataset size.

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

This page explains the design decisions behind PySuricata and when it might be a good fit for your workflow.

---

## Streaming Architecture

<figure class="ps-figure" markdown="0">
  <iframe src="../assets/diagrams/figures.html?only=memory-curve" title="Peak memory against dataset size" loading="lazy"></iframe>
  <figcaption>Illustrative curve, not a measurement. Run <code>python -m benchmarks.end_to_end --markdown results.md</code> for figures from your own machine; publishing a comparison table is <a href="https://github.com/alvarodiez20/pysuricata/issues/38">#38</a>.</figcaption>
</figure>

Most profiling tools load the entire dataset into memory to compute statistics. PySuricata takes a different approach — it processes data **one chunk at a time**, updating lightweight accumulators as it goes.

```python
from pysuricata import profile, ProfileConfig

# PySuricata processes data in chunks internally
config = ProfileConfig()
config.compute.chunk_size = 50_000  # 50k rows per chunk (default)

report = profile(df, config=config)
```

This means:

- **Memory is bounded by chunk size**, not dataset size. A 10 GB dataset uses roughly the same memory as a 100 MB one.
- **Each row is read exactly once** — there's no second pass over the data.
- **Accumulators are mergeable** — statistics computed on separate chunks (or machines) can be combined exactly.

You can also pass a generator of DataFrames for datasets that don't fit in memory at all:

```python
import pandas as pd
from pysuricata import profile

def read_in_parts():
    for i in range(100):
        yield pd.read_parquet(f"data/part-{i}.parquet")

# Processes 100 files without loading them all at once
report = profile(read_in_parts())
report.save_html("large_dataset_report.html")
```

---

## Algorithms

PySuricata uses well-known streaming algorithms from the academic literature. Here's what it computes and how:

### Exact Statistics — Welford & Pébay

Mean, variance, skewness, and kurtosis are computed **exactly** using Welford's online algorithm, extended with Pébay's merge formulas for combining results across chunks.

```python
# Conceptually, this is what happens per value:
n += 1
delta = value - mean
mean += delta / n
M2 += delta * (value - mean)
# variance = M2 / (n - 1)
```

These formulas are **numerically stable** (they avoid the catastrophic cancellation that can happen with naive sum-of-squares approaches) and **exactly mergeable** (combining two partial results gives the same answer as processing all data at once).

**References:**

- Welford, B.P. (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", *Technometrics*
- Pébay, P. (2008), "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments", Sandia Report

### Approximate Statistics — Sketches and Sampling

Some statistics can't be computed exactly in a single pass with bounded memory. PySuricata uses probabilistic data structures with known error bounds:

| Algorithm | Purpose | Space | Error Bound |
|-----------|---------|-------|-------------|
| **KMV sketch** | Distinct count estimation | O(k), default k=2048 | ε ≈ 1/√k (~2.2%) |
| **Misra-Gries** | Top-k frequent values | O(k), default k=50 | Finds all items with frequency > n/k |
| **Reservoir sampling** | Uniform random sample | O(s), default s=20,000 | Exact probability k/n per item |

These are standard algorithms with well-understood properties. The error bounds are **theoretical guarantees**, not empirical estimates.

---

## Pandas and Polars Support

PySuricata works natively with both pandas and polars DataFrames. The same `profile()` function handles both:

=== "Pandas"

    ```python
    import pandas as pd
    from pysuricata import profile

    df = pd.read_csv("data.csv")
    report = profile(df)
    report.save_html("report.html")
    ```

=== "Polars"

    ```python
    import polars as pl
    from pysuricata import profile

    df = pl.read_csv("data.csv")
    report = profile(df)
    report.save_html("report.html")
    ```

=== "Polars LazyFrame"

    ```python
    import polars as pl
    from pysuricata import profile

    lf = pl.scan_csv("large_file.csv")
    report = profile(lf)
    report.save_html("report.html")
    ```

---

## Self-Contained Reports

PySuricata generates a **single HTML file** with everything inlined:

- CSS styles embedded in `<style>` tags
- JavaScript embedded in `<script>` tags
- Charts rendered as inline SVG (no image files)
- Logo encoded as base64

The resulting file can be opened in any browser, shared via email, hosted on a static server, or committed to a repository. There are no external assets to manage.

---

## Configuration

All processing parameters are exposed through `ProfileConfig`:

```python
from pysuricata import profile, ProfileConfig

config = ProfileConfig()

# Processing
config.compute.chunk_size = 250_000       # Rows per chunk
config.compute.numeric_sample_size = 50_000  # Sample size for quantiles
config.compute.random_seed = 42           # Deterministic sampling

# Analysis
config.compute.compute_correlations = True
config.compute.corr_threshold = 0.5       # Min |r| to display
config.compute.top_k = 100           # Top values to track
config.compute.max_uniques = 4096 # KMV sketch size

# Rendering
config.render.title = "My Report"
config.render.description = "Custom **markdown** description"
config.render.include_sample = True   # show the sample table
config.render.sample_rows = 10

report = profile(df, config=config)
```

Most of that is unnecessary. The seven settings people actually reach for are
keywords, and two presets cover the common intents:

```python
from pysuricata import profile

report = profile(df, preset="thorough", seed=42, title="My Report")
```

Reports are **reproducible by default** — `random_seed` is `0`, not `None`, so
the same data produces the same output and re-running is a no-op. Pass a
different `seed=` to pin a different sample.

---

## What PySuricata Analyzes

PySuricata detects the type of each column and applies specialized analysis:

| Column Type | Statistics | Visualization |
|-------------|-----------|---------------|
| **Numeric** | Mean, variance, skewness, kurtosis, quantiles, IQR/MAD/z-score outliers | Histogram (SVG) |
| **Categorical** | Top-k values, distinct count, entropy, Gini impurity, string length stats | Donut chart (SVG) |
| **DateTime** | Range, hour/day/month distributions, monotonicity coefficient | Timeline (SVG) |
| **Boolean** | True/false ratios, entropy, balance score, imbalance ratio | Balance bar (SVG) |

Additionally, PySuricata computes:

- **Streaming correlations** (Pearson r) between numeric columns
- **Missing value analysis** per column and per chunk
- **Duplicate row estimation** using KMV hashing

---

## Use Cases

### Data Quality Checks in Pipelines

Use `summarize()` to get statistics as a dictionary, without generating HTML:

```python
import numpy as np
import pandas as pd

from pysuricata import summarize

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "user_id": range(1_000),
        "age": rng.integers(18, 80, 1_000),
        "country": rng.choice(["ES", "FR", "DE"], 1_000),
    }
)
stats = summarize(df)

# Assert data quality thresholds
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0

# Access per-column statistics
print(f"Mean age: {stats['columns']['age']['mean']:.1f}")
print(f"Distinct countries: {stats['columns']['country']['unique_est']}")  # KMV estimate
```

### Profiling Large Datasets

When your data doesn't fit in memory, pass a generator:

```python
import pandas as pd
from pysuricata import profile, ProfileConfig

config = ProfileConfig()
config.compute.chunk_size = 100_000
config.compute.random_seed = 42

def read_chunks():
    for chunk in pd.read_csv("large_file.csv", chunksize=100_000):
        yield chunk

report = profile(read_chunks(), config=config)
report.save_html("large_report.html")
```

### Jupyter Notebooks

Reports render inline in notebooks:

```python
from pysuricata import profile

report = profile(df)
report  # Auto-displays inline

# Or with custom height
report.display_in_notebook(height="800px")
```

### Gating a Build on Drift

The same single pass runs in CI. `pysuricata check` compares a dataset against a
stored baseline and **exits non-zero** when a threshold is crossed:

```bash
pysuricata check data/extract.parquet \
  --baseline baselines/extract.json \
  --max-missing-pct 5 \
  --require-fresh
```

`--require-fresh` is the one worth knowing about: stale data is not drifted
data. An extract that simply did not run looks identical to yesterday's, and
every distributional check passes.

See [Gating CI on drift](data-checks.md).

---

## Reading a Source Without Loading It

`profile()` and `summarize()` take more than a frame: a path, an Arrow table or
reader, anything exporting `__arrow_c_stream__`, or a DuckDB relation. Those are
read a **batch at a time** and never exist as one frame.

```python
from pysuricata import summarize

summarize("events.parquet")
```

Peak RSS on a 4,000,000 x 6 frame written as a 180 MB Parquet file, above a
118 MB bare-import floor:

| | above floor |
|---|---:|
| `profile(pd.read_parquet(path))` | 581 MB |
| `profile(path)` | 307 MB |

The DuckDB path goes further — a relation is a query that has not run yet, so a
join across several Parquet files, filtered, can be profiled without any of it
existing as a frame. See [Arrow, Parquet and DuckDB](data-sources.md).

---

## What It Does Not Do

Worth saying plainly, since the rest of this page is a case for the design.

- **Memory is bounded in rows, not in columns.** State is per column at roughly
  3 MB each, so a 20,000 x 600 frame costs more than a 5,000,000 x 8 one on less
  data. Tracked in
  [#207](https://github.com/alvarodiez20/pysuricata/issues/207).
- **Quantiles come from a sample.** Exact below the reservoir size, approximate
  above it, with the error bound published rather than hidden.
- **Category churn and distinct counts are sketch estimates.** They are marked
  approximate everywhere they appear, including in the JSON payload.
- **There is no HTML view for `compare()` yet** — the JSON contract and the text
  rendering are what it ships with.

---

## Learn More

- [Quick Start](quickstart.md) — Generate your first report
- [Configuration](configuration.md) — All available options
- [Statistical Methods](stats/overview.md) — Algorithm details and formulas
- [Architecture Diagrams](architecture-diagrams.md) — Visual overview of the processing pipeline
- [API Reference](api.md) — Function signatures and parameters
- [Command Line](cli.md) — `profile`, `summarize` and `check` from a shell
