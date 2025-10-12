---
title: Frequently Asked Questions
description: Common questions and answers about PySuricata
---

# Frequently Asked Questions

## General Questions

### What is PySuricata?

PySuricata is a lightweight Python library for exploratory data analysis (EDA) that generates self-contained HTML reports using streaming algorithms for memory efficiency.

### When should I use PySuricata vs pandas-profiling?

**Use PySuricata when:**
- Dataset > 1 GB (memory-constrained)
- Need streaming/bounded memory
- Want minimal dependencies
- Need reproducible reports (seeded sampling)
- Working with polars

**Use pandas-profiling when:**
- Dataset < 100 MB
- Need interactive widgets
- Want correlation heatmaps
- Don't mind heavy dependencies

### Is PySuricata production-ready?

Yes! PySuricata is actively maintained with:
- 90%+ test coverage
- CI/CD pipeline
- Semantic versioning
- Regular releases on PyPI

## Installation & Setup

### How do I install PySuricata?

```bash
pip install pysuricata
```

### What are the dependencies?

**Core dependencies:**
- pandas (or polars)
- markdown
- Python 3.9+

**Optional:**
- polars (for polars DataFrames)

### Why is my installation failing?

Common issues:
1. **Python version**: Requires 3.9+
   ```bash
   python --version  # Check version
   ```

2. **Conflicting packages**: Try fresh virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install pysuricata
   ```

## Usage Questions

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

Set random seed:

```python
config = ReportConfig()
config.compute.random_seed = 42

report = profile(df, config=config)
```

### Can I get statistics without HTML?

Yes, use `summarize()`:

```python
from pysuricata import summarize

stats = summarize(df)
print(stats["dataset"])
print(stats["columns"]["my_column"])
```

## Performance Questions

### How much memory does PySuricata use?

**Approximately:**
- Base overhead: ~50 MB
- Per numeric column: ~160 KB (default sample_size=20K)
- Per categorical column: ~100 KB
- Independent of dataset size (streaming)

### My report is slow. How can I speed it up?

**Quick wins:**
1. Disable correlations:
   ```python
   config.compute.compute_correlations = False
   ```

2. Reduce sample sizes:
   ```python
   config.compute.numeric_sample_size = 10_000  # Default: 20_000
   ```

3. Increase chunk size:
   ```python
   config.compute.chunk_size = 500_000  # Default: 200_000
   ```

### Can PySuricata handle 1 TB datasets?

Yes, with streaming:

```python
def read_large_dataset():
    for file in large_files:
        yield pd.read_parquet(file)

report = profile(read_large_dataset())
```

Memory usage stays constant regardless of dataset size.

### Why are correlations slow?

Correlation computation is O(p²) where p = number of numeric columns.

**Solutions:**
- Disable for > 100 columns
- Increase threshold to show fewer correlations
- Profile fewer columns

## Technical Questions

### Are the statistics exact or approximate?

**Exact:**
- Mean, variance, skewness, kurtosis (Welford/Pébay)
- Min, max, count
- Quantiles (if dataset fits in sample)

**Approximate:**
- Distinct count (KMV sketch, ~2% error with k=2048)
- Top-k values (Misra-Gries, guaranteed for freq > n/k)
- Quantiles for huge datasets (from reservoir sample)

### How accurate are the approximations?

**Distinct count (KMV):**
- k=1024: ~3% error
- k=2048: ~2% error (default)
- k=4096: ~1.5% error

**Top-k (Misra-Gries):**
- Guaranteed to find all items with frequency > n/k
- Frequency estimates within ±n/k

### What algorithms does PySuricata use?

- **Moments**: Welford's online algorithm, Pébay's parallel merge
- **Distinct count**: K-Minimum Values (KMV) sketch
- **Top-k**: Misra-Gries algorithm
- **Quantiles**: Reservoir sampling (exact uniform sample)
- **Correlations**: Streaming Pearson correlation

See [Algorithms](algorithms/streaming.md) for details.

### Does PySuricata support distributed computing?

Partially:
- **Accumulators are mergeable**: Can run on multiple machines and merge results
- **No built-in distribution**: Must use external framework (Spark, Dask)

Example with manual merge:

```python
# Machine 1
acc1 = NumericAccumulator("col")
acc1.update(data_part1)

# Machine 2
acc2 = NumericAccumulator("col")
acc2.update(data_part2)

# Merge
acc1.merge(acc2)
final_stats = acc1.finalize()
```

## Data Questions

### Does PySuricata modify my data?

No. PySuricata only reads data, never modifies it.

### What data formats are supported?

Anything that can be loaded into pandas or polars:
- CSV, Parquet, JSON, Excel
- SQL databases (via pandas read_sql)
- APIs (via pandas read_json)

Just load into DataFrame first:

```python
df = pd.read_csv("data.csv")
report = profile(df)
```

### Can I profile streaming data (Kafka, etc.)?

Yes, if you can iterate through chunks:

```python
def read_from_kafka():
    consumer = KafkaConsumer(...)
    chunk = []
    for message in consumer:
        chunk.append(parse(message))
        if len(chunk) >= 10_000:
            yield pd.DataFrame(chunk)
            chunk = []

report = profile(read_from_kafka())
```

### How does PySuricata handle missing values?

- **Excluded from calculations**: Missing values don't affect mean, variance, etc.
- **Reported separately**: Missing count and percentage shown
- **Visualization**: Missing data distribution per chunk

See [Missing Values](analytics/missing-values.md).

### What about duplicate rows?

PySuricata estimates duplicate percentage using KMV sketch (approximate). For exact duplicates:

```python
exact_duplicates = df.duplicated().sum()
dup_pct = (exact_duplicates / len(df)) * 100
```

## Report Questions

### Why is my HTML report so large?

Possible reasons:
1. **Many columns**: Each variable card adds HTML
2. **Large sample**: Reduce `sample_rows`
3. **Many top values**: Reduce `top_k_size`

**Typical sizes:**
- 10 columns: ~500 KB
- 50 columns: ~2 MB
- 100 columns: ~4 MB

### Can I customize the report appearance?

Not directly. The report uses inline CSS for portability. 

**Workaround:** Modify the template in `pysuricata/templates/report_template.html` (advanced).

### Can I export to PDF?

Not built-in. Options:
1. Print HTML to PDF in browser
2. Use tool like `wkhtmltopdf`:
   ```bash
   wkhtmltopdf report.html report.pdf
   ```

### How do I display reports in Jupyter?

```python
report = profile(df)
report  # Auto-displays

# Or with custom size
report.display_in_notebook(height="800px")
```

## Error Messages

### "Out of memory" error

Reduce memory usage:

```python
config = ReportConfig()
config.compute.chunk_size = 50_000  # Smaller chunks
config.compute.numeric_sample_size = 5_000  # Smaller samples
config.compute.compute_correlations = False  # Skip correlations
```

### "Cannot infer type" error

Some columns may have mixed types. Clean data first:

```python
# Convert to consistent type
df["mixed_col"] = df["mixed_col"].astype(str)

# Or drop problematic columns
df = df.drop(columns=["problematic_col"])
```

### "Build failed" (documentation)

If contributing and docs build fails:

```bash
# Check mkdocs.yml syntax
uv run python -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"

# Verify all files exist
ls docs/  # Check file names match mkdocs.yml

# Build with verbose output
uv run mkdocs build --verbose
```

## Contributing Questions

### How can I contribute?

See [Contributing Guide](contributing.md).

**Ways to contribute:**
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests
- Help others in Discussions

### Where do I report bugs?

[GitHub Issues](https://github.com/alvarodiez20/pysuricata/issues)

Include:
- Python version
- PySuricata version
- Minimal reproducible example
- Error message/traceback

### Where can I get help?

- 💬 [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- 🐛 [GitHub Issues](https://github.com/alvarodiez20/pysuricata/issues)
- 📧 Email: alvarodiez20@gmail.com

## Still have questions?

Ask in [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions) or open an [issue](https://github.com/alvarodiez20/pysuricata/issues).

---

*Last updated: 2025-10-12*




