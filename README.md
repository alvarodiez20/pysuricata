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
  
  <h3>Lightweight, High-Performance Exploratory Data Analysis for Python</h3>
  
  <p>
    <strong>Generate comprehensive, self-contained HTML reports using proven streaming algorithms</strong>
  </p>
  
  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/">Documentation</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/examples/">Examples</a> •
    <a href="#why-pysuricata">Why PySuricata?</a>
  </p>
</div>

---

## ✨ Features

- 🚀 **True Streaming Architecture** - Process datasets larger than RAM with bounded memory (~50MB regardless of size)
- 📊 **Mathematically Rigorous** - Welford/Pébay for exact moments, KMV/Misra-Gries with proven error bounds
- 📦 **Minimal Dependencies** - Just pandas/polars + markdown (~10 MB installed)
- 📄 **Portable Reports** - Self-contained HTML with inline CSS/JS/images
- 🔄 **Framework Flexible** - Native pandas and polars support
- 🎲 **Reproducible** - Seeded sampling for deterministic results
- ⚙️ **Highly Customizable** - Extensive configuration without code changes
- 🖥️ **CLI Tool** - Profile datasets from the command line

## Quick Start

### Installation

```bash
pip install pysuricata
```

### Generate Your First Report

```python
import pandas as pd
from pysuricata import profile

# Load data
df = pd.read_csv("your_data.csv")

# Generate report
report = profile(df)
report.save_html("report.html")
```

That's it! Open `report.html` in your browser to see a comprehensive analysis.

### Command Line Interface

```bash
# Generate HTML report
pysuricata profile data.csv --output report.html

# Get JSON statistics
pysuricata summarize data.csv
```

## Why PySuricata?

### 🆚 Comparison with Alternatives

| Feature | PySuricata | pandas-profiling | sweetviz | pandas-eda |
|---------|------------|------------------|----------|------------|
| **Memory model** | 🟢 Streaming (bounded) | 🔴 In-memory (full) | 🔴 In-memory | 🔴 In-memory |
| **Large datasets (>1GB)** | ✅ GB to TB | ❌ RAM limited | ❌ RAM limited | ❌ RAM limited |
| **Peak memory (1GB)** | 🟢 50 MB | 🔴 1.2 GB | 🔴 1.1 GB | 🔴 1.0 GB |
| **Dependencies** | 🟢 Minimal (~10 MB) | 🔴 Heavy (100+ MB) | 🟡 Medium (80 MB) | 🟡 Medium |
| **Report format** | 🟢 Single HTML | 🟡 HTML + assets | 🟡 HTML + assets | 🟡 HTML + assets |
| **Polars support** | ✅ Native | ❌ No | ❌ No | ❌ No |
| **Exact algorithms** | ✅ Welford/Pébay | ⚠️ NumPy/SciPy | ⚠️ NumPy/SciPy | ⚠️ NumPy/SciPy |
| **Reproducibility** | ✅ Seeded | ⚠️ Partial | ⚠️ Partial | ❌ No |

### 📊 Performance Benchmarks

**Memory usage** (1M+ rows dataset):

```
PySuricata:       ██░░░░░░░░░░░░░░░░░░  50 MB (constant!)
pandas-eda:       ████████████████████  1.0 GB
sweetviz:         █████████████████████ 1.1 GB
pandas-profiling: ██████████████████████ 1.2 GB
```

> **Key advantage**: PySuricata uses ~50MB regardless of dataset size, while competitors require memory proportional to data size.

## 🎯 What's in a Report?

PySuricata generates comprehensive reports with:

### Dataset Overview
- Rows, columns, memory usage
- Missing values summary
- Duplicate rows estimate
- Processing time and throughput

### Variable Analysis (4 types)

**📊 Numeric**: Mean, variance, skewness, kurtosis, quantiles, histograms, outliers, correlations  
**📝 Categorical**: Top values, distinct count, entropy, Gini, string statistics  
**📅 DateTime**: Temporal range, hour/day/month distributions, monotonicity, timeline charts  
**✓ Boolean**: True/false ratios, entropy, balance scores, imbalance detection

### Advanced Analytics
- Streaming correlations (Pearson r)
- Missing value patterns per chunk
- Data quality metrics
- Outlier detection (IQR, MAD, z-score)

All statistics computed using **mathematically proven algorithms** with error bounds.

## 📚 Examples

### Small Dataset (In-Memory)

```python
import pandas as pd
from pysuricata import profile

# Load Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Generate report
report = profile(df)
report.save_html("iris_report.html")
```

### Large Dataset (Streaming)

Process datasets larger than RAM with constant memory usage:

```python
from pysuricata import profile, ReportConfig
import pandas as pd

def read_large_dataset():
    """Generator yielding chunks"""
    for i in range(100):
        yield pd.read_parquet(f"data/part-{i}.parquet")

# Configure for large data
config = ReportConfig()
config.compute.chunk_size = 250_000
config.compute.random_seed = 42

# Profile in bounded memory
report = profile(read_large_dataset(), config=config)
report.save_html("large_dataset_report.html")
```

### Polars Support

```python
import polars as pl
from pysuricata import profile

# Works natively with polars
df = pl.read_csv("data.csv")
report = profile(df)
report.save_html("polars_report.html")

# Also supports LazyFrame
lf = pl.scan_csv("large_file.csv").filter(pl.col("value") > 0)
report = profile(lf)
```

### Statistics Only (No HTML)

Perfect for CI/CD data quality checks:

```python
from pysuricata import summarize

stats = summarize(df)

# Check data quality thresholds
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0

# Access per-column statistics
print(f"Mean age: {stats['columns']['age']['mean']:.1f}")
print(f"Distinct countries: {stats['columns']['country']['distinct']}")
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

### Custom Description (Markdown Support)

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()
config.render.description = """
# Q4 2024 Customer Analysis

**Dataset**: Production customer transactions  
**Period**: October - December 2024  

## Key Findings
- Revenue up 15% YoY
- Average transaction: $87.50
"""

report = profile(df, config=config)
```

## 🧮 Algorithms & Mathematical Rigor

PySuricata uses state-of-the-art streaming algorithms:

### Exact Statistics
- **Welford's algorithm** - Online mean/variance (numerically stable)
- **Pébay's formulas** - Parallel merging of moments (exact, mergeable)
- **Streaming correlations** - Sufficient statistics for Pearson r

### Approximate Statistics (with guarantees)
- **KMV sketch** - Distinct count (error ~2% with k=2048)
- **Misra-Gries** - Top-k heavy hitters (guaranteed for freq > n/k)
- **Reservoir sampling** - Uniform random sample (exact probability k/n)

All algorithms have **proven error bounds** and **mathematical guarantees**. See [full documentation](https://alvarodiez20.github.io/pysuricata/stats/overview/) for formulas and proofs.

## 🎨 Report Highlights

- **Self-contained**: Single HTML file, no external dependencies
- **Beautiful visualizations**: Inline SVG charts and histograms
- **Responsive design**: Works on desktop and mobile
- **Dark mode**: Toggle between light and dark themes
- **Professional styling**: Clean, modern interface
- **Shareable**: Email, cloud storage, or static hosting

## 💡 Use Cases

### Data Science & ML
- **EDA** - Understand distributions, correlations, missing patterns
- **Feature engineering** - Identify high-cardinality, constant columns
- **Data validation** - Check quality before training
- **Reproducibility** - Generate consistent reports with seeds

### Data Engineering
- **Pipeline monitoring** - Track data quality over time
- **CI/CD checks** - Assert quality thresholds
- **Documentation** - Auto-generate data dictionaries
- **Debugging** - Quickly profile large production datasets

### Business Analytics
- **Dashboard generation** - Automated reporting
- **Data documentation** - Share with stakeholders
- **Quality assurance** - Catch issues early
- **Compliance** - Document data characteristics

## ⚙️ Configuration

Highly customizable via `ReportConfig`:

```python
from pysuricata import profile, ReportConfig

config = ReportConfig()

# Processing
config.compute.chunk_size = 200_000  # Rows per chunk
config.compute.numeric_sample_size = 20_000  # Sample for quantiles
config.compute.random_seed = 42  # Reproducibility

# Analysis
config.compute.compute_correlations = True  # Enable correlations
config.compute.corr_threshold = 0.5  # Min |r| to show
config.compute.top_k_size = 50  # Top values to track

# Rendering
config.render.title = "My Analysis Report"
config.render.description = "Custom markdown description"
config.render.include_sample = True
config.render.sample_rows = 10

report = profile(df, config=config)
```

See [Configuration Guide](https://alvarodiez20.github.io/pysuricata/configuration/) for all options.

## 🔬 Statistical Methods

### Numeric Variables
- Central tendency: mean, median
- Dispersion: variance, std, IQR, MAD, CV
- Shape: skewness, kurtosis
- Quantiles: P1, P5, Q1, Q2, Q3, P95, P99
- Outliers: IQR fences, z-score, MAD-based
- Distribution: histogram (Freedman-Diaconis binning)

### Categorical Variables
- Frequency table (top-k via Misra-Gries)
- Distinct count (KMV sketch)
- Shannon entropy, Gini impurity
- String length statistics
- Case/trim variant detection

### DateTime Variables
- Temporal range and span
- Hour distribution (0-23)
- Day-of-week distribution
- Month distribution
- Monotonicity coefficient
- Timeline visualization

### Boolean Variables
- True/false counts and percentages
- Shannon entropy
- Balance score
- Imbalance ratio

Full mathematical formulas and derivations in [Statistical Methods](https://alvarodiez20.github.io/pysuricata/stats/overview/).

## 📈 Scalability

PySuricata scales **linearly** with dataset size:

| Dataset Size | Processing Time | Peak Memory | Throughput |
|--------------|-----------------|-------------|------------|
| 10K rows | ~3s | 20 MB | ~3,000 rows/s |
| 100K rows | ~13s | 30 MB | ~8,000 rows/s |
| 1M rows | ~3 min | 50 MB | ~5,500 rows/s |
| 10M rows | ~30 min | 50 MB | ~5,500 rows/s |
| 100M rows | ~5 hours | 50 MB | ~5,500 rows/s |

**Memory stays constant** regardless of dataset size! 🎉

> *Benchmarks measured on Apple Silicon with Python 3.13. Actual times vary by hardware and data complexity.*

## 🤝 Why "Suricata"?

Inspired by **suricatas (meerkats)** - small, vigilant animals that work cooperatively to survive in harsh desert environments:

- 👀 **Watchful** - Always scanning for patterns (like data analysis)
- 🤝 **Cooperative** - Parallel/distributed processing
- 🏜️ **Efficient** - Thrive with limited resources (bounded memory)
- ⚡ **Quick** - Fast reactions (streaming algorithms)
- 🔍 **Pattern recognition** - Identify important signals

Learn more about [why suricatas inspired this library](https://alvarodiez20.github.io/pysuricata/about-suricatas/).

## 📖 Documentation

Comprehensive documentation with mathematical formulas, algorithm details, and examples:

- 📘 [Quick Start Guide](https://alvarodiez20.github.io/pysuricata/quickstart/) - Get started in 5 minutes
- 📗 [User Guide](https://alvarodiez20.github.io/pysuricata/usage/) - Detailed usage patterns
- 📕 [API Reference](https://alvarodiez20.github.io/pysuricata/api/) - Complete API documentation
- 📙 [Statistical Methods](https://alvarodiez20.github.io/pysuricata/stats/overview/) - Mathematical formulas
- 📔 [Algorithms](https://alvarodiez20.github.io/pysuricata/algorithms/streaming/) - Welford, Pébay, KMV, Misra-Gries
- 📓 [Performance Tips](https://alvarodiez20.github.io/pysuricata/performance/) - Optimization strategies
- 📰 [Examples Gallery](https://alvarodiez20.github.io/pysuricata/examples/) - Real-world use cases

## 🛠️ Advanced Features

### Distributed Processing

Accumulators are **mergeable** - compute on multiple machines and combine:

```python
from pysuricata.accumulators import NumericAccumulator

# Worker 1
acc1 = NumericAccumulator("amount")
acc1.update(data_partition_1)

# Worker 2  
acc2 = NumericAccumulator("amount")
acc2.update(data_partition_2)

# Merge on coordinator (exact, no approximation)
acc1.merge(acc2)
final_stats = acc1.finalize()
```

### CI/CD Integration

```python
from pysuricata import summarize

def validate_data_quality(df):
    stats = summarize(df)  # Fast, stats-only
    
    assert stats["dataset"]["missing_cells_pct"] < 5.0, "Too many missing"
    assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0, "Too many duplicates"
    
    print("✓ Data quality checks passed")

# In your pipeline
validate_data_quality(df)
```

### Jupyter Notebooks

```python
from pysuricata import profile

report = profile(df)
report  # Auto-displays inline

# Or with custom size
report.display_in_notebook(height="800px")
```

## 🎓 Academic Use

If you use PySuricata in academic research, please reference the algorithms:

**Streaming moments:**
- Welford, B.P. (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", *Technometrics*
- Pébay, P. (2008), "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments", Sandia Report

**Sketch algorithms:**
- Bar-Yossef, Z. et al. (2002), "Counting Distinct Elements in a Data Stream", *RANDOM*
- Misra, J., Gries, D. (1982), "Finding repeated elements", *Science of Computer Programming*

See [References](https://alvarodiez20.github.io/pysuricata/stats/numeric/#references) for complete citations.

## 🗺️ Roadmap

### Current Version (0.0.11)
- ✅ Streaming architecture
- ✅ 4 variable types (numeric, categorical, datetime, boolean)
- ✅ Streaming correlations
- ✅ Missing value analysis
- ✅ Polars support
- ✅ Comprehensive documentation

### Planned Features
- 🔜 **Spearman rank correlation** (monotonic relationships)
- 🔜 **Little's MCAR test** (missing data mechanism)
- 🔜 **Chi-square uniformity test** (categorical distribution)
- 🔜 **Seasonality detection** (autocorrelation for datetime)
- 🔜 **Gap analysis** (missing time periods)
- 🔜 **Profile comparison** (compare two datasets)
- 🔜 **Export to JSON/CSV** (structured statistics)
- 🔜 **CLI tool** (command-line interface)
- 🔜 **Dask integration** (native distributed support)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://alvarodiez20.github.io/pysuricata/contributing/) to get started.

**Ways to contribute:**
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🧪 Add tests
- 🔧 Submit pull requests
- 💬 Help others in Discussions

### Development Setup

```bash
# Clone repository
git clone https://github.com/alvarodiez20/pysuricata.git
cd pysuricata

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Build documentation
uv run mkdocs serve
```

## 📊 Project Stats

- **Lines of code**: ~15,000
- **Test coverage**: 90%+
- **Documentation pages**: 25+
- **Supported Python versions**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Active development**: Regular releases

## 🆘 Support & Community

- 📖 [Documentation](https://alvarodiez20.github.io/pysuricata/)
- 💬 [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- 🐛 [Issue Tracker](https://github.com/alvarodiez20/pysuricata/issues)
- 📧 Email: alvarodiez20@gmail.com
- ⭐ [Star on GitHub](https://github.com/alvarodiez20/pysuricata)

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with inspiration from:
- **Suricatas (meerkats)** - Vigilant, cooperative, efficient
- **Welford & Pébay** - Streaming moments algorithms
- **Bar-Yossef, Misra & Gries** - Sketch algorithms
- Open-source community

## ⭐ Star History

If you find PySuricata useful, please star the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=alvarodiez20/pysuricata&type=Date)](https://star-history.com/#alvarodiez20/pysuricata&Date)

---

<div align="center">
  <p><strong>Ready to analyze like a suricata?</strong></p>
  <p>
    <a href="https://alvarodiez20.github.io/pysuricata/quickstart/">📚 Read the Docs</a> •
    <a href="https://github.com/alvarodiez20/pysuricata/issues/new">🐛 Report Bug</a> •
    <a href="https://github.com/alvarodiez20/pysuricata/discussions">💬 Discussions</a>
  </p>
  
  <p><em>"In the Kalahari of big data, be a suricata - vigilant, efficient, and always ready to dig for insights!"</em></p>
</div>
