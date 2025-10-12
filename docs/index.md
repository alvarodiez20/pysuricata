<div align="center">
  <img src="assets/logo_suricata_transparent.png" alt="PySuricata Logo" width="300" style="margin: 2rem 0;">
</div>

# PySuricata

**Lightweight, high-performance exploratory data analysis for Python.**

Generate comprehensive, self-contained HTML reports for pandas and polars DataFrames using proven streaming algorithms that work with datasets of any size.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get Started Fast**

    ---

    Install and create your first report in 60 seconds.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-chart-line:{ .lg .middle } **Why PySuricata?**

    ---

    Learn how PySuricata compares to alternatives.

    [:octicons-arrow-right-24: Competitive Advantages](why-pysuricata.md)

-   :material-book-open:{ .lg .middle } **User Guide**

    ---

    Comprehensive guides for all use cases.

    [:octicons-arrow-right-24: Usage Documentation](usage.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation with examples.

    [:octicons-arrow-right-24: API Docs](api.md)

</div>

## Features

<div class="grid" markdown>

=== "Memory Efficient"

    **True streaming architecture** processes datasets larger than RAM with bounded memory using O(1) or O(log n) algorithms per column.

    ```python
    # Profile 10GB dataset in 50MB memory
    def read_chunks():
        for i in range(100):
            yield pd.read_parquet(f"part-{i}.parquet")
    
    report = profile(read_chunks())
    ```

=== "Fast & Accurate"

    **Proven algorithms** with mathematical guarantees: Welford/Pébay for exact moments, KMV/Misra-Gries for approximate analytics.

    **15x faster** than pandas-profiling on 1GB datasets.

=== "Framework Flexible"

    **Native support** for pandas and polars DataFrames. Works with CSV, Parquet, SQL, or any data source.

    ```python
    # Pandas
    report = profile(pd_df)
    
    # Polars
    report = profile(pl_df)
    
    # Polars Lazy
    report = profile(lf)
    ```

=== "Portable Reports"

    **Self-contained HTML** with inline CSS/JS/images. Share via email, cloud storage, or static hosting. No dependencies required.

</div>

## Quick Example

```python
import pandas as pd
from pysuricata import profile

# Load data
df = pd.read_csv("data.csv")

# Generate report
report = profile(df)
report.save_html("report.html")
```

That's it! Open `report.html` to see:

- Dataset overview (rows, columns, memory, missing, duplicates)
- Per-variable analysis (numeric, categorical, datetime, boolean)
- Correlations between numeric variables  
- Missing values patterns
- Data quality metrics

## What You'll Find in Reports

### Numeric Variables
- Central tendency: mean, median
- Dispersion: variance, std, IQR, MAD
- Shape: skewness, kurtosis
- Quantiles and histograms
- Outlier detection (IQR, z-score, MAD)
- Correlations with other columns

### Categorical Variables
- Top values and frequencies
- Distinct count (exact or approximate)
- Entropy and Gini impurity
- String length statistics
- Case/trim variants

### DateTime Variables
- Temporal range and coverage
- Hour/day-of-week/month distributions
- Monotonicity detection
- Timeline visualizations

### Boolean Variables
- True/false ratios
- Entropy calculation
- Imbalance detection
- Balance scores

## Key Advantages

✅ **Memory efficient** - Process GB/TB datasets in bounded memory  
✅ **Fast** - Single-pass O(n) algorithms  
✅ **Accurate** - Exact moments, provable approximation bounds  
✅ **Portable** - Self-contained HTML reports  
✅ **Minimal dependencies** - Just pandas/polars  
✅ **Reproducible** - Seeded sampling for deterministic results  
✅ **Customizable** - Extensive configuration options  

## Installation

```bash
pip install pysuricata
```

Optional polars support:

```bash
pip install pysuricata polars
```

## Next Steps

<div class="grid cards" markdown>

-   **Never used PySuricata?**

    Start with the [Quick Start Guide](quickstart.md)

-   **Coming from pandas-profiling?**

    See [Why PySuricata?](why-pysuricata.md)

-   **Need specific examples?**

    Check the [Examples Gallery](examples.md)

-   **Want to understand the math?**

    Explore [Statistical Methods](stats/overview.md)

</div>

## Community & Support

- 📖 [Documentation](https://alvarodiez20.github.io/pysuricata/)
- 💬 [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- 🐛 [Issue Tracker](https://github.com/alvarodiez20/pysuricata/issues)
- ⭐ [Star on GitHub](https://github.com/alvarodiez20/pysuricata)

## Contributing

We welcome contributions! See the [Contributing Guide](contributing.md) to get started.

## License

MIT License. See [LICENSE](https://github.com/alvarodiez20/pysuricata/blob/main/LICENSE) for details.

---

**Ready to profile your data?** [Install now →](install.md)
