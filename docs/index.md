<div align="center">
  <img src="assets/logo_suricata_transparent.png" alt="PySuricata Logo" width="300" style="margin: 2rem 0;">
</div>

[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![Python versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/alvarodiez20/pysuricata/branch/main/graph/badge.svg)](https://codecov.io/gh/alvarodiez20/pysuricata)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://alvarodiez20.github.io/pysuricata/)
[![Downloads](https://static.pepy.tech/badge/pysuricata)](https://pepy.tech/project/pysuricata)

# PySuricata

**Exploratory data analysis for Python, built on streaming algorithms.**

PySuricata generates self-contained HTML reports for pandas and polars DataFrames. It processes data in chunks using streaming algorithms, so memory usage stays bounded regardless of dataset size.

<div class="grid cards" markdown>

-   **Quick Start**

    ---

    Install PySuricata and generate your first report.

    [:octicons-arrow-right-24: Get Started](quickstart.md)

-   **Why PySuricata?**

    ---

    Understand the streaming architecture and design decisions.

    [:octicons-arrow-right-24: Learn More](why-pysuricata.md)

-   **User Guide**

    ---

    Detailed guides for configuration, advanced features, and more.

    [:octicons-arrow-right-24: Read the Guide](usage.md)

-   **API Reference**

    ---

    Full API documentation generated from source code.

    [:octicons-arrow-right-24: API Docs](api.md)

</div>

## Features

- **Streaming processing** — Data is processed in configurable chunks, keeping memory usage bounded. Useful for datasets that don't fit in RAM.
- **Mathematically grounded** — Uses Welford's algorithm for numerically stable moments, Pébay's formulas for mergeable statistics, KMV sketches for distinct count estimation, and Misra-Gries for heavy hitters.
- **Pandas and Polars support** — Works natively with both `pandas.DataFrame` and `polars.DataFrame` / `polars.LazyFrame`.
- **Self-contained reports** — Generates a single HTML file with inline CSS, JS, and SVG charts. No external assets or dependencies needed to view.
- **Configurable** — Control chunk sizes, sample sizes, sketch parameters, correlation thresholds, and rendering options via `ReportConfig`.
- **Reproducible** — Seeded random sampling produces deterministic results across runs.

## Installation

=== "uv (Recommended)"

    ```bash
    uv add pysuricata
    ```

=== "pip"

    ```bash
    pip install pysuricata
    ```

This installs PySuricata along with its dependencies: **pandas**, **numpy** (on Python ≥3.13), **markdown**, and **psutil**.

To also install polars support:

```bash
pip install pysuricata[polars]
```

## Quick Example

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

This is the actual report generated from the code above (Titanic dataset, 891 rows × 12 columns):

<div style="border: 2px solid #7CB342; border-radius: 8px; overflow: hidden; margin: 2rem 0;">
  <iframe src="assets/titanic_report.html" width="100%" height="800px" style="border: none;"></iframe>
</div>

<div align="center">
  <p><em>Can't see the report? <a href="assets/titanic_report.html" target="_blank">Open in new tab →</a></em></p>
</div>

## How It Works

PySuricata reads data in chunks and updates lightweight accumulators for each column. This means:

| Aspect | Approach |
|--------|----------|
| **Memory** | Bounded by chunk size + accumulator state, not dataset size |
| **Speed** | Single pass over the data — each row is read once |
| **Accuracy** | Exact for moments (mean, variance, skewness, kurtosis); approximate with known error bounds for distinct counts and top-k |
| **Mergeability** | Accumulators can be merged across chunks or machines |

Reports include per-column statistics, histograms, correlation chips, missing value analysis, outlier detection, and more — all computed during the single streaming pass.

## Next Steps

<div class="grid cards" markdown>

-   **New to PySuricata?**

    Start with the [Quick Start Guide](quickstart.md)

-   **Want specific examples?**

    Check the [Examples Gallery](examples.md)

-   **Interested in the algorithms?**

    Explore [Statistical Methods](stats/overview.md)

-   **Want to contribute?**

    Read the [Contributing Guide](contributing.md)

</div>

## Community & Support

- [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- [Issue Tracker](https://github.com/alvarodiez20/pysuricata/issues)
- [Star on GitHub](https://github.com/alvarodiez20/pysuricata)

## License

MIT License. See [LICENSE](https://github.com/alvarodiez20/pysuricata/blob/main/LICENSE) for details.
