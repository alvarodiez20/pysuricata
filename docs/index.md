# PySuricata

Generate clean, self-contained EDA reports for pandas and large CSV/Parquet files.

!!! tip
    Works great on small and medium datasets; for very large datasets, sample first.

## Features
- Summary stats, missingness, duplicates
- Numeric, categorical, datetime, and boolean cards with inline SVG charts
- Out-of-core streaming (CSV/Parquet) with low peak memory
- Approximate distinct counts and heavy hitters for large columns
- Streaming correlations for numeric columns
- Self-contained HTML export (inline CSS/JS/images)

## Quick links
- [Installation](install.md)
- [Usage](usage.md)

## Quick start

=== "Classic (in-memory)"
    ```python
    import pandas as pd
    from pysuricata.report import generate_report

    df = pd.read_csv("/path/to/data.csv")
    html = generate_report(df, report_title="My EDA")
    ```

=== "Streaming (v2)"
    ```python
    from pysuricata.report_v2 import generate_report, ReportConfig

    html = generate_report(
        source="/path/to/data.parquet",  # or .csv
        config=ReportConfig(chunk_size=250_000, compute_correlations=True),
        output_file="report.html",
    )
    ```

## How it works (v2)

- Reads input in chunks (pandas for CSV, pyarrow for Parquet) and feeds type-specific accumulators.
- Numeric accumulators maintain Welford/Pébay moments, a reservoir sample, and KMV distinct.
- Categorical accumulators use Misra–Gries for top-k and KMV for distinct.
- Datetime accumulators count by hour/day/month and keep min/max.
- A lightweight streaming correlation estimator tracks Pearson r for numeric pairs.
- The template renders a self-contained HTML with precise duration (e.g., 0.02s) and processed bytes (≈).

See also: Numeric analysis details in numeric_var.md.
