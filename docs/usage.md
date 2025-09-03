# Usage

## Classic (in-memory DataFrame)

```python
import pandas as pd
from pysuricata.report import generate_report

df = pd.read_csv("data.csv")
html = generate_report(df, report_title="My EDA")
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html)
```

## Streaming (v2) for large CSV/Parquet

```python
from pysuricata.report_v2 import generate_report, ReportConfig

cfg = ReportConfig(
    chunk_size=250_000,
    compute_correlations=True,  # enable streaming correlation chips for numeric columns
)

# From a file path (CSV/Parquet)
html = generate_report("/data/big.parquet", config=cfg, output_file="report.html")

# From a DataFrame in-memory (single chunk)
import pandas as pd
df = pd.read_csv("/data/sample.csv")
html = generate_report(df, config=cfg)
```

### Programmatic summary

Ask the generator to return a compact JSON-like dictionary alongside the HTML:

```python
html, summary = generate_report("/data/big.csv", config=cfg, return_summary=True)
print(summary["dataset"])           # rows_est, cols, missing_cells, duplicates, top-missing
print(summary["columns"]["amount"]) # per-column stats by type
```

### Processed bytes and timing

The v2 report displays:
- Processed bytes (≈): total bytes handled across chunks (not peak RSS)
- Precise generation time in seconds (e.g., 0.02s)

### Chart builder helpers

You can also render the small SVGs programmatically:

```python
from pysuricata.report_v2 import (
    build_hist_svg_with_axes,
    build_cat_bar_svg,
    build_dt_line_svg,
)

hist_svg = build_hist_svg_with_axes(df["amount"], bins=25)
cat_svg = build_cat_bar_svg(df["country"], top=10)
dt_svg = build_dt_line_svg(df["ts"], bins=60)
```
