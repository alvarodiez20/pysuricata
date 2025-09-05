# Usage

## Basic


```python
import pandas as pd
from pysuricata import profile

df = pd.read_csv("data.csv")
rep = profile(df)
rep.save_html("report.html")
```

## Streaming large in-memory data

```python
from pysuricata import profile, ReportConfig
import pandas as pd

cfg = ReportConfig()

# From an iterable/generator yielding pandas DataFrame chunks
def chunk_iter():
    for i in range(10):
        yield pd.read_csv(f"/data/part-{i}.csv")  # you pre-chunk externally

rep = profile((ch for ch in chunk_iter()), config=cfg)
rep.save_html("report.html")

```

### Programmatic summary

Ask for a compact JSON-like dictionary of stats:

```python
from pysuricata import summarize
summary = summarize(df)
print(summary["dataset"])           # rows_est, cols, missing_cells, duplicates, top-missing
print(summary["columns"]["amount"]) # per-column stats by type
```

### Processed bytes and timing

The report displays:
- Processed bytes (≈): total bytes handled across chunks (not peak RSS)
- Precise generation time in seconds (e.g., 0.02s)
