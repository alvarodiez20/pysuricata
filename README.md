# PySuricata

[![Build Status](https://github.com/alvarodiez20/pysuricata/workflows/CI/badge.svg)](https://github.com/alvarodiez20/pysuricata/actions)
[![PyPI version](https://img.shields.io/pypi/v/pysuricata.svg)](https://pypi.org/project/pysuricata/)
[![Python versions](https://img.shields.io/pypi/pyversions/pysuricata.svg)](https://github.com/alvarodiez20/pysuricata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/alvarodiez20/pysuricata/branch/main/graph/badge.svg)](https://codecov.io/gh/alvarodiez20/pysuricata)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://alvarodiez20.github.io/pysuricata/)
[![Downloads](https://static.pepy.tech/badge/pysuricata)](https://pepy.tech/project/pysuricata)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-alvarodiez20-blue?logo=linkedin)](https://www.linkedin.com/in/alvarodiez20/)

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/pysuricata/static/images/logo_suricata_transparent.png" alt="PySuricata Logo" width="300">

  <h3>Single-pass exploratory data analysis, built on streaming algorithms</h3>

  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="#guarding-a-dataset-in-ci">CI Checks</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/">Documentation</a> •
    <a href="https://alvarodiez20.github.io/pysuricata/examples/">Examples</a>
  </p>
</div>

---

## What It Does

PySuricata reads your data **once**, in chunks, and produces a self-contained HTML
report — per-column statistics, histograms, correlations, missing-value analysis
and outlier detection in a single file with no external assets.

Because nothing is held but the accumulators, memory is bounded by your chunk
size rather than by the dataset. The same pass also produces a machine-readable
summary, so the tool that explores your data can also **guard it in CI**.

## Quick Start

```bash
uv add pysuricata        # or: pip install pysuricata
```

```python
import pandas as pd
from pysuricata import profile

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

report = profile(df)
report.save_html("titanic_report.html")
```

<div align="center">
  <img src="https://raw.githubusercontent.com/alvarodiez20/pysuricata/main/docs/assets/report-screenshot.png" alt="The top of a PySuricata report: dataset summary, column-type breakdown and the most incomplete columns" width="820">
</div>

**[▶ See a live example report →](https://alvarodiez20.github.io/pysuricata/assets/titanic_report.html)**

## Guarding a Dataset in CI

`summarize()` returns the same numbers as the report, as a plain mapping with a
`schema_version`, so a pipeline can assert on them:

```python
import numpy as np
import pandas as pd
from pysuricata import summarize

frame = pd.DataFrame({"age": np.arange(500.0), "city": ["ES", "FR"] * 250})
stats = summarize(frame)

assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["columns"]["age"]["mean"] > 0
```

Approximate figures say so and carry their error. The duplicate count is
`rows - distinct`, so the whole error of the distinct estimate lands on a much
smaller number — it is suppressed to `0` when it cannot be told apart from
zero, and `duplicate_rows_uncertainty` gives the bound:

```python
import numpy as np
import pandas as pd
from pysuricata import summarize

frame = pd.DataFrame({"id": np.arange(1_000)})
dataset = summarize(frame)["dataset"]

# 0 with an uncertainty of 0 is "exactly none"; 0 with an uncertainty of
# 2,201 is "nothing resolvable below about 2,201".
print(dataset["duplicate_rows_est"], dataset["duplicate_rows_uncertainty"])
```

Or skip Python entirely — `pysuricata check` compares a file against a stored
baseline and exits non-zero when a threshold is crossed (`0` pass, `1`
threshold crossed, `2` could not run):

```bash
pysuricata check data.csv --write-baseline baseline.json   # record
pysuricata check data.csv --baseline baseline.json \
    --max-missing-pct 5 --min-rows 1000 --fail-on-new-column
```

`compare()` is the same comparison from Python, returning a `Comparison` whose
`to_dict()` holds the findings.

## Reading Data

pandas and polars are native. Anything that streams in batches works without
loading the file:

```python
import pandas as pd
from pysuricata import profile

def read_in_chunks():
    """Any generator of frames — Parquet parts, a DB cursor, an API page."""
    for _ in range(3):
        yield pd.DataFrame({"value": range(1_000)})

report = profile(read_in_chunks())
```

Parquet files, PyArrow tables and DuckDB relations are streamed in batches too,
so a dataset larger than RAM never lands in it.

## Configuration

Common options are keywords — no config object needed. The full set is
`chunk_size`, `columns`, `sample`, `correlations`, `seed`, `title` and
`progress`:

```python
import numpy as np
import pandas as pd
from pysuricata import profile

frame = pd.DataFrame({"x": np.arange(2_000.0)})
report = profile(frame, chunk_size=100_000, seed=42, sample=5_000)
```

Two presets bundle the usual trade-off:

```python
import numpy as np
import pandas as pd
from pysuricata import summarize

frame = pd.DataFrame({"x": np.arange(2_000.0)})
quick = summarize(frame, preset="fast")       # smaller sketches, no correlations
careful = summarize(frame, preset="thorough") # larger sketches, all correlations
```

`ProfileConfig` is there when you want to set everything at once:

```python
import numpy as np
import pandas as pd
from pysuricata import ComputeOptions, ProfileConfig, RenderOptions, profile

frame = pd.DataFrame({"x": np.arange(2_000.0)})
config = ProfileConfig(
    compute=ComputeOptions(chunk_size=100_000, corr_threshold=0.5),
    render=RenderOptions(title="My Analysis"),
)
report = profile(frame, config=config)
```

An unknown keyword is rejected with the name of the one that works, rather than
being silently ignored. See the
[Configuration Guide](https://alvarodiez20.github.io/pysuricata/configuration/)
for the full set.

## How It Works

Well-known streaming algorithms, one pass, bounded memory:

| Algorithm | Purpose | Time | Space |
|-----------|---------|------|-------|
| **Welford/Pébay** | Exact mean, variance, skewness, kurtosis | O(1) per value | O(1) |
| **KMV sketch** | Distinct count estimation | O(log k) per value | O(k) |
| **Misra-Gries** | Top-k frequent values | O(1) amortized | O(k) |
| **Reservoir sampling** | Uniform random sample for quantiles | O(1) per value | O(s) |

*k = sketch size (default 2,048), s = numeric sample size (default 20,000),
chunk size 50,000 rows.*

Moments are exact. Distinct counts, top-k and quantiles are approximate, and
the report labels them as such — an estimate is never printed as though it were
counted.

Accumulators are **mergeable and order-independent**, so a chunked run gives
the same answer as an unchunked one. That invariant is asserted in
`benchmarks/accuracy.py`, not assumed.

## What's in a Report

Per column, by inferred type:

- **Numeric** — moments, quantiles, histogram (linear or log, 10/25/50 bins),
  outliers by IQR and modified z-score, correlations
- **Categorical** — top values, distinct count, entropy, label-length
  distribution, normalisation collisions
- **DateTime** — span, cadence, hour/day/month distributions, monotonicity
- **Boolean** — true/false split, imbalance flag

Plus dataset-level rows, columns, memory, missing cells and duplicate rows.

## CLI

```bash
pysuricata profile data.csv --output report.html   # HTML report
pysuricata summarize data.csv                      # JSON statistics
pysuricata check data.csv --baseline baseline.json # CI gate, exit code
```

## Documentation

- [Quick Start](https://alvarodiez20.github.io/pysuricata/quickstart/)
- [User Guide](https://alvarodiez20.github.io/pysuricata/usage/)
- [Configuration](https://alvarodiez20.github.io/pysuricata/configuration/)
- [API Reference](https://alvarodiez20.github.io/pysuricata/api/)
- [Statistical Methods](https://alvarodiez20.github.io/pysuricata/stats/overview/)
- [The `summarize()` schema](https://alvarodiez20.github.io/pysuricata/summary-schema/)
- [Examples](https://alvarodiez20.github.io/pysuricata/examples/)

The package ships `py.typed`, so the public API type-checks in your editor.

## Contributing

Contributions are welcome. See the
[Contributing Guide](https://alvarodiez20.github.io/pysuricata/contributing/).

```bash
git clone https://github.com/alvarodiez20/pysuricata.git
cd pysuricata
uv sync --dev
uv run pytest -m "not benchmark"
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built using algorithms from:

- Welford, B.P. (1962) — Streaming moments
- Pébay, P. (2008) — Parallel merging of moments
- Bar-Yossef, Z. et al. (2002) — KMV distinct count estimation
- Misra, J. & Gries, D. (1982) — Streaming heavy hitters

Named after **suricatas (meerkats)** — small, vigilant animals that work
cooperatively and thrive in harsh environments with limited resources.
