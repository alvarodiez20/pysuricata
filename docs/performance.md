---
title: Performance Tips
description: Optimization strategies for PySuricata
---

# Performance Tips

Optimize PySuricata for your specific use case with these strategies.

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

## Quick Wins

### 1. Disable Correlations for Many Columns

For datasets with > 100 numeric columns, correlation computation is O(p²) and can be slow.

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.compute_correlations = False  # Skip correlations

report = profile(df, config=config)
```

The step is \(O(p^2)\) in numeric columns, so the saving grows with the square
of the width. Measure it on your own frame with `benchmarks/end_to_end.py`
rather than trusting a ratio typed on someone else's machine.

### 2. Start From a Preset

`preset="fast"` sets the four knobs that actually move the clock — sample size,
sketch size, top-k and correlations — in one word.

```python
from pysuricata import profile

report = profile(df, preset="fast")
```

Keyword options layer on top, so you can take the preset and put one thing back:

```python
from pysuricata import profile

report = profile(df, preset="fast", correlations=True)
```

!!! warning "Raising `chunk_size` is not a speed lever"

    It reads like one and it is not. The sketch merges are **superlinear in
    batch size**, so one 200,000-row batch costs more than four 50,000-row ones
    — you spend memory and get less throughput. 50,000 is near the measured
    optimum across frame shapes; the useful range is roughly 25K–100K.

    Change it to trade memory against the number of chunk boundaries, not to go
    faster.

### 3. Reduce Sample Sizes

Smaller samples are faster to process.

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.numeric_sample_size = 10_000  # Default: 20_000

report = profile(df, config=config)
```

**Trade-off**: Slightly less accurate quantiles

## Memory Optimization

### Memory-Constrained Environments

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.chunk_size = 50_000  # Small chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.max_uniques = 1_024  # Small sketches
config.compute.top_k = 20  # Fewer top values
config.compute.compute_correlations = False  # Skip correlations

report = profile(df, config=config)
```

Every one of those trades accuracy for footprint, and the sample size is the
one that costs most: quantile error goes as \(1/\sqrt{k}\), so dropping the
reservoir from 20,000 to 5,000 moves it from about ±0.7% to ±1.4%.

### Proof: `pysuricata check` under a 512 MB ceiling

"Bounded memory, so it fits in CI" is the reason to prefer this over a
profiler that loads the frame — until it is measured under an *enforced*
ceiling it is an argument, not a result ([#92](https://github.com/alvarodiez20/pysuricata/issues/92)).
`benchmarks/memory_bounded_check.py` runs `pysuricata check` — first
`--write-baseline`, then a comparison against it — inside a child cgroup
capped at a fixed memory limit, against a Parquet file sized past that limit.
The cgroup is the kernel primitive Docker's own `--memory` flag rests on: the
process is genuinely killed by the kernel if it crosses the ceiling, not just
observed afterward.

| budget | file on disk | rows × cols | peak, `--write-baseline` | peak, compare |
|---:|---:|---|---:|---:|
| 512 MB | 775 MB (1.5×) | 10.8M × 12 | 301 MB | 296 MB |
| 350 MB | 531 MB (1.5×) | 7.4M × 12 | 295 MB | 295 MB |

Both runs completed. The frame is deliberately text-heavy (mixed numeric,
category-string and timestamp columns) — the shape
[docs/adr/memory-budget.md](adr/memory-budget.md) flags as most likely to
break its model, since that model was fitted on numeric columns only. It does:
the ADR's formula predicts 119 MB for this shape, and actual peak usage was
roughly 2.5× that. The gap is a real finding about the model, not about
`check` — both runs still landed comfortably under budget, and peak stayed
flat (295–301 MB) as the file grew from 531 MB to 775 MB, which is the actual
claim under test: memory does not grow with file size.

One genuine bug came out of getting this measurement to pass at all:
`stream_parquet` was calling pyarrow with its default `pre_buffer=True`, which
prefetches row groups ahead of what the reader has asked for — a sensible
trade on remote storage, and pure retained memory for the local files this
reader always sees. A 300 MB file's read alone was enough to breach a 512
MB-shaped ceiling with prefetching on; disabling it (now the default in
`stream_parquet`) roughly halved the reader's own footprint.

Two caveats, stated rather than glossed over. First, this measurement
environment has no Docker daemon, so the ceiling is a cgroup v1 child group
rather than a literal container — the same primitive, one layer down.
`RLIMIT_AS` was tried first and rejected: it caps virtual address space, and
`import pysuricata` alone reserves about 420 MB of it (pandas' and pyarrow's
arena and BLAS reservations, almost none of it resident), which makes it an
unrepresentative ceiling for this purpose. Second, "larger than RAM" in the
original acceptance criterion means larger than the *runner's* memory — the
host actually measured on has abundant physical RAM, so the file is sized
past the configured *ceiling* (1.5×) rather than past the host's total
memory, which is the dimension a CI runner is actually fixed at.

```bash
python -m benchmarks.memory_bounded_check --budget-mb 512
```

### Monitor Memory Usage

This recipe needs `psutil`, which is not a runtime dependency — nothing in the
library imports it. Install it with `pip install pysuricata[system]`.

```python
import os

import psutil

from pysuricata import ProfileConfig, profile

config = ProfileConfig()
process = psutil.Process(os.getpid())
print(f"Memory before: {process.memory_info().rss / 1024**2:.1f} MB")

report = profile(df, config=config)

print(f"Memory after: {process.memory_info().rss / 1024**2:.1f} MB")
```

## Speed Optimization

### Profile Only Key Columns

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.columns = ["user_id", "amount", "timestamp"]

report = profile(df, config=config)
```

**Speed improvement**: Linear in number of columns

### Use Polars for Large Datasets

Polars can be faster than pandas for certain operations:

```python
from pysuricata import profile
import polars as pl

df = pl.read_csv("large_file.csv")
report = profile(df)  # Native polars support
```

### Parallelize with Dask (Advanced)

```python
from pysuricata import profile
import dask.dataframe as dd

# Load with Dask
ddf = dd.read_csv("large_file.csv")

# Convert partitions to generator
def partition_generator():
    for partition in ddf.partitions:
        yield partition.compute()

report = profile(partition_generator())
```

## Benchmarks

Figures are not published on this page, on purpose. Two claims have already had
to be retracted for being paired across sessions rather than measured in one
round-robin — "0.0.21 is 1.24x faster" was really 0.88x, a regression, and a
3.56x headline was really 2.48x.

Run the harness instead. It interleaves every tool and version across rounds,
reads the load average, refuses to run above one process per core without
`--force`, and labels anything under three rounds *Not quotable*:

```bash
python -m benchmarks.end_to_end --markdown results.md   # vs ydata/sweetviz/skimpy
python -m benchmarks.hotspots                           # where profile() spends its time
python -m benchmarks.kernels                            # per-kernel timings + memory roofline
```

### What the shape is

- **Time is linear in rows.** One pass, O(1) work per value, so
  \(T(n) \approx k \cdot n\).
- **Memory is flat in rows.** Four float64 columns cost the same above the
  import floor at 500,000 rows and at 8,400,000.
- **Memory is *not* flat in columns.** It grows at roughly 1.2 MB per column, so
  a 20,000 x 600 frame costs 797 MB against 52 MB for a 1,000,000 x 14 one on
  more cells. This
  is a known limit, tracked in
  [#207](https://github.com/alvarodiez20/pysuricata/issues/207).
- **`summarize()` skips rendering entirely**, so it is the faster path whenever
  you only want the numbers.

## Advanced Configuration

### For Maximum Speed

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.numeric_sample_size = 5_000   # Small samples
config.compute.max_uniques = 512             # Tiny sketches
config.compute.top_k = 10                    # Few top values
config.compute.compute_correlations = False  # Skip the O(p^2) step
config.render.include_sample = False         # skip the sample table

report = profile(df, config=config)
```

`chunk_size` is deliberately not in that list — see the warning under Quick
Wins. This is `preset="fast"` with the sketches turned down further.

### For Maximum Accuracy

This is `preset="thorough"`, turned up. Note that `chunk_size` is absent again,
and for a different reason this time: **chunked results equal unchunked
results**. That invariant is asserted in `benchmarks/accuracy.py`, so no chunk
size is more accurate than another.

```python
from pysuricata import ProfileConfig, profile
config = ProfileConfig()
config.compute.numeric_sample_size = 100_000  # Large samples
config.compute.max_uniques = 8_192  # Large sketches
config.compute.top_k = 200  # Many top values
config.compute.corr_threshold = 0.0  # All correlations

report = profile(df, config=config)
```

## Profiling PySuricata

Use Python's profiler to find bottlenecks — and then check the answer:

!!! warning "`cProfile` over-weights kernels that make many small calls"

    It charges per Python call. It ranked the reservoir sampler at ~30% of self
    time on this codebase; swapping in a 5x-faster one moved wall clock by 4%.
    Confirm any ranking against wall clock **with the profiler off** before
    acting on it.


```python
from pysuricata import profile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

report = profile(df)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Common Bottlenecks

### 1. Correlation Computation

**Symptom**: Slow for > 100 numeric columns  
**Solution**: Disable correlations or increase threshold

### 2. Many Categorical Columns

**Symptom**: Slow with > 50 categorical columns  
**Solution**: Reduce `top_k` and `max_uniques`, or reach for `preset="fast"`

### 3. Very Wide Datasets (> 1000 columns)

**Symptom**: Slow overall  
**Solution**: Profile in batches, combine reports manually

### 4. Loading Before Profiling

**Symptom**: memory spikes to the size of the file before profiling starts  
**Solution**: hand `profile()` the **path**, not `pd.read_parquet(path)`. The
file is read a batch at a time and never exists as one frame — 307 MB against
581 MB on a 180 MB Parquet file. See
[Arrow, Parquet and DuckDB](data-sources.md).

## Production Optimization

### Scheduled Reports

For regular reporting, optimize for speed:

```python
from datetime import date
from pathlib import Path

from pysuricata import ProfileConfig, profile

# Fast config for nightly reports
config = ProfileConfig()
config.compute.compute_correlations = False
config.compute.numeric_sample_size = 10_000
config.render.title = f"Daily Report - {date.today()}"

report = profile(df, config=config)

Path("reports").mkdir(exist_ok=True)
report.save_html(f"reports/daily_{date.today()}.html")
```

### CI/CD Quality Checks

`summarize()` skips rendering, so it is the faster path when you only want the
numbers:

```python
from pysuricata import summarize

stats = summarize(df)  # no HTML built

assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

If the check is a build gate rather than an assertion inside your own code, the
CLI does the same single pass and exits non-zero on a breach:

```bash
pysuricata check data.parquet --baseline baseline.json --max-missing-pct 5
```

See [Gating CI on drift](data-checks.md).

## See Also

- [Configuration Guide](configuration.md) - All configuration options
- [Examples](examples.md) - Real-world use cases
- [Advanced Features](advanced.md) - Power user tips
