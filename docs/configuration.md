---
title: Configuration Guide
description: Complete reference for all configuration options in PySuricata
---

# Configuration Guide

Complete guide to configuring PySuricata for your specific needs.

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

## Overview

PySuricata is highly configurable via the `ProfileConfig` class hierarchy:

```
ProfileConfig
├── compute: ComputeOptions  # Analysis parameters
└── render: RenderOptions    # Display parameters
```

## Three ways to configure, in order of effort

Most settings never need touching. When one does, reach for the cheapest form
that says what you mean.

=== "Keyword options"

    Seven settings without the nesting. This is the common case.

    ```python
    from pysuricata import profile

    report = profile(df, seed=42, correlations=False, title="Q4 2024")
    ```

    | keyword | sets |
    |---|---|
    | `chunk_size` | `compute.chunk_size` |
    | `columns` | `compute.columns` |
    | `sample` | `compute.numeric_sample_size` |
    | `correlations` | `compute.compute_correlations` |
    | `seed` | `compute.random_seed` |
    | `progress` | `compute.progress` |
    | `title` | `render.title` |

    Anything else is rejected by name, and the error points at the keyword that
    does what you were reaching for.

=== "A preset"

    One word for an intent, rather than working out which of twenty-two knobs to
    turn.

    ```python
    from pysuricata import profile

    report = profile(df, preset="fast")
    ```

    | | `fast` | `thorough` |
    |---|---|---|
    | `numeric_sample_size` | 5,000 | 50,000 |
    | `max_uniques` | 1,024 | 8,192 |
    | `top_k` | 20 | 100 |
    | `compute_correlations` | off | on |
    | `corr_threshold` | — | 0.0 (report everything) |

    A preset and keyword options combine, and the keywords win.

=== "A `ProfileConfig`"

    The escape hatch, for anything the other two do not reach.

    ```python
    from pysuricata import profile, ProfileConfig

    config = ProfileConfig()
    config.compute.max_uniques = 8_192
    config.compute.force_column_types = {"grade": "categorical"}

    report = profile(df, config=config)
    ```

    `config=` **cannot be combined** with `preset=` or the keyword options. It
    takes everything, and a caller who built one means it.

Precedence, lowest to highest: defaults, then the preset, then keyword options.

## ComputeOptions

Control data processing and analysis.

### Basic Parameters

**`chunk_size: int = 50_000`**

Rows per chunk when processing data.

- **Larger**: more memory, and usually *slower* — the sketch merges are
  superlinear in batch size, so one 200K-row batch costs more than four 50K ones
- **Smaller**: less memory, more per-chunk overhead
- **Recommended**: leave it alone. 50K is near the measured optimum across
  frame shapes; the useful range is roughly 25K–100K

```python
config.compute.chunk_size = 250_000
```

**`columns: Optional[List[str]] = None`**

Analyze only specific columns. If `None`, analyze all.

```python
config.compute.columns = ["age", "income", "city"]
```

**`random_seed: int | None = 0`**

Seed for the reservoir sampling and the row-hash sketches.

The default is `0`, not `None` — PySuricata is **reproducible unless you ask for
otherwise**, so re-running a report over unchanged data is a no-op rather than a
set of sampling wobbles. `compare()` relies on this. Set a different integer to
pin a different sample, or `None` for a fresh one each run.

```python
config.compute.random_seed = 42   # a different, still-fixed sample
config.compute.random_seed = None # non-deterministic
```

### Numeric Configuration

**`numeric_sample_size: int = 20_000`**

Reservoir sample size for quantiles and histograms.

- **Larger**: More accurate quantiles, more memory
- **Smaller**: Less memory, slightly less accurate
- **Recommended**: 10K-50K

```python
config.compute.numeric_sample_size = 50_000
```

**`max_uniques: int = 2_048`**

KMV sketch size for distinct count estimation.

- **Error**: \(\approx 1/\sqrt{k}\)
- **k=1024**: ~3.1% error
- **k=2048**: ~2.2% error (default)
- **k=4096**: ~1.6% error

```python
config.compute.max_uniques = 4_096  # More accurate
```

### Categorical Configuration

**`top_k: int = 50`**

Number of top values to track (Misra-Gries algorithm).

- **Larger**: More top values, more memory
- **Smaller**: Fewer top values, less memory
- **Guarantee**: All items with frequency > n/k found

```python
config.compute.top_k = 100  # Track top 100
```

### Progress Reporting

**`progress: bool | str | Callable = False`**

Report how far a long run has got. Four shapes:

| value | behaviour |
|---|---|
| `False` | silent (default) |
| `True` | always report |
| `"auto"` | report only when stderr is a terminal |
| a callable | called with chunks, rows and elapsed time |

Progress always goes to **stderr**, never stdout, so a piped
`pysuricata summarize data.csv > stats.json` stays parseable.

```python
from pysuricata import profile

report = profile("events.parquet", progress="auto")
```

### Type Inference

**`force_column_types: dict[str, str] | None = None`**

Override the inferred type for named columns. Valid values are `"numeric"`,
`"categorical"`, `"datetime"` and `"boolean"`.

Worth knowing on a streamed source: a numeric column holding few enough distinct
whole numbers is reclassified as categorical, and on a stream that decision is
made from the first batch and never revisited. `force_column_types` is how you
settle it.

```python
config.compute.force_column_types = {"grade": "categorical", "zip": "categorical"}
```

**`enable_auto_boolean_detection: bool = True`**

Treat a 0/1 numeric column as boolean. Three parameters tune it:

| | default | meaning |
|---|---|---|
| `boolean_detection_min_samples` | `100` | fewer values than this, and the column is left numeric |
| `boolean_detection_max_zero_ratio` | `0.80` | above this share of zeros, the column is left numeric — a mostly-zero counter is not a flag |
| `boolean_detection_require_name_pattern` | `False` | when `True`, only names like `is_`, `has_`, `can_` are eligible |

The name pattern is **off** by default, so detection runs on the values whatever
the column is called.

### Correlation Configuration

Correlation settings are available through the public `ComputeOptions` API.

**`compute_correlations: bool = True`**

Enable/disable pairwise correlation computation.

```python
from pysuricata import profile, ProfileConfig, ComputeOptions

config = ProfileConfig(compute=ComputeOptions(
    compute_correlations=False  # Disable for speed
))
report = profile(df, config=config)
```

**`corr_threshold: float = 0.5`**

Minimum |r| to report.

```python
from pysuricata import ComputeOptions, ProfileConfig
config = ProfileConfig(compute=ComputeOptions(
    corr_threshold=0.7  # Only strong correlations
))
```

**`corr_max_cols: int = 50`**

Maximum columns for correlation computation. Skip if exceeded.

```python
from pysuricata import ComputeOptions, ProfileConfig
config = ProfileConfig(compute=ComputeOptions(
    corr_max_cols=100  # Higher limit
))
```

**`corr_max_per_col: int = 10`**

Maximum correlations to show per column.

```python
from pysuricata import ComputeOptions, ProfileConfig
config = ProfileConfig(compute=ComputeOptions(
    corr_max_per_col=5  # Show top 5
))
```

### Checkpointing Configuration

For long-running profiles over millions of rows across many chunks, you can save
periodic lightweight states to disk so an interrupted job is not lost.

Five of the twenty-two options serve this one concern. They are also reachable
as a named group, which is usually the more legible way to write them:

```python
options = ProfileConfig().compute
options.checkpoint.every_n_chunks = 10
options.checkpoint.dir = "./checkpoints"
options.checkpoint.write_html = True
```

`options.checkpoint` is a view, not a nested object — the fields below are the
same fields, under shorter names.

**`checkpoint_every_n_chunks: int = 0`**

Number of chunks between checkpoints. Set to > 0 to enable checkpointing.

```python
config.compute.checkpoint_every_n_chunks = 10  # Checkpoint every 10 chunks
```

**`checkpoint_dir: str = None`**

Directory to save checkpoint files. Defaults to current working directory or the directory of the final HTML report.

```python
from pysuricata import ProfileConfig

config = ProfileConfig()
config.compute.checkpoint_dir = "/tmp/pysuricata_checkpoints"
```

**`checkpoint_prefix: str = "pysuricata_ckpt"`**

Prefix for checkpoint filenames.

```python
config.compute.checkpoint_prefix = "daily_sales_job"
```

**`checkpoint_max_to_keep: int = 3`**

Maximum number of recent checkpoint files to keep before rotating out older ones.

**`checkpoint_write_html: bool = False`**

Whether to write a preview HTML snapshot alongside the binary pickle state.

```python
config.compute.checkpoint_write_html = True  # Useful for monitoring progress
```

## RenderOptions

Control report display and formatting.

### Basic Parameters

**`title: Optional[str] = None`**

Custom report title. If `None`, uses "Data Profile Report".

```python
config.render.title = "Customer Data Analysis - Q4 2024"
```

**`description: Optional[str] = None`**

Markdown-formatted description shown at top of report.

```python
config.render.description = """
# Analysis Overview

Dataset contains customer transactions from **2024 Q4**.

Key metrics:
- 1.5M transactions
- 50K unique customers
"""
```

**`include_sample: bool = True`**

Whether the report shows a handful of sample rows.

```python
config.render.include_sample = False  # no sample table in the output
```

!!! warning "This is not a redaction switch"

    It removes the sample table, and only the sample table. Raw values still
    reach the page from four other places:

    | card | what it prints verbatim |
    |---|---|
    | categorical | the top-value labels |
    | categorical | *Shortest seen* and *Longest seen* |
    | numeric | the smallest and largest values |
    | datetime | the earliest and latest timestamps |

    Turning the sample off removes the largest block of raw data from a report
    and is worth doing on a frame you are circulating. It does not make the
    report safe to circulate. If that is what you need, profile a redacted
    frame, or use `columns=` to leave the sensitive ones out of the pass
    entirely — which also means they never enter an accumulator.

**`sample_rows: int = 10`**

How many rows the sample shows, when it is shown. Ignored if `include_sample`
is `False`.

```python
config.render.sample_rows = 20  # Show 20 rows
```

## Example Configurations

### Memory-Constrained Environment

```python
from pysuricata import ProfileConfig
config = ProfileConfig()
config.compute.chunk_size = 50_000  # Small chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.max_uniques = 1_024  # Smaller sketches
config.compute.top_k = 20  # Fewer top values
config.compute.compute_correlations = False  # Skip correlations
```

### Maximum Accuracy

```python
from pysuricata import ProfileConfig
config = ProfileConfig()
config.compute.numeric_sample_size = 100_000  # Large samples
config.compute.max_uniques = 8_192  # Large sketches
config.compute.top_k = 200  # Many top values
config.compute.corr_threshold = 0.0  # All correlations
```

### Speed Optimized

This is what `preset="fast"` already does; the long form is here for when you
want to vary one part of it.

```python
from pysuricata import ProfileConfig
config = ProfileConfig()
config.compute.numeric_sample_size = 5_000   # Small samples
config.compute.max_uniques = 1_024           # Smaller sketches
config.compute.top_k = 20                    # Few top values
config.compute.compute_correlations = False  # Skip the O(p^2) step
# chunk_size is deliberately absent: raising it costs memory and buys nothing.
```

### Reproducible Reports

```python
from datetime import datetime

from pysuricata import ProfileConfig

config = ProfileConfig()
config.compute.random_seed = 42  # Deterministic
config.render.title = f"Report generated {datetime.now()}"
```

### Production Data Quality Checks

```python
from pysuricata import ProfileConfig
# Only check specific columns
config = ProfileConfig()
config.compute.columns = ["customer_id", "transaction_amount", "timestamp"]
config.render.include_sample = False  # drop the sample table -- see the note above

# Generate stats only (no HTML)
from pysuricata import summarize
stats = summarize(df, config=config)

# Assert quality thresholds
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

## Deprecated names

`ReportConfig` is an alias for `ProfileConfig`. It warns on **use** rather than
on import, and it is removed in **0.3.0**:

```python
# DeprecationWarning on use, naming 0.3.0 as the release that removes it
from pysuricata import ReportConfig
```

```python
# the name to use
from pysuricata import ProfileConfig
```

`pysuricata.config.EngineConfig` is not deprecated and not part of the public
API — it is the engine's internal configuration, built from `ComputeOptions` by
the boundary and not meant to be constructed by hand. Some of its fields have no
`ProfileConfig` counterpart; that is deliberate, not an omission to work around.

See [Versioning](versioning.md) for what a deprecation commits us to.

## Environment Variables

Not currently supported. All configuration via code.

## Configuration Validation

Every invariant is checked in one place, `ComputeOptions.validate()`, and that
one place is called **twice**: once at construction, and again when the options
reach the engine.

```python
from pysuricata import ComputeOptions, ConfigurationError, ProfileConfig, profile

try:
    ComputeOptions(chunk_size=0)
except ValueError as e:
    print(e)  # chunk_size must be positive

config = ProfileConfig()
config.compute.chunk_size = -1  # accepted -- the dataclass is mutable
try:
    profile(df, config=config)
except ConfigurationError as e:
    print(e)  # invalid compute options: chunk_size must be positive
```

The second call is the one that matters, because mutating a config after
building it is the path most people take. Validating only at construction
guarded a door nobody walks through.

`ConfigurationError` subclasses `ValueError`, so `except ValueError` still
catches it.

## Performance Impact

| Parameter | Increase → | Impact |
|-----------|-----------|--------|
| `chunk_size` | ↑ | More memory, usually *slower* — see above |
| `numeric_sample_size` | ↑ | More accurate quantiles, more memory |
| `max_uniques` | ↑ | More accurate distinct, more memory |
| `top_k` | ↑ | More top values, more memory |
| `compute_correlations` | False | Much faster, less memory |

## See Also

- [Performance Tips](performance.md) - Optimization strategies
- [Advanced Features](advanced.md) - Advanced usage patterns
- [API Reference](api.md) - Complete API documentation
