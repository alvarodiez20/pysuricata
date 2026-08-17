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

## Quick Start

```python
from pysuricata import profile, ProfileConfig

# Create config
config = ProfileConfig()

# Customize settings
config.compute.chunk_size = 250_000
config.compute.random_seed = 42
config.compute.compute_correlations = True

# Generate report
report = profile(df, config=config)
```

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

**`random_seed: Optional[int] = None`**

Random seed for deterministic sampling. Set for reproducibility.

```python
config.compute.random_seed = 42  # Same report every run
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

For long-running profiles overriding millions of rows across many chunks, you can save periodic lightweight states to disk to prevent data loss.

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

Include sample rows in report.

```python
config.render.include_sample = False  # Exclude sample
```

**`sample_rows: int = 10`**

Number of sample rows to show (if `include_sample=True`).

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

```python
from pysuricata import ProfileConfig
config = ProfileConfig()
config.compute.chunk_size = 500_000  # Large chunks
config.compute.numeric_sample_size = 10_000  # Small samples
config.compute.compute_correlations = False  # Skip correlations
config.compute.top_k = 20  # Few top values
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
config.render.include_sample = False  # No PII in reports

# Generate stats only (no HTML)
from pysuricata import summarize
stats = summarize(df, config=config)

# Assert quality thresholds
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

## Legacy EngineConfig

Older versions used `EngineConfig`. It's still supported but deprecated.

```python
# Old way (deprecated)
from pysuricata.config import EngineConfig
cfg = EngineConfig(chunk_size=50_000, numeric_sample_k=20_000)

# New way (recommended)
from pysuricata import ProfileConfig

config = ProfileConfig()
config.compute.chunk_size = 50_000
config.compute.numeric_sample_size = 20_000
```

## Environment Variables

Not currently supported. All configuration via code.

## Configuration Validation

Invalid configurations raise `ValueError`:

```python
from pysuricata import ProfileConfig
config = ProfileConfig()
config.compute.chunk_size = -1  # Invalid
# Raises: ValueError: chunk_size must be positive
```

## Performance Impact

| Parameter | Increase → | Impact |
|-----------|-----------|--------|
| `chunk_size` | ↑ | Faster, more memory |
| `numeric_sample_size` | ↑ | More accurate quantiles, more memory |
| `max_uniques` | ↑ | More accurate distinct, more memory |
| `top_k` | ↑ | More top values, more memory |
| `compute_correlations` | False | Much faster, less memory |

## See Also

- [Performance Tips](performance.md) - Optimization strategies
- [Advanced Features](advanced.md) - Advanced usage patterns
- [API Reference](api.md) - Complete API documentation
