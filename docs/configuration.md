---
title: Configuration Guide
description: Complete reference for all configuration options in PySuricata
---

# Configuration Guide

Complete guide to configuring PySuricata for your specific needs.

## Overview

PySuricata is highly configurable via the `ReportConfig` class hierarchy:

```
ReportConfig
├── compute: ComputeOptions  # Analysis parameters
└── render: RenderOptions    # Display parameters
```

## Quick Start

```python
from pysuricata import profile, ReportConfig

# Create config
config = ReportConfig()

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

**`chunk_size: int = 200_000`**

Rows per chunk when processing data.

- **Larger**: Faster processing, more memory
- **Smaller**: Less memory, more overhead
- **Recommended**: 100K-500K for most datasets

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

**`uniques_sketch_size: int = 2_048`**

KMV sketch size for distinct count estimation.

- **Error**: \(\approx 1/\sqrt{k}\)
- **k=1024**: ~3.1% error
- **k=2048**: ~2.2% error (default)
- **k=4096**: ~1.6% error

```python
config.compute.uniques_sketch_size = 4_096  # More accurate
```

### Categorical Configuration

**`top_k_size: int = 50`**

Number of top values to track (Misra-Gries algorithm).

- **Larger**: More top values, more memory
- **Smaller**: Fewer top values, less memory
- **Guarantee**: All items with frequency > n/k found

```python
config.compute.top_k_size = 100  # Track top 100
```

### Correlation Configuration

Correlation settings are available through the public `ComputeOptions` API.

**`compute_correlations: bool = True`**

Enable/disable pairwise correlation computation.

```python
from pysuricata import profile, ReportConfig, ComputeOptions

config = ReportConfig(compute=ComputeOptions(
    compute_correlations=False  # Disable for speed
))
report = profile(df, config=config)
```

**`corr_threshold: float = 0.5`**

Minimum |r| to report.

```python
config = ReportConfig(compute=ComputeOptions(
    corr_threshold=0.7  # Only strong correlations
))
```

**`corr_max_cols: int = 50`**

Maximum columns for correlation computation. Skip if exceeded.

```python
config = ReportConfig(compute=ComputeOptions(
    corr_max_cols=100  # Higher limit
))
```

**`corr_max_per_col: int = 10`**

Maximum correlations to show per column.

```python
config = ReportConfig(compute=ComputeOptions(
    corr_max_per_col=5  # Show top 5
))
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
config = ReportConfig()
config.compute.chunk_size = 50_000  # Small chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.uniques_sketch_size = 1_024  # Smaller sketches
config.compute.top_k_size = 20  # Fewer top values
config.compute.compute_correlations = False  # Skip correlations
```

### Maximum Accuracy

```python
config = ReportConfig()
config.compute.numeric_sample_size = 100_000  # Large samples
config.compute.uniques_sketch_size = 8_192  # Large sketches
config.compute.top_k_size = 200  # Many top values
config.compute.corr_threshold = 0.0  # All correlations
```

### Speed Optimized

```python
config = ReportConfig()
config.compute.chunk_size = 500_000  # Large chunks
config.compute.numeric_sample_size = 10_000  # Small samples
config.compute.compute_correlations = False  # Skip correlations
config.compute.top_k_size = 20  # Few top values
```

### Reproducible Reports

```python
config = ReportConfig()
config.compute.random_seed = 42  # Deterministic
config.render.title = f"Report generated {datetime.now()}"
```

### Production Data Quality Checks

```python
# Only check specific columns
config = ReportConfig()
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
cfg = EngineConfig(chunk_size=200_000, sample_k=20_000)

# New way (recommended)
from pysuricata import ReportConfig
config = ReportConfig()
config.compute.chunk_size = 200_000
config.compute.numeric_sample_size = 20_000
```

## Environment Variables

Not currently supported. All configuration via code.

## Configuration Validation

Invalid configurations raise `ValueError`:

```python
config = ReportConfig()
config.compute.chunk_size = -1  # Invalid
# Raises: ValueError: chunk_size must be positive
```

## Performance Impact

| Parameter | Increase → | Impact |
|-----------|-----------|--------|
| `chunk_size` | ↑ | Faster, more memory |
| `numeric_sample_size` | ↑ | More accurate quantiles, more memory |
| `uniques_sketch_size` | ↑ | More accurate distinct, more memory |
| `top_k_size` | ↑ | More top values, more memory |
| `compute_correlations` | False | Much faster, less memory |

## See Also

- [Performance Tips](performance.md) - Optimization strategies
- [Advanced Features](advanced.md) - Advanced usage patterns
- [API Reference](api.md) - Complete API documentation

