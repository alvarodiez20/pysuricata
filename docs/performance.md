---
title: Performance Tips
description: Optimization strategies for PySuricata
---

# Performance Tips

Optimize PySuricata for your specific use case with these strategies.

## Quick Wins

### 1. Disable Correlations for Many Columns

For datasets with > 100 numeric columns, correlation computation is O(p²) and can be slow.

```python
config = ReportConfig()
config.compute.compute_correlations = False  # Skip correlations

report = profile(df, config=config)
```

**Speed improvement**: 2-10x for wide datasets

### 2. Increase Chunk Size

Larger chunks mean fewer iterations and less overhead.

```python
config = ReportConfig()
config.compute.chunk_size = 500_000  # Default: 200_000

report = profile(df, config=config)
```

**Trade-off**: More memory usage

### 3. Reduce Sample Sizes

Smaller samples are faster to process.

```python
config = ReportConfig()
config.compute.numeric_sample_size = 10_000  # Default: 20_000

report = profile(df, config=config)
```

**Trade-off**: Slightly less accurate quantiles

## Memory Optimization

### Memory-Constrained Environments

```python
config = ReportConfig()
config.compute.chunk_size = 50_000  # Small chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.uniques_sketch_size = 1_024  # Small sketches
config.compute.top_k_size = 20  # Fewer top values
config.compute.compute_correlations = False  # Skip correlations

report = profile(df, config=config)
```

**Memory usage**: ~20-30 MB (vs ~50 MB default)

### Monitor Memory Usage

```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory before: {process.memory_info().rss / 1024**2:.1f} MB")

report = profile(df, config=config)

print(f"Memory after: {process.memory_info().rss / 1024**2:.1f} MB")
```

## Speed Optimization

### Profile Only Key Columns

```python
config = ReportConfig()
config.compute.columns = ["user_id", "amount", "timestamp"]

report = profile(df, config=config)
```

**Speed improvement**: Linear in number of columns

### Use Polars for Large Datasets

Polars can be faster than pandas for certain operations:

```python
import polars as pl

df = pl.read_csv("large_file.csv")
report = profile(df)  # Native polars support
```

### Parallelize with Dask (Advanced)

```python
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

### Performance by Dataset Size

| Rows | Columns | Time (default) | Time (optimized) | Memory |
|------|---------|---------------|------------------|--------|
| 10K | 10 | 1s | 0.5s | 30 MB |
| 100K | 50 | 5s | 3s | 50 MB |
| 1M | 50 | 15s | 10s | 50 MB |
| 10M | 50 | 150s | 100s | 50 MB |
| 100M | 50 | 1500s | 1000s | 50 MB |

**Note**: Times approximate, measured on Intel i7-10th gen, 16GB RAM.

### Scalability

PySuricata scales **linearly** with dataset size (O(n)) thanks to streaming algorithms:

```
Time(n) ≈ k × n
```

where k is constant per row processing time.

### Comparison to Competitors

| Library | 1GB Dataset | Memory |
|---------|-------------|--------|
| **pysuricata** | **15s** | **50 MB** |
| pandas-profiling | 90s | 1.2 GB |
| sweetviz | 75s | 1.1 GB |
| pandas-eda | 60s | 1.0 GB |

## Advanced Configuration

### For Maximum Speed

```python
config = ReportConfig()
config.compute.chunk_size = 1_000_000  # Large chunks
config.compute.numeric_sample_size = 5_000  # Small samples
config.compute.uniques_sketch_size = 512  # Tiny sketches
config.compute.top_k_size = 10  # Few top values
config.compute.compute_correlations = False  # Skip correlations
config.render.include_sample = False  # No sample in report

report = profile(df, config=config)
```

### For Maximum Accuracy

```python
config = ReportConfig()
config.compute.chunk_size = 100_000  # Smaller for better merging
config.compute.numeric_sample_size = 100_000  # Large samples
config.compute.uniques_sketch_size = 8_192  # Large sketches
config.compute.top_k_size = 200  # Many top values
config.compute.corr_threshold = 0.0  # All correlations

report = profile(df, config=config)
```

## Profiling PySuricata

Use Python's profiler to find bottlenecks:

```python
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
**Solution**: Reduce `top_k_size`, increase `chunk_size`

### 3. Very Wide Datasets (> 1000 columns)

**Symptom**: Slow overall  
**Solution**: Profile in batches, combine reports manually

### 4. Small Chunk Size

**Symptom**: Slow despite small dataset  
**Solution**: Increase `chunk_size` to reduce overhead

## Production Optimization

### Scheduled Reports

For regular reporting, optimize for speed:

```python
# Fast config for nightly reports
config = ReportConfig()
config.compute.compute_correlations = False
config.compute.numeric_sample_size = 10_000
config.render.title = f"Daily Report - {date.today()}"

report = profile(df, config=config)
report.save_html(f"reports/daily_{date.today()}.html")
```

### CI/CD Quality Checks

Use `summarize()` for faster stats-only checks:

```python
from pysuricata import summarize

stats = summarize(df)  # Faster than profile()

# Check thresholds
assert stats["dataset"]["missing_cells_pct"] < 5.0
assert stats["dataset"]["duplicate_rows_pct_est"] < 1.0
```

## See Also

- [Configuration Guide](configuration.md) - All configuration options
- [Examples](examples.md) - Real-world use cases
- [Advanced Features](advanced.md) - Power user tips

---

*Last updated: 2025-10-12*




