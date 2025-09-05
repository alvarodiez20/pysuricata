---
title: Architecture & Internals
description: How pysuricata generates EDA reports at scale — chunked ingestion, accumulators, sketches, rendering, and configuration.
---

# Architecture & Internals

This document explains how `pysuricata` profiles data efficiently and renders a self‑contained HTML report.


## Overview

```
┌────────┐   ┌──────────────┐   ┌────────────────────┐   ┌──────────────┐
│ Source │ → │ Chunk iterator│ → │ Typed accumulators │ → │ HTML template │
└────────┘   └──────────────┘   └────────────────────┘   └──────────────┘
     In-memory DataFrame(s)          numeric / categorical / datetime / boolean
```

## Chunk ingestion

- Iterable of pandas DataFrames: consumed as-is.
- Single pandas DataFrame: treated as one chunk (or sliced by rows if you pre-split it).

## Typed accumulators

Each column kind is handled by a specialized accumulator with small, mergeable state:

- NumericAccumulator
  - Moments (n, mean, M2, M3, M4) via Welford/Pébay (exact, mergeable)
  - Min/Max, zeros, negatives, ±inf, missing counters
  - Reservoir sample (default 20k) for quantiles, histograms, and shape hints
  - KMV (K‑Minimum Values) for approximate distinct
  - Misra–Gries top‑k for discrete integer‑like columns (on demand)
  - Heaping %, granularity (decimals/step), bimodality hint
  - Streaming correlation chips (optional, numeric vs numeric)
  - Extremes with row indices (min / max tracked across chunks)

- CategoricalAccumulator
  - KMV for distinct, Misra–Gries for top‑k
  - String length stats (avg, p90), empty strings
  - Case/trim variant distinctness

- DatetimeAccumulator
  - Min/Max timestamps (ns), counts by hour / day of week / month
  - Monotonicity hints

- BooleanAccumulator
  - True/False counts, missing, imbalance hints

All accumulators expose `update(...)` and `finalize() → SummaryDataclass` for rendering.

## Streaming correlations

`_StreamingCorr` maintains pairwise sufficient statistics for numeric columns and emits top absolute correlations above a configurable threshold for each column.

## Rendering pipeline

1. Infer column kinds from the first chunk.
2. Build accumulators and consume the first chunk.
3. Consume remaining chunks, update streaming correlations if enabled.
4. Compute summary metrics (missingness, duplicates, constant columns, etc.).
5. Render the template with:
   - Summary cards (rows, cols, processed bytes (≈), missing/duplicates)
   - Top missing columns
   - Variables (cards by type)
   - Optional dataset sample

The template is a single file with inline CSS/JS/images to produce a portable HTML.

## Configuration

`ReportConfig` controls chunk size, sample sizes, distinct/top‑k sketch sizes, and correlation settings, plus logging and checkpointing.

Key fields:

- `chunk_size`: rows per chunk (default 200k)
- `numeric_sample_k`: reservoir size for numeric sampling (default 20k)
- `uniques_k`: KMV sketch size (default 2048)
- `topk_k`: Misra–Gries capacity (default 50)
- `compute_correlations`: enable/disable streaming correlation chips
- `corr_threshold`, `corr_max_cols`, `corr_max_per_col`
- `include_sample`, `sample_rows`
- Checkpointing: write periodic pickles and (optional) partial HTML

## Processed bytes & timing

The report shows:
- Processed bytes (≈): cumulative bytes processed across chunks (not process RSS)
- Precise generation time in seconds (e.g., `0.02s`)

## Security & correctness notes

- HTML escaping: column names, labels, and chip text are escaped before rendering.
- Missing/inf handling: NaN and ±Inf are excluded from moment calculations but reported separately.
- Approximation badges: estimates are marked with `(≈)` or an `approx` badge.

## Extending

- Add backends: polars/Arrow datasets or DuckDB scans can be plugged into the chunk iterator.
- Add quantile sketches: t‑digest or KLL can replace the default reservoir for better tail accuracy.
- Add new sections: drift comparisons, profile JSON export to file, CLI wrapper.

