---
title: Architecture & Internals
description: How pysuricata generates EDA reports at scale — chunked ingestion, accumulators, sketches, rendering, and configuration.
---

# Architecture & Internals

How `pysuricata` profiles data efficiently and renders a self-contained HTML report.

## High-Level Pipeline

```mermaid
flowchart LR
    A["Data Source"] --> B["Chunk Iterator"]
    B --> C["Typed Accumulators"]
    C --> D["Summary Metrics"]
    D --> E["HTML Renderer"]

    style A fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    style B fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    style C fill:#A5D6A7,stroke:#2E7D32,color:#1B5E20
    style D fill:#81C784,stroke:#2E7D32,color:#1B5E20
    style E fill:#66BB6A,stroke:#2E7D32,color:#fff
```

**Data Sources** → pandas DataFrames, polars DataFrames, or any iterable of DataFrames (for streaming).

**Chunk Iterator** → If a single DataFrame is passed, it is treated as one chunk. Generators are consumed chunk-by-chunk to bound memory.

**Typed Accumulators** → Each column is assigned a specialized accumulator based on its inferred type. All accumulators are streaming: they accept one chunk at a time and maintain bounded state.

**Summary Metrics** → After all chunks are consumed, accumulators are finalized and dataset-wide metrics (missingness, duplicates, etc.) are computed.

**HTML Renderer** → A single-file Jinja2 template with inline CSS/JS produces a portable, self-contained HTML report.

---

## Accumulator Architecture

```mermaid
classDiagram
    class BaseAccumulator {
        +name: str
        +count: int
        +missing: int
        +update(chunk)
        +finalize() Summary
    }

    class NumericAccumulator {
        +StreamingMoments
        +ReservoirSampler
        +KMV sketch
        +MisraGries top-k
        +ExtremeTracker
    }

    class CategoricalAccumulator {
        +KMV sketch × 3
        +MisraGries top-k
        +String length stats
    }

    class DatetimeAccumulator {
        +min/max timestamps
        +hour/weekday/month counts
        +monotonicity tracker
    }

    class BooleanAccumulator {
        +true_count
        +false_count
    }

    BaseAccumulator <|-- NumericAccumulator
    BaseAccumulator <|-- CategoricalAccumulator
    BaseAccumulator <|-- DatetimeAccumulator
    BaseAccumulator <|-- BooleanAccumulator
```

Each accumulator follows the same interface:

1. **`update(chunk)`** — process a batch of values, update internal state
2. **`finalize()`** — compute final statistics from accumulated state

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant profile as profile()
    participant Infer as Type Inference
    participant Acc as Accumulators
    participant Corr as Correlations
    participant Render as HTML Renderer

    User->>profile: DataFrame / generator
    profile->>Infer: First chunk
    Infer-->>profile: Column types
    profile->>Acc: Create typed accumulators

    loop Each chunk
        profile->>Acc: update(chunk)
        profile->>Corr: update pairs (optional)
    end

    profile->>Acc: finalize()
    Acc-->>profile: Per-column summaries
    profile->>Corr: finalize()
    Corr-->>profile: Correlation matrix

    profile->>Render: Summaries + config
    Render-->>User: Report (HTML)
```

---

## Streaming Algorithms

Each accumulator uses algorithms chosen for **O(1) per-value update** and **bounded memory**:

```mermaid
flowchart TB
    subgraph Numeric["Numeric Accumulator"]
        N1["Welford/Pébay<br/>mean, var, skew, kurt<br/>O(1) space"]
        N2["Reservoir Sampling<br/>quantiles, histograms<br/>O(s) space"]
        N3["KMV Sketch<br/>distinct count<br/>O(k) space"]
        N4["Misra-Gries<br/>top-k values<br/>O(k) space"]
        N5["Extreme Tracker<br/>min/max with indices<br/>O(k) space"]
    end

    subgraph Categorical["Categorical Accumulator"]
        C1["KMV × 3<br/>distinct: original, lower, trimmed"]
        C2["Misra-Gries<br/>top-k values"]
        C3["String Length<br/>avg, p90"]
    end

    subgraph DateTime["DateTime Accumulator"]
        D1["Min/Max<br/>timestamps"]
        D2["Counters<br/>hour/weekday/month"]
        D3["Monotonicity<br/>pair comparison"]
    end

    subgraph Boolean["Boolean Accumulator"]
        B1["Counters<br/>true/false/missing"]
    end

    style Numeric fill:#E8F5E9,stroke:#2E7D32
    style Categorical fill:#FFF3E0,stroke:#E65100
    style DateTime fill:#E3F2FD,stroke:#1565C0
    style Boolean fill:#F3E5F5,stroke:#6A1B9A
```

---

## Rendering Pipeline

```mermaid
flowchart TB
    A["Finalized Summaries"] --> B["Dataset-Level Metrics"]
    B --> C["Jinja2 Template"]
    C --> D["Inline CSS + JS"]
    C --> E["Summary Cards"]
    C --> F["Variable Cards"]
    C --> G["Sample Table"]
    D --> H["Single HTML File"]
    E --> H
    F --> H
    G --> H

    style H fill:#66BB6A,stroke:#2E7D32,color:#fff
```

The template produces a **single portable HTML file** — no external dependencies, no server required.

**Summary cards** show: rows, columns, processed bytes, missing %, duplicates %.

**Variable cards** are rendered per-type with SVG charts, statistics, and quality flags.

### Shared Utilities

| Module | Functions | Purpose |
|--------|-----------|---------|
| `render/svg_utils.py` | `safe_col_id`, `nice_ticks`, `fmt_tick`, `svg_empty` | SVG chart helpers |
| `render/format_utils.py` | `human_bytes`, `fmt_num`, `fmt_compact` | Number formatting |

---

## Configuration

`ReportConfig` controls all behavior:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 200,000 | Rows per chunk |
| `numeric_sample_size` | 20,000 | Reservoir size for quantiles |
| `uniques_sketch_size` | 2,048 | KMV sketch size |
| `top_k_size` | 50 | Misra-Gries capacity |
| `compute_correlations` | `True` | Enable/disable correlation chips |
| `corr_threshold` | 0.5 | Minimum \|r\| to display |
| `random_seed` | `None` | Deterministic sampling |
| `include_sample` | `True` | Include data sample in report |

---

## Security & Correctness

- **HTML escaping** — column names and labels are escaped before rendering
- **Missing/Inf handling** — NaN and ±Inf excluded from moments, reported separately
- **Approximation badges** — estimates marked with `(≈)` or `approx` badge
- **Reproducibility** — set `random_seed` for deterministic results

## Extending

- **Backends** — polars/Arrow/DuckDB can be connected via the chunk iterator interface
- **Quantile sketches** — t-digest or KLL can replace the default reservoir
- **New sections** — drift comparisons, JSON export, CLI wrapper
