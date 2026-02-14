---
title: Architecture Diagrams
description: Workflow diagrams with time and space complexity annotations for PySuricata's streaming EDA pipeline.
---

# Architecture Diagrams

Visual reference for PySuricata's processing pipeline, accumulator internals, chunk processing, and rendering — annotated with algorithmic complexity.

---

## 1. Main Processing Pipeline

End-to-end data flow from user input to final `Report` object.

```mermaid
flowchart TD
    A["User Input\npd.DataFrame | pl.DataFrame\npl.LazyFrame | Iterable"] -->|"O(1)"| B["api.py — profile / summarize\n_coerce_input · _to_engine_config"]
    B -->|"O(1)"| C["report.py — ReportOrchestrator.build_report"]
    C -->|"O(1)"| D["engine.py — StreamingEngine.process_stream"]
    D -->|"O(1)"| E["EngineManager.select_adapter\nPandasAdapter | PolarsAdapter"]
    E -->|"O(1)"| F["AdaptiveChunker.chunks_from_source\nStrategy: ADAPTIVE | FIXED | MEMORY_AWARE"]
    F -->|"O(n/c) chunks"| G["First Chunk — infer_and_build\nO(cols): type inference + accumulator creation"]
    G --> K{"Stream Loop\nO(n/c) iterations"}
    K -->|"O(cols × chunk)"| L["consume_chunk\nPer-column accumulator updates"]
    K -->|"O(m² × chunk)"| M["update_corr\nPairwise sums for Pearson r\nm = numeric cols"]
    K -->|"O(cols × chunk)"| N["update_row_kmv\nRow hash for duplicate detection"]
    L --> K
    M --> K
    N --> K
    K -->|"Done"| O["_build_manifest_inputs\nO(cols): kinds_map · col_order · miss_list"]
    O --> P["_apply_correlation_chips\nO(cols): attach top correlations"]
    P --> Q["render_html_snapshot\nO(cols × card_render)"]
    Q --> R["_build_summary\nO(cols): acc.finalize per column"]
    R --> S["Report — html + stats"]
```

### Complexity Summary

| Metric | Value |
|--------|-------|
| **Total Time** | O(n_rows × n_cols) + O(m² × n_rows) for correlations |
| **Total Space** | O(cols × (s + k + b)) |
| **Peak Memory** | ~50 MB bounded by streaming architecture |

Where: `n` = total rows, `c` = chunk size (200k default), `m` = numeric columns, `s` = reservoir size (20k), `k` = KMV sketch size (2048), `b` = histogram bins (25).

---

## 2. Accumulator Architecture

Each column type has a specialized accumulator with small, bounded state.

```mermaid
flowchart TD
    subgraph Numeric["NumericAccumulator — Space: O(s + k + b) per column"]
        SM["StreamingMoments — Welford/Chan\nTime: O(n) per chunk | Space: O(1)\nmean · variance · skewness · kurtosis"]
        RS["ReservoirSampler — k = 20 000\nTime: O(1) add | Space: O(s)\nquantiles · histogram source"]
        KMV_N["KMV Sketch — k = 2 048\nTime: O(log k) add | Space: O(k)\napprox distinct count"]
        ET["ExtremeTracker — k = 5 bounded heaps\nTime: O(n log k) update | Space: O(k)\nmin/max with row indices"]
        MG_N["MisraGries — k = 50\nTime: O(1) add | Space: O(k)\ntop-k values (integer-like cols)"]
        SH["StreamingHistogram — bins = 25\nTime: O(1) add | Space: O(b)\ntrue distribution approximation"]
        OD["OutlierDetector\nTime: O(s) finalize | Space: O(1)\nIQR + MAD methods"]
    end

    subgraph Cat["CategoricalAccumulator — Space: O(k) per column"]
        KMV_C["KMV Sketch\nTime: O(log k) add | Space: O(k)\napprox distinct count"]
        MG_C["MisraGries\nTime: O(1) add | Space: O(k)\ntop-k categories"]
        SL["String Length Tracker\nTime: O(1) per value | Space: O(1)\navg_len · p90 · empty count"]
    end

    subgraph DT["DatetimeAccumulator — Space: O(1) per column"]
        DMM["Min/Max — nanosecond timestamps\nTime: O(1) | Space: O(1)"]
        DFC["Frequency Counters\nday-of-week · hour · month\nTime: O(1) | Space: O(1)"]
    end

    subgraph Bool["BooleanAccumulator — Space: O(1) per column"]
        BCT["True / False / Missing counters\nTime: O(1) per value | Space: O(1)"]
    end
```

### Per-Column Memory Budget

| Accumulator | Default Config | Approx Memory |
|-------------|---------------|---------------|
| NumericAccumulator | s=20k, k=2048, b=25 | ~170 KB |
| CategoricalAccumulator | k=2048+50 | ~20 KB |
| DatetimeAccumulator | — | < 1 KB |
| BooleanAccumulator | — | < 1 KB |

---

## 3. Chunk Processing Detail

What happens inside `consume_chunk()` for each column in a chunk.

```mermaid
flowchart TD
    A["Incoming Chunk\npd.DataFrame or pl.DataFrame"] --> B{"New columns\nin this chunk?"}
    B -->|"Yes"| C["UnifiedTypeInferrer.infer_series_type\n+ create accumulator via factory\nO(sample_size) per new column"]
    B -->|"No"| D["Iterate accs.items — O(cols)"]
    C --> D

    D --> E{"Column type?"}

    E -->|"Numeric"| F["to_numpy(float64)\nFast path: direct cast\nSlow path: pd.to_numeric(errors=coerce)\nO(chunk_size)"]
    F --> F1["acc.update(arr)\nmoments + reservoir + histogram\nO(chunk_size)"]
    F1 --> F2["KMV.add per value\nO(chunk_size × log k)"]
    F2 --> F3["ExtremeTracker.update\nO(chunk_size × log k)\nEvery 5th chunk only (pandas)"]
    F3 --> G["Next column"]

    E -->|"Categorical"| H["s.tolist → Python list\nO(chunk_size)"]
    H --> H1["KMV + MisraGries + string stats\nO(chunk_size)"]
    H1 --> G

    E -->|"Boolean"| I["Per-value str coercion\nstr(v).strip().lower()\nO(chunk_size) — Python loop"]
    I --> I1["True/False/Missing counting\nO(chunk_size)"]
    I1 --> G

    E -->|"Datetime"| J["pd.to_datetime(errors=coerce, utc=True)\nO(chunk_size)"]
    J --> J1["Min/max + frequency counters\nO(chunk_size)"]
    J1 --> G

    G --> K{"More columns?"}
    K -->|"Yes"| D
    K -->|"No"| L["Return to engine loop\n+ update memory estimate"]
```

### Per-Chunk Bottleneck Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Numeric KMV loop | O(chunk × log k) | Per-value Python → batch vectorizable |
| Boolean coercion | O(chunk) Python | str conversion per value → vectorizable |
| Correlation update | O(m² × chunk) | Pairwise; gated by `corr_max_cols=50` |
| Row KMV hashing | O(chunk × cols) | Per-row tuple hashing → vectorizable |
| Type conversion | O(chunk) | pd.to_numeric / pd.to_datetime |

---

## 4. Rendering Pipeline

How the final HTML report is assembled from accumulated statistics.

```mermaid
flowchart TD
    A["render_html_snapshot\nrender/html.py"] --> B["Build kinds_map\nO(cols): name → (kind, accumulator)"]
    B --> C["Compute miss_list\nO(cols): per-column missing percentage"]
    C --> D["Dataset metrics\nrow_kmv.approx_duplicates\nconstant / high-cardinality detection"]
    D --> E["Card Loop — O(cols)"]
    E --> F["acc.finalize(chunk_metadata)\nquantiles O(s log s)\nextremes O(k log k)\noutlier stats O(s)"]
    F --> G{"Card type?"}

    G -->|"Numeric"| G1["NumericCardRenderer\nSVG histogram + stats tables\n~200 lines HTML per card"]
    G -->|"Categorical"| G2["CategoricalCardRenderer\nDonut chart SVG + top-k table"]
    G -->|"Datetime"| G3["DatetimeCardRenderer\nTemporal bar charts + freq tables"]
    G -->|"Boolean"| G4["BooleanCardRenderer\nTrue/False bar + percentage"]

    G1 --> H["Collect card HTML strings"]
    G2 --> H
    G3 --> H
    G4 --> H

    H --> I["Load + inline static assets\nstyle.css · functionality.js\nchart.min.js (inlined)"]
    I --> J["CorrelationsSectionRenderer\nrender_section — O(m²)"]
    J --> K_node["MissingValuesSectionRenderer\nrender_section — O(cols)"]
    K_node --> L["DonutChartRenderer\nrender_dtype_donut — SVG"]
    L --> M["Template assembly\nreport_template.html.format\n~40 template variables"]
    M --> N["Self-contained HTML\n~1.2–1.6 MB typical"]
```

### Rendering Cost Breakdown

| Phase | Complexity | Output Size |
|-------|-----------|-------------|
| Finalization | O(cols × s log s) | — |
| Card rendering | O(cols) | ~5–20 KB per card |
| Asset inlining | O(1) | ~200 KB (CSS+JS) |
| Correlation section | O(m²) | Variable |
| Template assembly | O(1) | ~1.2–1.6 MB total |

---

## 5. End-to-End Complexity Table

| Stage | Time | Space | Key Parameter |
|-------|------|-------|---------------|
| Input coercion | O(1) | O(1) | — |
| Chunking | O(n/c) | O(c × cols) | `chunk_size` (200k) |
| Type inference | O(sample) | O(1) | first chunk only |
| Accumulator updates | O(n × cols) | O(cols × s) | `numeric_sample_k` (20k) |
| KMV sketching | O(n × log k) | O(cols × k) | `uniques_k` (2048) |
| Correlation | O(n × m²) | O(m²) | `corr_max_cols` (50) |
| Row deduplication | O(n × cols) | O(k) | KMV sketch |
| Finalization | O(cols × s log s) | O(cols) | — |
| HTML rendering | O(cols) | O(report_size) | ~1.5 MB |
| **Total** | **O(n × cols + n × m²)** | **O(cols × s)** | — |
