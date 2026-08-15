# Reading Arrow, Parquet and DuckDB

!!! info "Examples on this page assume a frame `df` and a Parquet file written from it"

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"amount": rng.standard_normal(2_000)})
    df.to_parquet("events.parquet")
    ```

`profile()` and `summarize()` take a pandas or polars frame, an iterable of
chunks, a file path — and, without materialising them first, Arrow tables and
readers, Parquet files, and DuckDB relations.

```python
from pysuricata import profile, summarize

df.to_parquet("events.parquet")
profile("events.parquet")          # read a batch at a time
```

```python
import duckdb

from pysuricata import summarize

df.to_parquet("events.parquet")
con = duckdb.connect()
relation = con.sql("SELECT * FROM 'events.parquet' WHERE amount > 0")
summarize(relation)                # the result set is never landed in memory
```

```python
import pyarrow as pa

from pysuricata import summarize

summarize(pa.Table.from_pandas(df))
```

DuckDB and pyarrow are **not dependencies**. The DuckDB path is duck-typed on
`fetch_record_batch`, so nothing imports it; Parquet and Arrow reading needs
`pyarrow`, and says so if it is missing.

## Streaming a query, not a table

The DuckDB path is the one worth knowing about. A relation is a query that has
not run yet, so a join across several Parquet files, filtered, can be profiled
without any of it existing as a frame:

```sql
SELECT o.*, c.segment
FROM 'orders/*.parquet' o
JOIN 'customers.parquet' c USING (customer_id)
WHERE o.created_at > '2026-01-01'
```

Hand that to `con.sql(...)` and pass the relation straight to `summarize()`.

## What it costs

Peak RSS on a 4,000,000 × 6 frame written as a 180 MB Parquet file, measured
with `getrusage` in fresh subprocesses, above a 118 MB bare-import floor:

| | above floor |
|---|---:|
| `profile(pd.read_parquet(path))` | 581 MB |
| `profile(path)` | 307 MB |

Lower, and it does not rise with the size of the file the way loading does.

Two honest qualifications:

- **This is not a zero-copy Arrow path.** The accumulators take numpy arrays,
  so each batch is converted on its way through. What changes is that one batch
  exists at a time rather than the whole file.
- **Memory is flat in rows for numeric columns and not yet for text.** Four
  float64 columns cost the same 19 MB above floor at 500,000 rows and at
  8,400,000. A string column does not — that is measured, filed, and open
  ([#95](https://github.com/alvarodiez20/pysuricata/issues/95)).

## Type inference on a stream

One behaviour follows from streaming, and it is worth knowing before it
surprises you.

A numeric column holding few enough distinct whole numbers is reclassified as
categorical — `grade` holding 0–11 is labels, not measurements. That decision
reads distinct values, which is sound evidence only when the whole column is in
hand. A stream cannot offer it: a leading run of one value looks
low-cardinality while the column is not, and the decision is never revisited.

So a **Parquet file that arrives in a single batch is handed over as a frame**,
and classifies exactly as `pd.read_parquet` would. Larger files are treated as
what they are, and their numeric columns stay numeric. The line is
`pysuricata.sources.DEFAULT_BATCH_ROWS`, 65,536 rows.

If you want whole-frame inference on a file that fits in memory, ask for it:

```python
profile(pd.read_parquet("events.parquet"))
```

## The readers on their own

`pysuricata.sources` exposes them, for when you want the batches rather than a
profile:

```python
from pysuricata.sources import stream_arrow, stream_duckdb, stream_parquet

df.to_parquet("events.parquet")
for batch in stream_parquet("events.parquet", batch_size=50_000, columns=["amount"]):
    print(len(batch))
```

`columns=` is worth using on a wide file: columns you do not read are never
decoded, which is where most of the saving is.
