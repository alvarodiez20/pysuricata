# Comparing two datasets

!!! info "Examples on this page assume two frames: `df`, last month's data, and `new_df`, this month's"

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame({
        "amount": rng.lognormal(3, 1.0, n),
        "region": rng.choice(["north", "south", "east"], n).astype(object),
    })
    new_df = pd.DataFrame({
        "amount": rng.lognormal(3.4, 1.0, n),
        "region": rng.choice(["north", "south", "west"], n).astype(object),
    })
    ```

`compare(a, b)` reports what moved between two datasets. Every delta, whether or
not it crosses any threshold — it is a description, not a verdict.

```python
from pysuricata import compare

diff = compare(df, new_df)
diff.schema.unchanged                      # ("amount", "region")
diff.columns["amount"].median_shift_sigma  # 0.24
diff.columns["region"].categories_added    # ("west",)
diff.columns["region"].categories_removed  # ("east",)
```

`pysuricata.comparison.render(diff)` turns that into text. Over the two frames
above, verbatim:

```text
rows: 20,000 → 20,000 (+0.0%)
  amount: median +0.24σ
  amount: spread ×1.70
  region: new categories: west (approx)
  region: gone: east (approx)
```

A frame that gained a column would show it under `diff.schema.added`, and a
datetime column would add a line like `seen_at: newest value moved +31.0 days`.

Both sides accept anything `profile()` does — frames, paths, Arrow tables,
DuckDB relations — or a `summarize()` payload you already have, which is not
re-profiled.

## Against `pysuricata check`

They are the same arithmetic with different jobs.

| | `check` | `compare` |
|---|---|---|
| Answers | should this build fail? | what moved? |
| Output | findings above a threshold | every delta |
| Verdict | `passed`, and an exit code | none |
| Category churn | no | yes |
| Quantiles | median | every quartile |

`check` is thresholds applied to the deltas `compare` computes, and they read
the payload through the same functions. That is deliberate: a gate and a diff
that disagreed about what counts as a change would be worse than either alone,
and there is a test asserting the two report the same numbers.

**Category churn is in the diff and not in the gate.** Which values fall in and
out of a top-k table moves for reasons that are not drift — Misra-Gries keeps
bounded counters, and the tail of a long list reshuffles on counting noise. That
makes it a poor thing to fail a build on, and the most legible thing you can
show a reader about a categorical column that changed.

## What it reports

**Schema** — `added`, `removed`, `retyped`, `unchanged`. A retyped column gets
no statistical delta: comparing a mean against a category count is noise on top
of a fact the reader already has.

**Dataset** — rows before and after with the relative change, missing-cell
percentage, approximate memory.

**Per column**, for columns present in both with the same type:

| | |
|---|---|
| `missing_pct_change` | in **percentage points** |
| `unique_change_pct` | **approximate**: relative change in the distinct estimate |
| `distinct_rate_change_pct` | the same, per row |
| `mean_shift_sigma`, `median_shift_sigma`, `q1_shift_sigma`, `q3_shift_sigma` | in baseline standard deviations, **signed** |
| `std_ratio` | spread after over spread before |
| `range_before`, `range_after` | `(min, max)` |
| `true_rate_change_pp` | boolean columns, in percentage points |
| `categories_added`, `categories_removed`, `top_category_*` | **approximate** |
| `span_days_*`, `newest_*` | datetime columns; timestamps are epoch nanoseconds |

Two numbers rather than one for cardinality, because neither answers the
question alone: doubling the rows doubles a continuous column's distinct
**count** while leaving its **rate** alone, and does the opposite to a
three-level enum. Reported separately, a reader can tell growth from a change in
shape — which is the same problem `check` solves with a rule, stated here as
data.

## Honesty about the estimates

`unique_change_pct` and the category lists rest on sketches. `ColumnDelta.approximate`
says so per column, and the text rendering marks those lines `(approx)`.

The rendering also **suppresses a distinct-count change smaller than the sketch's
own error** — about 2.2% at the default `uniques_k` — because printing a 1% move
as a finding is the same mistake as printing an estimate as an exact integer.
The structured delta still carries the raw number; only the prose is quiet.

## Reproducibility

`compare` profiles both sides with `seed=0` by default rather than leaving it to
chance, so comparing a dataset against itself is a no-op rather than a set of
sampling wobbles. Any other keyword is passed to both sides, since comparing two
profiles taken with different settings would report the settings as drift.

```python
from pysuricata import compare

compare(df, new_df, sample=5_000, columns=["amount"])
```

## As JSON

```python
import json

from pysuricata import compare

print(json.dumps(compare(df, new_df).to_dict())[:60])
```

Three sections — `dataset`, `schema`, `columns` — mirroring the object.

## What is not here

There is no HTML view yet. The JSON contract and the text rendering are what
this ships with; the report side of it belongs with the wider work on the
report's presentation rather than bolted on beside it. Tracked as
[#121](https://github.com/alvarodiez20/pysuricata/issues/121).
