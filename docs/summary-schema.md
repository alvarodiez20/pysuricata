# The `summarize()` schema

`summarize()` returns a plain dictionary, and `Report.stats` carries the same
one. This page is the contract: what the keys are, which of them are estimates,
and what a version bump means.

!!! info "Examples on this page assume a frame `df`"

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"amount": rng.lognormal(3, 1.2, 2_000)})
    ```

```python
from pysuricata import summarize

stats = summarize(df)
stats["schema_version"]                     # 2
stats["dataset"]["rows_est"]
stats["columns"]["amount"]["median"]
```

## Stability policy

`schema_version` is an integer at the top level.

- **Adding a key does not change it.** A consumer reading keys it knows is
  unaffected by new ones, so new statistics arrive without warning and without
  ceremony.
- **Renaming or removing a key bumps it.** So does changing what an existing key
  means, or its units.

This is not decoration. The payload drifted once before it was versioned:
`dataset["rows"]` became `dataset["rows_est"]`, which silently broke every
documented example and would have broken every downstream consumer.

`pysuricata check` reads `schema_version` off a stored baseline and **refuses**
to compare against one written by a different version, rather than quietly
matching whatever keys still line up.

### What changed in 2

Both outlier counts changed value: they were the number of crossings found
*inside* the 20,000-value reservoir and are now estimates for the column, so
above `numeric_sample_size` rows they are roughly `n / 20,000` times larger than
before (#327). By the rule above that is a **correction, not a break**, and it
would not have bumped the version on its own. The version moved anyway, because
a stored baseline written before the fix holds counts on the old scale and
comparing it against the new ones would silently mis-report drift: `check`
refuses across versions, which is exactly the signal wanted here.

`outliers_mod_zscore_est` is a new name for `outliers_mod_zscore`, which is
**deprecated and still published**. Renaming a key outright is a break, and a
break costs a major bump under [Versioning](versioning.md) -- so the old name
stays until 1.0.0 rather than the rename shipping in a minor release. New code
should read `_est`; nothing is required to change today.

Every column carries `type`, `count`, `missing` and `mem_bytes`, whatever its
kind. Those four are safe to read without checking `type` first.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | The version of this schema |
| `dataset` | dict | Whole-frame statistics |
| `columns` | dict | Column name → statistics |

## `dataset`

| Key | Type | Meaning |
|---|---|---|
| `rows_est` | int | Rows seen. Exact for a frame; a count of what was streamed otherwise |
| `cols` | int | Columns profiled |
| `missing_cells` | int | Missing cells across profiled columns |
| `missing_cells_pct` | float | As a percentage of `rows_est × cols` |
| `duplicate_rows_est` | int | **Approximate.** From a row-level KMV sketch. `0` when the estimate is below the sketch's own resolution — see below |
| `duplicate_rows_pct_est` | float | **Approximate.** As above |
| `duplicate_rows_uncertainty` | int | One standard deviation on the count, in rows. `0` when the count is exact |
| `memory_bytes` | int | Approximate in-memory size of the source |
| `top_missing` | list | Up to five `{column, pct, count}`, worst first |

### Reading the duplicate count

The duplicate count is `rows − distinct`. `rows` is exact and `distinct` is a
sketch estimate, so **the whole absolute error of the distinct estimate lands on
the duplicate count** — a quantity that is usually far smaller. On 200,000 rows
containing exactly 2,000 duplicates the distinct estimate was 0.48% off, well
inside spec, and the duplicate count came back 47% high.

So `duplicate_rows_est` is suppressed to `0` when it does not clear **3**
standard deviations of uncertainty, not 1 — a 1-sigma gate published a
duplicate count on a clean, duplicate-free frame about one run in ten (#248).
`duplicate_rows_uncertainty` is exported as one sigma either way; the ceiling
below which nothing is resolvable is `math.ceil(3 * duplicate_rows_uncertainty)`,
not `duplicate_rows_uncertainty` itself. Read the two together:

| `est` | `uncertainty` | Ceiling (`3 × uncertainty`, rounded up) | Means |
|---|---|---|---|
| `0` | `0` | — | Exactly none. The distinct count was exact, not estimated |
| `0` | `2201` | `6603` | Nothing resolvable. The true count is somewhere below roughly 6,603 |
| `50602` | `1651` | — | About 50,602, give or take 1,651 — cleared the ceiling, so no suppression |

Gating on `duplicate_rows_est > 0` is therefore safe against false positives and
will not fire on a count the sketch cannot support. If you need to fail on the
*possibility* of duplicates, gate on `duplicate_rows_uncertainty` instead — it
is nonzero exactly when `duplicate_rows_est` could be suppressing a real count.

## `columns[name]["type"]`

One of `numeric`, `categorical`, `datetime`, `boolean`, `identifier`.

`identifier` is a numeric column that is monotonic, integer-like and has a
distinct count equal to the row count — a key, not a measurement. It carries the
numeric keys, since it is a numeric column; the type is a statement about what
the numbers mean.

## Numeric columns

Shape and position:

| Key | Type | Notes |
|---|---|---|
| `count`, `missing`, `zeros`, `negatives`, `inf` | int | Exact |
| `mean`, `std`, `variance`, `se` | float | Exact (Welford/Pébay) |
| `skew`, `kurtosis`, `cv`, `gmean` | float | Exact |
| `min`, `max` | float | Exact — from the extreme tracker, which sees every value |
| `min_positive` | float \| null | Exact — the smallest strictly positive value, or `null` when the column has none. Where a log axis begins |
| `q1`, `median`, `q3`, `iqr`, `mad` | float | **From the reservoir sample** |
| `ci_lo`, `ci_hi` | float | 95% confidence interval for the mean |
| `jb_chi2` | float | Jarque–Bera statistic |
| `min_items`, `max_items` | list | `(row_index, value)`, most extreme first |
| `true_histogram_edges`, `true_histogram_counts` | list | The distribution, exact counts |

Quality and structure:

| Key | Type | Notes |
|---|---|---|
| `unique_est` | int | **Approximate.** KMV sketch |
| `unique_ratio_approx` | float | **Approximate.** `unique_est / count` |
| `top_values` | list \| None | `(value, count)`. **`None` means not tracked** |
| `top_values_uncertainty` | int | **Approximate.** How far below the truth each count above may sit. `0` means exact |
| `outliers_iqr_est`, `outliers_mod_zscore_est` | int | **Approximate.** Counted in the sample, scaled to the column |
| `mono_inc`, `mono_dec` | bool | Monotonic over the stream as it arrived |
| `int_like` | bool | Every value is a whole number |
| `heap_pct` | float | Share sitting on round numbers — heaping |
| `bimodal` | bool | Two modes detected in the histogram |
| `gran_decimals`, `gran_step` | int, float | Inferred granularity |
| `corr_top` | list | `(other_column, r)` above the threshold |
| `approx` | bool | Whether sampling was involved at all |
| `dtype` | str | The source dtype |

`top_values` distinguishes two situations that a plain empty list would not:
`None` means the top-k sketch was **switched off** for this column because its
cardinality made a "common values" table meaningless, while `[]` means it was
tracked and nothing was frequent enough.

## Categorical columns

| Key | Type | Notes |
|---|---|---|
| `count`, `missing`, `empty_zero` | int | Exact |
| `unique_est` | int | **Approximate.** KMV sketch |
| `top_items` | list | `(value, count)`. **Approximate** — Misra-Gries counts are lower bounds |
| `top_items_uncertainty` | int | **The bound on those counts.** `0` means the counters never evicted, so they are exact |
| `entropy`, `gini_impurity`, `diversity_ratio`, `most_common_ratio` | float | Derived |
| `avg_len`, `len_p90` | float, int | Value length in characters |
| `case_variants_est`, `trim_variants_est` | int | **Approximate.** Distinct counts after folding case / trimming whitespace |
| `approx`, `dtype` | bool, str | |

The two variant estimates are what the report's "looks like a case variant"
flags are drawn from: compare them against `unique_est` — a lower folded count
means some values differ only by case or by surrounding whitespace.

## Datetime columns

| Key | Type | Notes |
|---|---|---|
| `count`, `missing` | int | Exact |
| `min_ts`, `max_ts` | float | **Epoch nanoseconds**, UTC |
| `unique_est` | int | **Approximate** |
| `time_span_days` | float | `max_ts − min_ts`, in days |
| `avg_interval_seconds`, `interval_std_seconds` | float | Between consecutive values |
| `weekend_ratio`, `business_hours_ratio` | float | Share of values in each |
| `mono_inc`, `mono_dec` | bool | Monotonic over the stream as it arrived |
| `seasonal_pattern` | str \| None | A description, when one is detected |
| `source_timezone` | str \| None | The dtype's timezone, when it carries one |
| `by_hour`, `by_dow`, `by_month` | list | Tallies of length 24, 7 and 12 |
| `by_year` | dict | Year → count |
| `dtype` | str | |

**`min_ts` and `max_ts` are nanoseconds**, not seconds — that is the units
question most likely to be guessed wrong. Divide by `1_000_000_000` before
handing them to `datetime.fromtimestamp`.

## Boolean columns

| Key | Type | Notes |
|---|---|---|
| `count`, `missing`, `true`, `false` | int | Exact |
| `true_ratio`, `false_ratio` | float | Of the non-missing values |
| `entropy` | float | |
| `dtype` | str | |

`count` counts non-missing values, so the number of rows the column covers is
`true + false + missing`.

## What is estimated, and by how much

Anything marked **approximate** above rests on a bounded-memory sketch. The two
that matter:

- **`unique_est` and the variant counts** come from a KMV sketch with relative
  error near `1/√k`, about **2.2%** at the default `uniques_k=2048`. Raise
  `uniques_k` to tighten it.
- **Quantiles** (`q1`, `median`, `q3`, and the IQR and MAD derived from them)
  come from a reservoir sample of `numeric_sample_size` values, default 20,000,
  with error near `1/√k` — about **0.7%**, and **3.2%** if you drop the sample
  to 1,000.

`approx` on a numeric column tells you whether sampling was involved for that
column at all: below the sample size, the reservoir holds every value and the
quantiles are exact.

`top_items` and `top_values` come from Misra-Gries counters. Their counts are
**lower bounds** — a reported count never overstates, and the counters neither
partition the column nor sum to the row count.

How far below is published rather than left to the reader.
`top_items_uncertainty` (and `top_values_uncertainty` on numeric columns) is the
total weight the sketch decremented, and Misra-Gries guarantees

```text
true_count(x) ∈ [reported(x), reported(x) + uncertainty]
```

so `sku-0753: 37` with an uncertainty of 1,112 is honestly rendered as
`37 – 1,149`. Zero means no eviction ever happened and the counts are exact.

`approx` follows from the same number. It used to be `len(top_items) >=
top_k`, which reads the dangerous case backwards: eviction *deletes* counters,
so the list shrinks below the budget exactly when the sketch is under most
pressure, and a 1M-row column over 1,000 categories reported nine items,
`approx=False`, and a top count 30x low (#328).

## What is not in the payload

Deliberately withheld, and listed in `pysuricata.report.SUMMARY_FIELDS_WITHHELD`
with the reason for each: the reservoir sample itself (up to 20,000 floats per
column), the per-chunk bookkeeping the report's chunk strip is drawn from, and
the scale factor the renderer applies to sampled counts. None of these is a
statistic.

A test walks that list against the accumulators' own summary dataclasses and
fails if a computed statistic is neither published nor listed. That is what
stops the JSON drifting behind the HTML again — it has happened twice, with
correlations and with numeric top values, and both times it was only findable by
reading the renderer.
