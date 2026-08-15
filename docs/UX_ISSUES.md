# UX review — issue backlog

12 issues, grouped by the user each one unblocks.

---

## 1. Numeric columns with few distinct values are classified categorical

`correctness` `ux` `user:evaluator`

## What happens

Profiling a normal customer table with defaults, `age` comes back **categorical**:

```python
g = np.random.default_rng(0)
df = pd.DataFrame({"age": g.integers(18, 85, 20_000)})
summarize(df)["columns"]["age"]["type"]   # 'categorical'
```

The card then shows entropy and rare levels instead of median, quartiles and a
histogram.

## Why

`should_reclassify_numeric_as_categorical` fires on
`unique_count < 10 or unique_ratio < 0.05`. 67 distinct values in 20,000 rows is a
0.34% ratio, so the ratio arm triggers.

## Why this is the wrong shape of rule

**It gets more wrong as the table gets bigger.** Any bounded integer — age, year,
rating, day-of-month, HTTP status, US state code — crosses into "categorical"
purely by adding rows, with no change to the data's nature. For a library whose
pitch is bounded-memory profiling of large data, a heuristic that degrades with
scale is backwards.

The 0.0.26 fix (#50) guards the *streamed* path via `first_chunk_is_whole`.
In-memory frames — the common case — still hit this.

## Proposed fix

Replace the ratio arm with a cardinality **ceiling**, and require integer-like
values:

```python
return int_like and unique_est <= 50
```

A ceiling is stable under row count, which is the property the ratio lacks.

## Acceptance

- [ ] `age` (67 distinct, 20k rows) profiles as numeric
- [ ] A genuine 8-level category still profiles as categorical
- [ ] A test that profiles the same column at 1k / 100k / 10M rows and asserts the
      classification does not change

---

## 2. Identifier columns get meaningless statistics

`ux` `report` `user:evaluator`

## What happens

A monotonic, fully unique integer key is profiled as an ordinary numeric column:

- mean `1e+04`, median `1e+04`, a flat uniform histogram
- skewness and standard deviation, neither of which means anything
- `Zeros: 1 (0.0%)`, which is actively misleading

## Why it is worth fixing

Every competing profiler computes the mean of an ID column. Being the one that
says *"this is a key"* instead is a small, memorable difference — and it costs
less than the card it replaces.

## The signals already exist

The numeric accumulator already tracks everything needed:

- `mono_inc` is true
- the KMV estimate equals the row count
- `int_like` is true

## Proposed fix

When those hold, emit an **Identifier** badge and a card with the questions a key
actually raises: distinct count, gaps in the sequence, duplicates, nulls, min and
max. Skip moments, histogram and outliers entirely.

## Acceptance

- [ ] `pd.DataFrame({"id": np.arange(n)})` renders an Identifier card
- [ ] A numeric column that happens to be sorted but has duplicates does **not**
- [ ] The badge appears in `summarize()` output too, not only the HTML

---

## 3. Report has no triage: 60 columns render at equal weight in source order

`ux` `report` `user:analyst`

## The problem

Every column gets an identically sized card, in source order. A 60-column frame is
60 screens with no entry point. The question a reader arrives with — *which columns
are broken?* — is never answered.

## The signal already exists

The quality chips (*Missing · Skewed Right · Heavy-tailed · Many outliers ·
Positive-only*) are already computed per column. They are the best thing in the
report and they are currently decoration rather than navigation.

## Proposed fix

1. A **"needs attention"** block under the summary: *"3 of 60 columns have
   issues"*, listing them with their chips, each linking to its card.
2. Optional severity ordering, behind a toggle so source order stays available.
3. A chip filter in the Variables section: click *Missing* to show only those.

No new statistics required — only a use for the ones already there.

## Acceptance

- [ ] Summary names the columns with issues and links to them
- [ ] Clicking a chip filters the variable list
- [ ] Source order remains reachable in one click

---

## 4. Histogram ignores the log-scale flag the card itself computed

`ux` `report` `good first issue` `user:analyst`

## What happens

A lognormal `revenue` column is correctly labelled *Positive-only · Skewed Right ·
Heavy-tailed · **Log-scale?*** — and then the histogram is drawn on a linear axis,
where the whole distribution renders as one bar at the left edge.

The Log toggle is directly beneath the chart, defaulted to Linear.

## Why it matters

The report computed the right answer and then displayed the wrong picture. That is
worse than not detecting it, because it teaches the reader that the chips are
cosmetic.

## Proposed fix

When the log-scale heuristic fires, initialise the scale toggle to **Log**. Keep
the toggle so the reader can override, and consider marking the chip as active
rather than interrogative once it has driven the default.

## Acceptance

- [ ] A lognormal column opens on a log axis
- [ ] A normal column still opens on linear
- [ ] The toggle still switches both ways

---

## 5. Add `pysuricata check` — a CI gate with an exit code

`api` `dx` `user:engineer`

## Why

Bounded-memory profiling that fits in CI is the project's positioning, and the CLI
has no verb for it. `profile` and `summarize` both exit 0 regardless of what they
found.

Every existing gate tool (Great Expectations, Soda, pointblank) requires you to
author expectations first. A profiler can gate on *shape drift* with no
configuration at all, which is a genuinely unoccupied position.

## Proposed shape

```bash
pysuricata check data.parquet --against baseline.json --exit-code
pysuricata check data.parquet --baseline baseline.json --max-missing-pct 5
```

- writes a baseline from `summarize()` output
- compares a new run against it: row count, column set, dtypes, null rates,
  cardinality, and distribution drift on numeric columns
- exits non-zero when a threshold is crossed, printing what moved
- a `--thresholds thresholds.toml` file for anything beyond the defaults

## Why it is cheap

`summarize()` already produces both sides of the comparison. The work is the
comparison, the thresholds file and the CLI surface — not new statistics.

## Acceptance

- [ ] `check` exits 0 on an unchanged dataset, non-zero on a changed one
- [ ] Output names what changed and by how much
- [ ] A GitHub Actions snippet in the docs

---

## 6. Version the `summarize()` payload and reconcile it with the HTML

`api` `user:integrator`

## Why now

`summarize()` is the strongest quiet-adoption play available — the thing other
tools build on. It currently has no `schema_version`, and it has **already drifted
once**: `dataset.rows` became `dataset.rows_est`, which silently broke every doc
example and would break every downstream consumer.

Version it before anyone depends on it, not after.

## Two parts

**1. Add a version field**

```python
{"schema_version": 1, "dataset": {...}, "columns": {...}}
```

Document the compatibility promise: additive changes bump nothing, renames or
removals bump the major.

**2. Reconcile with the HTML**

`summarize()` returns `top_values: None` for numeric columns while the HTML
renders a populated "Common values" table from the same accumulator. The two
surfaces disagree about what a column has. Pick one answer.

## Acceptance

- [ ] `schema_version` present and asserted in a test
- [ ] Every statistic rendered in the HTML is reachable from `summarize()`
- [ ] A documented page listing the payload's keys and their meaning

---

## 7. No progress feedback on long runs

`ux` `dx` `user:bigdata`

## What happens

A 6-column, 300,000-row profile (1.8M cells) produced **46 bytes** of output, all
of it an unrelated notice. There is no chunk counter, no percentage, no ETA and no
completion line.

For the use case the library is positioned on — profiling data that does not fit
in memory, which takes minutes — a user cannot tell a working process from a hung
one.

## What exists

`log_every_n_chunks` routes to a logger that is off by default, so it is invisible
unless the caller configures logging first.

## Proposed fix

```python
profile(source, progress=True)     # or progress="auto"
```

- writes to **stderr**, never stdout, so piped output stays clean
- `"auto"` enables it only when stderr is a TTY
- shows chunks done, rows done, elapsed, and a rate; ETA only when the total is
  knowable
- a final one-line summary regardless: *"profiled 1.8M cells in 2.1 s"*
- accepts a callback for programmatic consumers

## Acceptance

- [ ] Nothing on stdout in any mode
- [ ] `progress="auto"` silent when piped, visible on a terminal
- [ ] Works for generator sources where the total is unknown

---

## 8. Let users specify a memory budget

`api` `ux` `user:bigdata`

## Proposal

```python
profile(source, memory_budget="512MB")
ProfileConfig.for_memory("512MB", n_columns=40)   # inspectable
```

A **planner that derives settings**, not a cap that enforces them. Full reasoning
in `docs/adr/memory-budget.md`.

## Why it works

The memory model was fitted from measurements at 0.0.27 and is clean:

```
peak_MB ≈ 75 + n_cols × (0.5 + k × 37B + chunk_size × 48B)
```

Every term is a knob the library owns, so the budget is invertible. Verified:

| budget | cols | chosen | actual peak |
|---:|---:|---|---:|
| 250 MB | 8 | chunk 200,000 · k 20,000 | 153 MB |
| 250 MB | 40 | chunk 48,437 · k 20,000 | 207 MB |
| 1000 MB | 100 | chunk 109,375 · k 20,000 | 560 MB |

Streaming memory is also genuinely flat in rows: 200k → 5M rows moved peak RSS
from 32 to 35 MB above the floor.

## Why it must not be a hard cap

- ~75 MB is gone before the first line runs (interpreter + numpy + pandas). A
  budget below ~100 MB is unsatisfiable and must error immediately.
- The library does not control pandas' allocations, fragmentation, or GC timing.
- For an **in-memory** DataFrame the user already paid: a 1M × 8 frame costs
  123 MB resident before `profile()` is called. The budget only bites on streaming
  sources.

A cap that raises `MemoryError` would reproduce the incumbent failure mode this
project is positioned against.

## Must report the trade

A tight budget shrinks `k`, and every quantile, the median, IQR, MAD and histogram
come from that sample. Error is `1/sqrt(k)`: ±0.7% at k=20,000, ±3.2% at k=1,000.
The chosen plan and its accuracy consequence must be logged, never silent.

## Before shipping

The model is fitted on **numeric columns only**. Categorical holds a Misra-Gries
table and string samples; datetime holds its own reservoir. Refit per column kind
or the budget will be optimistic on a text-heavy frame.

## Acceptance

- [ ] `memory_budget` accepts "512MB" / "2GB" / an int in bytes
- [ ] Errors below the floor with the real numbers in the message
- [ ] Logs the chosen settings and the resulting quantile error
- [ ] A test asserting measured peak RSS ≤ budget across a matrix of column counts
- [ ] Model refitted for categorical and datetime columns

---

## 9. Ship `py.typed` so the annotations are visible to type checkers

`api` `dx` `good first issue`

## What happens

Every public signature is fully annotated — and there is no `py.typed` marker, so
under [PEP 561](https://peps.python.org/pep-0561/) mypy and pyright treat the whole
package as untyped and infer `Any` for everything it returns.

All of the annotation work is invisible to exactly the audience the project
targets.

## Fix

1. Add an empty `pysuricata/py.typed`
2. Include it in `[tool.setuptools.package-data]`
3. Export `DataLike` at the top level so callers can annotate their own wrappers
4. Add a CI step that runs mypy against a small consumer snippet, so this cannot
   silently regress

## Acceptance

- [ ] `reveal_type(profile(df))` gives `Report`, not `Any`
- [ ] `py.typed` present in the built wheel

---

## 10. Tighten the public namespace: `__all__`, one-line repr, consistent errors

`api` `dx` `good first issue`

Four small things in the same area.

## 1. `__all__`

`pysuricata.<TAB>` offers 15 names, **8 of them internal modules**
(`accumulators`, `api`, `compute`, `config`, `logger`, `render`, `report`,
`utils`). One line makes the public contract explicit.

## 2. `Report.__repr__` dumps the whole document

```python
>>> rep
Report(html='<!DOCTYPE html>\n<html lang="en">...
```

1.1 MB into a REPL, a log line or a debugger. `_repr_html_` saves the notebook
case only. Proposed:

```python
<Report: 20,000 rows x 8 cols, 1.1 MB>
```

## 3. Inconsistent exceptions for the same mistake

| input | today |
|---|---|
| `profile({"a": [1,2]})` | `TypeError: Unsupported data type…` |
| `profile([1, 2, 3])` | `RuntimeError: Adapter selection failed: Unsupported input type: <class 'int'>` |
| `profile(df, config="oops")` | `AttributeError: 'str' object has no attribute 'compute'` |

Same class of user error, three exception types, one leaking internal vocabulary.
All should be `TypeError` with one message that names what is accepted.

## 4. `ReportConfig` is `ProfileConfig`

One class, two exported names, used interchangeably in the docs, so a reader
cannot tell whether there is a distinction they are missing. Keep
`ProfileConfig`, emit a `DeprecationWarning` from the alias for one release.

Same on `Report`: `save`/`save_html` and `show`/`display_in_notebook` are four
methods for two behaviours.

## Acceptance

- [ ] `dir(pysuricata)` shows only the public surface
- [ ] `repr(report)` fits on one line
- [ ] The three inputs above all raise `TypeError` with an actionable message

---

## 11. Configuration takes two constructors to change one number

`api` `ux` `dx`

## What it takes today

```python
from pysuricata import profile, ProfileConfig, ComputeOptions
profile(df, config=ProfileConfig(compute=ComputeOptions(chunk_size=50_000)))

ProfileConfig(chunk_size=50_000)   # TypeError: unexpected keyword argument
```

Three imports and two nested constructors to set one integer. The nesting models
the module layout, not the user's intent — nobody thinks *"I would like to
configure the compute subsystem"*, they think *"smaller chunks"*.

## Also

- **21 options in one flat namespace**, five of them checkpointing. An advanced
  recovery feature owns a quarter of what every user reads first.
- **No presets.** A user who wants "just make it faster" must work out which of 21
  knobs to turn. ydata-profiling's single most-used API feature is `minimal=True`
  — one word for an intent.

## Proposed

Purely additive; nothing existing breaks.

```python
profile(df)
profile(df, chunk_size=50_000, sample=20_000, correlations=False, seed=7)
profile(df, preset="fast")                     # or ProfileConfig.fast()
profile(df, config=ProfileConfig(...))         # unchanged escape hatch
```

Group the option object by concern (`sampling=`, `correlations=`, `checkpoint=`)
so the namespace is sorted by how many users need each thing.

## Acceptance

- [ ] The six most-used options are keywords on `profile()` and `summarize()`
- [ ] At least `fast` and `thorough` presets, documented with what they change
- [ ] `config=` still accepted and still wins over keywords

---

## 12. `profile()` rejects a file path that the CLI accepts

`api` `ux` `good first issue`

## What happens

```python
profile("data.csv")
# TypeError: Unsupported data type for this API...
```

```bash
pysuricata profile data.csv --output report.html   # works
```

The most natural first thing a Python user tries is the one thing the Python API
refuses while the CLI happily accepts it.

## Fix

Accept `str` and `os.PathLike` in `profile()` and `summarize()`, dispatching on
suffix to the reader the CLI already uses (`.csv`, `.parquet`, `.json`). Read in
chunks so the streaming guarantee holds for files too — which makes this a
*feature*, not just a convenience: `profile("huge.parquet", memory_budget="512MB")`
is the headline use case, expressed in one line.

Keep the error for genuinely unsupported types, and name the accepted ones.

## Acceptance

- [ ] `profile("data.csv")` and `profile(Path("data.parquet"))` work
- [ ] A file larger than memory profiles without loading it whole
- [ ] A missing file raises `FileNotFoundError`, not `TypeError`
