---
title: CLI Reference
description: The pysuricata command line — profile, summarize and check
---

# CLI Reference

Installing PySuricata puts a `pysuricata` command on your path. It does the same
three things the library does, without writing a script:

```bash
pysuricata profile   data.csv --output report.html   # an HTML report
pysuricata summarize data.csv --output stats.json    # the numbers, as JSON
pysuricata check     data.parquet --baseline b.json  # a gate, with an exit code
```

`pysuricata --version` prints the installed version. `pysuricata <command>
--help` prints the options below.

All three take a **path**. CSV, Parquet, JSON, Arrow IPC (`.arrow`, `.feather`,
`.ipc`) and Excel (`.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods`) are all read
directly — Parquet and Arrow IPC a batch at a time and never as one frame, the
rest loaded whole, Excel unavoidably so since no engine puts a `chunksize` on
`read_excel`. See [Arrow, Parquet and DuckDB](data-sources.md).

---

## `profile`

Analyse a dataset and write a self-contained HTML report.

```bash
pysuricata profile data.csv --output report.html
pysuricata profile data.parquet -o report.html --seed 42 --no-correlations
```

| option | default | meaning |
|---|---|---|
| `file` | — | path to the data file (positional, required) |
| `--output`, `-o` | — | where to write the HTML (**required**) |
| `--title`, `-t` | package default | custom title for the report |
| `--seed`, `-s` | `0` | random seed; leave it alone for a reproducible report |
| `--chunk-size` | `100000` | rows per chunk while streaming |
| `--sample-size` | `20000` | reservoir size for quantiles and histograms |
| `--no-correlations` | off | skip the \(O(p^2)\) correlation step |
| `--quiet`, `-q` | off | suppress progress output |

---

## `summarize`

Analyse a dataset and emit the statistics as JSON. No HTML is rendered, so this
is the faster path when you only want the numbers.

```bash
pysuricata summarize data.csv                     # to stdout
pysuricata summarize data.csv --output stats.json # to a file
```

| option | default | meaning |
|---|---|---|
| `file` | — | path to the data file (positional, required) |
| `--output`, `-o` | stdout | where to write the JSON |
| `--seed`, `-s` | `0` | random seed |
| `--chunk-size` | `100000` | rows per chunk while streaming |
| `--quiet`, `-q` | off | suppress progress output |

Progress goes to **stderr**, never stdout, so `pysuricata summarize data.csv |
jq .dataset` stays parseable without `--quiet`.

The payload's keys are a versioned contract — see
[the `summarize()` schema](summary-schema.md).

---

## `check`

Compare a dataset against a stored baseline and exit non-zero when a threshold
is crossed. This is the piece that makes the same single pass usable as a CI
gate. [Gating CI on drift](data-checks.md) is the guide; this is the reference.

```bash
pysuricata check data.parquet --write-baseline baseline.json   # record
pysuricata check data.parquet --baseline baseline.json         # compare
```

### Exit codes

| code | meaning |
|---|---|
| `0` | passed — nothing crossed a threshold |
| `1` | a threshold was crossed |
| `2` | the check could not run (missing file, unreadable baseline, schema mismatch) |

`2` is deliberately distinct from `1`: a broken check is not a passing one, and
it is not a failing dataset either.

### Baseline

| option | default | meaning |
|---|---|---|
| `file` | — | path to the data file (positional, required) |
| `--baseline`, `-b` | — | baseline JSON to compare against |
| `--write-baseline` | — | write a baseline from this dataset and exit |

`check` reads `schema_version` off a stored baseline and **refuses** to compare
against one written by a different version, rather than quietly matching
whatever keys still line up.

### Thresholds

| option | default | meaning |
|---|---|---|
| `--thresholds` | — | a `.json` or `.toml` file overriding the defaults |
| `--max-missing-pct` | — | fail if any column is missing more than this percentage |
| `--min-rows` | — | fail if the dataset has fewer rows than this |
| `--max-duplicate-pct` | — | fail if duplicate rows could be above this percentage — gated on the upper bound, so a count below the sketch's own resolution fails closed rather than passing as zero |
| `--max-rows-drift-pct` | — | fail if the row count moved more than this percentage from the baseline |
| `--fail-on-new-column` | off | treat an added column as a breach |
| `--fail-on-range-expansion` | off | treat a new minimum or maximum outside the baseline range as a breach |

A thresholds file keeps the rules with the repository rather than in a workflow
argument list, which is usually what you want once there is more than one.

### Freshness

Stale data is not drifted data — an extract that simply did not run looks
identical to yesterday's, and every distributional check passes.

| option | default | meaning |
|---|---|---|
| `--max-age` | — | fail if the newest timestamp in a datetime column is older than this, e.g. `26h`, `3d`, `90m`. Needs no baseline |
| `--require-fresh` | off | fail if a datetime column's newest timestamp did not advance past the baseline's — catches a re-run of yesterday's extract |

### Output and streaming

| option | default | meaning |
|---|---|---|
| `--json` | off | emit the result as JSON on stdout instead of text |
| `--warn-only` | off | report findings but always exit `0` |
| `--seed`, `-s` | `0` | random seed, so a re-run of the same data is a no-op |
| `--chunk-size` | `100000` | rows per chunk while streaming |
| `--quiet`, `-q` | off | suppress progress output |

`--warn-only` is for the period between adding a check and trusting it: you see
what it would have caught without turning the build red.

---

## In GitHub Actions

```yaml
- name: Check the daily extract
  run: |
    pysuricata check data/extract.parquet \
      --baseline baselines/extract.json \
      --max-missing-pct 5 \
      --require-fresh
```

A non-zero exit fails the step. Full workflow, including how to keep the
baseline current, in [Gating CI on drift](data-checks.md).

---

## See Also

- [Gating CI on drift](data-checks.md) — the workflow `check` was built for
- [The `summarize()` schema](summary-schema.md) — what the JSON contains
- [High-Level API](api.md) — the same three operations from Python
