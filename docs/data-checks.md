# Gating CI on drift

!!! info "Examples on this page assume two frames: `df`, the reference, and `new_df`, today's data"

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"amount": rng.standard_normal(2_000)})
    new_df = pd.DataFrame({"amount": rng.standard_normal(2_000) + 0.4})
    ```

`pysuricata check` compares a dataset against a stored baseline and exits
non-zero when something moved. It is the one command with a meaningful exit
code — `profile` and `summarize` exit 0 no matter what they found.

Unlike an expectations framework, there is nothing to author first. The
baseline is a `summarize()` payload, so the shape of yesterday's data *is* the
expectation.

## Two commands

Write a baseline from data you trust:

```bash
pysuricata check data.parquet --write-baseline baseline.json
```

Check the next batch against it:

```bash
pysuricata check data.parquet --baseline baseline.json
```

```text
check failed — 3 findings
  order_id: column is missing from the data
  amount: mean moved 2.02σ, -0.00453225 to 2.0105 (limit 0.5σ, baseline σ=0.995494)
  active: true rate moved 41.04 points, 50.64% to 9.60% (limit 10)
```

Commit `baseline.json` next to the data pipeline that produced it.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Nothing crossed a threshold |
| 1 | A threshold was crossed |
| 2 | The check could not run — file missing, baseline unreadable, bad thresholds |

The 1/2 split is deliberate: a pipeline should be able to treat drifted data
and a missing file as different incidents.

`--warn-only` reports findings and still exits 0, which is how you introduce
the gate to a repository that has never had one.

## What it compares

| Kind | Signal | Default |
|---|---|---|
| `schema` | Column removed, or its type changed | on |
| `schema` | Column added | off (`--fail-on-new-column`) |
| `rows` | Row count moved | off (`--max-rows-drift-pct`) |
| `missing` | A column's missing rate moved, in percentage points | 5 |
| `cardinality` | Approximate distinct count moved | 25% |
| `distribution` | Mean or median moved, in baseline standard deviations | 0.5σ |
| `distribution` | Standard deviation grew or shrank by a factor | 2× |
| `boolean` | Share of `True` moved, in percentage points | 10 |
| `range` | New minimum or maximum outside the baseline range | off |

Three choices in that table are worth explaining, because they are what stops
the gate from crying wolf.

**Growth is not drift.** Appending rows is the normal life of a dataset, so row
count drift is off by default. For the same reason the cardinality check
requires both the distinct *count* and the distinct *rate* to move: doubling
the rows doubles a continuous column's distinct count while leaving its rate
alone, and leaves a three-level enum's count alone while halving its rate.
Gating on either one alone fails every build that appends data.

The cost of that rule: when the row count also moved a lot, a small change in
levels sits inside the band growth could explain and is not reported. When the
row count holds — the same query run a day later, the common CI shape — it is
exactly as sensitive as gating on the raw count.

**Distribution drift is measured in standard deviations, not percent.** A
percentage change in the mean is meaningless when the mean is near zero and
incomparable across columns with different units. `|Δmean| / σ` is neither.

**Approximate quantities get loose thresholds.** `unique_est` is a KMV sketch
estimate with relative error around `1/√k` — about 2.2% at the default
`uniques_k=2048`. A cardinality threshold anywhere near that fires on
estimation noise rather than on drift, so the default sits an order of
magnitude above it, and any threshold you set inside the noise floor is called
out in the output:

```text
note: max_unique_drift_pct=1 is close to the KMV sketch error (~2.2% at k=2048);
this gate may fail on estimation noise. Raise it, or profile with a larger uniques_k.
```

Findings that rest on an estimate are labelled `(approximate)` in the output
and carry `"approximate": true` in the JSON.

## Thresholds

Common ones have flags:

```bash
pysuricata check data.parquet --baseline baseline.json \
    --max-missing-pct 5 --min-rows 10000 --max-rows-drift-pct 20
```

`--max-missing-pct` and `--min-rows` are absolute — they need no baseline, so
they work on the very first run.

Everything else lives in a file, in JSON or TOML:

```toml
# thresholds.toml
[thresholds]
max_missing_drift_pp = 2.0
max_mean_shift_sigma = 0.25
max_unique_drift_pct = 30.0
max_std_ratio = 1.5
fail_on_new_column = true
max_rows_drift_pct = 50.0
```

```bash
pysuricata check data.parquet --baseline baseline.json --thresholds thresholds.toml
```

A `[tool.pysuricata.check]` table in `pyproject.toml` is read the same way, so
the thresholds can live with the rest of the project's configuration.

Set any threshold to `null` (JSON) or omit it and set it to nothing to disable
that check. Precedence is defaults, then the file, then command-line flags.

A misspelled threshold is an error rather than a silent no-op — a typo that
quietly loosens a gate is the worst failure mode a gate has:

```text
Error: unknown threshold(s): max_mean_shift. Known thresholds: fail_on_new_column,
fail_on_range_expansion, max_mean_shift_sigma, ...
```

## In GitHub Actions

```yaml
name: data-check

on:
  schedule:
    - cron: "0 6 * * *"
  workflow_dispatch:

jobs:
  drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pysuricata
      - name: Fetch today's extract
        run: ./scripts/fetch_extract.sh   # writes data/extract.parquet
      - name: Check for drift
        run: |
          pysuricata check data/extract.parquet \
            --baseline baselines/extract.json \
            --thresholds thresholds.toml \
            --json | tee check.json
      - name: Keep the result
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: drift-report
          path: check.json
```

The job fails when the check exits 1. `--json` writes a machine-readable result
to stdout while progress goes to stderr, so the pipe stays parseable.

To refresh a baseline after an intentional change, run `--write-baseline` and
commit the result in the same pull request as the change that caused it. That
makes the drift visible in review rather than in a red build.

## From Python

The comparison is importable, and takes `summarize()` payloads:

```python
from pysuricata import summarize
from pysuricata.check import Thresholds, compare, make_baseline, write_baseline

baseline = make_baseline(summarize(df, seed=0), source="2026-08 reference")
write_baseline(baseline, "baseline.json")

result = compare(summarize(new_df, seed=0), baseline, Thresholds(max_mean_shift_sigma=0.25))
if not result.passed:
    for finding in result.findings:
        print(finding.kind, finding.column, finding.message)
```

`CheckResult.to_dict()` gives the same structure the CLI prints under `--json`.

## Reproducibility

`check` defaults to `--seed 0` rather than to no seed, so re-running it on
unchanged data is a no-op rather than a coin flip. Use the same seed when
writing the baseline and when checking against it.

The baseline records the `pysuricata` version and the payload's
`schema_version`. Reading a baseline whose schema does not match the running
version is an error that tells you to regenerate it, rather than a comparison
that silently succeeds against fields that moved.
