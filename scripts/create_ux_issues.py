#!/usr/bin/env python3
"""Create the GitHub issues from the UX review.

Twelve issues grouped by the five users they unblock, so the backlog reads as
"who is this for" rather than "what module does it touch".

    python scripts/create_ux_issues.py --print          # read them first
    python scripts/create_ux_issues.py --create         # file them
    python scripts/create_ux_issues.py --create --dry-run

Needs the GitHub CLI, authenticated:

    brew install gh && gh auth login

Every issue carries a measurement rather than an opinion, because a year from now
the number is what makes it re-checkable. Measurements are from a 20,000-row
customer table profiled with defaults at 0.0.26/0.0.27, and from the memory model
fitted at 0.0.27 — see docs/roadmap.md.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

REPO = "alvarodiez20/pysuricata"

# Labels are created if missing. Persona labels make the milestone story legible.
LABELS = {
    "ux": "8B5CF6",
    "api": "0EA5E9",
    "correctness": "DC2626",
    "report": "F59E0B",
    "dx": "10B981",
    "user:evaluator": "6B7280",
    "user:analyst": "6B7280",
    "user:engineer": "6B7280",
    "user:integrator": "6B7280",
    "user:bigdata": "6B7280",
    "good first issue": "7057FF",
}

ISSUES: list[dict] = [
    # ---------------------------------------------------------------- evaluator
    {
        "title": "Numeric columns with few distinct values are classified categorical",
        "labels": ["correctness", "ux", "user:evaluator"],
        "body": """\
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
""",
    },
    {
        "title": "Identifier columns get meaningless statistics",
        "labels": ["ux", "report", "user:evaluator"],
        "body": """\
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
""",
    },
    # ------------------------------------------------------------------ analyst
    {
        "title": "Report has no triage: 60 columns render at equal weight in source order",
        "labels": ["ux", "report", "user:analyst"],
        "body": """\
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
""",
    },
    {
        "title": "Histogram ignores the log-scale flag the card itself computed",
        "labels": ["ux", "report", "good first issue", "user:analyst"],
        "body": """\
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
""",
    },
    # ----------------------------------------------------------------- engineer
    {
        "title": "Add `pysuricata check` — a CI gate with an exit code",
        "labels": ["api", "dx", "user:engineer"],
        "body": """\
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
""",
    },
    # ---------------------------------------------------------------- integrator
    {
        "title": "Version the `summarize()` payload and reconcile it with the HTML",
        "labels": ["api", "user:integrator"],
        "body": """\
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
""",
    },
    # ------------------------------------------------------------------ bigdata
    {
        "title": "No progress feedback on long runs",
        "labels": ["ux", "dx", "user:bigdata"],
        "body": """\
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
""",
    },
    {
        "title": "Let users specify a memory budget",
        "labels": ["api", "ux", "user:bigdata"],
        "body": """\
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
""",
    },
    # -------------------------------------------------------------- cross-cutting
    {
        "title": "Ship `py.typed` so the annotations are visible to type checkers",
        "labels": ["api", "dx", "good first issue"],
        "body": """\
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
""",
    },
    {
        "title": "Tighten the public namespace: `__all__`, one-line repr, consistent errors",
        "labels": ["api", "dx", "good first issue"],
        "body": """\
Four small things in the same area.

## 1. `__all__`

`pysuricata.<TAB>` offers 15 names, **8 of them internal modules**
(`accumulators`, `api`, `compute`, `config`, `logger`, `render`, `report`,
`utils`). One line makes the public contract explicit.

## 2. `Report.__repr__` dumps the whole document

```python
>>> rep
Report(html='<!DOCTYPE html>\\n<html lang="en">...
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
""",
    },
    {
        "title": "Configuration takes two constructors to change one number",
        "labels": ["api", "ux", "dx"],
        "body": """\
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
""",
    },
    {
        "title": "`profile()` rejects a file path that the CLI accepts",
        "labels": ["api", "ux", "good first issue"],
        "body": """\
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
""",
    },
]


def run(cmd: list[str], dry: bool) -> None:
    """Execute ``cmd``, or print it verbatim when ``dry``.

    Issue bodies run to a few thousand characters, so the dry-run substitutes a
    readable placeholder for the ``--body`` argument. Everything else is shown
    exactly as it would be passed to ``gh``.
    """
    if dry:
        shown = []
        skip_next = False
        for part in cmd:
            if skip_next:
                shown.append(f"<body: {len(part)} chars>")
                skip_next = False
                continue
            shown.append(part)
            skip_next = part == "--body"
        print("  $ " + " ".join(shlex.quote(p) for p in shown))
        return
    subprocess.run(cmd, check=False)


def ensure_labels(dry: bool) -> None:
    for name, color in LABELS.items():
        run(
            [
                "gh",
                "label",
                "create",
                name,
                "--color",
                color,
                "--repo",
                REPO,
                "--force",
            ],
            dry,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="show")
    ap.add_argument("--create", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact gh commands without running any of them",
    )
    ap.add_argument("--markdown", default=None, help="write all issues to one file")
    args = ap.parse_args()

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write("# UX review — issue backlog\n\n")
            fh.write(
                f"{len(ISSUES)} issues, grouped by the user each one unblocks.\n\n"
            )
            for i, iss in enumerate(ISSUES, 1):
                fh.write(f"---\n\n## {i}. {iss['title']}\n\n")
                fh.write("`" + "` `".join(iss["labels"]) + "`\n\n")
                fh.write(iss["body"] + "\n")
        print(f"wrote {args.markdown}")

    if args.show:
        for i, iss in enumerate(ISSUES, 1):
            print(
                f"\n{'=' * 72}\n{i}. {iss['title']}\n   labels: {', '.join(iss['labels'])}\n{'=' * 72}"
            )
            print(iss["body"])

    if args.create or args.dry_run:
        if (
            not args.dry_run
            and subprocess.run(["which", "gh"], capture_output=True).returncode
        ):
            print("gh not found. brew install gh && gh auth login", file=sys.stderr)
            return 1
        ensure_labels(args.dry_run)
        for iss in ISSUES:
            cmd = [
                "gh",
                "issue",
                "create",
                "--repo",
                REPO,
                "--title",
                iss["title"],
                "--body",
                iss["body"],
            ]
            for label in iss["labels"]:
                cmd += ["--label", label]
            print(f"{'would create' if args.dry_run else 'creating'}: {iss['title']}")
            run(cmd, args.dry_run)

    if not (args.show or args.create or args.dry_run or args.markdown):
        for i, iss in enumerate(ISSUES, 1):
            print(f"{i:>2}. [{', '.join(iss['labels'])}]  {iss['title']}")
        print(f"\n{len(ISSUES)} issues. --print to read, --create to file them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
