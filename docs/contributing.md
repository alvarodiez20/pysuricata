---
title: Contributing Guide
description: How to contribute to PySuricata development
---

# Contributing to PySuricata

Thank you for considering contributing to PySuricata! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- `uv` package manager (recommended) or `pip`
- Git

### Clone Repository

```bash
git clone https://github.com/alvarodiez20/pysuricata.git
cd pysuricata
```

### Install Dependencies

=== "Using uv (recommended)"
    ```bash
    uv sync --dev
    uv run python -c "import pysuricata; print('Success!')"
    ```

=== "Using pip"
    ```bash
    pip install -e ".[dev]"
    python -c "import pysuricata; print('Success!')"
    ```

## Running Tests

```bash
# The suite. `not benchmark` is not optional -- a bare `pytest` pulls the
# benchmark modules in, and they take minutes.
uv run pytest -m "not benchmark"

# With coverage
uv run pytest -m "not benchmark" --cov=pysuricata --cov-report=html

# One file
uv run pytest tests/test_numeric.py
```

## The Gates

Five things guard behaviour that is easy to break without noticing. Run them
before opening a pull request; CI runs them all.

### The accuracy oracle

```bash
uv run pytest benchmarks/accuracy.py -v
```

Asserts that **chunked results equal unchunked results**. It is the invariant
the whole design rests on and the one most likely to break — a change that makes
this fail is wrong even if it is faster.

### The docs checker

```bash
uv run python -m benchmarks.check_docs --strict     # exit 1 on any ERROR
uv run python -m benchmarks.check_docs --quiet-info # the readable report
uv run python -m benchmarks.check_docs --json out.json
```

Runs every code fence in `docs/` **and the README** against the live API, checks
that every documented symbol is exported, and reports pages on disk that the nav
never renders. If you add a fence, it will be executed — give it its imports.

### The three ratchets

| ratchet | guards |
|---|---|
| `tests/test_report_layout.py` | report bytes, and elements per card |
| `tests/test_colour_tokens.py` | untokenised colours |
| `tests/test_processed_bytes_placement.py` | `Processed bytes` staying out of a stat row |

Each fails **in both directions**. Growth is a regression; shrinking asks you to
lower the baseline, so a win cannot be quietly respent.

### Data invariance

```bash
uv run pytest tests/test_report_data_invariance.py
```

Asserts the *facts* have not moved while the *presentation* has. This is what
made a seventeen-commit rewrite of every template reviewable.

### Browser layout checks

Deliberately not in `dev`: Chromium is ~300 MB and only 31 cases need it, so
they **skip themselves** when it is absent — which means you can break a layout
criterion and still see green locally.

```bash
uv sync --all-extras --group browser
uv run playwright install chromium
uv run pytest -m browser
```

`uv run python scripts/contact_sheet.py` produces six review captures. It is
never a gate.

## Benchmarks

```bash
python -m benchmarks.hotspots     # where profile() spends its time
python -m benchmarks.kernels      # per-kernel timings + memory roofline
python -m benchmarks.end_to_end --markdown results.md   # vs ydata/sweetviz/skimpy
python -m benchmarks.versions     # this version against previous ones
```

Two rules, both learned the hard way:

- **`cProfile` charges per Python call**, so it over-weights kernels that make
  many small ones. It ranked the reservoir sampler at ~30% of self time when a
  5x-faster replacement moved wall clock by 4%. Confirm any ranking against wall
  clock with the profiler off.
- **A ratio is only quotable when both sides were measured in the same
  round-robin, on the same machine, in the same run, with nothing else
  running.** The last clause is why both harnesses read the load average and
  refuse above one process per core without `--force`: a run once showed a 10.5%
  regression that was the coverage suite running in parallel.

## The Native Crate

The Rust kernels in `native/` are an **optional accelerator**. The pure-Python
path is the reference implementation and must never be deleted to "simplify" —
the two must agree within documented tolerance.

```bash
cargo test --lib --manifest-path native/Cargo.toml
maturin develop --release -m native/Cargo.toml
```

## Pre-commit

`pre-commit` is a dev dependency and the ruff revision in
`.pre-commit-config.yaml` is kept in sync with `pyproject.toml`.

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Code Style

PySuricata uses **Ruff** for linting and formatting.

Line length 88, target Python 3.10, `from __future__ import annotations` at the
top of every module.

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Auto-fix
uv run ruff check --fix .
```

### Style Guidelines

- Follow PEP 8
- Line length: 88 characters (Black-style)
- Use type hints for function signatures
- Docstrings: Google style

Example:

```python
def compute_mean(values: np.ndarray) -> float:
    """Compute arithmetic mean of values.

    Args:
        values: Array of numeric values

    Returns:
        Mean value

    Raises:
        ValueError: If array is empty
    """
    if len(values) == 0:
        raise ValueError("Cannot compute mean of empty array")
    return float(np.mean(values))
```

## Documentation

### Build Documentation Locally

```bash
# Install docs dependencies
uv sync --dev

# The example report is generated, not committed. Every docs workflow runs
# this before building; run it once locally so the iframe on the home page
# has something to show.
uv run python scripts/regenerate_example_report.py

# Build docs
uv run mkdocs serve

# Open http://localhost:8000 in browser
```

`docs/assets/titanic_report.html` is in `.gitignore`. It used to be committed,
and it drifted from 0.0.17 onward until it was 1,180,196 bytes against 600,028
of real output -- because every rendering change either produced a megabyte diff
or produced none, and nobody wanted the diff. `mkdocs build --strict` does not
need the file to be present.

### Documentation Style

- Use clear, concise language
- Include code examples
- Add mathematical formulas for algorithms
- Link to related pages
- Update relevant sections when changing code

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Changes

- Write tests for new functionality
- Update documentation
- Follow code style guidelines
- Keep commits atomic and well-described

### 3. Run Checks

```bash
uv run ruff format . && uv run ruff check .
uv run pytest -m "not benchmark"
uv run pytest benchmarks/accuracy.py -v
uv run python -m benchmarks.check_docs --strict
uv run python scripts/regenerate_example_report.py && uv run mkdocs build --strict
```

One more, if you are about to open the pull request from a branch you created
with `git checkout -B`: that command **aborts against uncommitted changes** and
says so in one line, which is easy to miss — a commit made afterwards lands on
the old base and looks fine.

```bash
git merge-base --is-ancestor origin/main HEAD && echo "based on main"
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add support for XYZ"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `test:` - Test updates
- `chore:` - Build/tooling changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots for UI changes
- Checklist of completed items

## Testing Guidelines

### Unit Tests

Test individual functions/classes in isolation.

```python
def test_welford_mean():
    """Test Welford mean computation"""
    from pysuricata.accumulators.algorithms import StreamingMoments

    moments = StreamingMoments()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    for v in values:
        moments.update(np.array([v]))

    result = moments.finalize()
    assert abs(result["mean"] - 3.0) < 1e-10
```

### Integration Tests

Test components working together.

```python
from pysuricata import profile
def test_full_profile():
    """Test end-to-end profiling"""
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    report = profile(df)

    assert report.html is not None
    assert len(report.stats["columns"]) == 2
```

### What to Assert

Two properties are worth reaching for beyond "the number is right", because they
are the ones this codebase's bugs hide behind.

**Order independence and mergeability.** An accumulator must produce the same
answer however the data is split, and `merge` must be exact:

```python
import numpy as np

from pysuricata.accumulators import NumericAccumulator

values = np.arange(1_000, dtype=float)

whole = NumericAccumulator("x")
whole.update(values)

left, right = NumericAccumulator("x"), NumericAccumulator("x")
left.update(values[:400])
right.update(values[400:])
left.merge(right)

assert left.finalize().mean == whole.finalize().mean
```

**That a control resolves.** A class renamed on one side of the
render/JavaScript boundary produces no error and no console warning, just a
button that goes quiet — `+ add a note` did nothing for eleven versions this
way, and 1,735 tests did not notice because not one asserted that a selector
*resolves*. `tests/test_js_selectors_match_markup.py` now pairs every
`closest()` selector against the markup that must match it; extend it when you
add a control.

One trap when asserting over rendered output: **the report inlines its own CSS
and JS**, so searching the whole document for a class name finds it in the very
source that references it. Strip `<script>` and `<style>` first, or require a
`class="` attribute. `"dt-svg" in html` was `True` for a class no element
carried.

## Architecture Overview

```
pysuricata/
├── api.py              # profile() / summarize() -- the public surface
├── cli.py              # the `pysuricata` command: profile, summarize, check
├── check.py            # baselines, thresholds and findings behind `check`
├── comparison.py       # compare() and the delta dataclasses
├── sources.py          # Parquet / Arrow IPC / DuckDB batch readers
├── config.py           # EngineConfig -- internal, built from ComputeOptions
├── report.py           # build_report(): the orchestration entry point
├── checkpoint.py       # periodic state to disk for long runs
├── progress.py         # progress= reporting, always to stderr
├── io/                 # shared reader plumbing
├── accumulators/       # the statistical core
│   ├── algorithms.py   # StreamingMoments, ExtremeTracker, monotonicity
│   ├── sketches.py     # KMV, MisraGries, ReservoirSampler, RowKMV
│   ├── numeric.py  categorical.py  datetime.py  boolean.py
│   └── factory.py      # accumulator selection and per-column seeding
├── compute/
│   ├── orchestration/  # engine.py -- the chunk loop and adapter dispatch
│   ├── adapters/       # pandas.py, polars.py -- frame-shaped I/O
│   ├── processing/     # chunking.py, inference.py
│   ├── analysis/       # correlation.py
│   ├── consume.py      # pandas chunk -> accumulator wiring
│   └── consume_polars.py
├── render/             # HTML generation; html.py drives the template
├── templates/          # report_template.html -- placeholders, one regex pass
└── static/             # CSS, JS, images, all inlined into the report
native/                 # optional Rust kernels (PyO3 + maturin)
benchmarks/             # accuracy oracle, performance harness, docs checker
```

Data flows one way: an adapter yields chunks, `consume_chunk_*` converts each
column to an array, the matching accumulator's `update()` folds it in,
`finalize()` produces a summary dataclass, and `render/` turns summaries into
HTML. **Accumulators never see the frame, only arrays.**

Two invariants to keep:

- Accumulators are **mergeable** and **order-independent** wherever the
  statistic allows it. Chunked must equal unchunked.
- Approximate values are **labelled approximate**. Sketches carry error bounds;
  surface them rather than printing an estimate as an exact integer.

And one rule: never touch the global RNG. Seeds belong to the accumulator
instance.

## Adding New Features

### Add New Statistic to Numeric Analysis

Five steps, and the fifth is the one people forget.

**1. Carry the state on the accumulator** (`pysuricata/accumulators/numeric.py`).
It must be foldable one array at a time and combinable in `merge`, or the
chunked-equals-unchunked invariant breaks:

```text
NumericAccumulator.__init__   add the state, seeded from the instance's own RNG
NumericAccumulator.update     fold one array of values into it
NumericAccumulator.merge      combine this state with another accumulator's
NumericAccumulator.finalize   emit it on the summary dataclass
```

**2. Add the field to the summary dataclass**, with a default, so an accumulator
that never saw the column still finalizes.

**3. Render it** in the matching card under `pysuricata/render/`.

**4. Test the arithmetic** *and* the invariant — that splitting the input in two
and merging gives the same answer.

**5. Publish it, or withhold it deliberately.** Every computed statistic must be
either in the `summarize()` payload or listed in
`pysuricata.report.SUMMARY_FIELDS_WITHHELD` with a reason. A test walks that
list against the summary dataclasses and fails if a statistic is neither. That
is what stops the JSON drifting behind the HTML — it has happened twice, with
correlations and with numeric top values, and both times it was only findable by
reading the renderer.

If the new value is an **estimate**, it carries its error bound. Adding an
approximate number without one is the thing the project is most careful about.
See [the `summarize()` schema](summary-schema.md).

5. **Update documentation** (`docs/stats/numeric.md`):
```markdown
### New Statistic

Mathematical definition:
\[
\text{NewStat} = \sum_{i=1}^{n} f(x_i)
\]

Interpretation: ...
```

## Release Process

Documented once, in [Versioning](versioning.md) — the five-stage pipeline, what
`scripts/check_version.py` enforces about a version bump, and the gates for
1.0.0.

A pull request does **not** have to bump the version. If it does, the bump must
be a legal step — one component raised, the ones below it reset, nothing
skipped, no downgrade — with a matching `CHANGELOG.md` section.

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Assume good intentions

## Getting Help

- [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- [GitHub Issues](https://github.com/alvarodiez20/pysuricata/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
