# PySuricata — working notes for Claude Code

Streaming EDA profiler. Single pass over the data, bounded memory, emits a
self-contained HTML report. pandas + polars native.

## Commands

```bash
uv sync --dev
uv run pytest -m "not benchmark"            # test suite
uv run pytest benchmarks/accuracy.py -v     # statistical accuracy oracle
uv run ruff check . && uv run ruff format . # lint (line-length 88, py310 target)
uv run mkdocs serve                         # docs
uv run python -m benchmarks.check_docs --strict   # docs + README vs the live API

# Layout acceptance criteria in a real browser (#124). Not in `dev`: Chromium is
# ~300 MB and only these 31 cases need it, so they skip when it is absent.
uv sync --all-extras --group browser && uv run playwright install chromium
uv run pytest -m browser
uv run python scripts/contact_sheet.py      # 6 review captures, never a gate

# Real Pyodide boot + profile run, asserted on rendered pixels not markup (#1).
# Slow and network-dependent (PyPI, jsDelivr) -- not part of `pytest -m browser`,
# runs post-release in cd.yml's demo-check job against the live site instead.
uv run python web/e2e.py                                    # local web/, over loopback
uv run python web/e2e.py --url https://pysuricata.pages.dev # the live demo

python -m benchmarks.hotspots               # where does profile() spend its time
python -m benchmarks.kernels                # per-kernel timings + memory roofline
python -m benchmarks.end_to_end --markdown results.md   # vs ydata/sweetviz/skimpy

cargo test --lib --manifest-path native/Cargo.toml      # native kernels
maturin develop --release -m native/Cargo.toml          # build + install locally
```

## Architecture

```
pysuricata/
  api.py              profile() / summarize() — public surface
  config.py           ProfileConfig; ComputeOptions is the user-facing knob set
  compute/
    orchestration/    engine.py — the chunk loop, adapter dispatch, checkpointing
    adapters/         pandas.py, polars.py — frame-shaped I/O
    processing/       chunking.py (chunk sizing), inference.py (column typing)
    analysis/         correlation.py
    consume.py        pandas chunk -> accumulator wiring
    consume_polars.py polars equivalent
  accumulators/       the statistical core — numeric, categorical, datetime, boolean
    algorithms.py     StreamingMoments (Welford/Pébay), ExtremeTracker, monotonicity
    sketches.py       KMV distinct-count, MisraGries top-k, ReservoirSampler, RowKMV
  render/             HTML generation; html.py is the template driver
  templates/, static/ report shell, CSS, JS
native/               optional Rust kernels (pysuricata-core, PyO3 + maturin)
benchmarks/           accuracy oracle + performance harness
```

Data flows one way: adapter yields chunks -> `consume_chunk_*` converts each
column to an array -> the matching accumulator's `update()` folds it in ->
`finalize()` produces a summary dataclass -> `render/` turns summaries into HTML.
Accumulators never see the frame, only arrays.

## Conventions

- Accumulators must be **mergeable** and **order-independent** where the statistic
  allows it. Chunked results must equal unchunked results; that invariant is
  asserted in `benchmarks/accuracy.py` and is the thing most likely to break.
- Approximate values must be labelled approximate. Sketches carry error bounds;
  surface them rather than printing a sketch estimate as an exact integer.
- Never touch the global RNG. Seeds belong to the accumulator instance.
- The pure-Python path is the reference implementation. The native crate is an
  optional accelerator and must agree with it within documented tolerance —
  never delete the Python path to "simplify".
- Ruff, line length 88, `from __future__ import annotations` at the top of modules.

## Current priorities

**The issue tracker is the authority.** `docs/roadmap.md` was v10, pinned to
0.0.62, and shipped in the docs nav describing a project ninety releases older
than the one readers were installing; it is deleted rather than re-synced,
because a roadmap in the docs dates the moment it is written and nothing after.
The working roadmap is **v15** and lives outside the repo. Half of #251 goes
with the file; what is left of it is the per-column figure, now measured.

**0.1.5 is published.** The report redesign is closed out, and so are
both packaging blockers, the README drift, the per-chunk missing counts and the
Missing-pane gate. Do not regress any of it — a change that makes
`benchmarks/accuracy.py` fail is wrong even if it is faster, and the same goes
for `tests/test_report_data_invariance.py`.

Three ratchets now guard things that only go one way. Each fails **in both
directions**: growth is a regression, and shrinking asks you to lower the
baseline so the win cannot be quietly respent.

| ratchet | where |
|---|---|
| report bytes, and elements per card | `tests/test_report_layout.py` |
| untokenised colours | `tests/test_colour_tokens.py` |
| `Processed bytes` still in a stat row | `tests/test_processed_bytes_placement.py` |

Next, in rough order:

1. **The reach ladder** — the half v12–v14 dropped. #247 (Arrow IPC does not
   load, which is the one format another language writes) then #250 (an
   `action.yml` over `pysuricata check`, and a JSON Schema for the payload).
   Cheapest reach per unit effort in the project.
2. **Two one-hour corrections**: #248 (the duplicate threshold is 1σ, so a clean
   frame false-alarms ~10% of runs) and #249 (a `sys.path` fall-through that
   `__version__` cannot detect and that corrupts every round *identically*).
3. **The column axis** — #207, and #39 for the report side. Bounded memory holds
   in rows (189 MB at 2M, 190 MB at 5M) and fails in columns (929 MB at
   20,000 × 600, on *less* data). Nothing in the field handles wide data, so
   this is where a weakness converts into a claim. Exit: a 600-column frame
   inside a 512 MB runner, which also answers #92 and unblocks #79.

   Note the largest remaining report saving is **no longer tracked**. #206 is
   closed on its cheap half (repeated constants out of every bar, 73,204 →
   ~63,600 bytes per numeric column), but the six pre-rendered histogram variants
   it was filed about are all still emitted, and they are ~65% of a numeric
   column. Collecting that needs a JS port of a ~170-line SVG renderer — a second
   implementation of the chart, which the reference-implementation rule under
   **Conventions** argues against. Re-file it before building it.
4. **Publish** (#38), then the native core (#44) — **KMV first, moments last**,
   since moments are ~1.4% of the numeric path and KMV was half of it. #108's
   abstraction boundary measured at 0.97–1.01×, so the preparation cost nothing.

**Two open issues are blocked on a decision, not on effort.** Do not guess at
them: #209 (categorical has no Statistics pane and boolean has no details
section by an explicit earlier decision, so neither has a home for the row) and
#150 (the best reachable demo dataset satisfies three of four acceptance
criteria; the one that satisfies all four is not reachable from CI).

Measurement discipline, all learned on this codebase:

- `cProfile` charges per Python call and over-weights kernels that make many
  small ones. It ranked the reservoir at ~30% of self time when swapping in a
  5x-faster one moved wall clock by 4%. Confirm rankings against wall clock with
  the profiler off.
- When checking whether a value reaches the report, search for the **formatted**
  string. An earlier audit wrongly concluded top-k output was discarded because
  it searched for `4248` in a report that renders `4,248`.
- A kernel benchmark only measures the call sites it calls. Holding `KMV._values`
  as an array won its own benchmark by 2.7x and lost 35% end to end, because the
  benchmark never touched the scalar insert path categorical columns hammer.
- **A ratio is only quotable when both sides were measured in the same
  round-robin, on the same machine, within the same run — and nothing else was
  running.** Two published claims came from cross-session pairing: "0.0.21 is
  1.24x faster" is really 0.88x, a regression, and a 3.56x headline is really
  2.48x. `benchmarks/end_to_end.py` and `benchmarks/versions.py` interleave
  every tool and version across rounds and label anything under three rounds
  *Not quotable*.

  The last clause is the one interleaving cannot buy you, because a neighbour
  is not in the round-robin. A run once put 0.0.61 at 1,599 ms against 0.0.42's
  1,448 — a 10.5% regression on a harness that reproduces to ±1%, with a
  ready-made culprit in the abstraction boundary #108 had just added to the
  accumulator hot path. Bisecting seven commits refused it (1,203–1,271 ms, no
  trend, HEAD at 1.008x): the coverage suite was running in parallel. Both
  harnesses now read the load average, **refuse above one per core** unless
  `--force`, and record the load at both ends in the exported results, so a
  contended run carries its own caveat (#212).
- **A check over rendered output is only as good as the markup the fixture
  reaches.** A frame of `[1.0, 2, 3, 4, 5] * 40` has five distinct values and
  profiles as *categorical*, so a report built from it has no numeric card and
  every numeric-card selector looks dead. A frame with no quality problems
  renders no `.needs-attention` block, so the flag filter looks dead too. Both
  controls work. A fixture that misses a branch reports "absent", not "unknown",
  and absent reads as broken — confirm in a browser before calling anything dead.
- **The report inlines its own CSS and JS**, so searching the whole document for
  a class name finds it in the very source that references it. Strip `<script>`
  and `<style>` before asserting anything about markup — or require a `class="`
  attribute. `"dt-svg" in html` was `True` for a class no element carried.
- **In a browser, the obvious measurement reads the wrong box.** Three of #124's
  acceptance criteria looked failed and were not: the header is 53px by
  `getBoundingClientRect()` and exactly `52px` by computed height, because the
  rect counts a 1px border the budget does not; `.icon-btn` is 30×30 and its
  *hit* area is a 44×44 absolutely positioned `::after`, which `elementFromPoint`
  confirms and a rect cannot see; and `scrollWidth > clientWidth` names nine
  elements at 1240px, none of which scrolls — a pane scrolls only if its content
  overflows **and** its `overflow-x` does. Encode the invariant, not the box that
  is easiest to read.
- **A parametrised axis can be inert.** The report's dark mode is the *absence*
  of a `light` class, not `prefers-color-scheme`, so Playwright's
  `color_scheme=` did nothing and six "theme" cases measured one state twice —
  visible only because the contact sheet came out byte-identical in pairs.
  Toggling the class is not enough either: `transition: background-color 0.3s`
  means an immediate read returns the old value. Make the axis prove it moved.
- **A failing coverage check is a finding, not a chore.** `codecov/patch`
  flagged an untested polars branch; writing the test showed the branch was
  *unreachable*, and its twin in the accumulator was putting
  `"time_zone='US/Eastern')"` into the `summarize()` payload. polars dtypes
  contain a comma, so the pandas branch matched them first.
- **`functionality.js` and the renderers never import each other.** A class
  renamed on one side produces no error and no console warning, just a control
  that goes quiet — the datetime timeline's tooltip was dead this way.
  `tests/test_js_selectors_match_markup.py` pairs every `closest()` selector
  against the markup that must match it.
- **`git checkout -B` aborts against uncommitted changes**, and prints one line
  saying so. A commit made afterwards lands on the old base and looks fine.
  Check `git merge-base --is-ancestor origin/main HEAD` before opening a PR.
