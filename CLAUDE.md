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

See `docs/roadmap.md` (**v8, re-audited at 0.0.62**) for the measured numbers and
the ordering. Everything this section used to list as pending has landed: the
chunk-size cap, the Misra-Gries gate, the KMV threshold pre-filter, the datetime
vectorisation and the stale `-2e18` bound.

**Phase 0 and Phase 1 are complete**, and the report redesign (#110–#125) is
fifteen issues of sixteen closed. Do not regress any of it — a change that makes
`benchmarks/accuracy.py` fail is wrong even if it is faster, and the same now
goes for `tests/test_report_data_invariance.py`.

Next, in order: close out the redesign (**#122**, which carries a height
decision that needs the user rather than more work; then #124, #121), **#139**
(reopened — per-column per-chunk missing counts are never produced), regenerate
the stale example report in `docs/assets/`, publish (#38), prove the memory
claim (#92) before shipping the budget (#79), then the native core (#44) —
**KMV first, moments last**, since moments are ~1.4% of the numeric path and KMV
was half of it.

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
  and `<style>` before asserting anything about markup.
