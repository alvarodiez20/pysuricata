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

See `docs/roadmap.md` (v3, re-audited at 0.0.26) for the measured numbers.

Phase 0 is **complete**: all eleven audit items are closed, the accuracy oracle
is green at 51 tests, and five targeted test files (63 tests) cover the items
that used to have none. Do not regress any of it — a change that makes
`benchmarks/accuracy.py` fail is wrong even if it is faster.

**Phase 1 has three changes left, worth 1.88x together. All pure Python.**

1. **Cap the auto-chosen chunk size near 50,000 rows** (`compute/processing/chunking.py`).
   Removing the `0.7*optimal + 0.3*requested` blend made `chunk_size` a real
   option and exposed that the heuristic's own value is too large: a 200k-row
   frame is now processed as one chunk. `chunk_size=50_000` is **1.34x** on
   mixed 200k x 14. Add a test asserting the chosen size stays in a sane band.
2. **Gate Misra-Gries on the KMV estimate** (`accumulators/numeric.py:335`).
   **34%** of the numeric accumulator. On high-cardinality columns it renders a
   "Common values" table of values that occurred *once* — so this removes
   misleading output as well as cost. `should_track_top_k` in
   `benchmarks/proposed_kernels.py` is written and verified against four column
   shapes.
3. **KMV threshold pre-filter** in `_batch_add_hashes` — **8.7x**, three lines,
   estimates provably identical. KMV is **52%** of the numeric accumulator, the
   largest single kernel. Code and proof in `benchmarks/proposed_kernels.py`.

Then: **vectorise `DatetimeAccumulator.update`** — four per-row Python loops
(`datetime.py:212, 234, 277, 280`), ~960 ms/column at 200k rows, the most
expensive column kind and the only accumulator never touched. And fix the stale
`-2e18` bound at `datetime.py:324`, which `_update_fallback` still carries after
the window was widened everywhere else.

**Then publish**, then the native core — **KMV first, moments last**. Moments are
1.4% of the numeric path; KMV is 52%. The crate under `native/` is vendored but
nothing imports it and there is no `[fast]` extra yet.

Measurement discipline, both learned on this codebase:

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
  round-robin, on the same machine, within the same run.** Two published claims
  came from cross-session pairing: "0.0.21 is 1.24x faster" is really 0.88x, a
  regression, and a 3.56x headline is really 2.48x. `benchmarks/end_to_end.py`
  and `benchmarks/versions.py` interleave every tool and version across rounds
  and label anything under three rounds *Not quotable*.
