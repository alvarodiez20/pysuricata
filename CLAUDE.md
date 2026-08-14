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

See `docs/roadmap.md` (v2, re-audited at 0.0.21) for the measured numbers behind
this. Shortest version:

Phase 0 is **done**: all six statistical bugs are fixed, the accuracy oracle is
at 578 lines with zero xfails, and it runs in CI. Do not regress it — a change
that makes `benchmarks/accuracy.py` fail is wrong even if it is faster.

1. **Finish Phase 1 (pure Python, no Rust), in measured value order:**
   - Gate Misra-Gries off high-cardinality numeric columns. It is **35%** of the
     numeric accumulator and `numeric_card.py:462` discards its output on the
     float columns where it never applies.
   - Apply the KMV threshold pre-filter in `_batch_add_hashes` — **8.8×**,
     three lines, estimates provably identical. Code and proof in
     `benchmarks/proposed_kernels.py`.
   - Vectorise `DatetimeAccumulator.update`. Four per-row Python loops,
     4.7 us/value, the most expensive column kind in the library.
   - Drop `format="mixed"` (`inference.py:384`, `consume.py:162`) for
     try-explicit-formats-first on a 200-row sample.
   - Vectorised Algorithm L (also in `proposed_kernels.py`) — **4.9×**, and
     bit-identical, which the naive Algorithm R rewrite is not.
   - `np.diff` for `MonotonicityDetector.update` — 61×.
2. **Then publish the benchmarks.** 9.1× vs ydata-profiling and 13× less
   marginal memory, both currently unmentioned in the README.
3. **Then the native core — KMV first, moments last.** The measured
   decomposition says moments are **1.3%** of the numeric path and KMV is
   **50%**; the original ordering had this backwards.

Measurement discipline, learned the hard way on this codebase: `cProfile` charges
per Python call and badly over-weights kernels that make many small ones. It
ranked the reservoir at ~30% of self time when swapping in a 5×-faster one moved
wall clock by 4%. Confirm any hot-spot ranking against wall clock with the
profiler off before acting on it.

Five items from the v1 audit are still open — first-chunk type decisions, the
2,000-row `RowKMV` fallback cap, chunk-local extreme indices, the dead
`corr_max_cols` option, and the pre-1906 datetime window. All six items that had
a failing test got fixed; all five without one did not. Write the test first.
