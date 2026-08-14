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

See `docs/roadmap.md` for the full plan, the measured numbers behind it, and the
market analysis. Ordering, shortest version:

1. **Correctness first.** `benchmarks/accuracy.py` has six `xfail`-marked tests,
   each pointing at a live bug with a file and line. Fix them, delete the markers.
   Three are critical: generator sources drop chunk 0, reservoir sampling is
   biased, and the M3/M4 batch merge is wrong for multi-chunk data.
2. **Then pure-Python performance.** SHA-1 in the KMV sketch is ~36% of runtime
   and `format="mixed"` date sniffing is ~21%. Neither needs Rust.
3. **Then publish benchmarks**, then the native core.

Do not start on `native/` before 1 and 2 are done — the numbers only mean
something against a measured, already-clean baseline.
