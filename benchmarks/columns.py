"""How does cost scale with the number of *columns*?

Every other benchmark here scales rows. That is the axis the streaming design
was built for, and it is the axis the memory claim holds on. Holding rows fixed
and widening the frame produces the opposite result, and until this script
existed nothing measured it (#207).

    python -m benchmarks.columns                  # the default sweep
    python -m benchmarks.columns --rows 5000      # faster, same shape of answer
    python -m benchmarks.columns --json out.json  # machine-readable
    python -m benchmarks.columns --budget 512     # exit 1 if RSS crosses it

What it reports, per shape:

* **Marginal RSS** — peak resident memory during ``profile()`` minus resident
  memory with the frame already built. The frame is the caller's; this is what
  profiling it costs on top.
* **Python peak** — ``tracemalloc``'s high-water mark, which counts only Python
  allocations. Reported beside RSS because the gap between them is the NumPy
  and interpreter arena that RSS includes and ``tracemalloc`` cannot see.
* **Report bytes** — the emitted HTML. A column card is in the document whether
  or not anyone scrolls to it, so this scales on the same axis.

The row-scaled control at the top is the point of comparison, and it is what
makes the result legible: it has *more cells* than the widest frame below it.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tracemalloc
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

#: Shapes to sweep. The first is the row-scaled control -- 14 million cells,
#: the shape every other benchmark in this directory measures.
DEFAULT_SHAPES: tuple[tuple[int, int], ...] = (
    (1_000_000, 14),
    (20_000, 14),
    (20_000, 50),
    (20_000, 100),
    (20_000, 200),
    (20_000, 400),
    (20_000, 600),
)


@dataclass
class Measurement:
    rows: int
    cols: int
    cells: int
    marginal_rss_mb: float
    python_peak_mb: float
    report_kb: float
    seconds: float


def _rss_mb() -> float:
    """Resident set size, or 0.0 where psutil is not installed.

    psutil is an extra (`pysuricata[system]`), not a runtime dependency (#204),
    so this degrades to the tracemalloc column rather than refusing to run.
    """
    try:
        import psutil
    except ImportError:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _frame(rows: int, cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{i}": rng.normal(size=rows) for i in range(cols)})


def measure(rows: int, cols: int) -> Measurement:
    import time

    import pysuricata as ps

    gc.collect()
    frame = _frame(rows, cols)
    gc.collect()
    before = _rss_mb()

    tracemalloc.start()
    started = time.perf_counter()
    report = ps.profile(frame, seed=0)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    after = _rss_mb()
    size_kb = len(report.html) / 1024

    result = Measurement(
        rows=rows,
        cols=cols,
        cells=rows * cols,
        marginal_rss_mb=round(after - before, 1),
        python_peak_mb=round(peak / 1024 / 1024, 1),
        report_kb=round(size_kb, 1),
        seconds=round(elapsed, 2),
    )
    del report, frame
    gc.collect()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=20_000, help="rows in wide frames")
    ap.add_argument(
        "--cols",
        type=int,
        nargs="*",
        default=None,
        help="column counts to sweep (default 14 50 100 200 400 600)",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="exit 1 if any shape's marginal RSS exceeds this many MB",
    )
    ap.add_argument(
        "--no-control",
        action="store_true",
        help="skip the row-scaled comparison at the top",
    )
    args = ap.parse_args(argv)

    if args.cols is None and args.rows == 20_000:
        shapes = list(DEFAULT_SHAPES)
    else:
        cols = args.cols or [14, 50, 100, 200, 400, 600]
        shapes = [(1_000_000, 14)] + [(args.rows, c) for c in cols]
    if args.no_control:
        shapes = [s for s in shapes if s[1] != 14 or s[0] == args.rows]

    print(
        f"{'shape':>18} {'cells':>12} {'marginal RSS':>14} "
        f"{'py peak':>10} {'report':>12} {'time':>8}"
    )
    print("-" * 80)

    results: list[Measurement] = []
    for rows, cols in shapes:
        m = measure(rows, cols)
        results.append(m)
        print(
            f"{m.rows:>11,} x {m.cols:<4} {m.cells:>12,} "
            f"{m.marginal_rss_mb:>11,.0f} MB {m.python_peak_mb:>7,.0f} MB "
            f"{m.report_kb:>9,.0f} KB {m.seconds:>7.2f}s"
        )

    wide = [m for m in results if m.cols >= 100]
    if len(wide) >= 2:
        first, last = wide[0], wide[-1]
        span = last.cols - first.cols
        print(
            f"\nmarginal cost per column: "
            f"{(last.marginal_rss_mb - first.marginal_rss_mb) / span:.2f} MB RSS, "
            f"{(last.report_kb - first.report_kb) / span:.1f} KB report"
        )

    control = next((m for m in results if m.rows > args.rows), None)
    widest = max(results, key=lambda m: m.cols)
    if control is not None and widest.cols > control.cols:
        print(
            f"\n{control.rows:,} x {control.cols} holds "
            f"{control.cells / widest.cells:.1f}x the cells of "
            f"{widest.rows:,} x {widest.cols} and costs "
            f"{widest.marginal_rss_mb / max(control.marginal_rss_mb, 1e-9):.0f}x "
            f"less memory. Bounded memory is a claim about rows (#207)."
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump([asdict(m) for m in results], handle, indent=2)
        print(f"\nwrote {args.json}")

    if args.budget is not None:
        over = [m for m in results if m.marginal_rss_mb > args.budget]
        if over:
            print(
                f"\nOVER BUDGET ({args.budget:,.0f} MB): "
                + ", ".join(f"{m.rows:,}x{m.cols}" for m in over),
                file=sys.stderr,
            )
            return 1
        print(f"\nevery shape inside the {args.budget:,.0f} MB budget")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
