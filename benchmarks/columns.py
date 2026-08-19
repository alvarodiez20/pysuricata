"""How does cost scale with the number of *columns*?

Every other benchmark here scales rows. That is the axis the streaming design
was built for, and it is the axis the memory claim holds on. Holding rows fixed
and widening the frame produces the opposite result, and until this script
existed nothing measured it (#207).

    python -m benchmarks.columns                  # the default sweep
    python -m benchmarks.columns --rows 5000      # faster, same shape of answer
    python -m benchmarks.columns --json out.json  # machine-readable
    python -m benchmarks.columns --budget 512     # exit 1 if peak crosses it

What it reports, per shape:

* **Marginal peak** — the process high-water mark during ``profile()`` minus
  the high-water mark with the frame already built. The frame is the caller's;
  this is what profiling it costs on top.
* **Peak** — the whole process at its high-water mark, which is the number a
  container ceiling is enforced against.
* **Report bytes** — the emitted HTML. A column card is in the document whether
  or not anyone scrolls to it, so this scales on the same axis.

The row-scaled control at the top is the point of comparison, and it is what
makes the result legible: it has *more cells* than the widest frame below it.

## Two things this used to get wrong, both by about 2x

**Each shape is measured in its own subprocess.** Peak memory is a high-water
mark: the allocator does not hand freed pages back, so a 600-column run
measured after a 400-column one in the same process reports only what it grew
*beyond* the earlier peak. That under-reported the widest shape -- the axis the
whole script exists to measure -- by half.

**Nothing measures under ``tracemalloc``.** It allocates a trace record per
allocation, so it inflates the very RSS being read, and it slowed the 600-column
run from 58s to 252s. ``--python-peak`` still asks for it, in its own run, for
when the split between Python and NumPy allocation is the question.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
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
    marginal_peak_mb: float
    peak_mb: float
    python_peak_mb: float
    report_kb: float
    seconds: float


def _peak_mb() -> float:
    """The process high-water mark, in MB.

    `ru_maxrss`, not a `psutil` reading of current RSS: the reading tells you
    where memory happens to sit when it is taken, and a peak that has already
    been freed is exactly what a container ceiling would have killed. Linux
    reports it in KB. Needs no dependency, which the previous psutil reading
    did (`pysuricata[system]`, an extra rather than a requirement -- #204).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _frame(rows: int, cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{i}": rng.normal(size=rows) for i in range(cols)})


def measure(rows: int, cols: int, *, python_peak: bool = False) -> Measurement:
    """Profile one shape and report what it cost. Call this in a fresh process.

    Running two shapes in one process reports the second one's growth *beyond*
    the first one's peak, not its cost -- see the module docstring. `main()`
    spawns a subprocess per shape; this function trusts that it is alone.
    """
    import time

    import pysuricata as ps

    gc.collect()
    frame = _frame(rows, cols)
    gc.collect()
    before = _peak_mb()

    if python_peak:
        tracemalloc.start()
    started = time.perf_counter()
    report = ps.profile(frame, seed=0)
    elapsed = time.perf_counter() - started
    if python_peak:
        _, traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        traced = 0

    after = _peak_mb()
    size_kb = len(report.html) / 1024

    result = Measurement(
        rows=rows,
        cols=cols,
        cells=rows * cols,
        marginal_peak_mb=round(after - before, 1),
        peak_mb=round(after, 1),
        python_peak_mb=round(traced / 1024 / 1024, 1),
        report_kb=round(size_kb, 1),
        seconds=round(elapsed, 2),
    )
    del report, frame
    gc.collect()
    return result


def _measure_in_subprocess(rows: int, cols: int, *, python_peak: bool) -> Measurement:
    """Run `measure` in a fresh interpreter and read back the one JSON line."""
    argv = [
        sys.executable,
        "-m",
        "benchmarks.columns",
        "--measure-one",
        str(rows),
        str(cols),
    ]
    if python_peak:
        argv.append("--python-peak")
    done = subprocess.run(argv, capture_output=True, text=True, check=True)
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("{")][-1]
    return Measurement(**json.loads(line))


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
        help="exit 1 if any shape's peak process memory exceeds this many MB",
    )
    ap.add_argument(
        "--no-control",
        action="store_true",
        help="skip the row-scaled comparison at the top",
    )
    ap.add_argument(
        "--python-peak",
        action="store_true",
        help="also report tracemalloc's peak. Costs 4x wall clock and inflates "
        "the memory readings beside it, so it is off by default.",
    )
    ap.add_argument(
        "--measure-one",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=None,
        help="internal: measure one shape and print it as JSON. This is how "
        "each shape gets a process of its own.",
    )
    args = ap.parse_args(argv)

    if args.measure_one is not None:
        rows, cols = args.measure_one
        print(json.dumps(asdict(measure(rows, cols, python_peak=args.python_peak))))
        return 0

    if args.cols is None and args.rows == 20_000:
        shapes = list(DEFAULT_SHAPES)
    else:
        cols = args.cols or [14, 50, 100, 200, 400, 600]
        shapes = [(1_000_000, 14)] + [(args.rows, c) for c in cols]
    if args.no_control:
        shapes = [s for s in shapes if s[1] != 14 or s[0] == args.rows]

    header = f"{'shape':>18} {'cells':>12} {'marginal':>11} {'peak':>10}"
    if args.python_peak:
        header += f" {'py peak':>10}"
    print(header + f" {'report':>12} {'time':>8}")
    print("-" * (92 if args.python_peak else 81))

    results: list[Measurement] = []
    for rows, cols in shapes:
        m = _measure_in_subprocess(rows, cols, python_peak=args.python_peak)
        results.append(m)
        row = (
            f"{m.rows:>11,} x {m.cols:<4} {m.cells:>12,} "
            f"{m.marginal_peak_mb:>8,.0f} MB {m.peak_mb:>7,.0f} MB"
        )
        if args.python_peak:
            row += f" {m.python_peak_mb:>7,.0f} MB"
        print(row + f" {m.report_kb:>9,.0f} KB {m.seconds:>7.2f}s")

    wide = [m for m in results if m.cols >= 100]
    if len(wide) >= 2:
        first, last = wide[0], wide[-1]
        span = last.cols - first.cols
        print(
            f"\nmarginal cost per column: "
            f"{(last.marginal_peak_mb - first.marginal_peak_mb) / span * 1024:.0f} KB, "
            f"{(last.report_kb - first.report_kb) / span:.1f} KB report"
        )

    control = next((m for m in results if m.rows > args.rows), None)
    widest = max(results, key=lambda m: m.cols)
    if control is not None and widest.cols > control.cols:
        # Peaks, not a ratio of marginals. The row-scaled control's marginal
        # is 0 MB -- profiling a million rows never exceeds what holding the
        # frame already cost, which is the claim -- and dividing by that
        # produced a headline in the hundreds of billions.
        print(
            f"\n{control.rows:,} x {control.cols} holds "
            f"{control.cells / widest.cells:.1f}x the cells of "
            f"{widest.rows:,} x {widest.cols} and peaks at "
            f"{control.peak_mb:,.0f} MB against {widest.peak_mb:,.0f} MB, of which "
            f"{control.marginal_peak_mb:,.0f} MB against "
            f"{widest.marginal_peak_mb:,.0f} MB is the profiling itself. "
            f"Bounded memory is a claim about rows (#207)."
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump([asdict(m) for m in results], handle, indent=2)
        print(f"\nwrote {args.json}")

    if args.budget is not None:
        over = [m for m in results if m.peak_mb > args.budget]
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
