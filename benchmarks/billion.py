#!/usr/bin/env python3
"""One billion rows of mixed dtypes under a kernel-enforced memory ceiling.

The claim this file exists to test is not that pysuricata is fast. At the
shape below it profiles roughly 200k rows a second, which wins no races. The
claim is that **the peak does not move**: a single pass whose working set is
set by the column count and the chunk size, and not by the row count, so the
same 4 GB laptop that profiles a million rows profiles a billion.

That is a memory claim, so it is measured as one -- against a ceiling the
kernel enforces, and sampled all the way through rather than read once at the
end.

    python -m benchmarks.billion --rows 1_000_000_000         # the headline
    python -m benchmarks.billion --rows 10_000_000            # a smoke run
    python -m benchmarks.billion --rows 200_000_000 --mode load  # the contrast

## What is measured, and against what

`--budget-mb` (default 4096) is imposed on the *child* process that does the
profiling, by the same two mechanisms `memory_bounded_check.py` uses and for
the same reason -- see its module docstring for why a cgroup beats an
after-the-fact reading:

1. a child cgroup (v1 `memory.limit_in_bytes`), which is what `docker
   --memory` itself rests on. `memory.max_usage_in_bytes` afterwards is the
   peak the kernel recorded for the group.
2. `RLIMIT_AS`, if the cgroup filesystem is not writable.

Both peaks are reported, and **RSS is the one to quote**. The cgroup's
`memory.max_usage_in_bytes` reads *lower* than RSS here -- 140 MB against
222 MB on a 2M-row run -- because a page is charged to whichever cgroup first
touched it, and the interpreter, numpy and pandas were already resident in the
parent's group before the child was ever created. The kernel is not
under-reporting the ceiling it enforces; it is reporting the pages this run is
responsible for having brought in. RSS counts the shared ones too, so it is the
conservative number and the one a reader should be given.

The ceiling is not decorative, and that is checked rather than assumed: a run
given 300 MB and a 4M-row chunk is killed by signal 9 at exactly 300 MB.
Either way the run is killed if it exceeds the ceiling. A row in the output
CSV is therefore evidence and not an assertion: the process that wrote it was
alive, and it could only be alive by being under budget.

The child samples its own `VmRSS` from `/proc/self/status` on a background
thread, next to a counter the source generator bumps as it yields chunks, so
every sample is a `(seconds, rows_processed, rss_mb)` triple and the CSV is a
curve rather than a number. It is flushed on every sample: this run takes
hours, and a container that is reclaimed at 700M rows should still leave
behind the graph up to 700M rows.

## Where the rows come from

Synthesised in-process, a chunk at a time, never materialised as a frame --
1B rows of this shape is ~80 GB in any file format worth writing, which is
not a thing you keep on a CI runner. The cost of that choice is stated rather
than hidden: **the generator's own memory is inside the measurement**, since
it runs in the same process under the same ceiling, so the reported peak is
higher than a pure read path would be, not lower.

The distributions are deliberately awkward -- lognormal tails, a Zipf head on
the high-cardinality column, seasonal timestamps, injected nulls, outliers and
duplicate rows. A uniform generator makes for a boring report and flatters
every sketch in it.

## The contrast curve

`--mode load` runs the same generator into a list and concatenates it, which is
what any profiler taking a frame rather than a stream must do before it can
start. It is *expected* to be killed by the ceiling; the row count it reaches
before it dies is the result, and it is measured in the same session, with the
same generator and the same ceiling, so the two curves in the graph are
comparable. Pairing a streaming number with a loading number taken from a
different run is how this project has twice published a ratio that was wrong.

## Reading the report it produces

At this row count several columns of the report are **estimates from
sketches**: distinct counts (KMV), top-k (Misra-Gries), quantiles, and the
sample. They carry their error bounds in the report and they are not exact
integers. This is a property of the design, not of the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: Rows per chunk handed to the engine. The ADR's model (docs/adr/memory-
#: budget.md) puts this at ~48 B per row per column of resident memory, so it
#: is the knob that trades throughput against the ceiling.
DEFAULT_CHUNK = 200_000

#: One "block" of columns: the unit `--cols` is rounded to. Ten columns
#: covering seven dtypes, in roughly the proportion an analytics table has
#: them -- numeric-heavy, a couple of strings, one timestamp, one boolean.
BLOCK = 10


def _cgroup_dir() -> str | None:
    """A writable cgroup v1 memory directory for *this* process, if any."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(":")
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if "memory" not in controllers.split(","):
            continue
        base = f"/sys/fs/cgroup/memory{path}"
        if os.path.isdir(base) and os.access(base, os.W_OK):
            return base
    return None


def read_rss_mb() -> float:
    """Resident set size of this process, in MB, from /proc.

    Deliberately not psutil: psutil is an optional extra of this project (it
    publishes no WASM wheel, see pyproject.toml), and a benchmark that only
    runs when an extra is installed is a benchmark that stops being run.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


# --------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------
#
# Pools are built once per process and indexed per chunk. Building the object
# arrays is the expensive half of generation, and rebuilding them 5,000 times
# would put the generator, not the profiler, on the critical path.

_SKU_POOL = 50_000
_TEXT_POOL = 800
_COUNTRIES = [
    "ES",
    "FR",
    "DE",
    "IT",
    "PT",
    "US",
    "MX",
    "BR",
    "JP",
    "IN",
    "GB",
    "NL",
    "SE",
    "PL",
    "AR",
    "CL",
    "CA",
    "AU",
    "ZA",
    "KR",
]


class Pools:
    """Immutable string pools, shared by every chunk."""

    def __init__(self) -> None:
        import numpy as np

        self.sku = np.array([f"SKU-{i:07d}" for i in range(_SKU_POOL)], dtype=object)
        self.text = np.array(
            [
                f"customer note {i}: " + "detail " * (1 + i % 9)
                for i in range(_TEXT_POOL)
            ],
            dtype=object,
        )
        self.country = np.array(_COUNTRIES, dtype=object)


def column_names(cols: int) -> list[str]:
    """The column names `make_chunk` produces for a given width."""
    blocks = max(1, round(cols / BLOCK))
    names: list[str] = []
    for b in range(blocks):
        names += [
            f"amount_{b}",
            f"score_{b}",
            f"ratio_{b}",
            f"qty_{b}",
            f"user_id_{b}",
            f"sku_{b}",
            f"country_{b}",
            f"free_text_{b}",
            f"event_at_{b}",
            f"is_active_{b}",
        ]
    return names


def make_chunk(g, n: int, pools: Pools, blocks: int, null_frac: float = 0.04):
    """One chunk: `blocks` x 10 columns spanning seven dtypes.

    The distributions are chosen to be awkward on purpose. A lognormal has a
    tail the mean does not describe; a power-law index gives the string column
    a Zipf head instead of the flat cardinality `choice` would give it; the
    timestamps carry a diurnal hump. Nulls, outliers and duplicate rows are
    injected because a report with no quality problems exercises none of the
    code that finds them.
    """
    import numpy as np
    import pandas as pd

    data: dict[str, object] = {}

    # Whole-row duplicates: the same source/destination index pair applied to
    # every column, so the duplicate is a duplicate *row* and the report's
    # duplicate detection has something real to find.
    n_dup = max(1, int(n * 0.005))
    dup_dst = g.integers(0, n, n_dup)
    dup_src = g.integers(0, n, n_dup)

    def finish(arr):
        arr[dup_dst] = arr[dup_src]
        return arr

    for b in range(blocks):
        # float64, heavy right tail, with genuine outliers an order of
        # magnitude past the tail so the extreme tracker has work to do.
        amount = g.lognormal(3.0, 1.1, n)
        outliers = g.integers(0, n, max(1, n // 20_000))
        amount[outliers] *= 500.0
        data[f"amount_{b}"] = finish(amount)

        # float64, bimodal, with nulls.
        score = np.where(
            g.random(n) < 0.65, g.normal(72.0, 8.0, n), g.normal(38.0, 12.0, n)
        )
        score[g.random(n) < null_frac] = np.nan
        data[f"score_{b}"] = finish(score)

        # float32 -- a second float width, so the report is not all float64.
        data[f"ratio_{b}"] = finish(g.random(n).astype(np.float32))

        # int64 counts.
        data[f"qty_{b}"] = finish(g.poisson(3.4, n).astype(np.int64))

        # int64, high cardinality, skewed -- an id column, the shape that
        # makes a distinct-count sketch earn its place.
        data[f"user_id_{b}"] = finish(
            (2_000_000 * g.power(0.4, n)).astype(np.int64) + 1
        )

        # object, high cardinality, Zipf head.
        sku_idx = (_SKU_POOL * g.power(0.35, n)).astype(np.int64)
        data[f"sku_{b}"] = finish(pools.sku[sku_idx])

        # category -- a dtype of its own, with categories pinned across chunks
        # so they do not drift chunk to chunk.
        cty_idx = (len(_COUNTRIES) * g.power(0.7, n)).astype(np.int64)
        data[f"country_{b}"] = pd.Categorical(
            finish(pools.country[cty_idx]), categories=_COUNTRIES
        )

        # object, free text, with nulls.
        text = pools.text[g.integers(0, _TEXT_POOL, n)]
        text[g.random(n) < null_frac] = None
        data[f"free_text_{b}"] = finish(text)

        # datetime64[ns], two years, with a diurnal hump rather than a flat
        # spread across the day.
        day = g.integers(0, 730, n)
        hour = (g.random(n) + g.random(n)) / 2.0  # triangular: a working day
        secs = day * 86_400 + (hour * 86_400).astype(np.int64)
        data[f"event_at_{b}"] = pd.to_datetime(
            finish(np.datetime64("2023-01-01", "s").astype("int64") + secs),
            unit="s",
        )

        # bool. Left as a true bool rather than a nullable one: pandas spells
        # a nullable boolean as object, and an object column of True/False
        # would be profiled as categorical, which is not what is being shown.
        data[f"is_active_{b}"] = finish(g.random(n) > 0.28)

    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# The child: profiles under the ceiling and samples itself while it does
# --------------------------------------------------------------------------


class Progress:
    """Shared state between the source generator and the sampler thread.

    Plain attribute reads and writes, deliberately unlocked: the sampler only
    ever reads, a torn read costs one slightly-stale row in a CSV of
    thousands, and a lock on the chunk boundary would show up in the timing
    this file exists to report.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.chunks = 0
        self.phase = "profile"
        self.done = False


def counting_source(
    g, pools: Pools, blocks: int, rows: int, chunk: int, progress: Progress
):
    """Yield `rows` rows as chunks, bumping `progress` as it goes.

    The generator is the source, so nothing larger than one chunk is ever
    resident -- and the cost of building it is inside the measured process,
    which makes the reported peak conservative rather than flattering.
    """
    left = rows
    while left > 0:
        n = min(chunk, left)
        df = make_chunk(g, n, pools, blocks)
        left -= n
        progress.rows += n
        progress.chunks += 1
        yield df


def _sampler(
    path: str, progress: Progress, interval: float, started: float, budget_mb: float
) -> None:
    """Append `(seconds, rows, rss)` to `path` until `progress.done`.

    Flushed and fsync-free but line-buffered on every sample: this run takes
    hours in a container that can be reclaimed, and a partial curve is still a
    curve. The peak is tracked here too so a run that is killed by the kernel
    still leaves its last reading behind.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "rows", "chunks", "rss_mb", "budget_mb", "phase"])
        while not progress.done:
            w.writerow(
                [
                    round(time.perf_counter() - started, 3),
                    progress.rows,
                    progress.chunks,
                    round(read_rss_mb(), 2),
                    budget_mb,
                    progress.phase,
                ]
            )
            f.flush()
            time.sleep(interval)
        w.writerow(
            [
                round(time.perf_counter() - started, 3),
                progress.rows,
                progress.chunks,
                round(read_rss_mb(), 2),
                budget_mb,
                "final",
            ]
        )
        f.flush()


def child_main(args) -> int:
    """Run the profile in this process, under whatever ceiling was applied."""
    import resource

    import numpy as np

    if args.mechanism == "rlimit":
        b = int(args.budget_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))

    from pysuricata import profile, summarize

    blocks = max(1, round(args.cols / BLOCK))
    pools = Pools()
    g = np.random.default_rng(args.seed)
    progress = Progress()
    started = time.perf_counter()

    t = threading.Thread(
        target=_sampler,
        args=(args.csv, progress, args.sample_interval, started, args.budget_mb),
        daemon=True,
    )
    t.start()

    src = counting_source(g, pools, blocks, args.rows, args.chunk_size, progress)
    kwargs = {"chunk_size": args.chunk_size}

    if args.mode == "load":
        # The contrast curve. Every profiler that takes a frame rather than a
        # stream needs the frame first, so this is the floor its memory cannot
        # go below -- measured with the same generator, the same ceiling and
        # the same sampler, in the same session, because a ratio assembled from
        # two separate runs is how this project has published wrong numbers
        # before. It is *expected* to be killed; where it dies is the result.
        import pandas as pd

        held = []
        for df in src:
            held.append(df)
        frame = pd.concat(held, ignore_index=True)
        del held
        progress.phase = "profile-in-memory"
        stats = summarize(frame, **kwargs)
        elapsed = time.perf_counter() - started
        progress.done = True
        t.join(timeout=args.sample_interval * 3)
        with open(os.environ["RESULT_OUT"], "w") as f:
            json.dump(
                {
                    "rows": progress.rows,
                    "chunks": progress.chunks,
                    "cols": blocks * BLOCK,
                    "seconds": round(elapsed, 2),
                    "rows_per_s": round(progress.rows / elapsed, 1),
                    "peak_rss_mb": round(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
                    ),
                },
                f,
            )
        return 0

    if args.html:
        result = profile(src, **kwargs)
        progress.phase = "render"
        result.save_html(args.html)
        stats = result.stats
    else:
        stats = summarize(src, **kwargs)

    elapsed = time.perf_counter() - started
    progress.done = True
    t.join(timeout=args.sample_interval * 3)

    if args.summary_json:
        from pysuricata.api import _convert_numpy_types

        with open(args.summary_json, "w") as f:
            json.dump(_convert_numpy_types(stats), f, indent=2, ensure_ascii=False)

    with open(os.environ["RESULT_OUT"], "w") as f:
        json.dump(
            {
                "rows": progress.rows,
                "chunks": progress.chunks,
                "cols": blocks * BLOCK,
                "seconds": round(elapsed, 2),
                "rows_per_s": round(progress.rows / elapsed, 1),
                "peak_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
                ),
            },
            f,
        )
    return 0


# --------------------------------------------------------------------------
# The parent: applies the ceiling, then reports what the kernel recorded
# --------------------------------------------------------------------------


@dataclass
class Run:
    mechanism: str
    budget_mb: float
    rows: int
    cols: int
    chunk_size: int
    passed: bool
    peak_rss_mb: float | None
    cgroup_peak_mb: float | None
    seconds: float
    rows_per_s: float | None
    csv: str
    note: str = ""


def run_capped(args) -> Run:
    """Spawn the profiling child under a ceiling the kernel enforces."""
    cgroup_base = _cgroup_dir() if args.mechanism == "auto" else None
    if args.mechanism == "auto":
        mechanism = "cgroup" if cgroup_base is not None else "rlimit"
    else:
        mechanism = args.mechanism

    budget_bytes = int(args.budget_mb * 1024 * 1024)
    result_out = tempfile.mktemp(prefix="pysuricata-billion-")
    env = {**os.environ, "RESULT_OUT": result_out, "PYTHONPATH": REPO_ROOT}

    my_cgroup = None
    if mechanism == "cgroup":
        my_cgroup = os.path.join(cgroup_base, f"billion-{uuid.uuid4().hex[:8]}")
        os.mkdir(my_cgroup)
        with open(os.path.join(my_cgroup, "memory.limit_in_bytes"), "w") as f:
            f.write(str(budget_bytes))

    cmd = [
        sys.executable,
        "-m",
        "benchmarks.billion",
        "--_child",
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--chunk-size",
        str(args.chunk_size),
        "--budget-mb",
        str(args.budget_mb),
        "--sample-interval",
        str(args.sample_interval),
        "--seed",
        str(args.seed),
        "--csv",
        args.csv,
        "--mechanism",
        mechanism,
        "--mode",
        args.mode,
    ]
    if args.html:
        cmd += ["--html", args.html]
    if args.summary_json:
        cmd += ["--summary-json", args.summary_json]

    def _preexec():
        with open(os.path.join(my_cgroup, "cgroup.procs"), "w") as f:
            f.write(str(os.getpid()))

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=REPO_ROOT,
        preexec_fn=_preexec if my_cgroup else None,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    cgroup_peak = None
    if my_cgroup is not None:
        try:
            with open(os.path.join(my_cgroup, "memory.max_usage_in_bytes")) as f:
                cgroup_peak = int(f.read().strip()) / 1024 / 1024
        except OSError:
            pass
        finally:
            shutil.rmtree(my_cgroup, ignore_errors=True)

    child: dict = {}
    if os.path.exists(result_out):
        try:
            with open(result_out) as f:
                child = json.load(f)
        except (OSError, json.JSONDecodeError):
            child = {}
        finally:
            os.remove(result_out)

    note = ""
    if proc.returncode < 0:
        note = f"killed by signal {-proc.returncode} -- the kernel intervened"
    elif proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        note = f"exit {proc.returncode}: {' | '.join(tail)}"

    return Run(
        mechanism=mechanism,
        budget_mb=args.budget_mb,
        rows=child.get("rows", 0),
        cols=child.get("cols", max(1, round(args.cols / BLOCK)) * BLOCK),
        chunk_size=args.chunk_size,
        passed=proc.returncode == 0 and bool(child),
        peak_rss_mb=child.get("peak_rss_mb"),
        cgroup_peak_mb=round(cgroup_peak, 2) if cgroup_peak is not None else None,
        seconds=round(elapsed, 2),
        rows_per_s=child.get("rows_per_s"),
        csv=args.csv,
        note=note,
    )


def _print(run: Run) -> None:
    print(f"mechanism:   {run.mechanism}  (budget {run.budget_mb:,.0f} MB)")
    print(f"shape:       {run.rows:,} rows x {run.cols} cols, chunk {run.chunk_size:,}")
    print(f"outcome:     {'PASSED under budget' if run.passed else 'FAILED'}")
    if run.peak_rss_mb is not None:
        head = run.budget_mb / run.peak_rss_mb
        print(
            f"peak RSS:    {run.peak_rss_mb:,.0f} MB  ({head:.1f}x under the ceiling)"
        )
    if run.cgroup_peak_mb is not None:
        print(f"cgroup peak: {run.cgroup_peak_mb:,.0f} MB  (kernel's own accounting)")
    if run.rows_per_s:
        print(f"throughput:  {run.rows_per_s / 1e6:.3f} M rows/s")
    print(f"wall clock:  {run.seconds / 60:.1f} min")
    print(f"curve:       {run.csv}")
    if run.note:
        print(f"note:        {run.note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rows", type=int, default=1_000_000_000)
    ap.add_argument(
        "--cols", type=int, default=BLOCK, help=f"rounded to a multiple of {BLOCK}"
    )
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--budget-mb", type=float, default=4096.0)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="billion-curve.csv")
    ap.add_argument("--html", default=None, help="render the HTML report here")
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--json", default=None, help="write the run's result here")
    ap.add_argument(
        "--mode",
        default="stream",
        choices=["stream", "load"],
        help="stream: profile chunk by chunk. load: materialise the whole frame "
        "first, the way a load-then-profile tool must -- expected to be killed",
    )
    ap.add_argument(
        "--mechanism", default="auto", choices=["auto", "cgroup", "rlimit", "none"]
    )
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._child:
        return child_main(args)

    run = run_capped(args)
    _print(run)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(asdict(run), f, indent=2)
    return 0 if run.passed else 1


if __name__ == "__main__":
    sys.exit(main())
