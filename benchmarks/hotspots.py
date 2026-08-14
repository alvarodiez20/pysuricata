"""Where does profile() actually spend its time?

Run:
    python -m benchmarks.hotspots                       # mixed suite, 200k rows
    python -m benchmarks.hotspots --suite numeric_wide --scale 2
    python -m benchmarks.hotspots --dump prof.out       # for snakeviz

Output has three sections:

* **By function** — the standard cProfile top-N by cumulative self time.
* **By subsystem** — self time rolled up into consume / accumulate / render /
  infer / correlate, so you can see whether the next hour is better spent on
  the compute path or the render path. It is easy to spend a week optimising
  accumulators when 40% of the wall clock is in HTML generation.
* **Flagged** — self time attributed to the specific functions identified in
  the audit, with what each one is doing wrong. This is the list that shrinks
  as the roadmap lands.

cProfile adds per-call overhead, so it *over*-weights functions called many
times with little work each — which is exactly the shape of the per-row Python
loops here. Treat the ordering as reliable and the absolute numbers as
directional; ``kernels.py`` gives untainted timings.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict

# Functions the audit flagged, with a one-line reason. Keyed by the substring
# that identifies them in a pstats entry.
FLAGGED = {
    "_update_vectorized": "3 full-length temporaries (d*d*d*d) + a wrong M3/M4 batch merge",
    "add_many": "SHA-1 per value, or extend-then-sort of a Python list of ints",
    "_batch_add_hashes": "list.extend + list.sort on every chunk",
    "MonotonicityDetector.update": "Python for-loop over every value; np.diff does this in one pass",
    "_add_to_min_heap": "O(k) max() scan + heapify per insert; heapq has a one-line idiom for this",
    "_add_to_max_heap": "same as _add_to_min_heap",
    "_to_datetime_ns_array_pandas": ".tolist() then a per-value NaT check",
    "_to_bool_array_pandas": "builds a Python list of bool|None, then loops over the indices",
    "_to_categorical_iter_pandas": "Series.tolist() materialises one PyObject per row",
    "_create_valid_mask": "per-value isinstance over an object array",
    "fromtimestamp": "one Python datetime object per row",
    "hash_pandas_object": "fine on its own — but its u64 output is then stringified and SHA-1'd",
    "missing_cells": "full isnull().sum().sum() pass per chunk, duplicating work the accumulators already did",
    "update_corr": "O(p^2) Python loop with ~10 array passes per pair",
    "memory_usage": "deep=True walks every Python string object",
    "value_counts": "full groupby per categorical column per chunk",
    "to_numeric": "coercion path; the fast dtype check should have caught this",
    "_u64": "SHA-1 per value — a cryptographic hash used for a distinct-count sketch",
    "get_token": "dateutil is parsing dates one row at a time in Python",
    "_strptime": "same: format='mixed' disables pandas' vectorised date parsing",
    "_identify_outliers": "render-time outlier pass over the full reservoir",
}

SUBSYSTEMS = [
    # Order matters: first match wins. Hashing and date parsing are pulled out
    # of the generic buckets because they are the two costs that hide inside
    # "other" and dominate real workloads.
    ("hashing", ("hashlib", "_hashlib", "sketches.py:11", "_u64", "sha1")),
    ("date parsing", ("dateutil", "_strptime", "strptime", "parser/_parser")),
    ("render", ("/render/", "report.py", "_card", "svg", "html")),
    ("accumulate", ("/accumulators/",)),
    ("consume", ("consume", "/adapters/", "conversion")),
    ("infer", ("inference",)),
    ("correlate", ("correlation",)),
    ("chunk", ("chunking",)),
    ("sort", ("builtins.sorted", "'sort' of 'list'")),
    (
        "pandas/numpy",
        (
            "site-packages/pandas",
            "site-packages/numpy",
            "dist-packages/pandas",
            "dist-packages/numpy",
        ),
    ),
]


def classify(filename: str, funcname: str) -> str:
    hay = f"{filename}::{funcname}"
    for label, needles in SUBSYSTEMS:
        if any(n in hay for n in needles):
            return label
    if "site-packages" in filename:
        return "other libs"
    if filename.startswith("<"):
        return "builtin"
    return "other"


def run(suite: str, scale: float, dump: str | None, top: int) -> int:
    try:
        from pysuricata import profile as ps_profile
    except ImportError:
        print("pysuricata is not installed in this environment.", file=sys.stderr)
        return 2

    from . import datasets

    df = datasets.build(suite, scale=scale)
    meta = datasets.describe(df)
    print(
        f"suite={suite}  rows={meta['rows']:,}  cols={meta['cols']}  {meta['bytes'] / 1e6:.0f} MB"
    )
    print(f"dtypes: {meta['dtypes']}\n")

    # Wall clock without the profiler, so the overhead is visible.
    t0 = time.perf_counter()
    ps_profile(df)
    wall = time.perf_counter() - t0

    pr = cProfile.Profile()
    pr.enable()
    ps_profile(df)
    pr.disable()

    if dump:
        pr.dump_stats(dump)
        print(f"wrote {dump}  (snakeviz {dump})")

    st = pstats.Stats(pr)
    profiled_total = st.total_tt
    print(f"wall clock (unprofiled): {wall:.3f}s")
    print(
        f"wall clock (profiled):   {profiled_total:.3f}s  "
        f"({profiled_total / wall:.1f}x cProfile overhead)\n"
    )

    # --- by function -------------------------------------------------------
    buf = io.StringIO()
    st.stream = buf
    st.sort_stats("tottime").print_stats(top)
    print("BY FUNCTION (self time)")
    print("-" * 72)
    for line in buf.getvalue().splitlines():
        if line.strip() and not line.startswith(
            ("   Ordered", "   List", "Wed", "Thu")
        ):
            print(line)

    # --- by subsystem ------------------------------------------------------
    rollup: dict[str, float] = defaultdict(float)
    for (fname, _lineno, func), (_cc, _nc, tt, _ct, _cal) in st.stats.items():
        rollup[classify(fname, func)] += tt
    print("\nBY SUBSYSTEM (self time)")
    print("-" * 72)
    for label, tt in sorted(rollup.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(50 * tt / max(profiled_total, 1e-9))
        print(f"{label:<14}{tt:>8.3f}s {100 * tt / profiled_total:>6.1f}%  {bar}")

    # --- flagged -----------------------------------------------------------
    hits: list[tuple[float, str, str, str]] = []
    for (fname, lineno, func), (_cc, nc, tt, _ct, _cal) in st.stats.items():
        for needle, why in FLAGGED.items():
            if needle in func or needle in f"{fname}::{func}":
                short = fname.split("/")[-1]
                hits.append((tt, f"{short}:{lineno} {func}", f"{nc:,} calls", why))
                break
    if hits:
        hits.sort(reverse=True)
        flagged_total = sum(h[0] for h in hits)
        print(
            f"\nFLAGGED BY AUDIT ({flagged_total:.3f}s self time, "
            f"{100 * flagged_total / profiled_total:.1f}% of profiled total)"
        )
        print("-" * 72)
        for tt, where, calls, why in hits[:20]:
            print(f"{tt:>8.3f}s  {where}")
            print(f"{'':>10}{calls:<16}{why}")
    else:
        print("\nNo flagged functions appeared in the profile for this suite.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite", default="mixed", help="one of benchmarks.datasets.SUITES"
    )
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dump", default=None, help="write raw pstats here for snakeviz")
    args = ap.parse_args(argv)
    return run(args.suite, args.scale, args.dump, args.top)


if __name__ == "__main__":
    sys.exit(main())
