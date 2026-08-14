"""Per-kernel microbenchmarks: pure Python/NumPy vs the native extension.

Run:

    python -m benchmarks.kernels                 # everything, default size
    python -m benchmarks.kernels --rows 5000000  # bigger
    python -m benchmarks.kernels --json out.json # machine-readable

Two things this measures that a naive timing script does not:

1. **A memory-bandwidth roofline.** Before anything else it measures how fast
   this machine can simply stream a float64 array (a bare ``np.sum``). Every
   kernel is then reported as a fraction of that. A kernel at 5% of roofline
   is compute-bound and worth optimising; one at 85% is bandwidth-bound and
   the only remaining win is making fewer passes. This is the number that
   tells you whether to reach for better instructions or for a better data
   layout (CS:APP ch. 5 vs ch. 6).

2. **Whether releasing the GIL actually buys anything.** Each native kernel is
   run once single-threaded and once across N threads on disjoint arrays. If
   the extension holds the GIL, the threaded time is the sum; if it releases
   it, the threaded time is close to the max.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

try:
    import pysuricata_core as native
except ImportError:  # pragma: no cover
    native = None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    name: str
    rows: int
    best_s: float
    median_s: float
    reps: int
    ns_per_row: float = 0.0
    m_rows_per_s: float = 0.0
    pct_of_roofline: float = float("nan")
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ns_per_row = self.best_s / max(1, self.rows) * 1e9
        self.m_rows_per_s = (
            self.rows / self.best_s / 1e6 if self.best_s > 0 else float("inf")
        )


def bench(fn, rows: int, name: str, reps: int = 5, warmup: int = 1) -> Timing:
    """Time ``fn`` with GC disabled, taking the best of ``reps``.

    Best-of, not mean: we want the machine's capability, not the average of
    whatever else the OS was doing. The median is recorded too so a noisy run
    is visible rather than hidden.
    """
    for _ in range(warmup):
        fn()
    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return Timing(
        name=name,
        rows=rows,
        best_s=min(samples),
        median_s=statistics.median(samples),
        reps=reps,
    )


def measure_roofline(rows: int) -> float:
    """Bytes/second this machine sustains streaming a float64 array."""
    arr = np.random.default_rng(0).standard_normal(rows)
    t = bench(lambda: float(arr.sum()), rows, "roofline:np.sum", reps=7)
    return arr.nbytes / t.best_s


def _apply_roofline(t: Timing, roofline_bps: float, bytes_touched: int) -> Timing:
    if roofline_bps > 0 and bytes_touched > 0:
        achieved = bytes_touched / t.best_s
        t.pct_of_roofline = 100.0 * achieved / roofline_bps
    return t


# ---------------------------------------------------------------------------
# Pure-Python reference implementations
#
# These are transcriptions of the shipped PySuricata code, kept here so the
# benchmark runs standalone and so a change in the library shows up as a diff
# against a fixed baseline rather than silently moving the goalposts.
# ---------------------------------------------------------------------------


def py_sha1_hash_batch(values: np.ndarray) -> np.ndarray:
    """What ``KMV.add_many`` does today: str() then SHA-1, per value."""
    import hashlib

    def u64(b: bytes) -> int:
        return int.from_bytes(hashlib.sha1(b).digest()[:8], "big", signed=False)

    return np.array(
        [u64(str(v).encode("utf-8", "ignore")) for v in values], dtype=np.uint64
    )


def py_kmv_add_many(state: list[int], k: int, hashes: np.ndarray) -> list[int]:
    """The extend-then-sort strategy from ``KMV._batch_add_hashes``."""
    if len(state) < k:
        needed = min(k - len(state), len(hashes))
        state.extend(int(h) for h in hashes[:needed])
        hashes = hashes[needed:]
        if len(state) == k:
            state.sort()
    if len(hashes) > 0 and state:
        max_hash = state[-1]
        cand = hashes[hashes < max_hash]
        if len(cand) > 0:
            state.extend(int(h) for h in cand)
            state.sort()
            if len(state) > k:
                del state[k:]
    return state


def py_moments_update(arr: np.ndarray) -> tuple[float, float, float, float]:
    """``StreamingMoments._update_vectorized``: five full passes, three temporaries."""
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    mean = finite.sum() / n
    d = finite - mean
    m2 = np.sum(d * d)
    m3 = np.sum(d * d * d)
    m4 = np.sum(d * d * d * d)
    return float(mean), float(m2), float(m3), float(m4)


def py_reservoir_add_many(
    buf: list, seen: int, k: int, arr: np.ndarray
) -> tuple[list, int]:
    """``ReservoirSampler.add_many``, including its batch-uniform draw."""
    if len(buf) < k:
        needed = min(k - len(buf), len(arr))
        buf.extend(arr[:needed])
        arr = arr[needed:]
        seen += needed
    if len(arr) > 0:
        r = np.random.randint(1, seen + len(arr) + 1, size=len(arr))
        mask = r <= k
        idxs = r[mask] - 1
        vals = arr[mask]
        for i, v in zip(idxs, vals, strict=False):
            buf[i] = v
        seen += len(arr)
    return buf, seen


def py_monotonic_update(arr: np.ndarray) -> tuple[bool, bool]:
    """``MonotonicityDetector.update``: a Python for-loop over every value."""
    finite = arr[np.isfinite(arr)]
    inc = dec = True
    last = None
    for v in finite:
        if last is not None:
            if v < last:
                inc = False
            if v > last:
                dec = False
        last = v
    return inc, dec


def py_extreme_update(arr: np.ndarray, k: int = 5) -> tuple[list, list]:
    """``ExtremeTracker._add_to_min_heap``: O(k) scan + heapify per insert."""
    import heapq

    finite = arr[np.isfinite(arr)]
    min_heap: list = []
    if len(finite) > 2 * k:
        cand = np.argpartition(finite, k)[:k]
    else:
        cand = range(len(finite))
    for i in cand:
        value = float(finite[i])
        if len(min_heap) < k:
            heapq.heappush(min_heap, (value, int(i)))
        else:
            largest = max(item[0] for item in min_heap)
            if value < largest:
                for j, item in enumerate(min_heap):
                    if item[0] == largest:
                        min_heap[j] = (value, int(i))
                        heapq.heapify(min_heap)
                        break
    return min_heap, []


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_hashing(rows: int, roofline: float, results: list[Timing]) -> None:
    g = np.random.default_rng(0)
    u64 = g.integers(0, 2**63, rows, dtype=np.int64).astype(np.uint64)

    # SHA-1 is slow enough that a full-size run dominates the whole suite;
    # measure a slice and extrapolate, and say so.
    slice_n = min(rows, 200_000)
    t = bench(
        lambda: py_sha1_hash_batch(u64[:slice_n]),
        slice_n,
        "hash u64: python sha1",
        reps=3,
    )
    t.notes["extrapolated_from"] = slice_n
    results.append(_apply_roofline(t, roofline, slice_n * 8))

    if native:
        t = bench(
            lambda: native.mix_u64_batch(u64), rows, "hash u64: native mix64", reps=7
        )
        results.append(_apply_roofline(t, roofline, rows * 8 * 2))  # read + write


def bench_kmv(rows: int, roofline: float, results: list[Timing]) -> None:
    g = np.random.default_rng(1)
    hashes = g.integers(0, 2**63, rows, dtype=np.int64).astype(np.uint64)
    k = 2048

    def run_py():
        state: list[int] = []
        for i in range(0, len(hashes), 250_000):
            py_kmv_add_many(state, k, hashes[i : i + 250_000])

    results.append(
        _apply_roofline(
            bench(run_py, rows, "kmv ingest: python", reps=3), roofline, rows * 8
        )
    )

    if native:

        def run_native():
            s = native.KmvSketch(k)
            for i in range(0, len(hashes), 250_000):
                s.offer_hashes(hashes[i : i + 250_000])

        results.append(
            _apply_roofline(
                bench(run_native, rows, "kmv ingest: native", reps=7),
                roofline,
                rows * 8,
            )
        )


def py_full_numeric_pass(arr: np.ndarray) -> dict:
    """Everything the numeric accumulator needs, the way the library gets it.

    The moments-only comparison flatters NumPy: in the real path the same
    column is also scanned for NaN, inf, zeros, negatives, min, max,
    int-likeness, monotonicity and the geometric-mean log-sum, each as its own
    pass. This is the honest baseline for the fused native scan.
    """
    nan = np.isnan(arr)
    inf = np.isinf(arr)
    finite = arr[~(nan | inf)]
    n = len(finite)
    mean = finite.sum() / n
    d = finite - mean
    out = {
        "n_nan": int(nan.sum()),
        "n_inf": int(inf.sum()),
        "n_zeros": int((finite == 0).sum()),
        "n_negatives": int((finite < 0).sum()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(mean),
        "m2": float(np.sum(d * d)),
        "m3": float(np.sum(d * d * d)),
        "m4": float(np.sum(d * d * d * d)),
        "int_like": bool(np.all(finite == np.trunc(finite))),
    }
    diffs = np.diff(finite)
    out["mono_inc"] = bool((diffs >= 0).all())
    out["mono_dec"] = bool((diffs <= 0).all())
    pos = finite[finite > 0]
    out["log_sum_pos"] = float(np.log(pos).sum()) if pos.size else 0.0
    out["n_pos"] = int(pos.size)
    return out


def bench_moments(rows: int, roofline: float, results: list[Timing]) -> None:
    arr = np.random.default_rng(2).standard_normal(rows)

    results.append(
        _apply_roofline(
            bench(
                lambda: py_moments_update(arr),
                rows,
                "moments only: numpy 5-pass",
                reps=5,
            ),
            roofline,
            # isfinite + boolean-index copy + sum + 3 temporaries: each n*8 read,
            # temporaries also n*8 written.
            rows * 8 * 11,
        )
    )
    results.append(
        _apply_roofline(
            bench(
                lambda: py_full_numeric_pass(arr),
                rows,
                "full column: numpy multi-pass",
                reps=5,
            ),
            roofline,
            rows * 8 * 22,
        )
    )
    if native:
        results.append(
            _apply_roofline(
                bench(
                    lambda: native.scan_numeric(arr),
                    rows,
                    "full column: native tiled scan",
                    reps=7,
                ),
                roofline,
                rows * 8,
            )
        )
        results.append(
            _apply_roofline(
                bench(
                    lambda: native.scan_numeric(arr, None, False),
                    rows,
                    "full column: native, no gmean",
                    reps=7,
                ),
                roofline,
                rows * 8,
            )
        )


def bench_reservoir(rows: int, roofline: float, results: list[Timing]) -> None:
    arr = np.random.default_rng(3).standard_normal(rows)
    k = 20_000

    def run_py():
        buf: list = []
        seen = 0
        for i in range(0, len(arr), 250_000):
            buf, seen = py_reservoir_add_many(buf, seen, k, arr[i : i + 250_000])

    results.append(
        _apply_roofline(
            bench(run_py, rows, "reservoir: python", reps=3), roofline, rows * 8
        )
    )
    if native:

        def run_native():
            r = native.Reservoir(k, 0)
            for i in range(0, len(arr), 250_000):
                r.add_many(arr[i : i + 250_000])

        results.append(
            _apply_roofline(
                bench(run_native, rows, "reservoir: native alg L", reps=7),
                roofline,
                rows * 8,
            )
        )


def bench_monotonic(rows: int, roofline: float, results: list[Timing]) -> None:
    arr = np.sort(np.random.default_rng(4).standard_normal(rows))
    slice_n = min(rows, 500_000)
    t = bench(
        lambda: py_monotonic_update(arr[:slice_n]),
        slice_n,
        "monotonic: python loop",
        reps=3,
    )
    t.notes["extrapolated_from"] = slice_n
    results.append(_apply_roofline(t, roofline, slice_n * 8))

    # np.diff is the obvious pure-Python fix, and worth showing: not every hot
    # spot needs Rust.
    def np_mono(a):
        d = np.diff(a)
        return bool((d >= 0).all()), bool((d <= 0).all())

    results.append(
        _apply_roofline(
            bench(lambda: np_mono(arr), rows, "monotonic: numpy diff", reps=5),
            roofline,
            rows * 8 * 3,
        )
    )
    if native:
        results.append(
            _apply_roofline(
                bench(
                    lambda: native.scan_numeric(arr),
                    rows,
                    "monotonic: native (in fused scan)",
                    reps=7,
                ),
                roofline,
                rows * 8,
            )
        )


def bench_extremes(rows: int, roofline: float, results: list[Timing]) -> None:
    arr = np.random.default_rng(5).standard_normal(rows)
    results.append(
        _apply_roofline(
            bench(
                lambda: py_extreme_update(arr), rows, "extremes: python heap", reps=5
            ),
            roofline,
            rows * 8,
        )
    )


def bench_gil(rows: int, results: list[Timing]) -> None:
    """Does the native extension actually release the GIL?

    Four threads, four disjoint arrays. If the GIL is held for the duration of
    each call, wall time is ~4x the single-threaded time. If it is released,
    wall time approaches the single-threaded time (up to core count).
    """
    if not native:
        return
    n_threads = min(
        4,
        (
            len(__import__("os").sched_getaffinity(0))
            if hasattr(__import__("os"), "sched_getaffinity")
            else 4
        ),
    )
    arrays = [
        np.random.default_rng(10 + i).standard_normal(rows) for i in range(n_threads)
    ]

    single = bench(
        lambda: native.scan_numeric(arrays[0]), rows, "gil: 1 thread", reps=5
    )
    results.append(single)

    def threaded():
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(native.scan_numeric, arrays))

    t = bench(threaded, rows * n_threads, f"gil: {n_threads} threads", reps=5)
    ideal = single.best_s * n_threads
    t.notes["threads"] = n_threads
    t.notes["serial_equivalent_s"] = round(ideal, 6)
    t.notes["parallel_efficiency"] = round(ideal / t.best_s / n_threads, 3)
    t.notes["gil_released"] = t.best_s < 0.75 * ideal
    results.append(t)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_table(results: list[Timing], roofline_bps: float) -> str:
    w = max(len(r.name) for r in results) + 2
    lines = [
        f"{'kernel':<{w}}{'ns/row':>10}{'M rows/s':>11}{'% roofline':>12}  notes",
        "-" * (w + 33 + 20),
    ]
    for r in results:
        pct = "" if math.isnan(r.pct_of_roofline) else f"{r.pct_of_roofline:>11.1f}%"
        note = ", ".join(f"{k}={v}" for k, v in r.notes.items())
        lines.append(
            f"{r.name:<{w}}{r.ns_per_row:>10.1f}{r.m_rows_per_s:>11.1f}{pct:>12}  {note}"
        )

    # Speedup pairs
    pairs = [
        ("hash u64: python sha1", "hash u64: native mix64"),
        ("kmv ingest: python", "kmv ingest: native"),
        ("moments only: numpy 5-pass", "full column: native tiled scan"),
        ("full column: numpy multi-pass", "full column: native tiled scan"),
        ("full column: numpy multi-pass", "full column: native, no gmean"),
        ("reservoir: python", "reservoir: native alg L"),
        ("monotonic: python loop", "monotonic: numpy diff"),
        ("monotonic: python loop", "monotonic: native (in fused scan)"),
    ]
    by_name = {r.name: r for r in results}
    speedups = []
    for a, b in pairs:
        if a in by_name and b in by_name:
            ratio = by_name[a].ns_per_row / by_name[b].ns_per_row
            flag = "  <-- SLOWER" if ratio < 1.0 else ""
            speedups.append(f"  {ratio:>7.1f}x   {a}  ->  {b}{flag}")
    if speedups:
        lines += ["", "speedups", "-" * 8, *speedups]

    lines += [
        "",
        f"memory roofline: {roofline_bps / 1e9:.2f} GB/s sequential read (np.sum on float64)",
        "  % roofline < 20  -> compute-bound, the instruction stream is the problem",
        "  % roofline > 70  -> bandwidth-bound, only fewer passes will help",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument(
        "--only", type=str, default=None, help="substring filter on kernel group"
    )
    args = ap.parse_args(argv)

    print(
        f"python {platform.python_version()}  {platform.machine()}  {platform.system()}"
    )
    print(
        f"native extension: {'yes v' + native.__version__ if native else 'NOT INSTALLED (python-only baseline)'}"
    )
    print(f"rows: {args.rows:,}\n")

    roofline = measure_roofline(min(args.rows, 4_000_000))
    results: list[Timing] = []

    groups = {
        "hashing": bench_hashing,
        "kmv": bench_kmv,
        "moments": bench_moments,
        "reservoir": bench_reservoir,
        "monotonic": bench_monotonic,
        "extremes": bench_extremes,
    }
    for name, fn in groups.items():
        if args.only and args.only not in name:
            continue
        fn(args.rows, roofline, results)
    if not args.only or "gil" in args.only:
        bench_gil(args.rows, results)

    print(render_table(results, roofline))

    if args.json:
        payload = {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "native": native.__version__ if native else None,
            "rows": args.rows,
            "roofline_bytes_per_s": roofline,
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
