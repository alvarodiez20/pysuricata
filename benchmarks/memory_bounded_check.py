"""Prove `check` runs in a memory-constrained runner against a file larger
than the ceiling, rather than claiming it (#92).

#76 shipped `pysuricata check` with one acceptance criterion carried over
unmet from #42: "runs in a 512 MB runner against a file larger than RAM."
"Bounded memory, so it fits in CI" is the reason to prefer this over a
profiler that loads the frame -- until it is measured under an enforced
ceiling it is an argument, not a result.

    python -m benchmarks.memory_bounded_check                 # the default: 512 MB
    python -m benchmarks.memory_bounded_check --budget-mb 256
    python -m benchmarks.memory_bounded_check --json out.json

## How the ceiling is enforced

A literal Docker container (`--memory=512m`) is the closest thing to what a
CI runner actually gives you, but it needs a daemon this environment does not
have. Two fallbacks, tried in order, both enforced by the kernel rather than
merely observed after the fact:

1. **A child cgroup** (v1 `memory.limit_in_bytes`), created under this
   process's own memory cgroup if it is writable. This is what `--memory`
   itself rests on, so a process that stays under a cgroup ceiling behaves
   the same way it would under Docker's. `memory.max_usage_in_bytes` after
   the run is the peak the kernel actually recorded for the group -- RSS and
   page cache both, not a sample taken from outside the process.
2. **`RLIMIT_AS`**, a POSIX resource limit every Python has, if the cgroup
   filesystem is not writable (a locked-down CI runner, most laptops). This
   caps virtual address space rather than resident memory, which is a
   stricter proxy -- a large `mmap` that is mostly untouched pages would trip
   it without actually pressuring a real container. `resource.getrusage(...)
   .ru_maxrss` (Linux: KB) is the peak reading in this mode.

Either way, the subprocess is killed by the kernel (`OOMKilled` / `MemoryError`
/ a nonzero return code from a signal) if it actually exceeds the ceiling --
this is a real constraint the run either satisfies or does not, not an
after-the-fact observation of what usage happened to be.

## The file

A single-file Parquet dataset, written in row-group batches so *building* it
does not itself need the ceiling (only the two `check` invocations run
constrained), sized comfortably past the budget on disk. Parquet, so the
reader streams it a batch at a time rather than loading the frame -- see
`docs/data-sources.md`. Text-heavy by default: `docs/adr/memory-budget.md`
flags this as the shape most likely to break its model, since that model was
fitted on numeric columns only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The ADR's fitted model (docs/adr/memory-budget.md), for a prediction to
#: put next to the measured number. Not pysuricata's memory -- a rough floor
#: for the whole process, including the interpreter and pandas/pyarrow.
_ADR_BASE_MB = 75.0
_ADR_PER_COLUMN = 0.5
_ADR_PER_SAMPLE_SLOT_BYTES = 37
_ADR_PER_CHUNK_ROW_BYTES = 48


def _adr_prediction_mb(n_cols: int, chunk_size: int, sample_k: int) -> float:
    return _ADR_BASE_MB + n_cols * (
        _ADR_PER_COLUMN
        + sample_k * _ADR_PER_SAMPLE_SLOT_BYTES / 1e6
        + chunk_size * _ADR_PER_CHUNK_ROW_BYTES / 1e6
    )


@dataclass
class Result:
    mechanism: str
    budget_mb: float
    file_bytes: int
    rows: int
    cols: int
    chunk_size: int
    write_baseline_passed: bool
    write_baseline_peak_mb: float | None
    compare_passed: bool
    compare_peak_mb: float | None
    adr_predicted_mb: float
    seconds: float


def _build_parquet(path: str, rows: int, target_bytes: int) -> tuple[int, int]:
    """Write a text-heavy Parquet file in row-group batches, growing it until
    it clears `target_bytes` on disk. Returns (rows_written, n_cols)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    batch_rows = 200_000
    n_cols = 12
    words = np.array(
        [f"category-{i:04d}-{'x' * (i % 37)}" for i in range(500)], dtype=object
    )

    writer = None
    written = 0
    try:
        while True:
            batch = {}
            for c in range(n_cols):
                if c % 3 == 0:
                    batch[f"num_{c}"] = rng.lognormal(2.0, 1.0, batch_rows)
                elif c % 3 == 1:
                    batch[f"txt_{c}"] = rng.choice(words, batch_rows)
                else:
                    batch[f"ts_{c}"] = rng.integers(
                        1_600_000_000, 1_700_000_000, batch_rows
                    ).astype("datetime64[s]")
            table = pa.table(batch)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            written += batch_rows
            if os.path.getsize(path) >= target_bytes and written >= rows:
                break
    finally:
        if writer is not None:
            writer.close()
    return written, n_cols


# The check subprocess writes its own peak-memory reading to this fd before
# exiting, so the parent does not have to reconstruct it from a killed
# process's absence of output.
_PROBE = """
import json, os, resource, sys

mode = sys.argv[1]
if mode == "rlimit":
    budget_bytes = int(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_AS, (budget_bytes, budget_bytes))

sys.path.insert(0, {repo_root!r})
from pysuricata.cli import main as cli_main

sys.argv = ["pysuricata"] + sys.argv[3:]
code = cli_main()

peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with open(os.environ["PEAK_OUT"], "w") as f:
    json.dump({{"exit_code": code, "peak_kb": peak_kb}}, f)
sys.exit(code)
"""


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


def _run_constrained(
    args: list[str], budget_mb: float, mechanism: str, cgroup_base: str | None
) -> tuple[bool, float | None]:
    """Run `pysuricata` CLI `args` under the chosen ceiling.

    Returns (passed, peak_mb). `passed` is False if the process was killed or
    exited nonzero for a reason other than pysuricata's own exit code 2 (a
    dataset error unrelated to memory would also be caught upstream, so this
    is specifically "did the kernel intervene").
    """
    budget_bytes = int(budget_mb * 1024 * 1024)
    peak_out = tempfile.mktemp(prefix="pysuricata-mem-probe-")
    env = {**os.environ, "PEAK_OUT": peak_out}
    script = _PROBE.format(repo_root=REPO_ROOT)

    my_cgroup = None
    if mechanism == "cgroup":
        my_cgroup = os.path.join(cgroup_base, f"check-probe-{uuid.uuid4().hex[:8]}")
        os.mkdir(my_cgroup)
        with open(os.path.join(my_cgroup, "memory.limit_in_bytes"), "w") as f:
            f.write(str(budget_bytes))

    cmd = [sys.executable, "-c", script, mechanism, str(budget_bytes), "check", *args]

    def _preexec():
        if my_cgroup is not None:
            with open(os.path.join(my_cgroup, "cgroup.procs"), "w") as f:
                f.write(str(os.getpid()))

    proc = subprocess.run(
        cmd, env=env, preexec_fn=_preexec if my_cgroup else None, capture_output=True
    )

    peak_mb = None
    if my_cgroup is not None:
        try:
            with open(os.path.join(my_cgroup, "memory.max_usage_in_bytes")) as f:
                peak_mb = int(f.read().strip()) / 1024 / 1024
        except OSError:
            pass
        finally:
            shutil.rmtree(my_cgroup, ignore_errors=True)

    if os.path.exists(peak_out):
        try:
            with open(peak_out) as f:
                probe_result = json.load(f)
            if peak_mb is None:
                peak_mb = probe_result["peak_kb"] / 1024
        except (OSError, json.JSONDecodeError, KeyError):
            probe_result = None
        finally:
            os.remove(peak_out)
    else:
        probe_result = None

    if proc.returncode < 0:
        return False, peak_mb  # killed by a signal -- the kernel intervened
    if probe_result is None:
        return False, peak_mb  # crashed before it could report -- e.g. MemoryError
    return probe_result["exit_code"] in (0, 1), peak_mb


def run(budget_mb: float, rows: int, seed: int) -> Result:
    cgroup_base = _cgroup_dir()
    mechanism = "cgroup" if cgroup_base is not None else "rlimit"

    tmpdir = tempfile.mkdtemp(prefix="pysuricata-memcheck-")
    data_path = os.path.join(tmpdir, "wide.parquet")
    baseline_path = os.path.join(tmpdir, "baseline.json")
    target_bytes = int(budget_mb * 1024 * 1024 * 1.5)

    started = time.perf_counter()
    written_rows, n_cols = _build_parquet(data_path, rows, target_bytes)
    file_bytes = os.path.getsize(data_path)

    chunk_size = 50_000
    write_passed, write_peak = _run_constrained(
        [
            data_path,
            "--write-baseline",
            baseline_path,
            "--chunk-size",
            str(chunk_size),
            "--seed",
            str(seed),
            "--quiet",
        ],
        budget_mb,
        mechanism,
        cgroup_base,
    )
    compare_passed, compare_peak = _run_constrained(
        [
            data_path,
            "--baseline",
            baseline_path,
            "--chunk-size",
            str(chunk_size),
            "--seed",
            str(seed),
            "--quiet",
            "--json",
        ],
        budget_mb,
        mechanism,
        cgroup_base,
    )
    elapsed = time.perf_counter() - started
    shutil.rmtree(tmpdir, ignore_errors=True)

    return Result(
        mechanism=mechanism,
        budget_mb=budget_mb,
        file_bytes=file_bytes,
        rows=written_rows,
        cols=n_cols,
        chunk_size=chunk_size,
        write_baseline_passed=write_passed,
        write_baseline_peak_mb=write_peak,
        compare_passed=compare_passed,
        compare_peak_mb=compare_peak,
        adr_predicted_mb=round(
            _adr_prediction_mb(n_cols, chunk_size, sample_k=20_000), 1
        ),
        seconds=round(elapsed, 1),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-mb", type=float, default=512.0)
    ap.add_argument(
        "--rows",
        type=int,
        default=6_000_000,
        help="minimum rows to write (the byte target usually binds first)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    result = run(args.budget_mb, args.rows, args.seed)

    print(f"mechanism:        {result.mechanism}")
    print(f"budget:           {result.budget_mb:.0f} MB")
    print(
        f"file on disk:     {result.file_bytes / 1024 / 1024:,.0f} MB "
        f"({result.rows:,} rows x {result.cols} cols) -- "
        f"{result.file_bytes / 1024 / 1024 / result.budget_mb:.1f}x the budget"
    )
    print(f"chunk_size:       {result.chunk_size:,}")
    print(
        f"write-baseline:   {'passed' if result.write_baseline_passed else 'FAILED'}"
        + (
            f", peak {result.write_baseline_peak_mb:.0f} MB"
            if result.write_baseline_peak_mb is not None
            else ""
        )
    )
    print(
        f"compare:          {'passed' if result.compare_passed else 'FAILED'}"
        + (
            f", peak {result.compare_peak_mb:.0f} MB"
            if result.compare_peak_mb is not None
            else ""
        )
    )
    print(f"ADR prediction:   {result.adr_predicted_mb:.0f} MB (this shape)")
    print(f"wall clock:       {result.seconds:.1f}s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(asdict(result), f, indent=2)

    return 0 if result.write_baseline_passed and result.compare_passed else 1


if __name__ == "__main__":
    sys.exit(main())
