"""Version-over-version timing, measured in one interleaved round-robin.

Run:

    python -m benchmarks.versions                       # installed versions
    python -m benchmarks.versions --versions 0.0.16,0.0.21,0.0.31
    python -m benchmarks.versions --rounds 5 --markdown curve.md

Why this file exists rather than a shell loop.

A version-over-version curve is a table of *ratios*, and a ratio is only as
trustworthy as the pairing behind it. Two claims in this project's history came
from pairing measurements taken at different times on a machine whose available
CPU varies between sessions:

* "0.0.21 is 1.24x faster than 0.0.16" -- measured properly, in one round-robin,
  it is **0.88x**. A regression, reported as an improvement.
* "3.56x since 0.0.16" -- really **2.48x**. The larger figure paired a slow
  0.0.16 run with a fast 0.0.31 run.

Both were honest mistakes with the same cause and the same fix: measure every
version in every round, interleaved, and never quote a number produced any other
way. This module makes that the only way to run it.

Each version is installed into its own throwaway virtualenv and timed in its own
subprocess, so import cost, allocator state and garbage from one version cannot
leak into the next.
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

from benchmarks.end_to_end import (
    MIN_QUOTABLE_ROUNDS,
    _report_load,
    environment,
    load_average,
    load_guard,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The measured body. Deliberately not indented: it is source for a subprocess.
RUNNER = """\
import json, sys, time
sys.path.insert(0, {repo!r})
from benchmarks import datasets
from pysuricata import summarize

df = datasets.build({suite!r}, scale={scale!r})
summarize(df)                       # warm imports and any lazy setup
t0 = time.perf_counter()
summarize(df)
elapsed = time.perf_counter() - t0
import pysuricata
print("__RESULT__" + json.dumps({{"seconds": elapsed, "version": pysuricata.__version__}}))
"""


def make_env(version: str, workdir: str) -> str | None:
    """Install one released version into a throwaway virtualenv.

    Args:
        version: Version specifier to install, or "." for the working tree.
        workdir: Directory to create the environment under.

    Returns:
        Path to the environment's python, or None if the install failed.
    """
    env_dir = os.path.join(workdir, f"v{version}")
    target = REPO if version == "." else f"pysuricata=={version}"
    create = subprocess.run(
        ["uv", "venv", env_dir], capture_output=True, text=True, cwd=REPO
    )
    if create.returncode != 0:
        print(f"  {version:<10} venv failed: {create.stderr.strip()[:120]}")
        return None
    python = os.path.join(env_dir, "bin", "python")
    install = subprocess.run(
        ["uv", "pip", "install", "--python", python, "--quiet", target],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if install.returncode != 0:
        print(f"  {version:<10} install failed: {install.stderr.strip()[-160:]}")
        return None
    return python


def time_once(python: str, suite: str, scale: float, timeout: int) -> dict:
    """Time one `summarize()` call in a fresh subprocess."""
    script = RUNNER.format(repo=REPO, suite=suite, scale=scale)
    try:
        proc = subprocess.run(
            [python, "-c", script], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            payload = json.loads(line[len("__RESULT__") :])
            payload["status"] = "ok"
            return payload
    return {"status": "crashed", "stderr": proc.stderr.strip()[-400:]}


def round_robin(
    pythons: dict[str, str], suite: str, scale: float, rounds: int, timeout: int
) -> dict[str, dict]:
    """Time every version in every round; keep each version's best.

    Args:
        pythons: Mapping of version label to interpreter path.
        suite: Dataset suite name.
        scale: Dataset scale factor.
        rounds: Number of interleaved rounds.
        timeout: Per-run timeout in seconds.

    Returns:
        Mapping of version label to its best result, retaining every round's
        timing under ``all_seconds``.
    """
    best: dict[str, dict] = {}
    for index in range(1, rounds + 1):
        print(f"  -- round {index}/{rounds}")
        for version, python in pythons.items():
            result = time_once(python, suite, scale, timeout)
            if result["status"] != "ok":
                best.setdefault(version, result)
                print(f"  {version:<10} {result['status'].upper()}")
                continue
            timings = best.get(version, {}).get("all_seconds", [])
            result["all_seconds"] = [*timings, result["seconds"]]
            if best.get(version, {}).get("status") != "ok" or result["seconds"] < best[
                version
            ].get("seconds", float("inf")):
                best[version] = result
            else:
                best[version]["all_seconds"] = result["all_seconds"]
            print(f"  {version:<10} {result['seconds']:>8.3f}s")
    return best


def to_markdown(payload: dict) -> str:
    env = payload["environment"]
    baseline = payload.get("baseline")
    results = payload["results"]
    base_seconds = results.get(baseline, {}).get("seconds")

    lines = [
        "# PySuricata version-over-version",
        "",
        f"- {env['system']} / {env['machine']} / {env['cpu_count']} cores",
        f"- Python {env['python']}, pandas {env['pandas']}, numpy {env['numpy']}",
        f"- suite `{payload['suite']}` at scale {payload['scale']}",
        f"- {payload['rounds']} interleaved rounds, best per version",
        "",
    ]
    if not payload["quotable"]:
        lines += [
            f"> **Not quotable.** Fewer than {MIN_QUOTABLE_ROUNDS} rounds. "
            "Re-run with `--rounds 5` before publishing.",
            "",
        ]
    lines += [
        f"| version | ms | x vs {baseline} | spread |",
        "|---|---:|---:|---:|",
    ]
    for version, result in results.items():
        if result.get("status") != "ok":
            lines.append(f"| {version} | — | — | {result['status']} |")
            continue
        ms = result["seconds"] * 1000
        ratio = (
            f"{base_seconds / result['seconds']:.2f}x"
            if base_seconds and result["seconds"]
            else "—"
        )
        timings = result.get("all_seconds") or []
        spread = (
            f"{(max(timings) - min(timings)) / min(timings) * 100:.0f}%"
            if len(timings) > 1
            else "—"
        )
        lines.append(f"| {version} | {ms:,.1f} | {ratio} | {spread} |")
    lines += [
        "",
        "Ratios are only comparable within this table: every version was "
        "measured in the same rounds, on the same machine, in one run. "
        "Absolute times are not comparable to any other run.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--versions",
        default="0.0.16,0.0.21,0.0.26,0.0.27,.",
        help="Comma-separated versions to compare. '.' is the working tree.",
    )
    ap.add_argument("--suite", default="mixed")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Measure even when the machine is busy. The load is still "
        "recorded with the results, so the caveat travels with them.",
    )
    args = ap.parse_args(argv)

    # Before the environments are built, not after: building five of them takes
    # minutes, and refusing at the end of that is a worse experience than
    # refusing at the start.
    load_start, refusal = load_guard(args.force)
    if refusal:
        print(f"refusing to measure: {refusal}", file=sys.stderr)
        return 2

    if shutil.which("uv") is None:
        print("uv is required to build the per-version environments", file=sys.stderr)
        return 2

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    workdir = tempfile.mkdtemp(prefix="pysuricata-versions-")
    print(f"building {len(versions)} environments under {workdir}\n")

    started = time.perf_counter()
    pythons: dict[str, str] = {}
    for version in versions:
        python = make_env(version, workdir)
        if python:
            label = "working tree" if version == "." else version
            pythons[label] = python
            print(f"  {label:<12} ready")
    print(f"  ({time.perf_counter() - started:.0f}s)\n")

    if not pythons:
        print("no versions could be installed", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    print(f"=== {args.suite} (scale={args.scale}, rounds={args.rounds}) ===")
    try:
        results = round_robin(
            pythons, args.suite, args.scale, args.rounds, args.timeout
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    payload = {
        "environment": environment(),
        "suite": args.suite,
        "scale": args.scale,
        "rounds": args.rounds,
        "quotable": args.rounds >= MIN_QUOTABLE_ROUNDS,
        # Both ends: a load average lags, so a job that starts during the run
        # is invisible to the reading taken before it.
        "load_start": load_start,
        "load_end": load_average(),
        "forced": bool(args.force),
        "baseline": next(iter(results), None),
        "results": results,
    }
    print()
    _report_load(payload)
    print()
    print(to_markdown(payload))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(to_markdown(payload))
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
