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


class VersionPathFallthrough(RuntimeError):
    """A subprocess resolved `pysuricata` to code outside the venv it was
    meant to measure -- the version label on that result cannot be trusted
    (#249)."""


# The measured body. Deliberately not indented: it is source for a subprocess.
#
# `import pysuricata` happens first, before `REPO` (this checkout) is added to
# `sys.path`, and is checked against `env_dir` immediately (#249). Two things
# can otherwise shadow the venv's own installed distribution with this
# checkout's working-tree source: the empty-string cwd entry `python -c` puts
# at `sys.path[0]` when the caller's cwd is the repo root, and `REPO` itself
# once it is added, since it holds a real `pysuricata/` package directory.
# Either shadow still reports the *venv's* `pysuricata.__version__`
# (`importlib.metadata` reads installed-distribution metadata, not the module
# actually running) while executing completely different code underneath it.
# A round-robin that hits this measures one version four times under four
# different labels, and does so identically every round -- clean, flat, and
# false. `time_once()` sets `cwd` away from `REPO` for the same reason, and
# `REPO` is appended rather than inserted at the front so it can never win
# against a venv's own site-packages for a name both provide.
RUNNER = """\
import json, os, sys, time
import pysuricata

env_dir = {env_dir!r}
# `__file__` is None for an implicit namespace package -- no __init__.py
# anywhere on sys.path, just a directory that happens to be named right (an
# incidental empty `pysuricata/` left over from something else entirely is
# enough). That is not "found nothing", it is "found something that is not
# the package", and belongs in the same refusal as a real shadow rather than
# an unhandled TypeError out of realpath(None).
found = pysuricata.__file__
expected = os.path.realpath(env_dir)
if found is None or not os.path.realpath(found).startswith(expected + os.sep):
    print("__RESULT__" + json.dumps(
        {{"status": "path_fallthrough", "wanted": env_dir, "got": found}}
    ))
    sys.exit(0)

sys.path.append({repo!r})
from benchmarks import datasets
from pysuricata import summarize

df = datasets.build({suite!r}, scale={scale!r})
summarize(df)                       # warm imports and any lazy setup
t0 = time.perf_counter()
summarize(df)
elapsed = time.perf_counter() - t0
print("__RESULT__" + json.dumps({{"seconds": elapsed, "version": pysuricata.__version__}}))
"""


def make_env(version: str, workdir: str) -> tuple[str, str] | None:
    """Install one released version into a throwaway virtualenv.

    Args:
        version: Version specifier to install, or "." for the working tree.
        workdir: Directory to create the environment under.

    Returns:
        `(python, env_dir)` -- the environment's interpreter and its own
        directory, the latter needed to check what actually got measured
        (#249) -- or `None` if the install failed.
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
    return python, env_dir


def time_once(
    python: str, env_dir: str, suite: str, scale: float, timeout: int
) -> dict:
    """Time one `summarize()` call in a fresh subprocess.

    Raises:
        VersionPathFallthrough: `pysuricata` resolved to something outside
            `env_dir` -- see the module docstring and `RUNNER` (#249).
    """
    script = RUNNER.format(repo=REPO, env_dir=env_dir, suite=suite, scale=scale)
    try:
        # cwd deliberately not REPO: `python -c` puts the empty string (cwd)
        # at sys.path[0], and REPO holds a real pysuricata/ directory that
        # would shadow the venv's own installation from there just as easily
        # as the sys.path.insert(0, ...) this fix also removed.
        proc = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=env_dir,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            payload = json.loads(line[len("__RESULT__") :])
            if payload.get("status") == "path_fallthrough":
                raise VersionPathFallthrough(
                    f"wanted pysuricata from {payload['wanted']!r}, got "
                    f"{payload['got']!r} -- this result cannot be trusted, "
                    "and neither can any other in this run"
                )
            payload["status"] = "ok"
            return payload
    return {"status": "crashed", "stderr": proc.stderr.strip()[-400:]}


def round_robin(
    pythons: dict[str, tuple[str, str]],
    suite: str,
    scale: float,
    rounds: int,
    timeout: int,
) -> dict[str, dict]:
    """Time every version in every round; keep each version's best.

    Args:
        pythons: Mapping of version label to `(python, env_dir)`.
        suite: Dataset suite name.
        scale: Dataset scale factor.
        rounds: Number of interleaved rounds.
        timeout: Per-run timeout in seconds.

    Returns:
        Mapping of version label to its best result, retaining every round's
        timing under ``all_seconds``.

    Raises:
        VersionPathFallthrough: A subprocess measured code outside the venv
            it was meant to (#249). Propagates uncaught -- deliberately: this
            failure mode corrupts every round identically, so a result
            gathered before it fired is exactly as untrustworthy as one
            gathered after, and there is no partial table worth returning.
    """
    best: dict[str, dict] = {}
    for index in range(1, rounds + 1):
        print(f"  -- round {index}/{rounds}")
        for version, (python, env_dir) in pythons.items():
            result = time_once(python, env_dir, suite, scale, timeout)
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
    pythons: dict[str, tuple[str, str]] = {}
    for version in versions:
        built = make_env(version, workdir)
        if built:
            label = "working tree" if version == "." else version
            pythons[label] = built
            print(f"  {label:<12} ready")
    print(f"  ({time.perf_counter() - started:.0f}s)\n")

    if not pythons:
        print("no versions could be installed", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    print(f"=== {args.suite} (scale={args.scale}, rounds={args.rounds}) ===")
    try:
        try:
            results = round_robin(
                pythons, args.suite, args.scale, args.rounds, args.timeout
            )
        except VersionPathFallthrough as exc:
            print(f"refusing to measure: {exc}", file=sys.stderr)
            return 2
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
