"""End-to-end: PySuricata vs the incumbents, on identical data.

Run:
    python -m benchmarks.end_to_end
    python -m benchmarks.end_to_end --tools pysuricata,ydata --scale 0.2
    python -m benchmarks.end_to_end --json results.json --markdown results.md

Rules this harness follows, so the numbers survive contact with a sceptical
reader:

* Every tool gets the **same DataFrame object**, generated once per suite.
* Each tool runs in a **fresh subprocess**, so import cost, allocator state and
  garbage from a previous tool cannot leak into the next measurement.
* **Peak RSS** is recorded alongside wall time. For this category memory is
  often the real story — the incumbent's most-cited failure mode is a
  MemoryError, not slowness.
* A tool that raises is recorded as a **failure with its exception**, not
  dropped. "ydata-profiling OOMs at 5M rows" is a result.
* **Output size** is recorded. A 40 MB HTML report that renders in 200 ms is
  not obviously better than a 2 MB one that takes 400 ms.
* Anything that could not be installed is reported as ``skipped``, never
  silently omitted.
* Every tool is measured in **every round**, and each tool's best is reported.
  Running one tool to completion and then the next compares them across two
  different stretches of machine time; on a shared runner the available CPU
  moves between them. Interleaving makes a slow patch penalise every tool in
  that round, so it cancels in the ratio. Fewer than three rounds and the
  output says so.

Publish the generated markdown table *with* the environment block. A benchmark
without the machine spec is a screenshot.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time

# Below this, a ratio between two tools is not quotable: it has not been
# measured enough times, interleaved, to separate the difference between the
# tools from the difference between two stretches of machine time.
MIN_QUOTABLE_ROUNDS = 3

TOOLS = {
    "pysuricata": {
        "import": "pysuricata",
        "code": """
from pysuricata import profile
rep = profile(df)
rep.save_html(out)
""",
    },
    "pysuricata-summarize": {
        "import": "pysuricata",
        "code": """
from pysuricata import summarize
import json as _j
s = summarize(df)
open(out, "w").write(_j.dumps({"n": len(s.get("columns", {}))}))
""",
    },
    "ydata": {
        "import": "ydata_profiling",
        "code": """
from ydata_profiling import ProfileReport
ProfileReport(df, minimal=False, progress_bar=False).to_file(out)
""",
    },
    "ydata-minimal": {
        "import": "ydata_profiling",
        "code": """
from ydata_profiling import ProfileReport
ProfileReport(df, minimal=True, progress_bar=False).to_file(out)
""",
    },
    "sweetviz": {
        "import": "sweetviz",
        "code": """
import sweetviz as sv
sv.analyze(df).show_html(out, open_browser=False, layout="vertical")
""",
    },
    "skimpy": {
        "import": "skimpy",
        "code": """
from skimpy import skim
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    skim(df)
open(out, "w").write(buf.getvalue())
""",
    },
}


# Deliberately not indented: this is source code for a subprocess, and the
# {body} substitution has to land at a known indentation level.
RUNNER = """\
import json, os, resource, sys, time
sys.path.insert(0, {repo!r})
from benchmarks import datasets

df = datasets.build({suite!r}, scale={scale!r})
out = {out!r}
t0 = time.perf_counter()
err = None
try:
{body}
except BaseException as e:
    err = "{{}}: {{}}".format(type(e).__name__, e)
elapsed = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# Linux reports KiB, macOS bytes.
rss_mb = rss / 1024 if sys.platform != "darwin" else rss / 1024 / 1024
size = os.path.getsize(out) if os.path.exists(out) else 0
print("__RESULT__" + json.dumps(
    {{"seconds": elapsed, "peak_rss_mb": rss_mb, "output_bytes": size, "error": err}}
))
"""


def have(module: str) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
        ).returncode
        == 0
    )


def run_one(tool: str, suite: str, scale: float, repo: str, timeout: int) -> dict:
    spec = TOOLS[tool]
    if not have(spec["import"]):
        return {"status": "skipped", "reason": f"{spec['import']} not installed"}

    suffix = ".html" if "summarize" not in tool and "skimpy" not in tool else ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        out = fh.name
    body = textwrap.indent(spec["code"].strip(), " " * 4)
    script = RUNNER.format(repo=repo, suite=suite, scale=scale, out=out, body=body)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": timeout}
    finally:
        if os.path.exists(out):
            try:
                os.unlink(out)
            except OSError:
                pass

    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            payload = json.loads(line[len("__RESULT__") :])
            payload["status"] = "error" if payload.get("error") else "ok"
            return payload
    return {
        "status": "crashed",
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip()[-800:],
    }


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    for mod in (
        "pandas",
        "numpy",
        "polars",
        "pysuricata",
        "pysuricata_core",
        "ydata_profiling",
    ):
        r = subprocess.run(
            [sys.executable, "-c", f"import {mod};print({mod}.__version__)"],
            capture_output=True,
            text=True,
        )
        env[mod] = r.stdout.strip() if r.returncode == 0 else None
    return env


def to_markdown(payload: dict) -> str:
    env = payload["environment"]
    lines = [
        "# PySuricata benchmark results",
        "",
        "## Environment",
        "",
        f"- {env['system']} / {env['machine']} / {env['cpu_count']} cores",
        f"- Python {env['python']}, pandas {env['pandas']}, numpy {env['numpy']}",
        f"- pysuricata {env['pysuricata']}"
        + (
            f" + native core {env['pysuricata_core']}"
            if env.get("pysuricata_core")
            else " (pure Python)"
        ),
        f"- ydata-profiling {env['ydata_profiling'] or 'not installed'}",
        f"- {payload.get('rounds', 1)} interleaved round(s), best per tool",
        "",
    ]
    if not payload.get("quotable", False):
        lines += [
            "> **Not quotable.** Fewer than "
            f"{MIN_QUOTABLE_ROUNDS} rounds were run, so a ratio between two "
            "tools here cannot be separated from machine noise. Re-run with "
            "`--rounds 5` before publishing any of these numbers.",
            "",
        ]
    for suite, rows in payload["suites"].items():
        shape = rows.pop("_shape", {})
        lines += [
            f"## {suite} — {shape.get('rows', '?'):,} rows x {shape.get('cols', '?')} cols "
            f"({shape.get('bytes', 0) / 1e6:.0f} MB in memory)",
            "",
            "| tool | wall (s) | spread | peak RSS (MB) | output (MB) | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        ok = {k: v for k, v in rows.items() if v.get("status") == "ok"}
        fastest = min((v["seconds"] for v in ok.values()), default=None)
        for tool, r in rows.items():
            if r.get("status") == "ok":
                mark = (
                    " **(fastest)**"
                    if fastest and abs(r["seconds"] - fastest) < 1e-9
                    else ""
                )
                spread = f"{r['spread_pct']:.0f}%" if "spread_pct" in r else "—"
                lines.append(
                    f"| {tool} | {r['seconds']:.2f} | {spread} | "
                    f"{r['peak_rss_mb']:.0f} | "
                    f"{r['output_bytes'] / 1e6:.2f} | ok{mark} |"
                )
            else:
                detail = r.get("error") or r.get("reason") or r.get("stderr", "")
                lines.append(
                    f"| {tool} | — | — | — | — | {r['status']}: {detail[:120]} |"
                )
        lines.append("")
    return "\n".join(lines)


def round_robin(
    tools: list[str], suite: str, scale: float, repo: str, timeout: int, rounds: int
) -> dict[str, dict]:
    """Measure every tool in every round, then keep each tool's best.

    The schedule is the point. Measuring tool A to completion and then tool B
    compares them across two different stretches of machine time -- and on a
    shared runner the available CPU moves between them. Interleaving means a
    slow patch penalises every tool in that round, so it cancels in the ratio.

    This is not hypothetical. Two published claims came from cross-session
    pairing: "0.0.21 is 1.24x faster than 0.0.16", which is a 0.88x *regression*
    when both are measured in one round-robin, and a 3.56x headline that is
    really 2.48x.

    Args:
        tools: Tool names to measure, all of them in every round.
        suite: Dataset suite name.
        scale: Dataset scale factor.
        repo: Repository root, put on the subprocess's sys.path.
        timeout: Per-run timeout in seconds.
        rounds: Number of interleaved rounds; each tool's best is reported.

    Returns:
        Mapping of tool name to its best result, with every round's timing
        retained under ``all_seconds`` so the spread stays visible.
    """
    best: dict[str, dict] = {}
    for round_index in range(1, rounds + 1):
        if rounds > 1:
            print(f"  -- round {round_index}/{rounds}")
        for tool in tools:
            if tool not in TOOLS:
                print(f"  {tool:<24} unknown tool, skipping")
                continue
            started = time.perf_counter()
            result = run_one(tool, suite, scale, repo, timeout)
            previous = best.get(tool)

            if result["status"] == "ok":
                timings = (previous or {}).get("all_seconds", [])
                result["all_seconds"] = [*timings, result["seconds"]]
                if previous is None or previous.get("status") != "ok":
                    best[tool] = result
                elif result["seconds"] < previous["seconds"]:
                    best[tool] = result
                else:
                    previous["all_seconds"] = result["all_seconds"]
                print(
                    f"  {tool:<24} {result['seconds']:>8.2f}s  "
                    f"{result['peak_rss_mb']:>7.0f} MB RSS  "
                    f"{result['output_bytes'] / 1e6:>6.2f} MB out"
                )
            else:
                if previous is None:
                    best[tool] = result
                print(
                    f"  {tool:<24} {result['status'].upper():>8}  "
                    f"{(result.get('error') or result.get('reason') or '')[:70]}"
                    f"  ({time.perf_counter() - started:.1f}s)"
                )

    for result in best.values():
        timings = result.get("all_seconds")
        if timings and len(timings) > 1:
            result["rounds"] = len(timings)
            result["spread_pct"] = (max(timings) - min(timings)) / min(timings) * 100
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--suites", default="mixed,numeric_wide,categorical_heavy")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Interleaved rounds; every tool is measured in each one and its "
        "best is reported. Fewer than 3 marks the results unquotable.",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)

    if args.rounds < 1:
        ap.error("--rounds must be at least 1")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    from benchmarks import datasets

    payload = {
        "environment": environment(),
        "rounds": args.rounds,
        "quotable": args.rounds >= MIN_QUOTABLE_ROUNDS,
        "suites": {},
    }
    print(json.dumps(payload["environment"], indent=2), "\n")
    if not payload["quotable"]:
        print(
            f"WARNING: --rounds {args.rounds} is below the {MIN_QUOTABLE_ROUNDS} "
            "needed for a quotable ratio. Results are indicative only.\n"
        )

    tools = [t.strip() for t in args.tools.split(",")]
    for suite in args.suites.split(","):
        suite = suite.strip()
        print(f"=== {suite} (scale={args.scale}, rounds={args.rounds}) ===")
        df = datasets.build(suite, scale=args.scale)
        payload["suites"][suite] = {"_shape": datasets.describe(df)}
        del df
        payload["suites"][suite].update(
            round_robin(tools, suite, args.scale, repo, args.timeout, args.rounds)
        )
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(to_markdown(json.loads(json.dumps(payload))))
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
