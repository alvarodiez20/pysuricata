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
        "",
    ]
    for suite, rows in payload["suites"].items():
        shape = rows.pop("_shape", {})
        lines += [
            f"## {suite} — {shape.get('rows', '?'):,} rows x {shape.get('cols', '?')} cols "
            f"({shape.get('bytes', 0) / 1e6:.0f} MB in memory)",
            "",
            "| tool | wall (s) | peak RSS (MB) | output (MB) | status |",
            "|---|---:|---:|---:|---|",
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
                lines.append(
                    f"| {tool} | {r['seconds']:.2f} | {r['peak_rss_mb']:.0f} | "
                    f"{r['output_bytes'] / 1e6:.2f} | ok{mark} |"
                )
            else:
                detail = r.get("error") or r.get("reason") or r.get("stderr", "")
                lines.append(f"| {tool} | — | — | — | {r['status']}: {detail[:120]} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--suites", default="mixed,numeric_wide,categorical_heavy")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    from benchmarks import datasets

    payload = {"environment": environment(), "suites": {}}
    print(json.dumps(payload["environment"], indent=2), "\n")

    for suite in args.suites.split(","):
        suite = suite.strip()
        print(f"=== {suite} (scale={args.scale}) ===")
        df = datasets.build(suite, scale=args.scale)
        payload["suites"][suite] = {"_shape": datasets.describe(df)}
        del df
        for tool in args.tools.split(","):
            tool = tool.strip()
            if tool not in TOOLS:
                print(f"  {tool:<24} unknown tool, skipping")
                continue
            t0 = time.perf_counter()
            res = run_one(tool, suite, args.scale, repo, args.timeout)
            payload["suites"][suite][tool] = res
            if res["status"] == "ok":
                print(
                    f"  {tool:<24} {res['seconds']:>8.2f}s  "
                    f"{res['peak_rss_mb']:>7.0f} MB RSS  "
                    f"{res['output_bytes'] / 1e6:>6.2f} MB out"
                )
            else:
                print(
                    f"  {tool:<24} {res['status'].upper():>8}  "
                    f"{(res.get('error') or res.get('reason') or '')[:70]}"
                    f"  ({time.perf_counter() - t0:.1f}s)"
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
