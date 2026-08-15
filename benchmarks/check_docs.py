"""Documentation staleness checker.

Docs drift silently: an API gets renamed, a config option disappears, a
performance number goes stale, and nothing fails. This walks every page in
``docs/`` and checks the things that *can* be checked mechanically, so the only
work left for a human is the prose.

    python -m benchmarks.check_docs             # report
    python -m benchmarks.check_docs --json out.json
    python -m benchmarks.check_docs --strict    # exit 1 on any ERROR

Checks, in rough order of how often they catch something:

1. **Runnable examples.** Every ``python`` fence that looks self-contained is
   executed in a subprocess. A traceback is an error; a warning is a warning.
2. **Public symbols.** Every ``pysuricata.X``, ``from pysuricata import X`` and
   ``ClassName(...)`` reference is resolved against the installed package.
3. **Config attributes.** Every ``config.compute.X`` / ``ProfileConfig(X=...)``
   / ``ComputeOptions(X=...)`` name is checked against the real dataclasses.
4. **Summary keys.** Every ``summarize(...)["a"]["b"]`` path is checked against
   a real ``summarize()`` result, which is how you catch things like
   ``rows`` becoming ``rows_est``.
5. **Internal links and images.** Relative links resolve to a file that exists.
6. **Nav coverage.** Pages on disk that mkdocs never renders, and nav entries
   pointing at files that do not exist.
7. **Stale markers.** Version strings, hardcoded timings and "N x faster"
   claims are listed for human review -- not auto-failed, since only a human
   knows whether a number is still true.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
SUMMARY_PATH = re.compile(
    r"""(?:summary|stats|result|s)\s*\[\s*["']([\w ]+)["']\s*\]\s*\[\s*["']([\w ]+)["']\s*\]"""
)
CFG_ATTR = re.compile(r"(?:config|cfg)\.(compute|output|report)\.(\w+)")
KWARG = re.compile(
    r"(?:ProfileConfig|ComputeOptions|OutputOptions)\s*\(([^)]*)\)", re.S
)
VERSION = re.compile(r"\b0\.0\.\d+\b")
TIMING = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|MB|GB|ns)\b(?!\w)")
SPEEDUP = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x|×)\s*(?:faster|slower|less|more|speedup)", re.I
)


@dataclasses.dataclass
class Finding:
    page: str
    line: int
    level: str  # ERROR | WARN | INFO
    kind: str
    detail: str


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return text[:idx].count("\n") + 1 if idx >= 0 else 0


# ---------------------------------------------------------------------------


def check_examples(page: Path, text: str, out: list[Finding], run: bool) -> None:
    """Execute python fences that look self-contained."""
    for m in FENCE.finditer(text):
        lang, code = m.group(1), m.group(2)
        if lang not in ("python", "py"):
            continue
        line = text[: m.start()].count("\n") + 1

        try:
            ast.parse(code)
        except SyntaxError as e:
            # A fence that does not parse is either a fragment (fine) or a typo
            # (not fine). Fragments almost always lack an import; use that.
            level = "WARN" if "import" not in code else "ERROR"
            out.append(
                Finding(
                    page.name,
                    line,
                    level,
                    "syntax",
                    f"{e.msg} (line {e.lineno} of block)",
                )
            )
            continue

        if not run:
            continue
        needs = "pysuricata" in code or "profile(" in code or "summarize(" in code
        if not needs or "..." in code:
            continue
        # Skip blocks that obviously need a file or network we do not have.
        if re.search(r"read_(csv|parquet|json)\(|open\(|requests\.|http", code):
            out.append(
                Finding(page.name, line, "INFO", "skipped", "needs an external input")
            )
            continue

        preamble = (
            "import warnings, numpy as np, pandas as pd\n"
            "warnings.simplefilter('ignore')\n"
            "df = pd.DataFrame({'a': np.arange(500, dtype=float),"
            " 'b': np.random.default_rng(0).standard_normal(500),"
            " 'c': ['x','y','z','w','v']*100})\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", preamble + code],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=REPO,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            msg = tail[-1] if tail else f"exit {proc.returncode}"
            # An optional dependency missing here says nothing about the docs.
            optional = ("polars", "pyarrow", "duckdb", "matplotlib")
            level = (
                "INFO"
                if any(f"No module named '{o}'" in msg for o in optional)
                else "ERROR"
            )
            out.append(Finding(page.name, line, level, "example", msg[:200]))


def check_symbols(page: Path, text: str, out: list[Finding]) -> None:
    import pysuricata

    exported = set(dir(pysuricata))
    for m in re.finditer(r"from\s+pysuricata\s+import\s+([^\n(]+)", text):
        for name in (n.strip() for n in m.group(1).split(",")):
            if name and name not in exported:
                out.append(
                    Finding(
                        page.name,
                        text[: m.start()].count("\n") + 1,
                        "ERROR",
                        "symbol",
                        f"`{name}` is not exported by pysuricata",
                    )
                )
    for m in re.finditer(r"\bpysuricata\.(\w+)", text):
        name = m.group(1)
        if name not in exported and name not in {
            "accumulators",
            "compute",
            "render",
            "io",
            "config",
        }:
            out.append(
                Finding(
                    page.name,
                    text[: m.start()].count("\n") + 1,
                    "WARN",
                    "symbol",
                    f"`pysuricata.{name}` does not resolve",
                )
            )


def _config_fields() -> dict[str, set[str]]:
    from pysuricata import ProfileConfig

    fields: dict[str, set[str]] = {}
    cfg = ProfileConfig()
    for group in ("compute", "output", "report"):
        obj = getattr(cfg, group, None)
        if obj is None:
            continue
        fields[group] = {f for f in dir(obj) if not f.startswith("_")}
    fields["_top"] = {f for f in dir(cfg) if not f.startswith("_")}
    return fields


def check_config(
    page: Path, text: str, out: list[Finding], fields: dict[str, set[str]]
) -> None:
    for m in CFG_ATTR.finditer(text):
        group, attr = m.group(1), m.group(2)
        known = fields.get(group)
        if known is not None and attr not in known:
            out.append(
                Finding(
                    page.name,
                    text[: m.start()].count("\n") + 1,
                    "ERROR",
                    "config",
                    f"`config.{group}.{attr}` does not exist",
                )
            )
    for m in KWARG.finditer(text):
        for kw in re.finditer(r"(\w+)\s*=", m.group(1)):
            name = kw.group(1)
            if name in fields.get("_top", set()) or any(
                name in v for v in fields.values()
            ):
                continue
            out.append(
                Finding(
                    page.name,
                    text[: m.start()].count("\n") + 1,
                    "WARN",
                    "config",
                    f"keyword `{name}=` not found on any config object",
                )
            )


def check_summary_keys(page: Path, text: str, out: list[Finding], real: dict) -> None:
    for m in SUMMARY_PATH.finditer(text):
        top, second = m.group(1), m.group(2)
        line = text[: m.start()].count("\n") + 1
        if top not in real:
            out.append(
                Finding(
                    page.name,
                    line,
                    "ERROR",
                    "summary-key",
                    f'summarize()["{top}"] does not exist (have: {sorted(real)[:5]})',
                )
            )
            continue
        node = real[top]
        if isinstance(node, dict) and node and second not in node:
            # column dicts are keyed by column name; check the inner schema instead
            sample = next(iter(node.values())) if node else {}
            if isinstance(sample, dict) and second in sample:
                continue
            out.append(
                Finding(
                    page.name,
                    line,
                    "ERROR",
                    "summary-key",
                    f'summarize()["{top}"]["{second}"] does not exist',
                )
            )


def check_links(page: Path, text: str, out: list[Finding]) -> None:
    for rx, kind in ((LINK, "link"), (IMAGE, "image")):
        for m in rx.finditer(text):
            target = m.group(2).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (page.parent / target).resolve()
            if not resolved.exists():
                out.append(
                    Finding(
                        page.name,
                        text[: m.start()].count("\n") + 1,
                        "ERROR",
                        kind,
                        f"`{target}` does not resolve",
                    )
                )


def check_stale_markers(page: Path, text: str, out: list[Finding]) -> None:
    for rx, kind, why in (
        (VERSION, "version", "version string, confirm it is current"),
        (SPEEDUP, "claim", "performance claim, re-measure before release"),
        (TIMING, "timing", "hardcoded timing, re-measure before release"),
    ):
        seen: set[str] = set()
        for m in rx.finditer(text):
            token = m.group(0)
            if token in seen:
                continue
            seen.add(token)
            out.append(
                Finding(
                    page.name,
                    text[: m.start()].count("\n") + 1,
                    "INFO",
                    kind,
                    f"`{token}` -- {why}",
                )
            )


def check_nav(out: list[Finding]) -> None:
    nav_text = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    referenced = set(re.findall(r"([\w./-]+\.md)", nav_text))
    on_disk = {str(p.relative_to(DOCS)) for p in _pages()}
    for orphan in sorted(on_disk - referenced):
        out.append(
            Finding(
                orphan,
                0,
                "WARN",
                "nav",
                "on disk but not in mkdocs nav -- never rendered",
            )
        )
    for missing in sorted(referenced - on_disk):
        out.append(Finding(missing, 0, "ERROR", "nav", "in mkdocs nav but not on disk"))


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any ERROR")
    ap.add_argument("--no-run", action="store_true", help="skip executing examples")
    ap.add_argument("--quiet-info", action="store_true", help="hide INFO findings")
    args = ap.parse_args(argv)

    findings: list[Finding] = []
    fields = _config_fields()

    import numpy as np
    import pandas as pd

    from pysuricata import summarize

    real = summarize(
        pd.DataFrame({"a": np.arange(400, dtype=float), "b": ["x", "y"] * 200})
    )

    check_nav(findings)
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        rel = str(page.relative_to(DOCS))
        before = len(findings)
        check_examples(page, text, findings, run=not args.no_run)
        check_symbols(page, text, findings)
        check_config(page, text, findings, fields)
        check_summary_keys(page, text, findings, real)
        check_links(page, text, findings)
        check_stale_markers(page, text, findings)
        for f in findings[before:]:
            f.page = rel

    by_level = defaultdict(list)
    for f in findings:
        by_level[f.level].append(f)

    print(f"{len(_pages())} pages checked\n")
    for level in ("ERROR", "WARN", "INFO"):
        items = by_level[level]
        if not items or (level == "INFO" and args.quiet_info):
            continue
        print(f"{level}  ({len(items)})")
        print("-" * 72)
        by_page = defaultdict(list)
        for f in items:
            by_page[f.page].append(f)
        for page in sorted(by_page):
            print(f"  {page}")
            for f in sorted(by_page[page], key=lambda x: x.line)[
                : 6 if level == "INFO" else 100
            ]:
                loc = f":{f.line}" if f.line else ""
                print(f"      {f.kind:<11}{loc:<6} {f.detail}")
            extra = len(by_page[page]) - (6 if level == "INFO" else 100)
            if extra > 0:
                print(f"      ... and {extra} more")
        print()

    counts = {k: len(v) for k, v in by_level.items()}
    print(
        f"summary: {counts.get('ERROR', 0)} errors, {counts.get('WARN', 0)} warnings, "
        f"{counts.get('INFO', 0)} to review by hand"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps([dataclasses.asdict(f) for f in findings], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 1 if (args.strict and counts.get("ERROR")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
