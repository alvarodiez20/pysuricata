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
import textwrap
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

# Indentation before the opening fence is captured so the body can be
# dedented by the same amount: fences nested in a pymdownx.tabbed block or a
# list item are indented, and feeding that to ast.parse reports every one of
# them as "unexpected indent" -- a property of the surrounding markdown, not
# of the code.
FENCE = re.compile(r"^([ \t]*)```(\w*)\n(.*?)^[ \t]*```", re.S | re.M)
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


# Planning documents, not documentation. They quote broken names on purpose
# (DOCS_PLAN) and propose APIs that do not exist yet (DIAGRAM_PROMPTS), so
# checking them against the current API reports the plan as the defect.
# Planning documents that live in docs/ for convenience but are not part of
# the published documentation: not in the nav, and their code fences
# illustrate test scaffolding rather than the public API.
_NOT_DOCUMENTATION = {
    "DOCS_PLAN.md",
    "DIAGRAM_PROMPTS.md",
    "MIGRATION_TESTING.md",
    "integration.md",
}


def _pages() -> list[Path]:
    """Every page this checks, `README.md` included.

    It was not included, and that is how the README came to claim a sketch `k`
    of 1,024 against a live 2,048, a sample of 10,000 against 20,000, and two
    CLI subcommands against three -- while every page under `docs/` stayed
    correct, because those were checked and it was not (#151).

    It is the most-read page in the project and the one PyPI renders as the
    package description, so it is the *last* file that should have been outside
    the net.
    """
    pages = [p for p in DOCS.rglob("*.md") if p.name not in _NOT_DOCUMENTATION]
    readme = REPO / "README.md"
    if readme.is_file():
        pages.append(readme)
    return sorted(pages)


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return text[:idx].count("\n") + 1 if idx >= 0 else 0


# ---------------------------------------------------------------------------


# Pages that share one DataFrame across their examples declare it once, in an
# admonition at the top. This pulls that block out so each fence can be run the
# way a reader following the page would actually run it.
SETUP_BLOCK = re.compile(
    r"""!!! info "Examples on this page assume.*?```python\n(.*?)```""", re.S
)


# The changelog is a record of past releases. Its snippets illustrate options as
# they were at the time, and some will legitimately no longer run; executing them
# would report history as a defect.
_NOT_RUNNABLE = {"changelog.md"}


def check_examples(page: Path, text: str, out: list[Finding], run: bool) -> None:
    """Execute python fences that look self-contained."""
    if page.name in _NOT_RUNNABLE:
        run = False
    setup_match = SETUP_BLOCK.search(text)
    setup = textwrap.dedent(setup_match.group(1)) if setup_match else ""
    for m in FENCE.finditer(text):
        lang, code = m.group(2), textwrap.dedent(m.group(3))
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
        if setup and code.strip() == setup.strip():
            continue
        # Skip blocks that obviously need a file or network we do not have.
        if re.search(
            r"(read|scan)_(csv|parquet|json|ndjson)\(|open\(|requests\.|http", code
        ):
            out.append(
                Finding(page.name, line, "INFO", "skipped", "needs an external input")
            )
            continue

        # Use the page's own stated setup block rather than injecting a df.
        # A synthetic frame made every fence pass while a reader pasting the
        # same code got NameError: df -- the checker was hiding the defect it
        # existed to find. A page that leans on a shared `df` must say so, and
        # what it says is what gets executed here.
        preamble = "import warnings\nwarnings.simplefilter('ignore')\n"
        if re.search(r"\bdf\b", code) and not re.search(r"^\s*df\s*=", code, re.M):
            if not setup:
                out.append(
                    Finding(
                        page.name,
                        line,
                        "ERROR",
                        "example",
                        "uses `df` but the page has no setup block defining it",
                    )
                )
                continue
            preamble += setup
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
    # "pysuricata.svg", "pysuricata.git", "pysuricata.md" are filenames inside
    # badge URLs and page links, not attribute access. Only look at dotted names
    # that are plausibly Python, and never inside a URL.
    _NOT_ATTRIBUTES = {
        "svg",
        "png",
        "ico",
        "git",
        "md",
        "html",
        "css",
        "js",
        "org",
        "com",
        "io",
    }
    for m in re.finditer(r"\bpysuricata\.(\w+)", text):
        name = m.group(1)
        prefix = text[max(0, m.start() - 120) : m.start()]
        if name in _NOT_ATTRIBUTES or "http" in prefix.rsplit("(", 1)[-1]:
            continue
        if name not in exported and name not in {
            "accumulators",
            "check",
            "compute",
            "config",
            "io",
            "progress",
            "render",
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
    if page.name in _NOT_RUNNABLE:
        # The changelog names options as they were at the time, including the
        # ones a release removed or renamed. Flagging those reports the record
        # of a fix as the bug it fixed.
        return
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
        if top == "columns":
            # The second key here is a column name from the example's own
            # DataFrame, not part of the API. Flagging it made every doc that
            # profiles a frame with domain column names look broken.
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
    # Strip comments first. The regex scans raw text, so a filename mentioned in
    # a YAML comment -- "# docs/changelog.md includes the root CHANGELOG.md" --
    # was read as a nav entry and reported as a missing page.
    nav_text = re.sub(r"(?m)#.*$", "", nav_text)
    referenced = set(re.findall(r"([\w./-]+\.md)", nav_text))
    # Only pages under `docs/`. This check asks whether mkdocs renders a page,
    # and `README.md` is not a site page -- it is the repository's front door
    # and PyPI's package description. It is checked for accuracy like every
    # other page, and it is correctly absent from the nav.
    on_disk = {str(p.relative_to(DOCS)) for p in _pages() if p.is_relative_to(DOCS)}
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
        # `README.md` sits at the repository root rather than under `docs/`, so
        # it is labelled by its path from there.
        rel = str(page.relative_to(DOCS if page.is_relative_to(DOCS) else REPO))
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
