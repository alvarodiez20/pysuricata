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
8. **Option defaults.** Every ``**`name: type = default`**`` heading and every
   row of a table with a Default column is resolved against
   ``dataclasses.fields()`` of ``ComputeOptions`` and ``RenderOptions``. Check 3
   verifies that a *name* resolves on a populated instance; this verifies that
   the field is declared and that the documented *default* is the real one.
9. **CLI flags.** ``cli.md``'s ``--flag`` tokens, per subcommand, against the
   options ``create_parser()`` actually defines -- in both directions.
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
# `render` was missing here for as long as this check has existed, and
# `output`/`report` are groups `ProfileConfig` has never had -- so two thirds of
# the pattern matched nothing and the third of the config that does exist went
# unchecked. That is how `config.render.include_sample` stayed documented on
# four pages while `RenderOptions` had two fields (#266).
CFG_ATTR = re.compile(r"(?:config|cfg)\.(compute|render)\.(\w+)")
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


# Working notes live under `docs/internal/`, excluded from the built site by
# `exclude_docs` in mkdocs.yml (#279). They quote broken names on purpose, propose
# APIs that do not exist yet, and illustrate test scaffolding rather than the
# public API -- checking them against the current API reports the plan as the
# defect. They used to sit beside the documentation behind a name-based
# allowlist here; a directory says the same thing where a reader can see it.
_NOT_DOCUMENTATION_DIR = "internal"


def _pages() -> list[Path]:
    return sorted(
        p
        for p in DOCS.rglob("*.md")
        if _NOT_DOCUMENTATION_DIR not in p.relative_to(DOCS).parts
    )


#: Documentation that does not live under `docs/`. The README is the page most
#: readers see first and it drifted furthest: at 0.0.62 it advertised a sketch
#: size, a sample size and a default chunk size that were all wrong, taught a
#: configuration ceremony removed in #87, and documented two of the three CLI
#: subcommands (#151). It was the only page outside this checker's reach, which
#: is most of why. Running its fences here is a smaller change than the rewrite
#: it protects.
_EXTRA_PAGES = (REPO / "README.md",)


def _checked_pages() -> list[Path]:
    """Everything the API checks run over: the docs tree plus the README."""
    return _pages() + [p for p in _EXTRA_PAGES if p.exists()]


def _page_label(page: Path) -> str:
    """`docs/`-relative for a docs page, repo-relative for anything else."""
    try:
        return str(page.relative_to(DOCS))
    except ValueError:
        return str(page.relative_to(REPO))


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

# The README is not an mkdocs page, so it cannot use the admonition above --
# `!!! info` renders as literal text on GitHub. This marker carries the same
# declaration in a form GitHub ignores, and the rule it enforces is unchanged:
# the fence it tags is *visible* to the reader, so a page leaning on a shared
# `df` still has to say what that `df` is. Hiding the setup inside the comment
# would recreate exactly the failure the note below describes.
SETUP_COMMENT = re.compile(r"<!--\s*docs-check:setup\s*-->\s*```python\n(.*?)```", re.S)


# The changelog is a record of past releases. Its snippets illustrate options as
# they were at the time, and some will legitimately no longer run; executing them
# would report history as a defect.
_NOT_RUNNABLE = {"changelog.md"}


def check_examples(page: Path, text: str, out: list[Finding], run: bool) -> None:
    """Execute python fences that look self-contained."""
    if page.name in _NOT_RUNNABLE:
        run = False
    setup_match = SETUP_BLOCK.search(text) or SETUP_COMMENT.search(text)
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
        # `stream_*` are the streaming readers in `pysuricata.sources`. They
        # need a file or a live relation exactly as `read_*` does, so they
        # belong in the same skip rather than failing on a missing path.
        # A path literal handed straight to the API needs that file just as much
        # as `read_parquet` does -- `profile("events.parquet")` is the shortest
        # way to show the streaming input and must not be reported as broken
        # code for want of a fixture.
        if re.search(
            r"(read|scan|stream)_(csv|parquet|json|ndjson|duckdb|arrow)\(|"
            r"(profile|summarize|check)\(\s*[\"']\S+\.(csv|parquet|json|arrow|feather|ipc)[\"']|"
            r"open\(|requests\.|http|duckdb\.",
            code,
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
    for group in ("compute", "render"):
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


#: The four per-kind key tables in `docs/summary-schema.md`, by the heading
#: that introduces each. This is the *contract* page -- #251 calls it the only
#: copy of "what each column kind reports" that is tied to the code, and this
#: check is what ties it.
_SCHEMA_SECTIONS = {
    "numeric": "Numeric columns",
    "categorical": "Categorical columns",
    "datetime": "Datetime columns",
    "boolean": "Boolean columns",
}


def _contract_frame():
    """One frame carrying all four column kinds.

    Deliberately not the checker's existing two-column frame: that one has no
    datetime and no boolean column, so half the contract could say anything at
    all and nothing would read it.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(list("abcde"), n),
            "boo": rng.random(n) > 0.5,
            "dt": pd.date_range("2024-01-01", periods=n, freq="h"),
        }
    )


def _documented_keys(body: str) -> set[str]:
    """Backticked identifiers in the **first** column of a table.

    The first column only: later columns carry prose that names other keys
    (`unique_est / count`, `true + false + missing`), and counting those as
    declarations would let a key be "documented" by being mentioned in someone
    else's note.
    """
    keys: set[str] = set()
    for row in re.findall(r"^\|(.+)$", body, flags=re.M):
        keys |= set(re.findall(r"`([a-z0-9_]+)`", row.split("|")[0]))
    return keys


def check_payload_contract(out: list[Finding]) -> None:
    """`summarize()`'s per-kind keys and the schema page must be the same set.

    #251 is about facts restated in several places drifting apart, and this is
    the instance with teeth: `docs/summary-schema.md` is a **contract**, so a
    key missing from it is a promise the payload makes and the documentation
    does not, while a key only in it is the `balance score` failure -- a
    statistic readers were told to expect that exists nowhere under
    `pysuricata/`, corrected in one copy at a time across three passes.

    Both directions, because they fail differently and neither is cosmetic:

    * a key in the payload and not in the table is **undocumented**. Three were,
      when this check was written: `unique_est_exact` on two kinds, and
      `singleton_levels` / `exact_levels`, which #297 added to the payload and
      to the card without adding to the contract.
    * a key in the table and not in the payload is a **ghost**. There are none
      today, and this is the half that keeps it that way.

    Keys documented before the first per-kind section -- `type`, `count`,
    `missing`, `mem_bytes` -- are stated once for every kind on purpose, and
    are not required to be repeated in each table.
    """
    from pysuricata import summarize

    page = DOCS / "summary-schema.md"
    if not page.exists():
        out.append(
            Finding(
                "summary-schema.md",
                0,
                "ERROR",
                "contract",
                "the payload contract page is missing",
            )
        )
        return

    text = page.read_text(encoding="utf-8")
    first = min(
        text.index(f"## {h}") for h in _SCHEMA_SECTIONS.values() if f"## {h}" in text
    )
    shared = set(re.findall(r"`([a-z0-9_]+)`", text[:first]))

    columns = summarize(_contract_frame())["columns"]
    by_kind = {c["type"]: c for c in columns.values()}
    everything = {k for col in by_kind.values() for k in col}

    for kind, heading in _SCHEMA_SECTIONS.items():
        marker = f"## {heading}"
        if marker not in text or kind not in by_kind:
            continue
        start = text.index(marker)
        nxt = text.find("\n## ", start + 1)
        body = text[start : nxt if nxt > 0 else len(text)]
        line = text[:start].count("\n") + 1

        documented = _documented_keys(body)
        actual = set(by_kind[kind])

        for key in sorted(actual - documented - shared):
            out.append(
                Finding(
                    "summary-schema.md",
                    line,
                    "ERROR",
                    "contract",
                    f"summarize() publishes `{key}` on {kind} columns and the "
                    f"contract does not document it",
                )
            )
        for key in sorted(documented - everything):
            out.append(
                Finding(
                    "summary-schema.md",
                    line,
                    "ERROR",
                    "contract",
                    f"the contract documents `{key}` under {kind} columns and "
                    f"no column kind publishes it",
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


#: `**`name: type = default`**`, which is how `configuration.md` heads each
#: option -- 22 of them. The type annotation is not checked: it is prose about a
#: union and gets written several ways, while the default is a literal that
#: either matches the dataclass or does not.
DOC_OPTION_HEADING = re.compile(
    r"^\*\*`([a-z_][a-z0-9_]*)\s*:\s*[^=`]+=\s*(.+?)`\*\*", re.M
)

#: A row of a table whose header carries a Default column. The group prefix is
#: captured rather than discarded: `compute.x` / `render.x` is an unambiguous
#: claim that `x` is a config field, while a bare first cell might be a CLI
#: flag, a payload key or a positional argument -- `cli.md` and `data-checks.md`
#: both have Default columns over things that are not options.
DOC_OPTION_ROW = re.compile(
    r"^\|\s*`(compute\.|render\.)?([a-z_][a-z0-9_]*)`\s*\|\s*`?([^|`]+?)`?\s*\|",
    re.M,
)


def _option_defaults() -> dict[str, object]:
    """Field name -> default, over the two public options dataclasses.

    `dataclasses.fields()` rather than `dir()` on an instance, which is the
    whole point: a plain dataclass has no slots, so a populated instance happily
    reports an attribute nobody declared. `dir()` cannot tell a real field from
    one the documentation invented, and that is precisely what #266 was.
    """
    from pysuricata import ComputeOptions, RenderOptions

    out: dict[str, object] = {}
    for cls in (ComputeOptions, RenderOptions):
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                out[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                out[f.name] = f.default_factory()  # type: ignore[misc]
    return out


def _as_literal(raw: str):
    """Parse a documented default, or return `None` if it is not a literal.

    Normalises the three ways the pages write a number -- `50_000`, `50000` and
    `50,000` -- so a thousands separator is not reported as a wrong default.
    """
    text = raw.strip().strip("`").strip()
    if not text:
        return None, False
    if re.fullmatch(r"-?[\d,_]+(?:\.\d+)?", text):
        text = text.replace(",", "").replace("_", "")
    try:
        return ast.literal_eval(text), True
    except (ValueError, SyntaxError):
        return None, False


def _table_sections_with_defaults(text: str) -> list[tuple[int, str]]:
    """Slices of `text` belonging to a table that has a Default column.

    A header is a row *followed by a separator row* whose cells include one
    called exactly "default". Both halves matter: matching any line containing
    the word reported a Guarantees table as a defaults table because one of its
    cells said "at the default `uniques_k=2048`".
    """
    spans: list[tuple[int, str]] = []
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    i = 0
    while i < len(lines) - 1:
        header, sep = lines[i], lines[i + 1]
        cells = [c.strip().lower() for c in header.strip().strip("|").split("|")]
        is_sep = bool(re.fullmatch(r"\s*\|[\s|:-]+\|\s*", sep))
        if header.lstrip().startswith("|") and is_sep and "default" in cells:
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            spans.append(
                (
                    starts[i],
                    text[starts[i] : starts[j] if j < len(lines) else len(text)],
                )
            )
            i = j
            continue
        i += 1
    return spans


def check_option_defaults(
    page: Path, text: str, out: list[Finding], defaults: dict[str, object]
) -> None:
    """Documented option names and defaults, against the real dataclasses.

    Three of the sixteen findings in the #266-#282 sweep were this: an option
    that does not exist, and two defaults that had moved. All three were
    mechanical, and none of them was caught -- so this is the ratchet under
    that correction (#284).
    """
    if page.name in _NOT_RUNNABLE:
        # The changelog names options as they were at the time. See check_config.
        return

    def report(
        name: str, documented: str, offset: int, *, declared: bool = True
    ) -> None:
        line = text[:offset].count("\n") + 1
        if name not in defaults:
            if not declared:
                # The row never claimed this was a config option. Saying nothing
                # is right: the alternative reported every CLI positional and
                # every threshold category as a missing field.
                return
            out.append(
                Finding(
                    page.name,
                    line,
                    "ERROR",
                    "option",
                    f"`{name}` is not a field of ComputeOptions or RenderOptions",
                )
            )
            return
        value, parsed = _as_literal(documented)
        if not parsed:
            return
        actual = defaults[name]
        if value != actual or type(value) is not type(actual):
            out.append(
                Finding(
                    page.name,
                    line,
                    "ERROR",
                    "option",
                    f"`{name}` documented as {value!r}, actual default {actual!r}",
                )
            )

    for m in DOC_OPTION_HEADING.finditer(text):
        report(m.group(1), m.group(2), m.start())

    for start, block in _table_sections_with_defaults(text):
        for m in DOC_OPTION_ROW.finditer(block):
            prefix, name, documented = m.group(1), m.group(2), m.group(3).strip()
            # A prose cell is a description, not a default. Only rows whose
            # second cell is a literal are making a claim this can check.
            if not _as_literal(documented)[1] and name in defaults:
                continue
            report(name, documented, start + m.start(), declared=bool(prefix))


def check_cli_flags(out: list[Finding]) -> None:
    """`cli.md`'s flags against the flags `create_parser()` actually defines.

    The page transcribes 31 options across three subcommands, and its whole
    value is being exhaustive -- so the first flag anyone adds or renames makes
    it wrong, silently, unless something pairs the two (#284).
    """
    page = DOCS / "cli.md"
    if not page.exists():
        return
    try:
        from pysuricata.cli import create_parser
    except Exception as e:  # pragma: no cover - import guard
        out.append(Finding("cli.md", 0, "WARN", "cli", f"cannot import parser: {e}"))
        return

    parser = create_parser()
    subparsers: dict[str, object] = {}
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            subparsers.update(choices)

    text = page.read_text(encoding="utf-8")
    # One slice of the page per `## subcommand` heading, so a flag documented
    # under `profile` is not credited to `check`.
    sections: dict[str, tuple[int, str]] = {}
    headings = list(re.finditer(r"^##\s+`?(\w+)`?\s*$", text, re.M))
    for i, m in enumerate(headings):
        name = m.group(1)
        if name not in subparsers:
            continue
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections[name] = (m.start(), text[m.start() : end])

    for name, sub in sorted(subparsers.items()):
        real = {
            opt
            for action in sub._actions  # type: ignore[attr-defined]
            for opt in action.option_strings
            if opt.startswith("--")
        }
        if name not in sections:
            out.append(
                Finding(
                    "cli.md",
                    0,
                    "WARN",
                    "cli",
                    f"subcommand `{name}` has no `## {name}` section",
                )
            )
            continue
        start, body = sections[name]
        documented = set(re.findall(r"(--[a-z][a-z0-9-]*)", body))
        for flag in sorted(documented - real - {"--help"}):
            out.append(
                Finding(
                    "cli.md",
                    text[:start].count("\n") + 1,
                    "ERROR",
                    "cli",
                    f"`{flag}` is documented under `{name}` but the parser has no such flag",
                )
            )
        for flag in sorted(real - documented - {"--help"}):
            out.append(
                Finding(
                    "cli.md",
                    text[:start].count("\n") + 1,
                    "WARN",
                    "cli",
                    f"`{name} {flag}` exists but is not documented",
                )
            )


def check_nav(out: list[Finding]) -> None:
    nav_text = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    # Strip comments first. The regex scans raw text, so a filename mentioned in
    # a YAML comment -- "# docs/changelog.md includes the root CHANGELOG.md" --
    # was read as a nav entry and reported as a missing page.
    nav_text = re.sub(r"(?m)#.*$", "", nav_text)
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
    defaults = _option_defaults()

    import numpy as np
    import pandas as pd

    from pysuricata import summarize

    real = summarize(
        pd.DataFrame({"a": np.arange(400, dtype=float), "b": ["x", "y"] * 200})
    )

    # Nav coverage stays scoped to `docs/`: the README is deliberately not an
    # mkdocs page, and including it here would report that as an orphan.
    check_nav(findings)
    check_cli_flags(findings)
    check_payload_contract(findings)
    for page in _checked_pages():
        text = page.read_text(encoding="utf-8")
        rel = _page_label(page)
        before = len(findings)
        check_examples(page, text, findings, run=not args.no_run)
        check_symbols(page, text, findings)
        check_config(page, text, findings, fields)
        check_option_defaults(page, text, findings, defaults)
        check_summary_keys(page, text, findings, real)
        check_links(page, text, findings)
        check_stale_markers(page, text, findings)
        for f in findings[before:]:
            f.page = rel

    by_level = defaultdict(list)
    for f in findings:
        by_level[f.level].append(f)

    print(f"{len(_checked_pages())} pages checked\n")
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
