"""Reduce a rendered report to the set of facts it asserts.

The UI migration rewrites every template, stylesheet and card renderer across
seventeen commits. An HTML snapshot test is worthless against that: the diff is
100% churn on every commit, so nobody reads it, so a real regression rides in
unnoticed. This module extracts the *numbers* instead, discarding every trace of
how they are presented, and produces a stable, sorted, diffable fingerprint.

The contract this enforces:

    Presentation may change on every commit of the migration.
    The facts may not, except at the two commits that deliberately change them.

Those two are the KMV clamp (``unique`` may only ever decrease, and only to the
row count) and the correlations change (below-threshold pairs become visible).
Everything else in a seventeen-commit rewrite of the render layer must leave
this file byte-identical.

Usage
-----
    # once, before the migration starts, on a report built from correct numbers
    python scripts/report_fingerprint.py --write tests/fixtures/fingerprint.txt

    # in CI on every commit
    pytest tests/test_report_data_invariance.py

Deliberately *not* captured: colours, class names, element order, tag names,
whitespace, SVG geometry, ids, ARIA text. If a change to any of those alters
this fingerprint, the extractor is over-fitted to the old markup and should be
loosened -- that judgement call is the price of the technique and it is much
cheaper than reviewing seventeen full-document diffs.
"""

from __future__ import annotations

import argparse
import html as htmllib
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# 1,234  |  1234.5  |  -0.004  |  1.2e+04  |  77.1%  |  121 KB
_NUMBER = re.compile(
    r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?\s*(?:%|KB|MB|GB|B|s|ms)?",
)


def _text(fragment: str) -> str:
    return _WS.sub(" ", htmllib.unescape(_TAG.sub(" ", fragment))).strip()


def _canon_number(raw: str) -> str:
    """Canonicalise a rendered figure so formatting changes do not register.

    ``1,234`` / ``1234`` / ``1.234e+03`` all reduce to the same token, because
    the migration explicitly reformats figures (thousands separators replace
    scientific notation) and that must not read as a data change.
    """
    s = raw.strip()
    unit = ""
    m = re.search(r"(%|KB|MB|GB|B|ms|s)$", s)
    if m:
        unit = m.group(1)
        s = s[: m.start()].strip()
    s = s.replace(",", "").replace("−", "-").replace("–", "-")
    try:
        v = float(s)
    except ValueError:
        return f"{raw.strip()}"
    # Twelve significant figures: enough that a real change shows, loose enough
    # that a float repr difference between Python versions does not.
    return f"{v:.12g}{unit}"


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def _pairs_from_attrs(doc: str) -> list[tuple[str, str]]:
    """Facts the renderer already tags in the DOM.

    ``data-count`` / ``data-pct`` / ``data-value`` / ``data-threshold`` are
    emitted next to the element they describe and survive any restyling, which
    makes them the most durable hooks in the document. ``data-col`` scopes them.
    """
    out: list[tuple[str, str]] = []
    for el in re.finditer(r"<[a-zA-Z][^>]*\sdata-[a-z-]+=[^>]*>", doc):
        tag = el.group(0)
        attrs = dict(re.findall(r'(data-[a-z-]+)="([^"]*)"', tag))
        scope = (
            attrs.get("data-col")
            or attrs.get("data-name")
            or attrs.get("data-label")
            or ""
        )
        for key in (
            "count",
            "pct",
            "value",
            "threshold",
            "percentage",
            "missing",
            "chunk",
        ):
            k = f"data-{key}"
            if k in attrs:
                out.append((f"attr::{scope}::{key}", _canon_number(attrs[k])))
    return out


# Adjacent label/value pairs, in either shape the report uses. Both reduce to
# the same key, so a statistic that moves from a table cell to a stat row keeps
# its identity in the fingerprint.
#
# The second pattern was added when #114 restacked the numeric card: the two
# `.kv` tables became a `<div class="vstat">` row, and the extractor -- which
# only knew about table cells -- reported `max` and `median` as *removed* from
# a report that still displayed both. That is the over-fitting this module's
# docstring warns about, and the fix is here rather than in the card.
_PAIR_PATTERNS = (
    # <th>Label</th><td>Value</td>  and  <td>Label</td><td>Value</td>
    re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>\s*<t[dh][^>]*>(.*?)</t[dh]>", re.S),
    # <div class="…__cap">Label</div><div class="…__val">Value</div>
    re.compile(
        r'<div class="[^"]*__cap"[^>]*>(.*?)</div>\s*'
        r'<div class="[^"]*__val"[^>]*>(.*?)</div>',
        re.S,
    ),
)


# Values that measure the *run* rather than the data: wall-clock duration and
# the generation timestamp. They differ between two renders of the same frame
# -- the elapsed figure moved from 0.02s to 0.04s simply by running the suite
# under load -- so including them makes every comparison flaky and trains
# whoever sees it to re-baseline on red, which is the one habit this file
# exists to prevent.
_RUN_DEPENDENT = (
    "elapsed",
    "generated",
    "profiled in",
    "duration",
    # Memory too. A column of a few repeated short strings -- "male"/"female",
    # "C85"/"B42" -- measures differently depending on whether those exact
    # string objects already exist in the process, because an object array
    # stores pointers and the accounting walks unique objects. Two runs of the
    # same frame in the same suite disagreed by 160 bytes for that reason. It
    # is a property of the process, not of the data.
    "processed",
    "memory",
)


def _is_run_dependent(label: str) -> bool:
    lowered = label.lower()
    return any(marker in lowered for marker in _RUN_DEPENDENT)


def _pairs_from_kv(doc: str) -> list[tuple[str, str]]:
    """Label/value pairs from the per-column statistics.

    Matched on adjacency rather than on class names, because the migration
    replaces ``.kv`` tables with a stat row and the pairing is the only thing
    common to both shapes.
    """
    out: list[tuple[str, str]] = []
    for pattern in _PAIR_PATTERNS:
        for m in pattern.finditer(doc):
            label, value = _text(m.group(1)), _text(m.group(2))
            if not label or not value or len(label) > 40:
                continue
            # A label is a word. Two adjacent numbers are a row of the sample
            # table, not a statistic and its name -- `<td>0</td><td>79</td>`
            # was being recorded as the fact `kv::0 = 79`. Those rows are a
            # random draw, so they made the fingerprint differ between machines
            # while looking like data had changed.
            if not any(character.isalpha() for character in label):
                continue
            if not _NUMBER.fullmatch(value):
                continue
            if _is_run_dependent(label):
                continue
            out.append((f"kv::{label.lower()}", _canon_number(value)))
    return out


def _pairs_from_columns(doc: str) -> list[tuple[str, str]]:
    """Per-column type and dtype, which must survive every layout change."""
    out: list[tuple[str, str]] = []
    for m in re.finditer(r'data-col="([^"]+)"[^>]*data-type="([^"]+)"', doc):
        out.append((f"type::{m.group(1)}", m.group(2)))
    for m in re.finditer(r'data-type="([^"]+)"[^>]*data-col="([^"]+)"', doc):
        out.append((f"type::{m.group(2)}", m.group(1)))
    return out


def _pairs_from_flags(doc: str) -> list[tuple[str, str]]:
    """Quality flags, sorted, per column.

    Order is presentation; membership is a fact. Phase 5.7 changes how a flag is
    *displayed* (value against threshold instead of a bare word) but must not
    change which flags fire.
    """
    out: list[tuple[str, str]] = []
    for m in re.finditer(r'data-col="([^"]+)"[^>]*data-flags="([^"]*)"', doc):
        flags = sorted(f.strip().lower() for f in m.group(2).split(",") if f.strip())
        out.append((f"flags::{m.group(1)}", "|".join(flags)))
    return out


def fingerprint(doc: str) -> str:
    """Return the sorted, deduplicated fact set of a rendered report."""
    facts: set[tuple[str, str]] = set()
    for extractor in (
        _pairs_from_attrs,
        _pairs_from_kv,
        _pairs_from_columns,
        _pairs_from_flags,
    ):
        facts.update(extractor(doc))
    return "\n".join(f"{k}\t{v}" for k, v in sorted(facts))


# --------------------------------------------------------------------------- #
# diffing
# --------------------------------------------------------------------------- #


def diff(before: str, after: str) -> tuple[list[str], list[str], list[str]]:
    """Return (removed, added, changed) between two fingerprints."""
    b = dict(line.split("\t", 1) for line in before.splitlines() if "\t" in line)
    a = dict(line.split("\t", 1) for line in after.splitlines() if "\t" in line)
    removed = sorted(f"{k}\t{b[k]}" for k in b.keys() - a.keys())
    added = sorted(f"{k}\t{a[k]}" for k in a.keys() - b.keys())
    changed = sorted(
        f"{k}\t{b[k]} -> {a[k]}" for k in b.keys() & a.keys() if b[k] != a[k]
    )
    return removed, added, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", nargs="?", help="rendered report HTML")
    ap.add_argument("--write", metavar="PATH", help="write the fingerprint here")
    ap.add_argument("--compare", metavar="PATH", help="diff against this fingerprint")
    args = ap.parse_args()

    if not args.report:
        ap.error("a report path is required")
    fp = fingerprint(Path(args.report).read_text(encoding="utf-8"))

    if args.write:
        Path(args.write).write_text(fp + "\n", encoding="utf-8")
        print(f"wrote {args.write} — {len(fp.splitlines())} facts")

    if args.compare:
        removed, added, changed = diff(
            Path(args.compare).read_text(encoding="utf-8"), fp
        )
        for title, rows in (
            ("REMOVED", removed),
            ("ADDED", added),
            ("CHANGED", changed),
        ):
            if rows:
                print(f"\n{title} ({len(rows)})")
                for r in rows[:40]:
                    print("  " + r)
                if len(rows) > 40:
                    print(f"  … and {len(rows) - 40} more")
        return 1 if (removed or changed) else 0

    if not args.write:
        print(fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
