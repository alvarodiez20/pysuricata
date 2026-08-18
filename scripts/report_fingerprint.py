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
#: A structural boundary that a single label cannot contain. Matching one means
#: the non-greedy group backtracked past the end of its own cell and the "label"
#: is really two elements glued together.
_CROSSES_CELL = re.compile(r"</t[dhr]\s*>|</thead|<tbody", re.I)

_PAIR_PATTERNS = (
    # <th>Label</th><td>Value</td>  and  <td>Label</td><td>Value</td>
    re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>\s*<t[dh][^>]*>(.*?)</t[dh]>", re.S),
    # <div class="…__cap">Label</div><div class="…__val">Value</div>
    #
    # `(?:\s[^"]*)?` because the marker is a class *token*, not the end of the
    # attribute. An element that borrows a second class -- the calendar panel's
    # figure carries `cal-base__value vstat__val` -- is the same fact in the
    # same shape, and anchoring on the closing quote silently stopped matching
    # it. Two facts read as removed from a report that still displayed both.
    re.compile(
        r'<div class="[^"]*__cap(?:\s[^"]*)?"[^>]*>(.*?)</div>\s*'
        r'<div class="[^"]*__val(?:\s[^"]*)?"[^>]*>(.*?)</div>',
        re.S,
    ),
    # <span class="…__label">Label</span><span class="…__value">Value</span>
    #
    # The third shape, and added for the same reason as the second. #291 drew
    # the two datetime calendar shares as bars against a flat-calendar rule
    # instead of printing them as stat-row figures, and the extractor -- which
    # knew table cells and `.vstat` rows -- collected neither, so two facts the
    # report still displays looked *removed*. Over-fitting again; the fix is
    # here, not in the card.
    re.compile(
        r'<span class="[^"]*__label(?:\s[^"]*)?"[^>]*>(.*?)</span>\s*'
        r'<span class="[^"]*__value(?:\s[^"]*)?"[^>]*>(.*?)</span>',
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


#: `457 (51.3%)` -- a count and its share, which the report writes as one cell
#: on every stat row. Both halves are facts and neither is a bare number, so
#: without this the whole cell is discarded: the boolean card's `True` and
#: `False` counts were visible to a reader and invisible here.
_COUNT_AND_SHARE = re.compile(r"^(-?[\d,]+(?:\.\d+)?)\s*\(\s*(-?[\d.]+\s*%)\s*\)$")


def _split_value(value: str) -> list[str]:
    """The figures in one rendered cell. Usually one; sometimes two."""
    match = _COUNT_AND_SHARE.match(value)
    if match:
        return [match.group(1), match.group(2)]
    return [value] if _NUMBER.fullmatch(value) else []


def _is_run_dependent(label: str) -> bool:
    lowered = label.lower()
    return any(marker in lowered for marker in _RUN_DEPENDENT)


#: Where one column's card starts. Everything until the next one belongs to it.
_CARD = re.compile(r'<article class="var-card"[^>]*\sid="([^"]+)"')


def _regions(doc: str) -> list[tuple[str, str]]:
    """Split the document into ``(scope, html)``, one region per column card.

    Without this every card's statistics land on the same key: `age` and `fare`
    both emit a `Median` row, so `kv::median` held two different values, and
    :func:`diff` -- reading the fingerprint as a dict -- compared one of them
    and dropped the other. 559 collected facts collapsed to 251 checked ones,
    and `age`'s median could have changed without turning this red.
    """
    starts = [(m.start(), m.group(1)) for m in _CARD.finditer(doc)]
    if not starts:
        return [("", doc)]

    out = [("", doc[: starts[0][0]])]
    for index, (offset, scope) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(doc)
        out.append((scope, doc[offset:end]))
    return out


def _pairs_from_kv(doc: str) -> list[tuple[str, str]]:
    """Label/value pairs from the per-column statistics, scoped to their card.

    Matched on adjacency rather than on class names, because the migration
    replaces ``.kv`` tables with a stat row and the pairing is the only thing
    common to both shapes.
    """
    out: list[tuple[str, str]] = []
    for scope, region in _regions(doc):
        for pattern in _PAIR_PATTERNS:
            for m in pattern.finditer(region):
                # A label lives in one cell. `(.*?)` backtracks across closing
                # tags, so a header could swallow a whole row boundary and glue
                # itself to the first cell of the body: the sample table's
                # `<th>booked</th></tr></thead><tbody><tr><td>311</td>` matched
                # with a "label" of `booked 311` and a value of `56.0`.
                #
                # That label carries data -- 311 is a sampled row index -- so
                # the fact was keyed on the draw. Chunking changes which rows
                # the reservoir keeps, and because the *key* moved with the
                # value it registered as removed-plus-added rather than
                # changed, which reads as a fact vanishing from the report.
                # The alphabetic guard below could not catch it, since `booked`
                # supplies the letters.
                if _CROSSES_CELL.search(m.group(1)):
                    continue
                label, value = _text(m.group(1)), _text(m.group(2))
                if not label or not value or len(label) > 40:
                    continue
                # A label is a word. Two adjacent numbers are a row of the
                # sample table, not a statistic and its name --
                # `<td>0</td><td>79</td>` was being recorded as the fact
                # `kv::0 = 79`. Those rows are a random draw, so they made the
                # fingerprint differ between machines while looking like data
                # had changed.
                if not any(character.isalpha() for character in label):
                    continue
                figures = _split_value(value)
                if not figures:
                    continue
                if _is_run_dependent(label):
                    continue
                for figure in figures:
                    out.append((f"kv::{scope}::{label.lower()}", _canon_number(figure)))
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
    """Return the sorted fact list of a rendered report.

    Sorted, so it is diffable; **not** deduplicated, so multiplicity survives.
    A histogram asserts one count per bin and several bins legitimately hold
    the same number: collapsing those to a set made `[12, 12, 15]` and
    `[12, 15, 15]` the same fingerprint. Repetition is part of what the report
    claims, so it is part of what is compared.
    """
    facts: list[tuple[str, str]] = []
    for extractor in (
        _pairs_from_attrs,
        _pairs_from_kv,
        _pairs_from_columns,
        _pairs_from_flags,
    ):
        facts.extend(extractor(doc))
    return "\n".join(f"{k}\t{v}" for k, v in sorted(facts))


# --------------------------------------------------------------------------- #
# diffing
# --------------------------------------------------------------------------- #


def _index(fingerprint_text: str) -> dict[str, list[str]]:
    """key -> every value recorded under it, in file order."""
    out: dict[str, list[str]] = {}
    for line in fingerprint_text.splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        out.setdefault(key, []).append(value)
    return out


def diff(before: str, after: str) -> tuple[list[str], list[str], list[str]]:
    """Return (removed, added, changed) between two fingerprints.

    A key may legitimately carry several values -- one bar's ``data-count`` per
    bin, 64 of them on a single histogram -- so this compares the **multiset**
    under each key rather than reading the file into a dict.

    Reading it into a dict is what it used to do, and it silently discarded
    every duplicate: 559 collected facts became 251 compared ones, with the
    survivor picked by sort order. Under that comparator 63 of `age`'s 64 bin
    counts could change without a red test. The per-column scoping above
    removes most of the duplication; this removes the consequence of the rest.
    """
    b, a = _index(before), _index(after)

    removed: list[str] = []
    added: list[str] = []
    changed: list[str] = []

    for key in sorted(b.keys() | a.keys()):
        before_values = sorted(b.get(key, []))
        after_values = sorted(a.get(key, []))
        lost = _multiset_difference(before_values, after_values)
        gained = _multiset_difference(after_values, before_values)

        # A value that left paired with one that arrived under the same key is
        # a figure that changed; an unpaired remainder is a fact that vanished
        # or appeared.
        for old, new in zip(lost, gained, strict=False):
            changed.append(f"{key}\t{old} -> {new}")
        removed.extend(f"{key}\t{v}" for v in lost[len(gained) :])
        added.extend(f"{key}\t{v}" for v in gained[len(lost) :])

    return sorted(removed), sorted(added), sorted(changed)


def _multiset_difference(left: list[str], right: list[str]) -> list[str]:
    remaining = list(right)
    out = []
    for value in left:
        if value in remaining:
            remaining.remove(value)
        else:
            out.append(value)
    return out


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
