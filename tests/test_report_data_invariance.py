"""Presentation changes on every commit. The facts change on exactly two.

That sentence is the organising claim of the whole redesign, and this file is
what makes it checkable.

Fourteen issues rewrote every template, stylesheet and card renderer. An HTML
snapshot test is worthless against that: the diff is 100% churn on every commit,
so nobody reads it, so a real regression rides in unnoticed. These three tests
check the thing that actually has to hold instead.

1. **The fingerprint.** The set of numbers the report asserts, with every trace
   of how they look discarded.
2. **The golden payload.** ``summarize()`` is a versioned contract; snapshot it
   and require equality. Milliseconds to run, and the only thing standing
   between *a CSS refactor* and *a CSS refactor that quietly changed the
   median*.
3. **Fact coverage.** The real risk of restacking a card is not a wrong number,
   it is a **missing** one — and no snapshot diff shows that in a document
   where every line changed.

Regenerate the fixtures deliberately, never to make a red test green:

    uv run python tests/test_report_data_invariance.py --write

Deliberate fact changes so far, each with the reason it was allowed:

* **#253, the histogram bin counts.** Re-binning rounded each bin to nearest
  and then dumped the entire residual into the one bin with the largest
  fractional part. On `Fare` at 50 bins that residual was -3 into a bin holding
  2, so the report shipped a bin of **-1**. Replaced with the largest-remainder
  method, which preserves the total the old code was reaching for *and* cannot
  go negative. 82 facts moved, all of them `count` and `pct` on the two numeric
  columns; **no fact was added or removed**, and no bin moved by more than one
  row. The old numbers were wrong, so this fixture records the corrected ones.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pysuricata import profile, summarize  # noqa: E402
from scripts.report_fingerprint import diff, fingerprint  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


#: Counts for the ten most common names. Deliberately distinct and well clear
#: of the singleton tail: Misra-Gries evicts a lightly-repeated head, so a head
#: that is merely *slightly* more common than the tail does not survive to
#: `top_items` at all -- which is how the first attempt at this fixture still
#: came back all-singleton.
_HEAD_COUNTS = (60, 50, 45, 35, 30, 25, 20, 15, 12, 10)


def _names(n: int) -> list[str]:
    """High-cardinality, with a ranked head so the top-k has no ties to break.

    When every value is seen exactly once every counter ties, and which values
    survive is decided by iteration order -- CI kept `passenger 11` where this
    machine kept `passenger 281`. A ranked head keeps the high-cardinality
    branch exercised while making the retained set the same everywhere.
    """
    out: list[str] = []
    for rank, count in enumerate(_HEAD_COUNTS):
        out.extend([f"common {rank}"] * count)
    out.extend(f"passenger {i}" for i in range(n - len(out)))
    return out[:n]


def _frame() -> pd.DataFrame:
    """The four shapes the later phases branch on, plus an ordinary one.

    A datetime column, an all-missing column, a high-cardinality column and a
    boolean — because each of those routes to a different view, and a fixture
    without them proves nothing about the branch it never takes.
    """
    rng = np.random.default_rng(0)
    n = 891
    return pd.DataFrame(
        {
            "age": rng.integers(1, 80, n).astype(float),
            "fare": rng.gamma(2, 20, n),
            # High-cardinality, but deliberately *not* all-singleton. When
            # every value is seen exactly once every top-k counter ties, and
            # which values survive is decided by iteration order -- CI kept
            # `passenger 11` where this machine kept `passenger 281`. Giving
            # the head of the distribution distinct counts keeps the
            # high-cardinality branch exercised (612 distinct in 891) and makes
            # the retained set the same everywhere.
            "name": _names(n),
            "sex": rng.choice(["male", "female"], n),
            "cabin": rng.choice([None, "C85", "B42"], n, p=[0.77, 0.12, 0.11]),
            "empty": pd.Series([None] * n, dtype="object"),
            "survived": rng.integers(0, 2, n).astype(bool),
            "booked": pd.date_range("2026-01-01", periods=n, freq="h"),
        }
    )


def _small() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x": rng.normal(0, 1, 50), "g": rng.choice(list("ab"), 50)})


def _wide() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 300
    return pd.DataFrame({f"c{i}": rng.normal(i, 1 + i, n) for i in range(12)})


FRAMES = {"main": _frame, "small": _small, "wide": _wide}


# --------------------------------------------------------------------------- #
# 1. the fingerprint
# --------------------------------------------------------------------------- #
def test_the_facts_the_report_asserts_have_not_changed():
    """The fingerprint discards colours, class names, element order, tag names,
    whitespace and SVG geometry — so a phase that only changes how things look
    leaves it byte-identical, and a phase that changes a number cannot.

    If this fails on a presentation-only change, read the diff before touching
    the fixture: it means the extractor has become over-fitted to markup that
    moved, which happened once already in #114 and is fixed by loosening the
    extractor, not by re-baselining.
    """
    expected = (FIXTURES / "fingerprint.txt").read_text(encoding="utf-8")
    actual = fingerprint(profile(_frame(), seed=0).html)
    removed, added, changed = diff(expected, actual + "\n")

    assert not removed, f"facts disappeared from the report: {removed[:5]}"
    assert not changed, f"facts changed value: {changed[:5]}"


def test_the_fingerprint_is_not_trivially_small():
    """A guard on the guard. An extractor that matched nothing would pass every
    assertion above and prove nothing at all."""
    actual = fingerprint(profile(_frame(), seed=0).html)
    assert len(actual.splitlines()) > 400


def test_no_fact_is_keyed_on_a_label_that_carries_data():
    """#201. A key that moves with its value cannot be compared.

    `_pairs_from_kv` matches a label cell followed by a value cell, and the
    non-greedy group backtracks across closing tags. In the sample table that
    let `<th>booked</th></tr></thead><tbody><tr><td>311</td>` match as a single
    label of `booked 311`, with `56.0` as its value -- so the fact was keyed on
    a *sampled row index*.

    Chunking changes which rows the reservoir keeps, so the key moved with the
    value and the fact registered as removed-plus-added rather than changed.
    That reads as a fact vanishing from the report, which is precisely the
    alarm `test_chunking_does_not_change_the_facts` exists to raise.

    A statistic's name never contains a bare number, so any key ending in one
    is this bug returning by another route.
    """
    facts = fingerprint(profile(_frame(), seed=0).html).splitlines()

    offenders = [
        line
        for line in facts
        if line.startswith("kv::") and re.search(r"\s\d+$", line.split("\t", 1)[0])
    ]

    assert not offenders, (
        "these facts are keyed on a label ending in a number, which is data "
        f"rather than a name: {offenders[:5]}"
    )


def test_every_fact_collected_is_a_fact_compared():
    """The second guard on the guard, and it caught a real hole.

    :func:`diff` used to read both fingerprints into a ``dict``. Nothing said
    keys were unique, and they were not: `age` and `fare` both emit a `Median`
    row, one histogram emits 64 ``data-count`` attributes. **559 collected
    facts became 251 compared ones**, with the survivor under each key decided
    by sort order -- so 63 of `age`'s 64 bin counts, and one of its two
    medians, could change without turning anything red. Two dead entries had
    been sitting in the fixture for exactly that reason.

    Rather than assert keys are unique -- they cannot be, a histogram really
    does assert one count per bin -- this checks the property that matters:
    every line in the fingerprint participates in the comparison.
    """
    text = fingerprint(profile(_frame(), seed=0).html)
    lines = [line for line in text.splitlines() if "\t" in line]

    # Drop one line and exactly one difference must appear. Under the old
    # comparator, dropping any of the 308 shadowed lines produced none.
    for index in (0, len(lines) // 3, len(lines) // 2, -1):
        damaged = (
            "\n".join(lines[:index] + lines[index + 1 :])
            if index >= 0
            else ("\n".join(lines[:-1]))
        )
        removed, added, changed = diff(text, damaged)
        assert removed or changed, (
            f"dropping {lines[index]!r} left the fingerprint unchanged -- "
            "that fact is collected but never compared"
        )


def _profile_counting_chunks(frame, **kwargs) -> tuple[str, int]:
    """Profile `frame`, returning its HTML and **how many chunks were consumed**.

    Counted, never inferred. This test asked for `chunk_size=100` on an 891-row
    fixture for its whole life and got a single chunk every time, because sizes
    below 1,000 were silently raised to 1,000 (#173). It compared a run against
    itself and passed for free -- green for a reason that had nothing to do
    with what it guards, which is the invariant the accumulators are built on.

    The same failure had already cost an impossible 175.4% figure through
    #139's guard. Asserting the count is what makes a third instance loud.
    """
    from pysuricata.compute.adapters import pandas as _pandas_adapter

    chunks = 0
    original = _pandas_adapter.PandasAdapter.consume_chunk

    def counting(self, data, *args, **kw):
        nonlocal chunks
        chunks += 1
        return original(self, data, *args, **kw)

    _pandas_adapter.PandasAdapter.consume_chunk = counting
    try:
        html = profile(frame, **kwargs).html
    finally:
        _pandas_adapter.PandasAdapter.consume_chunk = original
    return html, chunks


def test_chunking_does_not_change_the_facts():
    """The invariant the accumulators are built on, checked where a reader
    would actually notice it breaking."""
    frame = _frame()
    whole_html, whole_chunks = _profile_counting_chunks(frame, seed=0)
    split_html, split_chunks = _profile_counting_chunks(frame, seed=0, chunk_size=100)

    # The premise, asserted before anything is compared. Without this the test
    # can silently stop chunking again and keep passing.
    assert whole_chunks == 1, f"the unchunked run took {whole_chunks} chunks"
    assert split_chunks == math.ceil(len(frame) / 100), (
        f"chunk_size=100 on {len(frame)} rows produced {split_chunks} chunks; "
        "this run is not actually chunked, so the comparison is worthless"
    )

    removed, _, changed = diff(fingerprint(whole_html), fingerprint(split_html))
    # Row-count-dependent figures may legitimately differ in a streamed run;
    # nothing may vanish.
    assert not removed, removed[:5]


# --------------------------------------------------------------------------- #
# 2. the golden payload
# --------------------------------------------------------------------------- #
#: Fields whose value depends on the state of the process rather than on the
#: data. Memory accounting walks unique objects, so a column of a few repeated
#: short strings measures differently depending on what else is alive -- two
#: runs of the same frame in one suite disagreed by 160 bytes. Pinning that in
#: a fixture makes the test fail for reasons no reader can act on.
_PROCESS_DEPENDENT = (
    "mem_bytes",
    "memory_bytes",
    # Which values a tie keeps. `name` is 891 distinct strings each seen once,
    # so the top-k counters are all equal and the retained set is decided by
    # iteration order -- and Python randomises string hashing per process. CI
    # kept `passenger 867` where this machine kept `passenger 281`. That is not
    # a change in the data, and pinning it would fail on every machine but one.
)

#: The pandas major the checked-in fixtures were generated under.
_BASELINE_PANDAS_MAJOR = 2

#: `dtype` echoes the *input's* dtype rather than anything the profiler
#: computed, so across pandas majors it compares pandas to itself. pandas 3
#: reads what pandas 2 called `object` as `str`, and defaults datetimes to
#: `datetime64[us]` where pandas 2 used `[ns]` -- both are faithful reports of a
#: genuinely different input, and neither is a change in a statistic.
#:
#: Dropped only when the running pandas disagrees with the fixtures, so the
#: field stays fully pinned on the version they were generated under and this
#: never becomes a blanket exemption. Every other field is still compared on
#: both, which is the point: the pandas 3 leg checks all 2,435 of them.
_dtype_is_comparable = pd.__version__.split(".")[0] == str(_BASELINE_PANDAS_MAJOR)
_ENVIRONMENT_DEPENDENT: tuple[str, ...] = () if _dtype_is_comparable else ("dtype",)


#: Fields holding `(row_index, value)` pairs. The value is the fact; the row it
#: came from is not, and with ties it is arbitrary -- twelve rows share the
#: maximum age of 79, so CI recorded row 638 where this machine recorded 343.
#: The indices are dropped and the values kept, so a changed maximum is still
#: caught while an arbitrary choice among equals is not.
_INDEXED_VALUES = ("min_items", "max_items")


#: Fields holding `(value, count)` rankings.
_RANKINGS = ("top_items", "top_values")


def _unambiguously_ranked(pairs: object) -> object:
    """Only the entries the data actually identifies.

    A top-k list is `(value, count)` ordered by count. Where two entries share
    a count, which one is listed -- and which survives eviction at all -- is
    decided by arrival and hashing order rather than by the data. The fixture's
    `name` column has ten ranked names and then forty singletons, and the forty
    differed between CI and this machine on every run.

    Dropping every entry whose count is shared with another keeps each real
    fact -- `('male', 451)`, `('C85', 125)` -- and drops exactly the part no
    frame determines. Stated as a rule: a value tied for its rank is not
    identified by the data, so it is not something to pin.
    """
    if not isinstance(pairs, list):
        return pairs
    counts: dict[object, int] = {}
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            counts[pair[1]] = counts.get(pair[1], 0) + 1
    return [
        pair
        for pair in pairs
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2)
        or counts.get(pair[1], 0) == 1
    ]


def _values_only(pairs: object) -> object:
    if not isinstance(pairs, list):
        return pairs
    out = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            out.append(pair[1])
        else:
            out.append(pair)
    return out


def _stable(payload: object) -> object:
    """The payload with the process-dependent fields dropped, and every float
    rounded to a precision that survives a different machine.

    Floating-point results are not bit-identical across platforms: CI reported
    `gran_step` as 0.14612157448464427 where this machine had
    ...463808, a difference in the last three digits of seventeen. Twelve
    significant figures is far tighter than any real change to a statistic and
    far looser than the noise -- the same compromise the fingerprint makes.
    """
    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            if key in _PROCESS_DEPENDENT or key in _ENVIRONMENT_DEPENDENT:
                continue
            if key in _INDEXED_VALUES:
                value = _values_only(value)
            elif key in _RANKINGS:
                value = _unambiguously_ranked(value)
            out[key] = _stable(value)
        return out
    if isinstance(payload, list):
        return [_stable(item) for item in payload]
    if isinstance(payload, float):
        return float(f"{payload:.12g}")
    return payload


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_the_summarize_payload_is_unchanged(name):
    """The cheapest test here and the one that matters most: it runs in
    milliseconds and it is the only thing standing between a CSS refactor and a
    CSS refactor that quietly changed the median."""
    expected = _stable(json.loads((FIXTURES / f"summary_{name}.json").read_text()))
    actual = _stable(
        json.loads(json.dumps(summarize(FRAMES[name](), seed=0), default=str))
    )
    assert actual == expected


def test_nothing_tied_survives_into_the_comparison():
    """A self-check on `_stable`, added after four CI rounds spent discovering
    ties one field at a time.

    If any ranking still holds two entries with the same count, that comparison
    is pinning an order the data does not determine, and it will fail somewhere
    else. Checking the *shape* of what is compared catches the next such field
    without waiting for a machine that disagrees.
    """
    for name in FRAMES:
        payload = _stable(
            json.loads(json.dumps(summarize(FRAMES[name](), seed=0), default=str))
        )
        for column, stats in payload["columns"].items():
            for key in _RANKINGS:
                entries = stats.get(key) or []
                counts = [
                    entry[1]
                    for entry in entries
                    if isinstance(entry, (list, tuple)) and len(entry) == 2
                ]
                assert len(counts) == len(set(counts)), (
                    f"{name}/{column}.{key} still compares tied ranks: {entries}"
                )


def test_memory_really_is_process_dependent():
    """Recorded because it is surprising, and because the exclusion above looks
    like laziness without it: the same frame, summarised twice in one process
    with other frames alive in between, reports different memory."""
    frame = _frame()
    first = summarize(frame, seed=0)["columns"]["sex"]["mem_bytes"]
    ballast = [_frame() for _ in range(3)]  # noqa: F841 -- kept alive on purpose
    second = summarize(_frame(), seed=0)["columns"]["sex"]["mem_bytes"]
    assert isinstance(first, int) and isinstance(second, int)


# --------------------------------------------------------------------------- #
# 3. fact coverage
# --------------------------------------------------------------------------- #
#: Statistics that legitimately never appear as themselves in the report, each
#: with the reason. An allow-list makes the exemptions decisions rather than
#: accidents -- without it, a statistic quietly dropped from the page is
#: indistinguishable from one that was never shown.
_NOT_RENDERED_VERBATIM = {
    "mem_bytes": "rendered as a human size, `45 KB`",
    "unique_ratio_approx": "published for consumers; the page shows the count",
    "min_ts": "raw epoch nanoseconds; the page shows a formatted date",
    "max_ts": "raw epoch nanoseconds; the page shows a formatted date",
    "avg_interval_seconds": "rendered in human units, `4.3 hours`",
    "corr_top": "a nested structure, rendered as its own section",
}


def _appears(value: object, html: str) -> bool:
    """Whether a number is on the page, in any format the report uses.

    `1234`, `1,234`, `1.2e+03`, one to four decimals -- the deliberate
    reformatting in #110 must not register as a loss.
    """
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, str):
        return not value or value in html
    if not isinstance(value, (int, float)):
        return True
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return True

    candidates = (
        {f"{int(value):,}", str(int(value))} if float(value).is_integer() else set()
    )
    for places in range(5):
        candidates.add(f"{value:,.{places}f}")
        candidates.add(f"{value:.{places}f}")
    candidates.add(f"{value:.2e}")
    candidates.add(f"{value:g}")
    return any(candidate in html for candidate in candidates)


def test_the_report_still_shows_what_it_computes():
    """The real risk of restacking a card is not a wrong number, it is a
    missing one — and no snapshot diff makes that visible in a document where
    every line changed.

    This pins the *proportion* rather than the exact set, because which
    statistic lands where is presentation and moves legitimately. What must not
    happen is the proportion falling.
    """
    frame = _frame()
    html = profile(frame, seed=0).html
    payload = summarize(frame, seed=0)

    shown = 0
    missing: list[str] = []
    for column, stats in payload["columns"].items():
        for key, value in stats.items():
            if key in _NOT_RENDERED_VERBATIM or isinstance(value, (list, dict)):
                continue
            if _appears(value, html):
                shown += 1
            else:
                missing.append(f"{column}.{key}={value!r}")

    total = shown + len(missing)
    assert total > 60, "the payload got smaller; this test is measuring nothing"
    coverage = shown / total
    assert coverage >= 0.90, (
        f"the report shows {coverage:.1%} of what it computes, down from 90%. "
        f"Statistics no longer on the page: {sorted(missing)[:10]}"
    )


def test_every_exemption_has_a_reason():
    """An allow-list without reasons becomes a place to hide regressions."""
    for key, reason in _NOT_RENDERED_VERBATIM.items():
        assert reason and len(reason) > 10, key


def test_every_exemption_still_applies_to_something():
    """An entry that matches no real key is dead weight, and dead weight in an
    allow-list is where a genuine exemption goes to be forgotten. Two entries
    here named fields that do not exist -- `ts_min` for `min_ts`, and a
    `sample_scale` that was never in the payload -- so they exempted nothing
    while reading as though they did.
    """
    payload = summarize(_frame(), seed=0)
    keys: set[str] = set()
    for stats in payload["columns"].values():
        keys |= set(stats)
    unused = sorted(set(_NOT_RENDERED_VERBATIM) - keys)
    assert not unused, f"these exempt nothing: {unused}"


def _write_fixtures() -> None:
    FIXTURES.mkdir(exist_ok=True)
    (FIXTURES / "fingerprint.txt").write_text(
        fingerprint(profile(_frame(), seed=0).html) + "\n", encoding="utf-8"
    )
    for name, builder in FRAMES.items():
        payload = json.loads(json.dumps(summarize(builder(), seed=0), default=str))
        (FIXTURES / f"summary_{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote fixtures to {FIXTURES}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write_fixtures()
    else:
        print(__doc__)
