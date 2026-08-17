"""The datetime card may not claim a timezone the column does not have (#241).

`datetime_card.py` emitted the literal `("Timezone", "UTC", None)` on both its
stat row and its Statistics pane, and `_format_timestamp` appended `UTC` to
every rendered instant. So a `US/Eastern` column was labelled UTC, and a
**naive** column — which has no timezone at all — was labelled UTC too. The
report was stating a fact about the data that it did not get from the data,
which is the one thing this project is not allowed to do.

## Why `source_timezone` alone cannot fix it

It is the obvious source, and it is not sufficient. The accumulator stores it
only when the zone is *not* UTC:

    if tz_part and tz_part != "UTC":
        self._source_timezone = tz_part

so `None` means "naive **or** UTC" and cannot express the distinction the issue
is about. Measured: naive and UTC columns both report `source_timezone=None`,
and only `US/Eastern` reports a value. The dtype string carries the whole truth
(`datetime64[ns]` vs `datetime64[ns, UTC]`) and is on the summary already, so
`_timezone_of()` falls back to it.

## What each column should say

| column | Timezone row | rendered instant |
|---|---|---|
| naive | `— (naive)` | `2024-01-01 00:00:00` |
| UTC | `UTC` | `2024-01-01 00:00:00 UTC` |
| US/Eastern | `US/Eastern` | `2024-01-01 05:00:00 UTC` |

The last row is the one worth being deliberate about. The accumulator stores
epoch nanoseconds, so the instant genuinely *is* 05:00 UTC — midnight Eastern.
Rendering it in UTC is a correct conversion rather than a mislabelling, and the
Timezone row saying `US/Eastern` is what lets a reader reconcile the two. What
is not allowed is the naive case, where there is no instant at all, only a wall
clock, and `UTC` would be invented.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pysuricata import profile

CASES = {
    "naive": None,
    "utc": "UTC",
    "eastern": "US/Eastern",
}


def _card(timezone: str | None) -> str:
    index = pd.date_range("2024-01-01", periods=500, freq="h", tz=timezone)
    html = profile(pd.DataFrame({"when": pd.Series(index)}), seed=0).html
    found = re.search(r'<article[^>]*data-type="datetime".*?</article>', html, re.S)
    assert found, (
        "no datetime card rendered -- the fixture missed the branch, and "
        "absent reads as passing"
    )
    return found.group(0)


def _timezone_row(card: str) -> str:
    found = re.search(r"Timezone</div>\s*<div[^>]*>([^<]*)<", card)
    assert found, "no Timezone row in the card"
    return found.group(1).strip()


@pytest.fixture(scope="module")
def cards() -> dict[str, str]:
    return {name: _card(tz) for name, tz in CASES.items()}


class TestANaiveColumnIsNotCalledUTC:
    """The headline case. A column with no timezone has none to print."""

    def test_the_card_never_says_utc(self, cards):
        assert "UTC" not in cards["naive"], (
            "a naive datetime column's card mentions UTC. The column carries "
            "no timezone, so this is a fact the report invented"
        )

    def test_the_row_says_so_rather_than_going_blank(self, cards):
        row = _timezone_row(cards["naive"])

        # Silence would be ambiguous with a rendering failure; saying "naive"
        # is a statement, and a true one.
        assert "naive" in row.lower(), row

    def test_the_instants_carry_no_zone(self, cards):
        stamps = re.findall(
            r"\d{4}-\d{2}-\d{2}<br>\d{2}:\d{2}:\d{2}[^<]*", cards["naive"]
        )

        assert stamps, "no rendered instants in the naive card"
        for stamp in stamps:
            assert not stamp.strip().endswith("UTC"), stamp


class TestAZonedColumnKeepsItsZone:
    @pytest.mark.parametrize(
        "name,expected", [("utc", "UTC"), ("eastern", "US/Eastern")]
    )
    def test_the_row_names_the_source_zone(self, cards, name, expected):
        assert _timezone_row(cards[name]) == expected

    @pytest.mark.parametrize("name", ["utc", "eastern"])
    def test_the_instants_are_labelled(self, cards, name):
        """For a zone-aware column the epoch really is an instant, so saying
        which zone it is displayed in is information rather than invention."""
        stamps = re.findall(r"\d{4}-\d{2}-\d{2}<br>\d{2}:\d{2}:\d{2}[^<]*", cards[name])

        assert stamps
        assert all(stamp.strip().endswith("UTC") for stamp in stamps), stamps

    def test_eastern_is_converted_rather_than_relabelled(self, cards):
        """Midnight in New York is 05:00 UTC. If the card showed `00:00 UTC`
        it would have taken the wall clock and stamped a zone on it, which is
        the same error in a subtler form."""
        assert "05:00:00 UTC" in cards["eastern"], (
            "the Eastern column's first instant is not 05:00 UTC, so the "
            "wall clock was relabelled rather than converted"
        )


class TestTheLiteralIsGone:
    def test_no_hardcoded_timezone_row_remains(self):
        """Both sites emitted `("Timezone", "UTC", None)`. A third would be
        just as wrong, so the check is over the file rather than two lines."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "pysuricata"
            / "render"
            / "datetime_card.py"
        ).read_text(encoding="utf-8")

        assert '("Timezone", "UTC"' not in source, (
            "a hardcoded Timezone row is back in datetime_card.py"
        )

    def test_the_summary_still_reports_what_it_always_did(self):
        """`source_timezone` is in the `summarize()` payload and documented in
        `docs/summary-schema.md`. The fix reads it and falls back to the dtype
        in the renderer; it does not change what the payload says."""
        from pysuricata import summarize

        index = pd.date_range("2024-01-01", periods=200, freq="h", tz="US/Eastern")
        payload = summarize(pd.DataFrame({"when": pd.Series(index)}), seed=0)

        assert payload["columns"]["when"]["source_timezone"] == "US/Eastern"
