"""A tab has to earn itself (#154, 5b.4).

The Missing Values pane rendered on **every** column, including ones with no
missing values, where it drew a 100%-present bar and a one-segment chunk strip
reading 0.0%. A click to learn nothing.

Dropping it removed something worse than emptiness. The invariance fingerprint
lost four facts, which is what sent me to look at them:

    attr::::missing  0        a pane reporting nothing
    attr::::missing  1563     on an 891-row frame
    attr::::pct      175.4    ...that is 175.4% missing

The second pair is #139's chunk metadata, which counts *renders* rather than
chunks, rendered as a severity-coloured segment on a column that had no missing
values at all. The harness flagged that facts had disappeared; the facts turned
out to be impossible.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.card_base import CardRenderer


def _tabs(html: str, column: str) -> list[str]:
    markup = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)
    start = markup.find(f'id="col_{column}-details"')
    if start < 0:
        return []
    return re.findall(r'role="tab"[^>]*data-tab="(\w+)"', markup[start : start + 3000])


@pytest.fixture(scope="module")
def report() -> str:
    rng = np.random.default_rng(0)
    n = 400
    gappy = rng.normal(0, 1, n)
    gappy[rng.choice(n, 80, replace=False)] = np.nan
    when = pd.Series(pd.date_range("2026-01-01", periods=n, freq="h"))
    when[rng.choice(n, 40, replace=False)] = pd.NaT
    return profile(
        pd.DataFrame(
            {
                "num_gap": gappy,
                "num_full": rng.normal(0, 1, n),
                "cat_gap": rng.choice(["a", "b", None], n, p=[0.5, 0.4, 0.1]),
                "cat_full": rng.choice(list("xy"), n),
                "when_gap": when,
                "when_full": pd.date_range("2026-01-01", periods=n, freq="h"),
                "flag": rng.integers(0, 2, n).astype(bool),
            }
        ),
        seed=0,
    ).html


class TestMissingValuesEarnsItsTab:
    @pytest.mark.parametrize("column", ["num_gap", "cat_gap", "when_gap"])
    def test_a_column_with_gaps_keeps_the_pane(self, report, column):
        assert "missing" in _tabs(report, column)

    @pytest.mark.parametrize("column", ["num_full", "cat_full", "when_full", "flag"])
    def test_a_complete_column_does_not_render_it(self, report, column):
        assert "missing" not in _tabs(report, column)

    def test_every_card_kind_is_covered(self, report):
        """Not just the numeric card. The pane is built by four renderers and
        the fix has to reach all of them."""
        for column in ("num_full", "cat_full", "when_full", "flag"):
            assert _tabs(report, column), f"{column} has no details section at all"


class TestTheOrderIsFixed:
    """A tab appears or does not; it never moves. A control that changes
    position between two cards of the same kind is worse than one that is
    sometimes absent."""

    def test_numeric_order_is_stable_with_and_without_missing(self, report):
        with_gap = _tabs(report, "num_gap")
        without = _tabs(report, "num_full")
        assert without == [t for t in with_gap if t != "missing"]

    def test_datetime_order_is_stable(self, report):
        with_gap = _tabs(report, "when_gap")
        without = _tabs(report, "when_full")
        assert without == [t for t in with_gap if t != "missing"]


class TestSomethingIsAlwaysActive:
    """The first surviving pane is the active one, so dropping a tab can never
    leave a details section that opens on nothing."""

    @pytest.mark.parametrize(
        "column",
        ["num_gap", "num_full", "cat_gap", "cat_full", "when_gap", "when_full", "flag"],
    )
    def test_exactly_one_tab_is_active(self, report, column):
        markup = re.sub(r"<(script|style)\b.*?</\1>", "", report, flags=re.S | re.I)
        start = markup.find(f'id="col_{column}-details"')
        section = markup[start : start + 30_000]
        section = section[: section.find("</section>", section.find("tab-panes"))]
        assert section.count('class="tab-pane active"') == 1


class TestTheBuilder:
    def test_a_section_with_no_panes_renders_nothing(self):
        renderer = CardRenderer.__new__(CardRenderer)
        assert renderer._build_tabbed_details("c", [("a", "A", "", False)]) == ""

    def test_dropped_panes_leave_no_trace(self):
        renderer = CardRenderer.__new__(CardRenderer)
        out = renderer._build_tabbed_details(
            "c", [("a", "A", "<p>keep</p>", True), ("b", "B", "<p>drop</p>", False)]
        )
        assert "drop" not in out
        assert 'data-tab="b"' not in out

    def test_the_first_survivor_is_active(self):
        renderer = CardRenderer.__new__(CardRenderer)
        out = renderer._build_tabbed_details(
            "c", [("a", "A", "<p>x</p>", False), ("b", "B", "<p>y</p>", True)]
        )
        assert 'class="active" data-tab="b"' in out
        assert out.count("tab-pane active") == 1


class TestTheImpossibleNumberIsGone:
    """`data-missing="1563"` on an 891-row frame -- 175.4% -- was rendered as a
    severity-coloured segment inside a pane on a column with no missing values.
    It came from #139's chunk metadata, which counts renders rather than chunks.
    """

    def test_no_segment_claims_more_missing_than_there_are_rows(self, report):
        markup = re.sub(r"<(script|style)\b.*?</\1>", "", report, flags=re.S | re.I)
        for value in re.findall(r'data-missing="(\d+)"', markup):
            assert int(value) <= 400, f"data-missing={value} exceeds the row count"

    def test_no_percentage_exceeds_one_hundred(self, report):
        markup = re.sub(r"<(script|style)\b.*?</\1>", "", report, flags=re.S | re.I)
        for value in re.findall(r'data-pct="([\d.]+)"', markup):
            assert float(value) <= 100.0, f"data-pct={value}"
