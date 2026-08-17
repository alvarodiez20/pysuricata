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
from pathlib import Path
from types import SimpleNamespace

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
    """The rule tightened in #154 (5b.7): missing > 0 **and** more than one
    chunk.

    Gaps alone were not enough. With a single chunk the pane states one fact
    four times -- a Present stat, a Missing stat, a two-segment bar, and a
    one-segment chunk strip -- under a header already flagging the percentage.
    The only thing it knows that the card face does not is *where in the read*
    the gaps fall, and that needs more than one chunk to exist.

    The fixture below is a single chunk, so a gappy column correctly has no
    pane; `TestTheChunkGate` covers the other side.
    """

    @pytest.mark.parametrize(
        "column", ["num_gap", "when_gap", "num_full", "flag", "cat_gap"]
    )
    def test_one_chunk_never_renders_it(self, report, column):
        """`cat_gap` was the exception until #193.

        It was recorded rather than endorsed: `html.py` called `finalize()`
        without chunk metadata for categorical and boolean, so those
        accumulators had none to give, and applying the gate would not have
        tightened the rule -- it would have hidden the pane permanently, which
        is what the first attempt did. Both accumulators now track their own
        chunks, so the rule is the same on all four kinds.
        """
        assert "missing" not in _tabs(report, column)


class TestTheChunkGate:
    """Otherwise the rule above is satisfied by never rendering the pane."""

    @pytest.fixture(scope="class")
    def chunked(self) -> str:
        rng = np.random.default_rng(0)
        n = 12_000
        values = rng.normal(0, 1, n)
        values[rng.choice(n, n // 5, replace=False)] = np.nan
        labels = rng.choice(["a", "b", None], n, p=[0.5, 0.4, 0.1])
        return profile(
            pd.DataFrame({"num_gap": values, "cat_gap": labels}),
            seed=0,
            chunk_size=1000,
        ).html

    @pytest.mark.parametrize("column", ["num_gap", "cat_gap"])
    def test_more_than_one_chunk_brings_it_back(self, chunked, column):
        assert "missing" in _tabs(chunked, column)

    def test_every_card_kind_is_covered(self, report):
        """Not just the numeric card. The pane is built by three renderers now
        and the fix has to reach all of them."""
        for column in ("num_full", "cat_full", "when_full"):
            assert _tabs(report, column), f"{column} has no details section at all"

    def test_a_boolean_card_has_no_details_section_on_one_chunk(self):
        """5c.6, and a decision rather than an omission. Two values, two
        counts, one bar on the card face — nothing is withheld, so there is no
        second level of disclosure to offer.

        `Missing Values` was removed with it because a boolean accumulator was
        finalized without chunk metadata and so could never say *where in the
        read* the gaps fall, which is the only thing that pane knows and the
        card face does not. **#193 changed that**, and the pane came back under
        the same rule the other kinds use — so this frame is deliberately a
        single chunk, where the rule closes the gate and the whole section
        disappears again. `TestTheChunkGate` covers the open side."""
        # `boolean` dtype, not an object column of Python bools: an object
        # column with `None` in it infers as *categorical*, and the assertion
        # below would then pass or fail on a card of the wrong kind.
        frame = pd.DataFrame(
            {"flag": pd.array([True, False, None] * 60, dtype="boolean")}
        )
        html = profile(frame, seed=0).html
        assert 'id="col_flag"' in html and ">Boolean<" in html
        card = html[html.index('id="col_flag"') :]
        card = card[: card.index("</article>")]
        assert "details-section" not in card
        assert "details-toggle" not in card


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
        # `flag` is absent by design: a boolean card has no details section
        # at all since 5c.6, so there is no tab for one to be active among.
        ["num_gap", "num_full", "cat_gap", "cat_full", "when_gap", "when_full"],
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


class TestTheClosedStripNamesItsPanes:
    """5b.8. A `Details` button toggles a section containing up to six tabs.

    The tab set is known at render time and was not printed, so the word
    "Details" promised nothing -- and a reader had to open every card to learn
    whether opening was worth it. `11 outliers` beside the button is the reason
    to open it; `no outliers` is the reason not to.
    """

    @pytest.fixture(scope="class")
    def summary(self, report) -> str:
        found = re.search(r'class="details-panes">([^<]*)<', report)
        assert found, "no pane summary on any card"
        return found.group(1)

    def test_it_lists_the_panes(self, summary):
        assert "statistics" in summary
        assert "·" in summary

    def test_it_carries_the_figure_each_pane_is_worth_opening_for(self, report):
        strips = re.findall(r'class="details-panes">([^<]*)<', report)
        assert any(re.search(r"\d+ outliers", s) for s in strips), strips

    def test_zero_outliers_reads_as_a_phrase(self):
        """The reason *not* to open the pane, which is half the point.

        Needs its own frame: every numeric column in the shared fixture has
        outliers, so the zero case would never be reached. A linear ramp has
        none by the IQR rule, by construction.

        As a tab badge the phrasing would be nonsense -- `Outliers no` was the
        first attempt -- so it lives in the strip only.
        """
        ramp = profile(pd.DataFrame({"ramp": np.arange(500, dtype=float)}), seed=0).html
        strips = re.findall(r'class="details-panes">([^<]*)<', ramp)
        assert any("no outliers" in s for s in strips), strips
        assert not re.search(r'class="tab__count">no<', ramp)

    def test_a_dropped_pane_is_not_named(self, report):
        """The strip and the tab set have to agree, or the strip advertises a
        tab that is not there."""
        for card in re.split(r'(?=<article class="var-card")', report):
            strip = re.search(r'class="details-panes">([^<]*)<', card)
            if not strip:
                continue
            tabs = set(re.findall(r'data-tab="(\w+)"', card))
            if "missing" not in tabs:
                assert "missing" not in strip.group(1)
            if "corr" not in tabs:
                assert "correlations" not in strip.group(1)


class TestTheActiveMarkerIsOnTheLabel:
    """The tab button is 44px tall because it is a tap target (#122). A
    `border-bottom` on the button paints the rule ~29px below the word, where
    it reads as a second hairline floating under the strip rather than as an
    underline belonging to the text."""

    def test_the_label_has_its_own_element(self, report):
        assert 'class="tab__label"' in report

    def test_the_stylesheet_underlines_the_label_not_the_button(self):
        css = (
            Path(__file__).resolve().parents[1]
            / "pysuricata"
            / "static"
            / "css"
            / "_06-cards.css"
        ).read_text(encoding="utf-8")
        active = re.search(
            r'\.tabs \[role="tab"\]\.active \.tab__label \{(.*?)\}', css, re.S
        )
        assert active, "the active label rule is gone"
        assert "border-bottom-color: var(--data-1)" in active.group(1)

        button = re.search(r'\.tabs \[role="tab"\]\.active \{(.*?)\}', css, re.S)
        assert button, "the active button rule is gone"
        assert "border-bottom: 2px" not in button.group(1), (
            "the underline is back on the 44px tap box"
        )


class TestTheDatetimePanesSayWhatTheyKnow:
    """5c.4 and 5c.5 of #155."""

    @pytest.fixture(scope="class")
    def report(self) -> str:
        rng = np.random.default_rng(0)
        return profile(
            pd.DataFrame(
                {
                    # A record every 17 minutes: the design's motivating case.
                    "regular": pd.date_range("2026-01-01", periods=900, freq="17min"),
                    "events": pd.to_datetime(
                        np.sort(rng.integers(1_577_836_800, 1_735_689_600, 900)),
                        unit="s",
                    ),
                }
            ),
            seed=0,
        ).html

    def _card(self, report: str, column: str) -> str:
        start = report.index(f'id="col_{column}"')
        return report[start : report.index("</article>", start)]

    def test_a_generated_series_says_so_first(self, report):
        """It was a table row reading `Interval std dev — 0.0 seconds`, filed
        alphabetically between timezone and weekend ratio. A deviation of zero
        means every gap is identical, which is what anyone opens a datetime
        column to ask."""
        card = self._card(report, "regular")
        assert "Every gap is identical" in card
        assert "17.0 minutes" in card

    def test_an_event_stream_is_not_called_a_schedule(self, report):
        card = self._card(report, "events")
        assert "vary widely" in card
        assert "Every gap is identical" not in card

    def test_the_claim_of_identical_gaps_needs_exactly_zero(self):
        """A nearly-regular series is a different and weaker statement."""
        from pysuricata.render.datetime_card import DateTimeCardRenderer

        renderer = DateTimeCardRenderer()
        nearly = SimpleNamespace(
            avg_interval_seconds=1020.0, interval_std_seconds=0.001
        )
        assert "Every gap is identical" not in renderer._interval_sentence(nearly)

    def test_a_single_year_draws_no_year_chart(self, report):
        """`by_year` is a dict, so one year renders a single bar at full
        height — a chart whose only reading is "all of it"."""
        card = self._card(report, "regular")
        assert "Every record falls in 2026" in card
        assert "Year Distribution" not in card

    def test_several_years_keep_the_chart(self, report):
        card = self._card(report, "events")
        assert "Year Distribution" in card

    def test_each_panel_carries_its_own_peak(self, report):
        """A 211-record hour and a 2,626-record month drew identically, and the
        peaks that would resolve it lived in a different tab."""
        card = self._card(report, "regular")
        peaks = re.findall(r'class="temporal__peak">peak ([^<]+)<', card)
        assert len(peaks) >= 3, peaks

    def test_the_per_chart_scale_is_stated(self, report):
        """Heights compare within a chart and not between them, and a reader
        should not have to discover that."""
        assert "scaled to its own peak" in self._card(report, "regular")
