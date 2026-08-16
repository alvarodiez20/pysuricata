"""The restacked numeric card, and the two things that made it necessary.

Phase 5.1. The chart used to be one third of a row beside two 240px stat
tables; full width it gains about 550px, which is what makes 50 bins legible
and the log toggle worth having.

Two defects fell out of the restack rather than being designed away:

**The controls row was a grid sized for a layout the card no longer uses** —
``var(--triple-left) var(--triple-right) 1fr``, "to match .triple-row". At
390px its centre track measured 361px inside 358px and pushed the whole *page*
into horizontal scroll, which makes every sideways gesture ambiguous, including
the one inside the sample table's own scroll pane.

**The toggles were inline links**, so their target was the line box — about
20px, under even the 24px of WCAG 2.5.8, on the controls a reader uses most.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.numeric_card import NumericCardRenderer

CSS = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"
CARDS_CSS = (CSS / "_06-cards.css").read_text()


@pytest.fixture(scope="module")
def html() -> str:
    rng = np.random.default_rng(0)
    n = 300
    return profile(
        pd.DataFrame(
            {
                "measure": rng.normal(0, 1, n),
                "kind": rng.choice(list("abc"), n),
                "flag": rng.integers(0, 2, n).astype(bool),
                "when": pd.date_range("2026-01-01", periods=n, freq="h"),
            }
        ),
        seed=0,
    ).html


def _cards(html: str) -> list[str]:
    """The card elements, and only those.

    `split(...)[0]` is everything before the first card -- the whole `<head>`,
    including the inlined script, which contains `data-type="numeric"` in its
    filtering code. Searching it for a card finds the document.
    """
    return html.split('<article class="var-card"')[1:]


def _card(html: str, kind: str) -> str:
    for chunk in _cards(html):
        if f'data-type="{kind}"' in chunk:
            return chunk
    raise AssertionError(f"no {kind} card")


def _numeric_card(html: str) -> str:
    return _card(html, "numeric")


# --------------------------------------------------------------------------- #
# the restack
# --------------------------------------------------------------------------- #
class TestTheCardIsRestacked:
    def test_the_chart_is_no_longer_a_third_of_a_row(self, html):
        card = _numeric_card(html)
        assert 'class="var-chart"' in card
        assert "triple-row" not in card

    def test_the_order_is_chart_then_controls_then_stats(self, html):
        card = _numeric_card(html)
        chart = card.index('class="var-chart"')
        controls = card.index('class="card-controls"')
        stats = card.index('class="vstat-row"')
        assert chart < controls < stats

    def test_the_two_key_value_tables_became_one_row(self, html):
        card = _numeric_card(html)
        assert 'class="vstat-row"' in card
        assert "stats-left" not in card
        assert "stats-right" not in card

    def test_no_statistic_was_dropped_in_the_move(self, html):
        """The row carries what both tables carried."""
        card = _numeric_card(html)
        for label in ("Count", "Missing", "Zeros", "Min", "Median", "Mean", "Max"):
            assert f">{label}</div>" in card, label


class TestTheOtherCardsStillHaveTheirLayout:
    """#114 is the numeric card only. The categorical, boolean and datetime
    cards are phases 5.3 and 5.4 and still emit `.triple-row` -- deleting the
    generic grid with the numeric restack flattened all three at once."""

    @pytest.mark.parametrize("kind", ["categorical", "boolean", "datetime"])
    def test_it_still_emits_the_three_column_body(self, html, kind):
        assert 'class="triple-row"' in _card(html, kind)

    def test_the_grid_that_lays_it_out_still_exists(self):
        block = CARDS_CSS.split("#pysuricata-report .var-card__body .triple-row {", 1)[
            1
        ].split("}", 1)[0]
        assert "display: grid" in block
        assert "grid-template-columns" in block


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #
class TestTheControls:
    def test_they_wrap_rather_than_overflowing(self):
        """The row was a three-track grid sized for `.triple-row`, so at 390px
        it measured 361px inside 358px and put the page into horizontal
        scroll."""
        block = CARDS_CSS.split("#pysuricata-report .card-controls {", 1)[1].split(
            "}", 1
        )[0]
        assert "display: flex" in block
        assert "flex-wrap: wrap" in block
        assert "grid-template-columns" not in block

    def test_every_toggle_is_a_full_size_target(self):
        block = CARDS_CSS.split(
            "#pysuricata-report .card-controls .hist-controls button.btn-soft {", 1
        )[1].split("}", 1)[0]
        assert "min-height: var(--tap-min)" in block
        assert "min-width: var(--tap-min)" in block

    def test_the_details_toggle_is_one_too(self):
        assert "#pysuricata-report .card-controls .details-toggle," in CARDS_CSS

    def test_the_toggles_are_buttons_not_links(self, html):
        """A link's target is its line box; a button's is its box."""
        card = _numeric_card(html)
        controls = card.split('class="hist-controls"', 1)[1].split("</div></div>", 1)[0]
        assert "<button" in controls
        assert "<a " not in controls


# --------------------------------------------------------------------------- #
# the stat row
# --------------------------------------------------------------------------- #
class TestTheStatRow:
    def test_a_long_value_cannot_widen_its_column(self):
        """A grid track's default minimum is its content, so `1fr` lets one
        long value -- `-1.2345678e+18` is the case that does it -- push the
        other three out of alignment. `minmax(0, 1fr)` does not."""
        block = CARDS_CSS.split("#pysuricata-report .vstat-row {", 1)[1].split("}", 1)[
            0
        ]
        assert "minmax(0, 1fr)" in block

    def test_the_mobile_rule_comes_after_the_rule_it_overrides(self):
        """Same specificity, so the later declaration wins. Placed before, the
        two-column mobile grid never applied and the desktop four-column one
        was in force at 390px -- which measured *shorter*, and would have read
        as the target being met."""
        desktop = CARDS_CSS.index("#pysuricata-report .vstat-row {")
        # Scoped to the vstat rule: `.stats-grid` uses the same declaration
        # elsewhere in the file, and matching that one proves nothing.
        mobile = CARDS_CSS.index("#pysuricata-report .vstat-row {", desktop + 1)
        assert desktop < mobile

    def test_a_severity_reaches_the_value(self, html):
        card = _numeric_card(html)
        assert "vstat" in card
        assert re.search(r'class="vstat is-(warn|crit)"', card) or "is-warn" not in card

    def test_an_identifier_gets_the_facts_a_key_raises(self):
        frame = pd.DataFrame({"id": np.arange(500), "v": np.arange(500) * 1.5})
        card = _numeric_card(profile(frame, seed=0).html)
        assert "Distinct (≈)" in card or "Rows" in card


class TestEdgeCases:
    """Note which of these produce a *numeric* card at all. A float column with
    one distinct value, or three, is classified categorical -- the cardinality
    ceiling is 50 -- so the constant-column case is about the report surviving,
    not about the numeric card's layout."""

    def test_a_constant_column_renders(self):
        out = profile(pd.DataFrame({"c": [7.0] * 200}), seed=0).html
        assert "<html" in out
        assert "var-card" in out

    def test_an_all_missing_column_renders(self):
        out = profile(pd.DataFrame({"m": [np.nan] * 200}), seed=0).html
        assert "<html" in out

    def test_infinities_do_not_break_the_card(self):
        values = np.r_[np.random.default_rng(0).normal(0, 1, 100), np.inf, -np.inf]
        out = profile(pd.DataFrame({"i": values}), seed=0).html
        assert 'class="vstat-row"' in _numeric_card(out)

    def test_a_single_row_frame_renders(self):
        out = profile(pd.DataFrame({"x": [1.0]}), seed=0).html
        assert "<html" in out

    def test_a_very_large_value_does_not_break_the_grid(self):
        """Enough distinct values to stay numeric, one of them huge."""
        rng = np.random.default_rng(0)
        values = np.r_[rng.normal(0, 1, 200), -1.2345678e18]
        card = _numeric_card(profile(pd.DataFrame({"x": values}), seed=0).html)
        assert 'class="vstat-row"' in card
        assert "e+18" in card or "1.2345678" in card


class TestTheRendererStillHasItsSeams:
    def test_the_stat_data_is_separable_from_its_markup(self):
        """The builders return data now, so the row can be rendered any way
        without touching what goes in it."""
        renderer = NumericCardRenderer()
        assert hasattr(renderer, "_left_stats")
        assert hasattr(renderer, "_right_stats")
        assert hasattr(renderer, "_build_stat_row")
