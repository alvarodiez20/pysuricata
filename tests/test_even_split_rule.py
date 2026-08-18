"""Every level bar is read against what an even column would look like (#296).

`Embarked`'s S at 72.4% is a number. Against a rule at 33.3% it is a finding:
*dominated by one port*, with no arithmetic asked of the reader. That is the
same device as the flat-calendar rule on the datetime card and the outlier
fence on the numeric one, and sameness is the point — one reading convention
across the report rather than three.

`even_split_pct()` went in with the flag reference in 4b.2 and had **zero
callers** until this phase. A helper with no callers is indistinguishable from
a helper that does not work, so the last case here is the one that keeps it
honest.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.flag_reference import even_split_pct

_TAGS = re.compile(r"<script.*?</script>|<style.*?</style>", re.S)


@pytest.fixture(scope="module")
def body() -> str:
    return _TAGS.sub("", profile(pd.read_csv("web/sample/titanic.csv"), seed=0).html)


def _card(body: str, name: str) -> str:
    match = re.search(
        rf'<article class="var-card"[^>]*data-name="{name}".*?</article>', body, re.S
    )
    assert match, f"no card for {name}"
    return match.group(0)


class TestTheRuleIsDrawn:
    def test_a_level_chart_carries_one(self, body: str) -> None:
        assert 'class="cat-even"' in body

    @pytest.mark.parametrize(
        "column,levels", [("Pclass", 3), ("Sex", 2), ("Embarked", 3)]
    )
    def test_it_sits_where_an_even_split_would(
        self, body: str, column: str, levels: int
    ) -> None:
        """The position is checked against the **bars**, not against the
        helper — a rule placed by the same wrong arithmetic as the assertion
        would agree with itself and prove nothing.

        The chart's x scale runs 0..vmax where vmax is the largest bar, so an
        even split lands at `even_pct / largest_pct` along the widest bar.
        Percentages are read off the rendered `bar-value` labels; categorical
        bars carry no `data-count`, which is why this reads the text rather
        than an attribute.
        """
        card = _card(body, column)
        rule_x = float(re.search(r'class="cat-even" x1="([\d.]+)"', card).group(1))

        rects = [
            (float(x), float(w))
            for x, w in re.findall(
                r'<rect class="bar" x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', card
            )
        ]
        pcts = [
            float(p)
            for p in re.findall(
                r'class="bar-value[^"]*"[^>]*>[^<(]*\(([\d.]+)%\)', card
            )
        ]
        assert rects and pcts and len(rects) >= len(pcts), (rects, pcts)
        rects, pcts = rects[: len(pcts)], pcts

        x0 = min(x for x, _ in rects)
        widest_i = max(range(len(pcts)), key=lambda i: pcts[i])
        expected = x0 + (even_split_pct(levels) / pcts[widest_i]) * rects[widest_i][1]

        assert rule_x == pytest.approx(expected, abs=1.5), (
            f"{column}: rule at {rule_x}, an even split over {levels} levels "
            f"is at {expected}"
        )

    def test_it_is_drawn_before_the_bars(self, body: str) -> None:
        """Token rule 2: a mark crossing a data fill must protrude onto the
        paper or paint underneath it. Source order is the second half — a bar
        past the mark occludes it rather than crossing it."""
        card = _card(body, "Pclass")
        svg = re.search(r"<svg class=\"cat-svg\".*?</svg>", card, re.S).group(0)

        assert svg.index("cat-even") < svg.index("<rect")

    def test_it_protrudes_past_the_bars(self, body: str) -> None:
        """The first half of rule 2, and what keeps it findable at 390px."""
        card = _card(body, "Pclass")
        y1, y2 = (
            float(v)
            for v in re.search(
                r'class="cat-even" x1="[\d.]+" y1="(-?[\d.]+)" x2="[\d.]+" y2="([\d.]+)"',
                card,
            ).groups()
        )
        bar_ys = [
            (float(m.group(1)), float(m.group(2)))
            for m in re.finditer(r'<rect[^>]*y="([\d.]+)"[^>]*height="([\d.]+)"', card)
        ]
        assert bar_ys
        top = min(y for y, _ in bar_ys)
        bottom = max(y + h for y, h in bar_ys)

        assert y1 < top, (y1, top)
        assert y2 > bottom, (y2, bottom)


class TestItSaysNothingItCannotSay:
    def test_a_single_level_column_gets_no_rule(self) -> None:
        """One level splits evenly into itself, so the rule would sit exactly
        on the only bar and assert nothing."""
        html = profile(pd.DataFrame({"c": ["only"] * 300}), seed=0).html
        card = _card(_TAGS.sub("", html), "c")

        assert "cat-even" not in card

    def test_the_mark_does_not_move_when_the_control_does(self, body: str) -> None:
        """Every Top-N variant of one column is read against the same mark.

        A rule computed from the bars on screen rather than from the column
        would slide when a reader switched Top-5 to Top-10, which would make it
        a measure of the chart instead of the data.
        """
        card = _card(body, "Embarked")
        xs = set(re.findall(r'class="cat-even" x1="([\d.]+)"', card))

        assert len(xs) == 1, f"the rule moves between variants: {sorted(xs)}"


class TestTheCoverageNoteCarriesTheDenominator:
    def test_it_names_the_non_missing_row_count(self, body: str) -> None:
        """`Cabin` is 77.1% empty, so the same bars are 5.9% of its non-missing
        rows and 1.3% of the frame. A coverage figure without its denominator
        cannot tell those apart."""
        note = re.search(r'class="coverage-note">([^<]+)<', _card(body, "Sex")).group(1)

        assert re.search(r"covers [\d.]+% of the [\d,]+ non-missing rows", note), note

    def test_it_states_the_rule_once_rather_than_per_mark(self, body: str) -> None:
        """4b.2's lesson: a measure repeated in a tooltip on every mark is
        invisible on a phone and absent from paper. Said once, on the face."""
        card = _card(body, "Pclass")

        assert "rule at 33.3%, an even split" in card
        # No tooltip *on the rule*. The bars have carried their own `<title>`
        # since long before this phase, so a document-wide check for `<title>`
        # would fail on markup this change never touched.
        rule = re.search(r"<line class=\"cat-even\"[^>]*/?>", card).group(0)
        assert "<title>" not in rule
        assert "data-even-pct" not in rule


class TestTheHelperFinallyHasACaller:
    def test_the_module_imports_it(self) -> None:
        import pysuricata.render.categorical_card as module

        assert module.even_split_pct is even_split_pct

    @pytest.mark.parametrize(
        "levels,expected", [(2, 50.0), (3, 100 / 3), (7, 100 / 7), (147, 100 / 147)]
    )
    def test_it_still_says_what_it_said(self, levels: int, expected: float) -> None:
        assert even_split_pct(levels) == pytest.approx(expected)

    def test_it_refuses_to_divide_by_zero(self) -> None:
        assert even_split_pct(0) == 0.0
        assert even_split_pct(-1) == 0.0
