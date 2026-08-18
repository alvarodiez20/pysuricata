"""A percentage nobody can judge is not information (#291, #292).

`Weekend % 27.0` on a card face reads as a finding. A flat calendar gives
28.6%, so 27.0 is the *absence* of a weekend effect — and the renderer already
knew that, in a comment beside the flag threshold twelve lines up. That is the
Jarque–Bera problem: a number whose meaning lives somewhere the reader cannot
reach.

These tests pin the three things that make the fix real rather than cosmetic:
the baseline is drawn and not merely known, the verdict is stated in
percentage points so it can be checked against the mark, and the constants
have exactly one home.

The frame below has **two** datetime columns on purpose. One is a generated
series on a fixed interval; the other is a random draw. A fixture with a
single regular column exercises neither the irregular branch of the interval
sentence nor a non-flat verdict, and a branch a fixture misses reports as
absent — which reads as broken.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.flag_reference import (
    BUSINESS_HOURS_FLAT_PCT,
    FLAT_TOLERANCE_PP,
    WEEKEND_FLAT_PCT,
    flat_verdict,
)

#: The report inlines its own CSS and JS, so searching the whole document for
#: a class name finds it in the very source that references it. Everything
#: here asserts against the stripped body.
_TAGS = re.compile(r"<script.*?</script>|<style.*?</style>", re.S)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "signed_up": pd.date_range("2024-01-01", periods=n, freq="17min"),
            "events": pd.to_datetime(
                np.sort(
                    rng.integers(
                        pd.Timestamp("2024-01-01").value,
                        pd.Timestamp("2024-03-01").value,
                        n,
                    )
                )
            ),
            "amount": rng.lognormal(3, 1, n),
        }
    )


@pytest.fixture(scope="module")
def body() -> str:
    return _TAGS.sub("", profile(_frame(), seed=0).html)


@pytest.fixture(scope="module")
def card(body: str) -> str:
    """The first datetime card, header through to the details toggle."""
    match = re.search(
        r'<article class="var-card".*?<span class="badge">Datetime</span>.*?</article>',
        body,
        re.S,
    )
    assert match, "no datetime card in the report"
    return match.group(0)


class TestTheBaselineIsDrawn:
    def test_each_ratio_gets_a_rule(self, card: str) -> None:
        """Two shares, two bars, two marks. A bar without its rule is the bare
        percentage again in a more expensive shape."""
        assert card.count('class="cal-base__row"') == 2
        assert card.count("cal-base__fill") == 2
        assert card.count("cal-base__rule") == 2

    def test_the_rule_sits_at_the_flat_share(self, card: str) -> None:
        marks = [
            float(x) for x in re.findall(r"cal-base__rule[^>]*left:([\d.]+)%", card)
        ]
        assert marks == pytest.approx(
            [WEEKEND_FLAT_PCT, BUSINESS_HOURS_FLAT_PCT], abs=0.01
        )

    def test_the_rule_is_painted_before_the_fill(self, card: str) -> None:
        """Rule 2 in `tokens.css`: a quality mark crossing a data fill must
        protrude onto the paper or paint underneath it. Source order is the
        'underneath' half — a bar reaching past the mark occludes it rather
        than crossing it."""
        row = re.search(r'<div class="cal-base__track">.*?</div>', card, re.S).group(0)
        assert row.index("cal-base__rule") < row.index("cal-base__fill")

    def test_the_baseline_is_named_where_a_reader_can_reach_it(self, card: str) -> None:
        """The arithmetic, not just the number. `2 of 7 days` is why 28.6% is
        the right mark, and it was the thing living in a code comment."""
        assert "2 of 7 days" in card
        assert "8 of 24 hours on 5 of 7 days" in card


class TestAColumnWithNoValuesGetsNoPanel:
    """#315 made a zero-row frame render a report, which brought this panel
    within reach of a column that contains nothing.

    The ratios finalise to 0.0 there, and the verdict read
    `under-represented · −28.6pp vs 28.6%` — a confident finding about an empty
    column. This panel exists to stop a number being read as a finding when it
    is not one; it must not become the thing doing that.
    """

    @pytest.fixture(scope="class")
    def empty_card(self) -> str:
        frame = pd.DataFrame({"when": pd.Series([], dtype="datetime64[ns]")})
        return _TAGS.sub("", profile(frame, seed=0).html)

    def test_no_bar_is_drawn(self, empty_card: str) -> None:
        assert "cal-base__row" not in empty_card

    def test_no_verdict_is_stated(self, empty_card: str) -> None:
        assert "cal-base__verdict" not in empty_card
        assert "under-represented" not in empty_card

    def test_the_card_still_renders(self, empty_card: str) -> None:
        """Suppressing the panel must not take the column with it."""
        assert 'class="badge">Datetime</span>' in empty_card

    def test_a_column_with_values_is_unaffected(self, card: str) -> None:
        assert card.count('class="cal-base__row"') == 2


class TestTheVerdictIsInPercentagePoints:
    def test_a_flat_column_is_called_flat(self, card: str) -> None:
        verdicts = re.findall(r"cal-base__verdict[^>]*>([^<]+)<", card)
        assert verdicts, "no verdict rendered"
        for verdict in verdicts:
            assert "pp vs " in verdict, verdict

    @pytest.mark.parametrize(
        "actual,flat,expected_word,tone",
        [
            (27.0, WEEKEND_FLAT_PCT, "flat", "good"),
            (48.0, WEEKEND_FLAT_PCT, "over-represented", "warn"),
            (5.0, BUSINESS_HOURS_FLAT_PCT, "under-represented", "warn"),
        ],
    )
    def test_the_reading_matches_the_gap(
        self, actual: float, flat: float, expected_word: str, tone: str
    ) -> None:
        verdict, got_tone = flat_verdict(actual, flat)
        assert verdict.startswith(expected_word), verdict
        assert got_tone == tone
        # The gap is stated, and stated correctly, so it can be checked
        # against where the mark actually sits.
        assert f"{abs(actual - flat):.1f}pp vs {flat:.1f}%" in verdict

    def test_the_example_in_the_issue(self) -> None:
        assert flat_verdict(27.0, WEEKEND_FLAT_PCT)[0] == "flat · −1.6pp vs 28.6%"

    def test_the_tolerance_is_the_boundary(self) -> None:
        inside, _ = flat_verdict(
            WEEKEND_FLAT_PCT + FLAT_TOLERANCE_PP - 0.1, WEEKEND_FLAT_PCT
        )
        outside, _ = flat_verdict(
            WEEKEND_FLAT_PCT + FLAT_TOLERANCE_PP + 0.1, WEEKEND_FLAT_PCT
        )
        assert inside.startswith("flat")
        assert outside.startswith("over")

    def test_the_tone_is_a_slug_not_a_colour(self) -> None:
        """`test_colour_tokens.py` is a ratchet on untokenised colours. The
        helper must not be the place a hex code sneaks back in."""
        for actual in (0.0, 27.0, 50.0, 100.0):
            _, tone = flat_verdict(actual, WEEKEND_FLAT_PCT)
            assert tone in {"good", "warn"}


class TestTheFaceCarriesEightStatistics:
    """Phase 5e.3. Thirteen became eight; the five that left are not deleted."""

    def test_eight_and_only_eight(self, card: str) -> None:
        face = card.split('class="card-controls"')[0]
        caps = re.findall(r'class="vstat__cap">([^<]+)<', face)
        assert caps == [
            "Count",
            "Unique (≈)",
            "Missing",
            "Timezone",
            "Time span",
            "Min",
            "Max",
            "Data density",
        ], caps

    def test_the_two_ratios_are_no_longer_stat_cells(self, card: str) -> None:
        face = card.split('class="card-controls"')[0]
        assert "Weekend %" not in face
        assert "Business hrs %" not in face

    def test_min_and_max_are_single_height(self, card: str) -> None:
        """Two double-height cells in a four-column grid made every row in the
        grid taller, including the six carrying one line."""
        face = card.split('class="card-controls"')[0]
        assert "<br>" not in face

    def test_the_interval_pair_is_still_reachable(self, card: str) -> None:
        """Moved to the Statistics pane, not dropped. The sentence interprets
        them; a reader who wants the raw pair must still be able to get it."""
        pane = card.split('class="card-controls"')[1]
        assert "Avg interval" in pane
        assert "Interval std dev" in pane


class TestTheSentenceLeadsTheFace:
    def test_it_is_above_the_chart(self, card: str) -> None:
        face = card.split('class="card-controls"')[0]
        assert "dt-lede" in face
        assert face.index("dt-lede") < face.index("var-chart"), (
            "the interval sentence renders after the chart, so it is read as a "
            "conclusion rather than as the lede"
        )

    def test_it_says_the_strongest_thing_the_column_knows(self, card: str) -> None:
        lede = re.search(r'dt-lede">([^<]+)<', card).group(1)
        # `signed_up` is a 17-minute series with a standard deviation of
        # exactly zero, which is a generated series and not observed events.
        assert "Every gap is identical" in lede, lede

    def test_it_no_longer_leads_the_statistics_pane(self, card: str) -> None:
        pane = card.split('class="card-controls"')[1]
        assert "fence-lede" not in pane


class TestTheConstantsHaveOneHome:
    def test_the_card_spells_no_baseline_of_its_own(self) -> None:
        """The specific defect #291 was filed about: `expected ~28.5%` sat in a
        comment on the flag threshold, where a reader of the report can never
        get to it and a second copy can drift from the first.

        Checked against *executable* source with comments and docstrings
        stripped, because the prose explaining the fix necessarily quotes the
        numbers, and a test that cannot tell those apart would forbid
        documenting the change it exists to protect.
        """
        import ast
        import pathlib

        from pysuricata.render import datetime_card

        tree = ast.parse(pathlib.Path(datetime_card.__file__).read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                node.value.value = ""  # docstrings out; comments never parse
        code = ast.unparse(tree)

        for spelled in ("28.5", "28.6", "23.8", "2 / 7", "8 / 24"):
            assert spelled not in code, (
                f"datetime_card.py spells the baseline {spelled!r} itself. It "
                f"must read the constant in flag_reference.py -- two copies is "
                f"the drift #291 was filed about."
            )
        assert "WEEKEND_FLAT_PCT" in code
        assert "BUSINESS_HOURS_FLAT_PCT" in code

    def test_the_accumulator_computes_nothing_new(self) -> None:
        """Both baselines are arithmetic. If this ever needs a new field on
        DateTimeStats, the design was misread."""
        from pysuricata.render.flag_reference import (
            BUSINESS_HOURS_FLAT_PCT as b,
        )
        from pysuricata.render.flag_reference import (
            WEEKEND_FLAT_PCT as w,
        )

        assert w == pytest.approx(2 / 7 * 100)
        assert b == pytest.approx(8 / 24 * (5 / 7) * 100)
