"""Recognising a key column, and turning quality chips into navigation.

UX-2 and UX-3. Neither computes anything new: monotonicity, the distinct
estimate and `int_like` were already tracked, and the chips were already
rendered. Both findings were that the report had the answer and did not use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.render.identifier import identifier_facts, looks_like_identifier
from pysuricata.render.triage import (
    Chip,
    actionable_chips,
    annotate_flags,
    build_attention_block,
    extract_chips,
    flag_slug,
)


@dataclass
class FakeStats:
    """The subset of NumericStats the identifier rule reads."""

    count: int = 10_000
    missing: int = 0
    unique_est: int = 10_000
    int_like: bool = True
    mono_inc: bool = True
    mono_dec: bool = False
    min: float = 0.0
    max: float = 9_999.0


class TestIdentifierRule:
    def test_a_monotonic_unique_integer_key_is_recognised(self):
        assert looks_like_identifier(FakeStats()) is True

    def test_a_descending_key_counts_too(self):
        assert looks_like_identifier(FakeStats(mono_inc=False, mono_dec=True)) is True

    def test_sorted_but_duplicated_is_not_a_key(self):
        """The case most likely to be mistaken for one: a grouped measurement."""
        assert looks_like_identifier(FakeStats(unique_est=500)) is False

    def test_unsorted_is_not_a_key(self):
        assert looks_like_identifier(FakeStats(mono_inc=False, mono_dec=False)) is False

    def test_a_float_column_is_not_a_key(self):
        assert looks_like_identifier(FakeStats(int_like=False)) is False

    def test_a_column_with_nulls_is_not_a_key(self):
        assert looks_like_identifier(FakeStats(missing=1)) is False

    def test_a_tiny_frame_is_not_evidence(self):
        """1, 2, 3 is monotonic, integral and distinct, and is not a key."""
        assert looks_like_identifier(FakeStats(count=3, unique_est=3)) is False

    def test_the_sketch_error_does_not_disqualify_a_real_key(self):
        """KMV is ~2.2% off at k=2048; requiring equality would reject keys."""
        assert looks_like_identifier(FakeStats(unique_est=9_850)) is True


class TestIdentifierFacts:
    def test_the_distinct_estimate_is_clamped_to_the_row_count(self):
        """More distinct values than rows is impossible and reads as a bug."""
        facts = dict(identifier_facts(FakeStats(count=3_000, unique_est=3_064)))
        assert facts["Distinct (≈)"] == "3,000"
        assert facts["Duplicates (≈)"] == "0"

    def test_gaps_in_the_sequence_are_reported(self):
        facts = dict(identifier_facts(FakeStats(count=900, unique_est=900, max=999.0)))
        assert facts["Gaps in range"] == "100"

    def test_a_dense_key_has_no_gaps(self):
        facts = dict(identifier_facts(FakeStats(count=10_000, max=9_999.0)))
        assert facts["Gaps in range"] == "0"

    def test_the_order_is_named(self):
        assert dict(identifier_facts(FakeStats()))["Order"] == "ascending"
        assert (
            dict(identifier_facts(FakeStats(mono_inc=False, mono_dec=True)))["Order"]
            == "descending"
        )


class TestIdentifierEndToEnd:
    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        n = 3_000
        return pd.DataFrame(
            {
                "id": np.arange(n),
                "grouped": np.repeat(np.arange(n // 10), 10),
                "measure": rng.standard_normal(n),
            }
        )

    def test_the_key_gets_an_identifier_badge(self, frame):
        assert profile(frame).html.count(">Identifier<") == 1

    def test_summarize_says_so_too(self, frame):
        """The payload must not be poorer than the HTML."""
        columns = summarize(frame)["columns"]
        assert columns["id"]["type"] == "identifier"
        assert columns["grouped"]["type"] == "numeric"
        assert columns["measure"]["type"] == "numeric"

    def test_the_signals_reach_the_payload(self, frame):
        column = summarize(frame)["columns"]["id"]
        assert column["mono_inc"] is True
        assert column["int_like"] is True

    def _summary_tables(self, html: str) -> str:
        """The stat row at the top of the `id` card.

        Scoped deliberately: the expandable details pane still renders
        quantiles for an identifier. That is a separate surface from the
        summary UX-2 is about, and narrowing here keeps the test honest about
        what was fixed.

        The anchor is the stat row rather than the old `<div class="box
        chart">`: #114 restacked the card, so the two 240px key/value tables
        beside the chart became one row beneath it.
        """
        return self._stat_row(html, "id")

    @staticmethod
    def _stat_row(html: str, column: str) -> str:
        """The stat row of one column's card.

        Split on the article boundary rather than matching across it: a single
        regex spanning from the card header to the stat row has to describe
        every element in between, so it breaks on any layout change -- which is
        exactly what it did here.
        """
        cards = html.split('<article class="var-card"')
        card = next((c for c in cards if f'data-name="{column}"' in c), None)
        assert card, f"{column} card not found"
        row = re.search(
            r'<div class="vstat-row">.*?</div>\s*</div>\s*</div>', card, re.S
        )
        assert row, f"{column} stat row not found"
        return row.group(0)

    def test_the_summary_answers_the_questions_a_key_raises(self, frame):
        tables = self._summary_tables(profile(frame).html)
        for expected in ("Rows", "Distinct (≈)", "Duplicates (≈)", "Gaps in range"):
            assert expected in tables, expected

    def test_the_summary_drops_the_statistics_that_mean_nothing(self, frame):
        """Zeros: 1 (0.0%) on a key is true and meaningless."""
        tables = self._summary_tables(profile(frame).html)
        for unwanted in ("Q1 (P25)", "Zeros", "Outliers", "Infinites", "Mean"):
            assert unwanted not in tables, unwanted

    def test_an_ordinary_numeric_column_keeps_them(self, frame):
        row = self._stat_row(profile(frame).html, "measure")
        assert "Mean" in row
        assert "Q1 (P25)" in row


class TestChipExtraction:
    def test_a_threshold_containing_a_bracket_does_not_break_the_label(self):
        """data-threshold=">1" ends the tag early for a naive [^>]* match."""
        chips = extract_chips(
            '<li class="flag warn" data-threshold=">1" data-value="6.6">Skewed Right</li>'
        )
        assert chips == [Chip("warn", "Skewed Right", "skewed-right", "6.6", ">1")]

    def test_a_threshold_with_an_inequality_in_prose(self):
        chips = extract_chips(
            '<li class="flag bad" data-threshold="|kurtosis| > 3" '
            'data-value="9.1">Heavy&#8209;tailed</li>'
        )
        assert chips == [
            Chip("bad", "Heavy‑tailed", "heavy-tailed", "9.1", "|kurtosis| > 3")
        ]

    def test_an_unclassed_chip_reports_empty_severity(self):
        assert extract_chips('<li class="flag">Heaping</li>') == [
            Chip("", "Heaping", "heaping")
        ]

    def test_slugs_normalise_the_non_breaking_hyphen(self):
        assert flag_slug("Heavy‑tailed") == "heavy-tailed"
        assert flag_slug("Zero‑inflated") == "zero-inflated"


class TestActionableRule:
    def test_a_bad_chip_is_always_actionable(self):
        chip = Chip("bad", "Many outliers", "many-outliers")
        assert actionable_chips([chip]) == [chip]

    def test_a_good_chip_never_is(self):
        assert actionable_chips([Chip("good", "Positive‑only", "positive-only")]) == []

    def test_distribution_shape_warnings_are_not_defects(self):
        """A standard normal earns both of these; nine clean columns are not
        nine problems."""
        assert actionable_chips([Chip("warn", "Has negatives", "has-negatives")]) == []
        assert actionable_chips([Chip("warn", "Some outliers", "some-outliers")]) == []
        assert actionable_chips([Chip("warn", "Skewed Right", "skewed-right")]) == []

    def test_data_quality_warnings_are(self):
        for label in ("Missing", "Zero‑inflated", "Quasi‑constant", "Imbalanced"):
            chip = Chip("warn", label, flag_slug(label))
            assert actionable_chips([chip]) == [chip], label


class TestTheRuleSurvivesTheChipBeingRewritten:
    """#238. Every case above builds its chips by hand, in the shape a card
    emits them. No card ships that shape: `annotate_flags` rewrites each face
    to lead with the column's own value first, and it was that rewritten face
    the rule used to match against. `Missing` slugs to `missing` and matches;
    `19.9% missing` slugs to `19-9-missing` and matches nothing.

    So `_ACTIONABLE_WARNINGS` selected nothing at all -- eleven entries of dead
    configuration -- and the attention block was `bad`-only without saying so.
    The gap between what these tests fed the rule and what production fed it is
    the whole reason it survived, which is what this class closes.
    """

    def test_a_valued_warn_chip_is_still_actionable(self):
        annotated = annotate_flags(
            '<li class="flag warn" data-threshold=">10%" data-value="19.9%">'
            "Missing</li>"
        )
        assert "19.9% missing" in annotated, "the face should carry the value"
        chips = extract_chips(annotated)
        assert chips == [
            Chip("warn", "19.9% missing · limit 20%", "missing", "19.9%", ">10%")
        ]
        assert actionable_chips(chips) == chips

    def test_the_slug_does_not_move_with_the_value(self):
        """Two columns with the same defect must land on the same slug, or the
        chip filter can never group them -- it was selecting on a token unique
        to whichever column happened to be 77.1% missing."""
        slugs = {
            extract_chips(
                annotate_flags(
                    f'<li class="flag warn" data-value="{value}">Missing</li>'
                )
            )[0][2]
            for value in ("0.2%", "19.9%", "77.1%")
        }
        assert slugs == {"missing"}

    def test_a_column_whose_only_defect_is_a_warning_reaches_the_block(self):
        """`Embarked` is this shape on the Titanic report -- 72.4% dominant
        category and 0.2% missing, both `warn`, and it was absent entirely."""
        chips = extract_chips(
            annotate_flags(
                '<li class="flag warn" data-value="72.4%">Dominant category</li>'
                '<li class="flag warn" data-value="0.2%">Missing</li>'
            )
        )
        block = build_attention_block([("Embarked", "col_Embarked", chips)])
        assert 'href="#col_Embarked"' in block
        assert 'data-flag="dominant-category"' in block
        assert 'data-flag="missing"' in block


class TestAttentionBlock:
    def test_nothing_flagged_says_so(self):
        """It used to render nothing, on the grounds that an empty
        `0 of 60 columns have issues` banner is noise. #149 overturns that: an
        absence and a clean result look identical, and only one of them is
        information, so a block that vanishes reads as a broken feature. The
        same argument #138 already accepted for correlations.

        The banner it replaces is still not what gets rendered -- the statement
        is `All 1 column looks fine`, not `0 of 1`."""
        block = build_attention_block(
            [("a", "a", [Chip("good", "Positive‑only", "positive-only")])]
        )

        assert "look" in block and "fine" in block
        assert "need a look" not in block

    def test_a_frame_with_no_columns_still_renders_nothing(self):
        """There is no clean bill of health to give for zero columns."""
        assert build_attention_block([]) == ""

    def test_the_ranking_is_by_how_far_past_the_limit(self):
        """Severity first, then `value / threshold` -- the one quantity
        comparable across flag types. Both of these are `bad` missing chips, so
        only the magnitude separates them, and the flat list put them in
        whatever order the columns happened to arrive in."""
        block = build_attention_block(
            [
                ("mild", "m", [Chip("bad", "22% missing", "missing", "22%", "20%")]),
                ("severe", "s", [Chip("bad", "77% missing", "missing", "77%", "20%")]),
            ]
        )

        assert block.index("severe") < block.index("mild")

    def test_a_chip_with_an_unrankable_threshold_does_not_sort_first(self):
        """`data-threshold="one level dominates"` is real. A chip that cannot
        be ranked must not be ranked as zero, and must not beat a real ratio."""
        block = build_attention_block(
            [
                ("prose", "p", [Chip("bad", "X", "x", "5", "one level dominates")]),
                ("ranked", "r", [Chip("bad", "77% missing", "missing", "77%", "20%")]),
            ]
        )

        assert block.index("ranked") < block.index("prose")

    def test_the_list_is_capped_and_says_what_it_withheld(self):
        """Past roughly ten rows the block is the flat column list it
        replaced. Nothing disappears silently: the remainder is counted."""
        columns = [
            (f"c{i}", f"col_c{i}", [Chip("bad", "x", "missing", f"{99 - i}%", "20%")])
            for i in range(14)
        ]
        block = build_attention_block(columns)

        assert block.count("attention-item") == 10
        assert "4 further flagged columns" in block
        assert "<strong>14</strong> of 14 columns need a look" in block

    def test_the_chips_carry_the_number_and_the_limit(self):
        """#137 put both on the card chips; this block dropped them, so it said
        `37.8% missing` without saying why that is on the list."""
        block = build_attention_block(
            [("a", "a", [Chip("bad", "77% missing", "missing", "77%", "20%")])]
        )

        assert 'data-value="77%"' in block
        assert 'data-threshold="20%"' in block

    def test_the_banner_counts_flagged_against_total(self):
        block = build_attention_block(
            [
                ("a", "a", [Chip("bad", "Many outliers", "many-outliers")]),
                ("b", "b", []),
                ("c", "c", []),
            ]
        )
        assert "<strong>1</strong> of 3 columns need a look" in block

    def test_each_column_links_to_its_card(self):
        block = build_attention_block(
            [("revenue", "col_revenue", [Chip("bad", "X", "x")])]
        )
        assert 'href="#col_revenue"' in block

    def test_the_worst_column_comes_first(self):
        block = build_attention_block(
            [
                ("warned", "w", [Chip("warn", "Missing", "missing")]),
                ("broken", "b", [Chip("bad", "Many outliers", "many-outliers")]),
            ]
        )
        assert block.index("broken") < block.index("warned")

    def test_column_names_are_escaped(self):
        block = build_attention_block([("<script>", "x", [Chip("bad", "X", "x")])])
        assert "<script>" not in block
        assert "&lt;script&gt;" in block


class TestTriageEndToEnd:
    @pytest.fixture
    def html(self):
        rng = np.random.default_rng(0)
        n = 3_000
        frame = pd.DataFrame(
            {
                "id": np.arange(n),
                "revenue": rng.lognormal(3, 1.5, n),
                "clean": rng.standard_normal(n),
                "holey": np.where(rng.random(n) < 0.4, np.nan, rng.standard_normal(n)),
            }
        )
        return profile(frame).html

    def test_the_block_appears(self, html):
        assert 'id="needs-attention"' in html

    def test_a_clean_column_is_not_flagged(self, html):
        flags = dict(re.findall(r'data-name="([^"]+)" data-flags="([^"]*)"', html))
        assert flags["clean"] == ""
        assert flags["id"] == ""

    def test_a_column_with_missing_values_is(self, html):
        flags = dict(re.findall(r'data-name="([^"]+)" data-flags="([^"]*)"', html))
        assert "missing" in flags["holey"]

    def test_every_card_carries_its_flags_for_the_filter(self, html):
        assert len(re.findall(r'data-flags="[^"]*"', html)) == 4

    def test_the_chip_filter_is_wired(self, html):
        assert "setupFlagFilters" in html
