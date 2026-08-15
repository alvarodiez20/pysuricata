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
    actionable_chips,
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
        """The two key/value tables at the top of the `id` card.

        Scoped deliberately: the expandable details pane still renders
        quantiles for an identifier. That is a separate surface from the
        summary UX-2 is about, and narrowing here keeps the test honest about
        what was fixed.
        """
        card = re.search(
            r'<article class="var-card" id="[^"]*" data-type="numeric" '
            r'data-name="id".*?<div class="box chart">',
            html,
            re.S,
        )
        assert card, "identifier card not found"
        return card.group(0)

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
        html = profile(frame).html
        card = re.search(
            r'<article class="var-card" id="[^"]*" data-type="numeric" '
            r'data-name="measure".*?<div class="box chart">',
            html,
            re.S,
        )
        assert card
        assert "Mean" in card.group(0)
        assert "Q1 (P25)" in card.group(0)


class TestChipExtraction:
    def test_a_threshold_containing_a_bracket_does_not_break_the_label(self):
        """data-threshold=">1" ends the tag early for a naive [^>]* match."""
        chips = extract_chips(
            '<li class="flag warn" data-threshold=">1" data-value="6.6">Skewed Right</li>'
        )
        assert chips == [("warn", "Skewed Right")]

    def test_a_threshold_with_an_inequality_in_prose(self):
        chips = extract_chips(
            '<li class="flag bad" data-threshold="|kurtosis| > 3" '
            'data-value="9.1">Heavy&#8209;tailed</li>'
        )
        assert chips == [("bad", "Heavy‑tailed")]

    def test_an_unclassed_chip_reports_empty_severity(self):
        assert extract_chips('<li class="flag">Heaping</li>') == [("", "Heaping")]

    def test_slugs_normalise_the_non_breaking_hyphen(self):
        assert flag_slug("Heavy‑tailed") == "heavy-tailed"
        assert flag_slug("Zero‑inflated") == "zero-inflated"


class TestActionableRule:
    def test_a_bad_chip_is_always_actionable(self):
        assert actionable_chips([("bad", "Many outliers")]) == [
            ("bad", "Many outliers")
        ]

    def test_a_good_chip_never_is(self):
        assert actionable_chips([("good", "Positive‑only")]) == []

    def test_distribution_shape_warnings_are_not_defects(self):
        """A standard normal earns both of these; nine clean columns are not
        nine problems."""
        assert actionable_chips([("warn", "Has negatives")]) == []
        assert actionable_chips([("warn", "Some outliers")]) == []
        assert actionable_chips([("warn", "Skewed Right")]) == []

    def test_data_quality_warnings_are(self):
        for label in ("Missing", "Zero‑inflated", "Quasi‑constant", "Imbalanced"):
            assert actionable_chips([("warn", label)]) == [("warn", label)], label


class TestAttentionBlock:
    def test_nothing_flagged_renders_nothing(self):
        """An empty '0 of 60 columns have issues' banner is noise."""
        assert build_attention_block([("a", "a", [("good", "Positive‑only")])]) == ""

    def test_the_banner_counts_flagged_against_total(self):
        block = build_attention_block(
            [
                ("a", "a", [("bad", "Many outliers")]),
                ("b", "b", []),
                ("c", "c", []),
            ]
        )
        assert "<strong>1</strong> of 3 columns need a look" in block

    def test_each_column_links_to_its_card(self):
        block = build_attention_block([("revenue", "col_revenue", [("bad", "X")])])
        assert 'href="#col_revenue"' in block

    def test_the_worst_column_comes_first(self):
        block = build_attention_block(
            [
                ("warned", "w", [("warn", "Missing")]),
                ("broken", "b", [("bad", "Many outliers")]),
            ]
        )
        assert block.index("broken") < block.index("warned")

    def test_column_names_are_escaped(self):
        block = build_attention_block([("<script>", "x", [("bad", "X")])])
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
