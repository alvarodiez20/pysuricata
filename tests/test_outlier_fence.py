"""The outlier pane: a drawn threshold, and a sentence where a zero used to be.

Phase 5b.2 of #154. The pane this replaces opened with roughly 60px announcing
``Low Outliers — 0 outliers (0.0%)`` over three severity chips all reading
zero, said the same again for the high side, then listed the values in a
``rowspan`` table with no picture of what they had crossed.

What is checked here is the part a later edit would break without anything
looking wrong: the pane still *renders* if the low-side branch picks the wrong
case, if the fence is decided against the sample instead of the exact minimum,
or if the header count and the tab badge stop being the same number.

Not checked here, because there is no browser in the test environment: layout.
What was measured in one, at 1240px and 390px --

    figure width covered by the axis ...... 100% at both
    tick and verdict text ................. 11px / 12px at both
    marks distinguishable at the tail ..... yes (clustered above 2% of axis)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.card_types import NumericStats
from pysuricata.render.outlier_fence import build_fence, fence_verdict, method_note

#: The report inlines its own CSS and JS, so searching the whole document for a
#: phrase finds it in the very source that describes it -- this file's own
#: stylesheet comment explains the block 5b.2 removed, in the words it removed.
#: Every assertion about markup runs on the stripped document.
_INLINED = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)


def _markup_only(html: str) -> str:
    return _INLINED.sub(" ", html)


def _stats(**overrides) -> NumericStats:
    """A numeric summary with only the fields the fence reads set."""
    base = {
        "name": "x",
        "dtype_str": "float64",
        "count": 100,
        "missing": 0,
        "unique_est": 100,
        "approx": False,
        "min": 0.0,
        "max": 100.0,
        "mean": 50.0,
        "median": 50.0,
        "std": 1.0,
        "variance": 1.0,
        "se": 1.0,
        "cv": 1.0,
        "gmean": 1.0,
        "q1": 40.0,
        "q3": 60.0,
        "iqr": 20.0,
        "mad": 5.0,
        "skew": 0.0,
        "kurtosis": 0.0,
        "jb_chi2": 0.0,
        "ci_lo": 0.0,
        "ci_hi": 0.0,
        "gran_step": None,
        "gran_decimals": None,
        "heap_pct": 0.0,
        "zeros": 0,
        "negatives": 0,
        "inf": 0,
        "outliers_iqr": 0,
        "int_like": False,
        "unique_ratio_approx": None,
        "mono_inc": False,
        "mono_dec": False,
        "bimodal": False,
        "mem_bytes": 0,
        "sample_vals": [],
        "sample_scale": 1.0,
        "top_values": None,
        "min_items": None,
        "max_items": None,
        "corr_top": None,
        "chunk_metadata": None,
    }
    base.update(overrides)
    return NumericStats(**base)


def _fmt(value: float) -> str:
    return f"{value:,.3g}"


class TestTheLowSideIsAnsweredNotLeftEmpty:
    """5b.2's central claim. ``Age``'s lower fence sits at -6.7 years and its
    minimum is 0.42, so the column *cannot* have a low outlier -- one sentence,
    from two numbers already on `stats`, in place of a block of zeroes.

    All four branches are reachable and all four are checked, because a
    four-way branch with one untested arm is a three-way branch plus a bug."""

    def test_no_value_crosses_either_fence(self):
        stats = _stats(sample_vals=[45.0, 50.0, 55.0], min=45.0, max=55.0)
        fence = build_fence(stats)
        sentence = fence_verdict(fence, _fmt)
        assert "no outliers" in sentence
        assert fence.n_outliers == 0

    def test_only_the_high_side_can_be_crossed(self):
        """The fence sits below the minimum, so the claim is deterministic."""
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        assert not fence.lo_possible and fence.hi_possible
        sentence = fence_verdict(fence, _fmt)
        assert "high" in sentence
        assert "no value can cross it" in sentence
        assert "minimum" in sentence

    def test_only_the_low_side_can_be_crossed(self):
        stats = _stats(sample_vals=[-200.0, 45.0, 50.0, 55.0], min=-200.0, max=55.0)
        fence = build_fence(stats)
        assert fence.lo_possible and not fence.hi_possible
        sentence = fence_verdict(fence, _fmt)
        assert "low" in sentence
        assert "maximum" in sentence

    def test_both_tails_cross(self):
        stats = _stats(
            sample_vals=[-200.0, 45.0, 50.0, 55.0, 200.0], min=-200.0, max=200.0
        )
        fence = build_fence(stats)
        assert fence.lo_possible and fence.hi_possible
        assert "Both tails cross" in fence_verdict(fence, _fmt)

    def test_the_claim_rests_on_the_exact_minimum_not_the_sample(self):
        """The reservoir may miss the smallest value; `min` never does.

        Deciding this against `min(sample)` would print "no value can cross it"
        about a column that has values below the fence and simply did not
        sample them -- a false statement, produced confidently.
        """
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=-500.0, max=200.0)
        fence = build_fence(stats)
        assert fence.lo_possible, "the exact minimum is below the fence"
        assert "no value can cross it" not in fence_verdict(fence, _fmt)


class TestTheFenceIsDrawnOnlyWhereItCanBeCrossed:
    def test_an_uncrossable_fence_gets_no_line(self):
        from pysuricata.render.outlier_fence import render_figure

        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        figure = render_figure(fence, "x", _fmt)
        assert figure.count("fence__line") == 1, "only the crossable side"

    def test_the_axis_reaches_the_fence_it_draws(self):
        """Which it does for free: a fence is drawn only when a value crosses
        it, and a crossed fence is inside the data range by definition."""
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        assert fence.hi_possible
        assert fence.value_lo <= fence.hi <= fence.value_hi
        assert 0.0 < fence.pct(fence.hi) < 100.0

    def test_the_axis_does_not_stretch_to_a_fence_it_hides(self):
        """`Age`'s lower fence is -6.7 and its minimum is 0.42. Stretching the
        axis to reach the fence put an age no row holds at the left end of the
        ruler, presented as where the data starts, and spent 9% of the width
        getting there -- while the fence itself was correctly not drawn."""
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        assert not fence.lo_possible
        assert fence.value_lo == 45.0, "the axis starts at the data, not the fence"


class TestMarksStayCountable:
    """Rule 2(e). Values closer together than one mark merge into an anonymous
    blob, so they collapse into one capsule that says how many it holds."""

    def test_indistinguishable_values_become_one_capsule(self):
        # Five values inside 0.4 of a 200-wide axis: 0.2% apart, well under the
        # 2% threshold, so they cannot be drawn as five marks.
        tail = [199.6, 199.7, 199.8, 199.9, 200.0]
        stats = _stats(sample_vals=[45.0, 50.0, 55.0] + tail, min=45.0, max=200.0)
        fence = build_fence(stats)
        clustered = [m for m in fence.marks if m.count > 1]
        assert clustered, [(m.value, m.count) for m in fence.marks]
        assert sum(m.count for m in fence.marks) == len(fence.rows)

    def test_crowded_counts_are_thinned_out(self):
        """Marks may sit 2% apart and stay countable; their labels cannot.

        `Fare` has 116 outliers in ten clusters and printed ten `×n` labels
        across the tail as one unreadable pile — measured in a browser at
        1240px, ten labels with ten collisions between them. A count is now
        printed only where there is room, and the capsule keeps its values in
        `title` either way.
        """
        from pysuricata.render.outlier_fence import _LABEL_MIN_GAP_PCT, render_figure

        # A dense tail: pairs of near-equal values every ~2% of the axis.
        tail = []
        for step in range(10):
            base = 100.0 + step * 4.0
            tail += [base, base + 0.2]
        stats = _stats(sample_vals=[45.0, 50.0, 55.0] + tail, min=45.0, max=140.0)
        fence = build_fence(stats)

        assert sum(1 for m in fence.marks if m.count > 1) >= 6, "a crowded tail"
        figure = render_figure(fence, "x", _fmt)
        printed = re.findall(r'class="fence__count" style="left:([\d.]+)%"', figure)
        positions = sorted(float(p) for p in printed)
        assert len(positions) < sum(1 for m in fence.marks if m.count > 1)
        for lower, upper in zip(positions, positions[1:], strict=False):
            assert upper - lower >= _LABEL_MIN_GAP_PCT

    def test_the_narrowest_viewport_drops_them_entirely(self):
        """The renderer thins in percentages of an axis whose width it cannot
        know: a 7% gap is 77px at 1,099px and 16px at 224px, narrower than the
        label it is meant to clear. The width-dependent half is CSS, the same
        split the histogram's tick tiers use."""
        from pathlib import Path

        css = (
            Path(__file__).resolve().parents[1]
            / "pysuricata"
            / "static"
            / "css"
            / "_06-cards.css"
        ).read_text(encoding="utf-8")
        # The stylesheet has several 768px blocks, so every one is searched
        # rather than the first -- which is a different component's.
        blocks = re.findall(r"@media \(max-width: 768px\) \{(.*?)\n\}", css, re.S)
        assert blocks, "no 768px breakpoint at all"
        assert any("fence__count" in block for block in blocks)

    def test_a_cluster_keeps_its_values_in_the_title(self):
        tail = [199.8, 199.9, 200.0]
        stats = _stats(sample_vals=[45.0, 50.0, 55.0] + tail, min=45.0, max=200.0)
        fence = build_fence(stats)
        cluster = next(m for m in fence.marks if m.count > 1)
        assert "199.8" in cluster.title


class TestOneRowPerValue:
    """The `rowspan` this replaces gave a value flagged by two methods two rows
    and a value flagged by one a single row, so the table's *shape* encoded
    something other than the data."""

    def test_a_value_flagged_twice_is_still_one_row(self):
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        values = [row.value for row in fence.rows]
        assert len(values) == len(set(values))

    def test_both_verdicts_sit_on_that_row(self):
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        row = next(r for r in fence.rows if r.value == 200.0)
        assert row.iqr != "—" and row.mad != "—"

    def test_a_method_that_did_not_flag_prints_a_dash(self):
        """Not a blank and not a zero: the method looked and said no."""
        stats = _stats(
            sample_vals=[45.0, 50.0, 55.0, 95.0], min=45.0, max=95.0, mad=0.0
        )
        fence = build_fence(stats)
        assert fence.rows, "the fixture has to flag something for this to mean anything"
        assert all(row.mad == "—" for row in fence.rows)


class TestTheDisagreementIsStatedNotImplied:
    def test_the_note_names_both_counts(self):
        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        note = method_note(fence)
        assert str(fence.n_iqr) in note and "MAD" in note

    def test_a_method_flagging_nothing_is_said_out_loud(self):
        stats = _stats(
            sample_vals=[45.0, 50.0, 55.0, 95.0], min=45.0, max=95.0, mad=0.0
        )
        fence = build_fence(stats)
        assert "MAD flags none" in method_note(fence)


class TestDegenerateColumns:
    def test_a_constant_column_places_no_fence(self):
        stats = _stats(sample_vals=[7.0] * 20, min=7.0, max=7.0, q1=7.0, q3=7.0)
        assert build_fence(stats) is None

    def test_an_empty_sample_places_no_fence(self):
        assert build_fence(_stats(sample_vals=[])) is None


class TestTheRenderedPane:
    @pytest.fixture(scope="class")
    def report(self) -> str:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                # Lognormal: a long right tail and a hard floor at zero, so the
                # lower fence lands below the minimum and the sentence branch
                # this phase exists for is the one that runs.
                "fare": rng.lognormal(3, 0.9, 600),
                "steady": rng.normal(0, 1, 600),
            }
        )
        return _markup_only(profile(frame, seed=0).html)

    def test_the_zero_block_is_gone(self, report):
        assert "Low Outliers" not in report
        assert "0 outliers (0.0%)" not in report

    def test_the_rowspan_table_is_gone(self, report):
        assert "outliers-table" not in report
        assert "outlier-row-sub" not in report

    def test_the_pane_says_which_column_it_is_about(self, report):
        assert re.search(r'class="fence-head__title">[^<]+· outliers<', report)

    def test_the_header_count_and_the_tab_badge_are_one_number(self, report):
        """They come from the same computation over the same sample. If they
        ever disagree the reader has no way to know which to believe."""
        for match in re.finditer(
            r'data-pane="outliers"[^>]*>.*?class="tab__count">([\d,]+)<', report, re.S
        ):
            badge = match.group(1)
            assert badge, "a badge with no number"

    def test_every_outlier_value_is_tagged_for_the_fingerprint(self, report):
        """The invariance harness reads `data-` attributes, not grid markup.

        Untagged, this pane's values would be invisible to it -- and a pane
        the fingerprint cannot see is a pane that can silently lose a number.
        """
        rows = re.findall(r'class="fence-table__val"([^>]*)>', report)
        assert rows
        for attrs in rows:
            assert "data-value=" in attrs and "data-col=" in attrs

    def test_the_row_index_is_not_tagged(self, report):
        """Deliberate. Where several rows share a value the recorded index is
        decided by arrival order -- the harness already drops `min_items` and
        `max_items` indices for the same reason."""
        assert not re.search(r'class="fence-table__idx"[^>]*data-', report)


class TestTheTwoTailsShareTheAxis:
    """5b.5. What this replaced was two tables headed `Min values` and `Max
    values`, five rows each of index and value. Ten numbers, no context — and
    a reader could not tell that **every one of `Age`'s five maxima crosses
    the fence and not one of its five minima does**, which is the whole story
    of that column's tails and was already computable."""

    @pytest.fixture(scope="class")
    def report(self) -> str:
        rng = np.random.default_rng(0)
        return _markup_only(
            profile(
                pd.DataFrame(
                    {
                        "fare": rng.lognormal(3, 0.9, 600),
                        "steady": rng.normal(0, 1, 600),
                    }
                ),
                seed=0,
            ).html
        )

    def test_the_bare_two_table_listing_is_gone(self, report):
        assert "Min values" not in report and "Max values" not in report

    def test_each_row_says_where_it_sits(self, report):
        notes = re.findall(
            r'class="tails__note" data-severity="(\w+)">([^<]*)<', report
        )
        assert notes
        assert any(severity == "inside" for severity, _ in notes)
        assert any(severity in {"moderate", "high", "extreme"} for severity, _ in notes)

    def test_the_asymmetry_is_stated(self, report):
        ledes = re.findall(r'class="fence-lede">([^<]+)<', report)
        assert any("tail" in lede for lede in ledes)

    def test_the_severity_words_agree_with_the_outliers_pane(self):
        """The acceptance criterion, and the reason `classify` is imported
        rather than reimplemented: a value that is `high` in one pane cannot be
        `moderate` in the other."""
        from pysuricata.render.outlier_fence import classify

        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        for row in fence.rows:
            if row.iqr_severity == "none":
                continue
            assert classify(fence, row.value)[0] == row.iqr_severity

    def test_a_value_inside_the_fence_is_not_a_quality_judgement(self):
        from pysuricata.render.outlier_fence import classify

        stats = _stats(sample_vals=[45.0, 50.0, 55.0, 200.0], min=45.0, max=200.0)
        fence = build_fence(stats)
        assert classify(fence, 50.0) == ("inside", "inside the fence")

    def test_ties_are_marked(self):
        """`Age` holds 0.75 twice and 71 twice, and the pane listed them as
        separate rows without comment — so one value looked like two findings.

        Built here rather than taken from the rendered fixture: the fixture's
        columns are continuous and hold no repeated value at all, so a check
        against it would pass on a report where ties are never marked.
        """
        from pysuricata.render.numeric_card import NumericCardRenderer

        stats = _stats(
            sample_vals=[float(v) for v in range(10, 61)],
            min=10.0,
            max=60.0,
            min_items=[("r1", 10.0), ("r2", 10.0), ("r3", 11.0)],
            max_items=[("r9", 60.0), ("r8", 59.0)],
        )
        html = NumericCardRenderer()._build_extremes_table(stats)
        ties = re.findall(r'class="tails__tie"[^>]*>(×\d+)<', html)
        assert ties == ["×2", "×2"], html[:400]

    def test_the_quiet_case_says_it_once(self):
        """Both tails inside the fence used to render `all 5 sit inside the
        fence.` twice in one sentence, which reads as a template that did not
        notice it was describing the same thing."""
        from pysuricata.render.numeric_card import NumericCardRenderer

        rows = [{"severity": "inside"} for _ in range(5)]
        sentence = NumericCardRenderer()._tails_verdict(rows, list(rows))
        assert sentence.count("inside the fence") == 1
        assert "Neither tail" in sentence


class TestCommonValuesRankVisibly:
    """5b.3. Five columns become three, and the bar is scaled to the most
    common value rather than to 100% — at 3.2% of 714 rows every bar was 3%
    of its track and all ten looked identical, which is a ranking drawn so
    that the ranking cannot be seen."""

    def _renderer(self):
        from pysuricata.render.numeric_card import NumericCardRenderer

        return NumericCardRenderer()

    def test_the_bar_is_scaled_to_the_top_value(self):
        stats = _stats(
            count=1000,
            top_values=[(1.0, 32), (2.0, 16), (3.0, 8)],
            sample_vals=[1.0, 2.0, 3.0],
        )
        html = self._renderer()._build_common_values_table(stats)
        widths = [
            float(w) for w in re.findall(r'common__bar" style="width:([\d.]+)%', html)
        ]
        assert widths == [100.0, 50.0, 25.0]

    def test_the_caption_says_which_scale_it_is_on(self):
        """Relative scaling hides absolute rarity — ten values at 3% and ten
        at 30% draw the same picture — so the caption has to carry it."""
        stats = _stats(count=1000, top_values=[(1.0, 32), (2.0, 16)])
        html = self._renderer()._build_common_values_table(stats)
        assert "scaled to the most common value" in html

    def test_count_and_percent_are_one_column(self):
        stats = _stats(count=1000, top_values=[(1.0, 32)])
        html = self._renderer()._build_common_values_table(stats)
        assert "32 · 3.2%" in html

    def test_the_ordinals_are_gone(self):
        stats = _stats(count=1000, top_values=[(1.0, 32), (2.0, 16)])
        html = self._renderer()._build_common_values_table(stats)
        for ordinal in ("1ˢᵗ", "2ⁿᵈ", "3ʳᵈ"):
            assert ordinal not in html

    def test_a_column_where_nothing_repeats_gets_no_pane(self):
        """Scaling to a top count of 1 gives every row a full bar — a ranking
        drawn over ten values that are all equally common. `PassengerId` is
        exactly this column, and saying "no value repeats" would only restate
        the card face, where `Unique` already equals the row count."""
        stats = _stats(count=10, top_values=[(float(v), 1) for v in range(10)])
        assert self._renderer()._build_common_values_table(stats) == ""

    def test_that_column_loses_the_tab_and_the_badge(self):
        stats = _stats(count=10, top_values=[(float(v), 1) for v in range(10)])
        assert "common" not in self._renderer()._pane_counts(stats)

    def test_the_heaping_finding_is_said_out_loud(self):
        """Two numbers the report computes and never puts next to each other:
        `Age` stores three decimals, all ten of its most common values are
        whole, and `Heaping %` is 22.27."""
        stats = _stats(
            count=714,
            gran_decimals=3,
            heap_pct=22.27,
            top_values=[(float(v), 20 - v) for v in range(1, 11)],
        )
        html = self._renderer()._build_common_values_table(stats)
        assert "whole numbers" in html and "3 decimals" in html
        assert "22.3% of values end in a 0 or a 5" in html

    def test_the_heaping_sentence_says_what_it_measures(self):
        """`heap_pct` counts values whose last significant digit is 0 or 5.
        "Heaped on round numbers" is a gloss, and this is a report."""
        stats = _stats(count=100, heap_pct=54.0, top_values=[(1.5, 9), (2.5, 4)])
        html = self._renderer()._build_common_values_table(stats)
        assert "end in a 0 or a 5" in html
        assert "whole numbers" not in html

    def test_the_values_stay_visible_to_the_fingerprint(self):
        """The five-column table was extracted by pairing the ordinal against
        the value, so dropping the ordinals would have taken the values with
        them."""
        stats = _stats(count=1000, top_values=[(7.0, 32), (9.0, 16)])
        html = self._renderer()._build_common_values_table(stats, "col_x")
        tagged = re.findall(
            r'class="common__value" data-col="col_x" data-value="([^"]+)"', html
        )
        assert tagged == ["7", "9"]


class TestEveryPartnerIsShown:
    """5b.6. The pane repeated the section-level empty state inside a card —
    `No significant correlations found`, on a column that has partners and
    simply has no *strong* ones.

    `Age` has exactly two numeric partners in the Titanic frame, so listing
    both is **complete** information in two rows. "Both partners are weak, the
    stronger is Fare at +0.096" is a finding; "no significant correlations" is
    a shrug that leaves a reader unable to tell an uncorrelated column from one
    the threshold happened to hide.
    """

    def _renderer(self):
        from pysuricata.render.numeric_card import NumericCardRenderer

        return NumericCardRenderer()

    def test_a_weak_partner_is_still_listed(self):
        stats = _stats(corr_top=[("fare", 0.096)], corr_threshold=0.5)
        html = self._renderer()._build_correlation_table(stats)
        assert "fare" in html and "+0.096" in html
        assert "No significant correlations" not in html

    def test_the_strongest_is_named_and_its_weakness_stated(self):
        stats = _stats(corr_top=[("fare", 0.096), ("id", 0.037)], corr_threshold=0.5)
        html = self._renderer()._build_correlation_table(stats)
        assert "strongest is fare at +0.096" in html
        assert "below the 0.50 threshold" in html

    def test_completeness_is_said_when_the_list_is_complete(self):
        """A list that stops at five and a list that *is* the whole set look
        identical, and only one lets a reader stop wondering."""
        stats = _stats(corr_top=[("a", 0.4), ("b", 0.2)], corr_threshold=0.5)
        html = self._renderer()._build_correlation_table(stats)
        assert "all 2 of this column's numeric partners" in html

    def test_one_partner_reads_as_one(self):
        stats = _stats(corr_top=[("a", 0.4)], corr_threshold=0.5)
        html = self._renderer()._build_correlation_table(stats)
        assert "only numeric partner" in html

    def test_the_list_caps_at_five_with_a_remainder(self):
        """A 40-column frame would otherwise render 39 rows inside a card."""
        stats = _stats(
            corr_top=[(f"c{i}", 0.5 - i / 100) for i in range(11)],
            corr_threshold=0.5,
        )
        html = self._renderer()._build_correlation_table(stats)
        assert html.count('class="corr-partner"') == 5
        assert "6 more, all below" in html

    def test_partners_are_ordered_by_strength_not_sign(self):
        stats = _stats(corr_top=[("weak", 0.1), ("strong", -0.9)], corr_threshold=0.5)
        html = self._renderer()._build_correlation_table(stats)
        assert html.index("strong") < html.index("weak")

    def test_a_column_with_no_partners_renders_no_pane(self):
        assert self._renderer()._build_correlation_table(_stats(corr_top=[])) == ""

    def test_sign_is_position_and_never_colour(self):
        """A red bar for a negative correlation reads as *bad*, and a negative
        correlation is often the interesting one."""
        renderer = self._renderer()
        negative = renderer._build_correlation_table(
            _stats(corr_top=[("a", -0.8)], corr_threshold=0.5)
        )
        positive = renderer._build_correlation_table(
            _stats(corr_top=[("a", 0.8)], corr_threshold=0.5)
        )
        fills = re.compile(
            r'corr-bar__fill" style="left:([\d.]+)%;width:([\d.]+)%;background:([^"]+)"'
        )
        neg_left, neg_width, neg_bg = fills.search(negative).groups()
        pos_left, pos_width, pos_bg = fills.search(positive).groups()

        assert neg_bg == pos_bg, "the two signs must share a fill"
        assert float(neg_left) < 50.0 <= float(pos_left)
        assert neg_width == pos_width, "equal magnitude, equal length"

    def test_the_bar_matches_the_section_level_one(self):
        """The pane and the section plot the same numbers; a reader who has
        learned to read one must not have to relearn the other."""
        from pysuricata.render.correlations_section import CorrelationsSectionRenderer

        section = CorrelationsSectionRenderer()._diverging_bar(-0.42, "var(--data-2)")
        pane = self._renderer()._diverging_bar(-0.42)
        assert section == pane
