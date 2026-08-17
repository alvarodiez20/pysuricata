"""Label length: the accumulator had the answer, the card read the wrong object.

The design handoff reported this as an `Embarked` quirk -- a column whose three
labels are all one character long printing `Label length (avg)` as **`NaN`** and
`Length p90` as an em dash.

It was not about `Embarked`, and not about one-character labels. `_right_stats`
read both figures out of `cat_stats`, the dict `_compute_categorical_stats`
builds, and that dict has never carried either key. So `.get(..., float("nan"))`
and `.get(..., "—")` returned their defaults for **every categorical column in
every report** -- `Name`, whose labels average 26.97 characters, printed `NaN`
just as `Embarked` did.

The values were on the stats object the whole time. Same shape as #139: a field
read off an object that does not carry it, failing quietly because the call site
supplied a plausible default.

The em dash is now reserved for genuinely absent, which is what it was always
supposed to mean.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.accumulators.categorical import CategoricalAccumulator
from pysuricata.render.categorical_card import CategoricalCardRenderer


def _lengths(html: str, column: str) -> tuple[str | None, str | None]:
    markup = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)
    start = markup.find(f'id="col_{column}"')
    if start < 0:
        return None, None
    card = markup[start : start + 20_000]
    avg = re.search(r'Label length \(avg\)</div><div class="vstat__val">([^<]*)<', card)
    p90 = re.search(r'Length p90</div><div class="vstat__val">([^<]*)<', card)
    return (avg.group(1) if avg else None), (p90.group(1) if p90 else None)


@pytest.fixture(scope="module")
def titanic() -> str:
    return profile(pd.read_csv("docs/assets/titanic.csv"), seed=0).html


class TestTheFiguresReachTheCard:
    @pytest.mark.parametrize("column", ["Embarked", "Name", "Sex", "Ticket", "Cabin"])
    def test_neither_figure_is_nan_or_a_dash(self, titanic, column):
        avg, p90 = _lengths(titanic, column)
        assert avg not in (None, "NaN", "—"), f"{column} avg = {avg!r}"
        assert p90 not in (None, "NaN", "—"), f"{column} p90 = {p90!r}"

    def test_the_one_character_column_reports_one(self, titanic):
        """The case the handoff named. Three labels, all one character."""
        avg, p90 = _lengths(titanic, "Embarked")
        assert float(avg) == pytest.approx(1.0)
        assert float(p90) == pytest.approx(1.0)

    def test_a_long_label_column_reports_a_long_average(self, titanic):
        """`Name` is the proof this was never about short labels: it printed
        `NaN` too, for labels averaging about 27 characters."""
        avg, p90 = _lengths(titanic, "Name")
        assert float(avg) > 20
        assert float(p90) >= float(avg)

    def test_the_card_matches_the_accumulator(self):
        """The renderer must agree with the object that did the measuring."""
        values = np.array(["alpha", "bb", "c", "dddd"] * 60, dtype=object)
        acc = CategoricalAccumulator("label")
        acc.update(values)
        stats = acc.finalize()

        html = profile(pd.DataFrame({"label": values}), seed=0).html
        avg, _ = _lengths(html, "label")
        assert float(avg) == pytest.approx(float(stats.avg_len), rel=1e-3)


class TestTheEmDashMeansAbsent:
    """It was standing in for *read from the wrong place*, which is a different
    thing and hid the bug.

    Since 5c.2 the dash also says *why*, through `_unknown_cell`, like every
    other unknown on this card: a bare dash leaves a reader unable to tell a
    column with nothing to measure from a report that failed to measure it,
    and those are opposite conclusions about their data.
    """

    @pytest.mark.parametrize("value", [None, float("nan")])
    def test_absent_renders_as_a_dash(self, value):
        rendered = CategoricalCardRenderer()._length_display(value)
        assert rendered.endswith("—</span>")

    def test_the_dash_says_why(self):
        rendered = CategoricalCardRenderer()._length_display(None)
        assert "no non-missing values" in rendered
        assert "title=" in rendered

    @pytest.mark.parametrize("value", [0, 1, 1.0, 26.97])
    def test_a_real_length_renders_as_a_number(self, value):
        assert "—" not in CategoricalCardRenderer()._length_display(value)

    def test_zero_is_a_length_not_an_absence(self):
        """A column of empty strings has an average length of zero, and zero is
        a measurement."""
        assert CategoricalCardRenderer()._length_display(0) == "0"

    def test_something_unparseable_degrades_rather_than_raising(self):
        rendered = CategoricalCardRenderer()._length_display("not a number")
        assert rendered.endswith("—</span>")


class TestTheDictNeverHadTheKeys:
    """A regression guard aimed at the cause rather than the symptom: if someone
    reintroduces the `cat_stats.get("avg_len")` read, this fails even if the
    dict happens to carry the key that day."""

    def test_the_derived_dict_still_does_not_carry_them(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"c": rng.choice(["aa", "b", "ccc"], 300)})
        acc = CategoricalAccumulator("c")
        acc.update(frame["c"].to_numpy())
        derived = CategoricalCardRenderer()._compute_categorical_stats(acc.finalize())
        assert "avg_len" not in derived
        assert "len_p90" not in derived

    def test_the_stats_object_does(self):
        rng = np.random.default_rng(0)
        acc = CategoricalAccumulator("c")
        acc.update(rng.choice(["aa", "b", "ccc"], 300))
        stats = acc.finalize()
        assert stats.avg_len is not None
        assert stats.len_p90 is not None


class TestTheReservoirIsSpent:
    """5c.2 of #155. `categorical.py` has kept a 5,000-value reservoir of label
    lengths all along and the report spent it on two numbers. The whole
    distribution was sitting in it, and on an identifier column the shape *is*
    the finding."""

    def _hist(self, values: list[str]):
        from pysuricata.accumulators.categorical import CategoricalAccumulator
        from pysuricata.accumulators.config import CategoricalConfig

        acc = CategoricalAccumulator("x", CategoricalConfig())
        acc.update(np.array(values, dtype=object))
        return acc.finalize().len_hist

    def test_one_bar_per_distinct_length(self):
        """A label length *is* an integer, and binning hides the thing worth
        seeing: a column of 4- and 7-character values is two formats, and a
        bin of 4-7 is one blur."""
        assert self._hist(["a", "bb", "bb", "cccc"]) == [(1, 1), (2, 2), (4, 1)]

    def test_a_zero_count_is_never_emitted(self):
        """Rule 3. Ten empty lengths drawn as ten 1px bars assert data that is
        not there."""
        hist = self._hist(["a", "ccc"])
        assert all(count > 0 for _, count in hist)
        assert 2 not in [length for length, _ in hist]

    def test_a_wide_range_is_binned_rather_than_drawn_as_hundreds_of_bars(self):
        hist = self._hist(["x" * n for n in range(1, 300)])
        assert len(hist) <= 40


class TestTheLengthPaneOnlyClaimsWhatItCanSee:
    """Two earlier versions got the gap rule wrong in opposite directions, and
    both produced a confident sentence about the reader's data."""

    def _renderer(self):
        from pysuricata.render.categorical_card import CategoricalCardRenderer

        return CategoricalCardRenderer()

    def test_a_sparse_tail_is_not_a_finding(self):
        """`Name` runs 12 to 82 characters with almost every length above 60
        isolated. The first version reported that as "27 separate clusters"."""
        bins = [(12, 100), (14, 200), (16, 300)] + [(n, 1) for n in range(40, 82, 4)]
        finding = self._renderer()._length_finding(bins)
        assert "clusters" not in finding and "groups" not in finding

    def test_a_rare_second_format_is_a_finding(self):
        """`Ticket`'s 40 long tickets in 891 are exactly what this chart exists
        to surface, and a 10% mass rule rejected them."""
        # Titanic's real `Ticket` distribution, not a condensed one: leaving
        # intermediate lengths out invents gaps the data does not have, which
        # is how the first version of this test failed for the wrong reason.
        bins = [
            (3, 2),
            (4, 101),
            (5, 131),
            (6, 419),
            (7, 27),
            (8, 76),
            (9, 26),
            (10, 41),
            (11, 14),
            (12, 8),
            (13, 6),
            (15, 22),
            (16, 12),
            (17, 4),
            (18, 2),
        ]
        finding = self._renderer()._length_finding(bins)
        assert "Two formats in one column" in finding

    def test_the_bin_step_is_not_mistaken_for_a_gap(self):
        """Above the bin cap the histogram groups lengths, so adjacent bins sit
        a full bin apart and every neighbour looks like a gap. `Name` bins at
        width 2, and its twenty-seven gaps were all this artifact."""
        bins = [(n, 50) for n in range(10, 90, 2)]
        finding = self._renderer()._length_finding(bins)
        assert "no gap wide enough" in finding

    def test_a_single_length_is_a_sentence_not_a_chart(self):
        """A chart of one bar at full height says only "all of them"."""
        stats = SimpleNamespace(name="x", len_hist=[(1, 889)])
        html = self._renderer()._build_length_pane(stats)
        assert "Every label is 1 character long" in html
        assert "lenbar" not in html

    def test_two_lengths_are_a_sentence_too(self):
        stats = SimpleNamespace(name="x", len_hist=[(4, 577), (6, 314)])
        html = self._renderer()._build_length_pane(stats)
        assert "either 4 or 6 characters" in html
        assert "lenbar" not in html

    def test_three_lengths_earn_the_chart(self):
        stats = SimpleNamespace(name="x", len_hist=[(4, 10), (5, 20), (6, 5)])
        html = self._renderer()._build_length_pane(stats)
        assert html.count('class="lenbar"') == 3

    def test_no_reservoir_means_no_pane(self):
        assert self._renderer()._build_length_pane(SimpleNamespace(name="x")) == ""

    def test_a_non_zero_count_is_never_invisible(self):
        """The inverse of rule 3, and a different rule. `Ticket` has two labels
        of three characters against a peak of 419 — 0.6px of a 120px chart,
        present in the data and indistinguishable on the page from the zeros
        that are correctly absent."""
        css = (
            Path(__file__).resolve().parents[1]
            / "pysuricata"
            / "static"
            / "css"
            / "_08-categorical.css"
        ).read_text(encoding="utf-8")
        rule = re.search(r"\.lenbar \{(.*?)\}", css, re.S)
        assert rule and "min-height: 1px" in rule.group(1)
