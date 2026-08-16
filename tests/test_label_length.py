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
    thing and hid the bug."""

    @pytest.mark.parametrize("value", [None, float("nan")])
    def test_absent_renders_as_a_dash(self, value):
        assert CategoricalCardRenderer()._length_display(value) == "—"

    @pytest.mark.parametrize("value", [0, 1, 1.0, 26.97])
    def test_a_real_length_renders_as_a_number(self, value):
        assert CategoricalCardRenderer()._length_display(value) != "—"

    def test_zero_is_a_length_not_an_absence(self):
        """A column of empty strings has an average length of zero, and zero is
        a measurement."""
        assert CategoricalCardRenderer()._length_display(0) == "0"

    def test_something_unparseable_degrades_rather_than_raising(self):
        assert CategoricalCardRenderer()._length_display("not a number") == "—"


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
