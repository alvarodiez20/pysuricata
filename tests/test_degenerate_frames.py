"""Degenerate frames are the branch most likely to be absent from every fixture.

From #299. A frame with one column, no rows, one row, or a single column kind
was never designed for, and `CLAUDE.md` records the reason it stays broken
quietly: *a fixture that misses a branch reports "absent", not "unknown", and
absent reads as broken.* These shapes were absent from every fixture in the
suite, so nothing said whether they worked.

The investigation found they mostly do. This file is what keeps that true.

Four defects came out of the investigation and all four are now fixed, so the
cases below assert the right answers rather than pinning the wrong ones:

* #312 -- a zero-column frame reported 9 duplicate rows where pandas reports 0
* #313, #315 -- a zero-row frame rendered a bare page and returned `{}`
* #314 -- flags that fire by construction, contradictory quick facts, and `-0`

Each arrived here as a strict xfail while it was open, which is what turned the
fix into a passing test rather than a claim.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize

_TAGS = re.compile(r"<script.*?</script>|<style.*?</style>", re.S)

#: Every degenerate shape the issue names, plus the three neighbours the
#: investigation showed behave differently enough to be worth their own row.
SHAPES: dict[str, pd.DataFrame] = {
    "one_column_numeric": pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
    "one_column_categorical": pd.DataFrame({"a": list("abcabcabca")}),
    "one_row": pd.DataFrame(
        {
            "a": [1.0],
            "b": ["x"],
            "c": [True],
            "d": pd.to_datetime(["2024-01-01"]),
        }
    ),
    "one_col_one_row": pd.DataFrame({"a": [1.0]}),
    "all_numeric": pd.DataFrame({f"n{i}": np.arange(50.0) + i for i in range(3)}),
    "all_categorical": pd.DataFrame({f"c{i}": list("abcde") * 10 for i in range(3)}),
    "all_boolean": pd.DataFrame({f"b{i}": [True, False] * 25 for i in range(3)}),
    "zero_columns": pd.DataFrame(index=range(10)),
    "all_missing": pd.DataFrame({"a": [np.nan] * 20, "b": [None] * 20}),
    "constant_column": pd.DataFrame({"a": [7.0] * 50}),
}

#: Rendered separately: it currently takes the bare-page path (#313), so it
#: cannot be asserted against the report shell alongside the others.
ZERO_ROWS = pd.DataFrame(
    {"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")}
)


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return {name: profile(frame, seed=0).html for name, frame in SHAPES.items()}


@pytest.mark.parametrize("name", sorted(SHAPES))
class TestItRendersAtAll:
    """#299's first acceptance criterion, and the cheapest thing to lose."""

    def test_it_does_not_raise(self, rendered: dict[str, str], name: str) -> None:
        assert rendered[name]

    def test_it_is_a_whole_report_and_not_a_fallback_page(
        self, rendered: dict[str, str], name: str
    ) -> None:
        """The bare-page path is 221 bytes and carries no stylesheet. Landing
        on it is how #313 stayed invisible: it does not raise, it just stops
        being a report."""
        html = rendered[name]

        assert "Empty source." not in html
        assert "<style>" in html
        assert 'id="summary"' in html


@pytest.mark.parametrize("name", sorted(SHAPES))
class TestNothingUndefinedReachesThePage:
    """#299's second and third criteria."""

    #: Words that only appear when a division by zero, an empty reduction or a
    #: missing key has been formatted straight into the document. Bounded so
    #: `nan` does not match inside a column name.
    _LEAKS = ("nan", "NaN", "inf", "Infinity", "None", "undefined", "0/0")

    def test_no_arithmetic_leaks_into_the_text(
        self, rendered: dict[str, str], name: str
    ) -> None:
        text = re.sub(r"<[^>]+>", " ", _TAGS.sub("", rendered[name]))
        found = {
            leak
            for leak in self._LEAKS
            if re.search(rf"(?<![A-Za-z]){re.escape(leak)}(?![A-Za-z])", text)
        }

        assert not found, f"{name} renders {sorted(found)} as a value"

    def test_no_bar_is_drawn_for_a_zero_count(
        self, rendered: dict[str, str], name: str
    ) -> None:
        """Rule 3 in `tokens.css`: a zero count draws nothing. A one-pixel bar
        for no rows is a mark that says there is something there."""
        body = _TAGS.sub("", rendered[name])
        drawn = [
            bar
            for bar in re.findall(r'<rect class="bar"[^>]*data-count="0"[^>]*/>', body)
            if not re.search(r'height="0(\.0+)?"', bar)
        ]

        assert not drawn, f"{name} draws {len(drawn)} bars for a zero count"


class TestTheDuplicateCountAgreesWithPandas:
    """The neighbours of #312, which are correct and must stay correct.

    All-missing rows and constant rows genuinely *are* duplicate rows, and
    pandas says so. They look like the zero-column defect and are not it, so
    they are pinned here to stop #312's fix taking them with it.
    """

    @pytest.mark.parametrize("name", ["all_missing", "constant_column", "all_numeric"])
    def test_it_matches_pandas(self, name: str) -> None:
        frame = SHAPES[name]
        payload = summarize(frame)
        dataset = payload.get("dataset", payload)

        assert dataset["duplicate_rows_est"] == int(frame.duplicated().sum())

    def test_a_zero_column_frame_matches_pandas_too(self) -> None:
        """Was an xfail for #312.

        Ten empty rows came back as nine duplicates, 90%, labelled `exact`: the
        row hasher raised `IndexError` on `columns[0]` and the surrounding
        `except` routed it into the fallback for *unhashable* chunks, which
        stringified nothing into one signature for the lot. A frame with no
        columns has nothing in its rows to compare, which is why pandas reports
        no duplicates, and now so does this.
        """
        frame = SHAPES["zero_columns"]
        payload = summarize(frame)
        dataset = payload.get("dataset", payload)

        assert dataset["duplicate_rows_est"] == int(frame.duplicated().sum()) == 0
        assert dataset["duplicate_rows_pct_est"] == 0.0
        assert dataset["duplicate_rows_uncertainty"] == 0

    def test_the_zero_column_frame_is_not_reported_as_degraded(self) -> None:
        """It took the fallback for chunks that *could not be hashed*, and that
        flag is how a consumer learns the figure is an overestimate. Nothing
        failed here: there was simply nothing to hash."""
        payload = summarize(SHAPES["zero_columns"])
        dataset = payload.get("dataset", payload)

        assert dataset["duplicates_degraded"] is False


class TestTheZeroRowFrame:
    """#313. It does not raise, which is the part worth guarding now."""

    def test_it_does_not_raise(self) -> None:
        assert profile(ZERO_ROWS, seed=0).html

    def test_summarize_honours_its_contract(self) -> None:
        """`docs/versioning.md` makes this payload the one guaranteed surface,
        and a zero-row frame used to be where it returned nothing at all — not
        an error, not a zeroed payload, but `{}` (#315).

        The dataset key is `rows_est`, not `n_rows`. This test asserted
        `n_rows` while it was an xfail, so it was failing for a reason that had
        nothing to do with the defect it named — which an xfail cannot tell you,
        because a test that fails for the wrong reason looks exactly like a test
        that fails for the right one.
        """
        payload = summarize(ZERO_ROWS)

        assert "schema_version" in payload
        assert payload["dataset"]["rows_est"] == 0
        assert payload["dataset"]["cols"] == 2
        assert len(payload["columns"]) == 2

    def test_every_column_keeps_its_name_and_dtype(self) -> None:
        """The schema is the whole reason the payload is worth producing here:
        *did my filter match nothing, or did I select the wrong columns?*"""
        columns = summarize(ZERO_ROWS)["columns"]

        assert set(columns) == {"a", "b"}
        assert columns["a"]["dtype"] == "float64"
        assert columns["b"]["dtype"] == "object"

    def test_no_statistic_is_invented(self) -> None:
        """A count over an empty set is zero; a statistic over an empty set is
        undefined. `min` and `mean` of `0.0` for a column with no values is a
        reading invented rather than declined."""
        column = summarize(ZERO_ROWS)["columns"]["a"]

        assert column["count"] == 0
        for statistic in ("mean", "std", "min", "q1", "median", "q3", "max"):
            assert column[statistic] is None, statistic

    def test_the_payload_is_strict_json(self) -> None:
        """`NaN` is what Python emits by default and is not JSON any other
        language will read. The manifest is documented as JSON-safe."""
        import json

        json.dumps(summarize(ZERO_ROWS), allow_nan=False)

    def test_it_renders_the_schema_it_knows(self) -> None:
        """#313, fixed by the same change: the bare-page path is 221 bytes and
        carries no stylesheet, so a zero-row frame looked like a crash."""
        html = profile(ZERO_ROWS, seed=0).html

        assert "Empty source." not in html
        assert "<style>" in html
        assert 'id="summary"' in html


class TestAFlagThatCannotFailIsNotAFinding:
    """#314. A one-row frame raised `100.0% dominant category · limit 50%`.

    A column with one row has one value, so its most common value is 100% of it
    whatever the data: the flag could not *not* fire. That lands in the one
    block designed to say what needs a look, on exactly the frames a new user
    is most likely to start with.
    """

    def _attention(self, frame: pd.DataFrame) -> str:
        """The attention block's text, whitespace collapsed *first*.

        Stripping tags leaves a run of spaces wherever one was, so the phrase
        is searched for after collapsing rather than before -- otherwise the
        pattern misses and an empty string reads as "no flags raised", which is
        the answer these cases are trying to tell apart.
        """
        text = re.sub(r"<[^>]+>", " ", _TAGS.sub("", profile(frame, seed=0).html))
        text = re.sub(r"\s+", " ", text)
        found = re.search(r"\d+ of \d+ columns need a look.{0,200}", text)
        return found.group(0) if found else ""

    def test_a_one_row_frame_raises_nothing(self) -> None:
        assert self._attention(SHAPES["one_col_one_row"]) == ""

    def test_two_distinct_values_are_not_a_dominant_category(self) -> None:
        """The bar was `int(threshold * count)`, and truncation makes that 1 at
        two rows -- the smallest a mode can be -- so two distinct values were
        flagged as having a dominant category."""
        assert "dominant category" not in self._attention(
            pd.DataFrame({"a": ["x", "y"]})
        )

    def test_the_flag_still_fires_where_it_means_something(self) -> None:
        """The guard suppresses what cannot fail, not what is true."""
        attention = self._attention(pd.DataFrame({"a": ["x"] * 8 + ["y", "z"]}))

        assert "80.0% dominant category" in attention

    def test_an_all_missing_frame_still_reports_its_missingness(self) -> None:
        """100% missing is not an artefact of the row count."""
        assert "100.0% missing" in self._attention(SHAPES["all_missing"])


class TestTheQuickFactsAgreeWithThemselves:
    """#314. One column was counted as unique *and* constant *and*
    high-cardinality: `1 unique · 1 constant · 1 high-cardinality`.

    Each is individually defensible at n = 1 and the three together are
    nonsense. `unique` was not a property at all -- it was the column count --
    so every column was always in it.
    """

    def _facts(self, frame: pd.DataFrame) -> str:
        html = profile(frame, seed=0).html
        found = re.search(r'class="quick-facts">([^<]*)<', html)
        assert found, "the quick-facts line is missing"
        return found.group(1)

    def test_a_one_row_column_lands_in_one_bucket(self) -> None:
        facts = self._facts(SHAPES["one_col_one_row"])

        buckets = [
            b for b in ("all distinct", "constant", "high-cardinality") if b in facts
        ]
        assert buckets == ["constant"], facts

    def test_a_column_with_no_values_lands_in_none(self) -> None:
        """Neither unique nor constant is a property of a column holding
        nothing. Two all-NaN columns were counted as `2 unique · 2 constant`."""
        facts = self._facts(SHAPES["all_missing"])

        for bucket in ("all distinct", "constant", "high-cardinality"):
            assert bucket not in facts, facts

    def test_a_real_frame_still_describes_itself(self) -> None:
        """The buckets stay useful on a frame that has something to say."""
        frame = pd.DataFrame(
            {
                "id": [f"k{i}" for i in range(50)],
                "same": ["x"] * 50,
                "n": np.arange(50.0),
            }
        )

        facts = self._facts(frame)
        assert "1 constant" in facts
        assert "all distinct" in facts


class TestNegativeZeroNeverReachesThePage:
    """#314. The categorical card rendered `ENTROPY -0` on a one-row column.

    One level at p = 1 gives -(1 * log2(1)) = -0.0. The value is right and its
    rendering is not: a leading minus reads as a measurement that came out
    slightly negative. Caught in the shared formatter rather than at the call
    site, because every formatter is a place it can surface.
    """

    def test_the_formatter_normalises_it(self) -> None:
        from pysuricata.render.format_utils import fmt_compact, fmt_num

        assert fmt_num(-0.0) == "0"
        assert fmt_compact(-0.0) == "0"

    def test_it_still_formats_a_real_negative(self) -> None:
        from pysuricata.render.format_utils import fmt_num

        assert fmt_num(-0.4) == "-0.4"

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_no_shape_renders_a_bare_negative_zero(self, name: str) -> None:
        text = re.sub(
            r"<[^>]+>", " ", _TAGS.sub("", profile(SHAPES[name], seed=0).html)
        )

        assert not re.search(r"(?<![\d.])-0(?![\d.])", text), f"{name} renders -0"
