"""Degenerate frames are the branch most likely to be absent from every fixture.

From #299. A frame with one column, no rows, one row, or a single column kind
was never designed for, and `CLAUDE.md` records the reason it stays broken
quietly: *a fixture that misses a branch reports "absent", not "unknown", and
absent reads as broken.* These shapes were absent from every fixture in the
suite, so nothing said whether they worked.

The investigation found they mostly do. This file is what keeps that true.

**What it deliberately does not assert.** One defect is still filed rather than
pinned here, because a test asserting today's wrong answer makes the wrong
answer permanent:

* #312 -- a zero-column frame reports 9 duplicate rows where pandas reports 0

#313 and #315 were both fixed by treating a zero-row frame as a frame with a
schema rather than as an empty source, and their cases below are ordinary
assertions now rather than xfails. #314 is fixed too: the three quick-facts
buckets are mutually exclusive and empty below two values, a share-based flag
only fires where an even spread would not have fired it, and negative zero is
caught in the formatter. Those are ordinary assertions below as well.

The remaining cases are written to pass **either** side of #312 landing, so
they guard the shapes without freezing the bug.
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


def _quick_facts(html: str) -> dict[str, int]:
    """The Summary's `n unique · n constant · n high-cardinality` counts."""
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", _TAGS.sub("", html)))
    match = re.search(
        r"([\d,]+) unique · ([\d,]+) constant · ([\d,]+) high-cardinality", text
    )
    assert match, "the quick-facts line is gone -- this check reads nothing"
    unique, constant, high_card = (int(g.replace(",", "")) for g in match.groups())
    return {"unique": unique, "constant": constant, "high_card": high_card}


@pytest.mark.parametrize("name", sorted(SHAPES))
class TestAColumnIsCountedInAtMostOneBucket:
    """#314. `pd.DataFrame({"a": [1.0]})` was described as all-unique **and**
    constant **and** high-cardinality at once. Each is individually defensible
    at n = 1 and the three together are nonsense."""

    def test_the_three_buckets_do_not_double_count(
        self, rendered: dict[str, str], name: str
    ) -> None:
        facts = _quick_facts(rendered[name])
        n_cols = SHAPES[name].shape[1]

        assert sum(facts.values()) <= n_cols, (
            f"{name} has {n_cols} column(s) and counts {sum(facts.values())} "
            f"bucket memberships: {facts}"
        )

    def test_a_column_with_no_values_is_counted_in_none_of_them(
        self, rendered: dict[str, str], name: str
    ) -> None:
        """Neither unique nor constant is a property a column with zero values
        has, and `all_missing` holds two of them."""
        if name != "all_missing":
            pytest.skip("only the all-missing frame has valueless columns")

        assert _quick_facts(rendered[name]) == {
            "unique": 0,
            "constant": 0,
            "high_card": 0,
        }


class TestAFlagDoesNotFireWhereItCannotFail:
    """#314. A column with one row has one value, so its most common value is
    100% of it -- `dominant category` cannot *not* fire. Same family as #248,
    where the duplicate threshold false-alarms on a clean frame."""

    def test_a_single_row_raises_no_dominance_flag(
        self, rendered: dict[str, str]
    ) -> None:
        body = _TAGS.sub("", rendered["one_col_one_row"])

        assert "dominant-category" not in body
        assert "quasi-constant" not in body

    def test_the_same_flag_still_fires_where_it_means_something(self) -> None:
        """The guard must not have bought its silence by turning the flag off:
        75% of one level in 100 rows is a real dominant category."""
        frame = pd.DataFrame({"a": ["x"] * 75 + [f"v{i}" for i in range(25)]})
        body = _TAGS.sub("", profile(frame, seed=0).html)

        assert "dominant-category" in body

    def test_the_reference_table_states_the_limit_that_is_applied(self) -> None:
        """The block exists to tell a reader why a chip is on their column. It
        said 50% while `dominant_category_threshold` was 0.7, so a 60%-dominant
        column cleared the stated limit and did not fire."""
        from pysuricata.render.card_config import DEFAULT_QUALITY_THRESHOLDS
        from pysuricata.render.flag_reference import FLAG_MEANINGS

        stated = FLAG_MEANINGS["dominant-category"].limit
        applied = DEFAULT_QUALITY_THRESHOLDS.dominant_category_threshold

        assert stated == f"{applied:.0%}"


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_no_negative_zero_reaches_the_page(rendered: dict[str, str], name: str) -> None:
    """#314. A one-level column's entropy is `-sum([1.0 * log2(1.0)])`, which
    is IEEE negative zero. The value is correct; its rendering was not."""
    body = _TAGS.sub("", rendered[name])
    found = re.findall(r">\s*-0(?:\.0+)?\s*<", body)

    assert not found, f"{name} renders negative zero {len(found)} time(s)"


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

    @pytest.mark.xfail(
        reason="#312: the zero-column frame routes through the unhashable-chunk "
        "fallback, which lands on one distinct signature for every row",
        strict=True,
    )
    def test_a_zero_column_frame_matches_pandas_too(self) -> None:
        frame = SHAPES["zero_columns"]
        payload = summarize(frame)
        dataset = payload.get("dataset", payload)

        assert dataset["duplicate_rows_est"] == int(frame.duplicated().sum()) == 0


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
