"""Degenerate frames are the branch most likely to be absent from every fixture.

From #299. A frame with one column, no rows, one row, or a single column kind
was never designed for, and `CLAUDE.md` records the reason it stays broken
quietly: *a fixture that misses a branch reports "absent", not "unknown", and
absent reads as broken.* These shapes were absent from every fixture in the
suite, so nothing said whether they worked.

The investigation found they mostly do. This file is what keeps that true.

**What it deliberately does not assert.** Four defects were found and are filed
rather than pinned here, because a test asserting today's wrong answer makes
the wrong answer permanent:

* #312 -- a zero-column frame reports 9 duplicate rows where pandas reports 0
* #313 -- a zero-row frame renders a bare unstyled page
* #314 -- flags that fire by construction, contradictory quick facts, and `-0`
* #315 -- `summarize()` returns `{}` for a zero-row frame, so the one part of
  the surface `docs/versioning.md` guarantees is the part that breaks

The cases below are written to pass **either** side of #312 and #313 landing,
so they guard the shapes without freezing the bugs.
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

    @pytest.mark.xfail(
        reason="#315: summarize() returns {} for a zero-row frame, so every "
        "documented key -- schema_version included -- is absent",
        strict=True,
    )
    def test_summarize_honours_its_contract(self) -> None:
        """`docs/versioning.md` makes this payload the one guaranteed surface,
        and a zero-row frame is where it returns nothing at all."""
        payload = summarize(ZERO_ROWS)

        assert "schema_version" in payload
        assert payload["dataset"]["n_rows"] == 0
        assert len(payload["columns"]) == 2

    @pytest.mark.xfail(
        reason="#313: a zero-row frame takes the bare-page path, so the schema "
        "it does know is never rendered",
        strict=True,
    )
    def test_it_renders_the_schema_it_knows(self) -> None:
        html = profile(ZERO_ROWS, seed=0).html

        assert "Empty source." not in html
        assert "<style>" in html
