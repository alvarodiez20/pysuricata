"""An empty top-k sketch is an answer, not a zero.

Misra-Gries only guarantees that a value survives if it appears more than
`n/(k+1)` times. `Cabin` in the shipped demo has 204 non-missing values over
147 distinct levels, and its most frequent appears **4** times against a
threshold of exactly 4. Nothing qualifies, so the sketch comes back empty --
and it is right to be empty. The sketch is not the bug.

What the card did with it was. `entropy` became `float("nan")` and rendered
as the literal string `NaN` -- the only one in the whole report -- while
`Rare levels`, `Top 5 coverage` and `Mode %` fell through to their zero
initialisers and printed `0 (0.0%)`, `0.0%` and `0.0%`. Four statistics
stating facts about a column none of them had the data to describe, three of
them looking perfectly plausible.

That is the failure worth guarding: `NaN` announces itself, and `0.0%` does
not.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pysuricata import profile


def _body(html: str) -> str:
    """The report with its inlined CSS and JS removed.

    The report inlines its own stylesheet and scripts, so searching the whole
    document for a token finds it in the very source that defines it.
    """
    return re.sub(r"(?s)<(script|style)\b.*?</\1>", "", html)


@pytest.fixture(scope="module")
def titanic_body() -> str:
    frame = pd.read_csv("docs/assets/titanic.csv")
    return _body(profile(frame, seed=0).html)


class TestNoNaNReachesTheReader:
    def test_the_word_nan_never_renders(self, titanic_body):
        """A float repr leaking into a report is the clearest possible sign
        that a branch was not considered."""
        assert "NaN" not in titanic_body

    def test_nor_does_a_lowercase_one(self, titanic_body):
        assert not re.search(r">\s*nan\s*<", titanic_body)


class TestAnUntrackedColumnSaysSo:
    def test_it_renders_an_em_dash_with_a_reason(self, titanic_body):
        """`Cabin` is the column that triggers it in the demo data."""
        assert "no value repeats often enough" in titanic_body

    def test_the_reason_is_attached_to_a_dash_not_a_number(self, titanic_body):
        for match in re.finditer(
            r'<span title="no value repeats[^"]*">([^<]*)</span>', titanic_body
        ):
            assert match.group(1).strip() == "—", (
                f"expected an em dash, got {match.group(1)!r}"
            )

    def test_a_column_with_real_heavy_hitters_still_reports_numbers(self):
        """The guard must not swallow the normal case.

        Eight levels over 800 rows: the sketch is full of heavy hitters and
        every figure should be a figure.

        This used to be a two-level `sex` column, which #295 stopped rendering
        entropy for at all -- and a fixture that no longer reaches the branch
        reports *absent*, which reads exactly like a pass. Eight levels is
        clear of every suppression rule in `suppressed_statistics`, so the
        only thing that can empty this row is the untracked guard, which is
        what the test is about.
        """
        frame = pd.DataFrame({"grade": list("abcdefgh") * 100})
        body = _body(profile(frame, seed=0).html)
        assert "no value repeats often enough" not in body
        assert re.search(r"Entropy.{0,200}?\d", body, re.S)


class TestTheDistinctionIsRealAndNotCosmetic:
    def test_zero_and_unknown_render_differently(self):
        """The point of the fix: a column that genuinely has no rare levels
        must not look like a column that cannot say."""
        known = pd.DataFrame({"grade": ["a"] * 500 + ["b"] * 300 + ["c"] * 200})
        body = _body(profile(known, seed=0).html)
        # Three levels, all common: rare levels really is 0, and says 0.
        # Three rather than two, because #295 suppresses the row entirely on a
        # two-level column -- there, `0` is not a measurement of a tail, it is
        # the absence of one. The distinction this test is about is between a
        # measured zero and an unmeasurable one, and it needs a column that
        # can be measured.
        assert "no value repeats often enough" not in body
        assert re.search(r"Rare levels.{0,120}?0", body, re.S)

    def test_an_all_distinct_column_does_not_claim_zero_coverage(self):
        """Every value unique and none repeated -- the sketch cannot help, and
        `Top 5 coverage 0.0%` would be a false statement rather than a missing
        one.
        """
        frame = pd.DataFrame({"id": [f"id-{i:05d}" for i in range(3000)]})
        body = _body(profile(frame, seed=0).html)
        top5 = re.search(r"Top 5 coverage.{0,200}", body, re.S)
        assert top5, "the Top 5 coverage row is gone"
        assert "0.0%" not in top5.group(0), (
            "a column where nothing repeats still claims its top five cover 0.0%"
        )
