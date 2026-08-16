"""A case or trim flag fires only when normalising would actually merge levels.

The *Mobile UI 3* handoff reports `Embarked` carrying both `Case variants` and
`Trim variants` while its normalization pane finds no collisions to justify
either -- three levels, `S`, `C`, `Q`, all one upper-case character with no
surrounding whitespace.

**It does not reproduce.** The guard the handoff asks for is already in
`card_base`: a flag requires the lowercased or stripped distinct count to be
*strictly smaller* than the raw one, so "the estimator ran" is not mistaken
for "the estimator found something".

This file exists because a bug that has been fixed and never tested is a bug
waiting to come back, and because the next reader of that handoff needs to be
able to tell "already fixed" from "nobody looked". It asserts the guard in
both directions -- a clean column carries no flag, and a dirty one does --
since a guard that never fires would satisfy the first half and be useless.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.api import summarize


def _markup(frame: pd.DataFrame) -> str:
    """The report without its inlined CSS and JS."""
    return re.sub(r"(?s)<(script|style)\b.*?</\1>", "", profile(frame, seed=0).html)


class TestACleanColumnCarriesNoVariantFlag:
    def test_the_column_the_handoff_named(self):
        frame = pd.read_csv("docs/assets/titanic.csv")
        stats = summarize(frame, seed=0)["columns"]["Embarked"]
        assert stats["unique_est"] == stats["case_variants_est"]
        assert stats["unique_est"] == stats["trim_variants_est"]

    def test_and_no_flag_reaches_the_report(self):
        markup = _markup(pd.read_csv("docs/assets/titanic.csv")).lower()
        assert "case variant" not in markup
        assert "trim variant" not in markup


class TestADirtyColumnStillCarriesOne:
    """Otherwise the guard above is satisfied by a flag that never fires."""

    def test_case_variants_are_still_detected(self):
        frame = pd.DataFrame({"answer": ["yes", "Yes", "YES", "no", "No"] * 200})
        stats = summarize(frame, seed=0)["columns"]["answer"]
        assert stats["case_variants_est"] < stats["unique_est"], (
            "lowercasing should collapse five levels to two"
        )
        assert "case variant" in _markup(frame).lower()

    def test_trim_variants_are_still_detected(self):
        frame = pd.DataFrame({"city": ["Madrid", " Madrid", "Madrid ", "Lisbon"] * 250})
        stats = summarize(frame, seed=0)["columns"]["city"]
        assert stats["trim_variants_est"] < stats["unique_est"]
        assert "trim variant" in _markup(frame).lower()

    @pytest.mark.parametrize(
        ("frame", "fires", "quiet"),
        [
            (
                pd.DataFrame({"a": ["yes", "Yes", "YES", "no", "No"] * 200}),
                "case variant",
                "trim variant",
            ),
            (
                pd.DataFrame({"a": ["Madrid", " Madrid", "Madrid ", "Lisbon"] * 250}),
                "trim variant",
                "case variant",
            ),
        ],
        ids=["case-only", "trim-only"],
    )
    def test_the_two_flags_do_not_fire_for_each_other(self, frame, fires, quiet):
        """The specific contradiction the handoff describes: a column carrying
        both when only one condition holds."""
        markup = _markup(frame).lower()
        assert fires in markup
        assert quiet not in markup
