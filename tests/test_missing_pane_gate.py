"""The Missing Values pane is gated the same way on all four card kinds.

#154's 5b.7 set the rule: render it only when **missing > 0 and chunks > 1** —
the one condition under which the pane knows something the card face does not,
namely *where in the read* the gaps fall. With a single chunk it restates a
percentage the header already carries, four times over.

The rule could only land for numeric and datetime, because `render/html.py`
finalized categorical and boolean without chunk metadata and neither summary
had a field to put it in (#193). A single-chunk report was consistent by
accident — the numeric and datetime panes dropped, and the other two had
nothing to drop against. A multi-chunk report was not: `Age` got a strip
showing where its gaps fell and `Embarked` got a Present/Missing pair
restating its header.

**Why this file asserts both directions.** `getattr(stats, "chunk_metadata",
None)` returns `None` rather than raising, so applying the gate to a kind that
carries no such field *looks* like it works — it does not tighten the rule, it
hides the pane permanently. A test that only checks "absent when single-chunk"
passes just as happily against a pane that can never appear at all. So every
kind is checked open **and** closed.

Two fixture traps, both hit while writing this, and both of the kind that
reports "absent" where the truth is "your fixture missed the branch":

* `np.where(mask, pd.NaT, dates)` yields an **object** column, which does not
  infer as datetime — the datetime card simply never rendered, and the pane
  looked broken when the fixture was.
* a bool column with `None` punched into it is object too, and infers as
  categorical. A nullable `"boolean"` dtype with `pd.NA` is what produces a
  boolean card.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

KINDS = ("numeric", "categorical", "boolean", "datetime")

#: Small enough to force many chunks over the 900-row frame below.
MANY_CHUNKS = 100

#: Larger than the frame, so the whole read is one chunk.
ONE_CHUNK = 10_000


def _frame(rows: int = 900) -> pd.DataFrame:
    """One column of each kind, each with real missing values.

    Every column is built in its own dtype rather than through `np.where`,
    which silently produces object columns -- see the module docstring.
    """
    rng = np.random.default_rng(0)

    number = pd.Series(rng.normal(0, 1, rows))
    number[rng.random(rows) < 0.2] = np.nan

    label = pd.Series(rng.choice(list("abcde"), rows))
    label[rng.random(rows) < 0.2] = None

    flag = pd.Series(rng.integers(0, 2, rows).astype(bool)).astype("boolean")
    flag[rng.random(rows) < 0.2] = pd.NA

    moment = pd.Series(pd.date_range("2024-01-01", periods=rows, freq="h"))
    moment[rng.random(rows) < 0.2] = pd.NaT

    return pd.DataFrame(
        {"number": number, "label": label, "flag": flag, "moment": moment}
    )


def _card(report: str, kind: str) -> str:
    found = re.search(
        rf'<article[^>]*var-card[^>]*data-type="{kind}".*?</article>', report, re.S
    )
    assert found, (
        f"no {kind} card in the report. The fixture missed that branch, which "
        f"reads as a passing gate and is not one"
    )
    return found.group(0)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


@pytest.fixture(scope="module")
def chunked(frame) -> str:
    return profile(frame, seed=0, chunk_size=MANY_CHUNKS).html


@pytest.fixture(scope="module")
def unchunked(frame) -> str:
    return profile(frame, seed=0, chunk_size=ONE_CHUNK).html


class TestTheFixtureIsHonest:
    """If these fail, nothing below means anything."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_kind_renders_a_card(self, chunked, kind):
        _card(chunked, kind)

    @pytest.mark.parametrize("column", ["number", "label", "flag", "moment"])
    def test_every_column_actually_has_gaps(self, frame, column):
        assert frame[column].isna().sum() > 0, (
            f"{column} has no missing values, so the gate's open case is "
            f"unreachable through it"
        )

    def test_the_two_reads_really_differ_in_chunk_count(self, frame):
        """A gate on chunk count is untestable if both runs are one chunk."""
        assert len(frame) > MANY_CHUNKS
        assert len(frame) <= ONE_CHUNK


class TestTheGateOpens:
    """Many chunks and real gaps: the pane knows where they fall."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_pane_is_there(self, chunked, kind):
        assert "Missing Values" in _card(chunked, kind), (
            f"the {kind} card hides its Missing Values pane across "
            f"{MANY_CHUNKS}-row chunks, where the pane is the only thing that "
            f"can show where the gaps fall (#193)"
        )


class TestTheGateCloses:
    """One chunk: the pane would only restate the header."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_pane_is_absent(self, unchunked, kind):
        assert "Missing Values" not in _card(unchunked, kind), (
            f"the {kind} card renders a Missing Values pane on a single-chunk "
            f"read, where it restates the percentage already on the card face"
        )


class TestTheAccumulatorsCarryIt:
    """The summary field the gate reads, asserted at the source.

    A render-level check alone cannot tell "the pane is correctly hidden" from
    "the field does not exist", which is the whole trap.
    """

    @pytest.mark.parametrize("column", ["number", "label", "flag", "moment"])
    def test_each_column_alone_still_opens_the_gate(self, frame, column):
        """One column at a time, so a pane cannot be credited to a neighbour."""
        report = profile(frame[[column]], seed=0, chunk_size=MANY_CHUNKS).html

        assert "Missing Values" in report

    @pytest.mark.parametrize(
        "kind,values",
        [
            ("categorical", ["a", None, "b"]),
            ("boolean", [True, None, False]),
        ],
    )
    def test_the_summary_field_exists_and_fills(self, kind, values):
        """Driven with a plain list, which is what the accumulator is given.

        Accumulators never see the frame, only arrays: `_to_bool_array_pandas`
        hands over `[bool | None]`, having already turned `pd.NA` into `None`.
        Passing a nullable `"boolean"` Series here instead raises *"boolean
        value of NA is ambiguous"* from inside pandas -- a shape the pipeline
        never produces, so the crash would have been a fixture inventing an
        input rather than a bug in the accumulator.
        """
        if kind == "categorical":
            from pysuricata.accumulators.categorical import (
                CategoricalAccumulator as Acc,
            )
        else:
            from pysuricata.accumulators.boolean import BooleanAccumulator as Acc

        acc = Acc("c")
        for _ in range(3):
            acc.update(list(values))
            acc.mark_chunk_boundary()
        summary = acc.finalize()

        assert hasattr(summary, "chunk_metadata"), (
            "the summary has no chunk_metadata field, so the gate reads None "
            "and hides the pane permanently instead of tightening the rule"
        )
        assert len(summary.chunk_metadata) == 3
        assert [missing for _, _, missing in summary.chunk_metadata] == [1, 1, 1]

    def test_merging_offsets_the_second_run(self):
        """Accumulators must be mergeable, and merged chunks must stay
        contiguous rather than restarting at zero halfway through."""
        from pysuricata.accumulators.categorical import CategoricalAccumulator

        left, right = CategoricalAccumulator("c"), CategoricalAccumulator("c")
        left.update(pd.Series(["a", None]))
        left.mark_chunk_boundary()
        right.update(pd.Series(["b", "c", None]))
        right.mark_chunk_boundary()
        left.merge(right)

        assert left.finalize().chunk_metadata == [(0, 1, 1), (2, 4, 1)]
