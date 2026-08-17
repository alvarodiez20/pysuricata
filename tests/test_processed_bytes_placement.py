"""`Processed bytes (≈)` is bookkeeping, and does not belong in the stat row.

UX-21 asked for it to move out of the primary statistics and into the details
pane. #104 dropped the donut; the stat-row half did not land, so it sat in the
right-hand table on all four card kinds (#209).

The numeric card's right table is Min, Q1, Median, Mean, Q3, Max — six facts
about the distribution — and then a figure about the profiler's own
bookkeeping, in the position of highest attention on the card. It answers a
question about PySuricata, not about the data. It is not useless; it is
misplaced.

**Two of the four kinds have moved.** Numeric and datetime each have a
Statistics pane, which is the right home and where they now are. The other two
have nowhere to put it yet:

* **categorical** has details panes — Common values, Normalization, Label
  length, Missing Values — but no Statistics pane, and every one of those is
  conditional. Filing a fact that must always be in the document inside a pane
  that renders only sometimes would move it out of the stat row by making it
  disappear, which `test_report_data_invariance.py` would catch and should.
* **boolean** has no details section at all, and that is a documented decision
  rather than an omission (#155, 5c.6): a boolean column has two values and two
  counts, the card face shows both, and there is no second level of disclosure
  to offer. Giving it one to house a byte count would be the tail wagging the
  dog.

So the remaining two are recorded here rather than waived. Move one and this
file fails, telling you to shrink the set — the same ratchet idiom
`test_colour_tokens.py` uses, for the same reason: the number only goes down.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

LABEL = "Processed bytes"

#: Kinds whose `Processed bytes` has moved into the details pane.
MOVED = {"numeric", "datetime"}

#: Kinds where it is still in the primary stat row, and why they are not simply
#: oversights. See the module docstring.
NOT_YET = {"categorical", "boolean"}


@pytest.fixture(scope="module")
def report() -> str:
    """One frame with all four card kinds.

    Titanic has no datetime column, so the example report cannot exercise the
    datetime branch at all (#150). A fixture that misses a branch reports
    "absent", and absent reads as passing.
    """
    rng = np.random.default_rng(0)
    return profile(
        pd.DataFrame(
            {
                "amount": rng.normal(50, 12, 600),
                "seen_at": pd.date_range("2024-01-01", periods=600, freq="h"),
                "region": rng.choice(list("abcde"), 600),
                "active": rng.integers(0, 2, 600).astype(bool),
            }
        ),
        seed=0,
    ).html


def _card(report: str, kind: str) -> str:
    found = re.search(
        rf'<article[^>]*var-card[^>]*data-type="{kind}".*?</article>', report, re.S
    )
    assert found, f"no {kind} card in the report -- the fixture missed a branch"
    return found.group(0)


def _is_in_details(card: str) -> bool:
    """Whether the label sits inside the details section rather than the stat row.

    Decided by which container opened most recently before it. The stat row is
    a run of `.vstat` divs; the details panes are tables. A cruder split -- text
    before and after the `details-toggle` button -- gets categorical backwards,
    because the toggle is emitted ahead of the stat row in that card's markup.
    """
    index = card.find(LABEL)
    assert index != -1, "the label is not in this card at all"
    before = card[:index]
    return before.rfind("details") > before.rfind('class="vstat')


@pytest.mark.parametrize("kind", sorted(MOVED))
def test_it_has_left_the_stat_row(report, kind):
    card = _card(report, kind)

    assert _is_in_details(card), (
        f"the {kind} card shows `{LABEL}` in its primary stat row again. It is "
        f"a fact about the profiler, not about the column, and the Statistics "
        f"pane is where it belongs (#209)"
    )


@pytest.mark.parametrize("kind", sorted(NOT_YET))
def test_the_kinds_that_cannot_move_it_yet_are_recorded(report, kind):
    card = _card(report, kind)

    assert not _is_in_details(card), (
        f"the {kind} card has moved `{LABEL}` into a details pane. Good -- add "
        f"it to MOVED and remove it from NOT_YET in this file, so the win is "
        f"locked in and cannot quietly come back"
    )


@pytest.mark.parametrize("kind", sorted(MOVED | NOT_YET))
def test_the_fact_is_still_in_the_document(report, kind):
    """Moving it must not be a way of losing it.

    `test_report_data_invariance.py` guards this across the whole report; this
    says it per card kind, so a failure names the card."""
    card = _card(report, kind)

    assert card.count(LABEL) == 1, (
        f"the {kind} card carries `{LABEL}` {card.count(LABEL)} times; it "
        f"should appear exactly once, somewhere"
    )
