"""#150 — the example dataset can exercise what the report does.

Titanic is in the README, the docs quick start and the linked example report,
and it cannot show several of the things the library now does: no datetime
column, so the datetime card and its four temporal panels never rendered; no
numeric pair above 0.5, so the correlations section always took the
weak-result route. Three of four card kinds and one of three correlation
views, in the example that exists to demonstrate the library.

These are #150's four acceptance lines, asserted against the vendored file
rather than trusted. The dataset is a fixture for the *documentation*, and a
documentation fixture that quietly stops demonstrating the thing it was chosen
for is exactly the failure this file exists to prevent.

Titanic stays as the fixture for `tests/test_report_layout.py`: its byte and
height ratchets are pinned to it, and repinning them would throw away their
history for nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pysuricata import profile

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "docs" / "assets" / "bike_sharing.csv"

#: The hook in `.pre-commit-config.yaml` allows 1500 KB, raised from 500 for
#: this file alone. If the dataset is ever rebuilt larger than that, the commit
#: fails with a message about accidents rather than about this decision.
_MAX_KB = 1500


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(DEMO, parse_dates=["rented_at"])


@pytest.fixture(scope="module")
def report(frame) -> str:
    return profile(frame, seed=0).html


def _body(html: str) -> str:
    """The document without its inlined CSS and JS.

    A class name searched for in the whole document is found in the very source
    that references it -- the trap `CLAUDE.md` records.
    """
    return re.sub(r"<script\b.*?</script>|<style\b.*?</style>", "", html, flags=re.S)


class TestTheExampleRendersWhatTitanicCouldNot:
    def test_all_four_card_kinds_appear(self, report):
        kinds = set(
            re.findall(
                r'<article class="var-card" id="[^"]*" data-type="([a-z]+)"',
                _body(report),
            )
        )

        assert kinds >= {"numeric", "categorical", "boolean", "datetime"}, (
            f"the example renders only {sorted(kinds)}"
        )

    def test_the_correlation_view_is_populated(self, report):
        """Not the empty state. `temp` and `feels_like` correlate at 0.99 and
        `registered` with `rentals` at 0.97, which is why this dataset was
        chosen over one that merely has more rows."""
        assert "no-correlations-state" not in _body(report)

    @pytest.mark.parametrize("panel", ["Hour of day", "Day of week", "Month", "Year"])
    def test_every_temporal_panel_renders(self, report, panel):
        """All four, which is what two calendar years buy. The renderer drops
        the year panel inside a single year -- one bar reading "all of it" --
        so a dataset trimmed to one year would silently lose this."""
        assert panel in _body(report)


class TestItStaysUsableAsAnExample:
    def test_it_loads_instantly(self, frame):
        """#150's third line. A quick start that takes a visible pause is a
        worse advertisement than a smaller dataset."""
        import time

        start = time.perf_counter()
        profile(frame, seed=0)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"profiling the example took {elapsed:.1f}s"

    def test_it_is_vendored_so_no_job_fetches_it(self):
        """#150's fourth line, and the constraint the issue puts on every
        candidate: CI must never depend on network reachability."""
        assert DEMO.exists(), "the example dataset is not vendored"

    def test_it_stays_inside_the_size_the_repo_allows(self):
        kb = DEMO.stat().st_size / 1024

        assert kb < _MAX_KB, (
            f"the example dataset is {kb:.0f} KB, over the {_MAX_KB} KB the "
            f"large-file hook allows -- raise both together, or trim the file"
        )

    def test_the_licence_notice_travels_with_it(self):
        """The source terms ask for a citation. A dataset vendored without the
        line that says so is the kind of thing nobody notices until it
        matters."""
        notice = DEMO.with_suffix(".NOTICE")

        assert notice.exists()
        assert "Fanaee-T" in notice.read_text(encoding="utf-8")


class TestTheShapeTheExampleIsChosenFor:
    """If a rebuild ever changes these, the example stops demonstrating what it
    was picked to demonstrate, and the checks above would still pass."""

    def test_the_timestamp_spans_two_calendar_years(self, frame):
        years = frame["rented_at"].dt.year

        assert years.nunique() == 2, "one calendar year loses the year panel entirely"

    def test_the_strong_pair_is_still_strong(self, frame):
        assert frame["temp"].corr(frame["feels_like"]) > 0.95

    def test_every_hour_of_the_day_is_represented(self, frame):
        assert frame["rented_at"].dt.hour.nunique() == 24

    def test_both_boolean_columns_have_both_values(self, frame):
        for column in ("holiday", "working_day"):
            assert frame[column].nunique() == 2, column
