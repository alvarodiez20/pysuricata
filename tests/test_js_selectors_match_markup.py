"""Every selector the JS binds on must match markup the renderer emits (#233).

The report's interactivity is a pairing between two files that never import
each other: `static/js/functionality.js` binds handlers with `closest()`, and
the renderers emit the classes those selectors look for. Rename a class on one
side and the feature goes quiet — no error, no console warning, nothing in the
page that looks wrong. Just a control that stops responding.

That is the fragile part, and it is checkable without a browser: extract the
selectors from the JS, render a report that exercises every card kind, and
assert each selector has something to match.

## What #233 turned out to be

Filed as "the temporal bars carry tooltip data that never reaches a tooltip",
with the probe measuring `display: none` and empty text on two separate builds.
The issue asked first whether the tooltip was broken *or the probe was*, and
named two ways it could be the probe. It was both of them:

* it queried `.tooltip, .chart-tooltip, [role=tooltip], #tooltip`. The element
  is **`.hist-tooltip`**, and `ensureTip()` creates it lazily on first show — so
  before any successful hover it does not exist under *any* selector;
* the bar sits ~1,900px down a 900px viewport. Playwright's mouse coordinates
  are viewport-relative, so moving to it without scrolling delivers the event
  to empty space. My own first re-probe repeated this exact mistake.

With the right selector and the bar scrolled into view, the tooltip reads
`00:00 · 209 records (4.2%)` — the label, count and percentage the handler
builds. **The feature works.** This file is the third acceptance box: the guard
that would have caught a real drift, which is what the issue was worth.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile

REPO = Path(__file__).resolve().parents[1]
JS = REPO / "pysuricata" / "static" / "js" / "functionality.js"

#: Selectors that are matched against elements the *scripts* create, or against
#: state a user has to reach first, so a static report legitimately has none.
#: Named individually rather than skipped by pattern, so adding one is a
#: decision someone makes on purpose.
NOT_IN_A_STATIC_REPORT = {
    # Created by ensureTip() on the first hover, never rendered server-side.
    ".hist-tooltip",
    # Emitted only for columns with more than one chunk (#193), and this
    # fixture is deliberately a single chunk elsewhere.
    ".chunk-segment",
    ".spectrum-segment",
    ".chunk-spectrum",
    ".missing-spectrum-bar",
    ".dataprep-spectrum",
    ".chunk-distribution",
}


@pytest.fixture(scope="module")
def report() -> str:
    """All four card kinds, with correlations and missing values.

    A selector check is only as good as the markup the fixture reaches: a card
    kind that never renders makes its selectors look dead, and absent reads as
    broken.
    """
    rng = np.random.default_rng(0)
    rows = 900
    number = pd.Series(rng.normal(0, 1, rows))
    number[rng.random(rows) < 0.15] = np.nan
    return profile(
        pd.DataFrame(
            {
                "amount": number,
                "paired": rng.normal(0, 1, rows) + number.fillna(0) * 0.8,
                "region": rng.choice(list("abcde"), rows),
                "active": pd.Series(rng.integers(0, 2, rows).astype(bool)).astype(
                    "boolean"
                ),
                "seen_at": pd.date_range("2024-01-01", periods=rows, freq="h"),
            }
        ),
        seed=0,
    ).html


def _bound_selectors() -> set[str]:
    """Every simple class selector `functionality.js` binds a handler to."""
    source = JS.read_text(encoding="utf-8")
    found: set[str] = set()
    for group in re.findall(r"closest\(\s*['\"]([^'\"]+)['\"]\s*\)", source):
        for alternative in group.split(","):
            # `.temporal-chart .temporal-bar` is a descendant pair; both halves
            # have to exist, and checking each independently localises a break.
            for token in alternative.split():
                if token.startswith(".") and re.fullmatch(r"\.[\w-]+", token):
                    found.add(token)
    return found


def test_the_extraction_found_the_bindings():
    """If the regex stops matching, every assertion below passes vacuously."""
    selectors = _bound_selectors()

    assert len(selectors) >= 10, f"only found {selectors}; the binding style moved"
    assert ".temporal-bar" in selectors, "the selector #233 was filed about is missing"


@pytest.mark.parametrize(
    "selector", sorted(_bound_selectors() - NOT_IN_A_STATIC_REPORT)
)
def test_every_bound_selector_matches_rendered_markup(report, selector):
    """A handler bound to a class nothing emits is a dead control."""
    css_class = selector.lstrip(".")

    assert re.search(rf'class="[^"]*\b{re.escape(css_class)}\b', report), (
        f"{selector} is bound in functionality.js but no element in the report "
        f"carries it. Either the renderer's class was renamed and the handler "
        f"was not, or the reverse -- and the symptom is a control that silently "
        f"stops responding"
    )


class TestTheTemporalTooltipChain:
    """#233, end to end through the markup: the data, the hook, the handler."""

    def test_the_bars_carry_what_the_tooltip_prints(self, report):
        bars = re.findall(r"<rect[^>]*\btemporal-bar\b[^>]*>", report)
        assert bars, "no temporal bars in the report"

        for attribute in ("data-count", "data-pct", "data-label"):
            assert attribute in bars[0], f"a temporal bar has no {attribute}"

    def test_the_bars_sit_inside_the_container_the_handler_requires(self, report):
        """The binding is `.temporal-chart .temporal-bar`. A bar outside a
        `.temporal-chart` matches nothing, and looks identical in the page."""
        charts = re.findall(r"<svg[^>]*\btemporal-chart\b.*?</svg>", report, re.S)
        assert charts, "no .temporal-chart container"
        assert any("temporal-bar" in chart for chart in charts), (
            "temporal bars exist but none is inside a .temporal-chart, so the "
            "handler's descendant selector cannot reach them"
        )

    def test_the_handler_prints_all_three_values(self):
        """Reading the attributes and then not using one is the other way this
        goes quiet."""
        source = JS.read_text(encoding="utf-8")
        start = source.index(".temporal-chart .temporal-bar")
        handler = source[start : start + 900]

        for attribute in ("data-count", "data-pct", "data-label"):
            assert attribute in handler, f"the handler ignores {attribute}"
        assert "showTip(" in handler, "the handler never shows the tooltip"
