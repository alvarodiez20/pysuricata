"""The three header controls: which order they sit in, and that they work.

Order is pin, download, theme. The pin governs the bar it sits in, so it leads;
the theme toggle is set once and left, so it trails.

The rest of this file exists because of #142. `+ add a note` was inert for
eleven versions -- right size, right colour, right place, and dead -- and
nothing caught it because no test asserted that a control resolves. Moving a
button around in the template is exactly the edit that breaks a selector, so
each of these three is checked for a handler as well as a position.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pysuricata import profile

JS = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "js"

#: Left to right. Verified against rendered geometry in a browser, not only
#: against source order: at 390px these are 32px boxes at x = 153, 195, 237.
EXPECTED_ORDER = ["toggle-pin", "download-report", "toggle-dark-mode"]


@pytest.fixture(scope="module")
def markup() -> str:
    html = profile(pd.DataFrame({"a": [1.0, 2, 3, 4, 5]}), seed=0).html
    return re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)


class TestTheOrder:
    def test_pin_then_download_then_theme(self, markup):
        found = re.findall(
            r'data-action="(toggle-pin|download-report|toggle-dark-mode)"', markup
        )
        assert found == EXPECTED_ORDER

    def test_all_three_are_inside_the_action_group(self, markup):
        """Not merely present somewhere on the page. `ensurePinButton()` injects
        a pin when it finds none, and before #131 it injected into `.quick` --
        dropping an icon among the text section links and into the mobile rail
        that has to fit five labels at 390px."""
        group = markup[markup.index("bar-actions") :]
        group = group[: group.index("</div>", group.index("toggle-dark-mode"))]
        for action in EXPECTED_ORDER:
            assert f'data-action="{action}"' in group, action

    def test_the_pin_keeps_both_icon_states(self, markup):
        """Moving the button meant moving its SVG. Losing either group leaves
        the toggle with nothing to swap."""
        assert 'id="pinIconOn"' in markup
        assert 'id="pinIconOff"' in markup


class TestEachOneStillHasAHandler:
    """A reordered button that no longer dispatches is #142 again."""

    @pytest.fixture(scope="class")
    def dispatch(self) -> str:
        return (JS / "functionality.js").read_text(encoding="utf-8")

    @pytest.mark.parametrize("action", EXPECTED_ORDER)
    def test_the_action_is_dispatched(self, action, dispatch):
        assert f"case '{action}':" in dispatch

    def test_the_pin_handler_resolves_the_button_by_id(self, markup, dispatch):
        assert "const PIN_BTN_ID = 'pin-button'" in dispatch
        assert 'id="pin-button"' in markup

    def test_the_theme_handler_resolves_its_icon(self, markup, dispatch):
        assert "getElementById('toggle-icon')" in dispatch
        assert 'id="toggle-icon"' in markup


class TestTheAccessibleNameAndTheTooltipAgree:
    """`setPinned` updated `aria-label` and not `title`, so the tooltip said
    *Unpin header* on a header that was already unpinned. A sighted user and a
    screen-reader user were told different things about the same control."""

    @pytest.fixture(scope="class")
    def pin_block(self) -> str:
        source = (JS / "functionality.js").read_text(encoding="utf-8")
        return source[
            source.index("function setPinned") : source.index("function togglePin")
        ]

    @pytest.mark.parametrize("label", ["Unpin header", "Pin header"])
    def test_both_attributes_are_set_in_each_state(self, label, pin_block):
        assert f"'aria-label', '{label}'" in pin_block
        assert f"btn.title = '{label}'" in pin_block

    def test_neither_state_sets_only_one_of_them(self, pin_block):
        assert pin_block.count("btn.title =") == pin_block.count("'aria-label',")
