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


class TestNoEmojiReachesTheReport:
    """#119 removed three glyphs from the correlations section and guarded them
    there. Four more survived in the numeric card's outlier and context notes,
    for eleven versions, because the guard only ever looked at one section.

    This one looks at the whole document. The objection is unchanged: they are
    not part of this brand, and they render inconsistently -- `ℹ️` in particular
    takes a coloured emoji presentation on some platforms, so a note styled as
    quiet grey text acquires a blue box.
    """

    #: Everything the render layer has reached for at some point, plus the
    #: obvious neighbours, so a newly added one is caught rather than merely
    #: the ones already removed.
    _GLYPHS = ["💡", "ℹ️", "📊", "📈", "📉", "⚠️", "✅", "❌", "🔍", "🚨", "📌", "🎯"]

    @pytest.fixture(scope="class")
    def rich_report(self) -> str:
        """Wide enough to reach the panes the glyphs lived in: an outlier-heavy
        column, a dominant category, a datetime and a boolean."""
        import numpy as np

        rng = np.random.default_rng(0)
        n = 900
        gappy = rng.normal(0, 1, n)
        gappy[rng.choice(n, 300, replace=False)] = np.nan
        frame = pd.DataFrame(
            {
                "gappy": gappy,
                "skewed": rng.lognormal(0, 1.5, n),
                "cat": rng.choice(["a", "b", "c"], n, p=[0.9, 0.05, 0.05]),
                "when": pd.date_range("2026-01-01", periods=n, freq="h"),
                "flag": rng.integers(0, 2, n).astype(bool),
            }
        )
        return profile(frame, seed=0).html

    @pytest.mark.parametrize("glyph", _GLYPHS)
    def test_none_reach_the_report(self, rich_report, glyph):
        assert glyph not in rich_report

    @pytest.mark.parametrize("glyph", _GLYPHS)
    def test_none_remain_in_the_render_sources(self, glyph):
        """Belt and braces: a glyph on a branch this fixture does not take would
        pass the test above and still ship."""
        root = JS.parent.parent / "render"
        for path in sorted(root.glob("*.py")):
            assert glyph not in path.read_text(encoding="utf-8"), (
                f"{path.name}: {glyph}"
            )

    def test_no_pictograph_is_smuggled_in_as_an_html_entity(self):
        """`boolean_card.py` wrote them as `&#128680;` rather than as `🚨`.

        A literal-glyph search cannot see that, and neither could the first
        version of this guard -- so the boolean card kept four of them through
        a sweep that believed it had removed every one. Any numeric character
        reference above U+2000 is a symbol or a pictograph; the report's real
        typography is `·`, `—`, `→`, `≈` and `±`, all of which are written as
        the character itself.
        """
        root = JS.parent.parent
        pattern = re.compile(r"&#(\d+);")
        offenders: list[str] = []
        for path in (
            sorted(root.rglob("*.py"))
            + sorted(root.rglob("*.html"))
            + sorted(root.rglob("*.css"))
            + sorted(root.rglob("*.js"))
        ):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                if int(match.group(1)) >= 0x2000:
                    offenders.append(f"{path.name}: {match.group(0)}")
        assert not offenders, offenders
