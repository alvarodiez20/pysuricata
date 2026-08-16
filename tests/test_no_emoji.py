"""No emoji in the render layer, and none in a rendered report.

They are not part of this brand and they render inconsistently across
platforms: a `✓` has a different weight and baseline on every OS, which is
visible in a report that otherwise sets every figure in one mono face.

There is a second reason, learned in #122. Generated and inline glyphs are
**announced by screen readers**, so a `✓` in front of "No missing values
detected" is read out as "check mark no missing values detected". The redesign
already has the replacement idiom: the quality flags carry *drawn* shapes —
circle, triangle, square, made with `border-radius` and `clip-path` — which
are silent and survive greyscale.

Mathematical symbols are not emoji and are not caught here. `∞` names the
thing it counts and is the correct glyph for it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pysuricata import profile

ROOT = Path(__file__).resolve().parents[1] / "pysuricata"
#: Everything that can put a glyph on the page. The first version checked only
#: `render/` and passed while `tooltips.js` printed a clipboard on every header
#: tooltip -- the markup does not all come from Python.
SOURCES = sorted(
    [
        *(ROOT / "render").rglob("*.py"),
        *(ROOT / "static" / "js").glob("*.js"),
        *(ROOT / "templates").glob("*.html"),
    ]
)


def _code(path: Path) -> str:
    """Source with its comments removed.

    The fourth check in this project to be tripped by prose quoting the code it
    explains: `_06-cards.css` carries a comment saying a `content: "\u2713"` was
    rejected *because* generated glyphs are announced, and that sentence is a
    perfectly good explanation which must not fail the rule it explains.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # css and js block
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)  # js line
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)  # python line
    return text


#: Pictographs, dingbats, variation selectors and keycap combiners. Deliberately
#: not the whole of "symbols": `∞` (U+221E), `→`, `±` and `≈` are mathematical
#: notation the report uses on purpose.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols
    "☀-➿"  # miscellaneous symbols and dingbats
    "⬀-⯿"  # arrows and stars used as emoji
    "️"  # variation selector-16, the "render as emoji" marker
    "⃣"  # combining keycap, as in 0️⃣
    "]"
)


def test_there_are_sources_to_check():
    """Guards the guard: a glob matching nothing passes every test below."""
    assert len(SOURCES) > 5


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_emoji_in_the_render_layer(path: Path):
    found = [
        (number, line.strip())
        for number, line in enumerate(_code(path).splitlines(), 1)
        if _EMOJI.search(line)
    ]
    assert not found, f"{path.name} carries emoji: {found[:3]}"


def test_a_rendered_report_carries_none_either():
    """The source check would miss one arriving through a data-driven label.

    The report inlines its own stylesheet and scripts, so those are stripped
    first -- otherwise this finds the glyphs quoted in the comments that
    explain why they were removed, which is the source check's job anyway and
    with the comments masked.
    """
    frame = pd.read_csv("docs/assets/titanic.csv")
    html = profile(frame, seed=0).html
    markup = re.sub(r"(?s)<(script|style)\b.*?</\1>", "", html)
    found = sorted(set(_EMOJI.findall(markup)))
    assert not found, f"the report renders emoji: {found}"


def test_the_pattern_would_catch_the_ones_that_were_removed():
    """A regex asserting absence proves nothing unless it can find something.

    These are the six that were in `render/` before #180.
    """
    for glyph in ("✓", "🔗", "❓", "0️⃣", "➖", "📊"):
        assert _EMOJI.search(glyph), f"{glyph!r} would not have been caught"


def test_mathematical_notation_is_not_treated_as_emoji():
    """The report uses these on purpose and they must survive."""
    for glyph in ("∞", "≈", "→", "±", "—", "×", "χ²"):
        assert not _EMOJI.search(glyph), f"{glyph!r} is not emoji"
