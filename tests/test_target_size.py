"""Every interactive target is at least 44x44, and stays that way.

#122's item 1. The size was measured in a browser at 390px -- 80 targets in a
report with every card kind on it -- and this file is what keeps the rules that
produced that measurement from being edited away later. It cannot re-measure:
there is no browser in the test environment, so it asserts the *declarations*
whose absence would shrink a target, and names the browser result they were
derived from.

Two things learned while measuring, both of which this file encodes:

* **The painted box is not always the target.** `.icon-btn` paints 30px inside
  a 52px app bar and extends the hit area with an absolutely positioned
  `::after` at `--tap-min`. A pseudo-element is hit-tested like any other box,
  so that is a real 44x44 target -- and a check reading `getBoundingClientRect`
  alone reports it as a 30px failure and invites someone to "fix" it by
  growing the bar. The first version of the audit did exactly that.

* **`.report-meta a` is exempt, and exempt is not the same as missed.** WCAG
  2.5.8 excludes a target that sits inline in a run of text, which is what the
  two footer links are: they are the last item in a metadata sentence, not
  controls in a strip. They are named here so that "why is that one 14px" has
  an answer other than an oversight.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"
JS_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "js"
TOKENS = CSS_DIR / "_00-tokens.css"


def _code(path: Path) -> str:
    """A stylesheet with its comments removed.

    Three separate checks in this project have been tripped by prose that
    quoted the code it was explaining. Masking first is cheaper than being
    surprised again.
    """
    return re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)


def _rule(sheet: str, selector: str) -> str | None:
    match = re.search(
        rf"{re.escape(selector)}\s*\{{(.*?)\}}", _code(CSS_DIR / sheet), re.S
    )
    return match.group(1) if match else None


def test_the_token_exists_and_is_44():
    """Every rule below reads `var(--tap-min)` rather than a number, so this is
    the single place the value can be wrong."""
    match = re.search(r"--tap-min:\s*(\d+)px", TOKENS.read_text(encoding="utf-8"))
    assert match, "--tap-min is not defined"
    assert int(match.group(1)) == 44


#: (stylesheet, selector, which dimension the rule has to guarantee).
#: Every one of these measured under 44 in a browser at 390px before #122.
SIZED = [
    ("_06-cards.css", "#pysuricata-report .var-card__header .info-link", "both"),
    ("_06-cards.css", ".attention-item", "height"),
    ("_06-cards.css", ".attention-col", "both"),
    ("_13-utilities.css", "#pysuricata-report .controls-row input", "height"),
    ("_13-utilities.css", "#pysuricata-report .tab", "height"),
    ("_13-utilities.css", "#pysuricata-report .pagination button", "both"),
    ("_13-utilities.css", "#pysuricata-report .page-number", "both"),
    ("_05-sample.css", "#pysuricata-report details.card>summary", "height"),
]


@pytest.mark.parametrize("sheet,selector,axis", SIZED, ids=lambda v: v)
def test_the_target_is_sized_from_the_token(sheet: str, selector: str, axis: str):
    body = _rule(sheet, selector)
    assert body is not None, f"{selector} is gone from {sheet}"
    if axis in ("height", "both"):
        assert re.search(r"(min-)?height:\s*var\(--tap-min\)", body), (
            f"{selector} no longer guarantees a 44px height:\n{body}"
        )
    if axis in ("width", "both"):
        assert re.search(r"(min-)?width:\s*var\(--tap-min\)", body), (
            f"{selector} no longer guarantees a 44px width:\n{body}"
        )


def test_the_icon_button_keeps_its_extended_hit_area():
    """The one target whose painted box is deliberately smaller than itself.

    30px of paint in a 52px bar, with the target extended past it by a
    pseudo-element. Delete the `::after` and the button still *looks* right at
    every screen size, which is exactly why it needs a test.
    """
    body = _rule("_02-header.css", "#pysuricata-report .icon-btn::after")
    assert body is not None, "the extended hit area is gone"
    assert "position: absolute" in body
    assert body.count("var(--tap-min)") == 2, (
        f"the overlay is no longer --tap-min on both axes:\n{body}"
    )


def test_the_page_number_is_a_button():
    """It was a `<span>` with a click listener: no role, no keyboard access, and
    "2" as its whole accessible name. Sizing it would have left three of those
    four problems in place.
    """
    source = (JS_DIR / "pagination.js").read_text(encoding="utf-8")
    assert "<button" in source and 'class="page-number' in source, (
        "page numbers are not rendered as buttons"
    )
    assert "aria-label=" in source, "the page number has no accessible name"
    assert "aria-current=" in source, "the current page is not marked"
    assert '<span class="page-number' not in source


class TestReducedMotion:
    """`prefers-reduced-motion` was honoured nowhere, against 49 transitions
    and animations."""

    def test_the_stylesheets_honour_it(self):
        base = _code(CSS_DIR / "_01-base.css")
        assert "@media (prefers-reduced-motion: reduce)" in base
        block = base.split("prefers-reduced-motion: reduce)", 1)[1]
        for declaration in ("animation-duration", "transition-duration"):
            assert f"{declaration}: 0.01ms !important" in block

    @pytest.mark.parametrize("script", ["functionality.js", "pagination.js"])
    def test_the_scripts_honour_it_too(self, script: str):
        """An explicit `behavior` argument beats the CSS `scroll-behavior`
        property, so the media query in the stylesheet cannot reach a
        `scrollTo({behavior: 'smooth'})`. Both call sites check for
        themselves."""
        source = (JS_DIR / script).read_text(encoding="utf-8")
        if "behavior:" not in source:
            pytest.skip(f"{script} no longer scrolls programmatically")
        assert "prefers-reduced-motion" in source, (
            f"{script} scrolls with an explicit behavior and never asks "
            "whether the reader wants motion"
        )
