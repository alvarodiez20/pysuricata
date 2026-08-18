"""Contrast guard for the report palette.

Drop into ``tests/``. Parses the CSS custom properties out of
``pysuricata/static/css/_00-tokens.css`` and asserts that every pair the
design actually uses clears its WCAG minimum, in both themes.

This exists because four real failures got through design review by being
checked against the wrong background: a token that passes on the paper can
fail on a tinted surface, and a warning colour that works as a bar fill can
fail as text at 11px. Both happened.

Rules enforced:
  * text on its background            >= 4.5:1   (WCAG AA, normal text)
  * bar fill vs its track             >= 3.0:1   (AA non-text contrast)
  * bar fill vs the page background   >= 3.0:1
  * border vs its background          >= 3.0:1

Run: pytest tests/test_contrast.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TOKENS_CSS = (
    Path(__file__).resolve().parents[1]
    / "pysuricata"
    / "static"
    / "css"
    / "_00-tokens.css"
)

AA_TEXT = 4.5
AA_NON_TEXT = 3.0


# --------------------------------------------------------------------------- #
# colour maths
# --------------------------------------------------------------------------- #
def _channel(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------- #
# token extraction
# --------------------------------------------------------------------------- #
_DECL = re.compile(r"--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{3,6})\s*;")


def _parse_blocks(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (light, dark) token maps.

    Everything before the ``:not(.light)`` selector is light; the tokens
    redeclared inside it override for dark.
    """
    marker = css.find(":not(.light)")
    head = css if marker == -1 else css[:marker]
    tail = "" if marker == -1 else css[marker:]

    light = {name: hexval for name, hexval in _DECL.findall(head)}
    dark = dict(light)
    dark.update({name: hexval for name, hexval in _DECL.findall(tail)})
    return light, dark


@pytest.fixture(scope="module")
def themes() -> dict[str, dict[str, str]]:
    assert TOKENS_CSS.exists(), f"token file not found: {TOKENS_CSS}"
    light, dark = _parse_blocks(TOKENS_CSS.read_text(encoding="utf-8"))
    missing = [t for t in ("paper", "ink", "muted", "data-2") if t not in light]
    assert not missing, f"tokens missing from {TOKENS_CSS.name}: {missing}"
    return {"light": light, "dark": dark}


# --------------------------------------------------------------------------- #
# the pairs the design actually uses
# --------------------------------------------------------------------------- #
TEXT_PAIRS: list[tuple[str, str, str]] = [
    ("ink", "paper", "primary text"),
    ("ink-2", "paper", "body prose"),
    ("muted", "paper", "captions and secondary figures"),
    ("q-good", "paper", "pass flag text"),
    ("q-warn-text", "paper", "warning figures — e.g. '19.9% missing'"),
    ("q-bad", "paper", "fail flag text"),
    ("paper", "data-1", "count printed inside the darkest segment"),
    ("paper", "data-2", "count printed inside a default bar"),
    ("ink", "data-4", "count printed inside the palest segment"),
    # --data-3 deliberately absent: it is a fill that carries no text.
    # See test_data_3_carries_no_text.
]

# A mark on the paper or on an empty track needs 3:1 (WCAG 1.4.11).
# --data-4 is absent by design — it is stack-internal only, bounded by
# its neighbours, and never sits alone on a background.
NON_TEXT_PAIRS: list[tuple[str, str, str]] = [
    ("data-1", "paper", "darkest step as a standalone mark"),
    ("data-2", "paper", "histogram bar against the page"),
    ("data-3", "paper", "third step as a standalone mark"),
    ("data-1", "track", "darkest step against the track"),
    ("data-2", "track", "bar against its empty track"),
    ("data-3", "track", "third step against the track"),
    ("q-good", "track", "low-severity missing bar"),
    ("q-warn-fill", "track", "medium-severity missing bar"),
    ("q-bad", "track", "high-severity missing bar"),
    ("q-warn-fill", "paper", "warning chip border"),
    ("axis", "paper", "chart axis and ticks"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(
    "fg,bg,label", TEXT_PAIRS, ids=lambda v: v if isinstance(v, str) else ""
)
def test_text_contrast(themes, theme: str, fg: str, bg: str, label: str) -> None:
    tokens = themes[theme]
    if fg not in tokens or bg not in tokens:
        pytest.skip(f"{fg}/{bg} not declared in {theme}")
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= AA_TEXT, (
        f"[{theme}] {label}: --{fg} ({tokens[fg]}) on --{bg} ({tokens[bg]}) "
        f"is {ratio:.2f}:1, needs {AA_TEXT}:1"
    )


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize(
    "fg,bg,label", NON_TEXT_PAIRS, ids=lambda v: v if isinstance(v, str) else ""
)
def test_non_text_contrast(themes, theme: str, fg: str, bg: str, label: str) -> None:
    tokens = themes[theme]
    if fg not in tokens or bg not in tokens:
        pytest.skip(f"{fg}/{bg} not declared in {theme}")
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= AA_NON_TEXT, (
        f"[{theme}] {label}: --{fg} ({tokens[fg]}) on --{bg} ({tokens[bg]}) "
        f"is {ratio:.2f}:1, needs {AA_NON_TEXT}:1"
    )


def test_warn_fill_is_not_used_as_text(themes) -> None:
    """--q-warn-fill is deliberately below the text minimum.

    It exists so warning bars and borders can be lighter than warning text.
    If it ever clears 4.5:1 the two tokens have converged and one should be
    deleted; if it is used as a text colour somewhere, that is the bug this
    test is here to remember.
    """
    tokens = themes["light"]
    ratio = contrast(tokens["q-warn-fill"], tokens["paper"])
    assert AA_NON_TEXT <= ratio < AA_TEXT, (
        f"--q-warn-fill is {ratio:.2f}:1 on the paper. It must stay between "
        f"{AA_NON_TEXT} (usable as a fill) and {AA_TEXT} (not usable as text). "
        f"Use --q-warn-text for any warning figure."
    )


def test_data_scale_steps_are_distinguishable(themes) -> None:
    """Adjacent steps of the data scale must be tellable apart.

    Two datasets are drawn as --data-2 against --data-4, so those two in
    particular have to separate — including in greyscale, which is what the
    luminance ratio approximates.
    """
    for theme in ("light", "dark"):
        tokens = themes[theme]
        ratio = contrast(tokens["data-2"], tokens["data-4"])
        assert ratio >= 2.0, (
            f"[{theme}] --data-2 and --data-4 are only {ratio:.2f}:1 apart. "
            f"They carry the two-dataset comparison and must separate."
        )


def test_data_3_carries_no_text(themes) -> None:
    """--data-3 is a fill, never a label background.

    It sits mid-tone on purpose so it can clear 3:1 as a standalone mark
    on both the paper and the track. That same mid-tone means neither the
    ink nor the paper reaches 4.5:1 on it. If a caller needs to print a
    count inside a segment, that segment must be --data-1, --data-2 or
    --data-4.
    """
    for theme in ("light", "dark"):
        tokens = themes[theme]
        on_ink = contrast(tokens["ink"], tokens["data-3"])
        on_paper = contrast(tokens["paper"], tokens["data-3"])
        assert max(on_ink, on_paper) < AA_TEXT, (
            f"[{theme}] --data-3 now clears {AA_TEXT}:1 for text "
            f"(ink {on_ink:.2f}, paper {on_paper:.2f}). If that is deliberate, "
            f"add an --on-data-3 token and put it back in TEXT_PAIRS."
        )


def test_data_4_is_stack_internal_only(themes) -> None:
    """--data-4 is expected to FAIL against the paper.

    That is what makes it stack-internal: it is legal beside another
    segment, never alone on a background. The test exists so that if
    someone lightens the paper or darkens --data-4 into legality, the
    role comment in tokens.css gets revisited rather than silently
    becoming wrong. Rule 1 in tokens.css is the prose version.
    """
    tokens = themes["light"]
    ratio = contrast(tokens["data-4"], tokens["paper"])
    assert ratio < AA_NON_TEXT, (
        f"--data-4 is {ratio:.2f}:1 on the paper, which clears the "
        f"{AA_NON_TEXT}:1 floor. It is documented as stack-internal only; "
        f"either widen its documented role or keep it pale."
    )
