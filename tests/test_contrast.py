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
    # `--data-3` is deliberately absent. Nothing is printed on it: at 4.03:1
    # against the paper and 3.83:1 against the ink it reaches the 3:1 non-text
    # minimum and neither text minimum, so it is a fill and never a label
    # background. A segment on it sends its count to the legend, the way a
    # too-narrow segment already does. See `test_data_3_carries_no_text`.
    ("ink", "data-4", "count printed inside the palest segment"),
]

NON_TEXT_PAIRS: list[tuple[str, str, str]] = [
    ("data-2", "paper", "histogram bar against the page"),
    ("data-2", "track", "bar against its empty track"),
    ("data-1", "track", "darkest segment against the track"),
    ("q-good", "track", "low-severity missing bar"),
    ("q-warn-fill", "track", "medium-severity missing bar"),
    ("q-bad", "track", "high-severity missing bar"),
    ("q-warn-fill", "paper", "warning chip border"),
    # The axis, not the hairline. `--rule-strong` is a container edge: it is
    # decorative structure with no contrast minimum, and forcing it to 3:1
    # would make it the heavy black box this palette exists to remove. The
    # chart axis is part of a graphic required to understand the content, so
    # 1.4.11 does apply to it -- which is why the two are separate tokens.
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


# --------------------------------------------------------------------------- #
# roles a contrast pair cannot express
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_data_3_carries_no_text(themes, theme):
    """The third step exists to carry a *standalone mark*, not a label.

    It was `#7FA0B5`, which is **2.63:1** on the paper -- below the 3:1 non-text
    minimum, so it could not legally carry the one thing it exists for. `#5C7F99`
    clears 3:1 on both surfaces in both themes and reaches neither text minimum
    (paper 4.03, ink 3.83), which is the trade: a fill, never a label
    background.

    So this asserts a *failure* on purpose. Raising `--data-3` until text passes
    on it would break this test and force the conversation, rather than quietly
    reintroducing a label nobody measured.
    """
    tokens = themes[theme]
    for other in ("ink", "paper"):
        ratio = contrast(tokens["data-3"], tokens[other])
        assert ratio < AA_TEXT, (
            f"[{theme}] --{other} on --data-3 is now {ratio:.2f}:1. "
            "If --data-3 has been raised far enough to carry text, decide that "
            "deliberately: update this test and the composition bar, which "
            "currently sends the count to the legend instead."
        )


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_data_3_can_stand_alone(themes, theme):
    """The property the old value failed, and the reason for the change."""
    tokens = themes[theme]
    for surface in ("paper", "track"):
        ratio = contrast(tokens["data-3"], tokens[surface])
        assert ratio >= AA_NON_TEXT, (
            f"[{theme}] --data-3 on --{surface} is {ratio:.2f}:1, needs "
            f"{AA_NON_TEXT}:1 to carry a standalone mark"
        )


# --------------------------------------------------------------------------- #
# greyscale: the check a contrast ratio cannot make
# --------------------------------------------------------------------------- #
CSS_DIR = TOKENS_CSS.parent


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_data_scale_survives_greyscale(themes, theme):
    """Adjacent steps stay apart when the hue is removed.

    This is nearly free for the data scale, and that is the point of it being
    one hue in luminance steps rather than four colours: contrast *is* a
    luminance ratio, so a scale that passes on the page also passes in grey.
    """
    tokens = themes[theme]
    steps = ["data-1", "data-2", "data-3", "data-4"]
    for first, second in zip(steps, steps[1:], strict=False):
        ratio = contrast(tokens[first], tokens[second])
        assert ratio >= 1.3, (
            f"[{theme}] --{first} and --{second} are {ratio:.2f}:1 apart. "
            "Adjacent steps of the data scale must stay separable without hue."
        )


def test_the_quality_scale_does_not_survive_greyscale(themes):
    """And is therefore never allowed to be the only carrier.

    `--q-good` (#4A5D1E) and `--q-bad` (#963F1C) are 1.05:1 apart in
    luminance: olive and rust are, in grey, the same grey. That is not a bug
    in the palette -- the warm scale is chosen for *hue* distinctness, and
    forcing three bands onto separate luminances would push one of them out of
    its contrast budget against the paper.

    It is a bug in any component that leans on it alone, which is why the
    quality flags carry a shape as well. This test asserts the collapse so the
    reason for those shapes cannot be optimised away by someone who measures
    the tokens, finds them distinct in hue, and removes the marks.
    """
    tokens = themes["light"]
    ratio = contrast(tokens["q-good"], tokens["q-bad"])
    assert ratio < 1.3, (
        f"--q-good and --q-bad are now {ratio:.2f}:1 apart in luminance. If "
        "the scale has been re-chosen to separate in greyscale, that is good "
        "news -- update this test, and the shape marks on .flag become "
        "belt-and-braces rather than load-bearing."
    )


def test_each_quality_band_carries_a_shape_not_only_a_colour():
    """WCAG 1.4.1. The flags say "Positive-only" and "Missing" -- no number, no
    band name -- so with the hue gone they are the same chip.

    Checked as three distinct geometries rather than as three rules existing,
    because two bands given the same shape would satisfy 'a rule per band' and
    fail the reader.
    """
    css = (CSS_DIR / "_06-cards.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    signatures = {}
    for band in ("good", "warn", "bad"):
        match = re.search(
            rf"\.quality-flags \.flag\.{band}::before \{{(.*?)\}}", css, re.S
        )
        assert match, f"the {band} flag has no ::before mark"
        body = match.group(1)
        shape = re.findall(r"(border-radius|clip-path|width)\s*:\s*([^;]+);", body)
        signatures[band] = tuple(sorted(shape))

    assert len(set(signatures.values())) == 3, (
        f"two bands share a shape, so they are still telling apart by colour "
        f"alone: {signatures}"
    )


def test_the_mark_is_drawn_rather_than_spoken():
    """`content: "✓"` was the first version. Generated content is announced by
    screen readers, so every flag would have gained a spoken "check mark"
    before a label that already says what it is. An empty content with a drawn
    shape is silent.
    """
    css = (CSS_DIR / "_06-cards.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    block = re.search(r"\.quality-flags \.flag::before \{(.*?)\}", css, re.S)
    assert block, "the shared ::before rule is gone"
    content = re.search(r"content\s*:\s*([^;]+);", block.group(1))
    assert content and content.group(1).strip() in ('""', "''"), (
        f"the mark is {content and content.group(1)!r}, which a screen reader "
        "will read out. Draw the shape instead."
    )


# --------------------------------------------------------------------------- #
# Rule 1: the stack-only step stays in the stack
# --------------------------------------------------------------------------- #
#: The sites where `--data-4` is legal. Each is a segment of a stacked bar,
#: which is the one surface the token file permits: `#A8BECD` is **1.83:1 on
#: the paper**, so a mark drawn with it alone is a ghost on screen and gone in
#: print. `_00-tokens.css` states the rule; nothing enforced it, and it was
#: being broken in the correlations list at the time this was written.
_STACK_INTERNAL = {
    # The composition bar -- the stack the token exists for.
    "pysuricata/render/composition_bar.py",
    # The boolean card's true/false/missing bar, also a stack. Its labels
    # already take `--on-data-4`, so the pairing was understood here.
    "pysuricata/render/boolean_card.py",
    # That bar's geometry, and its legend swatch -- which is *not* a stack
    # segment and earns its legibility from a border instead. See below.
    "pysuricata/static/css/_10-boolean.css",
}

_REPO_ROOT = CSS_DIR.parent.parent.parent

_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _code_only(path: Path) -> str:
    """The file with its prose removed.

    Written the naive way first, and it failed on its own subject: the fix that
    moved the correlations bar off `--data-4` explains itself in a docstring
    that names the token, so a plain substring scan reported the file it had
    just cleaned. This repository has a standing note about that exact trap and
    this is one more instance of it -- a check over source is only as good as
    its ability to tell code from a sentence about code.

    Strings are *not* stripped: `--data-4` inside an f-string is a real use,
    which is precisely how the boolean card draws its stack. Only comments go,
    plus docstrings, which `ast` can identify exactly.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".css", ".js"}:
        return _CSS_COMMENT.sub("", text)

    import ast
    import io
    import tokenize

    stripped = "".join(
        token.string if token.type != tokenize.COMMENT else ""
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
    )
    docstrings = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.append(doc)
    for doc in docstrings:
        stripped = stripped.replace(doc, "")
    return stripped


def _files_spending_data_4() -> set[str]:
    """Every tracked source file naming `--data-4` in code, bar the tokens."""
    roots = (_REPO_ROOT / "pysuricata" / "render", _REPO_ROOT / "pysuricata" / "static")
    found = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".css", ".js"} or path == TOKENS_CSS:
                continue
            if "--data-4" in _code_only(path):
                found.add(str(path.relative_to(_REPO_ROOT)))
    return found


def test_data_4_is_only_spent_inside_a_stack():
    """A ratchet on rule 1, which no token pair can express.

    The rule is about which *surface* a mark lands on, not which two colours
    meet, so `test_non_text_contrast` cannot see a violation: `--data-4` on
    `--paper` is a pair nobody declares, it is a pair that happens because a
    bar was drawn on the page.
    """
    spent = _files_spending_data_4()
    assert spent <= _STACK_INTERNAL, (
        "--data-4 is 1.83:1 on the paper and stack-internal only. New site(s): "
        f"{sorted(spent - _STACK_INTERNAL)}. Use --data-3, which clears 3:1 on "
        "both surfaces in both themes."
    )


def test_the_below_threshold_correlation_bar_can_be_seen():
    """It was filled with `--data-4` -- a standalone bar on the paper, so the
    row rendered as a pair, a gap and a number."""
    source = (_REPO_ROOT / "pysuricata/render/correlations_section.py").read_text(
        encoding="utf-8"
    )
    bars = re.findall(r"_diverging_bar\([^)]*'var\((--data-\d)\)'\)", source)
    assert bars, "the weak row's bar no longer names its fill inline"
    assert "--data-4" not in bars, bars


def test_the_boolean_legend_swatch_is_discernible():
    """The swatch must keep the fill of the segment it labels -- that is what a
    legend is -- so its legibility has to come from a border, exactly as the
    `--track` swatch beside it already does."""
    css = (_REPO_ROOT / "pysuricata/static/css/_10-boolean.css").read_text(
        encoding="utf-8"
    )
    block = re.search(r"\.legend-color\.false\s*\{([^}]*)\}", css)
    assert block, "the false swatch rule is gone"
    body = block.group(1)
    assert "--data-4" in body, "the swatch must match the segment it labels"
    assert "border" in body, (
        "--data-4 is 1.83:1 on the paper; a swatch drawn with it and no border "
        "is invisible"
    )
