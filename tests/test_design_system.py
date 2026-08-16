"""The design decisions in phase 1 that a later edit would quietly undo.

`test_contrast.py` guards the palette's *values*. This guards its *reach*: that
the tokens are actually used, that the retired ones stay retired, and that the
two structural rules of the redesign hold across every stylesheet rather than
only in the file where they were introduced.

Every one of these is a rule the design states and nothing else enforces. CSS
has no type system: a stray `#4ea3f1` or a re-added `border-radius: 12px`
produces a valid stylesheet and a report that looks subtly off-system, and it
would only be caught by someone comparing against the palette by eye.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"
RENDER_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "render"

STYLESHEETS = sorted(CSS_DIR.glob("*.css"))
TOKENS = CSS_DIR / "_00-tokens.css"


def _declarations(pattern: str) -> list[tuple[str, int, str]]:
    """Every matching declaration across the stylesheets, with its location."""
    found = []
    for sheet in STYLESHEETS:
        for number, line in enumerate(sheet.read_text().splitlines(), 1):
            match = re.search(pattern, line)
            if match:
                found.append((sheet.name, number, line.strip()))
    return found


def test_there_are_stylesheets_to_check():
    """Guards the guard: a glob that silently matches nothing passes every
    test below and proves nothing."""
    assert len(STYLESHEETS) > 5


# --------------------------------------------------------------------------- #
# type is not a colour
# --------------------------------------------------------------------------- #
class TestTypeIsNotAColour:
    """The old palette gave each column type a hue and the cards inherited it.

    Inside a card the badge already names the type, so the hue carried nothing
    -- and it collided: olive meant both *categorical* and *passes*, rust meant
    both *boolean* and *fails*. A rust bar and a rust warning chip sat in the
    same card meaning unrelated things.
    """

    #: numeric, categorical, datetime, boolean.
    LEGACY = ("4ea3f1", "8ac926", "ffca3a", "ff595e")

    @pytest.mark.parametrize("hex_value", LEGACY)
    def test_the_retired_hue_is_gone_from_the_stylesheets(self, hex_value):
        hits = _declarations(rf"(?i)#{hex_value}")
        assert not hits, f"#{hex_value} survives: {hits[:3]}"

    def test_the_column_type_key_uses_the_data_scale_not_the_quality_scale(self):
        """The failure a hex grep cannot see.

        The four swatches in the column-type key were on `--q-good`,
        `--q-warn-fill` and `--q-bad` -- the data *quality* scale. They got
        there because the palette swap replaced each legacy hue with whichever
        new token looked closest: olive became `--q-good`, gold became
        `--q-warn-fill`, rust became `--q-bad`.

        The legacy hexes were gone, so every colour-literal check passed, while
        the report still said olive for both *categorical* and *passes* and rust
        for both *boolean* and *fails*. That collision is the entire reason this
        palette exists, and it had been reintroduced by the change meant to
        remove it.
        """
        text = (CSS_DIR / "_03-summary.css").read_text()
        key = text.split(".summary-card:first-child .summary-list li:nth-child(1)", 1)
        assert len(key) == 2, "the column-type key selector moved"
        block = key[1].split("/* Boolean */", 1)[0]
        swatches = re.findall(r"background:\s*var\((--[\w-]+)\)", block)
        assert len(swatches) == 4, swatches
        assert all(s.startswith("--data-") for s in swatches), (
            f"the column-type key must use the data scale, got {swatches}"
        )
        assert len(set(swatches)) == 4, f"each type needs its own step: {swatches}"

    def test_no_quality_token_is_used_for_a_column_type(self):
        """Stated as a rule rather than a location, so it still holds if the
        key moves to a different component."""
        quality = ("--q-good", "--q-warn-text", "--q-bad")
        offenders = []
        for sheet in STYLESHEETS:
            lines = sheet.read_text().splitlines()
            for number, line in enumerate(lines, 1):
                if not any(token in line for token in quality):
                    continue
                context = " ".join(lines[max(0, number - 12) : number]).lower()
                if re.search(r"numeric|categorical|datetime|boolean|dtype", context):
                    offenders.append(f"{sheet.name}:{number} {line.strip()}")
        assert not offenders, (
            f"a data-quality colour is labelling a column type: {offenders[:5]}"
        )

    @pytest.mark.parametrize("hex_value", LEGACY)
    def test_it_is_gone_from_the_renderers_too(self, hex_value):
        """The hues were never only in CSS.

        `temporal_charts.py`, `donut_chart.py` and `histogram_svg.py` emit
        colours directly into SVG, so a check that covered `static/css/` alone
        would pass while the charts stayed on the old palette.
        """
        hits = [
            f"{path.name}:{number}"
            for path in RENDER_DIR.rglob("*.py")
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if re.search(rf"(?i)#{hex_value}", line)
        ]
        assert not hits, f"#{hex_value} survives in render/: {hits[:5]}"


# --------------------------------------------------------------------------- #
# the structural motif
# --------------------------------------------------------------------------- #
class TestTheStructuralMotif:
    """Eight bordered boxes inside a bordered page was the strongest 'template'
    signal in the report. Hairlines and whitespace replace it."""

    def test_no_data_container_is_more_rounded_than_a_chip(self):
        """Radius survives on chips and buttons, at 6px or less, and nowhere
        else. `50%` is exempt: that is a circle, not a rounded rectangle."""
        offenders = []
        for sheet in STYLESHEETS:
            for number, line in enumerate(sheet.read_text().splitlines(), 1):
                match = re.search(r"border-radius:\s*([^;]+);", line)
                if not match:
                    continue
                value = match.group(1)
                if "%" in value or "inherit" in value or "var(" in value:
                    continue
                for part in value.split():
                    px = re.match(r"^(\d+(?:\.\d+)?)px$", part)
                    if px and float(px.group(1)) > 6:
                        offenders.append(f"{sheet.name}:{number} {line.strip()}")
        assert not offenders, offenders[:5]

    def test_the_decorative_shadows_and_gradients_are_gone(self):
        """45 of the 80 lines of the old token file were drop shadows, inset
        shadows and two-stop gradients on chart containers. None of them
        encoded anything."""
        text = TOKENS.read_text()
        for retired in (
            "--chart-shadow",
            "--segment-shadow",
            "--label-shadow",
            "--legend-shadow",
            "--chart-bg",
            "--svg-bg",
        ):
            assert retired not in text, retired


# --------------------------------------------------------------------------- #
# typography
# --------------------------------------------------------------------------- #
class TestTypography:
    def test_arial_is_no_longer_the_face(self):
        """The report was set in `Arial, sans-serif`.

        Arial itself is not banned -- it survives as a late fallback inside
        `--font-sans`, which is what a system stack is for. What goes is Arial
        as the *first* choice, which is what actually rendered.
        """
        leading = [
            entry
            for entry in _declarations(r"font-family:\s*")
            if re.match(r"font-family:\s*(Arial|Helvetica)\b", entry[2])
        ]
        assert not leading, leading

    def test_every_font_family_is_a_token_or_inherit(self):
        """Monospace was spelled out as a literal stack in 30 places across 8
        files. One of them differed from the others, which is what a copied
        literal always eventually does."""
        stragglers = [
            entry
            for entry in _declarations(r"font-family:")
            if "var(--font-" not in entry[2] and "inherit" not in entry[2]
        ]
        assert not stragglers, stragglers[:5]

    def test_both_font_tokens_are_defined(self):
        text = TOKENS.read_text()
        assert "--font-sans:" in text
        assert "--font-mono:" in text

    def test_the_micro_label_convention_ships(self):
        """Every section header and stat caption uses it, so it lives in the
        token layer rather than being respelled per component."""
        text = TOKENS.read_text()
        assert ".micro-label" in text
        block = text.split(".micro-label", 1)[1]
        assert "var(--font-mono)" in block
        assert "uppercase" in block


# --------------------------------------------------------------------------- #
# the shim
# --------------------------------------------------------------------------- #
class TestTokensAreDefinedInExactlyOnePlace:
    """The bug this exists for, found by inspecting a rendered report rather
    than the stylesheet:

    `_06-cards.css` carried a block redefining `--axis`, `--axis-text`,
    `--border-color` and `--bar`. The stylesheets are concatenated in filename
    order, so it loaded *after* `_00-tokens.css` and silently won. The report
    drew its axes in `rgba(0, 0, 0, 0.45)` while the token file said `#8F8474`
    -- and `test_contrast.py`, which reads the token file, passed the whole
    time, because it was measuring a value nothing rendered.

    A contrast guard that reads only the token definitions is worth exactly as
    much as the guarantee that those definitions are the ones in force.
    """

    def test_no_stylesheet_redefines_a_design_token(self):
        """Component-local variables are fine -- `--bool-true: var(--data-2)`
        derives from the system and reads better at the point of use. What is
        not fine is *reassigning* a name the token file already owns, because
        then the value in force is decided by filename order.
        """
        owned = set(re.findall(r"^\s*(--[\w-]+)\s*:", TOKENS.read_text(), re.M))
        assert len(owned) > 20, "the token file parsed as nearly empty"

        offenders = []
        for sheet in STYLESHEETS:
            if sheet.name == TOKENS.name:
                continue
            media_depth = 0  # nesting depth inside an @media block
            depth = 0
            for number, line in enumerate(sheet.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("@media"):
                    media_depth = depth + 1
                match = re.match(r"^\s*(--[\w-]+)\s*:", line)
                # A responsive adjustment inside @media is a legitimate
                # override of a *layout* token -- the mobile bar is taller, so
                # the anchor offset follows it. Shadowing at base scope is the
                # bug: it is unconditional and invisible.
                if match and match.group(1) in owned and media_depth == 0:
                    offenders.append(f"{sheet.name}:{number} {stripped}")
                depth += line.count("{") - line.count("}")
                if media_depth and depth < media_depth:
                    media_depth = 0
        assert not offenders, (
            "these reassign a token that _00-tokens.css owns, so load order "
            f"decides the value: {offenders[:5]}"
        )

    @pytest.mark.parametrize("token", ["--axis", "--axis-text", "--bar", "--ink"])
    def test_a_token_the_charts_rely_on_has_a_single_definition(self, token):
        definitions = [
            f"{sheet.name}:{number}"
            for sheet in STYLESHEETS
            for number, line in enumerate(sheet.read_text().splitlines(), 1)
            if re.match(rf"^\s*{token}\s*:", line)
        ]
        # One per theme: light and dark. Never more, and never in two files.
        assert definitions, f"{token} is never defined"
        assert all(d.startswith(TOKENS.name) for d in definitions), definitions


class TestTheCompatibilityShim:
    """The shim maps legacy variable names onto the new scale so the rest of
    the CSS keeps working while later phases land. It is scaffolding, and the
    thing about scaffolding is that it gets left up."""

    def test_it_is_labelled_as_temporary(self):
        text = TOKENS.read_text()
        assert "COMPATIBILITY SHIM" in text
        assert "DELETE THIS BLOCK" in text

    def test_every_legacy_name_it_maps_resolves_to_a_new_token(self):
        text = TOKENS.read_text()
        shim = text.split("COMPATIBILITY SHIM", 1)[1].split("}", 1)[0]
        mappings = re.findall(r"(--[\w-]+):\s*([^;]+);", shim)
        assert mappings, "the shim block parsed as empty"
        defined = set(re.findall(r"(--[\w-]+):", text))
        for name, value in mappings:
            for referenced in re.findall(r"var\((--[\w-]+)\)", value):
                assert referenced in defined, f"{name} maps to undefined {referenced}"
