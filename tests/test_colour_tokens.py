"""Every colour in the stylesheets must come from a token. A ratchet, not a ban.

#110's acceptance was that none of nine named hex values appears in
`static/css/`. **All nine are clear** -- and there are 94 distinct hex values in
the stylesheets, including Tailwind's amber, red and green and Material's
orange, green and red. The list named the *accent* colours being replaced and
missed the semantic ramp, so the check passes while both frameworks' defaults
sit in the file.

A ban list can always be outgrown by a colour nobody thought to ban. The
assertion runs the other way here: **every hex outside `_00-tokens.css` must
equal a value the token file defines.**

That cannot be satisfied today -- 67 file/value pairs do not -- so it ships as a
ratchet against a recorded baseline. A new literal fails immediately; removing
one is expected and fails too, loudly, telling you to shrink the baseline. The
number only goes down.

It started at 88 and reached 70 in #122. Eleven went without anyone choosing a
colour: they sat in rules for markup the redesign had already replaced, and
left with 285 dead selectors. That is worth noticing about this kind of debt --
a good deal of it is not a decision anyone still has to make.

The other seven were chosen, and they were chosen because they *failed*. A
contrast audit over the rendered report found 204 text nodes below 4.5:1, all
of them tracing back to eight framework literals printed on a 10% wash of
themselves. Every one would have passed if measured against `--paper`. See
#122's note on ancestor backgrounds, and `test_contrast.py`.

Two things are deliberately not counted:

* `var(--paper, #FBF9F5)` -- a fallback *for a token that exists* is the
  documented robustness pattern, not an untokenised colour.
* `body.suricata-standalone`, which sits **outside** `#pysuricata-report` and so
  cannot read tokens scoped to it. That is the only place in the stylesheets
  that has to repeat a palette value, and the file says so.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"
TOKENS = CSS_DIR / "_00-tokens.css"

#: Every file/value pair that does not yet resolve to a token, as of 0.1.0.
#: Delete entries as they are tokenised. Adding one requires a reason in review.
BASELINE: set[str] = {
    "_03-summary.css #3b8ad1",
    "_03-summary.css #4a9ae1",
    "_03-summary.css #d14e3e",
    "_03-summary.css #d1a53e",
    "_03-summary.css #e15e4e",
    "_03-summary.css #e1b54e",
    "_05-sample.css #1a1a1a",
    "_06-cards.css #0366d6",
    "_06-cards.css #111",
    "_06-cards.css #16a34a",
    "_06-cards.css #1e3a8a",
    "_06-cards.css #1e40af",
    "_06-cards.css #22c55e",
    "_06-cards.css #2563eb",
    "_06-cards.css #2a2a2a",
    "_06-cards.css #888",
    "_06-cards.css #93c5fd",
    "_06-cards.css #aaa",
    "_06-cards.css #bfdbfe",
    "_06-cards.css #d97706",
    "_06-cards.css #dbeafe",
    "_06-cards.css #e0e0e0",
    "_06-cards.css #ef4444",
    "_06-cards.css #f59e0b",
    "_06-cards.css #f9f9f9",
    "_06-cards.css #fff",
    "_08-categorical.css #007acc",
    "_08-categorical.css #2a2a2a",
    "_08-categorical.css #4db8ff",
    "_08-categorical.css #666",
    "_08-categorical.css #66c2ff",
    "_08-categorical.css #aaa",
    "_08-categorical.css #e0e0e0",
    "_09-datetime.css #2563eb",
    "_09-datetime.css #6b7280",
    "_09-datetime.css #9ca3af",
    "_09-datetime.css #d1d5db",
    "_09-datetime.css #f9f9f9",
    "_10-boolean.css #fff",
    "_11-correlations.css #2a2a2a",
    "_11-correlations.css #d32f2f",
    "_11-correlations.css #e0e0e0",
    "_11-correlations.css #f44336",
    "_11-correlations.css #f57c00",
    "_11-correlations.css #fbc02d",
    "_11-correlations.css #ff9800",
    "_11-correlations.css #ffeb3b",
    "_12-missing.css #10b981",
    "_12-missing.css #1a1a1a",
    "_12-missing.css #22c55e",
    "_12-missing.css #34d399",
    "_12-missing.css #45a049",
    "_12-missing.css #4caf50",
    "_12-missing.css #66bb6a",
    "_12-missing.css #86efac",
    "_12-missing.css #dc2626",
    "_12-missing.css #e5e7eb",
    "_12-missing.css #ef4444",
    "_12-missing.css #ef5350",
    "_12-missing.css #f44336",
    "_12-missing.css #f57c00",
    "_12-missing.css #f59e0b",
    "_12-missing.css #f87171",
    "_12-missing.css #fb923c",
    "_12-missing.css #fbbf24",
    "_12-missing.css #ff9800",
    "_12-missing.css #ffb74d",
}


def _token_values() -> set[str]:
    text = TOKENS.read_text(encoding="utf-8")
    return {
        value.lower()
        for _, value in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{3,8})\s*;", text)
    }


def _literals(text: str) -> list[str]:
    """Hex values that are not a `var()` fallback and not in a comment."""
    code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    code = re.sub(r"var\(\s*--[\w-]+\s*,\s*#[0-9a-fA-F]{3,8}\s*\)", "var(--x)", code)
    return [h.lower() for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", code)]


def _offenders() -> set[str]:
    tokens = _token_values()
    found: set[str] = set()
    for path in sorted(CSS_DIR.glob("*.css")):
        if path.name == TOKENS.name:
            continue
        for value in _literals(path.read_text(encoding="utf-8")):
            if value not in tokens:
                found.add(f"{path.name} {value}")
    return found


class TestTheRatchetOnlyTurnsOneWay:
    def test_no_new_untokenised_colour(self):
        added = sorted(_offenders() - BASELINE)
        assert not added, (
            "new hex values that no token defines:\n  "
            + "\n  ".join(added)
            + "\n\nAdd the colour to _00-tokens.css and reference it as var(--name)."
        )

    def test_the_baseline_does_not_outlive_the_debt(self):
        """Removing a literal is the goal, and it must shrink this list rather
        than leaving a stale entry that quietly permits the colour's return."""
        gone = sorted(BASELINE - _offenders())
        assert not gone, (
            "these are tokenised now -- delete them from BASELINE:\n  "
            + "\n  ".join(gone)
        )

    def test_the_count_is_what_the_roadmap_says(self):
        """So the number in the audit cannot drift from the number in the code."""
        assert len(BASELINE) == 67


class TestTheAssertionIsWorthHaving:
    def test_a_ban_list_would_have_missed_these(self):
        """#110 banned nine named accent colours and they are all gone. These
        are the framework defaults that walked in behind them."""
        offenders = {entry.split()[1] for entry in _offenders()}
        # Tailwind blue-600, green-500; Material red 700.
        for sneaked in ("#2563eb", "#22c55e", "#d32f2f"):
            assert sneaked in offenders, f"expected {sneaked} in the recorded debt"

    def test_the_retired_hues_really_are_clear(self):
        """The original acceptance still holds; it was just not enough.

        Checked against the list `test_design_system.py` actually carries --
        four retired column-type hues -- not the nine a first draft of this file
        invented. Asserting against a made-up list is how you get a green test
        that proves nothing; this one went red on `#93c5fd` and the list turned
        out to be mine rather than the project's.
        """
        from test_design_system import TestTypeIsNotAColour as Legacy

        retired = {f"#{value}" for value in Legacy.LEGACY}
        offenders = {entry.split()[1] for entry in _offenders()}
        assert not (retired & offenders), sorted(retired & offenders)


class TestTheExclusionsAreNarrow:
    def test_a_var_fallback_is_not_counted(self):
        assert _literals("a { color: var(--paper, #FBF9F5); }") == []

    def test_a_bare_literal_is_counted(self):
        assert _literals("a { color: #FBF9F5; }") == ["#fbf9f5"]

    def test_a_commented_literal_is_not_counted(self):
        """The third check this session to be tripped by prose it read as code."""
        assert _literals("/* was #ff0000 */\na { color: var(--ink); }") == []

    def test_the_token_file_itself_defines_colours(self):
        assert len(_token_values()) >= 25


class TestTheTokenFileIsTheOnlyPalette:
    @pytest.mark.parametrize(
        "path", sorted(CSS_DIR.glob("*.css")), ids=lambda p: p.name
    )
    def test_each_file_is_at_or_below_its_recorded_debt(self, path):
        if path.name == TOKENS.name:
            pytest.skip("the token file is where colours are allowed to be literal")
        recorded = {e for e in BASELINE if e.startswith(path.name + " ")}
        actual = {e for e in _offenders() if e.startswith(path.name + " ")}
        assert actual <= recorded, sorted(actual - recorded)
