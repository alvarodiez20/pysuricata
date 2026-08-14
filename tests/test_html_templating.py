"""Tests for report template placeholder substitution.

The template is filled with a single regex pass rather than ``str.format()``
(CSS custom properties and JS braces break ``.format()``) and rather than
sequential ``str.replace()`` calls (which rescan already-substituted values).
"""

import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.api import ProfileConfig, RenderOptions


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3], "g": ["a", "b", "a"]})


def _render(df, **render_kwargs):
    return profile(df, config=ProfileConfig(render=RenderOptions(**render_kwargs))).html


def test_title_containing_placeholder_is_not_expanded(df):
    """A user title with braces survives verbatim.

    Regression test: sequential str.replace() substituted {css} first, then kept
    scanning the whole document, so a title of "My {report_date} report" had its
    braces expanded into the actual timestamp.
    """
    html = _render(df, title="My {report_date} report")
    assert "My {report_date} report" in html


def test_description_containing_placeholder_is_not_expanded(df):
    html = _render(df, description="row count is {n_rows} today")
    assert "row count is {n_rows} today" in html


def test_title_cannot_inject_a_later_placeholder(df):
    """Keys substituted after report_title must not be reachable from it."""
    html = _render(df, title="{description_html}{variables_section}")
    assert "{description_html}{variables_section}" in html


def test_no_placeholders_remain_unsubstituted(df):
    """Every real placeholder in the template is filled."""
    html = _render(df, title="Report")
    for key in (
        "{css}",
        "{script}",
        "{ script }",
        "{favicon}",
        "{logo}",
        "{report_title}",
        "{report_date}",
        "{n_rows}",
        "{variables_section}",
        "{description_html}",
    ):
        assert key not in html, f"unsubstituted placeholder {key}"


def test_css_and_js_payloads_survive_substitution(df):
    """Brace-heavy CSS/JS is passed through untouched."""
    html = _render(df, title="Report")
    assert "<style>" in html and "</style>" in html
    # CSS custom properties use braces that .format() would have choked on.
    assert "--" in html
    # Delegated event handling from the inline script.
    assert "data-action" in html


def test_unknown_placeholder_in_template_is_left_verbatim():
    """Unrecognised keys are preserved rather than raising KeyError."""
    import re

    from pysuricata.render.html import _PLACEHOLDER_RE

    replacements = {"known": "VALUE"}

    def resolve(m: "re.Match[str]") -> str:
        return str(replacements.get(m.group(1), m.group(0)))

    out = _PLACEHOLDER_RE.sub(resolve, "a {known} b {unknown} c {--css-var} d")
    assert out == "a VALUE b {unknown} c {--css-var} d"
