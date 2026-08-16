"""CSS and HTML integrity tests.

Guards against regressions in CSS architecture (deduplication, !important budget,
breakpoint standardization) and HTML template quality (no inline event handlers).
"""

import glob
import os
import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _css_dir():
    """Return the path to the CSS partials directory."""
    return os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "pysuricata",
        "static",
        "css",
    )


def _concatenated_css():
    """Read all _*.css partials and concatenate in sorted order."""
    css_dir = _css_dir()
    parts = []
    for path in sorted(glob.glob(os.path.join(css_dir, "_*.css"))):
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    return "".join(parts)


def _template_path():
    return os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "pysuricata",
        "templates",
        "report_template.html",
    )


# ---------------------------------------------------------------------------
# Phase 5A — load_css_dir
# ---------------------------------------------------------------------------


def test_load_css_dir_concatenates_in_order(tmp_path):
    """Partials are concatenated in sorted filename order."""
    from pysuricata.utils import load_css_dir

    (tmp_path / "_02-second.css").write_text(".b{}", encoding="utf-8")
    (tmp_path / "_01-first.css").write_text(".a{}", encoding="utf-8")
    # Non-matching files should be ignored
    (tmp_path / "ignored.css").write_text(".x{}", encoding="utf-8")

    # lru_cache is keyed on the string arg, so tmp_path is unique per test
    result = load_css_dir(str(tmp_path))
    assert result == "<style>.a{}.b{}</style>"


def test_load_css_dir_empty_dir(tmp_path):
    """Empty directory returns empty string."""
    from pysuricata.utils import load_css_dir

    result = load_css_dir(str(tmp_path))
    assert result == ""


def test_load_css_dir_caching(tmp_path):
    """Second call returns cached result (same object identity)."""
    from pysuricata.utils import load_css_dir

    (tmp_path / "_01-a.css").write_text("body{}", encoding="utf-8")
    first = load_css_dir(str(tmp_path))
    second = load_css_dir(str(tmp_path))
    assert first is second  # same cached object


# ---------------------------------------------------------------------------
# Phase 5C — deduplication
# ---------------------------------------------------------------------------


def test_no_duplicate_hist_svg_blocks():
    """The canonical .hist-svg base block should appear exactly once."""
    css = _concatenated_css()
    count = css.count("#pysuricata-report .hist-svg {")
    assert count == 1, (
        f"Expected 1 canonical .hist-svg block, found {count}. "
        "Remove duplicate copies — keep only the one in _07-histogram.css."
    )


# ---------------------------------------------------------------------------
# Phase 5D — !important budget
# ---------------------------------------------------------------------------


def test_important_count_within_budget():
    """Only prefers-reduced-motion and documented exceptions may use !important."""
    css = _concatenated_css()
    count = css.count("!important")
    budget = 6  # 4 reduced-motion + 1 legacy kill switch + 1 buffer
    assert count <= budget, (
        f"Found {count} !important declarations (budget: {budget}). "
        "Fix specificity instead of adding !important."
    )


# ---------------------------------------------------------------------------
# Phase 5E — breakpoint standardization
# ---------------------------------------------------------------------------


def test_breakpoints_use_standard_values():
    """All @media breakpoints must use the approved set."""
    css = _concatenated_css()
    allowed = {480, 560, 768, 1024, 1440}
    pattern = re.compile(r"@media\s*\(\s*(?:max|min)-width:\s*(\d+)px\s*\)")
    found = {int(m) for m in pattern.findall(css)}
    unexpected = found - allowed
    assert not unexpected, (
        f"Non-standard breakpoints found: {sorted(unexpected)}px. "
        f"Allowed: {sorted(allowed)}px."
    )


# ---------------------------------------------------------------------------
# Phase 5B — partial file inventory
# ---------------------------------------------------------------------------


def test_css_partials_present():
    """All expected CSS partial files must exist."""
    expected = {
        "_00-tokens.css",
        "_01-base.css",
        "_02-header.css",
        "_03-summary.css",
        # _04-donut.css is deliberately absent: the donut it styled was
        # replaced by the composition bar (#104, #112), whose styles live with
        # the rest of the summary in _03. The numbering is left with a gap
        # rather than renumbering every later partial, which would make the
        # diff of that change unreadable for no benefit.
        "_05-sample.css",
        "_06-cards.css",
        "_07-histogram.css",
        "_08-categorical.css",
        "_09-datetime.css",
        "_10-boolean.css",
        "_11-correlations.css",
        "_12-missing.css",
        "_13-utilities.css",
    }
    css_dir = _css_dir()
    actual = {os.path.basename(p) for p in glob.glob(os.path.join(css_dir, "_*.css"))}
    missing = expected - actual
    assert not missing, f"Missing CSS partials: {sorted(missing)}"


# ---------------------------------------------------------------------------
# HTML template guards
# ---------------------------------------------------------------------------


def test_no_inline_event_handlers_in_template():
    """Template must use data-action attributes, not inline event handlers."""
    with open(_template_path(), encoding="utf-8") as f:
        html = f.read()
    forbidden = re.findall(r"\b(onclick|ontoggle|onload|onchange|onsubmit)\s*=", html)
    assert not forbidden, (
        f"Inline event handlers found in template: {forbidden}. "
        "Use data-action attributes and addEventListener instead."
    )


def test_template_has_data_action_attributes():
    """Key interactive elements must have data-action attributes."""
    with open(_template_path(), encoding="utf-8") as f:
        html = f.read()
    required_actions = [
        "scroll-to-top",
        "toggle-dark-mode",
        "download-report",
        "toggle-pin",
    ]
    for action in required_actions:
        assert f'data-action="{action}"' in html, (
            f'Missing data-action="{action}" in template'
        )
