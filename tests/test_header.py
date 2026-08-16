"""The header bar, and the metadata that used to hide in a tooltip.

Phase 2 of the redesign. The old header was a two-row grid with a 78px logo
column: about 96px of chrome before any data, carrying a bare timestamp, a bare
duration and a bare ``891 × 12`` whose meaning lived in a ``title`` attribute.

A tooltip survives neither printing, nor a screenshot, nor PDF export — which
are three of the four ways anyone looks at a report they did not just generate.
So the metadata moved into the page and got labels.

There was no test covering any of this before, which is why nothing failed when
the markup it described was deleted. Layout *measurements* (52px, 48px, the
44px targets, the rail fitting 390px) were taken in a browser; what a Python
test can hold is the contract the CSS states and the markup the renderer emits.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata._version import resolve_version
from pysuricata.render.html import _build_dataset_name

CSS = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"
HEADER_CSS = (CSS / "_02-header.css").read_text()


def _header_element(html: str) -> str:
    """Just the <header>…</header>.

    Splitting on "</header>" is not enough: the inlined stylesheet and script
    sit in <head>, ahead of it, and both contain strings this file looks for --
    the markdown renderer in `functionality.js` emits "<h1>", and the CSS names
    `.report-meta`. Two of these tests passed the wrong thing before this
    existed.
    """
    start = html.index("<header")
    return html[start : html.index("</header>", start)]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {"age": rng.integers(1, 80, n).astype(float), "sex": rng.choice(["m", "f"], n)}
    )


@pytest.fixture(scope="module")
def html(frame) -> str:
    return profile(frame, seed=0).html


# --------------------------------------------------------------------------- #
# the bar
# --------------------------------------------------------------------------- #
class TestTheBar:
    def test_it_is_one_row(self, html):
        assert 'class="container bar"' in html

    def test_the_two_row_grid_is_gone(self, html):
        """`header-grid`, `title-nav`, `nav-and-toggle` and the `meta` row were
        the shape that made the header 96px tall."""
        for dead in (
            'class="header-grid"',
            'class="title-nav"',
            'class="nav-and-toggle"',
        ):
            assert dead not in html, dead

    def test_its_height_comes_from_a_token(self):
        assert "height: var(--appbar-h);" in HEADER_CSS
        assert "height: var(--appbar-h-sm);" in HEADER_CSS

    def test_the_brand_links_to_the_top(self, html):
        assert 'class="brand"' in html
        assert 'data-action="scroll-to-top"' in html

    def test_the_mark_is_thirty_pixels(self):
        """A 52px bar cannot hold the 44px lockup the old header used."""
        block = HEADER_CSS.split("#pysuricata-report #logo .logo-mark", 1)[1]
        assert "height: 30px" in block.split("}", 1)[0]


# --------------------------------------------------------------------------- #
# what was profiled
# --------------------------------------------------------------------------- #
class TestTheDatasetName:
    def test_a_frame_has_no_name_and_no_dangling_separator(self, html):
        """An in-memory frame is most inputs. The separator is emitted with the
        name or not at all, so the bar does not show a rule with nothing after
        it on the common case."""
        assert 'class="dataset-name"' not in html
        assert 'class="bar-sep"' in html  # the one before the action icons
        assert html.count('class="bar-sep"') == 1

    def test_a_path_is_named_by_its_file(self, tmp_path, frame):
        target = tmp_path / "passengers.csv"
        frame.to_csv(target, index=False)
        rendered = profile(str(target), seed=0).html
        assert 'class="dataset-name"' in rendered
        assert "passengers.csv" in rendered

    def test_the_name_is_the_file_not_the_whole_path(self, tmp_path, frame):
        target = tmp_path / "passengers.csv"
        frame.to_csv(target, index=False)
        rendered = profile(str(target), seed=0).html
        assert str(tmp_path) not in rendered

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_nothing_is_emitted_for_an_absent_name(self, blank):
        assert _build_dataset_name(blank) == ""

    def test_a_name_is_escaped(self):
        """It reaches the DOM from a filename, and a filename can contain
        anything the filesystem allows."""
        out = _build_dataset_name("<script>alert(1)</script>")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_the_separator_travels_with_the_name(self):
        out = _build_dataset_name("sales.parquet")
        assert out.index("bar-sep") < out.index("dataset-name")

    def test_a_long_name_ellipses_rather_than_pushing_the_icons_off(self):
        block = HEADER_CSS.split("#pysuricata-report .dataset-name", 1)[1].split(
            "}", 1
        )[0]
        assert "text-overflow: ellipsis" in block
        assert "min-width: 0" in block


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
class TestTheSections:
    def test_all_five_are_present(self, html):
        for anchor in (
            "#summary",
            "#sample",
            "#vars",
            "#correlations",
            "#missing-values",
        ):
            assert f'href="{anchor}"' in html, anchor

    def test_they_are_plain_text_not_pills(self):
        block = HEADER_CSS.split("#pysuricata-report .quick a,", 1)[1].split("}", 1)[0]
        assert "border-radius" not in block
        assert "background" not in block

    def test_the_active_one_is_marked_by_a_rule_not_a_fill(self):
        block = HEADER_CSS.split("#pysuricata-report .quick a.active", 1)[1].split(
            "}", 1
        )[0]
        assert "border-bottom-color: var(--q-good)" in block
        assert "background" not in block

    def test_the_long_label_has_a_short_form_for_the_rail(self, html):
        """Five labels have to fit 390px without swiping; `Missing Values` is
        the one that does not."""
        assert 'class="nav-long">Missing Values<' in html
        assert 'class="nav-short">Missing<' in html


class TestTheMobileRail:
    def test_it_uses_min_height_never_height(self):
        """With `overflow-x` a fixed height has the scrollbar subtracted from
        it, which silently leaves a 29px rail that fails the 44px target while
        measuring as though it passed."""
        rail = HEADER_CSS.split("#pysuricata-report .appbar .quick {", 1)[1].split(
            "}", 1
        )[0]
        assert "min-height: var(--tap-min)" in rail
        assert not re.search(r"\n\s*height:", rail)

    def test_the_scrollbar_is_hidden_in_both_engines(self):
        assert "scrollbar-width: none" in HEADER_CSS
        assert "::-webkit-scrollbar" in HEADER_CSS

    def test_the_breakpoint_is_one_the_project_uses(self):
        """`test_css_integrity` owns the approved set; this states which one
        the header picked, so a change here is deliberate."""
        assert "@media (max-width: 768px)" in HEADER_CSS


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
class TestTheActions:
    def test_download_and_dark_mode_are_present(self, html):
        assert 'data-action="download-report"' in html
        assert 'data-action="toggle-dark-mode"' in html

    def test_the_pin_joined_the_icon_group(self, html):
        """It used to sit alone at `margin-left:auto` in a meta row that no
        longer exists. `functionality.js` injects one into `.quick` when it
        finds none, which would drop an icon into the text nav."""
        assert 'data-action="toggle-pin"' in html
        actions = html.split('class="bar-actions"', 1)[1].split("</div>", 1)[0]
        assert "toggle-pin" in actions

    def test_every_action_is_labelled(self, html):
        actions = _header_element(html).split('class="bar-actions"', 1)[1]
        for control in re.findall(r"<(?:a|button)\b[^>]*>", actions):
            assert "aria-label=" in control, control

    def test_the_target_is_larger_than_the_paint(self):
        """The visual box is 30px because a 52px bar has room for 30. A finger
        needs 44, so the hit area is extended past the paint rather than the
        box being grown, which would blow the bar height."""
        overlay = HEADER_CSS.split("#pysuricata-report .icon-btn::after", 1)[1].split(
            "}", 1
        )[0]
        assert "width: var(--tap-min)" in overlay
        assert "height: var(--tap-min)" in overlay

    def test_the_off_palette_icon_colours_are_gone(self):
        """Indigo-500 and amber-500 were on neither of the two scales."""
        for retired in ("#6366f1", "#f59e0b", "#0ea5e9", "#10b981"):
            assert retired not in HEADER_CSS, retired


# --------------------------------------------------------------------------- #
# the metadata line
# --------------------------------------------------------------------------- #
class TestTheMetadataIsLabelled:
    def test_it_left_the_bar(self, html):
        assert 'class="report-meta"' in html
        assert "report-meta" not in _header_element(html)

    @pytest.mark.parametrize("label", ["Generated", "Profiled in", "Shape"])
    def test_each_figure_says_what_it_is(self, html, label):
        assert f">{label}</span>" in html

    def test_the_shape_is_spelled_out(self, html):
        """`891 × 12` meant nothing without the tooltip that explained it."""
        assert re.search(r"[\d,]+ rows × [\d,]+ columns", html)

    def test_the_old_tooltip_chips_are_gone(self, html):
        assert 'data-tooltip-type="date"' not in html
        assert 'data-tooltip-type="duration"' not in html

    def test_the_version_is_read_not_written(self, html):
        """A literal in the template goes stale on the next release."""
        assert f"pysuricata {resolve_version()}" in html

    def test_it_is_set_in_the_monospace_token(self):
        block = HEADER_CSS.split("#pysuricata-report .report-meta {", 1)[1].split(
            "}", 1
        )[0]
        assert "font-family: var(--font-mono)" in block

    def test_the_title_moved_into_the_page(self, html):
        assert 'class="report-head"' in html
        assert "<h1" not in _header_element(html)
