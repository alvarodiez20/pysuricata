"""The sample table: what it shows, and what it must never print.

Phase 4 of the redesign, and it closes #103.

The old table drew a border on every cell — about 300 of them for a 10 × 13
sample — and striped alternate rows on top, so the grid competed with the data.
It also printed ``nan`` in every empty cell, which reads as a value rather than
as absence: a column of ``nan`` looks like text data.

The distinction this file exists to protect is between **a null** and **a
string that happens to spell one**. A column named ``nan``, and the three
characters ``n``, ``a``, ``n``, are real data. Dashing them would not be tidying
up; it would be corrupting a value and calling it missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.sections import (
    _build_simple_table_html,
    _is_null,
    render_sample_section,
)


@pytest.fixture(scope="module")
def html() -> str:
    frame = pd.DataFrame(
        {
            "n": [1.0, np.nan, 3.0],
            "s": ["x", "nan", None],
            "long": ["z" * 500, "q", "w"],
        }
    )
    return profile(frame, seed=0).html


# --------------------------------------------------------------------------- #
# nulls
# --------------------------------------------------------------------------- #
class TestANullIsADashNotTheWordNan:
    @pytest.mark.parametrize("value", [None, float("nan"), np.nan])
    def test_these_are_null(self, value):
        assert _is_null(value)

    @pytest.mark.parametrize("value", ["nan", "NaN", "None", "", 0, 0.0, False, "-"])
    def test_these_are_not(self, value):
        """Every one of these is a value somebody stored deliberately."""
        assert not _is_null(value)

    def test_a_null_renders_as_a_dash_and_keeps_its_value(self):
        table = _build_simple_table_html(["", "a"], [[0, None]], [0])
        assert "—" in table
        assert 'class="nil"' in table

    def test_a_string_that_spells_nan_is_left_alone(self):
        """The edge case named in #113. It is text, and it renders as text."""
        table = _build_simple_table_html(["", "a"], [[0, "nan"]], [0])
        assert ">nan</td>" in table
        assert "nil" not in table

    def test_a_column_named_nan_is_left_alone(self):
        table = _build_simple_table_html(["", "nan"], [[0, 1]], [0])
        assert ">nan</th>" in table

    def test_zero_is_not_missing(self):
        """Dashing a zero would be the most damaging version of this bug: it
        turns a measured value into an absent one."""
        table = _build_simple_table_html(["", "a"], [[0, 0]], [0, 1])
        assert "nil" not in table
        assert ">0</td>" in table

    def test_the_report_never_prints_a_bare_nan_for_a_null(self, html):
        body = html.split('class="sample-scroll"', 1)[1].split("</div>", 1)[0]
        assert ">nan</td>" in body  # the genuine string survives
        assert body.count('class="nil"') == 2  # the two real nulls


# --------------------------------------------------------------------------- #
# decoration
# --------------------------------------------------------------------------- #
class TestTheDecorationIsGone:
    def test_no_cell_carries_its_own_border(self):
        css = (
            pytest.importorskip("pathlib")
            .Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("pysuricata/static/css/_05-sample.css")
            .read_text()
        )
        table = css.split("table.sample-table td {", 1)[1].split("}", 1)[0]
        assert "border: 0" in table
        assert "border-bottom" in table  # the row rule stays

    def test_the_striping_is_gone(self):
        css = (
            pytest.importorskip("pathlib")
            .Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("pysuricata/static/css/_05-sample.css")
            .read_text()
        )
        assert "nth-child(even)" not in css

    def test_long_values_clamp_rather_than_stretch(self, html):
        """A 500-character cell would otherwise widen the pane until nothing
        else fits."""
        assert "text-overflow: ellipsis" in html
        assert "z" * 200 in html  # the value itself is not truncated in the DOM

    def test_a_long_value_keeps_its_full_text_in_a_title(self):
        table = _build_simple_table_html(["", "a"], [[0, "y" * 60]], [0])
        assert 'title="' + "y" * 60 in table


# --------------------------------------------------------------------------- #
# the frozen index
# --------------------------------------------------------------------------- #
class TestTheFrozenIndex:
    def test_the_index_cell_is_marked(self):
        table = _build_simple_table_html(["", "a"], [[0, 1]], [0])
        assert '<td class="idx num"' in table
        assert '<th class="idx num"' in table

    def test_it_is_sticky_rather_than_a_second_table(self):
        """Two tables side by side have to be kept in vertical step by hand,
        and a cell that wraps in one desynchronises the pair -- the drift the
        design package warns about. A sticky cell belongs to its own row, so
        the alignment cannot come apart."""
        css = (
            pytest.importorskip("pathlib")
            .Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("pysuricata/static/css/_05-sample.css")
            .read_text()
        )
        block = css.split("table.sample-table td.idx", 1)[1].split("}", 1)[0]
        assert "position: sticky" in block
        assert "left: 0" in block


# --------------------------------------------------------------------------- #
# what the reader is told
# --------------------------------------------------------------------------- #
class TestTheNotices:
    def test_the_rows_are_described_as_a_random_draw(self, html):
        """They are not the head of the file, and a reader who assumes they are
        will misread every value in the table."""
        assert "drawn at random from the first chunk" in html

    def test_the_overflow_is_stated(self):
        wide = pd.DataFrame({f"c{i}": [1, 2] for i in range(12)})
        out = render_sample_section(wide, 2, seed=0)
        assert "cols · scroll →" in out
        assert "12 cols" in out

    def test_a_single_column_frame_promises_no_scrolling(self):
        """There is nothing off-screen, so the arrow would point at nothing."""
        out = render_sample_section(pd.DataFrame({"a": [1, 2]}), 2, seed=0)
        assert "scroll →" not in out

    def test_an_empty_frame_does_not_crash(self):
        out = render_sample_section(pd.DataFrame({"a": []}), 5, seed=0)
        assert "<table" in out


class TestItOpensOnLoad:
    """#103. The sample is the fastest way to see whether the profile matches
    the data, and behind a click most readers never opened it."""

    def test_the_details_element_is_open(self, html):
        assert 'id="sample-details" open' in html

    def test_it_is_still_collapsible(self, html):
        assert "<summary>" in html.split('id="sample-details"', 1)[1][:400]

    def test_the_toggle_says_what_clicking_will_do(self, html):
        assert "Hide sample" in html
