"""The missing-values section, routed on chunk count rather than on tabs.

Phase 7 (#120). `Data Completeness` and `Missing per Chunk` over three rows was
two clicks for one screen of content — and with a single chunk the second tab
held one full-width block per column, a tab that hid nothing.

The by-chunk half of that issue is **not implemented here, and cannot be**: the
per-column per-chunk missing counts it needs are never produced. That is #139,
found while doing this. The renderer for the strip is written and correct; the
route degrades to the single-column view because the chunk count derived from
real data is always 1, which is what #120's own edge case asks for when the
metadata is unavailable.

The tests below therefore cover the view that ships, the degradation, and the
strip renderer in isolation so it is ready when the data arrives.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.missing_section import MissingValuesSectionRenderer


def _section(html: str) -> str:
    start = html.index('id="missing-values"')
    end = html.find("</section>", start)
    return html[start : end if end != -1 else len(html)]


@pytest.fixture(scope="module")
def with_missing() -> str:
    rng = np.random.default_rng(0)
    n = 900
    values = rng.normal(0, 1, n)
    values[rng.choice(n, 320, replace=False)] = np.nan
    return profile(
        pd.DataFrame(
            {
                "mostly_there": rng.choice([None, "x"], n, p=[0.03, 0.97]),
                "half_gone": values,
                "solid": rng.normal(0, 1, n),
            }
        ),
        seed=0,
    ).html


# --------------------------------------------------------------------------- #
# the tabs
# --------------------------------------------------------------------------- #
class TestTheTabsAreGone:
    def test_no_tab_markup_reaches_the_report(self, with_missing):
        section = _section(with_missing)
        assert "missing-tabs" not in section
        assert "data-tab" not in section
        assert "missing-tab-content" not in section

    def test_the_view_is_chosen_by_chunk_count(self):
        """One chunk of real data, so the single-column view. The routing is
        the conditional that replaced the tabs."""
        renderer = MissingValuesSectionRenderer()
        assert renderer._chunk_count([("a", 5.0, 10, None)]) == 1
        assert renderer._chunk_count([("a", 5.0, 10, [(0, 9, 2), (10, 19, 3)])]) == 2

    def test_missing_metadata_degrades_rather_than_drawing_an_empty_strip(self):
        renderer = MissingValuesSectionRenderer()
        assert renderer._chunk_count([("a", 1.0, 1, [])]) == 1


# --------------------------------------------------------------------------- #
# the row
# --------------------------------------------------------------------------- #
class TestOneRowPerColumn:
    def test_only_columns_with_missing_values_get_a_row(self, with_missing):
        section = _section(with_missing)
        assert section.count('class="miss-row"') == 2
        assert "solid" not in section.split('class="miss-rows"', 1)[1][:400]

    def test_the_row_carries_name_bar_and_figure(self, with_missing):
        section = _section(with_missing)
        assert "miss-row__name" in section
        assert "miss-row__fill" in section
        assert "miss-row__value" in section

    def test_the_bar_never_overflows(self, with_missing):
        widths = [
            float(w)
            for w in re.findall(
                r'miss-row__fill [^"]*" style="width:([\d.]+)%', _section(with_missing)
            )
        ]
        assert widths
        assert all(0.0 <= w <= 100.0 for w in widths)

    def test_the_rows_are_ordered_worst_first(self, with_missing):
        values = re.findall(
            r"miss-row__value [^\"]*\">[\d,]+ \(([\d.]+)%\)", _section(with_missing)
        )
        pcts = [float(v) for v in values]
        assert pcts == sorted(pcts, reverse=True)


class TestSeverityComesFromTheQualityScale:
    """This is the one place data uses the warm scale, because here the
    encoding *is* severity: 77% missing should look worse than 0.2%."""

    @pytest.mark.parametrize(
        ("pct", "expected"),
        [
            (0.0, "good"),
            (4.9, "good"),
            (5.0, "warn"),
            (20.0, "warn"),
            (20.1, "bad"),
            (99.9, "bad"),
        ],
    )
    def test_the_bands(self, pct, expected):
        assert MissingValuesSectionRenderer._severity(pct) == expected

    def test_the_legend_states_them(self, with_missing):
        section = _section(with_missing)
        assert "miss-legend" in section
        assert "≤5%" in section
        assert "5–20%" in section


# --------------------------------------------------------------------------- #
# edge cases from #120
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_nothing_missing_says_so_in_one_line(self):
        """Not an empty grid."""
        rng = np.random.default_rng(0)
        out = profile(pd.DataFrame({"a": rng.normal(0, 1, 300)}), seed=0).html
        section = _section(out)
        assert "No missing values in any of the" in section
        assert 'class="miss-row"' not in section

    def test_complete_columns_are_summarised_not_listed(self):
        """Sixty complete column names is not a summary."""
        rng = np.random.default_rng(0)
        n = 200
        frame = {f"c{i}": rng.normal(0, 1, n) for i in range(60)}
        frame["gap"] = rng.choice([None, "x"], n, p=[0.4, 0.6])
        out = profile(pd.DataFrame(frame), seed=0).html
        section = _section(out)
        assert re.search(r"\d+ of \d+ columns are complete", section)
        assert section.count('class="miss-row"') == 1

    def test_a_single_column_frame_renders(self):
        out = profile(pd.DataFrame({"only": [None, "a", "b", None] * 50}), seed=0).html
        assert 'class="miss-row"' in _section(out)


# --------------------------------------------------------------------------- #
# the strip, ready for #139
# --------------------------------------------------------------------------- #
class TestTheStripRendererIsReady:
    """It cannot draw yet — the counts it needs are never produced (#139) — so
    it is tested directly rather than through a report."""

    def test_it_draws_one_segment_per_chunk(self):
        renderer = MissingValuesSectionRenderer()
        out = renderer._render_strip([(0, 99, 0), (100, 199, 50), (200, 299, 100)])
        assert out.count("miss-seg") == 3

    def test_each_segment_is_coloured_by_its_own_share(self):
        renderer = MissingValuesSectionRenderer()
        out = renderer._render_strip([(0, 99, 0), (100, 199, 50), (200, 299, 99)])
        assert "miss-seg good" in out
        assert "miss-seg warn" in out or "miss-seg bad" in out

    def test_it_keeps_the_attributes_that_drive_the_title(self):
        renderer = MissingValuesSectionRenderer()
        out = renderer._render_strip([(0, 99, 7)])
        for attr in (
            "data-chunk",
            "data-start",
            "data-end",
            "data-missing",
            "data-pct",
        ):
            assert attr in out, attr

    def test_hundreds_of_chunks_are_sampled_not_drawn_sub_pixel(self):
        renderer = MissingValuesSectionRenderer()
        many = [(i * 10, i * 10 + 9, i % 3) for i in range(500)]
        out = renderer._render_strip(many)
        assert out.count("miss-seg") <= renderer.MAX_STRIP_SEGMENTS

    def test_an_untracked_column_gets_no_rail_at_all(self):
        """A blank rail would read as *no missing values in any chunk*, which
        is a different claim from *not measured*."""
        renderer = MissingValuesSectionRenderer()
        out = renderer._render_strip(None)
        assert "is-untracked" in out
        assert "miss-seg" not in out


class TestTheDataGapIsRealAndDetected:
    """#139. Recorded as a test so the day the engine starts marking chunk
    boundaries, this fails and tells someone the view can be turned on."""

    def test_no_accumulator_currently_reports_more_than_one_chunk(self):
        rng = np.random.default_rng(0)
        n = 900
        values = rng.normal(0, 1, n)
        values[rng.choice(n, 200, replace=False)] = np.nan
        import pysuricata.render.missing_section as module

        seen: list[int] = []
        original = module.MissingValuesSectionRenderer.render_section

        def spy(self, kinds_map, accs, n_rows, n_cols, total_missing_cells):
            for _, (_, acc) in kinds_map.items():
                per_chunk = self._per_chunk_missing(acc)
                seen.append(len(per_chunk) if per_chunk else 0)
            return original(self, kinds_map, accs, n_rows, n_cols, total_missing_cells)

        module.MissingValuesSectionRenderer.render_section = spy
        try:
            profile(pd.DataFrame({"num": values}), seed=0, chunk_size=150)
        finally:
            module.MissingValuesSectionRenderer.render_section = original

        assert seen, "the section was never rendered"
        assert max(seen) <= 1, (
            "an accumulator now reports per-chunk missing counts -- #139 may be "
            f"fixed, and the by-chunk view can be enabled: {seen}"
        )
