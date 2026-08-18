"""The per-chunk missing strip, on every severity branch.

This file used to drive `_build_dataprep_spectrum_visualization`, four
near-copies of a spectrum block that **no code path reached**. Its own
docstring said so: the blocks returned early without `chunk_metadata`, and the
suite existed because four emoji had survived a sweep by sitting on branches no
fixture takes.

#294 deleted those four renderers, along with `_generate_missing_insights`,
`_render_chunk_visualization`, `_build_simple_missing_distribution` and the 523
lines of CSS that dressed them. What actually ships is one renderer,
`CardRenderer._build_chunk_distribution_simple`, shared by all four card kinds
-- so the checks point at it now. The glyph guard is the part worth keeping and
it is stronger here, because it is finally reading markup a reader can see.
"""

from __future__ import annotations

import re

import pytest

from pysuricata.render.boolean_card import BooleanCardRenderer
from pysuricata.render.card_base import _where_the_gaps_fall
from pysuricata.render.categorical_card import CategoricalCardRenderer
from pysuricata.render.datetime_card import DateTimeCardRenderer
from pysuricata.render.numeric_card import NumericCardRenderer

#: Every pictograph the render layer has reached for, plus the neighbours a
#: future edit might reach for.
GLYPHS = ["💡", "ℹ️", "📊", "📈", "📉", "⚠️", "✅", "❌", "🔍", "🚨", "⚡", "📌", "🎯"]

#: (max missing pct, expected severity word). The strip's bands are >20, 5-20,
#: <=5, which the segment classes carry as high / medium / low.
BANDS = [(80.0, "high"), (30.0, "high"), (10.0, "medium"), (1.0, "low")]


def _chunks(pct: float, n: int = 4) -> list[tuple[int, int, int]]:
    """`n` chunks of 100 rows, each missing `pct` percent."""
    missing = int(round(pct))
    return [(i * 100, i * 100 + 99, missing) for i in range(n)]


class _Stats:
    """Minimal stand-in carrying only what the strip reads."""

    def __init__(self, pct: float, n_chunks: int = 4):
        rows = n_chunks * 100
        self.chunk_metadata = _chunks(pct, n_chunks)
        self.missing = int(rows * pct / 100)
        self.count = rows - self.missing
        self.total = rows


def _strip(renderer, pct: float, n_chunks: int = 4) -> str:
    stats = _Stats(pct, n_chunks)
    return renderer._build_chunk_distribution_simple(stats, stats.total)


RENDERERS = [
    pytest.param(BooleanCardRenderer, id="boolean"),
    pytest.param(CategoricalCardRenderer, id="categorical"),
    pytest.param(DateTimeCardRenderer, id="datetime"),
    pytest.param(NumericCardRenderer, id="numeric"),
]


class TestEverySeverityBranchDrawsWithoutAGlyph:
    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    @pytest.mark.parametrize(("pct", "severity"), BANDS)
    @pytest.mark.parametrize("glyph", GLYPHS)
    def test_no_pictograph_on_any_branch(self, renderer_cls, pct, severity, glyph):
        assert glyph not in _strip(renderer_cls(), pct)

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    @pytest.mark.parametrize(("pct", "severity"), BANDS)
    def test_no_entity_encoded_pictograph_on_any_branch(
        self, renderer_cls, pct, severity
    ):
        """The boolean card wrote them as `&#128680;`. A glyph search cannot see
        that, which is how four of them survived."""
        for match in re.finditer(r"&#(\d+);", _strip(renderer_cls(), pct)):
            assert int(match.group(1)) < 0x2000, match.group(0)


class TestTheSeverityStillReachesTheMarkup:
    """The glyph was removed; the meaning must not have gone with it. The class
    is what was carrying the severity all along."""

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    @pytest.mark.parametrize(("pct", "severity"), BANDS)
    def test_the_band_lands_in_a_class(self, renderer_cls, pct, severity):
        html = _strip(renderer_cls(), pct)

        assert f"chunk-segment {severity}" in html, (
            f"{renderer_cls.__name__} at {pct}% lost its severity entirely"
        )


class TestThePaneCarriesNoLegendAndNoHoverInstruction:
    """#294. Both belong once, in the Missing values *section* -- not repeated
    on every card that happens to have a gap."""

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_no_legend_inside_a_per_column_pane(self, renderer_cls):
        assert "chunk-legend" not in _strip(renderer_cls(), 30.0)

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_no_instruction_a_phone_cannot_follow(self, renderer_cls):
        assert "Hover" not in _strip(renderer_cls(), 30.0)

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_the_heading_is_not_title_case(self, renderer_cls):
        assert "Missing values per chunk" in _strip(renderer_cls(), 30.0)


class TestThePaneLeadsWithWhereTheGapsFall:
    """The strip shows where the missing values fall; the sentence says it, so
    the finding survives a phone, a PDF and a reader who does not hover."""

    def test_a_tail_heavy_column_names_the_tail(self):
        # Four chunks of 100: the last one holds 90 of the 120 missing.
        chunks = [(0, 99, 10), (100, 199, 10), (200, 299, 10), (300, 399, 90)]

        assert _where_the_gaps_fall(chunks) == (
            "The last chunk holds 75% of the 120 missing values."
        )

    def test_a_front_heavy_column_names_the_front(self):
        chunks = [(0, 99, 60), (100, 199, 40), (200, 299, 5), (300, 399, 5)]

        assert _where_the_gaps_fall(chunks) == (
            "The first chunk holds 55% of the 110 missing values."
        )

    def test_two_leading_chunks_are_named_together(self):
        chunks = [(0, 99, 50), (100, 199, 50), (200, 299, 1), (300, 399, 1)]

        assert _where_the_gaps_fall(chunks) == (
            "The first 2 chunks hold 98% of the 102 missing values."
        )

    def test_an_even_spread_is_not_dressed_up_as_a_finding(self):
        chunks = [(i * 100, i * 100 + 99, 10) for i in range(4)]

        assert _where_the_gaps_fall(chunks) == (
            "The 40 missing values are spread across all 4 chunks."
        )

    def test_a_column_with_no_gaps_says_nothing(self):
        assert _where_the_gaps_fall([(0, 99, 0), (100, 199, 0)]) == ""

    def test_a_short_trailing_chunk_is_measured_against_its_rows(self):
        """A file's last chunk is usually short. Two chunks holding 10,000
        gaps each is an even split by chunk and a threefold concentration by
        data, and the second reading is the true one."""
        chunks = [(0, 49_999, 10_000), (50_000, 59_999, 10_000)]

        assert _where_the_gaps_fall(chunks) == (
            "The last chunk holds 50% of the 20,000 missing values."
        )

    def test_a_middle_chunk_is_named_by_its_position(self):
        chunks = [(0, 99, 1), (100, 199, 80), (200, 299, 1), (300, 399, 1)]

        assert _where_the_gaps_fall(chunks) == (
            "Chunk 2 of 4 holds 96% of the 83 missing values."
        )

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_the_sentence_leads_the_pane(self, renderer_cls):
        html = _strip(renderer_cls(), 30.0)

        assert html.index("chunk-finding") < html.index("chunk-spectrum")


class TestItDegradesRatherThanRaising:
    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_no_chunk_metadata_does_not_raise(self, renderer_cls):
        class Bare:
            chunk_metadata = None

        assert renderer_cls()._build_chunk_distribution_simple(Bare(), 10) == ""

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_a_single_chunk_renders(self, renderer_cls):
        assert isinstance(_strip(renderer_cls(), 10.0, n_chunks=1), str)
