"""The per-chunk missing spectrum, on every severity branch.

These blocks return early unless the stats carry `chunk_metadata`, and #139
established that per-column per-chunk counts are never produced — so in a real
report the spectrum never draws and its four severity branches never run.

That is why four emoji survived a sweep that believed it was complete: they sat
on `severity_icon` assignments reachable only through a code path no fixture
takes. The boolean card wrote them as HTML entities on top of that, so neither
a rendered-report check nor a grep for the glyph could see them.

So the renderers are driven directly here, the way `test_missing_section_views.py`
drives the strip renderer for the same reason. It covers the branches, it proves
the removal held on all of them, and it is ready for the day #139 lands.
"""

from __future__ import annotations

import re

import pytest

from pysuricata.render.boolean_card import BooleanCardRenderer
from pysuricata.render.categorical_card import CategoricalCardRenderer
from pysuricata.render.datetime_card import DateTimeCardRenderer
from pysuricata.render.numeric_card import NumericCardRenderer

#: Every pictograph the render layer has reached for, plus the neighbours a
#: future edit might reach for.
GLYPHS = ["💡", "ℹ️", "📊", "📈", "📉", "⚠️", "✅", "❌", "🔍", "🚨", "⚡", "📌", "🎯"]

#: (max missing pct, expected severity word). The bands are >=50, >=20, >=5.
BANDS = [(80.0, "critical"), (30.0, "high"), (10.0, "medium"), (1.0, "low")]


def _chunks(pct: float, n: int = 4) -> list[tuple[int, int, int]]:
    """`n` chunks of 100 rows, each missing `pct` percent."""
    missing = int(round(pct))
    return [(i * 100, i * 100 + 99, missing) for i in range(n)]


class _Stats:
    """Minimal stand-in carrying only what the spectrum reads."""

    def __init__(self, pct: float, n_chunks: int = 4):
        rows = n_chunks * 100
        self.chunk_metadata = _chunks(pct, n_chunks)
        self.missing = int(rows * pct / 100)
        self.count = rows - self.missing
        # boolean
        self.true_n = self.count // 2
        self.false_n = self.count - self.true_n
        self.name = "col"
        self.unique_est = 2


def _spectrum(renderer, pct: float) -> str:
    return renderer._build_dataprep_spectrum_visualization(_Stats(pct))


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
        html = _spectrum(renderer_cls(), pct)
        assert glyph not in html

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    @pytest.mark.parametrize(("pct", "severity"), BANDS)
    def test_no_entity_encoded_pictograph_on_any_branch(
        self, renderer_cls, pct, severity
    ):
        """The boolean card wrote them as `&#128680;`. A glyph search cannot see
        that, which is how four of them survived."""
        html = _spectrum(renderer_cls(), pct)
        for match in re.finditer(r"&#(\d+);", html):
            assert int(match.group(1)) < 0x2000, match.group(0)


class TestTheSeverityStillReachesTheMarkup:
    """The glyph was removed; the meaning must not have gone with it. The class
    is what was carrying the severity all along."""

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    @pytest.mark.parametrize(("pct", "severity"), BANDS)
    def test_the_band_lands_in_a_class_or_a_word(self, renderer_cls, pct, severity):
        html = _spectrum(renderer_cls(), pct)
        assert severity in html.lower(), (
            f"{renderer_cls.__name__} at {pct}% lost its severity entirely"
        )

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_the_bands_are_distinguishable_from_each_other(self, renderer_cls):
        rendered = {sev: _spectrum(renderer_cls(), pct) for pct, sev in BANDS}
        assert len(set(rendered.values())) == len(BANDS), (
            "two severity bands render identically, so the encoding says nothing"
        )


class TestItDegradesRatherThanRaising:
    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_no_chunk_metadata_does_not_raise(self, renderer_cls):
        class Bare:
            chunk_metadata = None
            missing = 0
            count = 10
            true_n = 5
            false_n = 5
            name = "col"
            unique_est = 2

        renderer_cls()._build_dataprep_spectrum_visualization(Bare())

    @pytest.mark.parametrize("renderer_cls", RENDERERS)
    def test_a_single_chunk_renders(self, renderer_cls):
        html = renderer_cls()._build_dataprep_spectrum_visualization(_Stats(10.0, 1))
        assert isinstance(html, str)
