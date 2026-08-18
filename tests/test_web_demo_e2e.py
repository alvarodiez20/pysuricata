"""`web/e2e.py`'s blank-frame detector (#1).

The full check needs a real Pyodide boot against real PyPI and jsDelivr, which
is exactly the kind of network-dependent, minutes-long run
`tests/test_web_demo_layout.py` deliberately keeps out of the ordinary suite.
This file covers the one piece of it that is pure logic: telling a rendered
report apart from a frame that painted nothing. That is the failure mode a DOM
check cannot see -- Chrome has silently dropped a large `srcdoc` before with no
error, no console warning and a structurally intact page -- so it has to be
right on ordinary images, not just exercised end-to-end once in CI.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class TestInkStats:
    def test_a_solid_fill_reads_as_blank(self):
        Image = pytest.importorskip("PIL.Image")
        from e2e import MIN_DISTINCT_COLOURS, MIN_INK_FRACTION, _ink_stats

        blank = Image.new("RGB", (800, 600), (255, 255, 255))
        fraction, colours = _ink_stats(_png(blank))

        assert fraction == 0.0
        assert colours == 1
        assert fraction < MIN_INK_FRACTION
        assert colours < MIN_DISTINCT_COLOURS

    def test_a_dark_theme_solid_fill_also_reads_as_blank(self):
        """The background is read off the image itself, not assumed white --
        a dropped frame in dark theme must not slip through as "has ink"."""
        Image = pytest.importorskip("PIL.Image")
        from e2e import MIN_DISTINCT_COLOURS, MIN_INK_FRACTION, _ink_stats

        blank = Image.new("RGB", (800, 600), (18, 18, 22))
        fraction, colours = _ink_stats(_png(blank))

        assert fraction == 0.0
        assert colours == 1
        assert fraction < MIN_INK_FRACTION
        assert colours < MIN_DISTINCT_COLOURS

    def test_content_on_a_background_clears_both_bars(self):
        Image = pytest.importorskip("PIL.Image")
        ImageDraw = pytest.importorskip("PIL.ImageDraw")
        from e2e import MIN_DISTINCT_COLOURS, MIN_INK_FRACTION, _ink_stats

        img = Image.new("RGB", (800, 600), (250, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 380, 200], fill=(240, 240, 245), outline=(30, 30, 40))
        draw.rectangle([70, 100, 90, 180], fill=(220, 90, 90))
        draw.ellipse([420, 20, 600, 200], fill=(255, 200, 40))
        # A histogram-like run of bars, each a different colour -- what a real
        # report's charts contribute and the two flat rectangles above do not.
        for i in range(40):
            shade = 60 + i * 4
            draw.rectangle(
                [40 + i * 8, 220 - i, 44 + i * 8, 220], fill=(shade, 140, 220 - i)
            )

        fraction, colours = _ink_stats(_png(img))

        assert fraction >= MIN_INK_FRACTION
        assert colours >= MIN_DISTINCT_COLOURS

    def test_one_flat_block_is_not_enough_content(self):
        """A single coloured block can clear the ink-fraction bar on its own
        but must still fail on distinct colours -- a report is many colours
        over a real area, not a background plus one wrong-coloured rectangle."""
        Image = pytest.importorskip("PIL.Image")
        from e2e import MIN_DISTINCT_COLOURS, MIN_INK_FRACTION, _ink_stats

        img = Image.new("RGB", (10, 10), (255, 255, 255))
        for x in range(2, 7):
            for y in range(2, 7):
                img.putpixel((x, y), (255, 0, 0))
        fraction, colours = _ink_stats(_png(img))

        assert fraction == pytest.approx(25 / 100)
        assert fraction >= MIN_INK_FRACTION
        assert colours == 2
        assert colours < MIN_DISTINCT_COLOURS
