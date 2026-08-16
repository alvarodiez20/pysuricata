"""The logo is an asset the report inlines, so it is part of the contract.

Forty-seven percent of a 1.23 MB report was two base64 PNGs of the logo -- more
bytes than the data, the CSS, the JavaScript and the markup put together, to
draw a mark 30 CSS pixels tall. They were two because the artwork had the
wordmark baked in, and the wordmark needed a different colour in dark mode.

These tests cover three separate things, because they can each break alone:

1. the tracer, which turns the artwork into paths;
2. the committed SVG, which must still correspond to the artwork it came from;
3. the report, which must inline it and must not reacquire a raster payload.

The third is the one that regresses silently. Nothing fails when an image gets
big, so the size guard here is the only thing that would notice.
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.render.html import _build_logo
from scripts.trace_logo import (
    MIN_AREA,
    TOLERANCE,
    _area,
    build_svg,
    decode_png,
    path_data,
    simplify,
    trace,
)

REPO = Path(__file__).resolve().parent.parent
SOURCE_PNG = REPO / "assets" / "logo_mark.png"
SHIPPED_SVG = REPO / "pysuricata" / "static" / "images" / "logo_mark.svg"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_png(path: Path, rgba: np.ndarray) -> None:
    """Encode an (h, w, 4) array as an uncompressed-filter 8-bit RGBA PNG."""
    height, width = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].astype(np.uint8).tobytes() for y in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _polygons(d: str) -> list[list[tuple[float, float]]]:
    out = []
    for sub in d.split("Z"):
        pts = [
            (float(x), float(y))
            for x, y in re.findall(r"[ML](-?[\d.]+) (-?[\d.]+)", sub)
        ]
        if len(pts) >= 3:
            out.append(pts)
    return out


def _rasterise(polys, width: int, height: int) -> np.ndarray:
    """Even-odd scanline fill, sampling at pixel centres."""
    edges = []
    for poly in polys:
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
    mask = np.zeros((height, width), dtype=bool)
    arr = np.array(edges)
    x0, y0, x1, y1 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for row in range(height):
        yc = row + 0.5
        hit = ((y0 <= yc) & (y1 > yc)) | ((y1 <= yc) & (y0 > yc))
        if not hit.any():
            continue
        t = (yc - y0[hit]) / (y1[hit] - y0[hit])
        xs = np.sort(x0[hit] + t * (x1[hit] - x0[hit]))
        # Crossings pair up as spans; strict=False because a degenerate
        # scanline can yield an odd count and the trailing crossing is empty.
        for a, b in zip(xs[0::2], xs[1::2], strict=False):
            lo, hi = int(np.ceil(a - 0.5)), int(np.ceil(b - 0.5))
            if hi > lo:
                mask[row, max(0, lo) : min(width, hi)] = True
    return mask


@pytest.fixture(scope="module")
def report_html() -> str:
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame(
        {
            "age": rng.integers(1, 80, n).astype(float),
            "sex": rng.choice(["male", "female"], n),
            "survived": rng.integers(0, 2, n).astype(bool),
            "booked": pd.date_range("2026-01-01", periods=n, freq="h"),
        }
    )
    return profile(frame, seed=0).html


# --------------------------------------------------------------------------- #
# 1. the tracer
# --------------------------------------------------------------------------- #
class TestTheTracer:
    def test_a_rectangle_traces_to_its_own_outline(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8:22] = True
        loops = trace(mask)
        assert len(loops) == 1
        assert _area(loops[0]) == pytest.approx(10 * 14)

    def test_a_hole_comes_out_as_its_own_contour(self):
        """Two contours, not one, so ``fill-rule="evenodd"`` can leave the
        middle empty. This is what puts the eyes in the meerkat."""
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:25, 5:25] = True
        mask[12:18, 12:18] = False
        loops = trace(mask)
        assert len(loops) == 2
        assert sorted(round(_area(loop)) for loop in loops) == [36, 400]

    def test_disjoint_regions_each_get_a_contour(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[4:10, 4:10] = True
        mask[4:10, 24:30] = True
        assert len(trace(mask)) == 2

    def test_an_empty_mask_traces_to_nothing(self):
        assert trace(np.zeros((10, 10), dtype=bool)) == []

    def test_a_full_mask_traces_to_one_contour(self):
        """No padding bug: a region touching every edge still closes."""
        loops = trace(np.ones((10, 10), dtype=bool))
        assert len(loops) == 1
        assert _area(loops[0]) == pytest.approx(100)

    def test_a_single_pixel_is_traced_not_skipped(self):
        """``trace`` reports it; dropping it is ``MIN_AREA``'s decision, made
        one layer up, where it can be argued about."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        assert len(trace(mask)) == 1

    def test_simplify_never_moves_a_point_beyond_the_tolerance(self):
        """The guarantee the whole size argument rests on."""
        rng = np.random.default_rng(0)
        angles = np.linspace(0, 2 * np.pi, 400)
        pts = [
            (float(200 + 100 * np.cos(a)), float(200 + 100 * np.sin(a))) for a in angles
        ]
        for eps in (0.5, 1.0, 2.0):
            kept = simplify(pts, eps)
            assert len(kept) < len(pts)
            original = np.asarray(pts)
            poly = np.asarray(kept)
            for point in original:
                seg_a, seg_b = poly[:-1], poly[1:]
                seg = seg_b - seg_a
                rel = point - seg_a
                length = np.hypot(seg[:, 0], seg[:, 1])
                t = np.clip(
                    (rel[:, 0] * seg[:, 0] + rel[:, 1] * seg[:, 1])
                    / np.where(length == 0, 1, length**2),
                    0,
                    1,
                )
                closest = seg_a + t[:, None] * seg
                assert np.hypot(*(point - closest).T).min() <= eps + 1e-9

    def test_a_larger_tolerance_never_yields_more_points(self):
        rng = np.random.default_rng(1)
        pts = [(float(x), float(y)) for x, y in rng.integers(0, 200, (300, 2))]
        sizes = [len(simplify(pts, eps)) for eps in (0.25, 1.0, 4.0, 16.0)]
        assert sizes == sorted(sizes, reverse=True)

    def test_simplify_keeps_degenerate_input_intact(self):
        assert len(simplify([(0.0, 0.0), (1.0, 1.0)], 1.0)) == 2

    def test_contours_below_min_area_are_dropped_from_the_path(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:35, 5:35] = True
        mask[20, 20] = False  # a one-pixel pinhole, below MIN_AREA
        assert MIN_AREA > 1
        assert len(_polygons(path_data(mask))) == 1


class TestPngDecoding:
    def test_a_round_trip_through_every_filter_type(self, tmp_path):
        rng = np.random.default_rng(0)
        original = rng.integers(0, 256, (17, 23, 4), dtype=np.uint8)
        path = tmp_path / "x.png"
        _write_png(path, original)
        np.testing.assert_array_equal(decode_png(path), original)

    def test_a_real_encoder_using_adaptive_filters_decodes(self, tmp_path):
        """`_write_png` only emits filter 0. zlib's own output is not the
        concern -- adaptive per-scanline filters are, and only real encoders
        produce them, so the shipped artwork is the test case."""
        img = decode_png(SOURCE_PNG)
        assert img.shape == (620, 410, 4)
        # Very nearly binary: 51 pixels of 254,200 sit at alpha 76, and every
        # one of them falls below the opacity cut. This is worth pinning
        # because MIN_AREA is chosen on the strength of it -- if the artwork
        # were ever re-exported with soft edges, the dust that setting drops
        # would stop being dust and start being the antialiased boundary.
        alpha = img[:, :, 3]
        assert ((alpha == 0) | (alpha == 255)).mean() > 0.999

    def test_a_non_png_is_rejected_clearly(self, tmp_path):
        path = tmp_path / "not.png"
        path.write_bytes(b"GIF89a")
        with pytest.raises(ValueError, match="not a PNG"):
            decode_png(path)

    def test_an_unsupported_png_flavour_is_rejected_clearly(self, tmp_path):
        """Greyscale rather than RGBA. Better a named error than a wrong trace."""
        path = tmp_path / "grey.png"

        def chunk(tag, body):
            return (
                struct.pack(">I", len(body))
                + tag
                + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
            )

        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00\x00"))
            + chunk(b"IEND", b"")
        )
        with pytest.raises(ValueError, match="8-bit RGBA"):
            decode_png(path)


# --------------------------------------------------------------------------- #
# 2. the committed asset
# --------------------------------------------------------------------------- #
class TestTheShippedAsset:
    def test_it_exists_and_ships(self):
        assert SHIPPED_SVG.is_file()

    def test_it_still_matches_the_artwork_it_was_traced_from(self):
        """Regenerating reproduces the committed file exactly.

        This is the check that catches the asset and its source drifting apart
        -- someone editing the SVG by hand, or updating the artwork and
        forgetting to re-run the tracer.
        """
        committed = SHIPPED_SVG.read_text(encoding="utf-8")
        assert build_svg(SOURCE_PNG, TOLERANCE) == committed.rstrip("\n")

    def test_the_tracer_is_deterministic(self):
        assert build_svg(SOURCE_PNG, TOLERANCE) == build_svg(SOURCE_PNG, TOLERANCE)

    def test_it_depicts_the_artwork(self):
        """Rasterise the paths and compare against the source.

        Structure and determinism would both pass on a confidently wrong trace,
        so this renders the thing and counts pixels. The tolerance below is
        generous on purpose: the layers are nested, so their boundaries are long,
        and a sub-pixel band along a 2,700-pixel perimeter is the expected cost
        of simplification, not a defect.
        """
        svg = SHIPPED_SVG.read_text(encoding="utf-8")
        width, height = (
            int(v) for v in re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups()
        )
        img = decode_png(SOURCE_PNG)
        opaque = img[:, :, 3] > 127
        rgb = img[:, :, :3].astype(np.float64)
        lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        expected = [opaque, opaque & (lum >= 40.0), opaque & (lum >= 140.0)]

        paths = re.findall(r'd="([^"]+)"', svg)
        assert len(paths) == len(expected)
        for reference, d in zip(expected, paths, strict=True):
            drawn = _rasterise(_polygons(d), width, height)
            disagreement = (drawn ^ reference).sum() / reference.sum()
            assert disagreement < 0.10
            # And it must be the right shape, not merely the right area.
            overlap = (drawn & reference).sum() / (drawn | reference).sum()
            assert overlap > 0.92

    def test_it_scales(self):
        """A viewBox and no intrinsic width/height, so CSS `height` governs."""
        svg = SHIPPED_SVG.read_text(encoding="utf-8")
        assert "viewBox=" in svg
        assert not re.search(r"<svg[^>]*\swidth=", svg)
        assert not re.search(r"<svg[^>]*\sheight=", svg)

    def test_it_is_self_contained(self):
        """The report is a single file. An asset that fetches anything, or that
        carries script, would break that promise and inline straight into every
        report."""
        svg = SHIPPED_SVG.read_text(encoding="utf-8")
        for forbidden in (
            "<script",
            "<image",
            "xlink:href",
            "href=",
            "url(",
            "@import",
        ):
            assert forbidden not in svg, forbidden

    def test_it_is_smaller_than_the_raster_by_an_order_of_magnitude(self):
        assert SHIPPED_SVG.stat().st_size < SOURCE_PNG.stat().st_size / 10

    def test_it_does_not_announce_itself_twice(self):
        """The name is set as type beside the mark, so the mark is decorative."""
        svg = SHIPPED_SVG.read_text(encoding="utf-8")
        assert 'aria-hidden="true"' in svg
        assert "aria-label" not in svg


# --------------------------------------------------------------------------- #
# 3. the report
# --------------------------------------------------------------------------- #
class TestTheReportEmbedsIt:
    def test_the_mark_is_inline(self, report_html):
        assert 'class="logo-mark"' in report_html
        assert "<svg" in report_html

    def test_the_name_is_type_not_art(self, report_html):
        assert '<span class="wordmark">pysuricata</span>' in report_html

    def test_there_is_no_raster_logo_left(self, report_html):
        assert "data:image/png;base64," not in report_html

    def test_the_only_remaining_binary_payload_is_the_favicon(self, report_html):
        payloads = re.findall(r"data:([^;]+);base64,([A-Za-z0-9+/=]+)", report_html)
        assert [kind for kind, _ in payloads] == ["image/x-icon"]

    def test_the_dark_mode_duplicate_is_gone(self, report_html):
        """Two images and a CSS swap existed only to recolour a baked-in
        wordmark. Text follows currentColor, so there is nothing to swap."""
        assert "logo-light" not in report_html
        assert "logo-dark" not in report_html

    def test_the_embedded_payload_stays_small(self, report_html):
        """The guard on the thing that regressed silently for a year.

        Nothing fails when an inlined image gets big; the report just gets
        heavier every release until somebody measures it. 64 KB is well above
        the favicon and far below a mistake.
        """
        payloads = re.findall(r"data:[^;]+;base64,([A-Za-z0-9+/=]+)", report_html)
        assert sum(len(p) for p in payloads) < 64_000

    def test_a_missing_asset_does_not_fail_the_run(self, tmp_path):
        """A logo is not worth failing a profile over. The name survives."""
        html = _build_logo(str(tmp_path / "absent.svg"))
        assert "pysuricata" in html
        assert "<svg" not in html

    def test_the_report_still_renders_without_the_asset(self, monkeypatch, tmp_path):
        import pysuricata.render.html as module

        monkeypatch.setattr(
            module, "_build_logo", lambda _path: '<span id="logo">pysuricata</span>'
        )
        html = profile(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), seed=0).html
        assert "<html" in html
        assert "pysuricata" in html
