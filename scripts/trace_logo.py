"""Trace the flat-colour logo artwork into a compact SVG.

The report embeds its logo, and it embedded it as base64 PNG: 578 KB of a
1.23 MB report, 47% of the document, for a mark that renders at 30 CSS pixels.
Two copies, because the artwork carried a baked-in wordmark that needed a
different colour in dark mode.

This produces the vector the report should have been shipping. The artwork is
flat three-tone with hard edges -- exactly the case where tracing is lossless in
every way a reader can perceive, and where the vector is two orders of magnitude
smaller than the raster.

    uv run python scripts/trace_logo.py

It is a one-time asset tool, not part of the library. It is committed so the
mark can be regenerated if the artwork changes, and so the tolerance below is a
recorded decision rather than a lost one.

Why not a tracing library
-------------------------
`potrace` and friends fit Bezier curves, which is the right tool for scanned or
antialiased art. This artwork is flat colour with hard edges, so the boundaries
are *already* exact polygons; the only question is how much to simplify them.
Fitting curves to them would add a dependency and an approximation to something
that needs neither. Everything here is numpy plus the standard library.

How it works
------------
1. Decode the PNG (stdlib `zlib`, no imaging dependency).
2. Split the opaque pixels into the three tones by luminance.
3. Emit three *nested* masks rather than three disjoint ones:

       layer 1   every opaque pixel          filled black
       layer 2   brown and cream together    filled brown
       layer 3   cream alone                 filled cream

   Each mask is strictly contained in the one before it, so every layer is a
   simple region painted over the last. Tracing the tones disjointly instead
   would mean tracing the black keyline as a thin ring around the entire
   silhouette -- twice the contour length, for an identical picture.
4. Follow the cracks between inside and outside pixels to get exact closed
   polygons, then simplify with Ramer-Douglas-Peucker.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

# Simplification tolerance, in source pixels: no traced point moves further
# than this from the true boundary. The artwork is 410 x 620 and the mark
# renders at 30 px tall, so one source pixel is 1/20 of a rendered pixel.
#
# Measured, by rasterising the emitted paths back and comparing to the source:
#
#     eps    bytes   max deviation from the true boundary
#     0.50   45,373  1 px
#     0.75   23,964  1 px
#     1.00   10,794  2 px
#     1.50    5,228  2 px
#     2.00    3,915  3 px
#
# Every one of those is sub-pixel at the size the report draws. 1.0 is the
# conservative pick rather than the cheapest: the extra 5 KB over 1.5 is nothing
# against the 567 KB this removes, and it leaves headroom if the mark is ever
# used larger than the header bar.
TOLERANCE = 1.0

# Contours smaller than this are dust, not features. The source alpha channel is
# very nearly binary -- 125,857 pixels at 0, 128,292 at 255, and 51 at 76, all
# of which fall below the opacity cut -- so this is not antialiasing to be
# smoothed: it is 61 stray one- and two-pixel holes inside the body, 64 pixels
# in total, against a main contour of 128,356. Dropping them makes the vector
# marginally cleaner than the raster it came from.
MIN_AREA = 12.0

# Tone boundaries by luminance. The artwork clusters hard at 10, 65 and 245, so
# anything between the clusters is an antialiased edge pixel and the exact cut
# point does not matter.
_BLACK_MAX = 40.0
_BROWN_MAX = 140.0

INK = "#0B0906"
BROWN = "#412E17"
CREAM = "#F5E2C1"


def decode_png(path: Path) -> np.ndarray:
    """Decode an 8-bit RGBA non-interlaced PNG to an ``(h, w, 4)`` array."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")

    idat = bytearray()
    off = 0 + 8
    width = height = 0
    while off < len(data):
        (length,) = struct.unpack(">I", data[off : off + 4])
        typ = data[off + 4 : off + 8]
        body = data[off + 8 : off + 8 + length]
        if typ == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if (depth, ctype, interlace) != (8, 6, 0):
                raise ValueError(
                    f"{path}: expected 8-bit RGBA non-interlaced, got "
                    f"depth={depth} colour-type={ctype} interlace={interlace}"
                )
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
        off += 12 + length

    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        cur = np.frombuffer(raw[pos : pos + stride], dtype=np.uint8).copy()
        pos += stride
        if ftype == 1:  # Sub
            for i in range(4, stride):
                cur[i] = (cur[i] + cur[i - 4]) & 0xFF
        elif ftype == 2:  # Up
            cur = ((cur.astype(np.uint16) + prev) % 256).astype(np.uint8)
        elif ftype == 3:  # Average
            for i in range(stride):
                left = int(cur[i - 4]) if i >= 4 else 0
                cur[i] = (cur[i] + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = int(cur[i - 4]) if i >= 4 else 0
                b = int(prev[i])
                c = int(prev[i - 4]) if i >= 4 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pred) & 0xFF
        elif ftype != 0:
            raise ValueError(f"{path}: unknown PNG filter type {ftype}")
        out[y] = cur
        prev = cur
    return out.reshape(height, width, 4)


def trace(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return the closed boundary polygons of a binary mask.

    Walks the *cracks* between inside and outside pixels rather than the pixel
    centres, which makes every boundary an exact closed rectilinear polygon --
    no reconstruction, no ambiguity about whether a boundary pixel is in or out.
    Outer boundaries and holes both come out, wound in opposite directions, so
    ``fill-rule="evenodd"`` renders them correctly without further work.
    """
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    # A directed edge for each side of an inside pixel that faces outside.
    # Walking them start-to-end always closes, because every lattice point has
    # equal in- and out-degree.
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ys, xs = np.nonzero(padded)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if not padded[y - 1, x]:
            edges.setdefault((x, y), []).append((x + 1, y))
        if not padded[y, x + 1]:
            edges.setdefault((x + 1, y), []).append((x + 1, y + 1))
        if not padded[y + 1, x]:
            edges.setdefault((x + 1, y + 1), []).append((x, y + 1))
        if not padded[y, x - 1]:
            edges.setdefault((x, y + 1), []).append((x, y))

    loops: list[list[tuple[int, int]]] = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        cur = start
        while True:
            nxt_list = edges.get(cur)
            if not nxt_list:
                break
            nxt = nxt_list.pop()
            if not nxt_list:
                del edges[cur]
            loop.append(nxt)
            cur = nxt
            if cur == start:
                break
        if len(loop) > 3:
            loops.append(loop)
    return loops


def _area(poly: list[tuple[int, int]]) -> float:
    a = 0.0
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


def simplify(points: list[tuple[int, int]], eps: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker, iteratively so a long contour cannot blow the stack."""
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]
    pts = np.asarray(points, dtype=float)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = b - a
        norm = float(np.hypot(seg[0], seg[1]))
        sub = pts[i + 1 : j]
        rel = sub - a
        if norm == 0.0:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return [(float(x), float(y)) for x, y in pts[keep]]


def path_data(mask: np.ndarray, eps: float = TOLERANCE) -> str:
    """The SVG ``d`` attribute for every contour of a mask."""
    parts: list[str] = []
    for loop in trace(mask):
        if _area(loop) < MIN_AREA:
            continue
        pts = simplify(loop, eps)
        if len(pts) < 4:
            continue
        chunks = [f"M{pts[0][0]:g} {pts[0][1]:g}"]
        chunks += [f"L{x:g} {y:g}" for x, y in pts[1:-1]]
        parts.append("".join(chunks) + "Z")
    return "".join(parts)


def build_svg(png: Path, eps: float = TOLERANCE) -> str:
    img = decode_png(png)
    height, width = img.shape[:2]
    opaque = img[:, :, 3] > 127
    rgb = img[:, :, :3].astype(np.float64)
    lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

    cream = opaque & (lum >= _BROWN_MAX)
    brown_or_cream = opaque & (lum >= _BLACK_MAX)

    layers = [(opaque, INK), (brown_or_cream, BROWN), (cream, CREAM)]
    paths = "".join(
        f'<path fill="{colour}" fill-rule="evenodd" d="{path_data(mask, eps)}"/>'
        for mask, colour in layers
    )
    # aria-hidden, because everywhere this is used the mark sits next to the
    # word "pysuricata" -- as type in the report header, as an alt attribute in
    # the docs. Labelling the mark as well would make a screen reader announce
    # the product name twice.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'class="logo-mark" aria-hidden="true" focusable="false">{paths}</svg>'
    )


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=repo / "assets" / "logo_mark.png",
        help="flat-colour PNG of the mark alone",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=repo / "pysuricata" / "static" / "images" / "logo_mark.svg",
        help="where to write the SVG",
    )
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    svg = build_svg(args.source, args.tolerance)
    # Trailing newline: the repository's end-of-file-fixer hook adds one on
    # commit, and the test that asserts the committed asset still matches the
    # artwork would fail on the difference.
    args.out.write_text(svg + "\n", encoding="utf-8")
    before = args.source.stat().st_size
    print(
        f"{args.out.relative_to(repo)}  {len(svg):,} B "
        f"(source PNG {before:,} B, {before / len(svg):.0f}x smaller)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
