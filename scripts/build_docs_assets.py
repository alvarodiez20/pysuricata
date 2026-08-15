"""Generate documentation visuals from the live library.

A hand-drawn diagram is a claim nobody checks. Every asset here is replayed from
a real run of the real code with a fixed seed, so when an algorithm changes the
picture changes with it — and CI can fail if the committed asset no longer
matches, the same trick as a snapshot test.

    python scripts/build_docs_assets.py              # write docs/assets/generated/
    python scripts/build_docs_assets.py --check      # fail if anything drifted

Output is animated SVG, not GIF: a few KB instead of megabytes, scales, inherits
the page theme through CSS custom properties so dark mode works for free, and
honours prefers-reduced-motion. Each animation also renders a meaningful still
first frame, so print and screen readers lose nothing.

Currently generates one asset as a working reference; the pattern extends to the
rest of the list in docs/DOCS_PLAN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "assets" / "generated"


# ---------------------------------------------------------------------------
# KMV on the unit interval
# ---------------------------------------------------------------------------


def _kmv_trace(n_distinct: int = 4000, k: int = 32, seed: int = 7):
    """Replay a real KMV sketch and record the state after each batch.

    Uses the library's own hash so the marks on the line are the actual hashes
    the sketch stores, not a plausible-looking substitute.
    """
    from pysuricata.accumulators.sketches import KMV

    rng = np.random.default_rng(seed)
    values = rng.permutation(n_distinct).astype(float)

    sketch = KMV(k)
    frames = []
    step = max(1, n_distinct // 24)
    for i in range(step, n_distinct + 1, step):
        batch = values[i - step : i]
        sketch.add_many(np.ascontiguousarray(batch))
        stored = sorted(int(h) for h in sketch._values)
        unit = [h / 2**64 for h in stored]
        threshold = unit[-1] if len(stored) >= k else 1.0
        frames.append(
            {
                "seen": i,
                "marks": unit,
                "threshold": threshold,
                "estimate": float(sketch.estimate()),
                "exact": len(stored) < k,
            }
        )
    return frames, k, n_distinct


def kmv_unit_interval_svg() -> str:
    """Animated KMV, on a window that zooms as the sketch tightens.

    A fixed [0,1) axis is the honest picture and an unreadable one: with k=32 and
    a few thousand distinct values, every stored hash sits inside the leftmost
    1% and the marks collapse into a blob. So the view tracks the threshold —
    the axis always spans [0, 1.6t] — which keeps the marks spread out and turns
    the shrinking of t into the thing the eye actually follows: the window
    zooming in while the right-hand axis label counts down.
    """
    frames, k, truth = _kmv_trace()
    W, H = 640, 226
    X0, X1 = 62, W - 62
    Y = 112
    dur = len(frames) * 0.5

    def view_max(f) -> float:
        return max(1.6 * f["threshold"], 1e-4)

    def x_in(u: float, vm: float) -> float:
        return X0 + min(u / vm, 1.0) * (X1 - X0)

    def fmt(v: float) -> str:
        return f"{v:.3f}" if v >= 0.01 else f"{v:.1e}"

    P: list[str] = []
    P.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="A K-Minimum-Values sketch filling up. Hash values land on the unit '
        f"interval; only the {k} smallest are kept. As distinct values arrive the kth "
        f"smallest hash t shrinks, the view zooms in to follow it, and the estimate "
        f'(k-1)/t converges on the true count of {truth:,}.">'
    )
    P.append(
        "<style>"
        ".ax{stroke:var(--md-default-fg-color--lighter,#b6b5ae);stroke-width:1.5}"
        ".tk{stroke:var(--md-default-fg-color--lighter,#b6b5ae);stroke-width:1}"
        ".lbl{font:11px system-ui,-apple-system,sans-serif;"
        "fill:var(--md-default-fg-color--light,#52514e)}"
        ".big{font:600 13px system-ui,-apple-system,sans-serif;"
        "fill:var(--md-default-fg-color,#0b0b0b)}"
        ".mark{fill:var(--md-primary-fg-color,#2a78d6)}"
        ".thr{stroke:#eb6834;stroke-width:2}"
        ".thrl{font:600 11px system-ui,sans-serif;fill:#eb6834}"
        "@media (prefers-reduced-motion: reduce){animate{display:none}}"
        "</style>"
    )

    def anim(attr, vals):
        return (
            f'<animate attributeName="{attr}" values="{";".join(vals)}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
        )

    P.append(f'<line class="ax" x1="{X0}" y1="{Y}" x2="{X1}" y2="{Y}"/>')
    P.append(f'<line class="tk" x1="{X0}" y1="{Y}" x2="{X0}" y2="{Y + 6}"/>')
    P.append(f'<text class="lbl" x="{X0}" y="{Y + 20}" text-anchor="middle">0</text>')
    P.append(f'<line class="tk" x1="{X1}" y1="{Y}" x2="{X1}" y2="{Y + 6}"/>')
    # Right-hand label is the current window, and it shrinks.
    P.append(
        f'<text class="lbl" x="{X1}" y="{Y + 20}" text-anchor="middle">'
        f"{fmt(view_max(frames[0]))}{anim('textContent', [fmt(view_max(f)) for f in frames])}</text>"
    )
    P.append(
        f'<text class="lbl" x="{(X0 + X1) / 2:.0f}" y="{Y + 40}" text-anchor="middle">'
        "hash value &#183; the view zooms to follow the threshold</text>"
    )

    for slot in range(k):
        xs, ops = [], []
        for f in frames:
            vm = view_max(f)
            if slot < len(f["marks"]) and f["marks"][slot] <= vm:
                xs.append(f"{x_in(f['marks'][slot], vm):.1f}")
                ops.append("1")
            else:
                xs.append(f"{X0:.1f}")
                ops.append("0")
        P.append(
            f'<circle class="mark" cy="{Y}" r="4" cx="{xs[0]}" opacity="{ops[0]}">'
            f"{anim('cx', xs)}{anim('opacity', ops)}</circle>"
        )

    # The threshold sits at a fixed fraction of the window by construction, so it
    # reads as a stable gate while everything else moves through it.
    tx = f"{X0 + (1 / 1.6) * (X1 - X0):.1f}"
    P.append(f'<line class="thr" x1="{tx}" x2="{tx}" y1="{Y - 30}" y2="{Y + 12}"/>')
    P.append(
        f'<text class="thrl" x="{tx}" y="{Y - 36}" text-anchor="middle">t = kth smallest</text>'
    )
    P.append(
        f'<text class="lbl" x="{float(tx) + 8:.0f}" y="{Y - 14}">'
        "nothing to the right can enter</text>"
    )

    seen = [f"{f['seen']:,} distinct values seen" for f in frames]
    est = [
        ("exact count: " if f["exact"] else "estimate (k&#8722;1)/t = ")
        + f"{f['estimate']:,.0f}"
        for f in frames
    ]
    P.append(
        f'<text class="lbl" x="{X0}" y="28">{seen[0]}{anim("textContent", seen)}</text>'
    )
    P.append(
        f'<text class="big" x="{X0}" y="50">{est[0]}{anim("textContent", est)}</text>'
    )
    P.append(
        f'<text class="lbl" x="{X1}" y="50" text-anchor="end">true count {truth:,} &#183; k = {k}</text>'
    )
    P.append(
        f'<text class="lbl" x="{X0}" y="{H - 22}">Memory is fixed at {k} hashes, however long the '
        "stream runs.</text>"
    )
    P.append(
        f'<text class="lbl" x="{X0}" y="{H - 6}">Duplicates land on a mark that already exists, '
        "and change nothing.</text>"
    )
    P.append("</svg>")
    return "".join(P)


# ---------------------------------------------------------------------------

ASSETS = {
    "kmv-unit-interval.svg": kmv_unit_interval_svg,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="fail if an asset would change"
    )
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    drifted: list[str] = []

    for name, fn in ASSETS.items():
        # Trailing newline so end-of-file-fixer and --check agree: without it
        # the pre-commit hook rewrites every generated file and --check then
        # reports drift against the generator that produced it.
        content = fn().rstrip("\n") + "\n"
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        manifest[name] = digest
        path = OUT / name
        if args.check:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing != content:
                drifted.append(name)
            continue
        path.write_text(content, encoding="utf-8")
        print(f"  wrote {path.relative_to(REPO)}  ({len(content):,} bytes, {digest})")

    mpath = OUT / "MANIFEST.json"
    if args.check:
        if drifted:
            print("These assets no longer match the code that generates them:")
            for d in drifted:
                print(f"  {d}")
            print("\nRun: python scripts/build_docs_assets.py")
            return 1
        print(f"{len(ASSETS)} asset(s) up to date")
        return 0

    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {mpath.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
