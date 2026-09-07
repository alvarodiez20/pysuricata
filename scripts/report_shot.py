#!/usr/bin/env python3
"""Render regions of a pysuricata report to PNG, so a notebook can show it.

    uv sync --group browser && uv run playwright install chromium
    python scripts/report_shot.py benchmarks/results/billion-report.html \
        --out-dir benchmarks/results/shots --prefix billion \
        --card score_0 --card sku_0

Needs the `browser` dependency group, which is deliberately not in `dev` -- see
pyproject.toml for why a ~300 MB Chromium does not go in every test job.

## Why this exists

A saved report is a self-contained HTML document, and a notebook cannot show
one. The obvious answer, an `<iframe srcdoc="...">`, does render in JupyterLab
and in an nbconvert HTML export -- but **GitHub's notebook viewer sanitises
output HTML and drops iframes entirely**, so on the one surface where most
people will read the notebook, the report was an empty space. Size was not the
problem; the tag is simply not on GitHub's allowlist.

`image/png` is on it. So the report is rendered in a real browser and captured
region by region, and the notebook displays the PNGs -- which render in
JupyterLab, in VS Code, in an HTML export and on GitHub alike.

The cost is honest and worth stating where the images are used: a PNG is not
the report. It does not scroll, the histogram bin toggles do not toggle, and
the sample table does not scroll sideways. It is a picture of the report, and
the report itself is one file open away.

## Regions, not one tall screenshot

The billion-row report is 7,781 px tall at 1,280 wide. Captured whole it is a
strip nobody reads and a PNG nobody wants in a diff. These are the parts that
carry the argument: the summary, the quality flags, and one column card showing
what a per-column pane actually looks like.
"""

from __future__ import annotations

import argparse
import os
import sys

#: Selector -> file suffix, captured from every report. Missing selectors are
#: skipped rather than failing: a report with no quality problems renders no
#: `#needs-attention` block at all, and that is a property of the data rather
#: than a broken capture.
REGIONS: list[tuple[str, str]] = [
    ("#summary", "summary"),
    ("#needs-attention", "flags"),
]

#: Per-column cards carry `id="col_<name>"`, so a caller can ask for the ones
#: worth showing rather than taking whichever happens to be first. That matters:
#: the first card in the billion-row report is a lognormal whose histogram is a
#: single bar against its own tail. Honest, and a poor picture of what a card
#: looks like.
CARD_ID = "col_{}"

CHROME_CANDIDATES = (
    os.environ.get("CHROMIUM_PATH", ""),
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
)


def _chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if c and os.path.exists(c):
            return c
    return None  # let Playwright find its own


def shoot(
    report: str,
    out_dir: str,
    prefix: str,
    width: int = 1280,
    cards: tuple[str, ...] = (),
) -> list[str]:
    from playwright.sync_api import sync_playwright

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    exe = _chrome()

    with sync_playwright() as p:
        browser = p.chromium.launch(**({"executable_path": exe} if exe else {}))
        page = browser.new_page(viewport={"width": width, "height": 1000})
        page.goto(f"file://{os.path.abspath(report)}", wait_until="load")
        # The report animates a couple of things in on load, and a capture
        # taken immediately catches them mid-transition -- the same trap the
        # dark-mode contact sheet fell into.
        page.wait_for_timeout(1500)

        regions = list(REGIONS)
        for name in cards:
            regions.append((f"#{CARD_ID.format(name)}", f"card-{name}"))
        if not cards:
            regions.append((".var-card", "card"))

        for selector, suffix in regions:
            el = page.query_selector(selector)
            if el is None:
                print(f"  skip {selector}: not in this report")
                continue
            el.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            path = os.path.join(out_dir, f"{prefix}-{suffix}.png")
            el.screenshot(path=path)
            size = os.path.getsize(path)
            print(f"  {selector:18s} -> {path}  ({size / 1024:,.0f} KB)")
            written.append(path)

        browser.close()
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="path to a saved pysuricata HTML report")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument(
        "--card",
        action="append",
        default=[],
        metavar="COLUMN",
        help="capture this column's card by name; repeatable. "
        "Defaults to whichever card comes first.",
    )
    args = ap.parse_args(argv)

    if not os.path.exists(args.report):
        print(f"no such report: {args.report}", file=sys.stderr)
        return 1
    print(f"{args.report}:")
    shoot(args.report, args.out_dir, args.prefix, args.width, tuple(args.card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
