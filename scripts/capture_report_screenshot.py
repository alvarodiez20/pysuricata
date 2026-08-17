"""Capture the README's hero image from a real report.

    python scripts/regenerate_example_report.py     # produce the report
    uv run --with playwright python scripts/capture_report_screenshot.py

The previous screenshot sat in `docs/assets/` for thirty releases showing a
version string of 0.0.26 and a design two redesigns old, because regenerating
it meant reverse-engineering how it had been taken. This is that recipe, so the
next person can re-shoot it in one command.

It is deliberately not wired into CI: the check would need a browser, and a
picture that is a release behind is a cosmetic lag, not a false claim -- the
report header carries its own version, so staleness is visible in the image
itself.

The clip stops just above the Sample section. That boundary is found in the
DOM rather than hard-coded, so a layout change moves the crop instead of
slicing a table row in half.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "docs" / "assets" / "titanic_report.html"
OUT = REPO / "docs" / "assets" / "report-screenshot.png"

WIDTH = 1280
# Retina: the README displays it at 820px, so 2x keeps the type crisp.
SCALE = 2
FALLBACK_HEIGHT = 780

# Where the crop should stop. Everything above this is the dataset-level
# summary, which is what the README is advertising.
_STOP_AT = """() => {
  const h = [...document.querySelectorAll('h2, h3')]
    .find(e => /^sample$/i.test(e.textContent.trim()));
  return h ? h.getBoundingClientRect().top + window.scrollY : null;
}"""


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed; try:", file=sys.stderr)
        print("  uv run --with playwright python " + __file__, file=sys.stderr)
        return 2

    if not REPORT.exists():
        print(f"{REPORT} is missing; run scripts/regenerate_example_report.py first")
        return 2

    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": 900}, device_scale_factor=SCALE
        )
        page.goto(REPORT.as_uri())
        # The report renders its charts on load; give them a beat to settle.
        page.wait_for_timeout(1500)

        stop = page.evaluate(_STOP_AT)
        height = int(stop) - 12 if stop else FALLBACK_HEIGHT
        page.screenshot(
            path=str(OUT), clip={"x": 0, "y": 0, "width": WIDTH, "height": height}
        )
        browser.close()

    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size:,} bytes, {height}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
