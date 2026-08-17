#!/usr/bin/env python3
"""Screenshot the report at every breakpoint and theme, for review (#124).

    uv run --group browser python scripts/contact_sheet.py

Six images into `contact-sheet/`, uploaded by the `layout` CI job as an
artifact. **It never fails the build**, and that is a decision rather than an
oversight: thirteen redesign issues are *supposed* to change every pixel, so a
pixel-equality gate would be switched off during the first phase and stay off.
The structural assertions in `tests/test_report_layout.py` are the gate. These
are how a human reviews a phase in thirty seconds instead of thirty minutes.

Full-page captures, not viewports, because the criteria this supports are about
things that happen below the fold -- a card that overflows its column, a chart
that collapses at 390px, a theme that moves a box instead of recolouring it.
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "contact-sheet"
TITANIC = REPO_ROOT / "docs" / "assets" / "titanic.csv"

#: The three the redesign issues quote their acceptance numbers at.
BREAKPOINTS = (390, 768, 1240)
THEMES = ("light", "dark")


def _chrome() -> str | None:
    import os

    if explicit := os.environ.get("PYSURICATA_CHROME"):
        return explicit
    for candidate in pathlib.Path("/opt/pw-browsers").glob(
        "chromium-*/chrome-linux/chrome"
    ):
        return str(candidate)
    return None  # let Playwright resolve its own default


def _set_theme(page, theme: str) -> None:
    """Dark is the absence of a `light` class, not `prefers-color-scheme`.

    Passing Playwright's `color_scheme=` does nothing here, which is easy to
    miss because the run still produces six files -- they just come out
    byte-identical in pairs, and a contact sheet that shows one theme twice is
    worse than none. The wait is for `transition: background-color 0.3s`; shoot
    during it and the paper is caught mid-fade.
    """
    page.evaluate(
        """(theme) => {
            const light = theme === 'light';
            const report = document.getElementById('pysuricata-report');
            if (report) report.classList.toggle('light', light);
            document.body.classList.toggle('light', light);
        }""",
        theme,
    )
    page.wait_for_timeout(450)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed. Run with: uv sync --group browser")
        return 2

    import pandas as pd

    from pysuricata import profile

    OUT.mkdir(exist_ok=True)
    page_file = OUT / "_report.html"
    page_file.write_text(profile(pd.read_csv(TITANIC), seed=0).html, encoding="utf-8")

    launch = {}
    if chrome := _chrome():
        launch["executable_path"] = chrome

    written = []
    with sync_playwright() as play:
        browser = play.chromium.launch(**launch)
        for width in BREAKPOINTS:
            for theme in THEMES:
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.goto(page_file.as_uri())
                # The report draws its charts on load.
                page.wait_for_timeout(1200)
                _set_theme(page, theme)
                shot = OUT / f"{width:04d}-{theme}.png"
                page.screenshot(path=str(shot), full_page=True)
                written.append(shot)
                page.close()
        browser.close()

    page_file.unlink(missing_ok=True)
    for shot in written:
        print(f"  {shot.relative_to(REPO_ROOT)}  {shot.stat().st_size:,} bytes")
    print(f"{len(written)} images in {OUT.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
