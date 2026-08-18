"""#300 — dark mode measured in place, against the background actually painted.

The dark palette in `docs/internal/design/tokens.css` was **derived, not
verified**: the dark data scale was measured, and the rest was reasoned about
rather than looked at. `tests/test_contrast.py` guards the arithmetic of the
token pairs, and the plan lists what a token test structurally cannot reach.
The first item is *a foreground on a tinted surface* -- `test_contrast.py`
measures against `--paper`, and a token that clears 4.5:1 on the paper can fail
on a coloured segment.

This closes that gap by not using tokens at all. It walks every text-bearing
element in a rendered report, finds the first ancestor that actually paints a
background, composites any translucent layers on the way down, and computes the
real ratio. On Titanic that is ~690 elements measured against `rgb(37, 35, 32)`
-- the card surface -- rather than the `rgb(28, 26, 23)` paper the token test
uses, which is the whole point.

**Two traps, both already paid for and recorded in `CLAUDE.md`.**

1. The report's dark mode is the *absence* of a `light` class, not
   `prefers-color-scheme`, so Playwright's `color_scheme=` does nothing to it.
   Six "theme" cases once measured one state twice.
2. `transition: background-color 0.3s` means an immediate read after toggling
   returns the old value.

So the fixture toggles the class, waits, and **asserts the paper actually
moved** before measuring anything. Without that assertion a selector change
would leave this whole file passing against light mode.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysuricata import profile

REPO = Path(__file__).resolve().parents[1]
TITANIC = REPO / "docs" / "assets" / "titanic.csv"

BREAKPOINTS = (390, 1240)

#: The dark paper, from `tokens.css`. Named so a theme that silently stops
#: switching fails here rather than passing against the light palette.
DARK_PAPER = "rgb(28, 26, 23)"

#: `transition: background-color 0.3s` on `#pysuricata-report`, plus margin.
_TRANSITION_MS = 500

_TO_DARK = """() => {
  const root = document.getElementById('pysuricata-report');
  if (!root) return null;
  root.classList.remove('light');
  document.body.classList.remove('light');
  return true;
}"""

#: Walks the tree and returns one record per text-bearing element. The
#: interesting part is `painted()`: the background a reader actually sees is
#: the first ancestor that paints one, composited down through any translucent
#: layers above it. That is the step a token pair cannot take.
_AUDIT = """() => {
  const parse = (c) => {
    const m = c.match(/rgba?\\(([^)]+)\\)/);
    if (!m) return null;
    const p = m[1].split(',').map(s => parseFloat(s.trim()));
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const lin = (v) => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  const lum = (c) => 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  const over = (fg, bg) => ({
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a),
    a: 1,
  });

  const painted = (el) => {
    const stack = [];
    let node = el;
    while (node && node.nodeType === 1) {
      const bg = parse(getComputedStyle(node).backgroundColor);
      if (bg && bg.a > 0) {
        stack.push(bg);
        if (bg.a >= 1) break;
      }
      node = node.parentElement;
    }
    if (!stack.length) return {r: 255, g: 255, b: 255, a: 1};
    let base = stack[stack.length - 1];
    for (let i = stack.length - 2; i >= 0; i--) base = over(stack[i], base);
    return base;
  };

  const out = [];
  for (const el of document.querySelectorAll('*')) {
    const own = [...el.childNodes]
      .filter(n => n.nodeType === 3)
      .map(n => n.textContent.trim())
      .join(' ')
      .trim();
    if (!own) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    if (parseFloat(cs.opacity) === 0) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;

    const fg = parse(cs.color);
    if (!fg) continue;
    const bg = painted(el);
    const solid = over(fg, bg);
    const a = lum(solid), b = lum(bg);
    const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

    // WCAG AA: 3.0 for large text (>=24px, or >=18.66px bold), else 4.5.
    const size = parseFloat(cs.fontSize);
    const weight = parseInt(cs.fontWeight, 10) || 400;
    const large = size >= 24 || (size >= 18.66 && weight >= 700);

    out.push({
      text: own.slice(0, 40),
      cls: (el.className || '').toString().slice(0, 44),
      colour: cs.color,
      background: `rgb(${Math.round(bg.r)}, ${Math.round(bg.g)}, ${Math.round(bg.b)})`,
      ratio: Math.round(ratio * 100) / 100,
      need: large ? 3.0 : 4.5,
      section: (el.closest('section[id]') || {}).id || '(none)',
    });
  }
  return out;
}"""


def _chrome() -> str | None:
    import os

    if explicit := os.environ.get("PYSURICATA_CHROME"):
        return explicit
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return None


@pytest.fixture(scope="module")
def in_dark(tmp_path_factory):
    """Every text element in the report, measured in dark mode at both widths."""
    playwright = pytest.importorskip(
        "playwright.sync_api", reason="in-situ contrast needs a browser"
    )

    page_file = tmp_path_factory.mktemp("dark") / "report.html"
    page_file.write_text(profile(pd.read_csv(TITANIC), seed=0).html, encoding="utf-8")

    launch = {}
    if chrome := _chrome():
        launch["executable_path"] = chrome

    out = {}
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(**launch)
        except Exception as exc:  # no browser binary on this machine
            pytest.skip(f"Chromium is not available: {exc}")
        for width in BREAKPOINTS:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(page_file.as_uri())
            page.wait_for_timeout(900)

            assert page.evaluate(_TO_DARK), "no #pysuricata-report -- the shell moved"
            page.wait_for_timeout(_TRANSITION_MS)
            paper = page.evaluate(
                "() => getComputedStyle("
                "document.getElementById('pysuricata-report')).backgroundColor"
            )
            assert paper == DARK_PAPER, (
                f"the report is {paper} at {width}px, not the dark paper "
                f"{DARK_PAPER} -- every assertion below would be measuring "
                f"light mode"
            )

            out[width] = page.evaluate(_AUDIT)
            page.close()
        browser.close()
    return out


@pytest.mark.browser
@pytest.mark.parametrize("width", BREAKPOINTS)
class TestDarkModeReadsOnTheSurfaceItIsPaintedOn:
    def test_the_audit_is_reading_real_elements(self, in_dark, width):
        """A check over rendered output is only as good as the markup the
        fixture reaches, and an empty audit passes every assertion below it.
        Titanic renders ~650 text elements; a tenth of that means the walk
        stopped finding them."""
        measured = in_dark[width]

        assert len(measured) >= 300, (
            f"only {len(measured)} text elements measured at {width}px -- "
            f"this audit has stopped reading the report"
        )

    def test_it_measures_against_more_than_the_paper(self, in_dark, width):
        """The whole reason this exists. If every element resolves to the
        paper, it is doing what `test_contrast.py` already does and the tinted
        surfaces are still unchecked."""
        backgrounds = {row["background"] for row in in_dark[width]}

        assert len(backgrounds) >= 2, (
            f"every element resolved to {backgrounds} -- no nested or tinted "
            f"surface was measured, so this adds nothing to the token test"
        )

    def test_every_mark_clears_AA_on_its_actual_background(self, in_dark, width):
        failures = [row for row in in_dark[width] if row["ratio"] + 0.005 < row["need"]]

        assert not failures, "\n".join(
            f"  {row['ratio']} (need {row['need']}) in {row['section']} "
            f".{row['cls']}: {row['colour']} on {row['background']} "
            f"-- {row['text']!r}"
            for row in sorted(failures, key=lambda r: r["ratio"])[:12]
        )
