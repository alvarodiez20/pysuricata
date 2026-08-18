"""The browser demo's page is a desktop page, measured in a desktop window.

`web/index.html` laid every block out in one column capped at `--measure: 600px`
with no wider breakpoint anywhere in its stylesheet. On a phone that is a
deliberate reading measure; on a 1440px monitor it is a phone, centred, with
420px of empty paper on each side — and the report iframe broke out to 1120px
underneath it, so the one wide thing on the page made the narrow strip above it
look like a mistake rather than a choice.

Nothing could catch that from Python. The page is static markup with inlined
CSS, and the bug is a width that never widened: no selector is missing, no rule
is malformed, `ruff` and every existing test pass either way. It only exists at
a viewport, which is where these cases look for it.

The invariants encoded here are the ones that were actually violated, not the
boxes that were easiest to read:

* **the column tracks the window**, so a desktop viewport is not handed a phone
  measure. Asserted as a floor at 1280px, not as an exact width, so the design
  can be retuned without rewriting the test;
* **one left edge.** The hero, the report frame and the page chrome line up.
  The old breakout hack (`margin-left: 50%` and a translate) put the report
  56px left of the text it belonged to;
* **the panes stay readable.** A mono log and a label/value ledger get worse
  stretched to 1120px, so they carry their own cap and must keep it;
* **nothing scrolls sideways** at any width, phone widths included.

These need Playwright and a Chromium build; they are marked `browser` and skip
without one, like the report's own layout cases. The page is served over
`http://` rather than opened as a `file://` URI because it registers a module
worker, which Chromium refuses on a file origin. The worker then fails to boot
Pyodide unless the machine can reach jsDelivr — which is fine and is why the
fixture never waits for it. Every box measured here is laid out by CSS alone,
so the runtime's fate does not enter into it.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"

#: Phone, tablet, laptop, desktop. 1120 is the shell width, so 1280 is the
#: first width where the column stops growing and the page starts centring.
WIDTHS = (320, 390, 768, 1024, 1280, 1440, 1920)

#: `--shell` and `--pane` in `web/index.html`. Kept as the numbers the page
#: declares rather than as a copy of a measurement, so a deliberate retune
#: fails here and gets updated, and an accidental one is caught.
SHELL = 1120
PANE = 760

pytestmark = pytest.mark.browser


def _chrome() -> str | None:
    import os

    if explicit := os.environ.get("PYSURICATA_CHROME"):
        return explicit
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return None


@pytest.fixture(scope="module")
def demo_url():
    """`web/` over loopback. A worker cannot be registered from `file://`."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(WEB)
    )

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

        def handle_error(self, request, client_address):  # noqa: D102 - silence
            pass

    with Quiet(("127.0.0.1", 0), handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}/index.html"
        finally:
            server.shutdown()
            thread.join(timeout=5)


#: Every box this file asserts on, read in one pass per width.
_MEASURE = """
() => {
  const box = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {x: Math.round(r.x), width: Math.round(r.width)};
  };
  return {
    window: window.innerWidth,
    // scrollWidth against clientWidth on the document: the only definition of
    // "the page scrolls sideways" that does not also name elements that clip.
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    column: box('main.col'),
    nav: box('header nav'),
    footer: box('footer .col'),
    hero: box('h1'),
    frame: box('.frame-wrap'),
    log: box('.log'),
    ledger: box('.ledger'),
  };
}
"""

#: The result panel is hidden until a run finishes, and a run cannot finish
#: without the runtime. Unhide it: a pane nobody can measure reports "absent",
#: and absent is indistinguishable from correct.
_REVEAL = """
() => {
  document.getElementById('resultPanel').classList.remove('hidden');
  document.getElementById('log').classList.remove('hidden');
  document.getElementById('ledger').innerHTML = '<dt>rows</dt><dd>891</dd>';
}
"""


@pytest.fixture(scope="module")
def measurements(demo_url):
    """The demo page measured at every width, once."""
    playwright = pytest.importorskip(
        "playwright.sync_api", reason="browser layout checks need Playwright"
    )

    launch = {}
    if chrome := _chrome():
        launch["executable_path"] = chrome

    out = {}
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(**launch)
        except Exception as exc:  # no browser binary on this machine
            pytest.skip(f"Chromium is not available: {exc}")
        for width in WIDTHS:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(demo_url)
            page.evaluate(_REVEAL)
            out[width] = page.evaluate(_MEASURE)
            page.close()
        browser.close()
    return out


def test_the_server_is_serving_the_demo():
    """A guard on the fixture rather than on the page: if `web/index.html` moves,
    every case below would skip as "no element found" and read as passing."""
    assert (WEB / "index.html").is_file(), f"no demo page at {WEB / 'index.html'}"


@pytest.mark.parametrize("width", WIDTHS)
def test_the_page_never_scrolls_sideways(measurements, width):
    m = measurements[width]

    assert m["overflow"] <= 0, (
        f"at {width}px the document is {m['overflow']}px wider than its viewport"
    )


@pytest.mark.parametrize("width", [w for w in WIDTHS if w >= 1280])
def test_a_desktop_window_is_not_given_a_phone_column(measurements, width):
    """The bug, stated as the thing it broke. The column was 600px at 1920px."""
    m = measurements[width]

    assert m["column"]["width"] == SHELL, (
        f"at {width}px the page column is {m['column']['width']}px, not the "
        f"{SHELL}px shell — a desktop window is being laid out at a phone measure"
    )


@pytest.mark.parametrize("width", [w for w in WIDTHS if w < 1280])
def test_below_the_shell_the_column_fills_the_window(measurements, width):
    """Between a phone and the shell there is nothing to centre: the gutters own
    the edges, and a fixed cap here is what made a tablet look like a phone."""
    m = measurements[width]

    assert m["column"]["width"] == m["window"], (
        f"at {width}px the column is {m['column']['width']}px inside a "
        f"{m['window']}px window instead of filling it"
    )


@pytest.mark.parametrize("width", WIDTHS)
def test_the_hero_and_the_report_share_a_left_edge(measurements, width):
    """The report used to be centred on the viewport while the text was centred
    in a 600px column, so the two disagreed by 56px at every desktop width."""
    m = measurements[width]

    assert m["hero"]["x"] == m["frame"]["x"], (
        f"at {width}px the hero starts at {m['hero']['x']}px and the report at "
        f"{m['frame']['x']}px — they are meant to be one column"
    )


@pytest.mark.parametrize("width", WIDTHS)
def test_the_chrome_frames_the_same_column(measurements, width):
    m = measurements[width]

    assert m["nav"]["x"] == m["footer"]["x"], (
        f"at {width}px the header starts at {m['nav']['x']}px and the footer at "
        f"{m['footer']['x']}px"
    )


@pytest.mark.parametrize("pane", ["log", "ledger"])
@pytest.mark.parametrize("width", WIDTHS)
def test_the_panes_keep_their_reading_cap(measurements, width, pane):
    """Widening the shell is only right for the blocks that gain by it. A mono
    log set 1120px wide, or a ledger with its value a thousand pixels from its
    label, is worse than the narrow column this change was fixing."""
    m = measurements[width]

    assert m[pane]["width"] <= PANE, (
        f"at {width}px the {pane} is {m[pane]['width']}px wide, past the "
        f"{PANE}px cap it is meant to keep"
    )
