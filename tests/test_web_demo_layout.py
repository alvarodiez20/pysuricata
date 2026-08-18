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

The first fix over-corrected. Left-aligning every block in the shell gave the
page four widths — prose at 52ch, the panes at 760, the control row and the
report at 1064 — all starting at the same x and stacked down the left of an
empty monitor. Left edges only compose when the blocks are close in width. The
page now has one reading column, centred, and one full-width report centred on
the same axis: two widths on one line of symmetry.

The invariants encoded here are the ones that were actually violated, not the
boxes that were easiest to read:

* **the column tracks the window**, so a desktop viewport is not handed a phone
  measure. Asserted as a floor at 1280px, not as an exact width, so the design
  can be retuned without rewriting the test;
* **one axis.** The reading column and the report share a centre, which is what
  lets them differ in width without reading as a slip;
* **the reading track holds its measure**, as a grid track rather than a cap per
  block — capping each block centres each on its own width, and a 20ch heading
  beside a 52ch lede would give the page two left edges where it has one;
* **the control row ends where the report ends** (#317), since it labels it;
* **the report takes the whole window on a phone** (#317), where the page's
  gutter around the report's own is a gutter charged twice;
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

#: `--shell` and `--measure` in `web/index.html`. Kept as the numbers the page
#: declares rather than as a copy of a measurement, so a deliberate retune
#: fails here and gets updated, and an accidental one is caught.
SHELL = 1120
MEASURE = 720

#: At or below this the report frame breaks out of the page gutter and takes the
#: whole window. The page's tablet breakpoint, and also the widest of the three
#: the report's own layout criteria are measured at below desktop.
FULL_BLEED = 768

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
    actions: box('.result-actions'),
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


@pytest.mark.parametrize("width", [w for w in WIDTHS if w > FULL_BLEED])
def test_the_reading_track_holds_its_measure(measurements, width):
    """The blocks that are read sit in one track of `--measure`, not one cap per
    block: capping each would centre each on its own width, and a 20ch heading
    beside a 52ch lede would give the page two left edges where it has one."""
    m = measurements[width]
    track = min(MEASURE, m["column"]["width"] - 2 * 28)  # the gutter at this size

    assert m["log"]["width"] == track, (
        f"at {width}px the reading track is {m['log']['width']}px, not {track}px"
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


@pytest.mark.parametrize("width", [w for w in WIDTHS if w > FULL_BLEED])
def test_the_reading_column_and_the_report_share_a_centre(measurements, width):
    """Two widths on one axis, which is what stops two widths reading as a slip.

    They shared a *left* edge for two releases, and that is what made the page
    look wrong on a monitor: prose at 52ch, panes at 760 and the report at 1064
    all starting at the same x, stacked down the left of an empty screen. Left
    edges only compose when the blocks are close in width. Centring both on one
    axis makes the report visibly wider than the text on purpose.

    Above the full-bleed breakpoint only. Below it the frame is at the window
    edge and the reading column is inside the gutter, so their centres agree for
    a different reason and the next test is the one that means something.
    """
    m = measurements[width]
    text = m["log"]["x"] + m["log"]["width"] / 2
    frame = m["frame"]["x"] + m["frame"]["width"] / 2

    # A pixel of slack: an odd leftover width splits unevenly between the two
    # side tracks, and a page is not crooked because a rounding went one way.
    assert abs(text - frame) <= 1, (
        f"at {width}px the reading column is centred on {text}px and the report "
        f"on {frame}px — they are meant to share an axis"
    )


@pytest.mark.parametrize("width", [w for w in WIDTHS if w <= FULL_BLEED])
def test_a_phone_gives_the_report_the_whole_window(measurements, width):
    """The report is a document with its own gutters, so the page's gutter is
    charged twice on top of it.

    That is affordable on a monitor and not on a phone: at 390px it took 47px of
    page gutter before the report's own 40px of padding, and handed the report a
    341px viewport. No phone is 341px wide, and the report's own layout criteria
    (#124) are measured at 390, 768 and 1240 -- so the one width a visitor
    actually saw the report at was the one width nobody had checked. Its nav
    clipped `Missing Values` mid-word there.
    """
    m = measurements[width]

    assert m["frame"]["x"] == 0, (
        f"at {width}px the report starts at {m['frame']['x']}px, so the page "
        f"gutter is still inside the frame"
    )
    assert m["frame"]["width"] == width, (
        f"at {width}px the report frame is {m['frame']['width']}px wide, not the "
        f"{width}px the device has"
    )


@pytest.mark.parametrize("width", [w for w in WIDTHS if w > FULL_BLEED])
def test_the_control_row_ends_where_the_report_ends(measurements, width):
    """`.result-actions` labels and resets the frame directly beneath it.

    Capped at `--pane` under a `--shell` frame, `another file` sat 300px short of
    the right edge of the thing it resets -- one block with two right edges,
    which reads as a layout that slipped rather than as two measures chosen for a
    reason. The log and the ledger keep their cap; they are reading blocks, not
    controls for the frame.

    Above the full-bleed breakpoint only. Below it the frame is deliberately the
    one thing on the page outside the gutter, so the control row keeping the
    gutter is the point rather than a mismatch.
    """
    m = measurements[width]
    actions = m["actions"]["x"] + m["actions"]["width"]
    frame = m["frame"]["x"] + m["frame"]["width"]

    assert actions == frame, (
        f"at {width}px the control row ends at {actions}px and the report it "
        f"controls ends at {frame}px"
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
def test_the_panes_keep_their_reading_measure(measurements, width, pane):
    """Widening the shell is only right for the blocks that gain by it. A mono
    log set 1120px wide, or a ledger with its value a thousand pixels from its
    label, is worse than the narrow column this change was fixing."""
    m = measurements[width]

    assert m[pane]["width"] <= MEASURE, (
        f"at {width}px the {pane} is {m[pane]['width']}px wide, past the "
        f"{MEASURE}px measure it is meant to keep"
    )
