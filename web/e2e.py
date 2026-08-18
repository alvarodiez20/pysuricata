#!/usr/bin/env python3
"""End-to-end check that the browser demo actually renders a report (#1).

`worker.js` installs `pysuricata==<latest>` from PyPI at page load, which means
every release edits the demo's launch asset in production with nothing testing
it first. `tests/test_web_demo_layout.py` never waits for the runtime -- by
design, since it measures CSS layout and a network-dependent Pyodide boot has
no place in that fixture. Nothing else drives the real path: drop a file, boot
Pyodide, `micropip.install` the just-published wheel, profile it, render the
report.

And DOM presence would not be enough anyway. Chrome silently drops a `srcdoc`
document past ~700 KB -- no error, no console warning, no failed network
request, just a blank frame (`web/index.html` moved to a blob URL over this
exact failure). Nothing rules out a *different* silent blank in the same shape:
a stylesheet that fails to inline, a chart library returning empty SVG, a
report that renders structurally correct markup with no visible ink. The only
check that would have caught any of those is looking at what the pixels
actually show, so that is what this asserts on instead of `innerHTML`.

Usage:
    python web/e2e.py                                    # serves web/ locally
    python web/e2e.py --url https://pysuricata.pages.dev  # the live demo
    python web/e2e.py --out screenshots/                  # keep the capture

Exit 0 means a real profile ran and the report frame has visible content.
Anything else -- the runtime never became ready, the run failed, the frame
rendered blank -- exits 1 with the runtime's own status text and any console
errors, and still writes the screenshot so a failed CI run has something to
look at.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import io
import os
import socketserver
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"

#: Below this fraction of non-background pixels, a screenshot is "blank" --
#: the report failed to paint rather than merely being sparse. The real report
#: is mostly whitespace by design (cards, gutters, breathing room), so this is
#: set well under what an actual render produces, not at a midpoint.
MIN_INK_FRACTION = 0.02

#: Below this many distinct colours, a screenshot is a flat swatch rather than
#: a report -- a single wrong colour can clear the ink-fraction bar above.
MIN_DISTINCT_COLOURS = 20

#: Boot (Pyodide + micropip installing pysuricata from PyPI) and the profile
#: run itself are two different waits with two different budgets: boot is
#: bounded by CDN and PyPI latency, the run by the sample's own size.
BOOT_TIMEOUT_MS = 90_000
RUN_TIMEOUT_MS = 60_000


class DemoE2EFailure(RuntimeError):
    """The demo did not reach a visibly-rendered report."""


def _chrome() -> str | None:
    if explicit := os.environ.get("PYSURICATA_CHROME"):
        return explicit
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return None


@contextmanager
def _target_url(url: str | None):
    """The given URL verbatim, or `web/` served over loopback if there is none.

    A worker cannot be registered from `file://`, so testing the local
    checkout needs a real HTTP origin -- but a live `--url` needs no server of
    its own, and starting one anyway would just be an unused thread per run.
    """
    if url:
        yield url
        return

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


def _ink_stats(png_bytes: bytes) -> tuple[float, int]:
    """(non-background pixel fraction, distinct colour count) for a screenshot.

    The background colour is read off the image's own corner pixels rather
    than assumed as white, so this works the same in light and dark theme.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    width, height = img.size
    if width == 0 or height == 0:
        return 0.0, 0

    pixels = np.asarray(img).reshape(-1, 3)
    background = pixels[0]  # top-left corner, outside any card or chart
    non_background = int(np.any(pixels != background, axis=1).sum())
    distinct = len(np.unique(pixels, axis=0))
    return non_background / len(pixels), distinct


def run(url: str | None, out_dir: Path, headed: bool = False) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment guard
        raise DemoE2EFailure(
            "Playwright is not installed -- `uv sync --all-extras --group browser`"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = out_dir / "report.png"

    launch: dict[str, object] = {"headless": not headed}
    if chrome := _chrome():
        launch["executable_path"] = chrome

    console_errors: list[str] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(**launch)
        except Exception as exc:  # no browser binary on this machine
            raise DemoE2EFailure(f"Chromium is not available: {exc}") from exc

        with _target_url(url) as target:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error"
                else None,
            )
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))

            print(f"Loading {target}", file=sys.stderr)
            page.goto(target, wait_until="load")

            # `#sample` staying disabled and a boot error both mean "never
            # ready", but only one of them is announced -- a stuck boot never
            # touches `#status`, and a boot failure never enables `#sample`.
            # Racing both means a real boot failure is reported in seconds
            # instead of after the full boot timeout.
            try:
                page.wait_for_selector(
                    "#sample:not([disabled]), #status.failed",
                    timeout=BOOT_TIMEOUT_MS,
                )
            except Exception as exc:
                detail = page.inner_text("#statusDetail") or ""
                text = page.inner_text("#statusText") or ""
                raise DemoE2EFailure(
                    f"The runtime never became ready to profile anything: "
                    f"{text!r} {detail!r} ({exc})"
                ) from exc

            status_class = page.eval_on_selector("#status", "el => el.className")
            if "failed" in status_class.split():
                detail = page.inner_text("#statusDetail") or ""
                text = page.inner_text("#statusText") or ""
                page.screenshot(path=str(screenshot_path))
                raise DemoE2EFailure(f"The runtime failed to boot: {text!r} {detail!r}")

            page.click("#sample")

            try:
                page.wait_for_selector(
                    "#status.done, #status.failed", timeout=RUN_TIMEOUT_MS
                )
            except Exception as exc:
                raise DemoE2EFailure(
                    f"The sample run neither finished nor failed within "
                    f"{RUN_TIMEOUT_MS / 1000:.0f}s ({exc})"
                ) from exc

            status_class = page.eval_on_selector("#status", "el => el.className")
            text = page.inner_text("#statusText") or ""
            detail = page.inner_text("#statusDetail") or ""

            if "failed" in status_class.split():
                page.screenshot(path=str(screenshot_path))
                raise DemoE2EFailure(
                    f"The runtime reported failure: {text!r} {detail!r}"
                )

            # `renderResult()` assigns the blob URL and unhides the panel in the
            # same tick, but painting an iframe's new document is not: give the
            # compositor a beat before the screenshot is taken.
            page.wait_for_timeout(500)
            page.locator("#report").screenshot(path=str(screenshot_path))

    ink_fraction, distinct_colours = _ink_stats(screenshot_path.read_bytes())
    print(
        f"report frame: {ink_fraction:.1%} non-background, "
        f"{distinct_colours} distinct colours -- {screenshot_path}",
        file=sys.stderr,
    )

    if console_errors:
        print("console errors seen during the run:", file=sys.stderr)
        for line in console_errors:
            print(f"  {line}", file=sys.stderr)

    if ink_fraction < MIN_INK_FRACTION or distinct_colours < MIN_DISTINCT_COLOURS:
        raise DemoE2EFailure(
            f"The report frame looks blank: {ink_fraction:.1%} non-background "
            f"pixels (need >= {MIN_INK_FRACTION:.0%}), {distinct_colours} distinct "
            f"colours (need >= {MIN_DISTINCT_COLOURS}). Status said {text!r}; "
            f"the runtime believes it succeeded and the screen disagrees. "
            f"See {screenshot_path}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url",
        default=None,
        help="Demo URL to test (default: serve web/ locally over loopback)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "e2e-out",
        help="Directory to write the report screenshot into (default: e2e-out/)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window (for local debugging)",
    )
    args = parser.parse_args()

    try:
        run(args.url, args.out, headed=args.headed)
    except DemoE2EFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS: the demo profiled the sample and the report is visibly painted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
