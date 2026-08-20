"""#39, first acceptance box: download the report, re-open it, and require it
to still be the report.

The self-download button re-serialises the **live DOM** into a new document. It
does not ask the renderer what it emitted; it goes looking, with two regexes:

* styles are kept only if their text matches ``#pysuricata-report`` or
  ``suricata-standalone``;
* the script is whichever single ``<script>`` element mentions
  ``toggleDarkMode`` -- ``.find()``, so the first match and no other.

Both couplings are invisible from the Python side, and #39 says so in its own
words: *any restructuring breaks it silently*. That is not a hypothetical in
this repo. #142 renamed a CSS class and left a control inert for eleven
versions with a clean console and 1,735 green tests, which is the note
``CLAUDE.md`` keeps about ``functionality.js`` and the renderers never importing
each other.

The second regex is the one with a live trap under it. ``report/html.py``
concatenates ``functionality.js``, ``tooltips.js``, ``pagination.js`` and
``description-editor.js`` into **one** ``<script>`` placeholder, so today every
one of them rides along on the single element that happens to contain
``toggleDarkMode``. Split that tag -- a perfectly reasonable thing to do while
cutting report bytes, which is what #39 exists for -- and three of the four
vanish from every downloaded report, with no error anywhere.

So the guard here is deliberately **not** a list of features to spot-check. It
is an equivalence: every ``<style>`` and every ``<script>`` the live report
carries must survive into the download, whatever they are and however many
there are. A file added to ``static/js/`` is covered on the day it is added,
and a tag that gets split fails immediately.

Written before the restructuring, which is the order #39 asks for and the
reason this file exists ahead of the work it protects.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from pysuricata import profile

REPO = Path(__file__).resolve().parents[1]
TITANIC = REPO / "docs" / "assets" / "titanic.csv"

#: `transition: background-color 0.3s` on `#pysuricata-report`. An immediate
#: read after a class flip returns the *old* colour -- the axis has to be given
#: time to prove it moved. `CLAUDE.md` records six "theme" cases that measured
#: one state twice for want of this.
_THEME_TRANSITION_MS = 400


def _chrome() -> str | None:
    if explicit := os.environ.get("PYSURICATA_CHROME"):
        return explicit
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return None


@pytest.fixture(scope="module")
def report_html() -> str:
    """`seed` fixes the reservoir, so the document is byte-deterministic."""
    return profile(pd.read_csv(TITANIC), seed=0).html


@pytest.mark.browser
class TestTheDownloadedReportIsStillTheReport:
    @pytest.fixture(scope="class")
    def browser(self):
        playwright = pytest.importorskip(
            "playwright.sync_api", reason="the download round-trip needs Playwright"
        )
        launch = {}
        if chrome := _chrome():
            launch["executable_path"] = chrome
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(**launch)
            except Exception as exc:  # no browser binary on this machine
                pytest.skip(f"Chromium is not available: {exc}")
            yield browser
            browser.close()

    @pytest.fixture(scope="class")
    def round_trip(self, browser, report_html, tmp_path_factory):
        """Open the report, press its own download button, keep what came out.

        The button builds a Blob and clicks an anchor at it, so this is the
        real path a reader takes -- not a re-implementation of it in Python,
        which would pass even if the button were inert.
        """
        tmp = tmp_path_factory.mktemp("round_trip")
        live_file = tmp / "report.html"
        live_file.write_text(report_html, encoding="utf-8")

        page = browser.new_page(viewport={"width": 1240, "height": 900})
        page.goto(live_file.as_uri())
        page.wait_for_timeout(900)  # the report draws its charts on load

        live = page.evaluate(_INVENTORY)

        with page.expect_download() as info:
            page.click('[data-action="download-report"]')
        downloaded_file = tmp / "downloaded.html"
        info.value.save_as(str(downloaded_file))
        page.close()

        text = downloaded_file.read_text(encoding="utf-8")

        page2 = browser.new_page(viewport={"width": 1240, "height": 900})
        errors: list[str] = []
        page2.on("pageerror", lambda e: errors.append(str(e)))
        page2.on(
            "console",
            lambda m: errors.append(m.text) if m.type == "error" else None,
        )
        page2.goto(downloaded_file.as_uri())
        page2.wait_for_timeout(900)
        reopened = page2.evaluate(_INVENTORY)

        yield {
            "live": live,
            "reopened": reopened,
            "text": text,
            "page": page2,
            "errors": errors,
        }
        page2.close()

    # ---------------------------------------------------------------- assets

    def test_every_stylesheet_survives(self, round_trip):
        """The filter keeps a `<style>` only if its text names the report root
        or the standalone body. A partial that satisfies neither is dropped in
        silence, and the downloaded report renders unstyled in that region."""
        missing = [
            s[:80] for s in round_trip["live"]["styles"] if s not in round_trip["text"]
        ]

        assert not missing, (
            f"{len(missing)} of {len(round_trip['live']['styles'])} <style> blocks "
            f"did not survive the download -- the filter in downloadReport() "
            f"matches on '#pysuricata-report|suricata-standalone'. First missing "
            f"starts: {missing[0]!r}"
        )

    def test_every_script_survives(self, round_trip):
        """`.find()` keeps exactly one `<script>`. Four JS files are
        concatenated into one tag today, so all four ride on that single
        match -- split the tag and three disappear with no error raised."""
        missing = [
            s[:80] for s in round_trip["live"]["scripts"] if s not in round_trip["text"]
        ]

        assert not missing, (
            f"{len(missing)} of {len(round_trip['live']['scripts'])} <script> blocks "
            f"did not survive the download. downloadReport() takes the *first* "
            f"script matching /toggleDarkMode/ and discards the rest. First "
            f"missing starts: {missing[0]!r}"
        )

    def test_the_favicon_survives(self, round_trip):
        """The one binary payload the report is allowed to carry."""
        if not round_trip["live"]["favicon"]:
            pytest.skip("this report embeds no favicon")
        assert 'rel="icon"' in round_trip["text"]

    # ------------------------------------------------------------- integrity

    def test_the_downloaded_report_is_still_one_file(self, round_trip):
        """Re-serialising must not introduce a reference to something that is
        no longer beside it. The downloaded file is the copy most likely to be
        opened offline, on a machine that never had the original."""
        forbidden = {
            "a remote script": r'src\s*=\s*"https?:',
            "a remote stylesheet by href": r'href\s*=\s*"https?:[^"]*\.css',
            "a stylesheet link element": r'rel\s*=\s*"[^"]*stylesheet',
            "a CSS @import": r"@import",
            "a remote iframe or image": r'<(?:iframe|img)[^>]*\ssrc\s*=\s*"https?:',
        }
        offenders = {
            what: len(found)
            for what, pattern in forbidden.items()
            if (found := re.findall(pattern, round_trip["text"], re.I))
        }

        assert not offenders, (
            f"the downloaded report reaches the network for: {offenders}"
        )

    def test_no_card_is_lost_in_the_round_trip(self, round_trip):
        live, reopened = round_trip["live"], round_trip["reopened"]
        assert reopened["counts"] == live["counts"], (
            f"the document changed shape across the download: "
            f"{live['counts']} -> {reopened['counts']}"
        )

    def test_it_renders_identically(self, round_trip):
        """The measurement that makes this more than a byte check: the same
        elements must compute the same way in both documents."""
        live, reopened = round_trip["live"], round_trip["reopened"]
        drift = {
            sel: {
                prop: (values[prop], reopened["computed"][sel][prop])
                for prop in values
                if values[prop] != reopened["computed"][sel][prop]
            }
            for sel, values in live["computed"].items()
            if values is not None and reopened["computed"].get(sel) is not None
        }
        drift = {sel: d for sel, d in drift.items() if d}

        assert not drift, f"the downloaded report renders differently: {drift}"

    # ----------------------------------------------------------- it still runs

    def test_the_downloaded_report_raises_nothing(self, round_trip):
        """A downloaded report that throws on load is the silent failure this
        file exists to catch -- it still *looks* right in a screenshot."""
        assert not round_trip["errors"], (
            f"the downloaded report logged {len(round_trip['errors'])} error(s): "
            f"{round_trip['errors'][:3]}"
        )

    def test_dark_mode_still_toggles(self, round_trip):
        """The one control the download explicitly goes looking for, so the
        one whose loss would be most embarrassing. The theme is the *absence*
        of a `light` class rather than a media query, and the transition means
        an immediate read returns the old colour -- so this waits, and requires
        the colour to actually move."""
        page = round_trip["page"]
        read = "() => getComputedStyle(document.getElementById('pysuricata-report')).backgroundColor"

        before = page.evaluate(read)
        page.click('[data-action="toggle-dark-mode"]')
        page.wait_for_timeout(_THEME_TRANSITION_MS)
        after = page.evaluate(read)

        assert before != after, (
            f"the theme toggle did not change the report background in the "
            f"downloaded file (stayed {before}) -- the script was lost, or the "
            f"control's data-action no longer matches the handler"
        )


@pytest.mark.browser
class TestTheDownloadRemembersWhatYouWereLookingAt:
    """`downloadReport` reads the root's `light` class and stamps it onto the
    standalone body. That is the only piece of *state* the download carries, as
    opposed to content.

    Reading the **root** alone cannot test it, and the first version of this
    file did exactly that and passed against a deliberate break. The root
    carries its own `light` class inside `outerHTML`, so it arrives correct
    whatever the body does -- the box that is easiest to read is the one that
    proves nothing.

    What the `isLight` branch actually buys is the *body*.
    `body.suricata-standalone:not(.light)` paints `#1C1A17`, and the standalone
    body sits outside `#pysuricata-report` and cannot read its tokens, so those
    two literals in `_01-base.css` are the only place the palette is repeated.
    Lose the class and a light report is letterboxed in dark around every edge,
    with the report itself perfectly correct. So the invariant is that the two
    **agree**, which is what a reader sees and what the branch is for.

    Theme-agnostic on purpose -- it flips whatever the default is rather than
    pinning a colour, since the theme is the absence of a class rather than a
    media query and `CLAUDE.md` records six cases that measured one state twice
    for assuming otherwise.
    """

    @pytest.fixture(scope="class")
    def browser(self):
        playwright = pytest.importorskip(
            "playwright.sync_api", reason="the download round-trip needs Playwright"
        )
        launch = {}
        if chrome := _chrome():
            launch["executable_path"] = chrome
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(**launch)
            except Exception as exc:
                pytest.skip(f"Chromium is not available: {exc}")
            yield browser
            browser.close()

    @pytest.mark.parametrize("flip", [False, True], ids=["as-opened", "toggled"])
    def test_the_theme_survives_the_download(
        self, browser, report_html, tmp_path, flip
    ):
        """Both directions, because one of them is inert.

        `isLight` writes a class when the report is light and writes nothing
        when it is dark. A test that only ever downloads the dark state
        exercises the empty branch, and deleting the whole expression still
        passes -- which is what the first two versions of this test did. The
        default here is light, so `flip=False` is the case with something to
        lose and `flip=True` is its control.
        """
        read = """() => ({
            root: getComputedStyle(
                document.getElementById('pysuricata-report')).backgroundColor,
            body: getComputedStyle(document.body).backgroundColor,
        })"""
        live_file = tmp_path / "report.html"
        live_file.write_text(report_html, encoding="utf-8")

        page = browser.new_page(viewport={"width": 1240, "height": 900})
        page.goto(live_file.as_uri())
        page.wait_for_timeout(900)

        shown = page.evaluate(read)
        if flip:
            page.click('[data-action="toggle-dark-mode"]')
            page.wait_for_timeout(_THEME_TRANSITION_MS)
            flipped = page.evaluate(read)
            assert flipped["root"] != shown["root"], (
                "the theme did not move on the live report, so this cannot "
                f"test whether the download remembers it (both {shown['root']})"
            )
            shown = flipped

        with page.expect_download() as info:
            page.click('[data-action="download-report"]')
        downloaded = tmp_path / "downloaded.html"
        info.value.save_as(str(downloaded))
        page.close()

        reopened = browser.new_page(viewport={"width": 1240, "height": 900})
        reopened.goto(downloaded.as_uri())
        reopened.wait_for_timeout(_THEME_TRANSITION_MS)
        got = reopened.evaluate(read)
        reopened.close()

        assert got["root"] == shown["root"], (
            f"the report was downloaded showing {shown['root']} and re-opened "
            f"showing {got['root']}"
        )
        assert got["body"] == got["root"], (
            f"the downloaded report paints its body {got['body']} behind a "
            f"report painted {got['root']} -- downloadReport() stamps the "
            f"root's `light` class onto the standalone body, and that did not "
            f"survive, so the page is letterboxed in the other theme"
        )


#: Read once per document. Keeping the selector list short and the property
#: list explicit is deliberate: a full-DOM diff would drown a real regression
#: in the attributes the browser rewrites on serialisation.
_INVENTORY = """
() => {
  const props = ['color','background-color','font-size','font-family','font-weight',
                 'display','padding-top','margin-top','border-bottom-width',
                 'width','height'];
  const sels = ['#pysuricata-report','.var-card','.var-card__header','.badge',
                '.dtype','.attention-title','table','th','td','.flag','svg'];
  const computed = {};
  for (const sel of sels) {
    const el = document.querySelector(sel);
    if (!el) { computed[sel] = null; continue; }
    const cs = getComputedStyle(el);
    computed[sel] = Object.fromEntries(props.map(p => [p, cs.getPropertyValue(p)]));
  }
  const fav = document.querySelector('link[rel="icon"][href^="data:image"]');
  return {
    styles: Array.from(document.querySelectorAll('style')).map(s => s.textContent),
    scripts: Array.from(document.querySelectorAll('script')).map(s => s.textContent),
    favicon: !!fav,
    computed,
    counts: {
      cards: document.querySelectorAll('.var-card').length,
      svgs: document.querySelectorAll('svg').length,
      tables: document.querySelectorAll('table').length,
      flags: document.querySelectorAll('.flag').length,
    },
  };
}
"""
