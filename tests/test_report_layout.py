"""#124 — the redesign's acceptance criteria, executed instead of read.

Every redesign issue ends with numeric acceptance lines. They are already
phrased as assertions; this runs them. Two things had to change on the way,
both discovered by measuring rather than by reading the issue:

**The criteria were a specification, not a description.** #124 quotes
`len(html) < 400_000` and `elements_per_card < 400`. When this file was written
the Titanic report was 600,491 bytes and its widest card held 843 elements.
Those numbers are what #39 and #206 are open to deliver; asserting them now
would ship a red suite that someone disables on Monday. They ship here as
**ratchets** against a recorded baseline, the idiom `test_colour_tokens.py`
already uses: growth fails immediately, and shrinking fails loudly asking for
the baseline to come down. The number only goes one way.

It has already gone down once. #206's first pass took the report to 573,809 by
moving `vector-effect` out of every mark and trimming bar coordinates to two
decimals, and the ratchet is what noticed — it failed with *"lower the baseline
to lock the win in"*, which is the branch that exists so a saving cannot be
quietly spent again.

**Three criteria appear to fail and do not.** Measuring the obvious way says
the header is 53px against a ≤52px budget, that it is 49px against ≤48px on
mobile, and that the icon buttons are 30×30 against a 44×44 minimum. All three
are artifacts of measuring the wrong box:

* the header computes to exactly `height: 52px` (and `48px`) and carries a 1px
  bottom border, which `getBoundingClientRect()` counts and the budget does not;
* `.icon-btn` draws at 30×30 and extends its hit area with an absolutely
  positioned 44×44 `::after`. `elementFromPoint` six pixels outside the visible
  box still returns the button, so a finger lands on 44×44 even though the
  element's own rect is smaller.

A target-size check that reads `getBoundingClientRect()` therefore reports a
failure that does not exist. `_hit_box()` below unions the element's rect with
any absolutely positioned pseudo-element, which is how this report reaches
44×44 without drawing a 44px control.

The same care applies to horizontal scroll. Asking for `scrollWidth >
clientWidth` names nine elements at 1240px — `sr-only` clips, `icon-btn`'s
`::after` overflows, an SVG reports an animated string for `className`. None of
them scrolls. A pane scrolls only if its content overflows **and** its
`overflow-x` is `auto` or `scroll`; by that definition there is exactly one,
the sample table, at every breakpoint, and the document never overflows at all.

The browser cases are marked `browser` and skip when Playwright is absent, so
`uv run pytest` stays a pure-Python run. CI installs Chromium and runs them in
their own job.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pysuricata import profile

REPO = Path(__file__).resolve().parents[1]
TITANIC = REPO / "docs" / "assets" / "titanic.csv"

BREAKPOINTS = (390, 768, 1240)
THEMES = ("light", "dark")


@pytest.fixture(scope="module")
def report_html() -> str:
    """One report, reused. `seed` fixes the reservoir so byte counts are stable."""
    return profile(pd.read_csv(TITANIC), seed=0).html


def _without_script_and_style(doc: str) -> str:
    """The report inlines its own CSS and JS, so a search over the whole
    document finds a class name in the very source that references it. Strip
    both before asserting anything about markup."""
    return re.sub(
        r"<script\b.*?</script>|<style\b.*?</style>", "", doc, flags=re.S | re.I
    )


# --------------------------------------------------------------------------- #
# 1. self-containment
# --------------------------------------------------------------------------- #
class TestTheReportIsOneFile:
    """The single-file promise is the constraint most likely to be broken by
    accident, and until now nothing enforced it. The design comps broke it
    themselves: `Report Screen.dc.html` pulls React from unpkg and renders an
    empty page offline."""

    # Each pattern is a way a document can reach the network for something it
    # needs to render. `href="https:` alone is not one -- the report links out
    # to the docs site on purpose -- so only stylesheet hrefs count.
    FORBIDDEN = {
        "a remote script": r'src\s*=\s*"https?:',
        "a remote stylesheet by href": r'href\s*=\s*"https?:[^"]*\.css',
        "a stylesheet link element": r'rel\s*=\s*"[^"]*stylesheet',
        "a CSS @import": r"@import",
        "a CSS url() over the network": r"url\(\s*['\"]?https?:",
        "a remote iframe or image": r'<(?:iframe|img)[^>]*\ssrc\s*=\s*"https?:',
    }

    @pytest.mark.parametrize("what,pattern", sorted(FORBIDDEN.items()))
    def test_it_fetches_nothing(self, report_html, what, pattern):
        found = re.findall(pattern, report_html, re.I)

        assert not found, (
            f"the report contains {what} ({len(found)} occurrence(s)), so it no "
            f"longer renders offline and is not a single file"
        )

    def test_the_only_binary_payload_is_the_favicon(self, report_html):
        """#111 replaced the base64 PNG logo with inline SVG, and #39 measured
        564 KB riding on it. Naming the one payload that may remain means the
        raster cannot come back unnoticed."""
        kinds = set(re.findall(r"data:([^;]+);base64,", report_html))

        assert kinds <= {"image/x-icon"}, (
            f"unexpected embedded binary payloads: {sorted(kinds - {'image/x-icon'})}"
        )


# --------------------------------------------------------------------------- #
# 2. budgets, as ratchets
# --------------------------------------------------------------------------- #
#: Measured on the Titanic report. #124 wants 400,000; #39 is the issue that
#: gets there. Lower this when it drops -- the test says so when it does, and
#: #206 is the first time it did: 601,000 -> 574,000, by moving `vector-effect`
#: out of every mark and trimming coordinates to two decimals.
#:
#: 574,000 -> 500,000 by not shipping the stylesheet's comments. The report
#: inlines its own CSS, so all 545 of them were going out with every report:
#: **74,036 bytes, 33% of the inlined stylesheet and 12.9% of the document.**
#: They stay in `static/css/`, which is the only place anyone reads them.
#:
#: Found by the ratchet rather than by looking for it. A datetime-chart change
#: added 907 bytes of CSS and pushed the report 578 over, which is a fair thing
#: to be stopped by -- and the honest fix was not to write shorter comments.
#:
#: 500,000 -> 489,000 by not shipping the *scripts'* comments either. The same
#: argument and the same measurement, on the half that had been left out:
#: **15,551 bytes, 20% of the inlined JavaScript.** #240's print rules and deep
#: links cost 2,667 and were what pushed the report over -- the ratchet refused
#: them, correctly, and the way to pay for a feature turned out to be six times
#: larger than the feature.
#:
#: Set at 488,000 first, against a report measured before #246 and #252
#: landed; the rebase put it 3 bytes over. A baseline wants the headroom
#: of a round number above the measurement, not the measurement itself.
#:
#: 489,000 -> 491,000, and this is the **first rise**. #258's fix costs 1,040
#: bytes and none of it is waste: the log histogram was dropping a bin that
#: straddles zero, so `Fare` drew 372 of its 891 rows, and drawing the other
#: 504 emits **8 more bars** (~956 bytes) plus 84 bytes of caption saying how
#: many rows a log axis still cannot show.
#:
#: Measured before raising it, because "the fix needs the room" is what every
#: regression would say. The two savings that paid for #240 were real
#: redundancy -- comments nobody read. There is no equivalent here: the
#: candidates are `data-col` (10,224 bytes) and `data-pct` (6,944), and both
#: are read by `scripts/report_fingerprint.py` as facts, so removing either
#: deletes facts from the invariance guard rather than bytes from the report.
#: `data-col` was tried once already and put back for exactly that reason.
#:
#: So the rule stands and this is the exception it needs: a budget may not
#: grow to hold more presentation, and a chart that omits 58% of a column is
#: not presentation. #39 still wants 400,000.
#: 491,000 -> 494,000 for the flag reference (phase 4b.2). New content a
#: reader asked for, not drift: six rows saying what each raised flag measures
#: and means, 2,431 bytes, and nothing at all on a frame that raises none.
#:
#: Paid down first, which is the part worth recording. Putting the measure in
#: a `title` on every chip cost **5,548 bytes to say fourteen distinct
#: things** -- 154 copies across the report -- and a tooltip is exactly what
#: 4b.2 removes, being invisible on a phone and absent from paper. Dropping it
#: covered twice what the reference costs; the raise is the residue, mostly
#: the `· limit 20%` now on each chip face.
#: 494,000 -> 498,000 for the toolbar (phase 4b.4): a count on each type tab,
#: the sort control, and the script behind them. Trimmed first, again --
#: `data-missing-pct` is one decimal now and is omitted entirely on a complete
#: column, since the sort reads its absence as zero and most columns in most
#: frames are complete.
#: 498,000 -> 499,000 for the datetime baseline panel (phases 5e.2/5e.3,
#: #291 and #292). Two calendar shares were printed bare -- `Weekend % 27.0`
#: against a flat calendar's 28.6% -- and a percentage nobody can judge is not
#: information. Each is now a bar with a rule at the flat value and a verdict
#: in percentage points. The same argument as #258's raise: a figure the
#: reader cannot read is not presentation.
#:
#: Paid down first, and this one is worth recording because of *who* pays.
#: Titanic has no datetime column, so every byte of this is stylesheet the
#: example report carries and cannot use -- the report inlines its CSS into
#: every document. The first cut cost 2,025 bytes; borrowing `.vstat__cap` and
#: `.vstat__val` for the panel's caption and figure instead of defining a
#: second pair that looked the same, and grouping five near-identical
#: mono/size/colour blocks into one, brought it to 747. The remaining 747 is
#: geometry -- the two-column face, the track, the rule, and two breakpoints.
#:
#: The structural fix is not a smaller panel, it is not shipping datetime CSS
#: to a frame with no datetime column (#306). That is worth far more than this
#: raise and belongs to #39 rather than here.
#:
#: A dead-CSS sweep was tried as the paydown and abandoned deliberately: it
#: named 190 classes as never rendered, and the first one checked -- `dt-svg`
#: -- turned out to be live, carried by the empty-state chart. That is the
#: fixture trap in CLAUDE.md exactly. Absent from one fixture is not dead, and
#: an audit that needs a per-class branch check is its own change.
#: 499,000 -> 499,500 for the even-split rule (phase 5f.2, #296). A rule at
#: `100 / n_levels` beside every level bar: `Embarked`'s S at 72.4% against a
#: 33.3% mark says *dominated by one port* with no arithmetic asked of the
#: reader. Same argument as the two raises above it, and the same device as the
#: flat-calendar rule and the outlier fence -- one reading convention across
#: the report rather than three.
#:
#: **Unlike the #291 raise, the report being weighed is the one that benefits.**
#: Titanic is eight-twelfths categorical, so five of its columns get the mark;
#: the datetime raise bought a panel Titanic cannot render at all. That is the
#: difference between paying for a feature and paying for someone else's.
#:
#: Paid down 505 of 824 bytes before raising, and what went is worth recording
#: because it is the same lesson twice:
#:
#:   * a `<title>` on every rule -- 302 bytes to say one thing per column, and
#:     4b.2 already established that a measure in a tooltip is invisible on a
#:     phone and absent from paper. The value is stated once, visibly, in the
#:     coverage note under the chart.
#:   * `data-even-pct` on every rule -- 147 bytes with **no reader**.
#:     `report_fingerprint.py` keys on `data-pct` and does not match this;
#:     neither the stylesheet nor `functionality.js` mentions it. Checked
#:     rather than assumed, since `data-col` looks equally redundant and is
#:     load-bearing for exactly that file.
#:   * a second decimal on coordinates in a pixel viewBox, i.e. hundredths of
#:     a pixel.
#:
#: The residual 319 is seven `<line>` elements, the CSS rule, and the `of the
#: 891` denominator the coverage note now carries -- which #296 asks for by
#: name, because `Cabin` is 77.1% empty and a coverage figure without its
#: denominator cannot distinguish 5.9% of the non-missing rows from 1.3% of
#: the frame.
#:
#: 499,500 -> 486,500, and this one is a **paydown, not a raise**. #294 asked
#: for a legend and a hover instruction to come out of the per-card missing
#: pane. Following the legend to its renderer found that the pane those strings
#: lived in -- `_build_dataprep_spectrum_visualization`, four near-copies
#: across the card kinds, plus `_generate_missing_insights`,
#: `_render_chunk_visualization` and `_build_simple_missing_distribution` --
#: was **reached by no code path at all**. What ships is one shared renderer in
#: `card_base.py`. The dead markup took 523 lines of stylesheet with it, and
#: the report inlines its CSS into every document, so every report was carrying
#: it.
#:
#: This is the dead-CSS sweep the note above abandoned, done the way that note
#: says it has to be: per class, checked against the renderers rather than
#: against one fixture's output. Three untokenised colours went with it, which
#: `test_colour_tokens.py` noticed and asked to have written down.
#:
#: 486,500 -> 487,000 for the attention block's triage (#149). The block ranked
#: by severity class alone, so Titanic's `Age` (19.9% missing against a 20%
#: limit) sat above `Cabin` (77.1% against the same limit) on the strength of
#: having two chips rather than one. Ranking on `value / threshold` needs those
#: two numbers in this layer, and the 593 bytes are almost entirely the
#: `data-value` and `data-threshold` pairs on eleven chips.
#:
#: The same argument as the #291 and #296 raises above: a figure the reader
#: cannot act on is not presentation. Unlike #291, the report being weighed is
#: the one that benefits -- Titanic raises seven flagged columns, and the
#: reordering is visible on it.
#:
#: 487,000 -> 487,500 on the rebase, and the 81 bytes are not this branch's:
#: #338 landed underneath it and publishes a top-k error bound. Rounded up to
#: leave the headroom the #246/#252 note above asks for, so the next rebase
#: does not spend an afternoon on the same 3 bytes.
#:
#: Net against the 499,500 this branch started from: 12,419 bytes returned.
BYTES_BASELINE = 487_500

#: The widest card. #124 wants 400; #206 ("six pre-rendered histograms are 65%
#: of a numeric column's report bytes") is the issue that gets there.
ELEMENTS_PER_CARD_BASELINE = 850

#: How far below the baseline a measurement may sit before the test insists the
#: baseline be rewritten. Wide enough that ordinary noise does not trip it.
_SLACK = 0.06


def _ratchet(measured: int, baseline: int, what: str, issue: str) -> None:
    assert measured <= baseline, (
        f"{what} grew to {measured:,}, over the {baseline:,} baseline. "
        f"This budget only goes down ({issue})."
    )
    assert measured >= baseline * (1 - _SLACK), (
        f"{what} fell to {measured:,}, well under the {baseline:,} baseline. "
        f"Lower the baseline in this file to lock the win in -- otherwise the "
        f"space is free to be spent again."
    )


class TestTheBudgets:
    def test_the_report_does_not_grow(self, report_html):
        _ratchet(len(report_html), BYTES_BASELINE, "the Titanic report", "#39")

    def test_no_card_grows(self, report_html):
        cards = _cards(report_html)
        assert cards, "no variable cards found -- the markup moved, not a pass"

        widest = max(len(re.findall(r"<[a-zA-Z]", card)) for card in cards)
        _ratchet(widest, ELEMENTS_PER_CARD_BASELINE, "the widest card", "#206")


def _cards(doc: str) -> list[str]:
    return re.findall(
        r'<article[^>]*class="[^"]*var-card[^"]*".*?</article>',
        _without_script_and_style(doc),
        re.S,
    )


# --------------------------------------------------------------------------- #
# 2b. every correlation pair is represented
# --------------------------------------------------------------------------- #
class TestEveryCorrelationPairSurvives:
    """#124 asks that the matrix emit `n(n-1)/2` cells. **There is no matrix.**

    #122 removed the heatmap and #154's 5b.6 replaced it with a per-column
    partners pane, so the criterion as written targets markup that no longer
    exists -- searching for a `corr-cell` finds nothing and a test built on it
    would pass by being vacuous, which is the failure mode this whole file is
    about.

    The invariant behind it survives the redesign, and is stronger in the new
    shape. A matrix names each pair once; the panes name it from **both** sides,
    so the count is `n(n-1)` -- exactly twice the cell count. Measured at n = 3,
    4, 5 and 6, it is 6, 12, 20 and 30. A pair silently dropped, or one column's
    pane missing a partner, breaks the identity.
    """

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_each_pair_appears_from_both_sides(self, n):
        import numpy as np

        rng = np.random.default_rng(0)
        # Correlated on purpose: an all-independent frame is entitled to report
        # nothing, and would make this pass without rendering a single pair.
        common = rng.normal(0, 1, 400)
        frame = pd.DataFrame(
            {f"c{i}": rng.normal(0, 1, 400) + common * 0.3 for i in range(n)}
        )

        html = profile(frame, seed=0).html
        partners = len(re.findall(r'class="corr-partner"', html))

        assert partners == n * (n - 1), (
            f"{n} numeric columns make {n * (n - 1) // 2} pairs, which the "
            f"per-column panes should show twice over as {n * (n - 1)} partner "
            f"entries; found {partners}"
        )


# --------------------------------------------------------------------------- #
# 3. layout, in a browser
# --------------------------------------------------------------------------- #
#: The union of an element's own rect with any absolutely positioned pseudo
#: element. See the module docstring: `.icon-btn` is 30x30 with a 44x44
#: `::after`, and a finger lands on the larger box.
_HIT_BOX = """(el) => {
  const r = el.getBoundingClientRect();
  let w = r.width, h = r.height;
  for (const pseudo of ['::after', '::before']) {
    const cs = getComputedStyle(el, pseudo);
    if (cs.content === 'none' || cs.position !== 'absolute') continue;
    const pw = parseFloat(cs.width), ph = parseFloat(cs.height);
    if (!isNaN(pw)) w = Math.max(w, pw);
    if (!isNaN(ph)) h = Math.max(h, ph);
  }
  return {w, h};
}"""

_MEASURE = (
    """() => {
  const hit = __HIT_BOX__;
  const header = document.querySelector('.report-header') || document.querySelector('header');

  const undersized = [];
  for (const el of document.querySelectorAll('button, a, summary, input, select, [role=button]')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') continue;
    const {w, h} = hit(el);
    if (w < 44 || h < 44) {
      undersized.push({tag: el.tagName, cls: (el.className || '').toString(),
                       w: Math.round(w), h: Math.round(h)});
    }
  }

  // #206 moved `vector-effect` off every mark and into the .bar/.grid/.axis
  // rules. Computed style is the only form of this check that a declaration
  // which never applies cannot satisfy.
  const strokes = {};
  for (const mark of ['bar', 'grid', 'axis']) {
    const el = document.querySelector('.hist-svg .' + mark);
    strokes[mark] = el ? getComputedStyle(el).vectorEffect : null;
  }

  // Overflow is not scrolling. Only a box whose content is wider AND whose
  // overflow-x actually scrolls is a scroll pane.
  const scrollers = [];
  for (const el of document.querySelectorAll('*')) {
    if (el.clientWidth === 0 || el.scrollWidth <= el.clientWidth + 1) continue;
    const ox = getComputedStyle(el).overflowX;
    if (ox === 'auto' || ox === 'scroll') {
      scrollers.push((el.className || '').toString() || el.tagName);
    }
  }

  // #319/#145. A card that is paginated away lays out at a stub height, so
  // `offsetParent` decides what counts; the active `.variant` is the chart the
  // container is actually holding.
  const cards = [];
  for (const card of document.querySelectorAll('.var-card')) {
    if (card.offsetParent === null) continue;
    const badge = card.querySelector('.badge');
    const container = card.querySelector('.var-chart .hist-variants');
    const chart = container ? container.querySelector('.variant.active') : null;
    cards.push({
      id: card.id,
      kind: badge ? badge.textContent.trim() : '?',
      height: Math.round(card.getBoundingClientRect().height),
      container_height:
        container ? Math.round(container.getBoundingClientRect().height) : null,
      chart_height: chart ? Math.round(chart.getBoundingClientRect().height) : null,
    });
  }

  const summary = document.querySelector('#summary');
  return {
    header_height: header ? parseFloat(getComputedStyle(header).height) : null,
    undersized,
    scrollers: [...new Set(scrollers)],
    document_overflow:
      document.documentElement.scrollWidth - document.documentElement.clientWidth,
    summary_height: summary ? Math.round(summary.getBoundingClientRect().height) : null,
    strokes,
    cards,
  };
}"""
    # A plain replace, not %-format or .format(): the script is full of `{...}`
    # object literals and destructuring, which .format() would try to read.
    .replace("__HIT_BOX__", _HIT_BOX)
)


def assert_theme(page, theme: str) -> None:
    """Put the report into `theme`, and prove it moved.

    The report does **not** use `prefers-color-scheme`. Dark is the absence of a
    `light` class on `#pysuricata-report`, flipped by the header's toggle. So
    Playwright's `color_scheme=` argument does nothing here -- with it, all six
    contact-sheet images came out byte-identical in pairs and the theme
    parametrisation below was six runs of the same two states.

    Setting the class is not enough on its own either: if the selector ever
    changes, silently doing nothing would leave every theme assertion passing
    against one theme. So the background colour is read back and required to
    differ between the two.

    That read has to wait. `#pysuricata-report` carries
    `transition: background-color 0.3s`, so the class change flips `--paper` to
    `#1C1A17` immediately while the computed background is still interpolating
    from the old value -- read it at once and both themes report the light paper,
    which is what the check below was written to catch and did.
    """
    root = page.query_selector("#pysuricata-report")
    assert root is not None, "no #pysuricata-report root -- the shell markup moved"

    page.evaluate(
        """([el, theme]) => {
            el.classList.toggle('light', theme === 'light');
            // The standalone body sits outside the report and cannot read its
            // tokens, so it carries the same class and its own two literals.
            document.body.classList.toggle('light', theme === 'light');
        }""",
        [root, theme],
    )
    page.wait_for_timeout(_THEME_TRANSITION_MS)
    background = page.evaluate("""(el) => getComputedStyle(el).backgroundColor""", root)
    _SEEN_BACKGROUNDS.setdefault(theme, background)
    assert _SEEN_BACKGROUNDS[theme] == background, "a theme rendered inconsistently"
    other = _SEEN_BACKGROUNDS.get("light" if theme == "dark" else "dark")
    assert other is None or other != background, (
        f"light and dark both compute to {background} -- the theme is not "
        f"switching, and every theme assertion here is measuring one state twice"
    )


#: Filled by `assert_theme`, so the two themes can be required to differ.
_SEEN_BACKGROUNDS: dict[str, str] = {}

#: `transition: background-color 0.3s` on `#pysuricata-report`, plus margin.
_THEME_TRANSITION_MS = 400


def _chrome() -> str | None:
    import os

    if explicit := os.environ.get("PYSURICATA_CHROME"):
        return explicit
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return None


@pytest.fixture(scope="module")
def measurements(report_html, tmp_path_factory):
    """The report measured at every breakpoint and theme, once."""
    playwright = pytest.importorskip(
        "playwright.sync_api", reason="browser layout checks need Playwright"
    )

    page_file = tmp_path_factory.mktemp("layout") / "report.html"
    page_file.write_text(report_html, encoding="utf-8")

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
            for theme in THEMES:
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.goto(page_file.as_uri())
                assert_theme(page, theme)
                # The report draws its charts on load.
                page.wait_for_timeout(900)
                out[(width, theme)] = page.evaluate(_MEASURE)
                page.close()
        browser.close()
    return out


def _every_kind() -> pd.DataFrame:
    """A frame carrying all four card kinds, which Titanic cannot: it has no
    datetime column, so the only kind with a `time_span` has never been
    measured. Same 891 rows and same generator as the invariance suite's
    `_frame`, trimmed to one column per kind plus a second numeric."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 891
    return pd.DataFrame(
        {
            "age": rng.integers(1, 80, n).astype(float),
            "fare": rng.gamma(2, 20, n),
            "sex": rng.choice(["male", "female"], n),
            "cabin": rng.choice([None, "C85", "B42"], n, p=[0.77, 0.12, 0.11]),
            "survived": rng.integers(0, 2, n).astype(bool),
            "booked": pd.date_range("2026-01-01", periods=n, freq="h"),
        }
    )


@pytest.fixture(scope="module")
def kind_measurements(tmp_path_factory):
    """`_every_kind()` measured at every breakpoint, once.

    Light theme only. `test_the_themes_do_not_change_the_layout` already
    asserts that a theme moves no box, so measuring both here would buy a
    second reading of the same geometry at twice the browser time.
    """
    playwright = pytest.importorskip(
        "playwright.sync_api", reason="browser layout checks need Playwright"
    )

    page_file = tmp_path_factory.mktemp("kinds") / "report.html"
    page_file.write_text(profile(_every_kind(), seed=0).html, encoding="utf-8")

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
            page = browser.new_page(viewport={"width": width, "height": 844})
            page.goto(page_file.as_uri())
            page.wait_for_timeout(900)
            out[width] = page.evaluate(_MEASURE)["cards"]
            page.close()
        browser.close()
    return out


#: #111. The header is 48px on mobile and 52px from the first breakpoint up,
#: plus a 1px rule that the budget does not count.
_HEADER_BUDGET = {390: 48, 768: 48, 1240: 52}

#: #122's allow-list. Without one the no-horizontal-scroll assertion is
#: unpassable: the sample table is 12 Titanic columns and is *meant* to scroll.
_MAY_SCROLL = {"sample-scroll"}

#: Interactive targets still under 44x44, recorded so no new one can appear.
#: All three are inline links inside a sentence, which WCAG 2.5.8 exempts
#: precisely because padding them out would wreck the line box: the header's
#: version link, and the two in the footer's "Built with pysuricata, developed
#: by alvarodiez20". The third arrived with the author credit -- a ratchet is
#: meant to make exactly that visible, and it did. The desktop nav links are
#: 31px tall and are *not* exempt -- they are counted below.
_INLINE_LINK_EXEMPTIONS = 3

#: Desktop-only nav links, which are 31px tall. A real gap against #111's line,
#: not an exemption -- recorded so it cannot grow while it waits for a fix.
#: 1240 is the six nav links plus the three exempt inline ones.
_KNOWN_UNDERSIZED = {390: 3, 768: 3, 1240: 9}

#: #112 wants the summary under 560px at 390px. It is 624px. Recorded, not
#: waived.
#:
#: 620/575/340 -> 624/579/344 for the section-heading system (#298). Summary
#: was the only one of the five sections whose heading opted out of
#: `.section-title`, carrying a near-copy rule in `_03-summary.css` instead --
#: same size, same line height, an 8px bottom margin against the system's
#: 12px. Joining the system costs exactly those 4px, at all three widths.
#:
#: The raise is against this ratchet's own direction, so the reason matters:
#: this is not a new decision about spacing, it is the *removal* of an
#: undecided one. `docs/internal/design/Report Baseline.dc.html:87` draws the
#: Summary heading at `margin: 0 0 0.75rem`, which is exactly
#: `.section-title`. The shipped 0.5rem was drift away from the design, and
#: the heights recorded here were measuring the drift.
#:
#: **+4 and not more.** These numbers are CI's, and CI is the gate. Two
#: developer machines measure this summary 2-7px taller than the recorded
#: values *before any change* -- 342/580/627 against 340/575/620, at
#: eb0523b with nothing applied -- so a baseline set from a local reading
#: would have carried that machine's font rendering into the budget as
#: permanent slack, and the ratchet would have stopped catching the next 7px.
#: The change was therefore measured as a **delta** on one machine, which is
#: environment-independent, and added to what CI records. It is +4px at every
#: width, which is the margin and nothing else.
#:
#: The consequence is that these six cases fail locally on a machine that
#: reads high, and pass in CI. That split predates this change and is #309.
#:
#: The 64px still between 624 and #112's 560 is unaffected by this and is
#: where the work remains.
_SUMMARY_BASELINE = {390: 624, 768: 579, 1240: 344}

#: #145 — a height criterion for each of the four card kinds, which only the
#: numeric card had. Measured on `_every_kind()` at 844px tall, details
#: collapsed. Pinned to a dataset and a viewport, which is the thing #112 and
#: #114 did not do and why their original figures could not be reproduced.
#:
#: The premise #145 was filed on no longer holds: it recorded categorical as
#: the tallest kind at 923px against numeric's 820px. Categorical is now the
#: *shortest* of the three non-boolean kinds (480px at 390px) because #308
#: suppressed the statistics it could not say, and #319 took the padded 180px
#: chart container out from under it. Numeric and datetime are what run tall.
#:
#: These are a developer machine's readings, and #309 records that such a
#: machine measures 2-7px *taller* than CI. That direction is the safe one for
#: an upper bound -- CI reads under it, and by well under the 6% `_SLACK`, so
#: the lower branch does not fire either.
_CARD_HEIGHT_BASELINE = {
    390: {"Numeric": 883, "Categorical": 480, "Boolean": 343, "Datetime": 868},
    768: {"Numeric": 803, "Categorical": 449, "Boolean": 332, "Datetime": 849},
    1240: {"Numeric": 578, "Categorical": 405, "Boolean": 314, "Datetime": 551},
}


@pytest.mark.browser
@pytest.mark.parametrize("width", BREAKPOINTS)
class TestEveryCardKindHasAHeightCriterion:
    """#145. Three of the four kinds had no recorded expectation at all, so
    nothing distinguished "tall because it has more to say" from "tall because
    something regressed"."""

    @pytest.mark.parametrize("kind", sorted(_CARD_HEIGHT_BASELINE[390]))
    def test_no_card_kind_gets_taller(self, kind_measurements, width, kind):
        cards = [c for c in kind_measurements[width] if c["kind"] == kind]

        assert cards, (
            f"no {kind} card rendered at {width}px -- the fixture no longer "
            f"covers this kind, so its criterion is measuring nothing"
        )
        tallest = max(cards, key=lambda c: c["height"])
        _ratchet(
            tallest["height"],
            _CARD_HEIGHT_BASELINE[width][kind],
            f"the tallest {kind} card at {width}px ({tallest['id']})",
            "#145",
        )


@pytest.mark.browser
@pytest.mark.parametrize("width", BREAKPOINTS)
def test_no_chart_container_reserves_height_its_chart_does_not_use(measurements, width):
    """#319. A fixed `height` on `.hist-variants` under the mobile breakpoint
    forced every kind to 180px, which a two-level categorical chart (23px)
    padded by 157px and a numeric one (213px) overflowed by 33px. The chart is
    `height: auto` and sizes to the viewBox the renderer computed, so the
    container must take the height the chart asks for -- in both directions."""
    for card in measurements[(width, "light")]["cards"]:
        if card["container_height"] is None or card["chart_height"] is None:
            continue

        assert card["container_height"] == card["chart_height"], (
            f"{card['id']} ({card['kind']}) reserves "
            f"{card['container_height']}px at {width}px for a "
            f"{card['chart_height']}px chart"
        )


@pytest.mark.browser
@pytest.mark.parametrize("width", BREAKPOINTS)
@pytest.mark.parametrize("theme", THEMES)
class TestLayoutAtEveryBreakpoint:
    def test_the_header_keeps_its_budget(self, measurements, width, theme):
        """#111. Measured from the computed height, not the border box."""
        got = measurements[(width, theme)]["header_height"]

        assert got == _HEADER_BUDGET[width], (
            f"header is {got}px at {width}px, budget is {_HEADER_BUDGET[width]}px"
        )

    def test_the_page_never_scrolls_sideways(self, measurements, width, theme):
        """#122. The document itself must fit, at every width, in both themes."""
        overflow = measurements[(width, theme)]["document_overflow"]

        assert overflow <= 0, (
            f"the page overflows by {overflow}px at {width}px ({theme})"
        )

    def test_only_the_allowed_panes_scroll(self, measurements, width, theme):
        """A scroll pane is content wider than its box *and* an overflow-x that
        scrolls. Anything else is clipped or visible overflow."""
        scrollers = set(measurements[(width, theme)]["scrollers"])

        assert scrollers <= _MAY_SCROLL, (
            f"unexpected horizontal scroll panes at {width}px: "
            f"{sorted(scrollers - _MAY_SCROLL)}"
        )

    def test_no_new_undersized_target_appears(self, measurements, width, theme):
        """#111/#122 asked for 44x44 everywhere. Measured against the *hit* box,
        which includes the 44x44 `::after` the icon buttons extend themselves
        with -- otherwise every icon button reports a failure it does not have."""
        undersized = measurements[(width, theme)]["undersized"]

        assert len(undersized) <= _KNOWN_UNDERSIZED[width], (
            f"{len(undersized)} interactive targets are under 44x44 at {width}px, "
            f"up from the recorded {_KNOWN_UNDERSIZED[width]}: "
            f"{[(u['tag'], u['cls'], u['w'], u['h']) for u in undersized]}"
        )
        assert len(undersized) >= _KNOWN_UNDERSIZED[width], (
            f"only {len(undersized)} targets are now under 44x44 at {width}px, "
            f"down from {_KNOWN_UNDERSIZED[width]}. Lower the baseline."
        )

    @pytest.mark.parametrize("mark", ["bar", "grid", "axis"])
    def test_every_histogram_mark_keeps_a_hairline(
        self, measurements, width, theme, mark
    ):
        """#147's invariant, after #206 moved the declaration into CSS.

        A viewBox unit is a percent of the plot, so a 1-unit stroke is 11px at
        1,100px and 0.28px at 284px -- which is how the bars once merged into
        one block. `non-scaling-stroke` is what makes a hairline a hairline,
        and reading it from computed style is the check that survives the
        declaration moving from an attribute to a rule.
        """
        got = measurements[(width, theme)]["strokes"][mark]

        assert got == "non-scaling-stroke", (
            f".{mark} computes vector-effect={got!r} at {width}px, so its "
            f"stroke scales with the plot"
        )

    def test_the_summary_does_not_get_taller(self, measurements, width, theme):
        """#112 wants ≤560px at 390px; it is 620px. A ratchet, not a waiver."""
        got = measurements[(width, theme)]["summary_height"]

        assert got <= _SUMMARY_BASELINE[width], (
            f"the summary grew to {got}px at {width}px, over the recorded "
            f"{_SUMMARY_BASELINE[width]}px"
        )


@pytest.mark.browser
def test_the_themes_do_not_change_the_layout(measurements):
    """Dark mode swaps tokens, not geometry. If a theme moves a box, a colour
    token is being used for something that is not colour."""
    for width in BREAKPOINTS:
        light, dark = measurements[(width, "light")], measurements[(width, "dark")]
        assert light["header_height"] == dark["header_height"], width
        assert light["summary_height"] == dark["summary_height"], width
        assert len(light["undersized"]) == len(dark["undersized"]), width


# --------------------------------------------------------------------------- #
# Pagination must not take a card out of the document's reach
# --------------------------------------------------------------------------- #
@pytest.mark.browser
class TestAnOffPageCardIsStillReachable:
    """#240, then design 15d. Pagination hid the eleventh column onward with
    `display: none`, which is not a rendering choice but a removal: a browser
    find cannot match inside it, an anchor cannot land on it, and a printer
    will not print it.

    Nothing is hidden now. A column past the limit keeps its row and folds its
    body, so all four consequences go together -- including find, which the
    first pass at this could not fix because it left the hiding in place.
    """

    @staticmethod
    def _page(playwright, browser, report_html, tmp_path, hash_fragment=""):
        page_file = tmp_path / "report.html"
        page_file.write_text(report_html, encoding="utf-8")
        page = browser.new_page(viewport={"width": 1240, "height": 900})
        page.goto(page_file.as_uri() + hash_fragment)
        page.wait_for_timeout(700)
        return page

    @pytest.fixture(scope="class")
    def browser(self):
        playwright = pytest.importorskip(
            "playwright.sync_api", reason="this needs Playwright"
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

    def test_every_attention_link_resolves_to_a_card(self, report_html):
        """The block exists to be clicked. A link naming a column that is not
        in the document is the failure this cannot be allowed to have."""
        body = _without_script_and_style(report_html)
        targets = set(re.findall(r'class="attention-col" href="#([^"]+)"', body))
        if not targets:
            pytest.skip("this frame raised no quality flags")
        ids = set(re.findall(r'<article class="var-card" id="([^"]+)"', body))
        assert targets <= ids, sorted(targets - ids)

    def test_a_link_to_an_off_page_card_reveals_it(
        self, browser, report_html, tmp_path
    ):
        page = self._page(None, browser, report_html, tmp_path)
        hidden = page.evaluate(
            "[...document.querySelectorAll('#cards-grid .var-card')]"
            ".filter(c => c.classList.contains('is-collapsed')).map(c => c.id)"
        )
        if not hidden:
            pytest.skip("every column in this frame is expanded")
        target = hidden[-1]
        page.evaluate(
            "id => document.querySelector(`a[href='#${id}']`)?.click()"
            " ?? (location.hash = id)",
            target,
        )
        page.wait_for_timeout(400)
        assert page.evaluate(
            "id => document.getElementById(id).getBoundingClientRect().height > 0",
            target,
        ), f"the link to {target} did not open it"
        page.close()

    def test_a_deep_link_opens_on_the_right_page(self, browser, report_html, tmp_path):
        page = self._page(None, browser, report_html, tmp_path)
        hidden = page.evaluate(
            "[...document.querySelectorAll('#cards-grid .var-card')]"
            ".filter(c => c.classList.contains('is-collapsed')).map(c => c.id)"
        )
        page.close()
        if not hidden:
            pytest.skip("every column in this frame is expanded")
        target = hidden[-1]
        fresh = self._page(None, browser, report_html, tmp_path, f"#{target}")
        assert fresh.evaluate(
            "id => document.getElementById(id).getBoundingClientRect().height > 0",
            target,
        ), f"opening the report at #{target} left it folded"
        fresh.close()

    def test_print_unfolds_every_card(self, browser, report_html, tmp_path):
        """The worst of the four: a 60-column profile exported as 10 columns
        with nothing saying so. Read by re-targeting the print media query at
        the screen, which exercises the real cascade rather than the rule text.

        A folded card is now *in* the printed document, so what print has to do
        is unfold it -- a sheet of header rows answers less than the cards do,
        and paper has no affordance to expand one.
        """
        page = self._page(None, browser, report_html, tmp_path)
        result = page.evaluate("""() => {
  const cards = [...document.querySelectorAll('#cards-grid .var-card')];
  // A card counts as printed only when its *body* is laid out -- a folded
  // header has a box too, and counting boxes would pass on a page of headers.
  const body = c => [...c.children].find(x => !x.classList.contains('var-card__header'));
  const unfolded = () => cards.filter(c => {
    const b = body(c);
    return b && getComputedStyle(b).display !== 'none';
  }).length;
  const onScreen = unfolded();
  let rule = null;
  for (const s of [...document.styleSheets]) {
    let rules; try { rules = s.cssRules; } catch { continue; }
    for (const r of rules) {
      if (r.type === CSSRule.MEDIA_RULE && r.conditionText.includes('print')) { rule = r; break; }
    }
    if (rule) break;
  }
  if (!rule) return {error: 'no @media print block ships with the report'};
  rule.media.mediaText = 'screen';
  const onPaper = unfolded();
  const rail = document.getElementById('collapsed-rail');
  const controls = rail ? getComputedStyle(rail).display : 'none';
  rule.media.mediaText = 'print';
  return {onScreen, onPaper, total: cards.length, controls};
}""")
        page.close()
        assert "error" not in result, result.get("error")
        assert result["onPaper"] == result["total"], (
            f"{result['onPaper']} of {result['total']} cards would print in full"
        )
        assert result["onScreen"] < result["total"], (
            "nothing was folded on screen, so this proved nothing about print"
        )
        assert result["controls"] == "none", (
            "the expand control prints as an affordance the reader cannot use"
        )

    def test_no_column_is_ever_removed_from_the_document(
        self, browser, report_html, tmp_path
    ):
        """The property the whole mechanism exists for, and the one the first
        attempt could not deliver.

        A browser find matches rendered text. It cannot see into a
        `display: none` subtree, so paging a column away made it unfindable —
        and finding a column by name is the primary action in a profiling
        report. Folding keeps the name, the type and the flags laid out; only
        the charts, which nobody searches for, go.

        Filtering is the one case that still removes a card, and that is the
        intent: a reader who filtered a column away is asking not to see it.
        """
        page = self._page(None, browser, report_html, tmp_path)
        result = page.evaluate("""() => {
  const cards = [...document.querySelectorAll('#cards-grid .var-card')];
  const laidOut = c => c.getBoundingClientRect().height > 0;
  const nameShown = c => {
    const n = c.querySelector('.colname');
    return n && n.getBoundingClientRect().height > 0;
  };
  return {
    total: cards.length,
    laidOut: cards.filter(laidOut).length,
    namesVisible: cards.filter(nameShown).length,
    folded: cards.filter(c => c.classList.contains('is-collapsed')).length,
    inlineDisplayNone: cards.filter(c => c.style.display === 'none').length,
  };
}""")
        page.close()
        assert result["laidOut"] == result["total"], (
            f"{result['total'] - result['laidOut']} cards have no box, so a "
            "browser find cannot reach them"
        )
        assert result["namesVisible"] == result["total"], (
            "a column name is not laid out, which is the text a reader searches for"
        )
        assert result["inlineDisplayNone"] == 0, (
            "a card is hidden with an inline display:none — the mechanism 15d replaced"
        )
        assert result["folded"] > 0, "nothing was folded, so this frame proved nothing"


@pytest.mark.browser
class TestTheToolbarSaysWhatItIsShowing:
    """Design 15c, phase 4b.4. Three separate mechanisms narrow the variables
    list — a search box, a type tab and the collapse limit — and the toolbar
    described none of them. `Showing 1-10 of 12` describes a page, and there
    are no pages.
    """

    @pytest.fixture(scope="class")
    def page(self, request):
        playwright = pytest.importorskip(
            "playwright.sync_api", reason="this needs Playwright"
        )
        html = profile(pd.read_csv(TITANIC), seed=0).html
        tmp = Path(tempfile.mkdtemp()) / "r.html"
        tmp.write_text(html, encoding="utf-8")
        launch = {}
        if chrome := _chrome():
            launch["executable_path"] = chrome
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(**launch)
            except Exception as exc:
                pytest.skip(f"Chromium is not available: {exc}")
            page = browser.new_page(viewport={"width": 1240, "height": 900})
            page.goto(tmp.as_uri())
            page.wait_for_timeout(700)
            yield page
            browser.close()

    def test_no_tab_for_a_type_with_no_columns(self, page):
        """Titanic has no datetime columns and used to get a Datetime tab that
        filtered to an empty grid with nothing saying why — the same defect as
        a zero-width donut segment or a one-option Top-N chooser."""
        tabs = page.evaluate(
            "[...document.querySelectorAll('.tab')].map(t => t.dataset.filter)"
        )
        counts = page.evaluate("""() => {
  const out = {};
  for (const c of document.querySelectorAll('#cards-grid .var-card')) {
    out[c.dataset.type] = (out[c.dataset.type] || 0) + 1;
  }
  return out;
}""")
        for tab in tabs:
            if tab == "all":
                continue
            assert counts.get(tab, 0) > 0, f"a {tab} tab with no {tab} columns"
        for kind, n in counts.items():
            if n:
                assert kind in tabs, f"{n} {kind} columns and no tab for them"

    def test_each_tab_carries_its_count(self, page):
        labels = page.evaluate(
            "[...document.querySelectorAll('.tab')].map(t => t.textContent)"
        )
        assert all(re.search(r"\d+$", t) for t in labels), labels

    def test_the_count_sentence_is_gone(self, page):
        """It duplicated the Summary composition bar, and printed `0 datetime`
        for a type with no columns."""
        text = page.evaluate("document.getElementById('pysuricata-report').innerText")
        assert "Analyzing" not in text, "the count sentence is back"

    def test_one_line_covers_filter_search_and_collapse(self, page):
        unfiltered = page.evaluate(
            "document.getElementById('pagination-info').textContent"
        )
        assert "expanded" in unfiltered and "collapsed" in unfiltered, unfiltered
        assert "Showing 1-" not in unfiltered, "still describing a page"

        page.evaluate(
            """document.querySelector('.tab[data-filter="numeric"]').click()"""
        )
        filtered = page.evaluate(
            "document.getElementById('pagination-info').textContent"
        )
        assert "numeric" in filtered, filtered
        assert page.evaluate("!document.getElementById('clear-filter').hidden"), (
            "no way to clear a filter that is narrowing the list"
        )

        page.evaluate("document.getElementById('clear-filter').click()")
        assert page.evaluate("document.getElementById('clear-filter').hidden"), (
            "clear-filter stays visible with nothing to clear"
        )

    def test_sorting_reorders_the_document_and_returns(self, page):
        """The cards *are* the document, so a sort has to move them — that is
        what makes the order true for a browser find and for print, rather than
        only for the script's own bookkeeping. Dataset order is the default and
        has to come back exactly."""
        names = "[...document.querySelectorAll('#cards-grid .var-card')].map(c => c.dataset.name)"
        original = page.evaluate(names)

        def sort(value):
            page.evaluate(
                f"() => {{ const s = document.getElementById('sort-select');"
                f" s.value = '{value}'; s.dispatchEvent(new Event('change')); }}"
            )
            return page.evaluate(names)

        by_name = sort("name")
        assert by_name == sorted(original), by_name

        by_missing = sort("missing")
        assert by_missing[0] == "Cabin", by_missing[:3]

        assert sort("dataset") == original, "dataset order did not come back"
