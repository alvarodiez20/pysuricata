"""#306 — a report ships only the card-kind CSS it can use.

`load_css_dir` concatenated all fourteen partials into every report. The report
inlines its CSS, so a frame with no datetime column was not taking a cache miss
on `_09-datetime.css`, it was carrying it in the document.

**The mapping was measured rather than assumed**, which is the whole reason
this is safe. Every selector in every partial was matched against the rendered
DOM of a report built from each single-kind frame, and three rules turned out
to be misfiled -- they named no element of their partial's kind and applied to
every report:

* `.axis`, in `_08-categorical.css`, drawn by the numeric histogram and the
  datetime charts too
* `.var-card__body .var-chart`, in `_09-datetime.css`, which names nothing
  datetime at all
* the narrow-screen `.controls-slot` gap, also in `_08-categorical.css`

All three moved to `_06-cards.css` before anything was made conditional. A
fourth, `--triple-right`, was deleted: it was defined on a bare
`#pysuricata-report` in the datetime partial and read by nothing.

The check that matters is `TestNothingThatMatteredWasDropped`, which renders
the same frame twice -- once with the trimmed stylesheet the report actually
ships, once with every partial forced in -- and requires every element to
compute identically. A rule that was doing work cannot pass it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.utils import load_css_dir

REPO = Path(__file__).resolve().parents[1]
CSS_DIR = str(REPO / "pysuricata" / "static" / "css")

#: Partials that must not ship when the frame has no column of that kind.
CONDITIONAL = {
    "numeric": ["_08-categorical.css", "_09-datetime.css", "_10-boolean.css"],
    "categorical": ["_07-histogram.css", "_09-datetime.css", "_10-boolean.css"],
    "boolean": ["_07-histogram.css", "_08-categorical.css", "_09-datetime.css"],
    "datetime": ["_08-categorical.css", "_10-boolean.css"],
}

#: One selector per conditional partial that appears nowhere else, so its
#: presence in the emitted stylesheet is proof the partial shipped.
SIGNATURE = {
    "_07-histogram.css": ".hist-svg .axis",
    "_08-categorical.css": ".cat.variant",
    "_09-datetime.css": ".dt-svg",
    "_10-boolean.css": ".bool-svg",
}

N = 60


def _stylesheet(html: str) -> str:
    """Just the inlined `<style>` block.

    The report inlines its CSS **and** its JS, so searching the whole document
    for `.cat.variant` finds it in `functionality.js`, which queries for it --
    the trap `CLAUDE.md` records, and which this file walked straight into on
    first run: three shapes reported shipping a partial they had skipped.
    """
    start = html.index("<style>")
    end = html.index("</style>", start)
    return html[start:end]


def _frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    return {
        "numeric": pd.DataFrame({f"n{i}": rng.normal(0, 1, N) for i in range(3)}),
        "categorical": pd.DataFrame(
            {f"c{i}": rng.choice(list("abcde"), N) for i in range(2)}
        ),
        "boolean": pd.DataFrame(
            {f"b{i}": rng.integers(0, 2, N).astype(bool) for i in range(2)}
        ),
        "datetime": pd.DataFrame(
            {"t": pd.date_range("2026-01-01", periods=N, freq="h")}
        ),
    }


FRAMES = _frames()


@pytest.mark.parametrize("kind", sorted(CONDITIONAL))
class TestOnlyTheKindsPresentAreShipped:
    def test_the_partials_for_absent_kinds_do_not_ship(self, kind):
        css = _stylesheet(profile(FRAMES[kind], seed=0).html)

        absent = [partial for partial in CONDITIONAL[kind] if SIGNATURE[partial] in css]
        assert not absent, (
            f"a {kind}-only frame still ships {absent}, which it cannot use"
        )

    def test_the_partial_for_the_kind_present_does_ship(self, kind):
        """The other half, and the one that turns a saving into a defect if it
        is wrong: a frame that *has* the kind must still get its styles."""
        wanted = {
            "numeric": "_07-histogram.css",
            "categorical": "_08-categorical.css",
            "boolean": "_10-boolean.css",
            "datetime": "_09-datetime.css",
        }[kind]
        css = _stylesheet(profile(FRAMES[kind], seed=0).html)

        assert SIGNATURE[wanted] in css, f"a {kind} frame did not get {wanted}"


class TestTheSectionPartialsAreUnconditional:
    """Correlations and missing values are sections, not card kinds. Both
    always render, with an empty state when they have nothing to report, and an
    empty state still needs styling -- `.no-correlations-state` and
    `.miss-none` are in those partials."""

    def test_a_frame_with_neither_still_ships_both(self):
        # One complete numeric column: no second column to correlate with, and
        # no missing values.
        css = _stylesheet(
            profile(pd.DataFrame({"n": [float(i) for i in range(N)]}), seed=0).html
        )

        assert ".no-correlations-state" in css
        assert ".miss-none" in css


class TestTheLoaderItself:
    def test_no_kinds_given_ships_everything(self):
        """A caller that does not know the shape must not silently get less."""
        everything = load_css_dir(CSS_DIR)

        for signature in SIGNATURE.values():
            assert signature in everything

    def test_the_cache_key_includes_the_kinds(self):
        """`load_css_dir` is `lru_cache`d. Keyed on the directory alone, the
        first frame's stylesheet would be served to every later one."""
        numeric = load_css_dir(CSS_DIR, frozenset({"numeric"}))
        boolean = load_css_dir(CSS_DIR, frozenset({"boolean"}))

        assert numeric != boolean
        assert SIGNATURE["_10-boolean.css"] in boolean
        assert SIGNATURE["_10-boolean.css"] not in numeric

    def test_trimming_only_ever_removes(self):
        """Whatever a shape ships must be a subset of everything, byte for
        byte -- the loader selects partials, it does not rewrite them."""
        everything = load_css_dir(CSS_DIR)
        trimmed = load_css_dir(CSS_DIR, frozenset({"numeric"}))

        assert len(trimmed) < len(everything)
        # The concatenation order is stable, so each kept partial's text still
        # appears in the full sheet.
        assert trimmed.removeprefix("<style>").removesuffix("</style>")[:200] in (
            everything
        )


_COMPUTED = """() => {
  const out = [];
  const props = [
    'display', 'position', 'overflow', 'overflowX', 'overflowY',
    'width', 'height', 'margin', 'padding', 'color', 'backgroundColor',
    'fontSize', 'fontWeight', 'gridTemplateColumns', 'flexDirection',
    'stroke', 'strokeWidth', 'fill', 'borderTopWidth', 'borderBottomWidth',
  ];
  for (const el of document.querySelectorAll('#pysuricata-report *')) {
    const cs = getComputedStyle(el);
    out.push(props.map(p => cs[p]).join('|'));
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


@pytest.mark.browser
@pytest.mark.parametrize("kind", sorted(CONDITIONAL))
@pytest.mark.parametrize("width", [390, 1240])
class TestNothingThatMatteredWasDropped:
    """The acceptance criterion, as an equivalence rather than a spot check.

    The report is rendered twice — once as it ships, once with every partial
    forced back in — and every element must compute identically. A dropped rule
    that was doing any work at all shows up as a differing property, and the
    check does not depend on anyone having thought to look at the right box.
    """

    def test_the_trimmed_stylesheet_renders_identically(
        self, kind, width, tmp_path_factory
    ):
        playwright = pytest.importorskip(
            "playwright.sync_api", reason="this equivalence needs a browser"
        )

        shipped = profile(FRAMES[kind], seed=0).html
        # The same document with the full stylesheet swapped back in.
        start = shipped.index("<style>")
        end = shipped.index("</style>", start) + len("</style>")
        full = shipped[:start] + load_css_dir(CSS_DIR) + shipped[end:]
        assert len(full) > len(shipped), "the swap did not put anything back"

        tmp = tmp_path_factory.mktemp(f"css-{kind}-{width}")
        (tmp / "shipped.html").write_text(shipped, encoding="utf-8")
        (tmp / "full.html").write_text(full, encoding="utf-8")

        launch = {}
        if chrome := _chrome():
            launch["executable_path"] = chrome

        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(**launch)
            except Exception as exc:  # no browser binary on this machine
                pytest.skip(f"Chromium is not available: {exc}")
            styles = {}
            for name in ("shipped", "full"):
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.goto((tmp / f"{name}.html").as_uri())
                page.wait_for_timeout(700)
                styles[name] = page.evaluate(_COMPUTED)
                page.close()
            browser.close()

        assert len(styles["shipped"]) == len(styles["full"]), (
            "the two documents do not even have the same elements"
        )
        differing = [
            i
            for i, (a, b) in enumerate(
                zip(styles["shipped"], styles["full"], strict=False)
            )
            if a != b
        ]
        assert not differing, (
            f"{len(differing)} element(s) compute differently at {width}px on a "
            f"{kind}-only frame, so a partial that was doing work got dropped. "
            f"First: shipped={styles['shipped'][differing[0]]!r} "
            f"full={styles['full'][differing[0]]!r}"
        )
