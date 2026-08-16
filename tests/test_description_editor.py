"""The description block, and the contract between the template and the JS.

`+ add a note` did nothing. Not slowly, not wrongly -- nothing, with a clean
console. The redesign renamed the block's class from `.description-value` to
`.description-block`, and `description-editor.js` still looked for the old one.
Every entry point in that file guards on a null container and returns quietly,
so the rename turned a working control into an inert one and no test noticed,
because no test had ever asserted that the selector resolves.

That is the shape of the bug worth guarding, so the first class below does not
test the description block specifically: it checks **every** selector the
bundled JS uses to find report elements against a real rendered report. A
renamed class is now caught the moment it is renamed, in whichever module it
happens next.

The rest covers the state the editor has to move together. `.is-empty` sets
`display: none` on `.description-content`, so a note saved without clearing that
class is stored, rendered, inserted -- and invisible. Fixing the selector alone
would have produced exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import ProfileConfig, RenderOptions, profile

JS_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "js"
CSS_DIR = Path(__file__).resolve().parents[1] / "pysuricata" / "static" / "css"


def _rich_frame() -> pd.DataFrame:
    """Every card kind, plus the states the controls are attached to.

    Two mistakes were made writing this, and both are the same mistake: a
    selector check is only as good as the markup the fixture reaches.

    - `[1.0, 2, 3, 4, 5] * 40` has five distinct values, so it profiles as
      *categorical*. The frame had no numeric card at all, and the numeric
      card's Linear/Log control was reported dead when it is not.
    - A frame with no quality problems renders no `.needs-attention` block, so
      the flag filter was reported dead too. It is not; it filters correctly.

    Hence: enough columns to page, a column with real gaps, a constant column,
    correlated columns, a datetime, a boolean, and a lognormal one so the
    log-scale default fires.
    """
    rng = np.random.default_rng(0)
    n = 400
    gappy = rng.normal(0, 1, n)
    gappy[rng.choice(n, 150, replace=False)] = np.nan
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "gappy": gappy,
            "const": [1.0] * n,
            "skewed": rng.lognormal(0, 1.2, n),
            "x": x,
            "y": x * 0.95 + rng.normal(0, 0.15, n),
            "cat": rng.choice(["alpha", "beta", "gamma"], n),
            "flag": rng.integers(0, 2, n).astype(bool),
            "when": pd.date_range("2026-01-01", periods=n, freq="h"),
        }
    )


def _wide_frame() -> pd.DataFrame:
    """Above the matrix ceiling, so correlations take the ranked-list route.

    The matrix and the list emit different markup, and `.col-name` lives only on
    the list. One fixture cannot reach both.
    """
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {f"c{i}": x * (0.9 - 0.06 * i) + rng.normal(0, 0.2, n) for i in range(12)}
    )


def _markup(frame: pd.DataFrame) -> str:
    """Markup only.

    The report inlines its own CSS and JS, so a name searched for in the whole
    document is found in the very source that references it -- which made the
    first version of the selector check below pass while the control was dead.
    Stripping `<script>` and `<style>` is what makes it a test of the markup.
    """
    html = profile(frame, seed=0).html
    return re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)


@pytest.fixture(scope="module")
def report() -> str:
    return _markup(_rich_frame())


@pytest.fixture(scope="module")
def corpus(report) -> str:
    """Every route a selector might legitimately live on, concatenated.

    A class absent from *all* of these is dead; a class absent from one is only
    evidence that the fixture took the other branch -- which is the mistake this
    file has already made twice.
    """
    return report + _markup(_wide_frame())


def _js_files() -> list[Path]:
    return sorted(JS_DIR.glob("*.js"))


def _runtime_created(source: str) -> set[str]:
    """Names the script assigns to elements it builds itself.

    `pagination.js` creates `#no-results` and `#flag-filter-banner` on demand
    and `functionality.js` builds its own `.hist-tooltip`, so their absence from
    the served HTML is correct rather than a broken reference.
    """
    return (
        set(re.findall(r"\.id\s*=\s*['\"]([\w-]+)['\"]", source))
        | set(re.findall(r"\.className\s*=\s*['\"]([\w\s-]+)['\"]", source))
        | set(re.findall(r"""class=\\?["']([\w\s-]+)\\?["']""", source))
    )


#: Selectors that resolve against no report on any route.
#:
#: Three entries left this set when the code behind them was deleted: the
#: missing-values tab switcher (#120 replaced the tabs with a chunk-count
#: route) and a `.details-panel` fallback for a pre-refactor layout. An
#: exemption is a promise to come back, not a place to leave things.
_KNOWN_DEAD = {
    # Emitted only by the by-chunk missing view, which cannot render until the
    # engine produces per-column per-chunk counts (#139). Not stale -- ahead of
    # the data. Deleting it would mean writing it again.
    "compact-row",
}


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
class TestEveryJsSelectorResolves:
    """A selector that matches nothing is a dead control, and it fails silently
    in JavaScript. This is the test that was missing."""

    @pytest.mark.parametrize("path", _js_files(), ids=lambda p: p.name)
    def test_every_element_id_it_looks_up_exists(self, path, corpus):
        source = path.read_text(encoding="utf-8")
        created = _runtime_created(source)
        wanted = set(re.findall(r"getElementById\(\s*['\"]([\w-]+)['\"]", source))
        missing = sorted(i for i in wanted - created if f'id="{i}"' not in corpus)
        assert not missing, (
            f"{path.name} looks up ids that no rendered element carries and that "
            f"it does not create itself: {missing}"
        )

    @pytest.mark.parametrize("path", _js_files(), ids=lambda p: p.name)
    def test_every_class_it_selects_on_exists(self, path, corpus):
        source = path.read_text(encoding="utf-8")
        created = _runtime_created(source)
        wanted = set(
            re.findall(
                r"querySelector(?:All)?\(\s*[`'\"][^`'\"]*?\.([\w-]+)[`'\"\s\)]", source
            )
        )
        missing = sorted(c for c in wanted - created - _KNOWN_DEAD if c not in corpus)
        assert not missing, (
            f"{path.name} selects on classes absent from every report: {missing}"
        )

    def test_the_known_dead_list_does_not_outlive_the_dead_code(self, corpus):
        """An entry that starts resolving means the markup came back and the
        exemption should go -- otherwise the list quietly grows into a way of
        never fixing anything."""
        alive = sorted(c for c in _KNOWN_DEAD if c in corpus)
        assert not alive, (
            f"these are no longer dead and must leave _KNOWN_DEAD: {alive}"
        )

    def test_the_editor_finds_the_block_by_id_not_by_class(self):
        """The id is the template's guarantee; the class is presentation and has
        moved once already."""
        source = (JS_DIR / "description-editor.js").read_text(encoding="utf-8")
        assert "getElementById('summary-description')" in source

    def test_the_renamed_class_is_gone_from_js_and_css(self):
        """Left behind, `.description-value` is a ruleset that styles nothing and
        the next reader's false lead. Comments are stripped first -- the note
        recording *why* it was removed names it, and must not trip this."""
        for path in (*_js_files(), *sorted(CSS_DIR.glob("*.css"))):
            code = re.sub(
                r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S
            )
            code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
            assert ".description-value" not in code, (
                f"{path.name} still refers to .description-value"
            )

    def test_the_action_the_block_dispatches_has_a_handler(self, report):
        assert 'data-action="edit-description"' in report
        dispatch = (JS_DIR / "functionality.js").read_text(encoding="utf-8")
        assert "case 'edit-description':" in dispatch


# --------------------------------------------------------------------------- #
# the markup the editor drives
# --------------------------------------------------------------------------- #
class TestTheBlockCarriesWhatTheEditorNeeds:
    def test_the_block_is_addressable_and_labelled(self, report):
        assert 'id="summary-description"' in report
        assert "description-block__label" in report
        assert "description-block__action" in report
        assert 'class="description-content"' in report

    def test_an_empty_description_invites_one(self, report):
        block = _block(report)
        assert "is-empty" in block
        assert "+ add a note" in block
        assert ">Description<" in block

    def test_a_supplied_description_is_a_note_that_can_be_edited(self):
        out = profile(
            pd.DataFrame({"a": [1.0, 2, 3]}),
            config=ProfileConfig(render=RenderOptions(description="Backfilled.")),
        ).html
        block = _block(out)
        assert "is-empty" not in block
        assert ">edit<" in block
        assert ">Note<" in block

    def test_the_storage_key_source_is_on_the_block(self, report):
        """`getStorageKey` reads `data-report-id` off the container it finds. On
        the wrong element that silently degrades every reader to one shared
        `-default` key."""
        assert re.search(
            r'id="summary-description"[^>]*\n?\s*data-report-id="[^"]+"', report
        )

    def test_the_original_markdown_round_trips_through_the_attribute(self):
        """`startDescriptionEdit` seeds the textarea from this attribute, so it
        must hold the markdown the author typed, escaped -- not the rendered
        HTML, and not something a quote could break out of."""
        out = profile(
            pd.DataFrame({"a": [1.0, 2, 3]}),
            config=ProfileConfig(
                render=RenderOptions(description='**bold** & "quoted" <b>')
            ),
        ).html
        attr = re.search(r'data-original-markdown="([^"]*)"', _block(out))
        assert attr, "the editor has nothing to seed the textarea from"
        assert attr.group(1) == "**bold** &amp; &quot;quoted&quot; &lt;b&gt;"


# --------------------------------------------------------------------------- #
# the three parts that move together
# --------------------------------------------------------------------------- #
class TestSavingMovesTheWholeState:
    """Asserted against the source, since there is no JS runner here. Each of
    these was verified in a browser: empty -> typed -> reload -> cleared."""

    @pytest.fixture(scope="class")
    def source(self) -> str:
        return (JS_DIR / "description-editor.js").read_text(encoding="utf-8")

    def test_saving_applies_the_state(self, source):
        assert "applyState(container, newMarkdown)" in source

    def test_loading_from_storage_applies_it_too(self, source):
        """A note restored from localStorage into a block the server rendered as
        empty would land under `display: none`."""
        assert "applyState(container, saved)" in source

    def test_the_state_moves_class_label_and_action_together(self, source):
        body = source.split("function applyState")[1].split("\n  }")[0]
        assert "classList.toggle('is-empty'" in body
        assert "'Note'" in body and "'Description'" in body
        assert "'+ add a note'" in body

    def test_an_empty_note_renders_empty_not_a_placeholder(self, source):
        """`.is-empty` hides the content element, so a placeholder inside it is
        a string no reader can reach."""
        assert "Click to add description" not in source


class TestTheEditorHasRoomToType:
    def test_the_textarea_spans_the_whole_row(self):
        """It is appended as a child of a three-column grid; without an explicit
        span it lands in an 88px or `auto` track."""
        css = (CSS_DIR / "_03-summary.css").read_text(encoding="utf-8")
        block = css.split(".description-block .description-editor")[1].split("}")[0]
        assert "grid-column: 1 / -1" in block

    def test_the_invitation_meets_the_minimum_target(self):
        """WCAG 2.5.8. The row is clickable as a whole, but the words are what a
        reader aims at."""
        css = (CSS_DIR / "_03-summary.css").read_text(encoding="utf-8")
        rule = css.split("#pysuricata-report .description-block__action {")[-1].split(
            "}"
        )[0]
        assert "min-height: 24px" in rule


def _block(html: str) -> str:
    start = html.index('id="summary-description"')
    return html[html.rindex("<div", 0, start) : html.index("</div>", start) + 400]
