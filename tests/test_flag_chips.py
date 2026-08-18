"""Three defects found while designing the report, and easy to lose in a plan.

Phase 5.7 (#118).

1. **The distinct count could exceed the row count.** Fixed earlier, in the
   commit that had to come before any of the redesign was screenshotted — the
   baselines this migration measures against would otherwise have been taken
   from a report claiming 892 distinct values in 891 rows. Asserted here again
   because the acceptance list belongs to this issue, and because the clamp has
   to hold for *every* column kind, not only numeric.

2. **The chips hid the number they already had.** Every one carried
   ``data-threshold`` and ``data-value`` in the DOM and displayed neither, so a
   card said ``Missing`` where it could have said ``19.9% missing`` — and a
   reader had to open the details pane, or the inspector, to learn whether that
   meant two rows or two hundred.

3. **``.stat-badges`` was styled but never rendered.** The renderer had already
   gone; what remained was a block of CSS and a rule to hide the markup it
   styled. Both are bytes in a single-file report.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.render.triage import Chip, annotate_flags, extract_chips

CARDS_CSS = (
    Path(__file__).resolve().parents[1] / "pysuricata/static/css/_06-cards.css"
).read_text()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20_000
    return pd.DataFrame(
        {
            "score": rng.normal(0, 1, n),
            "name": [f"passenger {i}" for i in range(n)],
            "when": pd.date_range("2026-01-01", periods=n, freq="min"),
            "flag": rng.integers(0, 2, n).astype(bool),
        }
    )


# --------------------------------------------------------------------------- #
# 1. the impossible number
# --------------------------------------------------------------------------- #
class TestDistinctNeverExceedsTheRowCount:
    """More distinct values than rows is arithmetically impossible, and a
    reader who notices does not conclude *sketch tolerance* — they conclude the
    numbers cannot be trusted, which lands on every other figure on the page."""

    def test_for_every_column_that_publishes_one(self, frame):
        """Numeric, categorical and datetime carry `unique_est`. A boolean
        column does not, and should not: its distinct count is 2 by
        definition, so publishing an estimate of it would be inventing an
        approximation for something exactly known."""
        payload = summarize(frame, seed=0)
        checked = 0
        for name, column in payload["columns"].items():
            if "unique_est" not in column:
                assert column.get("type") == "boolean", name
                continue
            assert column["unique_est"] <= column["count"], name
            checked += 1
        assert checked >= 3, "expected numeric, categorical and datetime"

    def test_the_reported_case(self):
        """20,000 standard normals estimated at 20,197 before the clamp."""
        rng = np.random.default_rng(0)
        payload = summarize(pd.DataFrame({"score": rng.normal(0, 1, 20_000)}), seed=0)
        column = payload["columns"]["score"]
        assert column["unique_est"] <= column["count"]

    def test_the_approximation_marker_survives_the_clamp(self, frame):
        """Clamping must not turn an estimate into a claim of exactness."""
        payload = summarize(frame, seed=0)
        assert payload["columns"]["score"]["approx"] is True

    def test_an_exactly_counted_column_is_still_not_marked_approximate(self):
        payload = summarize(pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.0]}), seed=0)
        assert payload["columns"]["x"]["approx"] is False

    def test_an_empty_frame_does_not_raise(self):
        """The clamp still has to hold when there is nothing to clamp.

        This used to assert the payload had **no columns at all** -- a zero-row
        frame returned `{}` and the column never existed to be checked. #315
        changed that: the schema is known, so the column is reported with zero
        counts, which means the invariant this class exists for now applies to
        it rather than skipping it.

        The stronger statement is the one worth making. `0 <= 0` is trivially
        true, but a sketch that returned any positive estimate for a column it
        never saw a value in would be the same defect as the reported case
        above, and nothing else would catch it.
        """
        payload = summarize(pd.DataFrame({"x": pd.Series([], dtype=float)}), seed=0)
        column = payload["columns"]["x"]

        assert column["count"] == 0
        assert column["unique_est"] <= column["count"]


# --------------------------------------------------------------------------- #
# 2. the chips
# --------------------------------------------------------------------------- #
class TestTheChipsShowWhatTheyKnow:
    def test_the_value_reaches_the_face_of_the_chip(self, frame):
        html = profile(frame, seed=0).html
        labels = [chip.label for chip in extract_chips(html)]
        assert any(re.match(r"[\d.\-]", label) for label in labels), labels[:6]

    def test_the_value_leads_and_the_name_follows(self):
        out = annotate_flags(
            '<li class="flag warn" data-threshold=">10%" data-value="19.9%">Missing</li>'
        )
        assert ">19.9% missing" in out

    def test_the_threshold_goes_on_the_face_not_into_a_title(self):
        """It used to move into a `title`, and this test used to require that.

        A tooltip is invisible on a phone and absent from a printed report, so
        `19.9%` had nothing to be judged against in either — and the reader who
        cannot hover has the least context, not the most. Phase 4b.2 puts the
        limit on the face and drops the tooltip entirely; what the number *is*
        moves to the flag reference, stated once per flag.
        """
        out = annotate_flags(
            '<li class="flag warn" data-threshold=">10%" data-value="19.9%">Missing</li>'
        )
        assert "19.9% missing · limit 20%" in out
        assert "title=" not in out, "the tooltip is back"

    def test_a_threshold_containing_a_bracket_survives(self):
        """`data-threshold="|kurtosis| > 3"` ends the tag early for a naive
        `[^>]*` match — the bug that was sitting in this repository's own test
        helper until these chips started carrying values."""
        out = annotate_flags(
            '<li class="flag bad" data-threshold="|kurtosis| > 3" '
            'data-value="9.1">Heavy‑tailed</li>'
        )
        assert "9.1 heavy‑tailed" in out
        # The face carries the value and the limit; the slug still says which
        # flag it is; and the raw pair rides along for the triage block to rank
        # on (#149), bracket and all.
        assert extract_chips(out) == [
            Chip(
                "bad",
                "9.1 heavy‑tailed · limit 10",
                "heavy-tailed",
                "9.1",
                "|kurtosis| > 3",
            )
        ]

    def test_a_chip_with_no_value_is_left_alone(self):
        """Not every flag is a measurement. `Monotonic ↑` has no number, and
        inventing one would be worse than leaving the chip as it is."""
        original = '<li class="flag">Monotonic ↑</li>'
        assert annotate_flags(original) == original

    def test_a_chip_with_a_value_but_no_threshold_still_gains_the_value(self):
        out = annotate_flags('<li class="flag good" data-value="0.45">Normal</li>')
        assert ">0.45 normal<" in out
        assert "title=" not in out

    def test_the_data_attributes_are_kept(self):
        """The triage block and the chip filter both read them."""
        out = annotate_flags(
            '<li class="flag warn" data-threshold=">10%" data-value="19.9%">Missing</li>'
        )
        assert 'data-value="19.9%"' in out
        assert 'data-threshold="&gt;10%"' in out or 'data-threshold=">10%"' in out

    def test_the_triage_block_gets_the_values_too(self, frame):
        """#86 puts these chips at the top of the report, where a bare word is
        least useful."""
        html = profile(frame, seed=0).html
        block = html.split("needs a look", 1)
        if len(block) > 1:
            assert re.search(r"[\d.]+%?\s+\w", block[1][:600])


class TestTheChipsAreOutlinedNotFilled:
    @pytest.mark.parametrize("severity", ["bad", "warn", "good"])
    def test_the_severity_is_the_border_not_a_wash(self, severity):
        block = CARDS_CSS.split(
            f"#pysuricata-report .quality-flags .flag.{severity} {{", 1
        )[1].split("}", 1)[0]
        assert "background: transparent" in block
        assert "rgba(" not in block

    def test_the_warning_chip_uses_the_text_step(self):
        """`--q-warn-fill` sits deliberately below the text minimum so a bar can
        be lighter than a word. A chip label is a word."""
        block = CARDS_CSS.split("#pysuricata-report .quality-flags .flag.warn {", 1)[
            1
        ].split("}", 1)[0]
        assert "color: var(--q-warn-text)" in block


# --------------------------------------------------------------------------- #
# 3. the markup that was rendered and then hidden
# --------------------------------------------------------------------------- #
class TestStatBadgesAreGone:
    def test_no_report_contains_the_markup(self, frame):
        html = profile(frame, seed=0).html
        assert 'class="stat-badges"' not in html

    def test_and_no_css_is_left_styling_it(self):
        """A rule to hide markup that is never emitted is two kinds of dead at
        once, and still bytes in a single-file report."""
        rules = [
            line
            for line in CARDS_CSS.splitlines()
            if "stat-badges" in line and not line.strip().startswith(("/*", "*", "//"))
        ]
        assert not rules, rules

    def test_the_hide_rule_went_with_it(self):
        assert "Hide numeric header chips" not in CARDS_CSS


class TestTheFlagReference:
    """Design 15b. The chips name a conclusion — `heavy-tailed`, `dominant
    category` — and that vocabulary is only decodable if it is written down.
    Four columns: the flag, what was measured, the limit that fired it, and
    what it means for the data.
    """

    @staticmethod
    def _block(html: str) -> str:
        found = re.search(r'<details class="flagref".*?</details>', html, re.S)
        return found.group(0) if found else ""

    def test_it_renders_only_the_flags_the_report_raised(self, frame):
        html = profile(frame, seed=0).html
        block = self._block(html)
        assert block, "no flag reference in a report that raises flags"

        listed = set(re.findall(r'<tr id="flagref-([^"]+)"', block))
        raised = {chip.slug for chip in extract_chips(html) if chip.slug}
        assert listed <= raised, sorted(listed - raised)
        assert listed, "the reference is empty"

    def test_a_clean_frame_carries_no_reference(self):
        """Rendering the whole table regardless would put a glossary above the
        first card of every report, including ones with nothing to explain."""
        frame = pd.DataFrame({"a": np.arange(500.0), "b": np.arange(500.0) * 2})
        html = profile(frame, seed=0).html
        block = self._block(html)
        if not block:
            return
        listed = set(re.findall(r'<tr id="flagref-([^"]+)"', block))
        raised = {chip.slug for chip in extract_chips(html) if chip.slug}
        assert listed <= raised

    def test_every_row_states_a_measure_a_limit_and_a_meaning(self, frame):
        block = self._block(profile(frame, seed=0).html)
        rows = re.findall(r'<tr id="flagref-[^"]+">(.*?)</tr>', block, re.S)
        assert rows, "no rows"
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
            assert len(cells) == 4, cells
            assert all(re.sub(r"<[^>]+>", "", c).strip() for c in cells), cells

    def test_it_gives_no_advice(self, frame):
        """Open question 7 of the design package: whether pysuricata should
        recommend actions at all is undecided, so the reference states a
        consequence for the data and stops. "Drop before modelling" is wrong
        for a reader who is not modelling.
        """
        block = self._block(profile(frame, seed=0).html)
        prose = re.sub(r"<[^>]+>", " ", block).lower()
        for verb in ("you should", "drop the", "consider removing", "we recommend"):
            assert verb not in prose, verb

    def test_every_flag_the_renderers_can_raise_has_an_entry_or_is_dropped(self):
        """A flag with no entry must not render a blank row. Adding a chip in a
        card renderer should never be able to put an empty line in here.
        """
        from pysuricata.render.flag_reference import raised_flags

        assert raised_flags(["not-a-real-flag"]) == []
        assert [s for s, _ in raised_flags(["missing", "not-a-real-flag"])] == [
            "missing"
        ]
