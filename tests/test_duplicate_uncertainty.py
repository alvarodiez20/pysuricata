"""The duplicate count is a difference, and its error is not the sketch's.

``approx_duplicates`` returns ``rows - distinct``. ``rows`` is exact and
``distinct`` is a sketch estimate, so the whole *absolute* error of the distinct
estimate lands on a quantity that is usually far smaller. The relative error is
multiplied by ``distinct / duplicates``.

Measured on 200,000 rows with exactly 2,000 duplicates: the distinct estimate
was 0.48% low -- comfortably inside spec -- and the reported duplicate count came
back 2,942, **47% high**. The amplification factor is 99x, and 0.48% x 99 is
47%, so the model and the observation agree to the digit.

The card was not silent about this. It printed ``≈ KMV sketch``, which a reader
who knows what a KMV sketch is will read as ±1-2%. That is the error on
*distinct*. **An approximation marker that implies the wrong order of magnitude
is worse than none**, because it turns a naked estimate into a confident one.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.accumulators.sketches import (
    DUPLICATE_RESOLUTION_SIGMAS,
    RowKMV,
)


def _frame(n: int, duplicates: int) -> pd.DataFrame:
    """Exactly ``duplicates`` duplicate rows, shuffled."""
    base = np.arange(n - duplicates)
    values = np.concatenate([base, base[:duplicates]])
    rng = np.random.default_rng(0)
    rng.shuffle(values)
    return pd.DataFrame({"id": values})


def _sketch(n: int, duplicates: int) -> RowKMV:
    sketch = RowKMV()
    sketch.update_from_pandas(_frame(n, duplicates))
    return sketch


def _duplicates_stat(html: str) -> tuple[str, str]:
    markup = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S | re.I)
    start = markup.find("Duplicates")
    chunk = re.sub(r"\s+", " ", markup[start : start + 220])
    value = re.search(r'stat__val">([^<]*)<', chunk)
    note = re.search(r'stat__sub">([^<]*)<', chunk)
    return value.group(1), note.group(1)


class TestTheBoundCoversTheTruth:
    """The property that makes the figure usable: whatever it prints, the real
    answer is inside the stated interval."""

    @pytest.mark.parametrize("duplicates", [200, 2_000, 20_000, 100_000])
    def test_truth_lies_within_two_sigma(self, duplicates):
        sketch = _sketch(200_000, duplicates)
        reported, _ = sketch.approx_duplicates()
        sigma = sketch.duplicates_uncertainty()
        assert abs(reported - duplicates) <= 2 * max(sigma, 1)

    def test_the_error_really_is_amplified(self):
        """Not a hypothetical. The distinct estimate is well inside spec and the
        duplicate figure is still nearly 50% out."""
        sketch = _sketch(200_000, 2_000)
        reported, _ = sketch.approx_duplicates()
        distinct = 200_000 - reported
        distinct_error = abs(distinct - 198_000) / 198_000
        duplicate_error = abs(reported - 2_000) / 2_000

        assert distinct_error < 0.02, "the sketch itself should be well within spec"
        assert duplicate_error > 0.2, "and the derived figure should still be far out"
        # The amplification is distinct/duplicates, and it is what connects them.
        assert duplicate_error > distinct_error * 10


class TestAnExactCountSaysSo:
    """KMV counts exactly until it has seen `k` distinct values, so most frames
    have no estimation error at all here.

    This is the case the first version of the change got wrong -- it reported
    `< 10 (below sketch resolution)` for an 891-row frame whose duplicate count
    was known exactly. The invariance fingerprint caught it.
    """

    def test_a_small_frame_is_exact(self):
        sketch = _sketch(891, 0)
        assert sketch.kmv_is_exact()
        assert sketch.duplicates_uncertainty() == 0

    def test_zero_duplicates_is_a_result_not_a_ceiling(self):
        """`exactly none` must not be presented as `fewer than some bound`."""
        sketch = _sketch(891, 0)
        assert sketch.duplicates_are_resolvable()
        value, note = _duplicates_stat(profile(_frame(891, 0), seed=0).html)
        assert value == "0"
        assert note == "exact"

    def test_a_small_frame_with_duplicates_reports_them_exactly(self):
        value, note = _duplicates_stat(profile(_frame(891, 100), seed=0).html)
        assert value == "100"
        assert note == "exact"

    def test_the_old_label_no_longer_claims_approximation_where_there_is_none(self):
        _, note = _duplicates_stat(profile(_frame(891, 0), seed=0).html)
        assert "KMV sketch" not in note


class TestBelowResolutionStatesACeiling:
    """When the count is smaller than its own uncertainty, a figure would invite
    a conclusion the sketch cannot support."""

    def test_a_rare_duplicate_rate_is_not_resolvable(self):
        sketch = _sketch(200_000, 200)
        assert not sketch.duplicates_are_resolvable()

    def test_the_report_prints_a_ceiling(self):
        value, note = _duplicates_stat(profile(_frame(200_000, 200), seed=0).html)
        assert value.startswith("&lt;") or value.startswith("<")
        assert note == "below sketch resolution"

    def test_the_ceiling_is_above_the_truth(self):
        sketch = _sketch(200_000, 200)
        assert sketch.duplicates_uncertainty() > 200


class TestAResolvableCountCarriesItsBound:
    def test_the_note_states_the_bound(self):
        """50,000 duplicates in 200,000 rows: about 23 sigma, so it publishes.

        This case used to be 2,000 duplicates, which is 0.9 sigma and was
        published under the old 1-sigma gate on the strength of a draw that
        happened to land high (2,942 reported for a true 2,000). It is
        suppressed now, and correctly -- see
        `TestTheGateIsThreeSigma.test_a_one_percent_rate_is_below_resolution`.
        """
        value, note = _duplicates_stat(profile(_frame(200_000, 50_000), seed=0).html)
        assert re.fullmatch(r"[\d,]+", value)
        assert re.match(r"± [\d,]+ · KMV sketch", note), note

    def test_the_bound_is_not_zero_when_the_count_is_estimated(self):
        sketch = _sketch(200_000, 2_000)
        assert not sketch.kmv_is_exact()
        assert sketch.duplicates_uncertainty() > 0


class TestNothingDegradesOnEmptyOrTinyInput:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_no_division_by_zero(self, n):
        sketch = RowKMV()
        if n:
            sketch.update_from_pandas(pd.DataFrame({"id": np.arange(n)}))
        assert sketch.duplicates_uncertainty() == 0
        assert sketch.duplicates_are_resolvable()


class TestTheReportAndThePayloadAgree:
    """The invariant that was missing, and the reason #202 got through.

    #161 fixed the duplicate figure in `render/html.py` — the surface
    `docs/versioning.md` explicitly does *not* cover — and left it raw in
    `compute/manifest.py` and `report.py`, which build the payload that carries
    `schema_version`. Two call sites produced the figure and one of them
    corrected it.

    So on 200,000-row frames with exactly zero duplicate rows the report read
    `< 2,201 · below sketch resolution` while `summarize()` returned counts up
    to 880. A human was told the truth; a CI gate, a `pysuricata check` or a
    dbt hook reading the payload was told there were duplicates in a dataset
    that had none.

    Checking one surface could never have caught that. These tests check both
    against each other.
    """

    def _payload_and_report(self, frame):
        stats = summarize(frame, seed=0)["dataset"]
        value, note = _duplicates_stat(profile(frame, seed=0).html)
        return stats, value, note

    def test_a_suppressed_report_means_a_suppressed_payload(self):
        """200,000 distinct rows: below resolution, so both say so."""
        stats, value, note = self._payload_and_report(_frame(200_000, 0))

        assert note == "below sketch resolution"
        assert stats["duplicate_rows_est"] == 0, (
            "the report suppressed the count and the payload published it raw"
        )
        assert stats["duplicate_rows_pct_est"] == 0.0
        # The ceiling the report prints is the uncertainty the payload exports,
        # times the multiple the gate applies, so a consumer can reach the same
        # conclusion the reader can. `docs/summary-schema.md` states it, which
        # is what makes the payload readable without the report beside it.
        ceiling = math.ceil(
            DUPLICATE_RESOLUTION_SIGMAS * stats["duplicate_rows_uncertainty"]
        )
        assert value == f"&lt; {ceiling:,}"

    def test_a_resolvable_count_reaches_the_payload_intact(self):
        """Suppression must not become a blanket zero: 50,000 duplicates in
        200,000 rows are far above the sketch's resolution and must survive."""
        stats, value, _ = self._payload_and_report(_frame(200_000, 50_000))

        assert stats["duplicate_rows_est"] > 0
        assert value == f"{stats['duplicate_rows_est']:,}"
        assert abs(stats["duplicate_rows_est"] - 50_000) <= 2 * max(
            stats["duplicate_rows_uncertainty"], 1
        )

    def test_an_exact_zero_is_not_dressed_up_as_a_ceiling(self):
        """891 rows is below `k`, so the count is exact rather than estimated.

        "Exactly none" is a resolved result; reporting it as "fewer than some
        bound" would understate what the sketch knows.
        """
        stats, value, note = self._payload_and_report(_frame(891, 0))

        assert (value, note) == ("0", "exact")
        assert stats["duplicate_rows_est"] == 0
        assert stats["duplicate_rows_uncertainty"] == 0

    @pytest.mark.parametrize("seed", range(8))
    def test_frames_with_no_duplicates_report_none(self, seed):
        """#161's reproduction, through the payload this time.

        Every one of these frames has exactly zero duplicate rows; before the
        fix each returned a non-zero `duplicate_rows_est`.
        """
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"id": rng.permutation(200_000)})
        assert frame.duplicated().sum() == 0

        stats = summarize(frame, seed=0)["dataset"]

        assert stats["duplicate_rows_est"] == 0, (
            f"seed {seed}: reported {stats['duplicate_rows_est']} duplicates "
            "in a frame that has none"
        )


class TestTheUncertaintyIsExported:
    """`duplicates_uncertainty()` was computed and never exported, so a payload
    consumer had no key to apply the threshold themselves."""

    def test_the_payload_carries_it(self):
        stats = summarize(_frame(200_000, 0), seed=0)["dataset"]
        assert "duplicate_rows_uncertainty" in stats
        assert stats["duplicate_rows_uncertainty"] > 0

    def test_zero_with_zero_uncertainty_is_distinguishable_from_zero_without(self):
        """The distinction the extra key exists to make: "exactly none" and
        "nothing resolvable" are both `duplicate_rows_est == 0`."""
        exact = summarize(_frame(891, 0), seed=0)["dataset"]
        unresolved = summarize(_frame(200_000, 0), seed=0)["dataset"]

        assert exact["duplicate_rows_est"] == unresolved["duplicate_rows_est"] == 0
        assert exact["duplicate_rows_uncertainty"] == 0
        assert unresolved["duplicate_rows_uncertainty"] > 0


class TestTheIntervalIsPublished:
    """#329. `duplicate_rows_est == 0` cannot be told from "below my
    resolution" without also reading `duplicate_rows_uncertainty` and doing
    the arithmetic -- and the README's own CI-gate example
    (`duplicate_rows_est == 0`) passed identically on a clean frame and on
    one whose duplicates were merely unresolvable, which is a gate failing
    open in exactly the case it exists to catch.

    `duplicate_rows_lo` / `duplicate_rows_hi` publish the arithmetic done,
    so a consumer reaches the same conclusion the report already had without
    reconstructing it.
    """

    def test_an_unresolved_count_publishes_zero_to_the_ceiling(self):
        """`hi` must be the exact figure the report prints, not a second,
        independently-computed version of it that could drift from it."""
        stats = summarize(_frame(200_000, 0), seed=0)["dataset"]
        value, _ = _duplicates_stat(profile(_frame(200_000, 0), seed=0).html)

        assert stats["duplicate_rows_lo"] == 0
        assert value == f"&lt; {stats['duplicate_rows_hi']:,}"

    def test_a_resolved_count_publishes_a_bound_around_it(self):
        stats = summarize(_frame(200_000, 50_000), seed=0)["dataset"]

        est, sigma = stats["duplicate_rows_est"], stats["duplicate_rows_uncertainty"]
        assert stats["duplicate_rows_lo"] == max(0, est - sigma)
        assert stats["duplicate_rows_hi"] == est + sigma
        assert stats["duplicate_rows_lo"] <= est <= stats["duplicate_rows_hi"]

    def test_an_exact_count_has_no_interval_around_it(self):
        """891 rows is below `k` -- the count is exact, not estimated, so the
        interval collapses to the point rather than padding a known answer
        with a bound that does not apply to it."""
        stats = summarize(_frame(891, 0), seed=0)["dataset"]

        assert stats["duplicate_rows_lo"] == stats["duplicate_rows_hi"] == 0

    def test_a_gate_on_the_upper_bound_fails_closed_below_the_floor(self):
        """The motivating case: a frame with real duplicates just under the
        resolution floor must fail a strict gate compared against `hi`,
        where a gate compared against the suppressed `duplicate_rows_est`
        would pass it -- exactly the false-negative #329 was filed over."""
        stats = summarize(_frame(200_000, 2_000), seed=0)["dataset"]

        assert stats["duplicate_rows_est"] == 0, "suppressed, as #248 intends"
        assert stats["duplicate_rows_hi"] > 0
        max_allowed_pct = 0.1
        allowed_rows = 200_000 * max_allowed_pct / 100.0
        assert stats["duplicate_rows_hi"] > allowed_rows, (
            "a gate reading duplicate_rows_est alone would wrongly pass here"
        )


class TestTheGateIsThreeSigma:
    """#248. The gate was an implicit `>` — one sigma — and a clean frame
    published a duplicate count about one run in ten.

    The rate is what the issue measured; the *multiple* is what these assert.
    Separating 0.13% from 2.3% by simulation needs thousands of frames and
    still buys a flaky test; the multiple is one number and it is the thing
    that was wrong.
    """

    class _Sketch(RowKMV):
        """A sketch with the two inputs to the gate forced.

        The gate reads exactly `approx_duplicates()` and `duplicates_uncertainty()`,
        so driving those directly tests the boundary at a chosen number of sigma
        rather than at whatever a 200,000-row draw happened to produce.
        """

        def __init__(self, duplicates: int, sigma: int) -> None:
            super().__init__()
            self.rows = 1_000_000
            self._duplicates = duplicates
            self._sigma = sigma

        def approx_duplicates(self):
            return self._duplicates, self._duplicates / self.rows * 100.0

        def duplicates_uncertainty(self):
            return self._sigma

        def kmv_is_exact(self):
            return False

    def test_the_multiple_is_a_named_constant(self):
        assert DUPLICATE_RESOLUTION_SIGMAS == 3.0

    @pytest.mark.parametrize("sigmas", [0.5, 1.0, 1.5, 2.0, 2.9])
    def test_below_the_multiple_is_suppressed(self, sigmas):
        sketch = self._Sketch(int(1_000 * sigmas), 1_000)
        assert not sketch.duplicates_are_resolvable()
        assert sketch.duplicates().rows == 0

    @pytest.mark.parametrize("sigmas", [3.1, 4.0, 10.0])
    def test_above_the_multiple_is_published(self, sigmas):
        sketch = self._Sketch(int(1_000 * sigmas), 1_000)
        assert sketch.duplicates_are_resolvable()
        assert sketch.duplicates().rows == int(1_000 * sigmas)

    def test_the_ceiling_is_the_multiple_not_one_sigma(self):
        """The bound printed for a suppressed count has to be above the count
        it suppressed, or the report contradicts itself."""
        sketch = self._Sketch(2_500, 1_000)
        estimate = sketch.duplicates()

        assert estimate.resolvable is False
        assert estimate.uncertainty == 1_000, "the exported sigma stays one sigma"
        assert estimate.ceiling == 3_000
        assert estimate.ceiling > 2_500

    def test_a_resolvable_count_carries_no_ceiling(self):
        assert self._Sketch(10_000, 1_000).duplicates().ceiling == 0

    def test_an_exact_count_has_no_ceiling_either(self):
        """Below `k` distinct values KMV counts exactly, so there is no bound
        to state and "exactly none" must not be dressed up as one."""
        sketch = _sketch(891, 0)
        assert sketch.duplicates_ceiling() == 0
        assert sketch.duplicates() == (0, 0.0, 0, True, 0)

    def test_a_one_percent_rate_is_below_resolution(self):
        """200,000 rows with 2,000 duplicates is 0.9 sigma on a 2,048-value
        sketch. The old gate published it — as 2,942, 47% high — because the
        draw landed above one sigma. That is the false alarm, not a detection.
        """
        sketch = _sketch(200_000, 2_000)
        assert not sketch.duplicates_are_resolvable()
        assert sketch.duplicates().rows == 0

    @pytest.mark.parametrize("seed", range(25))
    def test_a_clean_frame_raises_no_alarm(self, seed):
        """The measurement from the issue, re-run at the new multiple.

        At 1 sigma this fired on 4 of 40 seeds. At 3 the normal-tail rate is
        0.13%, so 25 clean frames failing here is a real regression rather than
        an unlucky day — and the seeds are fixed, so it is the same 25 frames
        every run.
        """
        rng = np.random.default_rng(1_000 + seed)
        frame = pd.DataFrame(
            {"a": rng.permutation(200_000), "b": rng.normal(size=200_000)}
        )
        assert frame.duplicated().sum() == 0

        sketch = RowKMV()
        sketch.update_from_pandas(frame)

        assert sketch.duplicates().rows == 0
