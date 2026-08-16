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

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile
from pysuricata.accumulators.sketches import RowKMV


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
        value, note = _duplicates_stat(profile(_frame(200_000, 2_000), seed=0).html)
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
