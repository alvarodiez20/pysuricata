"""The `summarize()` payload is a contract, and this is what enforces it.

#43. The gap this closes is only findable by reading the renderer, which is how
it kept happening: #24 found correlations computed, rendered, and never emitted;
#59 found the same for numeric top values. Both times the payload was a strictly
poorer view than the HTML, with nothing saying so.

The central test walks the summary dataclasses and asserts that **every field is
either published or listed as deliberately withheld, with a reason**. Adding a
statistic to an accumulator therefore forces a decision about the contract
rather than quietly widening the gap again.
"""

from __future__ import annotations

import dataclasses as dc
import json

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.accumulators.boolean import BooleanSummary
from pysuricata.accumulators.categorical import CategoricalSummary
from pysuricata.accumulators.datetime import DatetimeSummary
from pysuricata.accumulators.numeric import NumericSummary
from pysuricata.report import (
    SUMMARY_FIELD_ALIASES,
    SUMMARY_FIELDS_WITHHELD,
    SUMMARY_SCHEMA_VERSION,
)

_SUMMARY_CLASSES = {
    "numeric": NumericSummary,
    "categorical": CategoricalSummary,
    "datetime": DatetimeSummary,
    "boolean": BooleanSummary,
}


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 4_000
    frame = pd.DataFrame(
        {
            "amount": rng.lognormal(3, 1.2, n),
            "region": rng.choice(["north", "south", "east"], n).astype(object),
            "seen_at": pd.date_range("2024-01-01", periods=n, freq="h"),
            "active": rng.random(n) > 0.4,
        }
    )
    frame.loc[:100, "amount"] = np.nan
    return frame


@pytest.fixture(scope="module")
def payload(frame):
    return summarize(frame, seed=0)


def _column_of_type(payload, kind):
    for stats in payload["columns"].values():
        if stats["type"] == kind:
            return stats
    raise AssertionError(f"no {kind} column in the fixture")


class TestEveryStatisticIsPublished:
    """The general form of the check #87 wrote for one field."""

    @pytest.mark.parametrize("kind", sorted(_SUMMARY_CLASSES))
    def test_no_field_is_silently_dropped(self, payload, kind):
        stats = _column_of_type(payload, kind)
        published = set(stats)
        missing = []
        for field in dc.fields(_SUMMARY_CLASSES[kind]):
            if field.name in SUMMARY_FIELDS_WITHHELD:
                continue
            key = SUMMARY_FIELD_ALIASES.get(field.name, field.name)
            if key not in published:
                missing.append(f"{field.name} (would be published as {key!r})")
        assert not missing, (
            f"{kind} statistics computed but not in summarize(): "
            + ", ".join(sorted(missing))
            + ". Publish them, or add them to SUMMARY_FIELDS_WITHHELD with a reason."
        )

    def test_every_withheld_field_names_a_reason(self):
        assert all(reason.strip() for reason in SUMMARY_FIELDS_WITHHELD.values())

    def test_withholding_is_limited_to_fields_that_exist(self):
        """A stale entry would silently excuse a field that no longer exists,
        and hide the next one that takes its name."""
        every_field = {
            field.name for cls in _SUMMARY_CLASSES.values() for field in dc.fields(cls)
        }
        assert set(SUMMARY_FIELDS_WITHHELD) <= every_field

    def test_aliases_are_limited_to_fields_that_exist(self):
        every_field = {
            field.name for cls in _SUMMARY_CLASSES.values() for field in dc.fields(cls)
        }
        assert set(SUMMARY_FIELD_ALIASES) <= every_field

    def test_the_reservoir_itself_is_not_in_the_payload(self):
        """Withholding `sample_vals` is a size decision, not an oversight: it
        holds up to 20,000 floats per column."""
        assert "sample_vals" in SUMMARY_FIELDS_WITHHELD


class TestThePayloadIsUsableWithoutReencoding:
    def test_it_serialises_with_the_plain_json_encoder(self, payload):
        """Numpy scalars are not JSON serialisable. A payload every consumer has
        to re-encode is not a contract."""
        json.dumps(payload)

    @pytest.mark.parametrize("kind", sorted(_SUMMARY_CLASSES))
    def test_no_numpy_scalars_survive(self, payload, kind):
        offenders = [
            key
            for key, value in _column_of_type(payload, kind).items()
            if type(value).__module__ == "numpy"
        ]
        assert offenders == []

    def test_the_dataset_block_serialises_too(self, payload):
        json.dumps(payload["dataset"])


class TestTheHtmlAndThePayloadAgree:
    """Spot checks that the published numbers are the rendered ones."""

    def test_the_row_count_matches(self, frame):
        report = profile(frame, seed=0)
        assert f"{report.stats['dataset']['rows_est']:,}" in report.html

    def test_a_numeric_extreme_matches(self, frame):
        report = profile(frame, seed=0)
        stats = _column_of_type(report.stats, "numeric")
        assert stats["max_items"], "expected extreme values"
        # The renderer formats to a fixed precision, so compare the value the
        # payload carries against the tracker's, not against the HTML text.
        assert stats["max_items"][0][1] == stats["max"]

    def test_the_boolean_ratio_matches_the_counts(self, payload):
        stats = _column_of_type(payload, "boolean")
        total = stats["true"] + stats["false"]
        assert stats["true_ratio"] == pytest.approx(stats["true"] / total)

    def test_the_datetime_tallies_add_up(self, payload):
        stats = _column_of_type(payload, "datetime")
        assert sum(stats["by_hour"]) == stats["count"]
        assert sum(stats["by_dow"]) == stats["count"]
        assert sum(stats["by_year"].values()) == stats["count"]


class TestTheDocumentedKeys:
    """What the schema page promises."""

    def test_the_top_level_keys(self, payload):
        assert set(payload) == {"schema_version", "dataset", "columns"}

    def test_the_dataset_keys(self, payload):
        assert set(payload["dataset"]) == {
            "rows_est",
            "cols",
            "missing_cells",
            "missing_cells_pct",
            "duplicate_rows_est",
            "duplicate_rows_pct_est",
            # The bound that makes the estimate readable. Zero with a zero
            # uncertainty is "exactly none"; zero with an uncertainty of 2,201
            # is "nothing resolvable below about 2,201". Without this key a
            # consumer could not reach the answer the report already printed.
            "duplicate_rows_uncertainty",
            "memory_bytes",
            "top_missing",
        }

    def test_every_column_declares_a_type(self, payload):
        assert all("type" in stats for stats in payload["columns"].values())

    def test_the_types_are_the_documented_five(self, payload):
        known = {"numeric", "categorical", "datetime", "boolean", "identifier"}
        assert {s["type"] for s in payload["columns"].values()} <= known

    def test_the_version_is_present(self, payload):
        assert payload["schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_the_four_shared_keys_are_on_every_column(self, payload):
        """The promise the docs make about what is safe to read blind."""
        for stats in payload["columns"].values():
            assert {"type", "count", "missing", "mem_bytes"} <= set(stats)


class TestAddingKeysIsNotBreaking:
    """The stated policy, exercised rather than asserted in prose."""

    def test_this_release_added_keys_without_bumping_the_version(self, payload):
        """Numeric gained skew, kurtosis, extremes and the histogram; datetime
        gained fifteen fields. A consumer reading the old keys is unaffected,
        which is exactly what the policy says."""
        stats = _column_of_type(payload, "numeric")
        assert {"skew", "kurtosis", "iqr", "true_histogram_counts"} <= set(stats)
        assert payload["schema_version"] == 1

    def test_correcting_a_wrong_value_does_not_bump_the_version(self, payload):
        """The rule that is easiest to get backwards, so it is pinned here.

        #327 changed `outliers_iqr_est` from a reservoir count to a population
        estimate. That looks like repurposing a key, and the instinct is to
        bump. `docs/versioning.md` rules the other way: correcting a wrong
        value is a bug fix, and "pinning it under the schema would mean the
        contract guaranteed the bug". The precedent it records is
        `duplicate_rows_est` (#202), which has the same shape -- published
        wrong, corrected in place, an uncertainty field added beside it.

        The two new keys are additive, which is free under the same policy.
        """
        stats = _column_of_type(payload, "numeric")
        assert {"outliers_iqr_sample", "outliers_mod_zscore_sample"} <= set(stats)
        assert payload["schema_version"] == 1

    def test_the_keys_that_existed_before_still_exist(self, payload):
        """The 0.0.38 numeric key set, pinned. Removing one of these is what
        bumps the version."""
        before = {
            "type",
            "count",
            "missing",
            "unique_est",
            "mean",
            "std",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "zeros",
            "negatives",
            "outliers_iqr_est",
            "approx",
            "mem_bytes",
            "corr_top",
            "mono_inc",
            "mono_dec",
            "int_like",
            "top_values",
        }
        assert before <= set(_column_of_type(payload, "numeric"))
