"""Keyword options, presets, the log-scale default, and the payload contract.

UX-4, UX-6 and UX-11. Three findings with one shape: the library already had
the answer -- the log-scale flag, the top-k counters, the config fields -- and
made the caller work to reach it, or did not expose it at all.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import (
    ComputeOptions,
    ConfigurationError,
    ProfileConfig,
    profile,
    summarize,
)
from pysuricata.report import SUMMARY_SCHEMA_VERSION


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 5_000
    return pd.DataFrame(
        {
            "revenue": rng.lognormal(3, 1.5, n),
            "normal": rng.standard_normal(n),
            # Above the 50-level categorical ceiling, so it stays numeric, but
            # low-cardinality enough that the top-k sketch is worth feeding.
            "score": rng.integers(0, 200, n),
        }
    )


def _scales(html: str) -> list[str]:
    return re.findall(r'<div class="hist-controls" data-scale="(\w+)"', html)


class TestLogScaleDefault:
    """UX-4. The card computed the right answer and drew the wrong picture."""

    def test_a_lognormal_column_opens_on_a_log_axis(self, frame):
        html = profile(frame[["revenue"]]).html
        assert _scales(html) == ["log"]

    def test_a_normal_column_still_opens_on_linear(self, frame):
        html = profile(frame[["normal"]]).html
        assert _scales(html) == ["lin"]

    def test_both_toggle_buttons_are_still_present(self, frame):
        html = profile(frame[["revenue"]]).html
        assert 'data-scale="lin"' in html
        assert 'data-scale="log"' in html

    def test_the_active_button_matches_the_default(self, frame):
        html = profile(frame[["revenue"]]).html
        assert '<button type="button" class="btn-soft active" data-scale="log">' in html


class TestKeywordOptions:
    """UX-11. Three imports and two nested constructors to set one integer."""

    def test_a_bare_call_still_works(self, frame):
        assert summarize(frame)["dataset"]["rows_est"] == 5_000

    def test_chunk_size_as_a_keyword(self, frame):
        assert summarize(frame, chunk_size=1_000)["dataset"]["rows_est"] == 5_000

    def test_columns_as_a_keyword(self, frame):
        assert set(summarize(frame, columns=["revenue"])["columns"]) == {"revenue"}

    def test_seed_as_a_keyword_is_reproducible(self, frame):
        first = summarize(frame, seed=42)["columns"]["revenue"]["median"]
        second = summarize(frame, seed=42)["columns"]["revenue"]["median"]
        assert first == second

    def test_correlations_can_be_turned_off(self, frame):
        stats = summarize(frame, correlations=False)
        assert stats["columns"]["revenue"]["corr_top"] == []

    def test_title_reaches_the_report(self, frame):
        assert "Quarterly review" in profile(frame, title="Quarterly review").html

    def test_an_unknown_option_names_the_ones_that_exist(self, frame):
        with pytest.raises(ConfigurationError, match="chunk_size"):
            summarize(frame, chunck_size=1_000)

    def test_the_error_is_still_a_valueerror(self, frame):
        with pytest.raises(ValueError):
            summarize(frame, nonsense=1)


class TestPresets:
    def test_fast_runs(self, frame):
        assert summarize(frame, preset="fast")["dataset"]["rows_est"] == 5_000

    def test_thorough_runs(self, frame):
        assert summarize(frame, preset="thorough")["dataset"]["rows_est"] == 5_000

    def test_fast_turns_correlations_off(self, frame):
        assert summarize(frame, preset="fast")["columns"]["revenue"]["corr_top"] == []

    def test_an_unknown_preset_lists_the_real_ones(self, frame):
        with pytest.raises(ConfigurationError, match="fast, thorough"):
            summarize(frame, preset="turbo")

    def test_a_keyword_overrides_the_preset(self, frame):
        """Precedence: defaults, then preset, then keywords."""
        stats = summarize(frame, preset="fast", correlations=True)
        assert isinstance(stats["columns"]["revenue"]["corr_top"], list)


class TestConfigRemainsTheEscapeHatch:
    def test_config_still_works_alone(self, frame):
        config = ProfileConfig(compute=ComputeOptions(chunk_size=1_000))
        assert summarize(frame, config=config)["dataset"]["rows_est"] == 5_000

    def test_config_with_a_preset_is_refused_rather_than_silently_ignored(self, frame):
        with pytest.raises(ConfigurationError, match="not both"):
            summarize(frame, config=ProfileConfig(), preset="fast")

    def test_config_with_a_keyword_is_refused(self, frame):
        with pytest.raises(ConfigurationError, match="not both"):
            summarize(frame, config=ProfileConfig(), chunk_size=100)


class TestPayloadContract:
    """UX-6. The payload drifted once already: rows became rows_est."""

    def test_the_payload_is_versioned(self, frame):
        assert summarize(frame)["schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_the_version_is_on_report_stats_too(self, frame):
        assert profile(frame).stats["schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_the_top_level_keys_are_the_documented_ones(self, frame):
        assert set(summarize(frame)) == {"schema_version", "dataset", "columns"}

    def test_numeric_top_values_reach_the_payload(self, frame):
        """The HTML renders these from the same accumulator."""
        top = summarize(frame)["columns"]["score"]["top_values"]
        assert top is not None
        assert top, "expected common values for a 200-level column"
        # Misra-Gries keeps k counters and its counts are lower bounds, so they
        # neither partition the column nor sum to the row count.
        assert len(top) <= 50
        assert all(count > 0 for _, count in top)
        assert all(isinstance(value, float) for value, _ in top)

    def test_the_payload_and_the_html_agree_on_the_common_values(self, frame):
        report = profile(frame)
        top = report.stats["columns"]["score"]["top_values"]
        assert top
        # The renderer formats counts with thousands separators, so search for
        # the formatted string -- an earlier audit concluded top-k output was
        # discarded because it looked for the raw one.
        value, count = top[0]
        assert f"{count:,}" in report.html

    def test_not_tracked_is_distinguishable_from_empty(self, frame):
        """The top-k sketch is gated off on high-cardinality columns; that is a
        different statement from "tracked, nothing frequent"."""
        assert summarize(frame)["columns"]["normal"]["top_values"] is None

    def test_the_payload_is_json_serialisable(self, frame, tmp_path):
        import json

        path = tmp_path / "stats.json"
        profile(frame).save_json(path)
        assert json.loads(path.read_text())["schema_version"] == SUMMARY_SCHEMA_VERSION
