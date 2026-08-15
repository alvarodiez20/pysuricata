"""`pysuricata check` — the gate, its thresholds, and its exit codes.

UX-5 / #76. `profile` and `summarize` both exit 0 no matter what they found,
which is why neither is usable in a pipeline.

Two properties carry the whole feature and are tested first:

* the **same data twice** must produce no findings, or the gate is noise;
* **changed data** must produce an exit code, or the gate is decoration.

Everything after that is about not crying wolf — thresholds loose enough that a
KMV estimate wobbling inside its own error bound does not fail a build.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.check import (
    BASELINE_VERSION,
    Baseline,
    Thresholds,
    compare,
    make_baseline,
    parse_duration,
    read_baseline,
    read_thresholds,
    render_findings,
    write_baseline,
)


def _frame(n: int = 4_000, *, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.standard_normal(n) + shift,
            "region": rng.choice(["north", "south", "east"], n),
            "active": rng.random(n) > 0.5,
            "code": rng.integers(0, 500, n),
        }
    )


@pytest.fixture
def base_summary():
    return summarize(_frame(), seed=0)


@pytest.fixture
def baseline(base_summary):
    return make_baseline(base_summary, source="fixture")


class TestTheGatePasses:
    """The property that makes it usable: no findings on unchanged data."""

    def test_the_same_frame_produces_no_findings(self, baseline):
        again = summarize(_frame(), seed=0)
        assert compare(again, baseline).passed

    def test_a_different_sample_of_the_same_distribution_passes(self, baseline):
        """A rebuild of the same pipeline draws different rows, not different
        shape. The gate must not fire on that."""
        other = summarize(_frame(seed=99), seed=0)
        result = compare(other, baseline)
        assert result.passed, [f.render() for f in result.findings]

    def test_the_columns_compared_are_reported(self, baseline):
        result = compare(summarize(_frame(), seed=0), baseline)
        assert set(result.checked_columns) == {"amount", "region", "active", "code"}

    def test_appending_rows_is_not_drift_by_default(self, baseline):
        doubled = summarize(pd.concat([_frame(), _frame(seed=7)]), seed=0)
        assert compare(doubled, baseline).passed

    def test_row_drift_is_caught_once_asked_for(self, baseline):
        doubled = summarize(pd.concat([_frame(), _frame(seed=7)]), seed=0)
        result = compare(doubled, baseline, Thresholds(max_rows_drift_pct=10.0))
        assert [f.kind for f in result.findings] == ["rows"]

    def test_growth_does_not_fire_the_cardinality_gate(self, baseline):
        """A continuous column's distinct count doubles when the rows double.
        Gating on the count alone fails every build that appends data."""
        doubled = summarize(pd.concat([_frame(), _frame(seed=7)]), seed=0)
        assert not any(
            f.kind == "cardinality" for f in compare(doubled, baseline).findings
        )

    def test_growth_does_not_fire_it_for_a_categorical_either(self, baseline):
        """The opposite shape: a three-level column keeps its count and halves
        its rate, so gating on the rate alone fails the same builds."""
        doubled = summarize(pd.concat([_frame(), _frame(seed=7)]), seed=0)
        assert all(f.column != "region" for f in compare(doubled, baseline).findings)

    def test_a_new_level_is_caught_when_the_row_count_holds(self, baseline):
        """The common CI shape: same query, next day, similar volume. Here the
        rule is exactly as sensitive as gating on the raw count."""
        rng = np.random.default_rng(5)
        frame = _frame()
        frame["region"] = rng.choice(["north", "south", "east", "west"], len(frame))
        result = compare(summarize(frame, seed=0), baseline)
        assert any(
            f.kind == "cardinality" and f.column == "region" for f in result.findings
        )

    def test_a_large_level_change_survives_concurrent_growth(self, baseline):
        rng = np.random.default_rng(5)
        grown = pd.concat([_frame(), _frame(seed=7)], ignore_index=True)
        grown["region"] = rng.choice([f"r{i}" for i in range(12)], len(grown))
        result = compare(summarize(grown, seed=0), baseline)
        assert any(
            f.kind == "cardinality" and f.column == "region" for f in result.findings
        )

    def test_a_small_level_change_during_large_growth_is_not_flagged(self, baseline):
        """The cost of the rule, stated rather than discovered. Three levels
        becoming five while the rows double sits inside the band that pure
        growth can explain, so it passes. `max_rows_drift_pct` is the gate for
        "the volume moved"; this one is for "the shape moved"."""
        rng = np.random.default_rng(5)
        grown = pd.concat([_frame(), _frame(seed=7)], ignore_index=True)
        grown["region"] = rng.choice(["a", "b", "c", "d", "e"], len(grown))
        result = compare(summarize(grown, seed=0), baseline)
        assert not any(f.kind == "cardinality" for f in result.findings)


class TestTheGateFails:
    def test_a_shifted_mean_is_caught(self, baseline):
        result = compare(summarize(_frame(shift=2.0), seed=0), baseline)
        assert not result.passed
        assert any(
            f.kind == "distribution" and f.column == "amount" for f in result.findings
        )

    def test_the_message_says_how_far_it_moved(self, baseline):
        result = compare(summarize(_frame(shift=2.0), seed=0), baseline)
        message = next(f.message for f in result.findings if f.kind == "distribution")
        assert "σ" in message
        assert "limit" in message

    def test_a_dropped_column_is_caught(self, baseline):
        result = compare(summarize(_frame().drop(columns=["code"]), seed=0), baseline)
        assert any(f.kind == "schema" and f.column == "code" for f in result.findings)

    def test_a_retyped_column_is_caught(self, baseline):
        frame = _frame()
        frame["region"] = np.arange(len(frame), dtype=float)
        result = compare(summarize(frame, seed=0), baseline)
        assert any("type changed" in f.message for f in result.findings)

    def test_a_retyped_column_is_not_also_compared_statistically(self, baseline):
        """Comparing a mean against a category count would be noise on top of a
        finding the reader already has."""
        frame = _frame()
        frame["region"] = np.arange(len(frame), dtype=float)
        result = compare(summarize(frame, seed=0), baseline)
        assert [f.column for f in result.findings].count("region") == 1

    def test_a_column_becoming_a_key_is_caught_as_a_retype(self, baseline):
        """`identifier` is a payload type, so "this column is now unique per
        row" arrives as a schema finding rather than a cardinality one."""
        frame = _frame()
        frame["code"] = np.arange(len(frame))
        result = compare(summarize(frame, seed=0), baseline)
        assert any(
            f.kind == "schema" and f.column == "code" and "identifier" in f.message
            for f in result.findings
        )

    def test_new_nulls_are_caught(self, baseline):
        frame = _frame()
        frame.loc[: len(frame) // 2, "amount"] = np.nan
        result = compare(summarize(frame, seed=0), baseline)
        assert any(
            f.kind == "missing" and f.column == "amount" for f in result.findings
        )

    def test_a_flipped_boolean_rate_is_caught(self, baseline):
        rng = np.random.default_rng(3)
        frame = _frame()
        frame["active"] = rng.random(len(frame)) > 0.95
        result = compare(summarize(frame, seed=0), baseline)
        assert any(f.kind == "boolean" for f in result.findings)

    def test_a_widened_spread_is_caught(self, baseline):
        frame = _frame()
        frame["amount"] = frame["amount"] * 5.0
        result = compare(summarize(frame, seed=0), baseline)
        assert any("spread changed" in f.message for f in result.findings)

    def test_a_new_column_passes_by_default(self, baseline):
        frame = _frame()
        frame["extra"] = 1.0
        assert compare(summarize(frame, seed=0), baseline).passed

    def test_a_new_column_fails_when_asked(self, baseline):
        frame = _frame()
        frame["extra"] = 1.0
        result = compare(
            summarize(frame, seed=0), baseline, Thresholds(fail_on_new_column=True)
        )
        assert [f.column for f in result.findings] == ["extra"]

    def test_range_expansion_is_opt_in(self, baseline):
        frame = _frame()
        frame.loc[0, "amount"] = 500.0
        assert not any(
            f.kind == "range"
            for f in compare(summarize(frame, seed=0), baseline).findings
        )
        result = compare(
            summarize(frame, seed=0),
            baseline,
            Thresholds(fail_on_range_expansion=True),
        )
        assert any(f.kind == "range" for f in result.findings)


class TestFreshness:
    """#91. The failure every other check here passes: yesterday's data again.

    When a scheduled job re-runs a stale extract, every distribution matches and
    every column is present, because the data is literally the same.
    """

    @staticmethod
    def _dated(start: str, periods: int = 500) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_time": pd.date_range(start, periods=periods, freq="h"),
                "value": np.arange(float(periods)),
            }
        )

    @pytest.fixture
    def yesterday(self):
        return make_baseline(summarize(self._dated("2026-08-01"), seed=0))

    def test_a_rerun_of_the_same_extract_is_caught(self, yesterday):
        again = summarize(self._dated("2026-08-01"), seed=0)
        result = compare(again, yesterday, Thresholds(require_max_ts_advances=True))
        assert [f.kind for f in result.findings] == ["freshness"]

    def test_the_same_extract_passes_every_other_check(self, yesterday):
        """Which is exactly why this gate has to exist."""
        again = summarize(self._dated("2026-08-01"), seed=0)
        assert compare(again, yesterday).passed

    def test_fresh_data_passes(self, yesterday):
        newer = summarize(self._dated("2026-08-22"), seed=0)
        assert compare(
            newer, yesterday, Thresholds(require_max_ts_advances=True)
        ).passed

    def test_a_backwards_jump_is_reported_as_such(self, yesterday):
        older = summarize(self._dated("2026-07-01"), seed=0)
        result = compare(older, yesterday, Thresholds(require_max_ts_advances=True))
        assert "backwards" in result.findings[0].message

    def test_advancement_is_off_by_default(self, yesterday):
        """A datetime column can be a birth date, not an event time."""
        again = summarize(self._dated("2026-08-01"), seed=0)
        assert compare(again, yesterday).passed

    def test_the_message_names_the_time_it_stopped_at(self, yesterday):
        again = summarize(self._dated("2026-08-01"), seed=0)
        result = compare(again, yesterday, Thresholds(require_max_ts_advances=True))
        assert "UTC" in result.findings[0].message

    def test_max_age_needs_no_baseline(self):
        stats = summarize(self._dated("2020-01-01"), seed=0)
        result = compare(stats, None, Thresholds(max_age="26h"))
        assert [f.column for f in result.findings] == ["event_time"]

    def test_max_age_passes_on_recent_data(self):
        stats = summarize(self._dated("2020-01-01"), seed=0)
        # 2020-01-21 19:00 UTC is the newest value; pretend it is an hour later.
        just_after = 1_579_640_400 + 3_600
        assert compare(stats, None, Thresholds(max_age="26h"), now=just_after).passed

    def test_max_age_is_measured_in_utc(self):
        """Reading epoch values through the runner's local timezone would make
        the gate fail differently depending on where CI runs."""
        stats = summarize(self._dated("2020-01-01"), seed=0)
        newest = stats["columns"]["event_time"]["max_ts"] / 1e9
        # Exactly at the limit passes; a second past it does not.
        assert compare(
            stats, None, Thresholds(max_age=3_600), now=newest + 3_600
        ).passed
        assert not compare(
            stats, None, Thresholds(max_age=3_600), now=newest + 3_601
        ).passed

    def test_a_non_datetime_column_is_never_stale(self):
        frame = pd.DataFrame({"n": np.arange(500.0)})
        assert compare(summarize(frame, seed=0), None, Thresholds(max_age="1s")).passed


class TestDurations:
    @pytest.mark.parametrize(
        "text,seconds",
        [
            ("90s", 90),
            ("90m", 5_400),
            ("26h", 93_600),
            ("3d", 259_200),
            ("2w", 1_209_600),
            ("450", 450),
            (" 12H ", 43_200),
        ],
    )
    def test_the_units_people_write(self, text, seconds):
        assert parse_duration(text) == seconds

    def test_a_number_passes_through_as_seconds(self):
        assert parse_duration(3_600) == 3_600.0

    def test_nonsense_names_the_units_that_work(self):
        with pytest.raises(ValueError, match="s, m, h, d or w"):
            parse_duration("soon")

    def test_a_negative_duration_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            parse_duration("-3h")

    def test_a_duration_string_works_in_the_constructor(self):
        assert Thresholds(max_age="26h").max_age == 93_600

    def test_a_duration_string_works_in_a_file(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"max_age": "2d"}')
        assert read_thresholds(path).max_age == 172_800


class TestAbsoluteThresholds:
    """Gates that need no baseline at all."""

    def test_missing_ceiling_without_a_baseline(self):
        frame = _frame()
        frame.loc[: int(len(frame) * 0.6), "amount"] = np.nan
        result = compare(
            summarize(frame, seed=0), None, Thresholds(max_missing_pct=10.0)
        )
        assert [f.column for f in result.findings] == ["amount"]

    def test_min_rows_without_a_baseline(self):
        result = compare(
            summarize(_frame(200), seed=0), None, Thresholds(min_rows=1_000)
        )
        assert [f.kind for f in result.findings] == ["rows"]

    def test_nothing_configured_means_nothing_to_say(self):
        assert compare(summarize(_frame(), seed=0), None).passed


class TestApproximationIsLabelled:
    """The house rule: an estimate must not be presented as a count."""

    def test_a_cardinality_finding_is_marked_approximate(self, baseline):
        frame = _frame()
        frame["code"] = np.random.default_rng(1).integers(0, 5_000, len(frame))
        result = compare(summarize(frame, seed=0), baseline)
        card = next(f for f in result.findings if f.kind == "cardinality")
        assert card.approximate
        assert "~" in card.message

    def test_a_missing_rate_finding_is_not(self, baseline):
        frame = _frame()
        frame.loc[: len(frame) // 2, "amount"] = np.nan
        result = compare(summarize(frame, seed=0), baseline)
        assert not next(f for f in result.findings if f.kind == "missing").approximate

    def test_the_default_cardinality_threshold_clears_the_sketch_error(self):
        """KMV relative error is ~2.2% at k=2048. A gate set near that fails on
        noise, so the default sits an order of magnitude above it."""
        assert Thresholds().warnings() == []

    def test_a_threshold_inside_the_noise_floor_is_called_out(self):
        notes = Thresholds(max_unique_drift_pct=1.0).warnings()
        assert len(notes) == 1
        assert "KMV" in notes[0]


class TestBaselineFile:
    def test_a_written_baseline_reads_back(self, base_summary, tmp_path):
        path = tmp_path / "b.json"
        write_baseline(make_baseline(base_summary, source="x"), path)
        assert read_baseline(path).source == "x"

    def test_the_round_trip_still_passes_the_gate(self, base_summary, tmp_path):
        path = tmp_path / "b.json"
        write_baseline(make_baseline(base_summary), path)
        assert compare(base_summary, read_baseline(path)).passed

    def test_the_envelope_carries_provenance(self, base_summary, tmp_path):
        path = tmp_path / "b.json"
        write_baseline(
            make_baseline(base_summary, source="s3://bucket/x.parquet"), path
        )
        raw = json.loads(path.read_text())
        assert raw["baseline_version"] == BASELINE_VERSION
        assert raw["pysuricata_version"]
        assert raw["created_at"]

    def test_a_baseline_from_a_future_version_is_refused(self, base_summary, tmp_path):
        path = tmp_path / "b.json"
        write_baseline(make_baseline(base_summary), path)
        raw = json.loads(path.read_text())
        raw["baseline_version"] = BASELINE_VERSION + 1
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="--write-baseline"):
            read_baseline(path)

    def test_a_payload_with_a_stale_schema_is_refused(self, base_summary, tmp_path):
        """The payload drifted once before -- rows became rows_est. A gate
        comparing across that silently compares nothing."""
        path = tmp_path / "b.json"
        write_baseline(make_baseline(base_summary), path)
        raw = json.loads(path.read_text())
        raw["summary"]["schema_version"] = 99
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="schema_version"):
            read_baseline(path)

    def test_an_unrelated_json_file_is_refused(self, tmp_path):
        path = tmp_path / "b.json"
        path.write_text('{"hello": 1}')
        with pytest.raises(ValueError, match="not a pysuricata baseline"):
            read_baseline(path)

    def test_a_bare_payload_works_as_a_baseline(self, base_summary):
        """Useful in a notebook, where nobody wants an envelope."""
        assert compare(base_summary, base_summary).passed

    def test_the_baseline_is_json_serialisable(self, base_summary, tmp_path):
        path = tmp_path / "b.json"
        write_baseline(make_baseline(base_summary), path)
        json.loads(path.read_text())


def _toml_is_readable() -> bool:
    """3.10 has no `tomllib`; `tomli` supplies it if the user installed one."""
    for module in ("tomllib", "tomli"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            continue
        return True
    return False


needs_toml = pytest.mark.skipif(
    not _toml_is_readable(), reason="no TOML parser on this interpreter"
)


class TestThresholdsFile:
    def test_a_json_file(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"max_mean_shift_sigma": 3.0}')
        assert read_thresholds(path).max_mean_shift_sigma == 3.0

    @needs_toml
    def test_a_toml_file(self, tmp_path):
        path = tmp_path / "t.toml"
        path.write_text("max_mean_shift_sigma = 3.0\n")
        assert read_thresholds(path).max_mean_shift_sigma == 3.0

    @needs_toml
    def test_a_thresholds_table_is_unwrapped(self, tmp_path):
        path = tmp_path / "t.toml"
        path.write_text("[thresholds]\nmin_rows = 10\n")
        assert read_thresholds(path).min_rows == 10

    @needs_toml
    def test_a_pyproject_table_is_unwrapped(self, tmp_path):
        path = tmp_path / "pyproject.toml"
        path.write_text(
            '[project]\nname = "x"\n\n[tool.pysuricata.check]\nmin_rows = 5\n'
        )
        assert read_thresholds(path).min_rows == 5

    @pytest.mark.skipif(_toml_is_readable(), reason="this interpreter can read TOML")
    def test_toml_without_a_parser_says_what_to_do(self, tmp_path):
        """On 3.10 the message has to name the way out, or the feature just
        looks broken."""
        path = tmp_path / "t.toml"
        path.write_text("min_rows = 10\n")
        with pytest.raises(ValueError, match="json"):
            read_thresholds(path)

    def test_unset_keys_keep_their_defaults(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"min_rows": 10}')
        assert (
            read_thresholds(path).max_mean_shift_sigma
            == Thresholds().max_mean_shift_sigma
        )

    def test_null_disables_a_check(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"max_mean_shift_sigma": null}')
        assert read_thresholds(path).max_mean_shift_sigma is None

    def test_a_typo_is_an_error_not_a_silent_no_op(self, tmp_path):
        """A misspelled threshold that is ignored loosens the gate without
        saying so, which is the worst failure mode a gate has."""
        path = tmp_path / "t.json"
        path.write_text('{"max_mean_shift": 3.0}')
        with pytest.raises(ValueError, match="unknown threshold"):
            read_thresholds(path)

    def test_the_error_lists_the_real_names(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"max_mean_shift": 3.0}')
        with pytest.raises(ValueError, match="max_mean_shift_sigma"):
            read_thresholds(path)

    def test_a_negative_threshold_is_refused(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"min_rows": -1}')
        with pytest.raises(ValueError, match="negative"):
            read_thresholds(path)

    def test_a_string_where_a_number_belongs_is_refused(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text('{"min_rows": "lots"}')
        with pytest.raises(ValueError, match="must be a number"):
            read_thresholds(path)

    def test_an_unknown_suffix_names_the_ones_that_work(self, tmp_path):
        path = tmp_path / "t.yaml"
        path.write_text("min_rows: 10")
        with pytest.raises(ValueError, match=r"\.json or \.toml"):
            read_thresholds(path)


class TestRendering:
    def test_a_pass_says_how_many_columns_were_compared(self, baseline, base_summary):
        assert "4 columns compared" in render_findings(compare(base_summary, baseline))

    def test_a_failure_counts_the_findings(self, baseline):
        text = render_findings(compare(summarize(_frame(shift=2.0), seed=0), baseline))
        assert text.startswith("check failed")

    def test_each_finding_names_its_column(self, baseline):
        text = render_findings(compare(summarize(_frame(shift=2.0), seed=0), baseline))
        assert "amount:" in text

    def test_a_dataset_level_finding_says_dataset(self, base_summary):
        result = compare(base_summary, None, Thresholds(min_rows=10_000))
        assert "dataset:" in render_findings(result)

    def test_the_result_is_json_serialisable(self, baseline):
        result = compare(summarize(_frame(shift=2.0), seed=0), baseline)
        assert json.loads(json.dumps(result.to_dict(), default=str))["passed"] is False


class TestExitCodes:
    """The whole point. Run through the real entry point, not the functions."""

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pysuricata.cli", "check", *args],
            capture_output=True,
            text=True,
        )

    @pytest.fixture
    def data(self, tmp_path):
        _frame().to_csv(tmp_path / "base.csv", index=False)
        _frame(shift=3.0).to_csv(tmp_path / "drift.csv", index=False)
        return tmp_path

    def test_writing_a_baseline_exits_zero(self, data):
        done = self._run(
            str(data / "base.csv"), "--write-baseline", str(data / "b.json")
        )
        assert done.returncode == 0
        assert (data / "b.json").exists()

    def test_unchanged_data_exits_zero(self, data):
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(str(data / "base.csv"), "--baseline", str(data / "b.json"))
        assert done.returncode == 0, done.stdout + done.stderr

    def test_changed_data_exits_one(self, data):
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(str(data / "drift.csv"), "--baseline", str(data / "b.json"))
        assert done.returncode == 1

    def test_the_output_names_what_moved(self, data):
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(str(data / "drift.csv"), "--baseline", str(data / "b.json"))
        assert "amount" in done.stdout
        assert "mean moved" in done.stdout

    def test_a_missing_file_exits_two_not_one(self, data):
        """A build must be able to tell drift from an outage."""
        done = self._run(str(data / "nope.csv"), "--baseline", str(data / "b.json"))
        assert done.returncode == 2

    def test_no_baseline_argument_exits_two(self, data):
        assert self._run(str(data / "base.csv")).returncode == 2

    def test_warn_only_exits_zero_but_still_reports(self, data):
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(
            str(data / "drift.csv"), "--baseline", str(data / "b.json"), "--warn-only"
        )
        assert done.returncode == 0
        assert "check failed" in done.stdout

    def test_json_output_is_parseable(self, data):
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(
            str(data / "drift.csv"), "--baseline", str(data / "b.json"), "--json"
        )
        payload = json.loads(done.stdout)
        assert payload["passed"] is False
        assert payload["findings"]

    def test_stdout_stays_parseable_with_progress_on(self, data):
        """Progress goes to stderr; --json output must survive a pipe."""
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(
            str(data / "base.csv"), "--baseline", str(data / "b.json"), "--json"
        )
        json.loads(done.stdout)

    def test_a_command_line_threshold_overrides_the_default(self, data):
        self._run(str(data / "base.csv"), "--write-baseline", str(data / "b.json"))
        done = self._run(
            str(data / "base.csv"),
            "--baseline",
            str(data / "b.json"),
            "--min-rows",
            "10000",
        )
        assert done.returncode == 1
        assert "below the required minimum" in done.stdout


class TestMissingStatisticsAreSkipped:
    """A summary that lacks a field must not crash the gate or invent a finding."""

    def test_an_absent_field_is_not_a_finding(self, base_summary):
        stripped = json.loads(json.dumps(base_summary, default=str))
        for stats in stripped["columns"].values():
            stats.pop("std", None)
            stats.pop("unique_est", None)
        assert compare(stripped, Baseline(summary=stripped)).passed

    def test_a_zero_variance_baseline_does_not_divide_by_zero(self):
        constant = pd.DataFrame({"k": np.ones(2_000)})
        before = summarize(constant, seed=0)
        after = summarize(pd.DataFrame({"k": np.ones(2_000) * 2}), seed=0)
        result = compare(after, before)
        assert all(f.kind != "distribution" for f in result.findings)
