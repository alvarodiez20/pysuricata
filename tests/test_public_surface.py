"""The public surface: classification stability, namespace, errors, inputs.

Four of the twelve UX findings meet here. The one that matters most is the
first: a classification rule that changed its answer as the table grew. The test
that pins it profiles the same column at three row counts and asserts the answer
does not move — that test is the fix, the rule change is one line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysuricata
from pysuricata import (
    ProfileConfig,
    PySuricataError,
    Report,
    UnsupportedDataError,
    profile,
    summarize,
)
from pysuricata.compute.processing.inference import (
    should_reclassify_numeric_as_categorical,
)


def _kind(df: pd.DataFrame, column: str) -> str:
    return summarize(df)["columns"][column]["type"]


class TestClassificationIsStableUnderRowCount:
    """UX-1. The ratio arm made the answer depend on how many rows you had."""

    @pytest.mark.parametrize("n_rows", [1_000, 20_000, 200_000])
    def test_a_bounded_integer_stays_numeric_at_every_size(self, n_rows):
        ages = np.random.default_rng(0).integers(18, 85, n_rows)
        assert _kind(pd.DataFrame({"age": ages}), "age") == "numeric"

    @pytest.mark.parametrize("n_rows", [1_000, 20_000, 200_000])
    def test_a_real_category_stays_categorical_at_every_size(self, n_rows):
        ratings = np.random.default_rng(0).integers(1, 6, n_rows)
        assert _kind(pd.DataFrame({"rating": ratings}), "rating") == "categorical"

    def test_the_verdict_does_not_move_across_three_orders_of_magnitude(self):
        rng = np.random.default_rng(0)
        verdicts = {
            n: _kind(pd.DataFrame({"age": rng.integers(18, 85, n)}), "age")
            for n in (1_000, 100_000, 1_000_000)
        }
        assert len(set(verdicts.values())) == 1, verdicts

    def test_continuous_values_are_never_categorical(self):
        """A measurement that repeats is still a measurement."""
        values = np.random.default_rng(0).standard_normal(20_000).round(1)
        assert _kind(pd.DataFrame({"m": values}), "m") == "numeric"


class TestReclassificationRule:
    def test_an_empty_column_is_not_reclassified(self):
        assert should_reclassify_numeric_as_categorical(0, 0) is False

    def test_a_handful_of_levels_is_categorical(self):
        assert should_reclassify_numeric_as_categorical(8, 1_000_000) is True

    def test_the_ceiling_is_fifty_levels(self):
        assert should_reclassify_numeric_as_categorical(50, 1_000_000) is True
        assert should_reclassify_numeric_as_categorical(51, 1_000_000) is False

    def test_the_ceiling_ignores_the_row_count(self):
        for total in (100, 10_000, 10_000_000):
            assert should_reclassify_numeric_as_categorical(67, total) is False
            assert should_reclassify_numeric_as_categorical(12, total) is True

    def test_non_integral_values_are_never_reclassified(self):
        assert (
            should_reclassify_numeric_as_categorical(8, 1_000, int_like=False) is False
        )


class TestPublicNamespace:
    """UX-10."""

    def test_all_is_defined(self):
        assert hasattr(pysuricata, "__all__")

    def test_everything_exported_actually_resolves(self):
        for name in pysuricata.__all__:
            assert hasattr(pysuricata, name), name

    def test_internal_modules_are_not_advertised(self):
        for internal in ("accumulators", "compute", "render", "report", "api"):
            assert internal not in pysuricata.__all__

    def test_the_config_alias_still_works(self):
        assert pysuricata.ReportConfig is ProfileConfig

    def test_py_typed_ships(self):
        from pathlib import Path

        assert (Path(pysuricata.__file__).parent / "py.typed").exists()


class TestReportRepr:
    """UX-10. The dataclass default printed the whole document."""

    def test_repr_is_one_short_line(self):
        report = profile(pd.DataFrame({"a": np.arange(500.0)}))
        assert len(repr(report)) < 120
        assert "\n" not in repr(report)

    def test_repr_names_the_shape(self):
        report = profile(pd.DataFrame({"a": np.arange(300.0), "b": np.arange(300.0)}))
        assert "300 rows" in repr(report)
        assert "2 columns" in repr(report)

    def test_the_html_is_still_reachable(self):
        report = profile(pd.DataFrame({"a": np.arange(100.0)}))
        assert report.html.startswith("<!DOCTYPE html>")
        assert isinstance(report, Report)


class TestErrorsShareOneBase:
    """UX-10. Three exception types for one user mistake."""

    def test_an_unsupported_input_raises_the_library_base(self):
        with pytest.raises(PySuricataError):
            profile(42)

    def test_it_is_still_a_typeerror_for_existing_handlers(self):
        with pytest.raises(TypeError):
            profile(42)

    def test_the_message_names_what_was_passed(self):
        with pytest.raises(PySuricataError, match="int"):
            profile(42)

    def test_a_missing_file_raises_the_library_base(self):
        with pytest.raises(PySuricataError, match="File not found"):
            profile("/nonexistent/nowhere.csv")

    def test_an_unreadable_format_says_which_are_supported(self, tmp_path):
        path = tmp_path / "data.xlsx"
        path.write_bytes(b"not really a spreadsheet")
        with pytest.raises(UnsupportedDataError, match=r"\.csv"):
            profile(path)


class TestPathInput:
    """UX-12. The CLI accepted a path; the API raised TypeError on the same one."""

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "amount": rng.lognormal(3, 1, 500),
                "country": rng.choice(["ES", "FR"], 500),
            }
        )

    def test_a_csv_path_as_a_string(self, tmp_path, frame):
        path = tmp_path / "data.csv"
        frame.to_csv(path, index=False)
        assert summarize(str(path))["dataset"]["rows_est"] == 500

    def test_a_csv_path_as_a_pathlike(self, tmp_path, frame):
        path = tmp_path / "data.csv"
        frame.to_csv(path, index=False)
        assert summarize(path)["dataset"]["rows_est"] == 500

    def test_a_json_path(self, tmp_path, frame):
        path = tmp_path / "data.json"
        frame.to_json(path)
        assert summarize(path)["dataset"]["rows_est"] == 500

    def test_the_path_and_the_frame_agree(self, tmp_path, frame):
        path = tmp_path / "data.csv"
        frame.to_csv(path, index=False)
        config = ProfileConfig()
        from_path = summarize(str(path), config=config)
        from_frame = summarize(pd.read_csv(path), config=config)
        assert from_path["dataset"]["rows_est"] == from_frame["dataset"]["rows_est"]
        assert set(from_path["columns"]) == set(from_frame["columns"])

    def test_a_string_is_not_mistaken_for_an_iterable_of_chunks(self):
        """str is Iterable, so it used to fall through to the chunk path."""
        with pytest.raises(PySuricataError):
            profile("this is not a path")
