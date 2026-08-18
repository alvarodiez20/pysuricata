"""Excel workbooks load from a path, which the library never did (#4).

The browser demo already reads five spreadsheet formats through
`python-calamine` (`web/README.md`) -- the gap was in the library itself:
`profile("data.xlsx")` raised `UnsupportedDataError` while `profile("data.csv")`
and `profile("data.parquet")` both just worked. Publishing widened that gap
rather than revealing it, since the demo's own README already documented the
five formats the library did not read.

`_read_excel` (`pysuricata/api.py`) tries `python-calamine` first -- one
dependency across all five formats, and the engine the demo settled on for
the same reason -- and falls back to pandas' own per-format engine
(openpyxl, xlrd, pyxlsb, odfpy) when calamine is not installed or the
installed pandas predates its support (added in pandas 2.2; this project's
floor is 2.0).
"""

from __future__ import annotations

import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.api import _read_excel

openpyxl = pytest.importorskip("openpyxl", reason="writes the .xlsx fixtures below")

ROWS = 200


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [i * 1.5 for i in range(ROWS)],
            "region": [f"r{i % 7}" for i in range(ROWS)],
        }
    )


@pytest.fixture
def workbook(frame: pd.DataFrame, tmp_path) -> str:
    path = tmp_path / "data.xlsx"
    frame.to_excel(path, index=False, sheet_name="data")
    return str(path)


def _rows(payload: dict) -> int:
    return payload["dataset"]["rows_est"]


class TestEveryEntryPointLoads:
    def test_summarize_reads_a_path(self, workbook):
        assert _rows(summarize(workbook, seed=0)) == ROWS

    def test_profile_reads_a_path_and_renders(self, workbook):
        report = profile(workbook, seed=0)

        assert _rows(report.stats) == ROWS
        assert "<style" in report.html

    def test_the_columns_come_through(self, workbook, frame):
        result = summarize(workbook, seed=0)

        assert set(result["columns"]) == set(frame.columns)

    def test_the_cli_loads_it(self, workbook):
        from pysuricata.cli import load_data

        loaded = load_data(workbook)

        assert len(loaded) == ROWS


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm"])
class TestOfficeOpenXmlSuffixes:
    """`.xlsm` is the same container format as `.xlsx` with macros allowed --
    same reader, so one test parametrised over both rather than two copies."""

    def test_a_path_loads(self, frame, tmp_path, suffix):
        path = tmp_path / f"data{suffix}"
        frame.to_excel(path, index=False)

        assert _rows(summarize(str(path), seed=0)) == ROWS


class TestOnlyTheFirstSheetIsRead:
    """`profile()` is a one-shot call over one table with no prompt to put a
    sheet chooser behind, unlike the demo, which pauses and asks
    (`web/README.md`). Silently taking the first sheet must actually mean
    the first sheet, not an arbitrary one."""

    def test_a_later_sheet_is_not_what_gets_profiled(self, tmp_path):
        path = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"a": [1, 2, 3]}).to_excel(
                writer, index=False, sheet_name="first"
            )
            pd.DataFrame({"a": [1] * 500, "b": [2] * 500}).to_excel(
                writer, index=False, sheet_name="second"
            )

        result = summarize(str(path), seed=0)

        assert _rows(result) == 3
        assert set(result["columns"]) == {"a"}


class TestTheEngineFallsBackWithoutCalamine:
    """The candidate-resolution logic itself, isolated from whether calamine
    happens to be installed in the environment running this suite."""

    def test_calamine_is_tried_first_when_available(self, workbook, monkeypatch):
        calls = []
        real_read_excel = pd.read_excel

        def spy(*args, **kwargs):
            calls.append(kwargs.get("engine"))
            return real_read_excel(*args, **kwargs)

        monkeypatch.setattr(pd, "read_excel", spy)

        _read_excel(workbook)

        assert calls == ["calamine"]

    def test_missing_calamine_falls_back_to_pandas_default(self, workbook, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "python_calamine":
                raise ImportError("no calamine here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        result = _read_excel(workbook)

        assert len(result) == ROWS

    def test_a_pandas_that_predates_calamine_support_still_reads_the_file(
        self, workbook, monkeypatch
    ):
        """`engine="calamine"` on a pandas older than 2.2 raises `ValueError`
        (unknown engine name), not `ImportError` -- calamine itself is fine,
        pandas just does not know it yet. That must retry, not propagate."""
        real_read_excel = pd.read_excel
        calls = []

        def fake_read_excel(*args, **kwargs):
            calls.append(kwargs.get("engine"))
            if kwargs.get("engine") == "calamine":
                raise ValueError("Unknown engine: calamine")
            return real_read_excel(*args, **{k: v for k, v in kwargs.items()})

        monkeypatch.setattr(pd, "read_excel", fake_read_excel)

        result = _read_excel(workbook)

        assert calls == ["calamine", None]
        assert len(result) == ROWS

    def test_no_engine_at_all_raises_a_readable_error(self, workbook, monkeypatch):
        def always_missing_engine(*args, **kwargs):
            raise ImportError("Missing optional dependency 'openpyxl'")

        monkeypatch.setattr(pd, "read_excel", always_missing_engine)

        with pytest.raises(ImportError, match="python-calamine"):
            _read_excel(workbook)

    def test_a_value_error_with_no_calamine_to_blame_still_propagates(
        self, workbook, monkeypatch
    ):
        """The retry is specifically for `engine="calamine"` being unknown to
        an old pandas. A `ValueError` with `engine=None` already -- nothing
        left to fall back to -- has to reach the caller, not vanish."""
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "python_calamine":
                raise ImportError("no calamine here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        def raises_for_an_unrelated_reason(*args, **kwargs):
            assert kwargs.get("engine") is None
            raise ValueError("Excel file format cannot be determined")

        monkeypatch.setattr(pd, "read_excel", raises_for_an_unrelated_reason)

        with pytest.raises(ValueError, match="cannot be determined"):
            _read_excel(workbook)


class TestErrorsMatchTheOtherFormats:
    def test_a_missing_file_is_reported_before_any_reader_runs(self, tmp_path):
        from pysuricata.api import PySuricataError

        with pytest.raises(PySuricataError, match="not found"):
            summarize(str(tmp_path / "nope.xlsx"), seed=0)

    def test_the_cli_reports_a_missing_workbook_as_file_not_found(self, tmp_path):
        from pysuricata.cli import load_data

        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nope.xlsx"))
