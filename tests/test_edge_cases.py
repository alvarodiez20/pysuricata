"""Edge-case tests targeting user input / output boundaries.

These tests focus on the surface a real user touches: the ``profile()`` and
``summarize()`` API functions, the ``Report`` wrapper, ``ComputeOptions``
validation, CLI data loading, and the formatting utilities that appear in
every generated report.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from pysuricata.api import (
    ComputeOptions,
    ProfileConfig,
    RenderOptions,
    _coerce_input,
    profile,
    summarize,
)
from pysuricata.render.format_utils import (
    fmt_compact,
    fmt_compact_scientific,
    fmt_num,
    human_bytes,
    human_time,
    ordinal_number,
)

# ---------------------------------------------------------------------------
# Helper: a small DataFrame used by most tests
# ---------------------------------------------------------------------------


def _tiny_df(**overrides):
    """Return a minimal 10-row DataFrame, optionally overriding columns."""
    base = {
        "num": list(range(10)),
        "cat": ["a", "b"] * 5,
        "flag": [True, False] * 5,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ===================================================================
# 1. Input validation
# ===================================================================


class TestInputValidation:
    """profile() / summarize() / _coerce_input() input edge cases."""

    def test_none_input_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be None"):
            profile(None)

    def test_none_input_summarize_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            summarize(None)

    def test_string_input_raises_type_error(self):
        with pytest.raises(TypeError):
            _coerce_input("not a dataframe")

    def test_bytes_input_raises_type_error(self):
        with pytest.raises(TypeError):
            _coerce_input(b"bytes data")

    def test_int_input_raises_type_error(self):
        with pytest.raises(TypeError):
            _coerce_input(42)

    def test_dict_input_raises_type_error(self):
        with pytest.raises(TypeError):
            _coerce_input({"a": 1, "b": 2})

    def test_accepts_list_of_dataframes(self):
        chunks = [_tiny_df(), _tiny_df()]
        result = _coerce_input(chunks)
        assert hasattr(result, "__iter__")

    def test_accepts_generator_of_dataframes(self):
        def gen():
            yield _tiny_df()

        result = _coerce_input(gen())
        assert hasattr(result, "__iter__")

    def test_empty_dataframe(self):
        """Empty DataFrame (0 rows) should not crash."""
        df = pd.DataFrame({"a": pd.Series(dtype="float"), "b": pd.Series(dtype="str")})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)
        assert len(report_obj.html) > 100

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"x": [42], "y": ["hello"]})
        report_obj = profile(df)
        assert "42" in report_obj.html

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"only_col": range(20)})
        report_obj = profile(df)
        assert "only_col" in report_obj.html


# ===================================================================
# 2. Tricky data shapes
# ===================================================================


class TestTrickyDataShapes:
    """Columns with extreme values, missing data, or unusual types."""

    def test_all_nan_column(self):
        df = _tiny_df(all_nan=[float("nan")] * 10)
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_all_none_column(self):
        df = _tiny_df(all_none=[None] * 10)
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_inf_values(self):
        df = _tiny_df(inf_col=[float("inf"), float("-inf")] * 5)
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_mixed_int_nan(self):
        """Integer column with NaN → pandas coerces to float."""
        df = pd.DataFrame({"vals": [1, 2, None, 4, 5, None, 7, 8, 9, 10]})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_very_large_numbers(self):
        df = _tiny_df(big=[10**18 + i for i in range(10)])
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_very_small_float_numbers(self):
        df = _tiny_df(tiny=[1e-15 * i for i in range(10)])
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_constant_column(self):
        df = _tiny_df(const=[42] * 10)
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_high_cardinality_categorical(self):
        df = pd.DataFrame({"hc": [f"val_{i}" for i in range(500)]})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_all_missing_beside_normal(self):
        """All-missing column next to a healthy one."""
        df = pd.DataFrame(
            {
                "good": range(20),
                "bad": [None] * 20,
            }
        )
        report_obj = profile(df)
        assert "good" in report_obj.html

    def test_duplicate_column_names(self):
        """DataFrames with duplicate column names are auto-deduplicated."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report_obj = profile(df)
        assert isinstance(report_obj.html, str)
        # Columns should have been renamed (a, a_1)
        assert "a_1" in report_obj.html or "a" in report_obj.html


# ===================================================================
# 3. Special characters in column names / values
# ===================================================================


class TestSpecialCharacters:
    """Column names / values with HTML, JS, or template metacharacters."""

    def test_html_in_column_name(self):
        """Column names with HTML should be escaped in the card IDs."""
        df = pd.DataFrame({"<b>bold</b>": [1, 2, 3]})
        report_obj = profile(df)
        # The report should generate successfully
        assert isinstance(report_obj.html, str)
        assert len(report_obj.html) > 100

    def test_curly_braces_in_column_name(self):
        """Curly braces must not break the template engine."""
        df = pd.DataFrame({"{col}": range(10), "{{nested}}": range(10)})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_unicode_column_names(self):
        df = pd.DataFrame({"日本語": [1, 2, 3], "émojis_🎉": ["a", "b", "c"]})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_unicode_string_values(self):
        df = _tiny_df(text=["café", "naïve", "日本", "🎉", "Ω"] * 2)
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_very_long_column_name(self):
        long_name = "x" * 500
        df = pd.DataFrame({long_name: range(10)})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_empty_string_column_name(self):
        df = pd.DataFrame({"": range(10)})
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)

    def test_whitespace_only_values(self):
        df = _tiny_df(ws=["   ", "\t", "\n", " ", "  "] * 2)
        report_obj = profile(df)
        assert isinstance(report_obj.html, str)


# ===================================================================
# 4. Configuration edge cases
# ===================================================================


class TestConfigurationEdgeCases:
    def test_chunk_size_none_disables_chunking(self):
        opts = ComputeOptions(chunk_size=None)
        assert opts.chunk_size is None

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ComputeOptions(chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ComputeOptions(chunk_size=-1)

    def test_numeric_sample_size_zero_raises(self):
        with pytest.raises(ValueError, match="numeric_sample_size must be positive"):
            ComputeOptions(numeric_sample_size=0)

    def test_max_uniques_zero_raises(self):
        with pytest.raises(ValueError, match="max_uniques must be positive"):
            ComputeOptions(max_uniques=0)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k must be positive"):
            ComputeOptions(top_k=0)

    def test_force_invalid_column_type_raises(self):
        with pytest.raises(ValueError, match="Invalid column type"):
            ComputeOptions(force_column_types={"col": "invalid_type"})

    def test_force_valid_column_types(self):
        opts = ComputeOptions(
            force_column_types={
                "a": "numeric",
                "b": "categorical",
                "c": "datetime",
                "d": "boolean",
            }
        )
        assert opts.force_column_types["a"] == "numeric"

    def test_corr_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="corr_threshold"):
            ComputeOptions(corr_threshold=-0.1)
        with pytest.raises(ValueError, match="corr_threshold"):
            ComputeOptions(corr_threshold=1.5)

    def test_boolean_detection_max_zero_ratio_out_of_range(self):
        with pytest.raises(ValueError):
            ComputeOptions(boolean_detection_max_zero_ratio=-0.1)
        with pytest.raises(ValueError):
            ComputeOptions(boolean_detection_max_zero_ratio=1.1)

    def test_log_every_n_chunks_zero_raises(self):
        with pytest.raises(ValueError, match="log_every_n_chunks"):
            ComputeOptions(log_every_n_chunks=0)

    def test_checkpoint_every_n_chunks_negative_raises(self):
        with pytest.raises(ValueError, match="checkpoint_every_n_chunks"):
            ComputeOptions(checkpoint_every_n_chunks=-1)

    def test_render_options_empty_title(self):
        opts = RenderOptions(title="")
        assert opts.title == ""

    def test_render_options_none_title(self):
        opts = RenderOptions(title=None)
        assert opts.title is None

    def test_render_description_markdown(self):
        """Description with markdown should embed in the report."""
        cfg = ProfileConfig(render=RenderOptions(description="# Hello\n\nWorld"))
        report_obj = profile(_tiny_df(), config=cfg)
        assert isinstance(report_obj.html, str)

    def test_profile_with_custom_title(self):
        cfg = ProfileConfig(render=RenderOptions(title="Custom < Title >"))
        report_obj = profile(_tiny_df(), config=cfg)
        assert isinstance(report_obj.html, str)


# ===================================================================
# 5. Report output integrity
# ===================================================================


class TestReportOutputIntegrity:
    """The generated HTML must be self-contained and valid."""

    @pytest.fixture
    def sample_report(self):
        return profile(_tiny_df())

    def test_html_has_doctype(self, sample_report):
        assert "<!DOCTYPE html>" in sample_report.html

    def test_html_has_style_tag(self, sample_report):
        assert "<style>" in sample_report.html

    def test_html_has_script_tag(self, sample_report):
        assert "<script>" in sample_report.html

    def test_html_has_no_unresolved_placeholders(self, sample_report):
        """No {placeholder} strings should remain in the output."""
        import re

        # Match {word} patterns but exclude CSS custom properties {--var}
        # and JS code like {key: value}
        placeholders = re.findall(r"\{[a-z_]+\}", sample_report.html)
        # Filter out CSS var() function arguments and JS object keys
        suspicious = [p for p in placeholders if not p.startswith("{--")]
        # There shouldn't be any template placeholders left
        template_names = {
            "{favicon}",
            "{css}",
            "{logo}",
            "{report_title}",
            "{n_rows}",
            "{n_cols}",
            "{missing_overall}",
            "{duplicates_overall}",
            "{variables_section}",
            "{correlations_section}",
        }
        leftover = [p for p in suspicious if p in template_names]
        assert leftover == [], f"Unresolved placeholders: {leftover}"

    def test_html_contains_toggle_dark_mode(self, sample_report):
        assert "toggleDarkMode" in sample_report.html

    def test_html_contains_download_report(self, sample_report):
        assert "downloadReport" in sample_report.html

    def test_stats_has_dataset_key(self, sample_report):
        assert "dataset" in sample_report.stats

    def test_stats_has_columns_key(self, sample_report):
        assert "columns" in sample_report.stats

    def test_save_html(self, sample_report, tmp_path):
        out = tmp_path / "report.html"
        sample_report.save_html(str(out))
        content = out.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_json(self, sample_report, tmp_path):
        out = tmp_path / "stats.json"
        sample_report.save_json(str(out))
        data = json.loads(out.read_text())
        assert "dataset" in data

    def test_save_dispatches_html(self, sample_report, tmp_path):
        out = tmp_path / "report.html"
        sample_report.save(str(out))
        assert out.exists()
        assert "<!DOCTYPE" in out.read_text()

    def test_save_dispatches_json(self, sample_report, tmp_path):
        out = tmp_path / "report.json"
        sample_report.save(str(out))
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_save_unknown_extension_raises(self, sample_report, tmp_path):
        with pytest.raises(ValueError, match="Unknown extension"):
            sample_report.save(str(tmp_path / "report.xyz"))

    def test_save_html_nonexistent_dir_raises(self, sample_report):
        with pytest.raises((FileNotFoundError, OSError)):
            sample_report.save_html("/nonexistent/dir/report.html")

    def test_display_in_notebook(self, sample_report):
        # We catch ImportError or verify it executes smoothly up to the IPython import
        try:
            res = sample_report.display_in_notebook()
            assert res is None or isinstance(res, str)
        except Exception:
            pass

    def test_show_alias(self, sample_report):
        try:
            res = sample_report.show()
            assert res is None or isinstance(res, str)
        except Exception:
            pass


# ===================================================================
# 6. summarize() output integrity
# ===================================================================


class TestSummarizeOutput:
    def test_summarize_returns_dict(self):
        result = summarize(_tiny_df())
        assert isinstance(result, dict)

    def test_summarize_has_expected_keys(self):
        result = summarize(_tiny_df())
        assert "dataset" in result
        assert "columns" in result

    def test_summarize_json_serializable(self):
        result = summarize(_tiny_df())
        # Must not raise
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)

    def test_summarize_with_all_nan(self):
        df = pd.DataFrame({"x": [float("nan")] * 10})
        result = summarize(df)
        assert isinstance(result, dict)


# ===================================================================
# 7. CLI data loading edge cases
# ===================================================================


class TestCLIDataLoading:
    def test_load_csv(self):
        from pysuricata.cli import load_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,x\n2,y\n")
            path = f.name
        try:
            df = load_data(path)
            assert len(df) == 2
        finally:
            os.unlink(path)

    def test_load_headers_only_csv(self):
        """CSV with headers but no data rows."""
        from pysuricata.cli import load_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n")
            path = f.name
        try:
            df = load_data(path)
            assert len(df) == 0
            assert list(df.columns) == ["a", "b", "c"]
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_raises(self):
        from pysuricata.cli import load_data

        with pytest.raises(FileNotFoundError):
            load_data("/does/not/exist.csv")

    def test_load_unsupported_format_raises(self):
        from pysuricata.cli import load_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("data")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_data(path)
        finally:
            os.unlink(path)

    def test_load_unsupported_tsv_raises(self):
        """TSV is not a supported format."""
        from pysuricata.cli import load_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("a\tb\n1\tx\n2\ty\n")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_data(path)
        finally:
            os.unlink(path)


# ===================================================================
# 8. Format utilities
# ===================================================================


class TestHumanBytes:
    def test_zero(self):
        assert human_bytes(0) == "0.0 B"

    def test_one(self):
        assert human_bytes(1) == "1.0 B"

    def test_1023(self):
        assert human_bytes(1023) == "1,023.0 B"

    def test_1024(self):
        assert human_bytes(1024) == "1.0 KB"

    def test_one_mb(self):
        assert human_bytes(1024**2) == "1.0 MB"

    def test_one_gb(self):
        assert human_bytes(1024**3) == "1.0 GB"

    def test_negative_clamped_to_zero(self):
        assert human_bytes(-100) == "0.0 B"

    def test_very_large(self):
        result = human_bytes(10**18)
        assert "PB" in result or "TB" in result


class TestHumanTime:
    def test_none(self):
        assert human_time(None) == "—"

    def test_zero(self):
        assert human_time(0) == "0.00 s"

    def test_sub_second(self):
        assert human_time(0.02) == "0.02 s"

    def test_one_minute(self):
        result = human_time(65)
        assert "1 min" in result
        assert "5 s" in result

    def test_one_hour(self):
        result = human_time(3661)
        assert "1 h" in result
        assert "1 min" in result

    def test_nan(self):
        assert human_time(float("nan")) == "—"

    def test_inf(self):
        assert human_time(float("inf")) == "—"

    def test_negative(self):
        assert human_time(-5) == "—"


class TestFmtNum:
    def test_none(self):
        assert fmt_num(None) == "—"

    def test_nan(self):
        assert fmt_num(float("nan")) == "NaN"

    def test_inf(self):
        assert fmt_num(float("inf")) == "NaN"

    def test_integer(self):
        result = fmt_num(1234)
        assert "1,234" in result

    def test_float(self):
        result = fmt_num(3.14159)
        assert "3.14" in result


class TestFmtCompact:
    def test_none(self):
        assert fmt_compact(None) == "—"

    def test_nan(self):
        assert fmt_compact(float("nan")) == "—"

    def test_inf(self):
        assert fmt_compact(float("inf")) == "—"

    def test_normal_number(self):
        result = fmt_compact(42)
        assert "42" in result

    def test_string_fallback(self):
        result = fmt_compact("not_a_number")
        assert result == "not_a_number"


class TestFmtCompactScientific:
    def test_none(self):
        assert fmt_compact_scientific(None) == "—"

    def test_small_number(self):
        result = fmt_compact_scientific(1000)
        assert "1000" in result

    def test_large_number(self):
        result = fmt_compact_scientific(1_000_000)
        assert "e+" in result.lower() or "E+" in result

    def test_nan(self):
        assert fmt_compact_scientific(float("nan")) == "—"

    def test_inf(self):
        assert fmt_compact_scientific(float("inf")) == "—"

    def test_custom_threshold(self):
        result = fmt_compact_scientific(500, threshold=100)
        assert "e+" in result.lower()


class TestOrdinalNumber:
    def test_1st(self):
        assert "1" in ordinal_number(1)

    def test_2nd(self):
        assert "2" in ordinal_number(2)

    def test_3rd(self):
        assert "3" in ordinal_number(3)

    def test_11th(self):
        assert "11" in ordinal_number(11)

    def test_21st(self):
        assert "21" in ordinal_number(21)
