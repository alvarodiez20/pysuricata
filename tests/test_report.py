import os
import tempfile
from pathlib import Path

import pytest

from pysuricata.config import EngineConfig
from pysuricata.report import ReportOrchestrator, build_report


def _has_pandas():
    try:
        import pandas as pd  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_basic_html():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    html = build_report(df, config=EngineConfig(), compute_only=False)
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_with_summary():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})
    html, summary = build_report(
        df, config=EngineConfig(), return_summary=True, compute_only=False
    )
    assert isinstance(html, str)
    assert isinstance(summary, dict)
    assert "dataset" in summary and "columns" in summary


# Enhanced edge case tests
@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_empty_dataframe():
    """Test build_report with empty DataFrame."""
    import pandas as pd

    df = pd.DataFrame()
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_single_row():
    """Test build_report with single row DataFrame."""
    import pandas as pd

    df = pd.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_all_nan_column():
    """Test build_report with column containing all NaN values."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_mixed_types():
    """Test build_report with DataFrame containing mixed data types."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "date_col": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "mixed_col": [1, "a", 2.5],
        }
    )
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_very_large_numbers():
    """Test build_report with very large numbers."""
    import pandas as pd

    # Use smaller numbers to avoid numpy type issues
    df = pd.DataFrame(
        {
            "large_int": [10**15, 10**16, 10**17],
            "large_float": [1e15, 1e16, 1e17],
            "small_float": [1e-15, 1e-16, 1e-17],
        }
    )
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_unicode_strings():
    """Test build_report with unicode strings."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "ascii": ["hello", "world"],
            "unicode": ["你好", "世界"],
            "emoji": ["😀", "🚀"],
            "special": ["αβγ", "∑∏∫"],
        }
    )
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_very_long_strings():
    """Test build_report with very long strings."""
    import pandas as pd

    long_string = "a" * 10000
    df = pd.DataFrame(
        {"long_strings": [long_string, long_string + "b", long_string + "c"]}
    )
    html = build_report(df, config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_with_custom_config():
    """Test build_report with custom configuration."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Test with custom config
    config = EngineConfig(
        chunk_size=1000,
        numeric_sample_k=100,
        uniques_k=512,
        topk_k=10,
        random_seed=42,
        title="Custom Report",
    )
    html = build_report(df, config=config)
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_compute_only():
    """Test build_report with compute_only=True."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    html = build_report(df, config=EngineConfig(), compute_only=True)
    assert html == ""  # Should return empty string when compute_only=True


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_with_output_file():
    """Test build_report with output file."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_file = f.name

    try:
        html = build_report(df, config=EngineConfig(), output_file=output_file)
        assert isinstance(html, str)
        assert "<html" in html.lower()

        # Check that file was written
        assert os.path.exists(output_file)
        with open(output_file, "r") as f:
            file_content = f.read()
        assert "<html" in file_content.lower()
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_with_report_title():
    """Test build_report with custom report title."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    html = build_report(df, config=EngineConfig(), report_title="My Custom Title")
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "My Custom Title" in html


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_with_iterable():
    """Test build_report with iterable of DataFrames."""
    import pandas as pd

    chunks = [
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
    ]
    html = build_report(iter(chunks), config=EngineConfig())
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_build_report_invalid_type():
    """Test build_report with invalid input type."""
    with pytest.raises(TypeError):
        build_report(123, config=EngineConfig())


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_class():
    """Test ReportOrchestrator class directly."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Test with default config
    orchestrator = ReportOrchestrator()
    html = orchestrator.build_report(df)
    assert isinstance(html, str)
    assert "<html" in html.lower()

    # Test with custom config
    config = EngineConfig(chunk_size=1000, random_seed=42)
    orchestrator = ReportOrchestrator(config)
    html = orchestrator.build_report(df)
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_with_summary():
    """Test ReportOrchestrator with return_summary=True."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    orchestrator = ReportOrchestrator()
    html, summary = orchestrator.build_report(df, return_summary=True)

    assert isinstance(html, str)
    assert isinstance(summary, dict)
    assert "dataset" in summary and "columns" in summary
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_compute_only():
    """Test ReportOrchestrator with compute_only=True."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    orchestrator = ReportOrchestrator()
    html = orchestrator.build_report(df, compute_only=True)

    assert html == ""  # Should return empty string when compute_only=True


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_with_output_file():
    """Test ReportOrchestrator with output file."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        output_file = f.name

    try:
        orchestrator = ReportOrchestrator()
        html = orchestrator.build_report(df, output_file=output_file)
        assert isinstance(html, str)
        assert "<html" in html.lower()

        # Check that file was written
        assert os.path.exists(output_file)
        with open(output_file, "r") as f:
            file_content = f.read()
        assert "<html" in file_content.lower()
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_with_report_title():
    """Test ReportOrchestrator with custom report title."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    orchestrator = ReportOrchestrator()
    html = orchestrator.build_report(df, report_title="Orchestrator Test Title")

    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "Orchestrator Test Title" in html


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_edge_cases():
    """Test ReportOrchestrator with various edge cases."""
    import pandas as pd

    orchestrator = ReportOrchestrator()

    # Test with empty DataFrame
    df_empty = pd.DataFrame()
    html = orchestrator.build_report(df_empty)
    assert isinstance(html, str)
    assert "<html" in html.lower()

    # Test with single row
    df_single = pd.DataFrame({"a": [1]})
    html = orchestrator.build_report(df_single)
    assert isinstance(html, str)
    assert "<html" in html.lower()

    # Test with all NaN column
    df_nan = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    html = orchestrator.build_report(df_nan)
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_config_validation():
    """Test ReportOrchestrator with various configuration validations."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Test with very small chunk size
    config = EngineConfig(chunk_size=1)
    orchestrator = ReportOrchestrator(config)
    html = orchestrator.build_report(df)
    assert isinstance(html, str)
    assert "<html" in html.lower()

    # Test with very small sample size
    config = EngineConfig(numeric_sample_k=1)
    orchestrator = ReportOrchestrator(config)
    html = orchestrator.build_report(df)
    assert isinstance(html, str)
    assert "<html" in html.lower()

    # Test with very small max_uniques
    config = EngineConfig(uniques_k=1)
    orchestrator = ReportOrchestrator(config)
    html = orchestrator.build_report(df)
    assert isinstance(html, str)
    assert "<html" in html.lower()

    # Test with very small top_k
    config = EngineConfig(topk_k=1)
    orchestrator = ReportOrchestrator(config)
    html = orchestrator.build_report(df)
    assert isinstance(html, str)
    assert "<html" in html.lower()


@pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
def test_report_orchestrator_random_seed():
    """Test ReportOrchestrator with different random seeds."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})

    # Test with same seed produces valid results
    config1 = EngineConfig(random_seed=42)
    config2 = EngineConfig(random_seed=42)

    orchestrator1 = ReportOrchestrator(config1)
    orchestrator2 = ReportOrchestrator(config2)

    html1 = orchestrator1.build_report(df)
    html2 = orchestrator2.build_report(df)

    # Both should produce valid HTML
    assert isinstance(html1, str)
    assert isinstance(html2, str)
    assert "<html" in html1.lower()
    assert "<html" in html2.lower()

    # Test with different seed produces valid results
    config3 = EngineConfig(random_seed=123)
    orchestrator3 = ReportOrchestrator(config3)
    html3 = orchestrator3.build_report(df)

    # Should produce valid HTML
    assert isinstance(html3, str)
    assert "<html" in html3.lower()
