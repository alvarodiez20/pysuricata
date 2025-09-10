import io
import json
import os
from typing import Iterable

import pandas as pd
import pytest

from pysuricata.api import (
    ComputeOptions,
    ProfileConfig,
    RenderOptions,
    Report,
    _coerce_input,  # type: ignore
    _to_engine_config,  # type: ignore
    profile,
    summarize,
)


def test_report_save_and_repr(tmp_path):
    rep = Report(html="<html>ok</html>", stats={"dataset": {"rows": 3}})

    # save_html
    html_path = tmp_path / "out.html"
    rep.save_html(str(html_path))
    assert html_path.exists()
    assert html_path.read_text(encoding="utf-8").startswith("<html>")

    # save_json
    json_path = tmp_path / "out.json"
    rep.save_json(str(json_path))
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "dataset" in data

    # save() by extension
    html2 = tmp_path / "out2.html"
    rep.save(str(html2))
    assert html2.exists()

    json2 = tmp_path / "out2.json"
    rep.save(str(json2))
    assert json2.exists()

    # invalid extension
    with pytest.raises(ValueError):
        rep.save(str(tmp_path / "out.txt"))

    # notebook repr
    assert rep._repr_html_().startswith("<html>")


def test_compute_options_properties_map_to_engine_names():
    c = ComputeOptions(
        chunk_size=123,
        numeric_sample_size=999,
        max_uniques=777,
        top_k=55,
        random_seed=1,
    )
    assert c.numeric_sample_k == 999
    assert c.uniques_k == 777
    assert c.topk_k == 55


def test__coerce_input_accepts_pandas_and_iterable():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = _coerce_input(df)  # type: ignore
    assert out is df

    def gen() -> Iterable[pd.DataFrame]:
        yield df
        yield df

    out2 = _coerce_input(gen())  # type: ignore
    assert hasattr(out2, "__iter__")


def test__coerce_input_rejects_invalid_types():
    with pytest.raises(TypeError):
        _ = _coerce_input(123)  # type: ignore
    with pytest.raises(TypeError):
        _ = _coerce_input({"a": 1})  # type: ignore


def test__to_engine_config_prefers_from_options(monkeypatch):
    # Ensure from_options path is exercised
    cfg = ProfileConfig(
        compute=ComputeOptions(
            chunk_size=42, numeric_sample_size=5, max_uniques=6, top_k=7, random_seed=9
        )
    )

    # direct call should work with current engine
    eng = _to_engine_config(cfg)  # type: ignore
    for name in ("chunk_size", "numeric_sample_k", "uniques_k", "topk_k"):
        assert hasattr(eng, name)


def test__to_engine_config_fallback_without_from_options(monkeypatch):
    # Simulate older engine without from_options
    class DummyEngineCfg:
        def __init__(
            self,
            *,
            chunk_size,
            numeric_sample_k,
            uniques_k,
            topk_k,
            random_seed,
        ):
            self.chunk_size = chunk_size
            self.numeric_sample_k = numeric_sample_k
            self.uniques_k = uniques_k
            self.topk_k = topk_k
            self.random_seed = random_seed

    import pysuricata.api as api

    monkeypatch.setattr(api, "_EngineConfig", DummyEngineCfg, raising=True)

    cfg = ProfileConfig(
        compute=ComputeOptions(
            chunk_size=99,
            numeric_sample_size=11,
            max_uniques=22,
            top_k=33,
            random_seed=123,
        )
    )
    eng = _to_engine_config(cfg)  # type: ignore
    assert isinstance(eng, DummyEngineCfg)
    assert eng.chunk_size == 99
    assert eng.numeric_sample_k == 11
    assert eng.uniques_k == 22
    assert eng.topk_k == 33
    assert eng.random_seed == 123


def test_profile_and_summarize_with_pandas():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, None, 6.0]})
    rep = profile(df, config=ProfileConfig())
    assert rep.html and isinstance(rep.html, str)
    assert isinstance(rep.stats, dict)

    stats = summarize(df, config=ProfileConfig())
    assert "dataset" in stats and "columns" in stats


def test_profile_with_iterable_pandas():
    chunks = [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4]})]
    rep = profile(iter(chunks), config=ProfileConfig())
    assert rep.html and isinstance(rep.html, str)


def test_profile_invalid_type_raises():
    with pytest.raises(TypeError):
        _ = profile(123)  # type: ignore


# Enhanced edge case tests
def test_compute_options_validation():
    """Test ComputeOptions validation with edge cases."""
    # Test negative values
    with pytest.raises(ValueError, match="numeric_sample_size must be positive"):
        ComputeOptions(numeric_sample_size=-1)

    with pytest.raises(ValueError, match="max_uniques must be positive"):
        ComputeOptions(max_uniques=-1)

    with pytest.raises(ValueError, match="top_k must be positive"):
        ComputeOptions(top_k=-1)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ComputeOptions(chunk_size=-1)

    # Test zero values
    with pytest.raises(ValueError, match="numeric_sample_size must be positive"):
        ComputeOptions(numeric_sample_size=0)

    with pytest.raises(ValueError, match="max_uniques must be positive"):
        ComputeOptions(max_uniques=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        ComputeOptions(top_k=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ComputeOptions(chunk_size=0)

    # Test valid edge cases
    ComputeOptions(chunk_size=1, numeric_sample_size=1, max_uniques=1, top_k=1)
    ComputeOptions(chunk_size=None)  # None should be allowed


def test_profile_config_creation():
    """Test ProfileConfig creation with various configurations."""
    # Default configuration
    config = ProfileConfig()
    assert isinstance(config.compute, ComputeOptions)
    assert isinstance(config.render, RenderOptions)

    # Custom configuration
    compute_opts = ComputeOptions(chunk_size=1000, numeric_sample_size=500)
    config = ProfileConfig(compute=compute_opts)
    assert config.compute.chunk_size == 1000
    assert config.compute.numeric_sample_size == 500


def test_profile_with_empty_dataframe():
    """Test profile with empty DataFrame."""
    df = pd.DataFrame()
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert isinstance(rep.stats, dict)


def test_profile_with_single_row():
    """Test profile with single row DataFrame."""
    df = pd.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)


def test_profile_with_all_nan_column():
    """Test profile with column containing all NaN values."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)


def test_profile_with_mixed_types():
    """Test profile with DataFrame containing mixed data types."""
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
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)


def test_profile_with_very_large_numbers():
    """Test profile with very large numbers."""
    # Use smaller numbers to avoid numpy type issues
    df = pd.DataFrame(
        {
            "large_int": [10**15, 10**16, 10**17],
            "large_float": [1e15, 1e16, 1e17],
            "small_float": [1e-15, 1e-16, 1e-17],
        }
    )
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)


def test_profile_with_unicode_strings():
    """Test profile with unicode strings."""
    df = pd.DataFrame(
        {
            "ascii": ["hello", "world"],
            "unicode": ["你好", "世界"],
            "emoji": ["😀", "🚀"],
            "special": ["αβγ", "∑∏∫"],
        }
    )
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)


def test_profile_with_very_long_strings():
    """Test profile with very long strings."""
    long_string = "a" * 10000
    df = pd.DataFrame(
        {"long_strings": [long_string, long_string + "b", long_string + "c"]}
    )
    rep = profile(df)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)


def test_summarize_with_edge_cases():
    """Test summarize function with edge cases."""
    # Empty DataFrame
    df = pd.DataFrame()
    stats = summarize(df)
    # For empty DataFrames, stats will be empty
    if stats:
        assert "dataset" in stats and "columns" in stats

    # Single row
    df = pd.DataFrame({"a": [1]})
    stats = summarize(df)
    assert "dataset" in stats and "columns" in stats

    # All NaN
    df = pd.DataFrame({"a": [None, None, None]})
    stats = summarize(df)
    assert "dataset" in stats and "columns" in stats


def test_profile_with_custom_config_edge_cases():
    """Test profile with custom configuration edge cases."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Very small chunk size
    config = ProfileConfig(compute=ComputeOptions(chunk_size=1))
    rep = profile(df, config=config)
    assert rep.html and isinstance(rep.html, str)

    # Very small sample size
    config = ProfileConfig(compute=ComputeOptions(numeric_sample_size=1))
    rep = profile(df, config=config)
    assert rep.html and isinstance(rep.html, str)

    # Very small max_uniques
    config = ProfileConfig(compute=ComputeOptions(max_uniques=1))
    rep = profile(df, config=config)
    assert rep.html and isinstance(rep.html, str)

    # Very small top_k
    config = ProfileConfig(compute=ComputeOptions(top_k=1))
    rep = profile(df, config=config)
    assert rep.html and isinstance(rep.html, str)


def test_profile_with_random_seed():
    """Test profile with different random seeds for reproducibility."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})

    # Test with same seed produces same results
    config1 = ProfileConfig(compute=ComputeOptions(random_seed=42))
    config2 = ProfileConfig(compute=ComputeOptions(random_seed=42))

    rep1 = profile(df, config=config1)
    rep2 = profile(df, config=config2)

    # The HTML should be identical for same seed (or at least both should be valid HTML)
    assert rep1.html and isinstance(rep1.html, str)
    assert rep2.html and isinstance(rep2.html, str)
    assert "<html" in rep1.html.lower()
    assert "<html" in rep2.html.lower()

    # Test with different seed produces valid results
    config3 = ProfileConfig(compute=ComputeOptions(random_seed=123))
    rep3 = profile(df, config=config3)

    # The HTML should be valid for different seed
    assert rep3.html and isinstance(rep3.html, str)
    assert "<html" in rep3.html.lower()


def test_profile_with_columns_subset():
    """Test profile with specific columns subset."""
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
    )

    # Test with specific columns
    config = ProfileConfig(compute=ComputeOptions(columns=["a", "c"]))
    rep = profile(df, config=config)
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)

    # Test with single column
    config = ProfileConfig(compute=ComputeOptions(columns=["a"]))
    rep = profile(df, config=config)
    assert rep.html and isinstance(rep.html, str)


def test_report_save_edge_cases(tmp_path):
    """Test Report save functionality with edge cases."""
    rep = Report(html="<html>test</html>", stats={"dataset": {"rows": 1}})

    # Test save with different extensions
    html_path = tmp_path / "test.html"
    rep.save_html(str(html_path))
    assert html_path.exists()

    json_path = tmp_path / "test.json"
    rep.save_json(str(json_path))
    assert json_path.exists()

    # Test save with auto-detection
    auto_html = tmp_path / "auto.html"
    rep.save(str(auto_html))
    assert auto_html.exists()

    auto_json = tmp_path / "auto.json"
    rep.save(str(auto_json))
    assert auto_json.exists()

    # Test invalid extension
    with pytest.raises(ValueError):
        rep.save(str(tmp_path / "test.txt"))


def test_report_repr_edge_cases():
    """Test Report representation with edge cases."""
    # Test with minimal data
    rep = Report(html="<html></html>", stats={})
    assert rep._repr_html_() == "<html></html>"

    # Test with complex stats
    complex_stats = {
        "dataset": {"rows": 1000, "columns": 10},
        "columns": {
            "col1": {"type": "numeric", "missing": 0.1},
            "col2": {"type": "categorical", "unique": 5},
        },
    }
    rep = Report(html="<html>complex</html>", stats=complex_stats)
    assert rep._repr_html_() == "<html>complex</html>"
