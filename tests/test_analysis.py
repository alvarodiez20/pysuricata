import pandas as pd
from pysuricata.analysis import summary_statistics, missing_values, correlation_matrix


def test_summary_statistics():
    """Test that summary_statistics returns a DataFrame with expected columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    stats = summary_statistics(df)
    assert "a" in stats.columns


def test_missing_values():
    """Test that missing_values correctly computes missing value counts."""
    df = pd.DataFrame({"a": [1, None, 3]})
    missing = missing_values(df)
    assert missing["missing_count"].iloc[0] == 1


def test_correlation_matrix():
    """Test that correlation_matrix returns perfect correlation for identical columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    corr = correlation_matrix(df)
    assert corr.loc["a", "b"] == 1.0
