import pandas as pd

from pysuricata import profile, summarize, ReportConfig


def test_profile_returns_html_and_stats():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rep = profile(df, config=ReportConfig())
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)
    assert rep.stats.get("dataset") is not None
    assert rep.stats.get("columns") is not None


def test_summarize_stats_only():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    stats = summarize(df)
    assert "dataset" in stats and "columns" in stats
