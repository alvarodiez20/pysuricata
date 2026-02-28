import numpy as np
import pandas as pd
import polars as pl

from pysuricata.compute.analysis.correlation import StreamingCorr


def test_streaming_corr_pandas_basic():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],  # Perfect correlation (r=1.0)
            "C": [5, 4, 3, 2, 1],  # Perfect negative correlation (r=-1.0)
            "D": [1, 0, 1, 0, 1],  # Weak/no correlation
        }
    )

    corr = StreamingCorr(["A", "B", "C", "D"])
    corr.update_from_pandas(df)

    # Check default threshold (0.5), we expect A-B and A-C to pass
    res = corr.top_map(threshold=0.5, max_per_col=5)

    # A should correlate perfectly with B (1.0) and C (-1.0)
    assert len(res["A"]) == 2
    b_corr = next(val for col, val in res["A"] if col == "B")
    c_corr = next(val for col, val in res["A"] if col == "C")

    assert np.isclose(b_corr, 1.0)
    assert np.isclose(c_corr, -1.0)

    # Check D has no strong correlations
    assert len(res["D"]) == 0


def test_streaming_corr_pandas_chunks():
    # Test that chunking data produces the same result as all at once
    df1 = pd.DataFrame({"X": [1, 2], "Y": [1, 2]})
    df2 = pd.DataFrame({"X": [3, 4], "Y": [3, 4]})

    corr = StreamingCorr(["X", "Y"])
    corr.update_from_pandas(df1)
    corr.update_from_pandas(df2)

    res = corr.top_map()
    assert np.isclose(res["X"][0][1], 1.0)


def test_streaming_corr_pandas_missing_data():
    # Test handling of NaNs and mixed types that pandas fallback handles
    df = pd.DataFrame(
        {
            "A": [1, np.nan, 3, 4],
            "B": [1, 2, 3, np.nan],
            "C": ["1", "nan", "3", "bad"],  # Force the except block fallback
        }
    )

    corr = StreamingCorr(["A", "B", "C"])
    corr.update_from_pandas(df)

    # The valid overlapping masks are minimal but top_map shouldn't crash
    res = corr.top_map(threshold=0.0)
    assert "A" in res


def test_streaming_corr_polars_basic():
    df = pl.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10], "C": [5, 4, 3, 2, 1]}
    )

    corr = StreamingCorr(["A", "B", "C"])
    corr.update_from_polars(df)

    res = corr.top_map(threshold=0.5)

    b_corr = next(val for col, val in res["A"] if col == "B")
    assert np.isclose(b_corr, 1.0)


def test_streaming_corr_polars_bad_data():
    # Covers Polars casting fallback
    df = pl.DataFrame({"X": [1.0, 2.0, None], "Y": ["1", "not_a_number", "3"]})
    corr = StreamingCorr(["X", "Y"])
    corr.update_from_polars(df)
    res = corr.top_map(threshold=0.0)
    assert "X" in res


def test_streaming_corr_zero_variance():
    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [2, 2, 2, 2]})
    corr = StreamingCorr(["A", "B"])
    corr.update_from_pandas(df)
    res = corr.top_map(threshold=0.0)
    # Zero variance standard deviation denominator should safely yield 0.0
    assert len(res["A"]) == 1
    assert res["A"][0][1] == 0.0


def test_streaming_corr_no_valid_pairs():
    # Nothing valid
    df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
    corr = StreamingCorr(["A", "B"])
    corr.update_from_pandas(df)
    res = corr.top_map()
    assert len(res["A"]) == 0


def test_streaming_corr_too_few_cols():
    corr = StreamingCorr(["A"])
    df = pd.DataFrame({"A": [1, 2, 3]})

    corr.update_from_pandas(df)
    corr.update_from_polars(pl.DataFrame({"A": [1, 2, 3]}))

    res = corr.top_map()
    assert len(res["A"]) == 0
