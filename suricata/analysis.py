"""Analysis functions for computing summary statistics, missing values,
and correlation matrices for Pandas, Dask, and Polars DataFrames.
"""

import pandas as pd
from typing import Union
from .logger import timeit

# Try to import Dask and Polars.
try:
    import dask.dataframe as dd
except ImportError:
    dd = None

try:
    import polars as pl
except ImportError:
    pl = None


@timeit
def summary_statistics(
    df: Union[pd.DataFrame, "dd.DataFrame", "pl.DataFrame"],
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """Compute summary statistics for numeric columns.

    For Dask DataFrames the result is computed and returned as a Pandas DataFrame,
    while for Polars the native .describe() method is used.

    Args:
        df: A Pandas, Dask, or Polars DataFrame.

    Returns:
        Summary statistics as a Pandas DataFrame (for Pandas/Dask) or a Polars DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return df.describe()
    if dd is not None and isinstance(df, dd.DataFrame):
        return df.describe().compute()
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.describe()
    raise TypeError("Unsupported data type for summary_statistics")


@timeit
def missing_values(
    df: Union[pd.DataFrame, "dd.DataFrame", "pl.DataFrame"],
) -> pd.DataFrame:
    """Compute missing value counts and percentages per column.

    For Dask DataFrames the computation is done lazily and then computed,
    while for Polars the missing counts are computed using native methods.

    Args:
        df: A Pandas, Dask, or Polars DataFrame.

    Returns:
        A Pandas DataFrame with 'missing_count' and 'missing_percent'.
    """
    if isinstance(df, pd.DataFrame):
        counts = df.isnull().sum()
        percent = 100 * counts / len(df)
        return pd.DataFrame({"missing_count": counts, "missing_percent": percent})
    if dd is not None and isinstance(df, dd.DataFrame):
        counts = df.isnull().sum().compute()
        percent = 100 * counts / len(df)
        return pd.DataFrame({"missing_count": counts, "missing_percent": percent})
    if pl is not None and isinstance(df, pl.DataFrame):
        null_counts = df.null_count().to_dict()
        total = df.height
        summary = {"column": [], "missing_count": [], "missing_percent": []}
        for col in df.columns:
            summary["column"].append(col)
            count = null_counts.get(col, 0)
            summary["missing_count"].append(count)
            summary["missing_percent"].append(100 * count / total)
        return pd.DataFrame(summary)
    raise TypeError("Unsupported data type for missing_values")


@timeit
def correlation_matrix(
    df: Union[pd.DataFrame, "dd.DataFrame", "pl.DataFrame"],
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """Compute the correlation matrix for numeric columns.

    For Dask DataFrames the result is computed and returned as a Pandas DataFrame,
    while for Polars the native .corr() method is used.

    Args:
        df: A Pandas, Dask, or Polars DataFrame.

    Returns:
        The correlation matrix as a Pandas DataFrame (for Pandas/Dask) or as a Polars DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return df.select_dtypes(include=["number"]).corr()
    if dd is not None and isinstance(df, dd.DataFrame):
        return df.select_dtypes(include=["number"]).corr().compute()
    if pl is not None and isinstance(df, pl.DataFrame):
        # For Polars, we use its native .corr() (assuming numeric columns)
        return df.select(pl.all().exclude(pl.Utf8)).corr()
    raise TypeError("Unsupported data type for correlation_matrix")
