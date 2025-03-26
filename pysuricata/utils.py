"""Utility functions for converting inputs and rendering HTML."""

import pandas as pd
import numpy as np
from typing import Union, Optional, List

# Support only Pandas, Dask, and Polars.
try:
    import dask.dataframe as dd
except ImportError:
    dd = None

try:
    import polars as pl
except ImportError:
    pl = None


def to_dataframe(
    data: Union[pd.DataFrame, np.ndarray, "dd.DataFrame", "pl.DataFrame"],
    columns: Optional[List[str]] = None,
) -> Union[pd.DataFrame, "dd.DataFrame", "pl.DataFrame"]:
    """Convert input data to a DataFrame-like object.

    Supported types:
      - Pandas DataFrame: returned unchanged.
      - Dask DataFrame: returned unchanged.
      - Polars DataFrame: returned unchanged.
      - NumPy ndarray: converted to a Pandas DataFrame (with optional column names).

    Args:
        data: The input data.
        columns: Optional list of column names (for 2D NumPy arrays).

    Returns:
        A DataFrame-like object.

    Raises:
        TypeError: If the data type is unsupported.
        ValueError: If the NumPy array is not 2D or columns length mismatches.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if dd is not None and isinstance(data, dd.DataFrame):
        return data
    if pl is not None and isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Only 2D arrays can be converted")
        use_cols = (
            columns
            if columns is not None
            else [f"col_{i}" for i in range(data.shape[1])]
        )
        if len(use_cols) != data.shape[1]:
            raise ValueError("Length of columns must match the number of array columns")
        return pd.DataFrame(data, columns=use_cols)
    raise TypeError("Unsupported data type for to_dataframe")


def df_to_html(df: Union[pd.DataFrame, "dd.DataFrame", "pl.DataFrame"]) -> str:
    """Convert a DataFrame-like object to an HTML table.

    For Dask DataFrames, the data is computed; for Polars, conversion is done via its .to_pandas().

    Args:
        df: A Pandas, Dask, or Polars DataFrame.

    Returns:
        A string containing the HTML table.
    """
    if dd is not None and isinstance(df, dd.DataFrame):
        df = df.compute()
    if isinstance(df, pd.DataFrame):
        return df.to_html(classes="table")
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas().to_html(classes="table")
    raise TypeError("Unsupported data type for HTML conversion")
