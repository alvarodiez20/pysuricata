# eda_tool/eda_tool/analysis.py
import pandas as pd
from .logger import timeit


@timeit
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Summary statistics.
    """
    return df.describe()


@timeit
def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing values per column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the count and percentage of missing values.
    """
    missing_count = df.isnull().sum()
    missing_percent = 100 * missing_count / len(df)
    return pd.DataFrame(
        {"missing_count": missing_count, "missing_percent": missing_percent}
    )


@timeit
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix of numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Correlation matrix computed on numeric columns.
    """
    numeric_df = df.select_dtypes(include=["number"])
    return numeric_df.corr()
