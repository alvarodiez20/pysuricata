"""Generate an HTML EDA report from a DataFrame-like object."""

import pandas as pd
from typing import Union, Optional, List
from .analysis import summary_statistics, missing_values, correlation_matrix
from .utils import to_dataframe, df_to_html


def generate_report(
    data: Union[pd.DataFrame, "np.ndarray", "dd.DataFrame", "pl.DataFrame"],
    output_file: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> str:
    """Generate an HTML report containing summary statistics, missing values, and a correlation matrix.

    The report is styled with inline CSS.

    Args:
        data: Input data (Pandas, Dask, Polars, or a 2D NumPy array).
        output_file: Optional file path to save the HTML report.
        columns: Optional column names (for 2D NumPy arrays).

    Returns:
        A string containing the HTML report.
    """
    df = to_dataframe(data, columns=columns)
    stats = summary_statistics(df)
    miss = missing_values(df)
    corr = correlation_matrix(df)

    html = f"""
    <html>
      <head>
        <title>EDA Report</title>
      </head>
      <body>
        <h1>Summary Statistics</h1>
        {df_to_html(stats)}
        <h1>Missing Values</h1>
        {df_to_html(miss)}
        <h1>Correlation Matrix</h1>
        {df_to_html(corr)}
      </body>
    </html>
    """
    if output_file:
        with open(output_file, "w") as f:
            f.write(html)
    return html
