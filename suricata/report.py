import pandas as pd
from .analysis import summary_statistics, missing_values, correlation_matrix


def generate_report(df: pd.DataFrame, output_file: str = None) -> str:
    """Generate an HTML report of the DataFrame analysis.

    The report includes summary statistics, missing values, and the correlation matrix,
    formatted with inline CSS.

    Args:
        df (pd.DataFrame): Input DataFrame.
        output_file (str, optional): File path to save the HTML report. Defaults to None.

    Returns:
        str: HTML content of the generated report.
    """
    stats = summary_statistics(df)
    missing = missing_values(df)
    corr = correlation_matrix(df)

    html = """
    <html>
      <head>
        <title>EDA Report</title>
      </head>
      <body>
        <h1>Summary Statistics</h1>
        {stats_table}
        <h1>Missing Values</h1>
        {missing_table}
        <h1>Correlation Matrix</h1>
        {corr_table}
      </body>
    </html>
    """.format(
        stats_table=stats.to_html(classes="table"),
        missing_table=missing.to_html(classes="table"),
        corr_table=corr.to_html(classes="table"),
    )

    if output_file:
        with open(output_file, "w") as f:
            f.write(html)

    return html
