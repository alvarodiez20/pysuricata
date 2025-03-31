import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Union, Optional, List
from .analysis import summary_statistics, missing_values, correlation_matrix
from .utils import (
    to_dataframe,
    df_to_html,
    load_css,
    load_template,
    embed_favicon,
    embed_image,
    load_script,
)

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

try:
    import polars as pl
except ImportError:
    pl = None


def generate_report(
    data: Union[pd.DataFrame, np.ndarray, "dd.DataFrame", "pl.DataFrame"],
    output_file: Optional[str] = None,
    report_title: Optional[str] = "PySuricata EDA Report",
    columns: Optional[List[str]] = None,
) -> str:
    """
    Generate an HTML report containing summary statistics, missing values, and a correlation matrix.

    This function converts the input data into a DataFrame, computes summary statistics,
    missing values, and the correlation matrix. It then loads an HTML template and embeds
    CSS and images (logo and favicon) using Base64 encoding so that the report is self-contained.
    Optionally, the report can be written to an output file.

    Args:
        data (Union[pd.DataFrame, np.ndarray, dd.DataFrame, pl.DataFrame]):
            The input data.
        output_file (Optional[str]):
            File path to save the HTML report. If None, the report is not written to disk.
        report_title (Optional[str]):
            Title of the report. Defaults to "PySuricata EDA Report" if not provided.
        columns (Optional[List[str]]):
            Column names (used when the data is a 2D NumPy array).

    Returns:
        str: A string containing the complete HTML report.
    """

    start_time = time.time()

    # Convert input data to a DataFrame.
    df = to_dataframe(data, columns=columns)

    # Perform analyses.
    stats = summary_statistics(df)
    miss = missing_values(df)
    corr = correlation_matrix(df)

    # Determine directory paths.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(module_dir, "static")
    template_dir = os.path.join(module_dir, "templates")

    # Load the HTML template and resource files.
    template_path = os.path.join(template_dir, "report_template.html")
    template = load_template(template_path)

    # Load CSS and embed it inline.
    css_path = os.path.join(static_dir, "css", "style.css")
    css_tag = load_css(css_path)

    # Load the JavaScript for dark mode toggle.
    script_path = os.path.join(static_dir, "js", "darkModeToggle.js")
    script_content = load_script(script_path)

    # Load and embed the PNG logo.
    logo_path = os.path.join(static_dir, "images", "logo.png")
    logo_html = embed_image(
        logo_path, element_id="logo", alt_text="Logo", mime_type="image/png"
    )

    # Load and embed the favicon.
    favicon_path = os.path.join(static_dir, "images", "favicon.png")
    favicon_tag = embed_favicon(favicon_path)

    # Compute how long it took to generate the report.
    end_time = time.time()
    duration_seconds = end_time - start_time

    # Set defaults for new information.
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    repo_url = "https://github.com/alvarodiez20/pysuricata"

    # Replace placeholders in the template.
    html = template.format(
        favicon=favicon_tag,
        css=css_tag,
        script=script_content,
        logo=logo_html,
        report_title=report_title,
        report_date=report_date,
        report_duration=f"{duration_seconds:.2f}",
        repo_url=repo_url,
        stats_table=df_to_html(stats),
        missing_table=df_to_html(miss),
        corr_table=df_to_html(corr),
    )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    return html
