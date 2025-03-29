import os
import base64
import pandas as pd
import numpy as np
from typing import Union, Optional, List
from .analysis import summary_statistics, missing_values, correlation_matrix
from .utils import to_dataframe, df_to_html

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
    columns: Optional[List[str]] = None,
) -> str:
    """
    Generate an HTML report containing summary statistics, missing values, and a correlation matrix.
    
    The report embeds CSS and a PNG logo from a static folder into the HTML template so that
    all styling and resources remain intact even if the report is moved.

    Args:
        data: Input data (Pandas, Dask, Polars, or a 2D NumPy array).
        output_file: Optional file path to save the HTML report.
        columns: Optional column names (for 2D NumPy arrays).

    Returns:
        A string containing the complete HTML report.
    """
    # Convert input data to a DataFrame-like object.
    df = to_dataframe(data, columns=columns)
    stats = summary_statistics(df)
    miss = missing_values(df)
    corr = correlation_matrix(df)

    # Determine the paths to the static and template folders.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(module_dir, "static")
    template_dir = os.path.join(module_dir, "templates")
    
    # Load the HTML template.
    template_path = os.path.join(template_dir, "report_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Load CSS and embed it inline.
    css_path = os.path.join(static_dir, "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        css_tag = f"<style>{css_content}</style>"
    else:
        css_tag = ""

    # Load the PNG logo from static/images/logo.png and encode it as Base64.
    logo_path = os.path.join(static_dir, "images", "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            encoded_logo = base64.b64encode(img_file.read()).decode("utf-8")
        logo_html = f'<img id="logo" src="data:image/png;base64,{encoded_logo}" alt="Logo">'
    else:
        logo_html = ""
    
    # Load the favicon from static/images/favicon.ico and encode it as Base64.
    favicon_path = os.path.join(static_dir, "images", "favicon.png")
    if os.path.exists(favicon_path):
        with open(favicon_path, "rb") as icon_file:
            encoded_favicon = base64.b64encode(icon_file.read()).decode("utf-8")
        # Since it's an ICO file, we use the appropriate MIME type.
        favicon_tag = f'<link rel="icon" href="data:image/x-icon;base64,{encoded_favicon}" type="image/x-icon">'
    else:
        favicon_tag = ""

    # Replace the placeholders in the template with the actual content.
    html = template.format(
        favicon=favicon_tag,
        css=css_tag,
        logo=logo_html,
        stats_table=df_to_html(stats),
        missing_table=df_to_html(miss),
        corr_table=df_to_html(corr)
    )

    # Optionally write the report to an output file.
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    return html
