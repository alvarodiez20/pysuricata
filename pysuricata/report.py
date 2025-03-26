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
    """Generate an HTML report containing summary statistics, missing values, and a correlation matrix.

    The report automatically embeds CSS and a PNG logo from a static folder so that the
    styling remains intact even if the report is moved.

    Args:
        data: Input data (Pandas, Dask, Polars, or a 2D NumPy array).
        output_file: Optional file path to save the HTML report.
        columns: Optional column names (for 2D NumPy arrays).

    Returns:
        A string containing the complete HTML report.
    """
    # Convert the input data to a DataFrame-like object.
    df = to_dataframe(data, columns=columns)
    stats = summary_statistics(df)
    miss = missing_values(df)
    corr = correlation_matrix(df)

    # Determine the path to the static folder (relative to this module).
    module_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(module_dir, "static")

    # Load CSS from static/css/style.css.
    css_path = os.path.join(static_dir, "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        css_tag = f"<style>{css_content}</style>"
    else:
        # Fallback default CSS.
        css_tag = """
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; transition: background-color 0.3s, color 0.3s; }
          body.dark { background-color: #333; color: #f9f9f9; }
          h1 { color: inherit; }
          table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          body.dark th { background-color: #555; }
          .toggle-button { position: fixed; top: 20px; right: 20px; padding: 8px 12px; background-color: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; }
          #logo { max-width: 150px; margin-bottom: 20px; }
        </style>
        """

    # Load the PNG logo from static/images/logo.png and encode it as Base64.
    logo_path = os.path.join(static_dir, "images", "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            encoded_logo = base64.b64encode(img_file.read()).decode("utf-8")
        # Since the logo is a PNG image, use the appropriate MIME type.
        logo_html = (
            f'<img id="logo" src="data:image/png;base64,{encoded_logo}" alt="Logo">'
        )
    else:
        logo_html = ""

    html = f"""
    <html>
      <head>
        <title>EDA Report</title>
        {css_tag}
        <script>
          function toggleDarkMode() {{
            document.body.classList.toggle('dark');
          }}
        </script>
      </head>
      <body>
        <button class="toggle-button" onclick="toggleDarkMode()">Toggle Dark Mode</button>
        {logo_html}
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
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    return html
