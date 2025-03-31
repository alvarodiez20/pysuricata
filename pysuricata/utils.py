import os
import base64
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


def load_template(template_path: str) -> str:
    """
    Load an HTML template from a file.

    Args:
        template_path (str): The file path to the HTML template.

    Returns:
        str: The content of the HTML template.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def load_css(css_path: str) -> str:
    """
    Load a CSS file and return its content wrapped in a <style> tag.

    Args:
        css_path (str): The file path to the CSS file.

    Returns:
        str: A string with the CSS content wrapped in a <style> tag, or an empty string if the file is not found.
    """
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        return f"<style>{css_content}</style>"
    return ""


def embed_image(
    image_path: str, element_id: str, alt_text: str = "", mime_type: str = "image/png"
) -> str:
    """
    Embed an image into an HTML <img> tag using Base64 encoding.

    Args:
        image_path (str): The file path to the image.
        element_id (str): The HTML id attribute for the image.
        alt_text (str): Alternate text for the image.
        mime_type (str): MIME type of the image (default "image/png").

    Returns:
        str: An HTML <img> tag containing the embedded Base64 image.
             Returns an empty string if the image file does not exist.
    """
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f'<img id="{element_id}" src="data:{mime_type};base64,{encoded}" alt="{alt_text}">'
    return ""


def embed_favicon(favicon_path: str) -> str:
    """
    Embed a favicon into an HTML <link> tag using Base64 encoding.

    Args:
        favicon_path (str): The file path to the favicon image.

    Returns:
        str: An HTML <link> tag containing the embedded favicon.
             Returns an empty string if the favicon file does not exist.
    """
    if os.path.exists(favicon_path):
        with open(favicon_path, "rb") as icon_file:
            encoded = base64.b64encode(icon_file.read()).decode("utf-8")
        return f'<link rel="icon" href="data:image/x-icon;base64,{encoded}" type="image/x-icon">'
    return ""


def load_script(script_path: str) -> str:
    """
    Load a JavaScript file and return its content.

    Args:
        script_path (str): The file path to the JavaScript file.

    Returns:
        str: The JavaScript content as a string, or an empty string if the file is not found.
    """
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""
