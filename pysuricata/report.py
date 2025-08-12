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

    # Build a concise, modern summary section (self-contained styles)
    def _human_bytes(n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(n)
        for u in units:
            if size < 1024.0:
                return f"{size:,.1f} {u}"
            size /= 1024.0
        return f"{size:,.1f} PB"

    n_rows, n_cols = df.shape
    mem_bytes = int(df.memory_usage(deep=True).sum())

    # Type counts
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    categorical_cols = df.select_dtypes(include=["object", "category"]).shape[1]
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).shape[1]
    bool_cols = df.select_dtypes(include=["bool"]).shape[1]

    # Extra metrics for Quick Insights
    n_unique_cols = df.nunique(dropna=False).count()
    constant_cols = int((df.nunique(dropna=False) <= 1).sum())
    # High-cardinality categoricals: unique ratio > 0.5
    high_card_cols = int(((df.select_dtypes(include=["object", "category"])\
                          .nunique(dropna=False) / n_rows) > 0.5).sum())
    # Date range (min -> max) for any datetime col
    if datetime_cols > 0:
        date_min = str(df.select_dtypes(include=["datetime", "datetime64[ns]"]).min().min())
        date_max = str(df.select_dtypes(include=["datetime", "datetime64[ns]"]).max().max())
    else:
        date_min, date_max = "—", "—"
    # Likely ID cols: all unique & non-null
    likely_id_cols = df.columns[(df.nunique(dropna=False) == n_rows) & (~df.isna().any())].tolist()
    if len(likely_id_cols) > 3:
        likely_id_cols = likely_id_cols[:3] + ["..."]
    likely_id_cols_str = ", ".join(likely_id_cols) if likely_id_cols else "—"
    # Text columns & average length
    text_cols = df.select_dtypes(include=["object"]).shape[1]
    if text_cols > 0:
        avg_text_len = df.select_dtypes(include=["object"]).apply(lambda s: s.dropna().astype(str).str.len().mean()).mean()
        avg_text_len = f"{avg_text_len:.1f}"
    else:
        avg_text_len = "—"

    # Missing & duplicates
    total_cells = int(df.size) if df.size else 0
    total_missing = int(df.isna().sum().sum()) if total_cells else 0
    missing_pct = (total_missing / total_cells * 100.0) if total_cells else 0.0
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / n_rows * 100.0) if n_rows else 0.0

    # Format for display
    missing_overall = f"{total_missing:,} ({missing_pct:.1f}%)"
    duplicates_overall = f"{dup_rows:,} ({dup_pct:.1f}%)"

    # Top columns by missing percentage (up to 5)
    missing_by_col = df.isna().mean().sort_values(ascending=False)
    top_missing_list = ""
    for col, frac in missing_by_col.head(5).items():
        count = int(df[col].isna().sum())
        pct = frac * 100
        # Decide severity for bar color
        if pct <= 5:
            severity_class = "low"
        elif pct <= 20:
            severity_class = "medium"
        else:
            severity_class = "high"

        top_missing_list += f"""
        <li class="missing-item">
          <div class="missing-info">
            <code class="missing-col" title="{col}">{col}</code>
            <span class="missing-stats">{count:,} ({pct:.1f}%)</span>
          </div>
          <div class="missing-bar">
            <div class="missing-fill {severity_class}" style="width: {pct:.1f}%;"></div>
          </div>
        </li>
        """

    if not top_missing_list:
        top_missing_list = """
        <li class="missing-item">
          <div class="missing-info">
            <code class="missing-col">None</code>
            <span class="missing-stats">0 (0.0%)</span>
          </div>
          <div class="missing-bar"><div class="missing-fill low" style="width: 0%;"></div></div>
        </li>
        """


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
    logo_path = os.path.join(static_dir, "images", "logo_suricata_transparent.png")
    logo_html = embed_image(
        logo_path, element_id="logo", alt_text="Logo", mime_type="image/png"
    )

    # Load and embed the favicon.
    favicon_path = os.path.join(static_dir, "images", "favicon.ico")
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
      n_rows=f"{n_rows:,}",
      n_cols=f"{n_cols:,}",
      memory_usage=_human_bytes(mem_bytes),
      missing_overall=missing_overall,
      duplicates_overall=duplicates_overall,
      numeric_cols=numeric_cols,
      categorical_cols=categorical_cols,
      datetime_cols=datetime_cols,
      bool_cols=bool_cols,
      top_missing_list=top_missing_list,
      n_unique_cols=f"{n_unique_cols:,}",
      constant_cols=f"{constant_cols:,}",
      high_card_cols=f"{high_card_cols:,}",
      date_min=date_min,
      date_max=date_max,
      likely_id_cols=likely_id_cols_str,
      text_cols=f"{text_cols:,}",
      avg_text_len=avg_text_len,
  )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    return html
