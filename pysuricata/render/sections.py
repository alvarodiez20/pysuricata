from __future__ import annotations

"""Sample-section renderers for pandas and polars.

This module contains small, testable helpers to build the sample content
for the report for both pandas and polars backends. It avoids heavyweight 
dependencies where possible and provides a pandas-free HTML path for 
polars datasets.
"""

from collections.abc import Iterable, Sequence
from typing import Any, List, Tuple

import numpy as np


def _build_sample_table_html(pdf: pd.DataFrame) -> str:  # type: ignore[name-defined]
    """Build an HTML table for a pandas sample frame.

    The function right-aligns numeric columns by wrapping values in
    ``<span class="num">``. It expects the first column to contain
    positional row numbers, which are kept as-is.

    Args:
        pdf: Pandas DataFrame already containing a positional first column.

    Returns:
        str: An HTML table string with inline spans for numeric values.
    """
    try:
        import pandas as pd  # type: ignore

        # Right-align numeric columns via a span
        try:
            num_cols = pdf.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                pdf[c] = pdf[c].map(
                    lambda v: f'<span class="num">{v}</span>' if pd.notna(v) else ""
                )
        except Exception:
            pass
        return pdf.to_html(classes="sample-table", index=False, escape=False)
    except Exception:
        return "<em>Unable to render sample preview.</em>"


def _build_simple_table_html(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    numeric_idx: Sequence[int],
) -> str:
    """Build a plain HTML table without requiring pandas.

    Args:
        columns: Column headers in display order.
        rows: Iterable of row tuples/lists matching ``columns`` order.
        numeric_idx: Zero-based indices of columns to right-align using the
            ``<span class="num">`` wrapper.

    Returns:
        str: HTML for a table suitable for embedding in the sample section.
    """
    try:
        num_set = set(int(i) for i in numeric_idx)
    except Exception:
        num_set = set()
    # Header
    thead = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead>"
    # Body
    body_cells: List[str] = []
    for r in rows:
        try:
            cells = []
            for j, v in enumerate(r):
                if j in num_set and v is not None and v != "":
                    cells.append(f'<td><span class="num">{v}</span></td>')
                else:
                    cells.append(f"<td>{'' if v is None else v}</td>")
            body_cells.append("<tr>" + "".join(cells) + "</tr>")
        except Exception:
            continue
    tbody = "<tbody>" + "".join(body_cells) + "</tbody>"
    return f'<table class="sample-table">{thead}{tbody}</table>'


def _sample_pandas(df: pd.DataFrame, sample_rows: int) -> Tuple[pd.DataFrame, int]:  # type: ignore[name-defined]
    """Sample rows from a pandas DataFrame and add a positional column.

    Args:
        df: Pandas DataFrame to sample from (typically the first chunk).
        sample_rows: Maximum number of rows to sample (capped by DataFrame length).

    Returns:
        Tuple[pd.DataFrame, int]: The sampled DataFrame with a first positional
        column, and the number of sampled rows ``n``.
    """
    import pandas as pd  # type: ignore

    n = max(0, min(int(sample_rows), len(df.index)))
    sample_df = df.sample(n=n) if n > 0 else df.head(0)
    # Derive original positional row numbers within this chunk
    row_pos = pd.Index(df.index).get_indexer(sample_df.index)
    sample_df = sample_df.copy()
    sample_df.insert(0, "", row_pos)
    return sample_df, n


def _sample_polars(
    df: pl.DataFrame, sample_rows: int
) -> Tuple[Tuple[List[str], List[Sequence[Any]], List[int]], int]:  # type: ignore[name-defined]
    """Sample rows from a polars DataFrame and build HTML-friendly payload.

    This path never converts to pandas. It adds a positional column via
    ``with_row_count``, samples rows (without replacement), and returns the
    sequences required by :func:`_build_simple_table_html`.

    Args:
        df: Polars DataFrame to sample from (typically the first chunk).
        sample_rows: Maximum number of rows to sample (capped by height).

    Returns:
        Tuple[((columns, rows, numeric_idx), n_rows)]:
        - ``columns``: Display column names including the positional column.
        - ``rows``: Sampled rows as sequences.
        - ``numeric_idx``: Indices (including the positional column 0) that
          should be right-aligned.
        - ``n_rows``: Number of sampled rows actually returned.
    """
    try:
        pass  # type: ignore
    except Exception:
        # No polars, return empty
        return ([], [], []), 0

    n = max(0, min(int(sample_rows), int(df.height)))
    if n <= 0:
        cols = [""] + list(df.columns)
        return (cols, [], []), 0
    try:
        with_idx = df.with_row_index(name="")
        sampled = with_idx.sample(n=n, with_replacement=False, shuffle=True)
    except Exception:
        # If sample not available (older polars), fall back to head
        sampled = df.with_row_index(name="").head(n)
    # Build simple table without pandas
    cols = [""] + list(df.columns)
    try:
        rows = sampled.rows()
    except Exception:
        rows = []
    # numeric columns: detect using polars dtypes
    try:
        from polars import selectors as cs  # type: ignore

        # Use selectors to detect numeric columns
        numeric_cols = set(df.select(cs.numeric()).columns)
        numeric_idx = [0] + [
            i + 1 for i, c in enumerate(df.columns) if c in numeric_cols
        ]
    except Exception:
        numeric_idx = [0]
    return (cols, rows, numeric_idx), n


def render_sample_section_pandas(df: pd.DataFrame, sample_rows: int = 10) -> str:  # type: ignore[name-defined]
    """Render the sample content for a pandas chunk.

    Args:
        df: Pandas DataFrame (first chunk).
        sample_rows: Desired number of rows in the sample table.

    Returns:
        str: HTML string for the sample table with metadata.
    """
    try:
        pdf, n = _sample_pandas(df, sample_rows)
        # Build simple HTML to ensure stable structure across pandas versions
        columns = list(pdf.columns)
        rows = pdf.to_numpy().tolist()
        # Numeric alignment indices (include positional column 0)
        import pandas as pd  # type: ignore

        num_idx = [0] + [
            i
            for i, c in enumerate(columns[1:], start=1)
            if pd.api.types.is_numeric_dtype(pdf[c])
        ]
        sample_html_table = _build_simple_table_html(columns, rows, num_idx)
    except Exception:
        sample_html_table, n = "<em>Unable to render sample preview.</em>", 0
    return _wrap_sample_content(sample_html_table, n)


def render_sample_section_polars(df: pl.DataFrame, sample_rows: int = 10) -> str:  # type: ignore[name-defined]
    """Render the sample content for a polars chunk.

    This function relies solely on polars to compute the sample and build the
    HTML table; it does not require pandas.

    Args:
        df: Polars DataFrame (first chunk).
        sample_rows: Desired number of rows in the sample table.

    Returns:
        str: HTML string for the sample table with metadata.
    """
    try:
        (cols, rows, numeric_idx), n = _sample_polars(df, sample_rows)
        sample_html_table = _build_simple_table_html(cols, rows, numeric_idx)
    except Exception:
        sample_html_table, n = "<em>Unable to render sample preview.</em>", 0
    return _wrap_sample_content(sample_html_table, n)


def render_sample_section(df_like: Any, sample_rows: int = 10) -> str:
    """Render sample content for pandas or polars chunks.

    Dispatches based on runtime type and gracefully degrades if optional
    dependencies are missing.

    Args:
        df_like: Pandas or polars DataFrame to sample from.
        sample_rows: Desired number of rows in the sample table.

    Returns:
        str: HTML string for the sample table with metadata.
    """
    try:
        import pandas as pd  # type: ignore

        if isinstance(df_like, pd.DataFrame):
            return render_sample_section_pandas(df_like, sample_rows)
    except Exception:
        pass
    try:
        import polars as pl  # type: ignore

        if isinstance(df_like, pl.DataFrame):
            return render_sample_section_polars(df_like, sample_rows)
    except Exception:
        pass
    # Fallback
    return _wrap_sample_content("<em>Unable to render sample preview.</em>", 0)


def _wrap_sample_content(sample_html_table: str, n_rows: int) -> str:
    """Wrap a table HTML string for embedding in the new sample section.

    Args:
        sample_html_table: HTML string for the sample table body.
        n_rows: Number of rows included in the sample (for the caption).

    Returns:
        str: A ready-to-embed content containing the sample table with scrolling.
    """
    return f"""
    <div class="sample-scroll">{sample_html_table}</div>
    <p class="muted small">Showing {n_rows} randomly sampled rows from the first chunk.</p>
    """
