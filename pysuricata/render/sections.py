from __future__ import annotations

"""Sample-section renderers for pandas and polars.

This module contains small, testable helpers to build the sample content
for the report for both pandas and polars backends. It avoids heavyweight
dependencies where possible and provides a pandas-free HTML path for
polars datasets.
"""

import html as _html
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


# A real null renders as an em dash, never as the string "nan". The literal is
# what pandas prints and it reads as a value -- a column of "nan" looks like
# text data rather than absence. The true value goes in a title so nothing is
# lost, and the glyph carries a class so it can meet the 4.5:1 text minimum:
# it is data, not decoration.
_NULL_GLYPH = '<span class="nil" title="{title}">—</span>'

# Long values clamp with an ellipsis and keep the whole string in a title. A
# 500-character cell otherwise stretches the pane until nothing else fits.
_CELL_CLAMP = 260


def _is_null(value: Any) -> bool:
    """Whether a cell holds an actual null.

    Deliberately not a string test. A column named ``nan``, and a string whose
    characters happen to be ``n``, ``a``, ``n``, are both real data and must
    render as themselves -- which is the difference between reporting absence
    and corrupting a value.
    """
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN is the only value unequal to itself
    except Exception:
        return False


def _cell(value: Any) -> tuple[str, str]:
    """Return the cell's HTML body and its title attribute."""
    if _is_null(value):
        return _NULL_GLYPH.format(title=_html.escape(str(value))), ""
    text = str(value)
    escaped = _html.escape(text)
    title = f' title="{escaped}"' if len(text) > 24 else ""
    return escaped, title


def _build_simple_table_html(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    numeric_idx: Sequence[int],
) -> str:
    """Build the sample table.

    One builder for both backends -- pandas hands over rows and polars hands
    over rows, so there is no reason for the two to produce different markup,
    and they used to.

    The row index is frozen with ``position: sticky`` rather than split into a
    second table beside the scroll pane. Two tables have to be kept in vertical
    step by hand, and any cell that wraps in one of them silently desynchronises
    the pair -- the failure the design package warns about. A sticky column is
    the same element as the row it belongs to, so it cannot drift from it.

    Args:
        columns: Column headers in display order, index column first.
        rows: Row values matching ``columns``.
        numeric_idx: Indices to right-align.

    Returns:
        str: The table markup.
    """
    try:
        num_set = {int(i) for i in numeric_idx}
    except Exception:
        num_set = set()

    head_cells = []
    for index, name in enumerate(columns):
        classes = ["idx"] if index == 0 else []
        if index in num_set:
            classes.append("num")
        attr = f' class="{" ".join(classes)}"' if classes else ""
        head_cells.append(f"<th{attr}>{_html.escape(str(name))}</th>")
    thead = f"<thead><tr>{''.join(head_cells)}</tr></thead>"

    body_rows: list[str] = []
    for row in rows:
        try:
            cells = []
            for index, value in enumerate(row):
                classes = ["idx"] if index == 0 else []
                if index in num_set:
                    classes.append("num")
                body, title = _cell(value)
                attr = f' class="{" ".join(classes)}"' if classes else ""
                cells.append(f"<td{attr}{title}>{body}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        except Exception:
            continue
    tbody = f"<tbody>{''.join(body_rows)}</tbody>"
    return f'<table class="sample-table">{thead}{tbody}</table>'


def _sample_pandas(
    df: pd.DataFrame, sample_rows: int, *, seed: int | None = None
) -> tuple[pd.DataFrame, int]:
    """Sample rows from a pandas DataFrame and add a positional column.

    Args:
        df: Pandas DataFrame to sample from (typically the first chunk).
        sample_rows: Maximum number of rows to sample (capped by DataFrame length).
        seed: Seed for the row draw. None gives an independent generator; either
            way the process-global RNG is left alone.

    Returns:
        Tuple[pd.DataFrame, int]: The sampled DataFrame with a first positional
        column, and the number of sampled rows ``n``.
    """
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    n = max(0, min(int(sample_rows), len(df.index)))
    # Explicit random_state: bare df.sample() draws from the global numpy RNG,
    # which would make the preview both irreproducible and visible to callers.
    sample_df = (
        df.sample(n=n, random_state=np.random.default_rng(seed))
        if n > 0
        else df.head(0)
    )
    # Derive original positional row numbers within this chunk
    row_pos = pd.Index(df.index).get_indexer(sample_df.index)
    sample_df = sample_df.copy()
    sample_df.insert(0, "", row_pos)
    return sample_df, n


def _sample_polars(
    df: pl.DataFrame, sample_rows: int, *, seed: int | None = None
) -> tuple[tuple[list[str], list[Sequence[Any]], list[int]], int]:
    """Sample rows from a polars DataFrame and build HTML-friendly payload.

    This path never converts to pandas. It adds a positional column via
    ``with_row_index``, samples rows (without replacement), and returns the
    sequences required by :func:`_build_simple_table_html`.

    Args:
        df: Polars DataFrame to sample from (typically the first chunk).
        sample_rows: Maximum number of rows to sample (capped by height).
        seed: Seed for the row draw. None lets polars pick one.

    Returns:
        tuple[((columns, rows, numeric_idx), n_rows)]:
        - ``columns``: Display column names including the positional column.
        - ``rows``: Sampled rows as sequences.
        - ``numeric_idx``: Indices (including the positional column 0) that
          should be right-aligned.
        - ``n_rows``: Number of sampled rows actually returned.
    """

    n = max(0, min(int(sample_rows), int(df.height)))
    if n <= 0:
        cols = [""] + list(df.columns)
        return (cols, [], []), 0
    try:
        with_idx = df.with_row_index(name="")
        sampled = with_idx.sample(n=n, with_replacement=False, shuffle=True, seed=seed)
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


def render_sample_section_pandas(
    df: pd.DataFrame, sample_rows: int = 10, *, seed: int | None = None
) -> str:
    """Render the sample content for a pandas chunk.

    Args:
        df: Pandas DataFrame (first chunk).
        sample_rows: Desired number of rows in the sample table.
        seed: Seed for the row draw, so the preview is reproducible.

    Returns:
        str: HTML string for the sample table with metadata.
    """
    try:
        pdf, n = _sample_pandas(df, sample_rows, seed=seed)
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
        columns = []
    return _wrap_sample_content(sample_html_table, n, max(0, len(columns) - 1))


def render_sample_section_polars(
    df: pl.DataFrame, sample_rows: int = 10, *, seed: int | None = None
) -> str:
    """Render the sample content for a polars chunk.

    This function relies solely on polars to compute the sample and build the
    HTML table; it does not require pandas.

    Args:
        df: Polars DataFrame (first chunk).
        sample_rows: Desired number of rows in the sample table.
        seed: Seed for the row draw, so the preview is reproducible.

    Returns:
        str: HTML string for the sample table with metadata.
    """
    try:
        (cols, rows, numeric_idx), n = _sample_polars(df, sample_rows, seed=seed)
        sample_html_table = _build_simple_table_html(cols, rows, numeric_idx)
    except Exception:
        sample_html_table, n, cols = "<em>Unable to render sample preview.</em>", 0, []
    return _wrap_sample_content(sample_html_table, n, max(0, len(cols) - 1))


def render_sample_section(
    df_like: Any, sample_rows: int = 10, *, seed: int | None = None
) -> str:
    """Render sample content for pandas or polars chunks.

    Dispatches based on runtime type and gracefully degrades if optional
    dependencies are missing.

    Args:
        df_like: Pandas or polars DataFrame to sample from.
        sample_rows: Desired number of rows in the sample table.
        seed: Seed for the row draw, so the preview is reproducible.

    Returns:
        str: HTML string for the sample table with metadata.
    """
    try:
        import pandas as pd  # type: ignore

        if isinstance(df_like, pd.DataFrame):
            return render_sample_section_pandas(df_like, sample_rows, seed=seed)
    except Exception:
        pass
    try:
        import polars as pl  # type: ignore

        if isinstance(df_like, pl.DataFrame):
            return render_sample_section_polars(df_like, sample_rows, seed=seed)
    except Exception:
        pass
    # Fallback
    return _wrap_sample_content("<em>Unable to render sample preview.</em>", 0)


def _wrap_sample_content(sample_html_table: str, n_rows: int, n_cols: int = 0) -> str:
    """Wrap the table with the two facts a reader needs to trust it.

    Both were previously missing or buried: that the rows are a random draw
    from the first chunk rather than the head of the file, and that there is
    more to the right than fits. Stating the overflow is the difference between
    a table that looks complete and one that says it is not.

    Args:
        sample_html_table: The table markup.
        n_rows: Rows drawn.
        n_cols: Columns in the frame, excluding the row index. Zero suppresses
            the column notice.

    Returns:
        str: The sample content.
    """
    # Only when there is something to scroll. A one-column frame has nothing
    # off-screen, and the arrow would point at nothing.
    scroll_note = (
        f'<span class="sample-note__cols">{n_cols:,} cols · scroll →</span>'
        if n_cols > 1
        else ""
    )
    return f"""
    <p class="sample-note">{scroll_note}<span class="sample-note__rows">{n_rows:,} rows drawn at random from the first chunk</span></p>
    <div class="sample-scroll">{sample_html_table}</div>
    """
