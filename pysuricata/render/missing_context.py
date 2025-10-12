"""
Enhanced per-column missing values context and navigation.

This module provides navigation and context for individual column
missing values views (Option B).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def build_missing_values_context(
    current_column: str,
    per_column_chunk_metadata: Optional[Dict[str, List[Tuple[int, int, int]]]],
    all_column_names: List[str],
) -> str:
    """Build context and navigation for a column's missing values section.

    Args:
        current_column: Name of the current column
        per_column_chunk_metadata: Dict mapping column names to chunk metadata
        all_column_names: List of all columns in order

    Returns:
        HTML string with context and navigation
    """
    if not per_column_chunk_metadata:
        return ""

    # Find all columns with missing values
    columns_with_missing = []
    for col_name in all_column_names:
        if col_name in per_column_chunk_metadata:
            metadata = per_column_chunk_metadata[col_name]
            total_missing = sum(missing for _, _, missing in metadata)
            if total_missing > 0:
                total_vals = sum(end - start + 1 for start, end, _ in metadata)
                missing_pct = (total_missing / max(1, total_vals)) * 100.0
                columns_with_missing.append((col_name, total_missing, missing_pct))

    if not columns_with_missing:
        return ""

    # Sort by missing percentage (descending)
    columns_with_missing.sort(key=lambda x: x[2], reverse=True)

    # Find current column's rank
    current_rank = None
    for i, (col_name, _, _) in enumerate(columns_with_missing, 1):
        if col_name == current_column:
            current_rank = i
            break

    if current_rank is None:
        return ""  # Current column has no missing values

    total_cols_with_missing = len(columns_with_missing)

    # Build navigation links
    prev_col = None
    next_col = None

    if current_rank > 1:
        prev_col = columns_with_missing[current_rank - 2][0]

    if current_rank < total_cols_with_missing:
        next_col = columns_with_missing[current_rank][0]

    # Build navigation buttons
    nav_html = _build_navigation_buttons(prev_col, next_col)

    # Build context info
    context_html = f"""
    <div class="missing-context">
        <div class="context-info">
            <span class="rank-badge">#{current_rank} of {total_cols_with_missing}</span>
            <span class="context-label">columns with missing values</span>
        </div>
        {nav_html}
    </div>
    """

    return context_html


def _build_navigation_buttons(prev_col: Optional[str], next_col: Optional[str]) -> str:
    """Build previous/next navigation buttons.

    Args:
        prev_col: Previous column name (or None)
        next_col: Next column name (or None)

    Returns:
        HTML string with navigation buttons
    """
    from .svg_utils import safe_col_id

    prev_button = ""
    if prev_col:
        prev_col_id = safe_col_id(prev_col)
        prev_display = prev_col if len(prev_col) <= 20 else prev_col[:17] + "..."
        prev_button = f"""
        <button class="nav-column-btn prev" 
                onclick="document.getElementById('{prev_col_id}').scrollIntoView({{behavior: 'smooth', block: 'center'}}); 
                         document.getElementById('{prev_col_id}').querySelector('.details-toggle').click();"
                title="Previous column with missing values: {prev_col}">
            ← <code>{prev_display}</code>
        </button>
        """
    else:
        prev_button = '<button class="nav-column-btn prev" disabled>← First</button>'

    next_button = ""
    if next_col:
        next_col_id = safe_col_id(next_col)
        next_display = next_col if len(next_col) <= 20 else next_col[:17] + "..."
        next_button = f"""
        <button class="nav-column-btn next"
                onclick="document.getElementById('{next_col_id}').scrollIntoView({{behavior: 'smooth', block: 'center'}}); 
                         document.getElementById('{next_col_id}').querySelector('.details-toggle').click();"
                title="Next column with missing values: {next_col}">
            <code>{next_display}</code> →
        </button>
        """
    else:
        next_button = '<button class="nav-column-btn next" disabled>Last →</button>'

    return f"""
    <div class="column-navigation">
        {prev_button}
        {next_button}
    </div>
    """
