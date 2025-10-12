from __future__ import annotations

import html as _html
import os
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

from .._version import resolve_version as _resolve_pysuricata_version
from ..compute.core.types import ColumnKinds
from ..utils import embed_favicon, embed_image, load_css, load_script, load_template
from .cards import render_bool_card as _render_bool_card
from .cards import render_cat_card as _render_cat_card
from .cards import render_dt_card as _render_dt_card
from .cards import render_numeric_card as _render_numeric_card
from .format_utils import human_bytes as _human_bytes
from .format_utils import human_time as _human_time
from .markdown_utils import render_markdown_to_html
from .missing_columns import create_missing_columns_renderer
from .svg_utils import safe_col_id as _safe_col_id


def render_html_snapshot(
    *,
    kinds: ColumnKinds,
    accs: dict[str, Any],
    first_columns: list[str],
    row_kmv: Any,
    total_missing_cells: int,
    approx_mem_bytes: int,
    start_time: float,
    cfg: Any,
    report_title: Optional[str],
    sample_section_html: str,
    chunk_metadata: Optional[list[tuple[int, int, int]]] = None,
) -> str:
    kinds_map = {
        **{name: ("numeric", accs[name]) for name in kinds.numeric},
        **{name: ("categorical", accs[name]) for name in kinds.categorical},
        **{name: ("datetime", accs[name]) for name in kinds.datetime},
        **{name: ("boolean", accs[name]) for name in kinds.boolean},
    }

    # Build missing columns list using intelligent analysis
    miss_list: list[tuple[str, float, int]] = []
    for name, (kind, acc) in kinds_map.items():
        miss = getattr(acc, "missing", 0)
        cnt = getattr(acc, "count", 0) + miss
        pct = (miss / cnt * 100.0) if cnt else 0.0
        miss_list.append((name, pct, miss))
    miss_list.sort(key=lambda t: t[1], reverse=True)

    # Use intelligent missing columns renderer with configuration
    n_rows = int(getattr(row_kmv, "rows", 0))
    n_cols = len(kinds_map)
    missing_renderer = create_missing_columns_renderer(
        min_threshold_pct=getattr(cfg, "missing_columns_threshold_pct", 0.5)
    )
    top_missing_list = missing_renderer.render_missing_columns_html(
        miss_list, n_cols, n_rows
    )
    total_cells = n_rows * n_cols
    missing_overall = f"{total_missing_cells:,} ({(total_missing_cells / max(1, total_cells) * 100):.1f}%)"
    dup_rows, dup_pct = row_kmv.approx_duplicates()
    duplicates_overall = f"{dup_rows:,} ({dup_pct:.1f}%)"

    constant_cols = 0
    high_card_cols = 0
    for name, (kind, acc) in kinds_map.items():
        if kind in ("numeric", "categorical"):
            u = (
                acc._uniques.estimate()
                if hasattr(acc, "_uniques")
                else getattr(acc, "unique_est", 0)
            )
        elif kind == "datetime":
            u = acc.unique_est
        else:
            present = (acc.true_n > 0) + (acc.false_n > 0)
            u = int(present)
        _ = getattr(acc, "count", 0) + getattr(acc, "missing", 0)
        if u <= 1:
            constant_cols += 1
        if kind == "categorical" and n_rows:
            if (u / n_rows) > 0.5:
                high_card_cols += 1

    if kinds.datetime:
        mins, maxs = [], []
        for name in kinds.datetime:
            acc = accs[name]
            if acc._min_ts is not None:
                mins.append(acc._min_ts)
            if acc._max_ts is not None:
                maxs.append(acc._max_ts)
        if mins and maxs:
            dt_min = datetime.fromtimestamp(min(mins) / 1_000_000_000, tz=timezone.utc)
            dt_max = datetime.fromtimestamp(max(maxs) / 1_000_000_000, tz=timezone.utc)
            # Format: date on one line, time on next (inline with <br>)
            date_min = (
                f"{dt_min.strftime('%Y-%m-%d')}<br>{dt_min.strftime('%H:%M:%S UTC')}"
            )
            date_max = (
                f"{dt_max.strftime('%Y-%m-%d')}<br>{dt_max.strftime('%H:%M:%S UTC')}"
            )
        else:
            date_min = date_max = "—"
    else:
        date_min = date_max = "—"

    text_cols = len(kinds.categorical)
    avg_text_len_vals = [
        acc.avg_len
        for name, (k, acc) in kinds_map.items()
        if k == "categorical" and acc.avg_len is not None
    ]
    avg_text_len = (
        f"{(sum(avg_text_len_vals) / len(avg_text_len_vals)):.1f}"
        if avg_text_len_vals
        else "—"
    )

    col_order = [
        c
        for c in list(first_columns)
        if c in kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean
    ] or (kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean)
    all_cards_list: List[str] = []
    for name in col_order:
        acc = accs[name]
        card_html = ""
        data_type = ""

        if name in kinds.numeric:
            card_html = _render_numeric_card(acc.finalize(chunk_metadata))
            data_type = "numeric"
        elif name in kinds.categorical:
            card_html = _render_cat_card(acc.finalize())
            data_type = "categorical"
        elif name in kinds.datetime:
            card_html = _render_dt_card(acc.finalize(chunk_metadata))
            data_type = "datetime"
        elif name in kinds.boolean:
            card_html = _render_bool_card(acc.finalize())
            data_type = "boolean"

        # Add data attributes for filtering and search
        if card_html:
            # Insert data attributes into the var-card element
            card_html = card_html.replace(
                f'<article class="var-card" id="{_safe_col_id(name)}">',
                f'<article class="var-card" id="{_safe_col_id(name)}" data-type="{data_type}" data-name="{_html.escape(name)}">',
            )
            all_cards_list.append(card_html)
    # Build variables section with pagination and search
    total_variables = (
        len(kinds.numeric)
        + len(kinds.categorical)
        + len(kinds.datetime)
        + len(kinds.boolean)
    )
    variables_section_html = f"""
          <p class=\"muted small\">Analyzing {total_variables} variables ({len(kinds.numeric)} numeric, {len(kinds.categorical)} categorical, {len(kinds.datetime)} datetime, {len(kinds.boolean)} boolean).</p>

          <div class=\"vars-controls\">
            <div class=\"controls-row\">
              <input type=\"text\" placeholder=\"Search columns...\" id=\"search-input\">
              <div class=\"filter-buttons\">
                <button class=\"tab active\" data-filter=\"all\">All</button>
                <button class=\"tab\" data-filter=\"numeric\">Numeric</button>
                <button class=\"tab\" data-filter=\"categorical\">Categorical</button>
                <button class=\"tab\" data-filter=\"datetime\">Datetime</button>
                <button class=\"tab\" data-filter=\"boolean\">Boolean</button>
              </div>
            </div>
            <div class=\"info\" id=\"pagination-info\">Showing 1-{min(8, total_variables)} of {total_variables}</div>
          </div>

          <div class=\"cards-grid\" id=\"cards-grid\">
            {"".join(all_cards_list)}
          </div>

          <div class=\"pagination\" id=\"pagination\">
            <button id=\"prev-btn\" {"disabled" if total_variables <= 8 else ""}>←</button>
            <div class=\"pages\" id=\"page-numbers\"></div>
            <button id=\"next-btn\" {"disabled" if total_variables <= 8 else ""}>→</button>
          </div>
    """

    module_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(module_dir)
    static_dir = os.path.join(pkg_dir, "static")
    template_dir = os.path.join(pkg_dir, "templates")
    template_path = os.path.join(template_dir, "report_template.html")
    template = load_template(template_path)
    css_path = os.path.join(static_dir, "css", "style.css")
    css_tag = load_css(css_path)
    script_path = os.path.join(static_dir, "js", "functionality.js")
    script_content = load_script(script_path)

    # Add tooltips.js and pagination.js
    tooltips_script_path = os.path.join(static_dir, "js", "tooltips.js")
    tooltips_script_content = load_script(tooltips_script_path)

    pagination_script_path = os.path.join(static_dir, "js", "pagination.js")
    pagination_script_content = load_script(pagination_script_path)

    # Add description editor
    description_editor_path = os.path.join(static_dir, "js", "description-editor.js")
    description_editor_content = load_script(description_editor_path)

    # Combine all scripts
    combined_script_content = (
        script_content
        + "\n"
        + tooltips_script_content
        + "\n"
        + pagination_script_content
        + "\n"
        + description_editor_content
    )

    # Generate missing values section
    from .missing_section import MissingValuesSectionRenderer

    missing_section_renderer = MissingValuesSectionRenderer()
    missing_values_section_html = missing_section_renderer.render_section(
        kinds_map, accs, n_rows, n_cols, total_missing_cells
    )
    logo_light_path = os.path.join(
        static_dir, "images", "logo_suricata_transparent.png"
    )
    logo_dark_path = os.path.join(
        static_dir, "images", "logo_suricata_transparent_dark_mode.png"
    )
    logo_light_img = embed_image(
        logo_light_path, element_id="logo-light", alt_text="Logo", mime_type="image/png"
    )
    logo_dark_img = embed_image(
        logo_dark_path,
        element_id="logo-dark",
        alt_text="Logo (dark)",
        mime_type="image/png",
    )
    logo_html = f'<span id="logo">{logo_light_img}{logo_dark_img}</span>'
    favicon_path = os.path.join(static_dir, "images", "favicon.ico")
    favicon_tag = embed_favicon(favicon_path)

    end_time = time.time()
    duration_seconds = end_time - start_time
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pysuricata_version = _resolve_pysuricata_version()
    repo_url = "https://github.com/alvarodiez20/pysuricata"

    # Process description
    description_raw = getattr(cfg, "description", None) or ""
    # Treat whitespace-only descriptions as empty
    if description_raw and not description_raw.strip():
        description_raw = ""
    description_html = (
        render_markdown_to_html(description_raw) if description_raw else ""
    )
    # Escape the raw markdown for the data attribute
    description_attr = _html.escape(description_raw) if description_raw else ""

    html = template.format(
        favicon=favicon_tag,
        css=css_tag,
        script=combined_script_content,
        logo=logo_html,
        report_title=report_title or cfg.title,
        report_date=report_date,
        pysuricata_version=pysuricata_version,
        report_duration=_human_time(duration_seconds),
        repo_url=repo_url,
        n_rows=f"{n_rows:,}",
        n_cols=f"{n_cols:,}",
        memory_usage=_human_bytes(approx_mem_bytes) if approx_mem_bytes else "—",
        missing_overall=missing_overall,
        duplicates_overall=duplicates_overall,
        numeric_cols=len(kinds.numeric),
        categorical_cols=len(kinds.categorical),
        datetime_cols=len(kinds.datetime),
        bool_cols=len(kinds.boolean),
        top_missing_list=top_missing_list,
        n_unique_cols=f"{n_cols:,}",
        constant_cols=f"{constant_cols:,}",
        high_card_cols=f"{high_card_cols:,}",
        date_min=date_min,
        date_max=date_max,
        text_cols=f"{text_cols:,}",
        avg_text_len=avg_text_len,
        dataset_sample_section=sample_section_html or "",
        variables_section=variables_section_html,
        missing_values_section=missing_values_section_html,
        description_html=description_html,
        description_attr=description_attr,
    )
    return html


def render_empty_html(title: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title></head>
    <body><div class=\"container\"><h1>{title}</h1><p>Empty source.</p></div></body></html>
    """
