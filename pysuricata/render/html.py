from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import time
import html as _html
from datetime import datetime

from ..utils import load_css, load_template, embed_favicon, embed_image, load_script
from ..accumulators import NumericAccumulator, CategoricalAccumulator, DatetimeAccumulator, BooleanAccumulator
from ..compute.infer import ColumnKinds
from .cards import render_numeric_card as _render_numeric_card, render_cat_card as _render_cat_card, render_dt_card as _render_dt_card, render_bool_card as _render_bool_card
from ..report_v2 import _resolve_pysuricata_version  # reuse version resolver


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(max(0, n))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024.0


def render_html_snapshot(
    *,
    kinds: ColumnKinds,
    accs: Dict[str, Any],
    first_columns: List[str],
    row_kmv: Any,
    total_missing_cells: int,
    approx_mem_bytes: int,
    start_time: float,
    cfg: Any,
    report_title: Optional[str],
    sample_section_html: str,
) -> str:
    kinds_map = {
        **{name: ("numeric", accs[name]) for name in kinds.numeric},
        **{name: ("categorical", accs[name]) for name in kinds.categorical},
        **{name: ("datetime", accs[name]) for name in kinds.datetime},
        **{name: ("boolean", accs[name]) for name in kinds.boolean},
    }

    miss_list: List[tuple[str, float, int]] = []
    for name, (kind, acc) in kinds_map.items():
        miss = getattr(acc, "missing", 0)
        cnt = getattr(acc, "count", 0) + miss
        pct = (miss / cnt * 100.0) if cnt else 0.0
        miss_list.append((name, pct, miss))
    miss_list.sort(key=lambda t: t[1], reverse=True)
    top_missing_list = ""
    for col, pct, count in miss_list[:5]:
        severity_class = "low" if pct <= 5 else ("medium" if pct <= 20 else "high")
        top_missing_list += f"""
        <li class=\"missing-item\"> 
          <div class=\"missing-info\"> 
            <code class=\"missing-col\" title=\"{_html.escape(str(col))}\">{_html.escape(str(col))}</code>
            <span class=\"missing-stats\">{count:,} ({pct:.1f}%)</span>
          </div>
          <div class=\"missing-bar\"><div class=\"missing-fill {severity_class}\" style=\"width:{pct:.1f}%;\"></div></div>
        </li>
        """
    if not top_missing_list:
        top_missing_list = """
        <li class=\"missing-item\"><div class=\"missing-info\"><code class=\"missing-col\">None</code><span class=\"missing-stats\">0 (0.0%)</span></div><div class=\"missing-bar\"><div class=\"missing-fill low\" style=\"width:0%;\"></div></div></li>
        """

    n_rows = int(getattr(row_kmv, "rows", 0))
    n_cols = len(kinds_map)
    total_cells = n_rows * n_cols
    missing_overall = f"{total_missing_cells:,} ({(total_missing_cells/max(1,total_cells)*100):.1f}%)"
    dup_rows, dup_pct = row_kmv.approx_duplicates()
    duplicates_overall = f"{dup_rows:,} ({dup_pct:.1f}%)"

    constant_cols = 0
    high_card_cols = 0
    for name, (kind, acc) in kinds_map.items():
        if kind in ("numeric", "categorical"):
            u = acc._uniques.estimate() if hasattr(acc, "_uniques") else getattr(acc, "unique_est", 0)
        elif kind == "datetime":
            u = acc.unique_est
        else:
            present = (acc.true_n > 0) + (acc.false_n > 0)
            u = int(present)
        total = getattr(acc, "count", 0) + getattr(acc, "missing", 0)
        if u <= 1:
            constant_cols += 1
        if kind == "categorical" and n_rows:
            if (u / n_rows) > 0.5:
                high_card_cols += 1

    if kinds.datetime:
        mins, maxs = [], []
        for name in kinds.datetime:
            acc = accs[name]
            if acc._min_ts is not None: mins.append(acc._min_ts)
            if acc._max_ts is not None: maxs.append(acc._max_ts)
        if mins and maxs:
            date_min = datetime.utcfromtimestamp(min(mins) / 1_000_000_000).isoformat() + "Z"
            date_max = datetime.utcfromtimestamp(max(maxs) / 1_000_000_000).isoformat() + "Z"
        else:
            date_min = date_max = "—"
    else:
        date_min = date_max = "—"

    text_cols = len(kinds.categorical)
    avg_text_len_vals = [acc.avg_len for name, (k, acc) in kinds_map.items() if k == "categorical" and acc.avg_len is not None]
    avg_text_len = f"{(sum(avg_text_len_vals)/len(avg_text_len_vals)):.1f}" if avg_text_len_vals else "—"

    col_order = [c for c in list(first_columns) if c in kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean] or (kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean)
    all_cards_list: List[str] = []
    for name in col_order:
        acc = accs[name]
        if name in kinds.numeric:
            all_cards_list.append(_render_numeric_card(acc.finalize()))
        elif name in kinds.categorical:
            all_cards_list.append(_render_cat_card(acc.finalize()))
        elif name in kinds.datetime:
            all_cards_list.append(_render_dt_card(acc.finalize()))
        elif name in kinds.boolean:
            all_cards_list.append(_render_bool_card(acc.finalize()))
    variables_section_html = f"""
        <section id=\"vars\">  
          <span id=\"numeric-vars\" class=\"anchor-alias\"></span>
          <h2 class=\"section-title\">Variables</h2>
          <p class=\"muted small\">Analyzing {len(kinds.numeric)+len(kinds.categorical)+len(kinds.datetime)+len(kinds.boolean)} variables ({len(kinds.numeric)} numeric, {len(kinds.categorical)} categorical, {len(kinds.datetime)} datetime, {len(kinds.boolean)} boolean).</p>
          <div class=\"cards-grid\">{''.join(all_cards_list)}</div>
        </section>
    """
    sections_html = (sample_section_html or "") + variables_section_html

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
    logo_light_path = os.path.join(static_dir, "images", "logo_suricata_transparent.png")
    logo_dark_path  = os.path.join(static_dir, "images", "logo_suricata_transparent_dark_mode.png")
    logo_light_img = embed_image(logo_light_path, element_id="logo-light", alt_text="Logo", mime_type="image/png")
    logo_dark_img  = embed_image(logo_dark_path,  element_id="logo-dark",  alt_text="Logo (dark)", mime_type="image/png")
    logo_html = f'<span id="logo">{logo_light_img}{logo_dark_img}</span>'
    favicon_path = os.path.join(static_dir, "images", "favicon.ico")
    favicon_tag = embed_favicon(favicon_path)

    end_time = time.time()
    duration_seconds = end_time - start_time
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pysuricata_version = _resolve_pysuricata_version()
    repo_url = "https://github.com/alvarodiez20/pysuricata"

    html = template.format(
      favicon=favicon_tag,
      css=css_tag,
      script=script_content,
      logo=logo_html,
      report_title= report_title or cfg.title,
      report_date=report_date,
      pysuricata_version=pysuricata_version,
      report_duration=f"{duration_seconds:.2f}",
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
      dataset_sample_section=sections_html,
    )
    return html
