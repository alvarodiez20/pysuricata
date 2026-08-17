from __future__ import annotations

import html as _html
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from .._version import resolve_version as _resolve_pysuricata_version
from ..compute.core.types import ColumnKinds
from ..utils import (
    embed_favicon,
    load_css_dir,
    load_script,
    load_template,
)
from .cards import render_bool_card as _render_bool_card
from .cards import render_cat_card as _render_cat_card
from .cards import render_dt_card as _render_dt_card
from .cards import render_numeric_card as _render_numeric_card
from .composition_bar import CompositionBarRenderer
from .format_utils import human_bytes as _human_bytes
from .format_utils import human_time as _human_time
from .markdown_utils import render_markdown_to_html
from .missing_columns import create_missing_columns_renderer
from .svg_utils import safe_col_id as _safe_col_id
from .triage import actionable_chips as _actionable_chips
from .triage import build_attention_block as _build_attention_block
from .triage import extract_chips as _extract_chips

# Template placeholders are bare identifiers in braces ({report_title}). Anything
# else that looks brace-wrapped -- CSS custom properties, JS object literals --
# either fails to match or resolves to no key and is left verbatim.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _build_logo(svg_path: str) -> str:
    """The header lockup: the mark inline as SVG, the product name as type.

    This used to be two base64 PNGs -- 578 KB of a 1.23 MB report, 47% of the
    document, to draw a mark 30 CSS pixels tall. There were two because the
    artwork had the wordmark baked into it, and the wordmark needed a different
    colour in dark mode, so the report shipped both and hid one with CSS.

    Setting the name as type instead of art removes the duplicate outright: text
    follows ``currentColor`` into dark mode, so there is nothing to swap and
    nothing to keep in sync. It also reads better, because the drawn wordmark is
    a display face whose letters land about eight pixels tall at header size.

    If the asset is missing -- a partial install, a stripped wheel -- the report
    still renders, with the name and without the mark. A logo is not worth
    failing a profile run over.
    """
    try:
        with open(svg_path, encoding="utf-8") as handle:
            mark = handle.read().strip()
    except OSError:
        mark = ""
    return f'<span id="logo">{mark}<span class="wordmark">pysuricata</span></span>'


def _build_dataset_name(name: str | None) -> str:
    """The name of what was profiled, for the header bar.

    Returns the empty string when there is nothing to show, and the separator
    is part of what is returned rather than part of the template. An in-memory
    frame has no name -- most inputs are in-memory frames -- so the absent case
    is the common one, and a template that hard-coded the divider would render
    a rule with nothing after it on most reports.
    """
    text = (name or "").strip()
    if not text:
        return ""
    return (
        '<span class="bar-sep" aria-hidden="true"></span>'
        f'<span class="dataset-name" title="{_html.escape(text)}">{_html.escape(text)}</span>'
    )


def _column_mix(numeric: int, categorical: int, datetime: int, boolean: int) -> str:
    """`3 num · 8 cat · 1 bool`, naming only the types that are present.

    Printing every type regardless gives `1 num · 0 cat · 0 date · 0 bool` for
    a single-column frame, which is four facts to convey one. A type with no
    columns is already stated by the composition legend below.
    """
    parts = [
        (numeric, "num"),
        (categorical, "cat"),
        (datetime, "date"),
        (boolean, "bool"),
    ]
    present = [f"{count:,} {name}" for count, name in parts if count]
    return " · ".join(present) if present else "none"


def _quick_facts(
    *,
    unique_cols: int,
    constant_cols: int,
    high_card_cols: int,
    text_cols: int,
    avg_text_len: str,
    date_min: str,
    date_max: str,
) -> str:
    """One mono run in place of five bordered pills.

    Five pills is five borders to state five short facts, and the borders were
    doing none of the work. The date range is dropped rather than shown empty
    when there are no datetime columns.
    """
    facts = [
        f"{unique_cols:,} unique",
        f"{constant_cols:,} constant",
        f"{high_card_cols:,} high-cardinality",
    ]
    if text_cols:
        facts.append(f"{text_cols:,} text (avg len {avg_text_len})")
    if date_min != "—" and date_max != "—":
        span = f"{date_min} → {date_max}".replace("<br>", " ")
        facts.append(f"range {span}")
    else:
        facts.append("no date range")
    return " · ".join(facts)


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
    report_title: str | None,
    sample_section_html: str,
    chunk_metadata: list[tuple[int, int, int]] | None = None,
    corr_est: Any | None = None,
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
    # The duplicate count is `rows - distinct`, so the whole absolute error of
    # the distinct estimate lands on it -- a quantity that is usually far
    # smaller. `≈ KMV sketch` alone read as the sketch's own ~1%, which is the
    # error on a different number: at 200,000 rows with 2,000 true duplicates
    # the figure came back 47% high while the distinct estimate was 0.48% off.
    #
    # The threshold itself is applied by `RowKMV.duplicates()`, not here. It
    # used to live in this function alone, so the report suppressed an
    # unresolvable count while `summarize()` published it raw -- the report
    # correct, the versioned payload wrong.
    if hasattr(row_kmv, "duplicates"):
        dup_rows, dup_pct, dup_sigma, dup_resolvable = row_kmv.duplicates()
    else:  # pragma: no cover - a row sketch that predates `duplicates()`
        dup_rows, dup_pct = row_kmv.approx_duplicates()
        dup_sigma, dup_resolvable = 0, True
    if dup_resolvable:
        duplicates_value = f"{dup_rows:,}"
        # No bound when the count is exact -- KMV counts exactly until it has
        # seen k distinct values, so most frames have no estimation error here
        # and "± 0" would be noise.
        duplicates_note = f"± {dup_sigma:,} · KMV sketch" if dup_sigma else "exact"
        duplicates_overall = f"{dup_rows:,} ({dup_pct:.1f}%)"
    else:
        # Below the resolution of the sketch. A figure here would invite a
        # conclusion it cannot support, so state the ceiling instead.
        ceiling_pct = (dup_sigma / n_rows * 100.0) if n_rows else 0.0
        duplicates_value = f"&lt; {dup_sigma:,}"
        duplicates_note = "below sketch resolution"
        duplicates_overall = f"under {dup_sigma:,} ({ceiling_pct:.1f}%)"

    constant_cols = 0
    high_card_cols = 0
    for name, (kind, acc) in kinds_map.items():
        # `unique_est` is a property on every accumulator kind, so this needs no
        # branch and no reach into `_uniques` -- which a native accumulator
        # could not expose with those names and semantics anyway (#64).
        u = int(getattr(acc, "unique_est", 0))
        if kind == "boolean":
            # A boolean column reports 2 by definition; what the constant-column
            # count wants to know is how many values actually turned up.
            u = int((acc.true_n > 0) + (acc.false_n > 0))
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
    all_cards_list: list[str] = []
    column_chips: list[tuple[str, str, list[tuple[str, str]]]] = []
    for name in col_order:
        acc = accs[name]
        card_html = ""
        data_type = ""

        if name in kinds.numeric:
            card_html = _render_numeric_card(acc.finalize(chunk_metadata))
            data_type = "numeric"
        elif name in kinds.categorical:
            card_html = _render_cat_card(acc.finalize(chunk_metadata))
            data_type = "categorical"
        elif name in kinds.datetime:
            card_html = _render_dt_card(acc.finalize(chunk_metadata))
            data_type = "datetime"
        elif name in kinds.boolean:
            card_html = _render_bool_card(acc.finalize(chunk_metadata))
            data_type = "boolean"

        # Add data attributes for filtering and search
        if card_html:
            # The chips this card already emitted, carried on the article so the
            # triage block and the chip filter can both use them without
            # recomputing anything.
            chips = _extract_chips(card_html)
            # The chip's stamped slug, not one re-derived from its face: the
            # face leads with the column's own value, so deriving here gave
            # every card a set of flags no other card could share and no filter
            # could group. See #238.
            flags = " ".join(sorted({slug for _, _, slug in _actionable_chips(chips)}))
            card_id = _safe_col_id(name)
            column_chips.append((name, card_id, chips))
            card_html = card_html.replace(
                f'<article class="var-card" id="{card_id}">',
                f'<article class="var-card" id="{card_id}" data-type="{data_type}"'
                f' data-name="{_html.escape(name)}" data-flags="{flags}">',
            )
            all_cards_list.append(card_html)
    # Build variables section with pagination and search
    total_variables = (
        len(kinds.numeric)
        + len(kinds.categorical)
        + len(kinds.datetime)
        + len(kinds.boolean)
    )
    attention_html = _build_attention_block(column_chips)
    variables_section_html = f"""
          {attention_html}
          <p class=\"muted small\">Analyzing {total_variables} variables ({len(kinds.numeric)} numeric, {len(kinds.categorical)} categorical, {len(kinds.datetime)} datetime, {len(kinds.boolean)} boolean).</p>

          <div class=\"vars-controls\">
            <div class=\"controls-row\">
              <label for=\"search-input\" class=\"sr-only\">Search columns</label>
              <input type=\"text\" placeholder=\"Search columns...\" id=\"search-input\" aria-label=\"Search columns\">
              <div class=\"filter-buttons\">
                <button class=\"tab active\" data-filter=\"all\">All</button>
                <button class=\"tab\" data-filter=\"numeric\">Numeric</button>
                <button class=\"tab\" data-filter=\"categorical\">Categorical</button>
                <button class=\"tab\" data-filter=\"datetime\">Datetime</button>
                <button class=\"tab\" data-filter=\"boolean\">Boolean</button>
              </div>
            </div>
            <div class=\"info\" id=\"pagination-info\">Showing 1-{min(10, total_variables)} of {total_variables}</div>
          </div>

          <div class=\"cards-grid\" id=\"cards-grid\">
            {"".join(all_cards_list)}
          </div>

          <div class=\"pagination\" id=\"pagination\">
            <button id=\"prev-btn\" {"disabled" if total_variables <= 10 else ""}>←</button>
            <div class=\"pages\" id=\"page-numbers\"></div>
            <button id=\"next-btn\" {"disabled" if total_variables <= 10 else ""}>→</button>
          </div>
    """

    module_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(module_dir)
    static_dir = os.path.join(pkg_dir, "static")
    template_dir = os.path.join(pkg_dir, "templates")
    template_path = os.path.join(template_dir, "report_template.html")
    template = load_template(template_path)
    css_dir = os.path.join(static_dir, "css")
    css_tag = load_css_dir(css_dir)
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

    # Generate correlations section (before missing values)
    from .correlations_section import CorrelationsSectionRenderer

    correlations_renderer = CorrelationsSectionRenderer()
    correlations_section_html = correlations_renderer.render_section(
        corr_est, kinds.numeric, cfg.corr_threshold
    )

    # Generate missing values section
    from .missing_section import MissingValuesSectionRenderer

    missing_section_renderer = MissingValuesSectionRenderer()
    missing_values_section_html = missing_section_renderer.render_section(
        kinds_map, accs, n_rows, n_cols, total_missing_cells
    )
    logo_html = _build_logo(os.path.join(static_dir, "images", "logo_mark.svg"))
    favicon_path = os.path.join(static_dir, "images", "favicon.ico")
    favicon_tag = embed_favicon(favicon_path)

    end_time = time.time()
    duration_seconds = end_time - start_time
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Generate unique report ID for description localStorage isolation
    report_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:20]
    pysuricata_version = _resolve_pysuricata_version()
    repo_url = "https://github.com/alvarodiez20/pysuricata"
    author_url = "https://github.com/alvarodiez20"
    # The GitHub handle, not the legal name: the link goes to the profile, and
    # the handle is what a reader can act on. Matches the bare "pysuricata" that
    # sits beside it — a repo name with no owner prefix.
    author_name = "alvarodiez20"

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

    composition_bar = CompositionBarRenderer().render(
        numeric=len(kinds.numeric),
        categorical=len(kinds.categorical),
        datetime=len(kinds.datetime),
        boolean=len(kinds.boolean),
    )

    # The description is a margin note. Empty, it is one hairline row offering
    # to add one -- reports generated in a loop never carry a description, and
    # must not be disfigured by an invitation nobody will accept.
    has_description = bool(description_html)
    description_state = "" if has_description else " is-empty"
    description_label = "Note" if has_description else "Description"
    description_action = "edit" if has_description else "+ add a note"

    missing_pct_value = (
        (total_missing_cells / total_cells * 100.0) if total_cells else 0.0
    )
    # Past the threshold the sub-line takes the warning colour. --q-warn-text,
    # not --q-warn-fill: this is 11.5px type, and the fill step is deliberately
    # below the text minimum so bars can be lighter than words.
    missing_tone = "is-warn" if missing_pct_value >= 5.0 else ""
    complete_columns = sum(1 for _, pct, _ in miss_list if pct <= 0.0)
    complete_note = (
        f'<p class="miss-complete">{complete_columns:,} of {n_cols:,} columns'
        f" {'is' if complete_columns == 1 else 'are'} complete</p>"
        if n_cols
        else ""
    )

    # Substitution is done with a single regex pass rather than str.format()
    # because CSS custom properties ({--var-name}) and JavaScript braces would be
    # read by .format() as named placeholders and raise KeyError. A single pass is
    # also required for correctness: with sequential str.replace() calls, a value
    # substituted early (e.g. a user-supplied title containing "{report_date}")
    # would itself be rescanned and expanded by a later replacement.
    replacements = {
        "css": css_tag,
        "script": combined_script_content,
        "favicon": favicon_tag,
        "logo": logo_html,
        "report_title": report_title or cfg.title,
        "dataset_name_html": _build_dataset_name(getattr(cfg, "dataset_name", "")),
        "report_date": report_date,
        "report_id": report_id,
        "pysuricata_version": pysuricata_version,
        "report_duration": _human_time(duration_seconds),
        "repo_url": repo_url,
        "author_url": author_url,
        "author_name": author_name,
        "n_rows": f"{n_rows:,}",
        "n_cols": f"{n_cols:,}",
        "memory_usage": _human_bytes(approx_mem_bytes) if approx_mem_bytes else "—",
        "missing_overall": missing_overall,
        "duplicates_overall": duplicates_overall,
        "numeric_cols": str(len(kinds.numeric)),
        "categorical_cols": str(len(kinds.categorical)),
        "datetime_cols": str(len(kinds.datetime)),
        "bool_cols": str(len(kinds.boolean)),
        "composition_bar": composition_bar,
        "col_mix": _column_mix(
            len(kinds.numeric),
            len(kinds.categorical),
            len(kinds.datetime),
            len(kinds.boolean),
        ),
        "missing_pct": f"{missing_pct_value:.1f}%",
        "missing_cells": f"{total_missing_cells:,} cells",
        "missing_tone": missing_tone,
        "duplicates_value": duplicates_value,
        "duplicates_note": duplicates_note,
        "complete_columns_note": complete_note,
        "description_state": description_state,
        "description_label": description_label,
        "description_action": description_action,
        "quick_facts": _quick_facts(
            unique_cols=n_cols,
            constant_cols=constant_cols,
            high_card_cols=high_card_cols,
            text_cols=text_cols,
            avg_text_len=avg_text_len,
            date_min=date_min,
            date_max=date_max,
        ),
        "top_missing_list": top_missing_list,
        "n_unique_cols": f"{n_cols:,}",
        "constant_cols": f"{constant_cols:,}",
        "high_card_cols": f"{high_card_cols:,}",
        "date_min": date_min,
        "date_max": date_max,
        "text_cols": f"{text_cols:,}",
        "avg_text_len": avg_text_len,
        "dataset_sample_section": sample_section_html or "",
        "variables_section": variables_section_html,
        "correlations_section": correlations_section_html,
        "missing_values_section": missing_values_section_html,
        "description_html": description_html,
        "description_attr": description_attr,
    }

    def _resolve(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in replacements:
            return str(replacements[key])
        # Unknown key: leave the original text untouched so CSS/JS braces survive.
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_resolve, template)


def render_empty_html(title: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title></head>
    <body><div class=\"container\"><h1>{title}</h1><p>Empty source.</p></div></body></html>
    """
