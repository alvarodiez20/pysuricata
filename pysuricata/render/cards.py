from __future__ import annotations

from typing import Any, List, Optional, Tuple, Sequence
import html as _html
import math

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _safe_col_id(name: str) -> str:
    return "col_" + "".join(ch if str(ch).isalnum() else "_" for ch in str(name))


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(max(0, n))
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024.0


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "NaN"
        return f"{x:,.4g}"
    except Exception:
        return str(x)


def _svg_empty(css_class: str, width: int, height: int, aria_label: str = "no data") -> str:
    return f'<svg class="{css_class}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{aria_label}"></svg>'


def _nice_ticks(vmin: float, vmax: float, n: int = 5):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return [0, 1], 1
    if vmin == vmax:
        vmax = vmin + 1
    rng = vmax - vmin
    raw_step = rng / max(1, n)
    def nice_num(rng: float, do_round: bool = True) -> float:
        if rng <= 0 or not np.isfinite(rng):
            return 1.0
        exp = math.floor(math.log10(rng))
        frac = rng / (10 ** exp)
        if do_round:
            if frac < 1.5:
                nice = 1
            elif frac < 3:
                nice = 2
            elif frac < 7:
                nice = 5
            else:
                nice = 10
        else:
            if frac <= 1:
                nice = 1
            elif frac <= 2:
                nice = 2
            elif frac <= 5:
                nice = 5
            else:
                nice = 10
        return nice * (10 ** exp)
    step = nice_num(raw_step, True)
    start = math.floor(vmin / step) * step
    ticks = []
    x = start
    while x <= vmax + 1e-9:
        ticks.append(float(x))
        x += step
    return ticks, step


def _fmt_tick(v: float, step: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if step >= 1:
        return f"{v:.0f}"
    if step >= 0.1:
        return f"{v:.1f}"
    if step >= 0.01:
        return f"{v:.2f}"
    try:
        return f"{v:.4g}"
    except Exception:
        return str(v)


def _fmt_compact(x) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
    except Exception:
        pass
    try:
        return f"{x:.4g}"
    except Exception:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return str(x)


def _build_hist_svg_from_vals(
    base_title: str,
    vals: Sequence[float],
    *,
    bins: int = 25,
    width: int = 420,
    height: int = 160,
    margin_left: int = 45,
    margin_bottom: int = 36,
    margin_top: int = 8,
    margin_right: int = 8,
    scale: str = "lin",
    scale_count: float = 1.0,
    x_min_override: Optional[float] = None,
    x_max_override: Optional[float] = None,
) -> str:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _svg_empty("hist-svg", width, height)
    if scale == "log":
        arr = arr[arr > 0]
        if arr.size == 0:
            return _svg_empty("hist-svg", width, height)
        arr = np.log10(arr)

    x_min, x_max = float(np.min(arr)), float(np.max(arr))
    if x_min_override is not None and np.isfinite(x_min_override):
        x_min = float(x_min_override)
    if x_max_override is not None and np.isfinite(x_max_override):
        x_max = float(x_max_override)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    counts, edges = np.histogram(arr, bins=int(bins), range=(x_min, x_max))
    counts_scaled = np.maximum(0, np.round(counts * max(1.0, float(scale_count)))).astype(int)
    y_max = int(max(1, counts_scaled.max()))
    total_n = int(counts_scaled.sum()) if counts_scaled.size else 0

    iw = width - margin_left - margin_right
    ih = height - margin_top - margin_bottom

    def sx(x):
        return margin_left + (x - x_min) / (x_max - x_min) * iw

    def sy(y):
        return margin_top + (1 - y / y_max) * ih

    x_ticks, x_step = _nice_ticks(x_min, x_max, 6)
    xt = [x for x in x_ticks if x >= x_min - 1e-9 and x <= x_max + 1e-9]
    if not xt or abs(xt[0] - x_min) > 1e-9:
        xt = [x_min] + [x for x in xt if x > x_min]
    x_ticks = xt
    y_ticks, y_step = _nice_ticks(0, y_max, 5)

    parts = [
        f'<svg class="hist-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Histogram">',
        '<g class="plot-area">'
    ]

    for i, c in enumerate(counts_scaled):
        x0 = edges[i]
        x1 = edges[i + 1]
        x = sx(x0)
        w = max(1.0, sx(x1) - sx(x0) - 1.0)
        y = sy(int(c))
        h = (margin_top + ih) - y
        pct = (c / total_n * 100.0) if total_n else 0.0
        parts.append(
            f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="1" ry="1" '
            f'data-count="{int(c)}" data-pct="{pct:.1f}" data-x0="{_fmt_compact(x0)}" data-x1="{_fmt_compact(x1)}">'
            f'<title>{int(c)} rows ({pct:.1f}%)&#10;[{_fmt_compact(x0)} – {_fmt_compact(x1)}]</title>'
            f'</rect>'
        )
    parts.append('</g>')

    x_axis_y = margin_top + ih
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left + iw}" y2="{x_axis_y}"></line>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')

    for xt in x_ticks:
        px = sx(xt)
        parts.append(f'<line class="tick" x1="{px}" y1="{x_axis_y}" x2="{px}" y2="{x_axis_y + 4}"></line>')
        parts.append(f'<text class="tick-label" x="{px}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_tick(xt, x_step)}</text>')
    for yt in y_ticks:
        py = sy(yt)
        parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py}" x2="{margin_left}" y2="{py}"></line>')
        parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3}" text-anchor="end">{_fmt_tick(yt, y_step)}</text>')

    x_title = f"log10({base_title})" if scale == "log" else base_title
    parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">{x_title}</text>')
    parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">Count</text>')

    parts.append('</svg>')
    return ''.join(parts)


def render_numeric_card(s: Any) -> str:
    # For now, import and delegate to the implementation in report_v2 to keep parity
    # This shim can be expanded to fully isolate rendering logic if needed.
    # Avoid circular import at module import time
    from ..report_v2 import _render_numeric_card as _impl  # type: ignore
    return _impl(s)


def render_dt_card(s: Any) -> str:
    from ..report_v2 import _render_dt_card as _impl  # type: ignore
    return _impl(s)


def _build_cat_bar_svg_from_items(
    items: List[Tuple[str, int]],
    total: int,
    *,
    width: int = 420,
    height: int = 160,
    margin_top: int = 8,
    margin_bottom: int = 8,
    margin_left_min: int = 120,
    margin_right: int = 12,
    scale: str = "count",
) -> str:
    if total <= 0 or not items:
        return _svg_empty("cat-svg", width, height)
    labels = [_html.escape(str(k)) for k, _ in items]
    counts = [int(c) for _, c in items]
    pcts = [(c / total * 100.0) for c in counts]

    max_label_len = max((len(l) for l in labels), default=0)
    char_w = 7
    gutter = max(60, min(180, char_w * min(max_label_len, 28) + 16))
    mleft = max(margin_left_min, gutter)

    n = len(labels)
    iw = width - mleft - margin_right
    ih = height - margin_top - margin_bottom
    if n <= 0 or iw <= 0 or ih <= 0:
        return _svg_empty("cat-svg", width, height)
    bar_gap = 6
    bar_h = max(4, (ih - bar_gap * (n - 1)) / max(n, 1))

    if scale == "pct":
        vmax = max(pcts) or 1.0
        values = pcts
    else:
        vmax = float(max(counts)) or 1.0
        values = counts

    def sx(v: float) -> float:
        return mleft + (v / vmax) * iw

    parts = [f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Top categories">']
    for i, (label, c, p, val) in enumerate(zip(labels, counts, pcts, values)):
        y = margin_top + i * (bar_h + bar_gap)
        x0 = mleft
        x1 = sx(float(val))
        w = max(1.0, x1 - x0)
        short = (label[:24] + "…") if len(label) > 24 else label
        parts.append(
            f'<g class="bar-row">'
            f'<rect class="bar" x="{x0:.2f}" y="{y:.2f}" width="{w:.2f}" height="{bar_h:.2f}" rx="2" ry="2">'
            f'<title>{label}\n{c:,} rows ({p:.1f}%)</title>'
            f'</rect>'
            f'<text class="bar-label" x="{mleft-6}" y="{y + bar_h/2 + 3:.2f}" text-anchor="end">{short}</text>'
            f"<text class=\"bar-value\" x=\"{(x1 - 6 if w >= 56 else x1 + 4):.2f}\" y=\"{y + bar_h/2 + 3:.2f}\" text-anchor=\"{('end' if w >= 56 else 'start')}\">{c:,} ({p:.1f}%)</text>"
            f'</g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_bool_card(s: Any) -> str:
    col_id = _safe_col_id(s.name)
    total = int(s.true_n + s.false_n + s.missing)
    cnt = int(s.true_n + s.false_n)
    miss_pct = (s.missing / max(1, total)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    true_pct_total = (s.true_n / max(1, total)) * 100.0
    false_pct_total = (s.false_n / max(1, total)) * 100.0

    mem_display = _human_bytes(getattr(s, 'mem_bytes', 0)) + ' (≈)'

    flags = []
    if miss_pct > 0:
        flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    if cnt > 0 and (s.true_n == 0 or s.false_n == 0):
        flags.append('<li class="flag bad">Constant</li>')
    if cnt > 0:
        p = s.true_n / cnt
        if p <= 0.05 or p >= 0.95:
            flags.append('<li class="flag warn">Imbalanced</li>')
    quality_flags_html = f"<ul class='quality-flags'>{''.join(flags)}</ul>" if flags else ""

    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{cnt:,}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Unique</th><td class=\"num\">{(int(s.true_n>0)+int(s.false_n>0))}</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display}</td></tr>
    </tbody></table>
    """
    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>True</th><td class=\"num\">{s.true_n:,} ({true_pct_total:.1f}%)</td></tr>
      <tr><th>False</th><td class=\"num\">{s.false_n:,} ({false_pct_total:.1f}%)</td></tr>
    </tbody></table>
    """

    def _bool_stack_svg(true_n: int, false_n: int, miss: int, width: int = 420, height: int = 48, margin: int = 4) -> str:
        total = max(1, int(true_n + false_n + miss))
        inner_w = width - 2 * margin
        seg_h = height - 2 * margin
        w_false = int(inner_w * (false_n / total))
        w_true = int(inner_w * (true_n / total))
        w_miss = max(0, inner_w - w_false - w_true)
        parts = [f'<svg class="bool-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
        x = margin
        parts.append(f'<rect class="false" x="{x}" y="{margin}" width="{w_false}" height="{seg_h}"><title>False: {false_n:,}</title></rect>')
        x += w_false
        parts.append(f'<rect class="true" x="{x}" y="{margin}" width="{w_true}" height="{seg_h}"><title>True: {true_n:,}</title></rect>')
        x += w_true
        if w_miss:
            parts.append(f'<rect class="missing" x="{x}" y="{margin}" width="{w_miss}" height="{seg_h}"><title>Missing: {miss:,}</title></rect>')
        parts.append('</svg>')
        return ''.join(parts)

    chart_html = _bool_stack_svg(int(s.true_n), int(s.false_n), int(s.missing))
    safe_name = _html.escape(str(s.name))
    col_id = _safe_col_id(s.name)
    return f"""
    <article class=\"var-card\" id=\"{col_id}\">
      <header class=\"var-card__header\"><div class=\"title\"><span class=\"colname\">{safe_name}</span>
        <span class=\"badge\">Boolean</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {quality_flags_html}
      </div></header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}</div>
          <div class=\"box chart\">{chart_html}</div>
        </div>
      </div>
    </article>
    """


def render_cat_card(s: Any) -> str:
    col_id = _safe_col_id(s.name)
    safe_name = _html.escape(str(s.name))
    total = s.count + s.missing
    miss_pct = (s.missing / max(1, total)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    approx_badge = '<span class="badge">approx</span>' if s.approx else ''

    mode_label, mode_n = (s.top_items[0] if s.top_items else ("—", 0))
    safe_mode_label = _html.escape(str(mode_label))
    mode_pct = (mode_n / max(1, s.count)) * 100.0 if s.count else 0.0

    import math
    if s.count > 0 and s.top_items:
        probs = [c / s.count for _, c in s.top_items]
        entropy = float(-sum(p * math.log2(max(p, 1e-12)) for p in probs))
    else:
        entropy = float('nan')

    rare_count = 0
    rare_cov = 0.0
    if s.count > 0:
        for _, c in s.top_items:
            pct = c / s.count * 100.0
            if pct < 1.0:
                rare_count += 1
                rare_cov += pct
    rare_cls = 'crit' if rare_cov > 60 else ('warn' if rare_cov >= 30 else '')
    top5_cov = 0.0
    if s.count > 0 and s.top_items:
        top5_cov = sum(c for _, c in s.top_items[:5]) / s.count * 100.0
    top5_cls = 'good' if top5_cov >= 80 else ('warn' if top5_cov <= 40 else '')
    empty_cls = 'warn' if s.empty_zero > 0 else ''

    flags = []
    if s.unique_est > max(200, int(0.5 * max(1, s.count))):
        flags.append('<li class="flag warn">High cardinality</li>')
    if mode_n >= int(0.7 * max(1, s.count)) and s.count:
        flags.append('<li class="flag warn">Dominant category</li>')
    if rare_cov >= 30.0:
        flags.append('<li class="flag warn">Many rare levels</li>')
    if s.case_variants_est > 0:
        flags.append('<li class="flag">Case variants</li>')
    if s.trim_variants_est > 0:
        flags.append('<li class="flag">Trim variants</li>')
    if s.empty_zero > 0:
        flags.append('<li class="flag">Empty strings</li>')
    if miss_pct > 0:
        flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    quality_flags_html = f"<ul class=\"quality-flags\">{''.join(flags)}</ul>" if flags else ""

    mem_display = _human_bytes(int(getattr(s, 'mem_bytes', 0)))
    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{s.count:,}</td></tr>
      <tr><th>Unique</th><td class=\"num\">{s.unique_est:,}{' (≈)' if s.approx else ''}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Mode</th><td><code>{safe_mode_label}</code></td></tr>
      <tr><th>Mode %</th><td class=\"num\">{mode_pct:.1f}%</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display} (≈)</td></tr>
    </tbody></table>
    """

    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Entropy</th><td class=\"num\">{_fmt_num(entropy)}</td></tr>
      <tr><th>Rare levels</th><td class=\"num {rare_cls}\">{rare_count:,} ({rare_cov:.1f}%)</td></tr>
      <tr><th>Top 5 coverage</th><td class=\"num {top5_cls}\">{top5_cov:.1f}%</td></tr>
      <tr><th>Label length (avg)</th><td class=\"num\">{_fmt_num(s.avg_len)}</td></tr>
      <tr><th>Length p90</th><td class=\"num\">{s.len_p90 if s.len_p90 is not None else '—'}</td></tr>
      <tr><th>Empty strings</th><td class=\"num {empty_cls}\">{s.empty_zero:,}</td></tr>
    </tbody></table>
    """

    items = s.top_items or []
    maxN = max(1, min(15, len(items)))
    candidates = [5, 10, 15, maxN]
    topn_list = sorted({n for n in candidates if 1 <= n <= maxN})
    default_topn = 10 if 10 in topn_list else (max(topn_list) if topn_list else maxN)

    variants_html_parts = []
    for n in topn_list:
        if len(items) > n:
            keep = max(1, n - 1)
            head = items[:keep]
            other = sum(c for _, c in items[keep:])
            data = head + [("Other", other)]
        else:
            data = items[:n]
        svg = _build_cat_bar_svg_from_items(data, total=max(1, s.count + s.missing))
        active = " active" if n == default_topn else ""
        variants_html_parts.append(f'<div class="topn{active}" data-topn="{n}">{svg}</div>')
    topn_switch = " ".join(f'<button type="button" class="btn-soft btn-topn{(" active" if n == default_topn else "")}" data-topn="{n}">{n}</button>' for n in topn_list)
    chart_html = f"""
      <div class=\"topn-chart\">
        <div class=\"chart-variants\">{''.join(variants_html_parts)}</div>
        <div class=\"chart-controls\"><span>Top‑N:</span> {topn_switch}</div>
      </div>
    """

    return f"""
    <article class=\"var-card\" id=\"{col_id}\"> 
      <header class=\"var-card__header\"><div class=\"title\"><span class=\"colname\">{safe_name}</span>
        <span class=\"badge\">Categorical</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {approx_badge}
        {quality_flags_html}
      </div></header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}</div>
          <div class=\"box chart\">{chart_html}</div>
        </div>
      </div>
    </article>
    """
