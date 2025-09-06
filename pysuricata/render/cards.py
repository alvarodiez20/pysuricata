from __future__ import annotations

from typing import Any, List, Optional, Tuple, Sequence
import html as _html
import math

import numpy as np

try:  # optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


from .format_utils import human_bytes as _human_bytes, fmt_num as _fmt_num, fmt_compact as _fmt_compact
from .svg_utils import safe_col_id as _safe_col_id, nice_ticks as _nice_ticks, fmt_tick as _fmt_tick, svg_empty as _svg_empty

# Tiny spark histogram SVG for numeric preview
def _spark_hist_svg(counts: Optional[List[int]], width: int = 180, height: int = 48, pad: int = 2) -> str:
    if not counts:
        return '<svg class="spark spark-hist" width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg>'.format(w=width, h=height)
    max_c = max(counts) or 1
    bar_w = (width - 2 * pad) / len(counts)
    parts = [f'<svg class="spark spark-hist" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="distribution">']
    for i, c in enumerate(counts):
        h = 0 if max_c == 0 else (c / max_c) * (height - 2 * pad)
        x = pad + i * bar_w
        y = height - pad - h
        parts.append(f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-1,1):.2f}" height="{h:.2f}" rx="1" ry="1"></rect>')
    parts.append("</svg>")
    return "".join(parts)


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
    col_id = _safe_col_id(s.name)
    safe_name = _html.escape(str(s.name))
    miss_pct = (s.missing / max(1, s.count + s.missing)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    zeros_pct = (s.zeros / max(1, s.count)) * 100.0 if s.count else 0.0
    neg_pct = (s.negatives / max(1, s.count)) * 100.0 if s.count else 0.0
    out_pct = (s.outliers_iqr / max(1, s.count)) * 100.0 if s.count else 0.0
    zeros_cls = 'warn' if zeros_pct > 30 else ''
    neg_cls = 'warn' if 0 < neg_pct <= 10 else ('crit' if neg_pct > 10 else '')
    out_cls = 'crit' if out_pct > 1 else ('warn' if out_pct > 0.3 else '')
    inf_cls = 'crit' if s.inf else ''
    approx_badge = '<span class="badge">approx</span>' if s.approx else ''

    # Discrete heuristic (approximate)
    discrete = False
    try:
        if s.int_like:
            if (s.unique_est <= max(1, min(50, int(0.05 * max(1, s.count))))) or (isinstance(s.unique_ratio_approx, float) and not math.isnan(s.unique_ratio_approx) and s.unique_ratio_approx <= 0.05):
                discrete = True
    except Exception:
        discrete = False

    # Quality flags
    flag_items = []
    if miss_pct > 0:
        flag_items.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    if s.inf:
        flag_items.append('<li class="flag bad">Has ∞</li>')
    if neg_pct > 0:
        cls = 'warn' if neg_pct > 10 else ''
        flag_items.append(f'<li class="flag {cls}">Has negatives</li>' if cls else '<li class="flag">Has negatives</li>')
    if zeros_pct >= 50.0:
        flag_items.append('<li class="flag bad">Zero‑inflated</li>')
    elif zeros_pct >= 30.0:
        flag_items.append('<li class="flag warn">Zero‑inflated</li>')
    if isinstance(s.min, (int, float)) and math.isfinite(s.min) and s.min > 0:
        flag_items.append('<li class="flag good">Positive‑only</li>')
    if isinstance(s.skew, float) and math.isfinite(s.skew):
        if s.skew >= 1:
            flag_items.append('<li class="flag warn">Skewed Right</li>')
        elif s.skew <= -1:
            flag_items.append('<li class="flag warn">Skewed Left</li>')
    if isinstance(s.kurtosis, float) and math.isfinite(s.kurtosis) and abs(s.kurtosis) >= 3:
        flag_items.append('<li class="flag bad">Heavy‑tailed</li>')
    if isinstance(s.jb_chi2, float) and math.isfinite(s.jb_chi2) and s.jb_chi2 <= 5.99:
        flag_items.append('<li class="flag good">≈ Normal (JB)</li>')
    if discrete:
        flag_items.append('<li class="flag warn">Discrete</li>')
    if isinstance(s.heap_pct, float) and math.isfinite(s.heap_pct) and s.heap_pct >= 30.0:
        flag_items.append('<li class="flag">Heaping</li>')
    if getattr(s, 'bimodal', False):
        flag_items.append('<li class="flag warn">Possibly bimodal</li>')
    if (isinstance(s.min, (int,float)) and math.isfinite(s.min) and s.min > 0) and (isinstance(s.skew, float) and math.isfinite(s.skew) and s.skew >= 1):
        flag_items.append('<li class="flag good">Log‑scale?</li>')
    try:
        uniq_est = max(0, int(s.unique_est))
        total_nonnull = max(1, int(s.count))
        unique_ratio = (uniq_est / total_nonnull) if total_nonnull else 0.0
        if uniq_est == 1:
            flag_items.append('<li class="flag bad">Constant</li>')
        elif unique_ratio <= 0.02 or uniq_est <= 2:
            flag_items.append('<li class="flag warn">Quasi‑constant</li>')
        if out_pct > 1.0:
            flag_items.append('<li class="flag bad">Many outliers</li>')
        elif out_pct > 0.3:
            flag_items.append('<li class="flag warn">Some outliers</li>')
        if total_nonnull > 1 and s.mono_inc:
            flag_items.append('<li class="flag good">Monotonic ↑</li>')
        elif total_nonnull > 1 and s.mono_dec:
            flag_items.append('<li class="flag good">Monotonic ↓</li>')
    except Exception:
        pass
    quality_flags_html = f"<ul class='quality-flags'>{''.join(flag_items)}</ul>" if flag_items else ""

    mem_display = _human_bytes(int(getattr(s, 'mem_bytes', 0)))
    inf_pct = (s.inf / max(1, s.count)) * 100.0 if s.count else 0.0
    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{s.count:,}</td></tr>
      <tr><th>Unique</th><td class=\"num\">{s.unique_est:,}{' (≈)' if s.approx else ''}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Outliers</th><td class=\"num {out_cls}\">{s.outliers_iqr:,} ({out_pct:.1f}%)</td></tr>
      <tr><th>Zeros</th><td class=\"num {zeros_cls}\">{s.zeros:,} ({zeros_pct:.1f}%)</td></tr>
      <tr><th>Infinites</th><td class=\"num {inf_cls}\">{s.inf:,} ({inf_pct:.1f}%)</td></tr>
      <tr><th>Negatives</th><td class=\"num {neg_cls}\">{s.negatives:,} ({neg_pct:.1f}%)</td></tr>
    </tbody></table>
    """

    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Min</th><td class=\"num\">{_fmt_num(s.min)}</td></tr>
      <tr><th>Median</th><td class=\"num\">{_fmt_num(s.median)}</td></tr>
      <tr><th>Mean</th><td class=\"num\">{_fmt_num(s.mean)}</td></tr>
      <tr><th>Max</th><td class=\"num\">{_fmt_num(s.max)}</td></tr>
      <tr><th>Q1</th><td class=\"num\">{_fmt_num(s.q1)}</td></tr>
      <tr><th>Q3</th><td class=\"num\">{_fmt_num(s.q3)}</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display} (≈)</td></tr>
    </tbody></table>
    """

    # Quantiles from sample
    svals = getattr(s, "sample_vals", None) or []
    def _q_from_sample(p: float) -> float:
        if not svals:
            return float("nan")
        i = (len(svals) - 1) * p
        lo = math.floor(i); hi = math.ceil(i)
        if lo == hi:
            return float(svals[int(i)])
        return float(svals[lo] * (hi - i) + svals[hi] * (i - lo))
    p1 = _q_from_sample(0.01)
    p5 = _q_from_sample(0.05)
    p10 = _q_from_sample(0.10)
    p90 = _q_from_sample(0.90)
    p95 = _q_from_sample(0.95)
    p99 = _q_from_sample(0.99)
    range_val = (s.max - s.min) if (isinstance(s.max, (int,float)) and isinstance(s.min, (int,float))) else float('nan')

    quant_stats_table = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Min</th><td class=\"num\">{_fmt_num(s.min)}</td></tr>
      <tr><th>P1</th><td class=\"num\">{_fmt_num(p1)}</td></tr>
      <tr><th>P5</th><td class=\"num\">{_fmt_num(p5)}</td></tr>
      <tr><th>P10</th><td class=\"num\">{_fmt_num(p10)}</td></tr>
      <tr><th>Q1 (P25)</th><td class=\"num\">{_fmt_num(s.q1)}</td></tr>
      <tr><th>Median (P50)</th><td class=\"num\">{_fmt_num(s.median)}</td></tr>
      <tr><th>Q3 (P75)</th><td class=\"num\">{_fmt_num(s.q3)}</td></tr>
      <tr><th>P90</th><td class=\"num\">{_fmt_num(p90)}</td></tr>
      <tr><th>P95</th><td class=\"num\">{_fmt_num(p95)}</td></tr>
      <tr><th>P99</th><td class=\"num\">{_fmt_num(p99)}</td></tr>
      <tr><th>Range</th><td class=\"num\">{_fmt_num(range_val)}</td></tr>
      <tr><th>Std Dev</th><td class=\"num\">{_fmt_num(s.std)}</td></tr>
    </tbody></table>
    """

    # Variants of histograms: lin/log axes with bins preset and adjustable
    hist_lin = _build_hist_svg_from_vals(safe_name, s.sample_vals or [], bins=25, scale='lin', width=420, height=160, scale_count=getattr(s, 'sample_scale', 1.0))
    hist_log = _build_hist_svg_from_vals(safe_name, s.sample_vals or [], bins=25, scale='log', width=420, height=160, scale_count=getattr(s, 'sample_scale', 1.0))
    spark = _spark_hist_svg(getattr(s, 'hist_counts', None))

    bins_list = [10, 25, 50]
    bin_buttons = " ".join(f'<button type="button" class="btn-soft btn-bins{(" active" if b==25 else "")}" data-bins="{b}">{b}</button>' for b in bins_list)
    chart_html = f"""
      <div class=\"hist-chart\">
        <div class=\"chart-variants\" data-col=\"{col_id}\"> 
          <div class=\"spark-wrap\">{spark}</div>
          <div class=\"hist variant active\" data-scale=\"lin\" data-bins=\"25\">{hist_lin}</div>
          <div class=\"hist variant\" data-scale=\"log\" data-bins=\"25\">{hist_log}</div>
        </div>
        <div class=\"chart-controls\"><span>Scale:</span> <button type=\"button\" class=\"btn-soft btn-scale active\" data-scale=\"lin\">Linear</button> <button type=\"button\" class=\"btn-soft btn-scale\" data-scale=\"log\">Log</button> | <span>Bins:</span> {bin_buttons}</div>
      </div>
    """

    corr_html = ""
    try:
        if getattr(s, 'corr_top', None):
            items = [f"<li><code>{_html.escape(str(k))}</code> <span class='num'>{v:+.2f}</span></li>" for k, v in s.corr_top]
            corr_html = f"<div class='corr-top'><div class='hdr'>Top correlations</div><ul>{''.join(items)}</ul></div>"
    except Exception:
        corr_html = ""

    details_html = f"""
      <section id=\"{col_id}-details\" class=\"details-section\" hidden>
        <nav class=\"tabs\" role=\"tablist\" aria-label=\"More details\">
          <button role=\"tab\" class=\"active\" data-tab=\"quantiles\">Quantiles</button>
        </nav>
        <div class=\"tab-panes\">
          <section class=\"tab-pane active\" data-tab=\"quantiles\">{quant_stats_table}</section>
        </div>
      </section>
    """

    return f"""
    <article class=\"var-card\" id=\"{col_id}\">
      <header class=\"var-card__header\"><div class=\"title\"><span class=\"colname\" title=\"{safe_name}\">{safe_name}</span>
        <span class=\"badge\">Numeric</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {approx_badge}
        {quality_flags_html}
      </div></header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}{corr_html}</div>
          <div class=\"box chart\">{chart_html}</div>
        </div>
        <div class=\"card-controls\" role=\"group\" aria-label=\"Numeric controls\">
          <div class=\"details-slot\">
            <button type=\"button\" class=\"details-toggle btn-soft\" aria-controls=\"{col_id}-details\" aria-expanded=\"false\">Details</button>
          </div>
          <div class=\"controls-slot\"></div>
        </div>
        {details_html}
      </div>
    </article>
    """


def render_dt_card(s: Any) -> str:
    col_id = _safe_col_id(s.name)
    safe_name = _html.escape(str(s.name))
    miss_pct = (s.missing / max(1, s.count + s.missing)) * 100.0
    miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
    flags = []
    if miss_pct > 0:
        flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
    if s.count > 1 and s.mono_inc:
        flags.append('<li class="flag good">Monotonic ↑</li>')
    if s.count > 1 and s.mono_dec:
        flags.append('<li class="flag good">Monotonic ↓</li>')
    quality_flags_html = f"<ul class='quality-flags'>{''.join(flags)}</ul>" if flags else ""

    def _fmt_ts(ts: Optional[int]) -> str:
        if ts is None:
            return '—'
        try:
            return datetime.utcfromtimestamp(ts / 1_000_000_000).isoformat() + 'Z'
        except Exception:
            return str(ts)

    mem_display = _human_bytes(getattr(s, 'mem_bytes', 0)) + ' (≈)'
    left_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Count</th><td class=\"num\">{s.count:,}</td></tr>
      <tr><th>Missing</th><td class=\"num {miss_cls}\">{s.missing:,} ({miss_pct:.1f}%)</td></tr>
      <tr><th>Min</th><td>{_fmt_ts(s.min_ts)}</td></tr>
      <tr><th>Max</th><td>{_fmt_ts(s.max_ts)}</td></tr>
      <tr><th>Processed bytes</th><td class=\"num\">{mem_display}</td></tr>
    </tbody></table>
    """

    def spark(counts: List[int]) -> str:
        if not counts:
            return ''
        m = max(counts) or 1
        blocks = '▁▂▃▄▅▆▇█'
        levels = [blocks[min(len(blocks)-1, int(c * (len(blocks)-1) / m))] for c in counts]
        return ''.join(levels)

    right_tbl = f"""
    <table class=\"kv\"><tbody>
      <tr><th>Hour</th><td class=\"small\">{spark(s.by_hour)}</td></tr>
      <tr><th>Day of week</th><td class=\"small\">{spark(s.by_dow)}</td></tr>
      <tr><th>Month</th><td class=\"small\">{spark(s.by_month)}</td></tr>
    </tbody></table>
    """

    def _dt_line_svg_from_sample(sample: Optional[List[int]], tmin: Optional[int], tmax: Optional[int], bins: int = 60, scale_count: float = 1.0) -> str:
        if not sample or tmin is None or tmax is None:
            return _svg_empty("dt-svg", 420, 160)
        a = np.asarray(sample, dtype=np.int64)
        if a.size == 0:
            return _svg_empty("dt-svg", 420, 160)
        if tmin == tmax:
            tmax = tmin + 1
        counts, edges = np.histogram(a, bins=int(max(10, min(bins, 180))), range=(int(tmin), int(tmax)))
        counts = np.maximum(0, np.round(counts * max(1.0, float(scale_count)))).astype(int)
        y_max = int(max(1, counts.max()))
        width, height = 420, 160
        margin_left, margin_right, margin_top, margin_bottom = 45, 8, 8, 32
        iw = width - margin_left - margin_right
        ih = height - margin_top - margin_bottom
        def sx(x):
            return margin_left + (x - tmin) / (tmax - tmin) * iw
        def sy(y):
            return margin_top + (1 - y / y_max) * ih
        centers = (edges[:-1] + edges[1:]) / 2.0
        pts = " ".join(f"{sx(x):.2f},{sy(float(c)):.2f}" for x, c in zip(centers, counts))
        y_ticks, _ = _nice_ticks(0, y_max, 5)
        n_xt = 5
        xt_vals = np.linspace(tmin, tmax, n_xt)
        span_ns = tmax - tmin
        def _fmt_xt(v):
            try:
                ts = pd.to_datetime(int(v))
                if span_ns <= 3 * 24 * 3600 * 1e9:
                    return ts.strftime('%Y-%m-%d %H:%M')
                return ts.date().isoformat()
            except Exception:
                return str(v)
        parts = [
            f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Timeline">',
            '<g class="plot-area">'
        ]
        for yt in y_ticks:
            parts.append(f'<line class="grid" x1="{margin_left}" y1="{sy(yt):.2f}" x2="{margin_left + iw}" y2="{sy(yt):.2f}"></line>')
        parts.append(f'<polyline class="line" points="{pts}"></polyline>')
        parts.append('<g class="hotspots">')
        for i, c in enumerate(counts):
            if not np.isfinite(c):
                continue
            x0p = sx(edges[i])
            x1p = sx(edges[i+1])
            wp = max(1.0, x1p - x0p)
            cp = (edges[i] + edges[i+1]) / 2.0
            label = _fmt_xt(cp)
            title = f"{int(c)} rows&#10;{label}"
            parts.append(
                f'<rect class="hot" x="{x0p:.2f}" y="{margin_top}" width="{wp:.2f}" height="{ih:.2f}" fill="transparent" pointer-events="all">'
                f'<title>{title}</title>'
                f'</rect>'
            )
        parts.append('</g>')
        parts.append('</g>')
        x_axis_y = margin_top + ih
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left+iw}" y2="{x_axis_y}"></line>')
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')
        for yt in y_ticks:
            py = sy(yt)
            parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py:.2f}" x2="{margin_left}" y2="{py:.2f}"></line>')
            lab = int(round(yt))
            parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3:.2f}" text-anchor="end">{lab}</text>')
        for xv in xt_vals:
            px = sx(xv)
            parts.append(f'<line class="tick" x1="{px:.2f}" y1="{x_axis_y}" x2="{px:.2f}" y2="{x_axis_y + 4}"></line>')
            parts.append(f'<text class="tick-label" x="{px:.2f}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_xt(xv)}</text>')
        parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">Time</text>')
        parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">Count</text>')
        parts.append('</svg>')
        return ''.join(parts)

    chart_html = _dt_line_svg_from_sample(getattr(s, 'sample_ts', None), s.min_ts, s.max_ts, bins=60, scale_count=getattr(s, 'sample_scale', 1.0))

    # Details tables
    hours_tbl = ''.join(f'<tr><th>{h:02d}</th><td class="num">{c:,}</td></tr>' for h, c in enumerate(s.by_hour))
    dows = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    dows_tbl = ''.join(f'<tr><th>{dows[i]}</th><td class="num">{c:,}</td></tr>' for i, c in enumerate(s.by_dow))
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_tbl = ''.join(f'<tr><th>{months[i]}</th><td class="num">{c:,}</td></tr>' for i, c in enumerate(s.by_month))

    details_html = f'''
      <section id="{col_id}-details" class="details-section" hidden>
        <nav class="tabs" role="tablist" aria-label="More details">
          <button role="tab" class="active" data-tab="breakdown">Breakdown</button>
        </nav>
        <div class="tab-panes">
          <section class="tab-pane active" data-tab="breakdown">
            <div class="grid-2col">
              <table class="kv"><thead><tr><th>Hour</th><th>Count</th></tr></thead><tbody>{hours_tbl}</tbody></table>
              <table class="kv"><thead><tr><th>Day</th><th>Count</th></tr></thead><tbody>{dows_tbl}</tbody></table>
            </div>
            <div class="grid-2col" style="margin-top:8px;">
              <table class="kv"><thead><tr><th>Month</th><th>Count</th></tr></thead><tbody>{months_tbl}</tbody></table>
            </div>
          </section>
        </div>
      </section>
    '''

    return f"""
    <article class=\"var-card\" id=\"{col_id}\"> 
      <header class=\"var-card__header\"><div class=\"title\"><span class=\"colname\">{safe_name}</span>
        <span class=\"badge\">Datetime</span>
        <span class=\"dtype chip\">{s.dtype_str}</span>
        {quality_flags_html}
      </div></header>
      <div class=\"var-card__body\">
        <div class=\"triple-row\">
          <div class=\"box stats-left\">{left_tbl}</div>
          <div class=\"box stats-right\">{right_tbl}</div>
          <div class=\"box chart\">{chart_html}</div>
        </div>
        <div class=\"card-controls\" role=\"group\" aria-label=\"Column controls\">
          <div class=\"details-slot\">
            <button type=\"button\" class=\"details-toggle btn-soft\" aria-controls=\"{col_id}-details\" aria-expanded=\"false\">Details</button>
          </div>
          <div class=\"controls-slot\"></div>
        </div>
        {details_html}
      </div>
    </article>
    """


# === Public builder helpers (standalone) ===
def build_hist_svg_with_axes(
    s: "pd.Series",  # type: ignore[name-defined]
    bins: int = 20,
    width: int = 420,
    height: int = 160,
    margin_left: int = 45,
    margin_bottom: int = 36,
    margin_top: int = 8,
    margin_right: int = 8,
    sample_cap: Optional[int] = 200_000,
    scale: str = "lin",
    auto_bins: bool = True,
) -> str:
    try:
        ss = s.dropna()
        if ss.empty:
            return _svg_empty("hist-svg", width, height)
        vals = pd.to_numeric(ss, errors="coerce").dropna().to_numpy()  # type: ignore[name-defined]
        if vals.size == 0:
            return _svg_empty("hist-svg", width, height)
        if scale == "log":
            vals = vals[vals > 0]
            if vals.size == 0:
                return _svg_empty("hist-svg", width, height)
        if sample_cap is not None and vals.size > int(sample_cap):
            try:
                rng = np.random.default_rng(0)
                idx = rng.choice(vals.size, size=int(sample_cap), replace=False)
                vals = vals[idx]
            except Exception:
                vals = vals[: int(sample_cap)]
        if auto_bins and vals.size > 1:
            try:
                q1 = float(np.quantile(vals, 0.25))
                q3 = float(np.quantile(vals, 0.75))
                iqr_local = q3 - q1
                if iqr_local > 0:
                    h = 2.0 * iqr_local * (vals.size ** (-1.0 / 3.0))
                    if h > 0:
                        fd_bins = int(np.clip(np.ceil((float(np.max(vals)) - float(np.min(vals))) / h), 10, 200))
                        bins = fd_bins
            except Exception:
                pass
        name = str(getattr(s, "name", "Value"))
        return _build_hist_svg_from_vals(
            name, vals, bins=int(bins), width=width, height=height,
            margin_left=margin_left, margin_bottom=margin_bottom,
            margin_top=margin_top, margin_right=margin_right,
            scale=scale,
        )
    except Exception:
        return _svg_empty("hist-svg", width, height)


def build_cat_bar_svg(
    s: "pd.Series",  # type: ignore[name-defined]
    top: int = 10,
    width: int = 420,
    height: int = 160,
    margin_left: int = 120,
    margin_right: int = 12,
    margin_top: int = 8,
    margin_bottom: int = 8,
    scale: str = "count",
    include_other: bool = True,
) -> str:
    try:
        s2 = s.astype(str).dropna()
        if s2.empty:
            return _svg_empty("cat-svg", width, height)
        vc = s2.value_counts()
        if vc.empty:
            return _svg_empty("cat-svg", width, height)
        items: List[Tuple[str, int]] = []
        if int(top) > 0:
            head = vc.head(int(top))
            items = [(str(k), int(v)) for k, v in head.items()]
            if include_other and len(vc) > int(top):
                other = int(vc.iloc[int(top):].sum())
                items.append(("Other", other))
        else:
            items = [(str(k), int(v)) for k, v in vc.items()]
        total = int(vc.sum())
        return _build_cat_bar_svg_from_items(items, total=total, scale=scale)
    except Exception:
        return _svg_empty("cat-svg", width, height)


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
