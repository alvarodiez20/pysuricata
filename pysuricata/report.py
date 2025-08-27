import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional, List
import json
from .utils import (
    load_css,
    load_template,
    embed_favicon,
    embed_image,
    load_script,
)


# Resolve pysuricata version in a robust way (installed or source checkout)
try:
    try:
        from importlib.metadata import version as _pkg_version  # Python 3.8+
        from importlib.metadata import PackageNotFoundError as _PkgNotFound
    except Exception:  # pragma: no cover - fallback for older envs
        from importlib_metadata import version as _pkg_version  # type: ignore
        from importlib_metadata import PackageNotFoundError as _PkgNotFound  # type: ignore
except Exception:  # last resort
    _pkg_version = None
    class _PkgNotFound(Exception):
        pass

def _resolve_pysuricata_version() -> str:
    # Prefer installed metadata
    if _pkg_version is not None:
        try:
            return _pkg_version("pysuricata")
        except _PkgNotFound:
            pass
        except Exception:
            pass
    # Fallback to module attribute when running from source
    try:
        from . import __version__  # type: ignore
        if isinstance(__version__, str) and __version__:
            return __version__
    except Exception:
        pass
    # Environment override or default
    return os.getenv("PYSURICATA_VERSION", "dev")


def generate_report(
    data: pd.DataFrame,
    output_file: Optional[str] = None,
    report_title: Optional[str] = "PySuricata EDA Report",
    columns: Optional[List[str]] = None,
    include_sample: bool = True,
) -> str:
    """
    Generate an HTML report containing summary statistics, missing values, and a correlation matrix.

    This function expects the input data as a Pandas DataFrame, computes summary statistics,
    missing values, and the correlation matrix. It then loads an HTML template and embeds
    CSS and images (logo and favicon) using Base64 encoding so that the report is self-contained.
    Optionally, the report can be written to an output file.

    Args:
        data (pd.DataFrame):
            The input data as a Pandas DataFrame.
        output_file (Optional[str]):
            File path to save the HTML report. If None, the report is not written to disk.
        report_title (Optional[str]):
            Title of the report. Defaults to "PySuricata EDA Report" if not provided.
        columns (Optional[List[str]]):
            Column names (used when the data is a 2D NumPy array).
        include_sample (bool):
            Whether to include a collapsible Dataset Sample section (first 5 rows). Defaults to True.

    Returns:
        str: A string containing the complete HTML report.
    """

    start_time = time.time()

    df = data

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
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]", "datetimetz"]).shape[1]
    bool_cols = df.select_dtypes(include=["bool"]).shape[1]

    # Extra metrics for Quick Insights
    n_unique_cols = df.nunique(dropna=False).count()
    constant_cols = int((df.nunique(dropna=False) <= 1).sum())
    # High-cardinality categoricals: unique ratio > 0.5
    high_card_cols = int(((df.select_dtypes(include=["object", "category"])\
                          .nunique(dropna=False) / n_rows) > 0.5).sum())
    # Date range (min -> max) for any datetime col (robust to tz-aware/naive mixing)
    if datetime_cols > 0:
        _dt_all = df.select_dtypes(include=["datetime", "datetime64[ns]", "datetimetz"])
        mins, maxs = [], []
        for _c in _dt_all.columns:
            _s = _dt_all[_c].dropna()
            if _s.empty:
                continue
            try:
                if hasattr(_s.dt, "tz") and _s.dt.tz is not None:
                    _s_use = _s.dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    _s_use = _s
                mins.append(_s_use.min())
                maxs.append(_s_use.max())
            except Exception:
                try:
                    _s2 = pd.to_datetime(_s, errors="coerce")
                    if hasattr(_s2.dt, "tz") and _s2.dt.tz is not None:
                        _s2 = _s2.dt.tz_convert("UTC").dt.tz_localize(None)
                    mins.append(_s2.min())
                    maxs.append(_s2.max())
                except Exception:
                    pass
        if mins and maxs:
            date_min = str(min(mins))
            date_max = str(max(maxs))
        else:
            date_min, date_max = "—", "—"
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

    # Build optional dataset sample section
    if include_sample:
        try:
            n = min(10, len(df.index))
            # Take a random sample (or an empty frame if there are no rows)
            sample_df = df.sample(n=n) if n > 0 else df.head(0)

            # Compute original positional row numbers regardless of index type
            row_pos = pd.Index(df.index).get_indexer(sample_df.index)

            # Insert row numbers as the first column on the left
            sample_df = sample_df.copy()
            sample_df.insert(0, "", row_pos)

            # Render the HTML table (no pandas index column)
            # Mark numeric columns for right alignment using a span
            num_cols = sample_df.columns.intersection(df.select_dtypes(include=[np.number]).columns)
            for c in num_cols:
                sample_df[c] = sample_df[c].map(lambda v: f'<span class="num">{v}</span>' if pd.notna(v) else "")

            # Render the HTML table (no pandas index column); allow spans
            sample_html_table = sample_df.to_html(classes="sample-table", index=False, escape=False)
        except Exception:
            sample_html_table = "<em>Unable to render sample preview.</em>"

        dataset_sample_section = f"""
        <section id="sample" class="collapsible-card">
          <span id="dataset-sample" class="anchor-alias"></span>
          <details class="card">
            <summary>
              <span>Sample</span>
              <span class="chev" aria-hidden="true">▸</span>
            </summary>
            <div class="card-content">
              <div class="sample-scroll">{sample_html_table}</div>
              <p class="muted small">Showing 10 randomly sampled rows.</p>
            </div>
          </details>
        </section>
        """
    else:
        dataset_sample_section = ""

    # =============================
    # PER-COLUMN ANALYSIS (Numeric) – TABLE VIEW
    # =============================
    def _hist_svg_from_series(
        s: pd.Series,
        bins: int = 20,
        width: int = 180,
        height: int = 48,
        pad: int = 2,
        scale: str = "lin",
        sample_cap: int = 50_000,
    ) -> str:
        """Return an inline SVG histogram for a numeric Series.
        scale: 'lin' or 'log' (log uses log10 on strictly positive values).
        Designed for spark-sized cells.
        """
        s = s.dropna()
        if s.empty:
            return f'<svg class="spark spark-hist" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
        vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            return f'<svg class="spark spark-hist" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
        if scale == "log":
            vals = vals[vals > 0]
            if vals.size == 0:
                return f'<svg class="spark spark-hist" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
            vals = np.log10(vals)
        if vals.size > sample_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(vals.size, size=sample_cap, replace=False)
            vals = vals[idx]
        counts, _edges = np.histogram(vals, bins=bins)
        max_c = counts.max() if counts.max() > 0 else 1
        bar_w = (width - 2 * pad) / bins
        parts = [
            f'<svg class="spark spark-hist" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="distribution">'
        ]
        for i, c in enumerate(counts):
            h = 0 if max_c == 0 else (c / max_c) * (height - 2 * pad)
            x = pad + i * bar_w
            y = height - pad - h
            parts.append(
                f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-1,1):.2f}" height="{h:.2f}" rx="1" ry="1"></rect>'
            )
        parts.append("</svg>")
        return "".join(parts)

    def _ecdf_svg_from_series(
        s: pd.Series,
        width: int = 180,
        height: int = 48,
        pad: int = 2,
        scale: str = "lin",
        points: int = 192,
        sample_cap: int = 200_000,
    ) -> str:
        """Return a tiny inline SVG ECDF polyline for a numeric Series (lin/log)."""
        s = s.dropna()
        if s.empty:
            return f'<svg class="spark spark-ecdf" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
        vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        if scale == "log":
            vals = vals[vals > 0]
            if vals.size == 0:
                return f'<svg class="spark spark-ecdf" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
            vals = np.log10(vals)
        if vals.size > sample_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(vals.size, size=sample_cap, replace=False)
            vals = vals[idx]
        vals = np.sort(vals)
        n = vals.size
        if n == 0:
            return f'<svg class="spark spark-ecdf" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
        if n > points:
            qs = np.linspace(0, 1, points)
            xs = np.quantile(vals, qs)
            ys = qs
        else:
            xs = vals
            ys = np.arange(1, n + 1) / n
        x_min, x_max = np.min(xs), np.max(xs)
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        def _map_x(x):
            return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
        def _map_y(y):
            return height - pad - y * (height - 2 * pad)
        pts = " ".join(f"{_map_x(x):.2f},{_map_y(y):.2f}" for x, y in zip(xs, ys))
        return (
            f'<svg class="spark spark-ecdf" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="ECDF">'
            f'<polyline fill="none" stroke-width="1" points="{pts}"></polyline>'
            f"</svg>"
        )

    def _hist_svg_with_axes(
        s: pd.Series,
        bins: int = 20,
        width: int = 420,
        height: int = 160,
        margin_left: int = 45,
        margin_bottom: int = 36,
        margin_top: int = 8,
        margin_right: int = 8,
        sample_cap: int = 200_000,
        scale: str = "lin",
        auto_bins: bool = True,
    ) -> str:
        """Return an inline SVG histogram with axes for a numeric Series.
        scale: 'lin' (default). Values are sampled for speed when very large.
        Adds subtle quantile bands and native tooltips on bars for quick insights.
        """
        s = s.dropna()
        if s.empty:
            return f'<svg class="hist-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
        vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            return f'<svg class="hist-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
        if vals.size > sample_cap:
            rng = np.random.default_rng(0)
            vals = vals[rng.choice(vals.size, size=sample_cap, replace=False)]
        # Optional log10 scale
        if scale == "log":
            vals = vals[vals > 0]
            if vals.size == 0:
                return f'<svg class="hist-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
            vals = np.log10(vals)

        x_min, x_max = float(np.min(vals)), float(np.max(vals))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        # IQR for bin-width selection (Freedman–Diaconis)
        q1 = float(np.quantile(vals, 0.25)) if vals.size else np.nan
        q3 = float(np.quantile(vals, 0.75)) if vals.size else np.nan
        iqr_local = q3 - q1 if np.isfinite(q3) and np.isfinite(q1) else 0.0

        # Freedman–Diaconis bin rule (keeps visuals crisp for large n)
        if auto_bins:
            if vals.size > 1 and iqr_local > 0:
                h = 2.0 * iqr_local * (vals.size ** (-1.0 / 3.0))
                if h > 0:
                    fd_bins = int(np.clip(np.ceil((x_max - x_min) / h), 10, 200))
                    bins = fd_bins

        counts, edges = np.histogram(vals, bins=bins, range=(x_min, x_max))
        y_max = int(max(1, counts.max()))

        total_n = int(counts.sum()) if counts.size else 0

        iw = width - margin_left - margin_right
        ih = height - margin_top - margin_bottom

        def sx(x):
            return margin_left + (x - x_min) / (x_max - x_min) * iw

        def sy(y):
            return margin_top + (1 - y / y_max) * ih

        # Nicer ticks (D3-like) for more representative scales
        def _nice_num(rng, do_round=True):
            import math
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

        def _nice_ticks(vmin, vmax, n=5):
            import math
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            if vmax == vmin:
                vmax = vmin + 1
            rng = _nice_num(vmax - vmin, do_round=False)
            step = _nice_num(rng / (n - 1), do_round=True)
            nice_min = math.floor(vmin / step) * step
            nice_max = math.ceil(vmax / step) * step
            ticks = []
            t = nice_min
            while t <= nice_max + step * 1e-9 and len(ticks) < 50:
                ticks.append(t)
                t += step
            return ticks, step

        x_ticks, x_step = _nice_ticks(x_min, x_max, 5)
        y_ticks, y_step = _nice_ticks(0, y_max, 5)

        def _fmt_tick(v, step):
            if not np.isfinite(v):
                return ''
            if step >= 1:
                return f"{int(round(v))}"
            elif step >= 0.1:
                return f"{v:.1f}"
            elif step >= 0.01:
                return f"{v:.2f}"
            else:
                return _fmt(v)

        parts = [
            f'<svg class="hist-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Histogram">',
            '<g class="plot-area">'
        ]

        # Bars with native tooltips
        for i, c in enumerate(counts):
            x0 = edges[i]
            x1 = edges[i + 1]
            x = sx(x0)
            w = max(1.0, sx(x1) - sx(x0) - 1.0)
            y = sy(c)
            h = (margin_top + ih) - y
            pct = (c / total_n * 100.0) if total_n else 0.0
            parts.append(
                f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="1" ry="1" '
                f'data-count="{c}" data-pct="{pct:.1f}" data-x0="{_fmt(x0)}" data-x1="{_fmt(x1)}">'
                f'<title>{c} rows ({pct:.1f}%)&#10;[{_fmt(x0)} – {_fmt(x1)}]</title>'
                f'</rect>'
            )
        parts.append('</g>')

        # Axes
        x_axis_y = margin_top + ih
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left + iw}" y2="{x_axis_y}"></line>')
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')

        # (Percentile/quantile guide lines removed)

        # X ticks & labels
        for xt in x_ticks:
            px = sx(xt)
            parts.append(f'<line class="tick" x1="{px}" y1="{x_axis_y}" x2="{px}" y2="{x_axis_y + 4}"></line>')
            parts.append(f'<text class="tick-label" x="{px}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_tick(xt, x_step)}</text>')
        # Y ticks & labels
        for yt in y_ticks:
            py = sy(yt)
            parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py}" x2="{margin_left}" y2="{py}"></line>')
            parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3}" text-anchor="end">{_fmt_tick(yt, y_step)}</text>')

        # Axis titles
        base_title = str(getattr(s, "name", None)) if getattr(s, "name", None) is not None else "Value"
        x_title = (f"log10({base_title})" if scale == "log" else base_title)
        y_title = "Count"
        parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">{x_title}</text>')
        parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">{y_title}</text>')

        parts.append('</svg>')
        return ''.join(parts)

    def _safe_col_id(name: str) -> str:
        return "col_" + "".join(ch if ch.isalnum() else "_" for ch in str(name))

    def _fmt(x) -> str:
        try:
            if pd.isna(x):
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

    # =============================
    # Tiny horizontal bar chart for categorical Top-N
    # =============================
    def _cat_bar_svg(
        s: pd.Series,
        top: int = 10,
        width: int = 420,
        height: int = 160,
        margin_left: int = 120,
        margin_right: int = 12,
        margin_top: int = 8,
        margin_bottom: int = 8,
        scale: str = "count",   # 'count' or 'pct'
        include_other: bool = True,
    ) -> str:
        """Return an inline SVG with horizontal bars for the Top-N categories (+ optional Other).
        - Labels are ellipsized visually; full label is available via <title>.
        - scale='count' or 'pct' controls axis values in the tooltip; bar lengths scale to the chosen metric.
        """
        vc = s.value_counts(dropna=False)
        total = int(vc.sum()) if vc.size else 0
        if total == 0:
            return f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'

        # Build display labels handling NaN
        labels = ["(Missing)" if pd.isna(idx) else str(idx) for idx in vc.index]
        df_counts = pd.DataFrame({"label": labels, "count": vc.values})
        df_counts = df_counts.groupby("label", as_index=False)["count"].sum().sort_values("count", ascending=False)

        if include_other and df_counts.shape[0] > top:
            keep = max(1, top - 1)
            top_df = df_counts.head(keep).copy()
            other_count = int(df_counts["count"].iloc[keep:].sum())
            top_df = pd.concat(
                [top_df, pd.DataFrame([["Other", other_count]], columns=["label", "count"])],
                ignore_index=True,
            )
        else:
            top_df = df_counts.head(top).copy()
        top_df["pct"] = top_df["count"] / total * 100.0
        # Dynamically size the left label gutter so short labels don't waste horizontal space
        max_label_len = int(top_df["label"].map(lambda x: len(str(x))).max()) if not top_df.empty else 0
        _char_w = 7  # approx px per char for our 11px font
        _gutter = max(60, min(180, _char_w * min(max_label_len, 28) + 16))
        mleft = _gutter
        mright = 6

        n = len(top_df)
        if n == 0:
            return f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'

        iw = width - mleft - mright
        ih = height - margin_top - margin_bottom
        bar_gap = 6
        bar_h = max(4, (ih - bar_gap * (n - 1)) / max(n, 1))

        if scale == "pct":
            vmax = float(top_df["pct"].max()) or 1.0
            vals = top_df["pct"].to_numpy()
        else:
            vmax = float(top_df["count"].max()) or 1.0
            vals = top_df["count"].to_numpy()

        def sx(v: float) -> float:
            return mleft + (v / vmax) * iw

        parts = [
            f'<svg class="cat-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Top categories">'
        ]
        for i, (label, c, p, val) in enumerate(zip(top_df["label"], top_df["count"], top_df["pct"], vals)):
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

    # =============================
    # PER-COLUMN ANALYSIS (Numeric) – CARD VIEW
    # =============================
    numeric_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cards = {}

    # Precompute correlations (lightweight cap)
    corr_matrix = None
    try:
        num_df_for_corr = df.select_dtypes(include=[np.number])
        if num_df_for_corr.shape[1] >= 2 and num_df_for_corr.shape[1] <= 40:
            corr_matrix = num_df_for_corr.corr()
    except Exception:
        corr_matrix = None

    for _col in numeric_cols_list:
        s = df[_col]
        col_id = _safe_col_id(_col)
        n_total = int(s.size)
        miss = int(s.isna().sum())
        miss_pct = (miss / n_total * 100.0) if n_total else 0.0
        uniq = int(s.nunique(dropna=False))
        s_nonnull = s.dropna()
        cnt = int(s_nonnull.size)
        uniq_nonnull = int(s_nonnull.nunique()) if cnt > 0 else 0
        unique_ratio = (uniq_nonnull / cnt) if cnt > 0 else 0.0

        if cnt > 0:
            qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            q_all = s_nonnull.quantile(qs)
            p1 = q_all.loc[0.01]; q5 = q_all.loc[0.05]; p10 = q_all.loc[0.10]
            q1 = q_all.loc[0.25]; med = q_all.loc[0.50]; q3 = q_all.loc[0.75]
            p90 = q_all.loc[0.90]; q95 = q_all.loc[0.95]; p99 = q_all.loc[0.99]
            mn = s_nonnull.min(); mx = s_nonnull.max()
            range_val = mx - mn
            mean = s_nonnull.mean(); std = s_nonnull.std()
            # New lightweight descriptive stats
            var = float(std ** 2) if np.isfinite(std) else float("nan")
            se = float(std / np.sqrt(cnt)) if cnt > 1 else float("nan")
            cv = float(std / mean) if np.isfinite(std) and np.isfinite(mean) and mean != 0 else float("nan")
            gmean = float(np.exp(np.log(s_nonnull).mean())) if mn > 0 else float("nan")
            iqr = q3 - q1
            mad = float(np.median(np.abs(s_nonnull - med)))
            skew = s_nonnull.skew() if cnt > 2 else float("nan")
            kurt = s_nonnull.kurtosis() if cnt > 3 else float("nan")
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            out_n = int(((s_nonnull < lo) | (s_nonnull > hi)).sum())
            out_pct = (out_n / cnt * 100.0)
            zeros_n = int((s_nonnull.eq(0)).sum())
            zeros_pct = float(zeros_n / cnt * 100.0)
            neg_n = int((s_nonnull.lt(0)).sum())
            neg_pct = float(neg_n / cnt * 100.0)
            vals_np = pd.to_numeric(s_nonnull, errors="coerce").to_numpy()
            inf_n = int(np.isinf(vals_np).sum())
            inf_pct = float(inf_n / cnt * 100.0)
            # 95% CI for mean
            if np.isfinite(se):
                ci_lo = float(mean - 1.96 * se)
                ci_hi = float(mean + 1.96 * se)
            else:
                ci_lo = ci_hi = float("nan")
            ci_str = f"[{_fmt(ci_lo)}, {_fmt(ci_hi)}]" if np.isfinite(ci_lo) and np.isfinite(ci_hi) else "—"
        else:
            mn = mx = mean = std = iqr = mad = skew = kurt = float("nan")
            var = se = cv = gmean = float("nan")
            ci_lo = ci_hi = float("nan"); ci_str = "—"
            out_n = 0; out_pct = 0.0; zeros_n = 0; zeros_pct = 0.0; neg_n = 0; neg_pct = 0.0
            q1 = med = q3 = q5 = q95 = p1 = p10 = p90 = p99 = range_val = float("nan")
            inf_n = 0; inf_pct = 0.0

        try:
            col_mem_bytes = int(s.memory_usage(index=False, deep=True))
        except TypeError:
            # pandas < 1.3 fallback (index kw not supported)
            try:
                col_mem_bytes = int(s.memory_usage(deep=True))
            except Exception:
                col_mem_bytes = 0
        except Exception:
            col_mem_bytes = 0
        col_mem_display = _human_bytes(col_mem_bytes)

        # Semantic flags & helpers
        is_int = str(s.dtype).startswith("int")
        discrete = bool(cnt and is_int and (uniq <= min(50, int(0.05 * cnt))))
        positive_only = bool(cnt and (mn > 0))
        skewed_right = bool(np.isfinite(skew) and skew >= 1)
        skewed_left = bool(np.isfinite(skew) and skew <= -1)
        heavy_tailed = bool(np.isfinite(kurt) and abs(kurt) >= 3)
        likely_id = bool(is_int and cnt and (uniq >= int(0.98 * cnt)) and miss == 0)

        # --- Lightweight extra diagnostics: JB, heaping, granularity, bimodality ---
        # Jarque–Bera normality (χ² df=2), no SciPy needed; threshold 5.99 ≈ p<0.05
        try:
            jb = float(cnt/6.0 * ((skew if np.isfinite(skew) else 0.0)**2 + 0.25*(kurt if np.isfinite(kurt) else 0.0)**2)) if cnt > 3 else float("nan")
        except Exception:
            jb = float("nan")
        jb_is_normal = bool(np.isfinite(jb) and jb <= 5.99)

        # Heaping / round-number bias (% at round values)
        heap_pct = float("nan")
        try:
            if cnt > 0:
                if is_int:
                    v_int = s_nonnull.astype("Int64")
                    heap_pct = float((((v_int % 10 == 0) | (v_int % 5 == 0)).mean()) * 100.0)
                else:
                    v = pd.to_numeric(s_nonnull, errors="coerce").dropna()
                    if v.size:
                        frac = (v - v.round()).abs().to_numpy()
                        heap_pct = float(((frac <= 0.002) | (np.abs(frac - 0.5) <= 0.002)).mean() * 100.0)
        except Exception:
            pass

        # Granularity (decimals) and step size (approx)
        gran_decimals = None
        gran_step = None
        try:
            vals_for_gran = pd.to_numeric(s_nonnull, errors="coerce").dropna()
            if vals_for_gran.size > 50000:
                vals_for_gran = vals_for_gran.sample(50000, random_state=0)
            if not is_int and not vals_for_gran.empty:
                decs = vals_for_gran.map(lambda x: len(str(x).split('.')[-1]) if ('.' in str(x)) else 0)
                if not decs.empty:
                    gran_decimals = int(decs.mode().iloc[0])
            uniq_vals = np.sort(vals_for_gran.unique())
            if uniq_vals.size >= 2:
                diffs = np.diff(uniq_vals)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size > 0:
                    gran_step = float(np.median(diffs))
        except Exception:
            pass

        # Bimodality hint (valley detection on binned counts)
        bimodal = False
        try:
            vals_bi = pd.to_numeric(s_nonnull, errors="coerce").dropna().to_numpy()
            if vals_bi.size >= 10:
                nb = min(25, max(10, int(np.sqrt(vals_bi.size))))
                c_bi, _e_bi = np.histogram(vals_bi, bins=nb)
                if c_bi.size >= 3:
                    valley = (c_bi[1:-1] < c_bi[:-2]) & (c_bi[1:-1] < c_bi[2:])
                    bimodal = bool((valley & (c_bi[1:-1] <= 0.8 * c_bi.max())).sum() >= 1)
        except Exception:
            bimodal = False

        flag_items = []
        # Existing semantic flags
        if discrete:
            flag_items.append('<li class="flag warn">Discrete</li>')
        if positive_only:
            flag_items.append('<li class="flag good">Positive-only</li>')
        if skewed_right:
            flag_items.append('<li class="flag warn">Skewed Right</li>')
        if skewed_left:
            flag_items.append('<li class="flag warn">Skewed Left</li>')
        if heavy_tailed:
            flag_items.append('<li class="flag bad">Heavy-tailed</li>')
        if likely_id:
            flag_items.append('<li class="flag bad">Likely ID</li>')

        # New quality flags
        if miss_pct > 0:
            flag_items.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
        if inf_pct > 0:
            flag_items.append('<li class="flag bad">Has ∞</li>')
        if neg_pct > 0 and not positive_only:
            cls = 'warn' if neg_pct > 10 else ''
            flag_items.append(f'<li class="flag {cls}">Has negatives</li>' if cls else '<li class="flag">Has negatives</li>')
        # Zero-inflation flag
        if zeros_pct >= 50.0:
            flag_items.append('<li class="flag bad">Zero‑inflated</li>')
        elif zeros_pct >= 30.0:
            flag_items.append('<li class="flag warn">Zero‑inflated</li>')
        # Constant / quasi-constant
        if uniq_nonnull == 1:
            flag_items.append('<li class="flag bad">Constant</li>')
        elif unique_ratio <= 0.02 or uniq_nonnull <= 2:
            flag_items.append('<li class="flag warn">Quasi‑constant</li>')
        # Outlier flags
        if out_pct > 1.0:
            flag_items.append('<li class="flag bad">Many outliers</li>')
        elif out_pct > 0.3:
            flag_items.append('<li class="flag warn">Some outliers</li>')
        # Distribution shape hints
        if jb_is_normal:
            flag_items.append('<li class="flag good">≈ Normal (JB)</li>')
        if bimodal:
            flag_items.append('<li class="flag warn">Possibly bimodal</li>')
        if np.isfinite(heap_pct) and heap_pct >= 30.0:
            flag_items.append('<li class="flag">Heaping</li>')
        if positive_only and skewed_right:
            flag_items.append('<li class="flag good">Log‑scale?</li>')
        # Monotonic trends
        try:
            if uniq_nonnull > 1 and s_nonnull.is_monotonic_increasing:
                flag_items.append('<li class="flag good">Monotonic ↑</li>')
            elif uniq_nonnull > 1 and s_nonnull.is_monotonic_decreasing:
                flag_items.append('<li class="flag good">Monotonic ↓</li>')
        except Exception:
            pass

        quality_flags_html = f"<ul class=\"quality-flags\">{''.join(flag_items)}</ul>" if flag_items else ""

        # Optional: Top-5 values for discrete small-cardinality integer columns
        top5_section_html = ""
        if discrete:
            try:
                vc = s_nonnull.value_counts().head(5)
                rows = ''.join(
                    f'<tr><th>{_fmt(k)}</th><td class="num">{v:,} ({(v/cnt*100):.1f}%)</td></tr>'
                    for k, v in vc.items()
                )
                top5_table = f'<table class="kv"><tbody>{rows}</tbody></table>'
                top5_section_html = f'<section class="subtable"><h4 class="small muted">Top 5 values</h4>{top5_table}</section>'
            except Exception:
                top5_section_html = ""

        # Optional: Top correlations chips from precomputed matrix
        corr_section_html = ""
        if corr_matrix is not None and _col in corr_matrix.columns:
            try:
                s_corr = corr_matrix[_col].drop(index=_col).dropna()
                order = s_corr.abs().sort_values(ascending=False)
                chips = []
                for other, val in order.head(2).items():
                    if abs(val) >= 0.6:
                        sign = '+' if s_corr.loc[other] >= 0 else '−'
                        cls = 'good' if abs(val) >= 0.8 else 'warn'
                        chips.append(f'<li class="flag {cls}"><code>{other}</code> {sign}{abs(val):.2f}</li>')
                if chips:
                    corr_section_html = f'<section class="subtable"><h4 class="small muted">Top correlations</h4><ul class="quality-flags">{"".join(chips)}</ul></section>'
            except Exception:
                corr_section_html = ""

        # Precompute linear histogram variants (fixed bin counts)
        hist_lin_10 = _hist_svg_with_axes(s_nonnull, bins=10, auto_bins=False)
        hist_lin_25 = _hist_svg_with_axes(s_nonnull, bins=25, auto_bins=False)
        hist_lin_50 = _hist_svg_with_axes(s_nonnull, bins=50, auto_bins=False)
        # Precompute log-scale variants when applicable (positive-only)
        if positive_only and cnt > 0:
            hist_log_10 = _hist_svg_with_axes(s_nonnull, bins=10, auto_bins=False, scale="log")
            hist_log_25 = _hist_svg_with_axes(s_nonnull, bins=25, auto_bins=False, scale="log")
            hist_log_50 = _hist_svg_with_axes(s_nonnull, bins=50, auto_bins=False, scale="log")
        else:
            hist_log_10 = hist_log_25 = hist_log_50 = ""

        zeros_cls = "warn" if zeros_pct > 30 else ""
        neg_cls = "warn" if 0 < neg_pct <= 10 else ("crit" if neg_pct > 10 else "")
        miss_cls = "crit" if miss_pct > 20 else ("warn" if miss_pct > 5 else "")
        out_cls = "crit" if out_pct > 1 else ("warn" if out_pct > 0.3 else "")
        inf_cls = "crit" if inf_pct > 0 else ""


        # --- Main stats tables (split 6/6, requested order) ---
        main_stats_left = f"""
        <table class="kv"><tbody>
        <tr><th>Count</th><td class="num">{cnt:,}</td></tr>
        <tr><th>Unique</th><td class="num">{uniq:,}</td></tr>
        <tr><th>Missing</th><td class="num {miss_cls}">{miss:,} ({miss_pct:.1f}%)</td></tr>
        <tr><th>Outliers</th><td class="num {out_cls}">{out_n:,} ({out_pct:.1f}%)</td></tr>
        <tr><th>Zeros</th><td class="num {zeros_cls}">{zeros_n:,} ({zeros_pct:.1f}%)</td></tr>
        <tr><th>Memory</th><td class="num">{col_mem_display}</td></tr>
        </tbody></table>
        """

        # This mini table will live inside the chart column (left sub-column)
        main_stats_right = f"""
        <table class="kv"><tbody>
        <tr><th>Min</th><td class="num">{_fmt(mn)}</td></tr>
        <tr><th>Median</th><td class="num">{_fmt(med)}</td></tr>
        <tr><th>Mean</th><td class="num">{_fmt(mean)}</td></tr>
        <tr><th>Max</th><td class="num">{_fmt(mx)}</td></tr>
        <tr><th>Negatives</th><td class="num {neg_cls}">{neg_n:,} ({neg_pct:.1f}%)</td></tr>
        <tr><th>Infinites</th><td class="num {inf_cls}">{inf_n:,} ({inf_pct:.1f}%)</td></tr>
        </tbody></table>
        """

        main_stats_table = f"""
        <div class=\"kv-2col\">
          <div class=\"kv-box left\">{main_stats_left}</div>
          <div class=\"kv-box right\">{main_stats_right}</div>
        </div>
        """

        # --- Quantile Statistics (expandable, percentile grid) ---
        quant_stats_table = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Min</th><td class=\"num\">{_fmt(mn)}</td></tr>
          <tr><th>P1</th><td class=\"num\">{_fmt(p1)}</td></tr>
          <tr><th>P5</th><td class=\"num\">{_fmt(q5)}</td></tr>
          <tr><th>P10</th><td class=\"num\">{_fmt(p10)}</td></tr>
          <tr><th>Q1 (P25)</th><td class=\"num\">{_fmt(q1)}</td></tr>
          <tr><th>Median (P50)</th><td class=\"num\">{_fmt(med)}</td></tr>
          <tr><th>Q3 (P75)</th><td class=\"num\">{_fmt(q3)}</td></tr>
          <tr><th>P90</th><td class=\"num\">{_fmt(p90)}</td></tr>
          <tr><th>P95</th><td class=\"num\">{_fmt(q95)}</td></tr>
          <tr><th>P99</th><td class=\"num\">{_fmt(p99)}</td></tr>
          <tr><th>Max</th><td class=\"num\">{_fmt(mx)}</td></tr>
          <tr><th>Range</th><td class=\"num\">{_fmt(range_val)}</td></tr>
          <tr><th>IQR</th><td class=\"num\">{_fmt(iqr)}</td></tr>
        </tbody></table>
        """

        # --- Descriptive Statistics (expandable, with 95% CI) ---
        desc_stats_table = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Mean</th><td class=\"num\">{_fmt(mean)}</td></tr>
          <tr><th>Std</th><td class=\"num\">{_fmt(std)}</td></tr>
          <tr><th>Variance</th><td class=\"num\">{_fmt(var)}</td></tr>
          <tr><th>SE (mean)</th><td class=\"num\">{_fmt(se)}</td></tr>
          <tr><th>95% CI (mean)</th><td class=\"num\">{ci_str}</td></tr>
          <tr><th>Coeff. Variation</th><td class=\"num\">{_fmt(cv)}</td></tr>
          <tr><th>Geo-Mean</th><td class=\"num\">{_fmt(gmean)}</td></tr>
          <tr><th>MAD</th><td class=\"num\">{_fmt(mad)}</td></tr>
          <tr><th>Skewness</th><td class=\"num\">{_fmt(skew)}</td></tr>
          <tr><th>Kurtosis</th><td class=\"num\">{_fmt(kurt)}</td></tr>
          <tr><th>Jarque–Bera χ² (df=2)</th><td class=\"num\">{_fmt(jb)}</td></tr>
          <tr><th>Heaping at round values</th><td class=\"num\">{('—' if not np.isfinite(heap_pct) else f'{heap_pct:.1f}%')}</td></tr>
          <tr><th>Granularity (decimals)</th><td class=\"num\">{gran_decimals if gran_decimals is not None else '—'}</td></tr>
          <tr><th>Step ≈</th><td class=\"num\">{_fmt(gran_step)}</td></tr>
          <tr><th>Bimodality hint</th><td>{'Yes' if bimodal else 'No'}</td></tr>
        </tbody></table>
        """

        # --- Distribution hint, log-scale histogram, extreme values sections ---
        # Distribution hint (prefer JB if available)
        hints = []
        if jb_is_normal:
            hints = ["≈ normal (JB)"]
        else:
            if np.isfinite(skew):
                if skew >= 1:
                    hints.append("right‑skewed")
                elif skew <= -1:
                    hints.append("left‑skewed")
            if np.isfinite(kurt) and abs(kurt) >= 3:
                hints.append("heavy‑tailed")
            if bimodal:
                hints.append("bimodal?")
        if positive_only:
            hints.append("positive‑only")
        if np.isfinite(heap_pct) and heap_pct >= 30.0:
            hints.append("heaping")
        dist_hint_str = ", ".join(hints) if hints else "—"
        dist_section_html = f'<section class="subtable"><h4 class="small muted">Distribution</h4><p class="muted small">{dist_hint_str}</p></section>'

        # Optional log-scale histogram (only if positive-only)
        log_hist_section = ""
        if positive_only and cnt > 0:
            try:
                log_hist_svg = _hist_svg_with_axes(s_nonnull, scale="log")
                log_hist_section = f'<section class="subtable"><h4 class="small muted">Log‑scale histogram</h4>{log_hist_svg}</section>'
            except Exception:
                log_hist_section = ""

        # Extreme values (top-5)
        extreme_section_html = ""
        try:
            if cnt > 0:
                s_min = s_nonnull.nsmallest(5)
                rows_min = ''.join(f'<tr><th><code>{idx}</code></th><td class="num">{_fmt(val)}</td></tr>' for idx, val in s_min.items())
                min_table = f'<section class="subtable"><h4 class="small muted">Top 5 minima</h4><table class="kv"><tbody>{rows_min}</tbody></table></section>'
                s_max = s_nonnull.nlargest(5)
                rows_max = ''.join(f'<tr><th><code>{idx}</code></th><td class="num">{_fmt(val)}</td></tr>' for idx, val in s_max.items())
                max_table = f'<section class="subtable"><h4 class="small muted">Top 5 maxima</h4><table class="kv"><tbody>{rows_max}</tbody></table></section>'
                outliers_section_html = ""
                if out_n > 0:
                    s_out = s_nonnull[(s_nonnull < lo) | (s_nonnull > hi)]
                    if not s_out.empty:
                        def _score(v):
                            return (lo - v) if v < lo else (v - hi)
                        scores = s_out.apply(_score).sort_values(ascending=False).head(5)
                        rows_out = ''.join(f'<tr><th><code>{idx}</code></th><td class="num">{_fmt(s_nonnull.loc[idx])}</td></tr>' for idx in scores.index)
                        outliers_section_html = f'<section class="subtable"><h4 class="small muted">Top outliers</h4><table class="kv"><tbody>{rows_out}</tbody></table></section>'
                extreme_section_html = min_table + max_table + outliers_section_html
        except Exception:
            extreme_section_html = ""

        # --- Only main stats table in the stats column (minimal) ---
        # (Unused: stats_html = f"""<div class=\"stats\">{main_stats_table}</div>""")

        card_html = f"""
        <article class="var-card" id="{col_id}">

        <header class="var-card__header">
            <div class="title"><span class="colname" title="{_col}">{_col}</span>
            <span class="badge">Numeric</span>
            <span class="dtype chip">{str(s.dtype)}</span>
            {quality_flags_html}
            </div>
        </header>

        <div class="var-card__body">
          <div class="triple-row">
            <div class="box stats-left">{main_stats_left}</div>
            <div class="box stats-right">{main_stats_right}</div>
            <div class="box chart">
                <div class="hist-variants">
                    <!-- Linear scale (default visible: 25 bins) -->
                    <div id="{col_id}-lin-bins-10" class="hist variant" style="display:none">{hist_lin_10}</div>
                    <div id="{col_id}-lin-bins-25" class="hist variant">{hist_lin_25}</div>
                    <div id="{col_id}-lin-bins-50" class="hist variant" style="display:none">{hist_lin_50}</div>
                    <!-- Log scale (only for positive-only cols; hidden by default) -->
                    {"".join([
                        f'<div id="{col_id}-log-bins-10" class="hist variant" style="display:none">{hist_log_10}</div>',
                        f'<div id="{col_id}-log-bins-25" class="hist variant" style="display:none">{hist_log_25}</div>',
                        f'<div id="{col_id}-log-bins-50" class="hist variant" style="display:none">{hist_log_50}</div>',
                    ]) if (positive_only and cnt > 0) else ""}
                </div>
            </div>
          </div>
          <div class="card-controls" role="group" aria-label="Column controls">
            <div class="details-slot">
              <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
            </div>
            <div class="controls-slot">
              <div class="hist-controls" data-col="{col_id}" data-scale="lin" data-bin="25" role="group" aria-label="Histogram controls">
                <div class="center-controls">
                  {'<div class="scale-group"><button type="button" class="btn-soft btn-scale active" data-scale="lin">Linear</button><button type="button" class="btn-soft btn-scale" data-scale="log">Log</button></div>' if (positive_only and cnt > 0) else ''}
                  <div class="bin-group">
                    <button type="button" class="btn-soft btn-bins" data-bin="10">10</button>
                    <button type="button" class="btn-soft btn-bins active" data-bin="25">25</button>
                    <button type="button" class="btn-soft btn-bins" data-bin="50">50</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Full-width Details section with tabs (pestañas) -->
          <section id="{col_id}-details" class="details-section" hidden>
            <nav class="tabs" role="tablist" aria-label="More details">
              <button role="tab" class="active" data-tab="stats">Statistics</button>
              <button role="tab" data-tab="values">Common values</button>
              <button role="tab" data-tab="extremes">Extreme values</button>
              {('<button role="tab" data-tab="corr">Correlations</button>') if corr_section_html else ''}
            </nav>
            <div class="tab-panes">
              <section class="tab-pane active" data-tab="stats">
                <div class="grid-2col">
                  {quant_stats_table}
                  {desc_stats_table}
                </div>
              </section>
              <section class="tab-pane" data-tab="values">
                {top5_section_html or '<p class="muted small">No common values available.</p>'}
              </section>
              <section class="tab-pane" data-tab="extremes">
                {extreme_section_html or '<p class="muted small">No extreme values found.</p>'}
              </section>
              {f'<section class="tab-pane" data-tab="corr">{corr_section_html}</section>' if corr_section_html else ''}
            </div>
          </section>
        </div>
        </article>
        """
        numeric_cards[_col] = card_html

    # Defer building the Variables section until after categorical cards are generated.
    numeric_analysis_section = ""

    # =============================
    # PER-COLUMN ANALYSIS (Categorical) – CARD VIEW
    # =============================
    categorical_cols_list = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cards = {}

    for _col in categorical_cols_list:
        s = df[_col]
        col_id = _safe_col_id(_col)
        n_total = int(s.size)
        miss = int(s.isna().sum())
        miss_pct = (miss / n_total * 100.0) if n_total else 0.0
        vc_all = s.value_counts(dropna=False)
        uniq = int(vc_all.size)
        s_nonnull = s.dropna().astype(str)
        cnt = int(s_nonnull.size)
        uniq_nonnull = int(s_nonnull.nunique()) if cnt > 0 else 0

        if cnt > 0:
            vc = s_nonnull.value_counts()
            mode_val = str(vc.index[0])
            mode_n = int(vc.iloc[0])
            mode_pct = (mode_n / cnt * 100.0)
        else:
            mode_val = "—"; mode_n = 0; mode_pct = 0.0

        probs = (vc / cnt) if cnt > 0 else pd.Series(dtype=float)
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if cnt > 0 else float("nan")

        rare_mask = probs < 0.01 if cnt > 0 else pd.Series(dtype=bool)
        rare_count = int(rare_mask.sum()) if cnt > 0 else 0
        rare_cov = float(probs[rare_mask].sum() * 100.0) if cnt > 0 else 0.0
        top_norm = probs.sort_values(ascending=False) if cnt > 0 else pd.Series(dtype=float)
        top5_cov = float(top_norm.head(5).sum() * 100.0) if cnt > 0 else 0.0

        if cnt > 0:
            lens = s_nonnull.str.len()
            len_min = int(lens.min()) if not lens.empty else 0
            len_mean = float(lens.mean()) if not lens.empty else 0.0
            len_p90 = int(np.quantile(lens, 0.90)) if not lens.empty else 0
            len_max = int(lens.max()) if not lens.empty else 0
            empty_str_n = int((s_nonnull.str.len() == 0).sum())
        else:
            len_min = len_max = len_p90 = 0; len_mean = 0.0; empty_str_n = 0

        case_variants = 0
        trim_variants = 0
        try:
            lower_map = s_nonnull.str.lower()
            case_variants = int((s_nonnull.groupby(lower_map).nunique() > 1).sum())
            strip_map = s_nonnull.str.strip()
            trim_variants = int((s_nonnull.groupby(strip_map).nunique() > 1).sum())
        except Exception:
            pass

        try:
            col_mem_bytes = int(s.memory_usage(index=False, deep=True))
        except TypeError:
            try:
                col_mem_bytes = int(s.memory_usage(deep=True))
            except Exception:
                col_mem_bytes = 0
        except Exception:
            col_mem_bytes = 0
        col_mem_display = _human_bytes(col_mem_bytes)

        flags = []
        unique_ratio = (uniq_nonnull / cnt) if cnt else 0.0
        if uniq_nonnull > max(200, int(0.5 * cnt)):
            flags.append('<li class="flag warn">High cardinality</li>')
        if mode_n >= int(0.7 * cnt) and cnt:
            flags.append('<li class="flag warn">Dominant category</li>')
        if rare_cov >= 30.0:
            flags.append('<li class="flag warn">Many rare levels</li>')
        if case_variants > 0:
            flags.append('<li class="flag">Case variants</li>')
        if trim_variants > 0:
            flags.append('<li class="flag">Trim variants</li>')
        if empty_str_n > 0:
            flags.append('<li class="flag">Empty strings</li>')
        if miss_pct > 0:
            flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
        quality_flags_html = f"<ul class=\"quality-flags\">{''.join(flags)}</ul>" if flags else ""

        # Classes for table highlights (match numeric semantics)
        miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
        rare_cls = 'crit' if rare_cov > 60 else ('warn' if rare_cov >= 30 else '')
        top5_cls = 'good' if top5_cov >= 80 else ('warn' if top5_cov <= 40 else '')
        empty_cls = 'warn' if empty_str_n > 0 else ''

        left_tbl = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Count</th><td class=\"num\">{cnt:,}</td></tr>
          <tr><th>Unique</th><td class=\"num\">{uniq:,}</td></tr>
          <tr><th>Missing</th><td class=\"num {miss_cls}\">{miss:,} ({miss_pct:.1f}%)</td></tr>
          <tr><th>Mode</th><td><code>{mode_val}</code></td></tr>
          <tr><th>Mode %</th><td class=\"num\">{mode_pct:.1f}%</td></tr>
          <tr><th>Memory</th><td class=\"num\">{col_mem_display}</td></tr>
        </tbody></table>
        """

        right_tbl = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Entropy</th><td class=\"num\">{_fmt(entropy)}</td></tr>
          <tr><th>Rare levels</th><td class=\"num {rare_cls}\">{rare_count:,} ({rare_cov:.1f}%)</td></tr>
          <tr><th>Top 5 coverage</th><td class=\"num {top5_cls}\">{top5_cov:.1f}%</td></tr>
          <tr><th>Label length (avg)</th><td class=\"num\">{len_mean:.1f}</td></tr>
          <tr><th>Length p90</th><td class=\"num\">{len_p90}</td></tr>
          <tr><th>Empty strings</th><td class=\"num {empty_cls}\">{empty_str_n:,}</td></tr>
        </tbody></table>
        """

        # Details: levels table (Top 20)
        levels_df = pd.DataFrame({
            "Level": ["(Missing)" if pd.isna(idx) else str(idx) for idx in vc_all.index],
            "Count": vc_all.values,
        })
        total_all = int(vc_all.sum()) if vc_all.size else 1
        levels_df["%"] = (levels_df["Count"] / max(1, total_all) * 100.0).round(1)
        levels_table_html = levels_df.head(20).to_html(index=False, classes='kv', escape=False)

        # Dynamic Top-N choices based on available levels (cap at 15)
        labels_for_levels = ["(Missing)" if pd.isna(idx) else str(idx) for idx in vc_all.index]
        levels_unique = int(pd.Series(labels_for_levels).nunique())
        maxN = max(1, min(15, levels_unique))
        candidates = [5, 10, 15, maxN]
        topn_list = sorted({n for n in candidates if 1 <= n <= maxN})
        default_topn = 10 if 10 in topn_list else (max(topn_list) if topn_list else maxN)

        # Pre-render only generic variants (no Count/% scale toggle)
        variants_html_parts = []
        for n in topn_list:
            svg = _cat_bar_svg(s, top=n, scale="count")
            style = "" if n == default_topn else "display:none"
            variants_html_parts.append(f'<div id="{col_id}-cat-top-{n}" class="cat variant" style="{style}">{svg}</div>')
        variants_html = "".join(variants_html_parts)

        # Build Top-N buttons only for available sizes
        bin_buttons = "".join(
            f'<button type="button" class="btn-soft btn-bins{(" active" if n == default_topn else "" )}" data-topn="{n}">Top {n}</button>'
            for n in topn_list
        )

        card_html = f"""
        <article class=\"var-card\" id=\"{col_id}\">
          <header class=\"var-card__header\"> 
            <div class=\"title\"><span class=\"colname\" title=\"{_col}\">{_col}</span>
            <span class=\"badge\">Categorical</span>
            <span class=\"dtype chip\">{str(s.dtype)}</span>
            {quality_flags_html}
            </div>
          </header>

          <div class=\"var-card__body\">
            <div class=\"triple-row\">
              <div class=\"box stats-left\">{left_tbl}</div>
              <div class=\"box stats-right\">{right_tbl}</div>
              <div class=\"box chart\">
                <div class=\"hist-variants\">{variants_html}</div>
              </div>
            </div>
            <div class="card-controls" role="group" aria-label="Column controls">
              <div class="details-slot">
                <button type="button" class="details-toggle btn-soft" aria-controls="{col_id}-details" aria-expanded="false">Details</button>
              </div>
              <div class="controls-slot">
                <div class="hist-controls" data-col="{col_id}" data-topn="{default_topn}" role="group" aria-label="Categorical controls">
                  <div class="center-controls">
                    <div class="bin-group">{bin_buttons}</div>
                  </div>
                </div>
              </div>
            </div>

            <section id=\"{col_id}-details\" class=\"details-section\" hidden>
              <nav class=\"tabs\" role=\"tablist\" aria-label=\"More details\">
                <button role=\"tab\" class=\"active\" data-tab=\"levels\">Levels</button>
                <button role=\"tab\" data-tab=\"quality\">Quality</button>
              </nav>
              <div class=\"tab-panes\">
                <section class=\"tab-pane active\" data-tab=\"levels\">
                  {levels_table_html}
                </section>
                <section class=\"tab-pane\" data-tab=\"quality\">
                  <table class=\"kv\"><tbody>
                    <tr><th>Case variants (groups)</th><td class=\"num\">{case_variants:,}</td></tr>
                    <tr><th>Trim variants (groups)</th><td class=\"num\">{trim_variants:,}</td></tr>
                    <tr><th>Empty strings</th><td class=\"num\">{empty_str_n:,}</td></tr>
                    <tr><th>Unique ratio</th><td class=\"num\">{unique_ratio:.2f}</td></tr>
                  </tbody></table>
                </section>
              </div>
            </section>
          </div>
        </article>
        """
        categorical_cards[_col] = card_html

    # =============================
    # PER-COLUMN ANALYSIS (Datetime) – CARD VIEW
    # =============================
    datetime_cols_list = df.select_dtypes(include=["datetime", "datetime64[ns]", "datetimetz"]).columns.tolist()

    def _datetime_card(s: pd.Series, col_id: str):
        s_nn = s.dropna()
        cnt = int(s_nn.size)
        miss = int(s.isna().sum())
        n_total = int(s.size)
        miss_pct = (miss / n_total * 100.0) if n_total else 0.0
        miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')
        tz_name = str(getattr(s_nn.dt.tz, 'zone', None)) if cnt and hasattr(s_nn.dt, 'tz') and s_nn.dt.tz is not None else ""
        dt_min = s_nn.min() if cnt else None
        dt_max = s_nn.max() if cnt else None
        span = (dt_max - dt_min) if (cnt and dt_min is not None and dt_max is not None) else None
        delta = s_nn.diff().median() if cnt > 1 else None
        def _fmt_dt(dt):
            if pd.isna(dt) or dt is None:
                return "—"
            return str(dt)

        # Additional diagnostics (sorted diffs, monotonic %, duplicates)
        try:
            t_sorted = s_nn.sort_values()
            diffs_sorted = t_sorted.diff().dropna()
        except Exception:
            diffs_sorted = pd.Series(dtype='timedelta64[ns]')
        try:
            delta_sorted = diffs_sorted.median() if not diffs_sorted.empty else None
        except Exception:
            delta_sorted = None
        try:
            cv_delta = float(diffs_sorted.std() / diffs_sorted.mean()) if not diffs_sorted.empty else None
        except Exception:
            cv_delta = None
        try:
            gaps_n = int((diffs_sorted > (1.5 * delta_sorted)).sum()) if (delta_sorted is not None) else 0
            max_gap = diffs_sorted.max() if not diffs_sorted.empty else None
        except Exception:
            gaps_n = 0
            max_gap = None
        try:
            monotonic_inc = bool(s_nn.is_monotonic_increasing) if cnt > 1 else True
            inc_ratio = float((s_nn.diff().dropna() >= pd.Timedelta(0)).mean()) if cnt > 1 else 1.0
        except Exception:
            monotonic_inc = False
            inc_ratio = 0.0
        try:
            dup_ts = int(s_nn.duplicated().sum())
        except Exception:
            dup_ts = 0
        # Quality flags shown in the header
        dt_flags = []
        if miss_pct > 0:
            dt_flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
        if monotonic_inc:
            dt_flags.append('<li class="flag good">Monotonic ↑</li>')
        quality_flags_html = f"<ul class=\"quality-flags\">{''.join(dt_flags)}</ul>" if dt_flags else ""
        # Unique values (including NaN) and memory footprint
        try:
            uniq = int(s.nunique(dropna=False))
        except Exception:
            uniq = int(s_nn.nunique()) if cnt else 0
        try:
            col_mem_bytes = int(s.memory_usage(index=False, deep=True))
        except TypeError:
            try:
                col_mem_bytes = int(s.memory_usage(deep=True))
            except Exception:
                col_mem_bytes = 0
        except Exception:
            col_mem_bytes = 0
        col_mem_display = _human_bytes(col_mem_bytes)

        def _fmt_td_small(td) -> str:
            try:
                if td is None or (isinstance(td, float) and pd.isna(td)):
                    return "—"
                secs = int(pd.to_timedelta(td).total_seconds())
                days, rem = divmod(secs, 86400)
                hours, rem = divmod(rem, 3600)
                minutes, seconds = divmod(rem, 60)
                parts = []
                if days:
                    parts.append(f"{days}d")
                if hours:
                    parts.append(f"{hours}h")
                if minutes and not days:
                    parts.append(f"{minutes}m")
                if not parts:
                    parts.append(f"{seconds}s")
                return " ".join(parts)
            except Exception:
                return "—"

        def _dt_line_svg(ts: pd.Series,
                         bins: int = 60,
                         width: int = 420,
                         height: int = 160,
                         margin_left: int = 45,
                         margin_right: int = 8,
                         margin_top: int = 8,
                         margin_bottom: int = 32) -> str:
            ts = ts.dropna()
            if ts.empty:
                return f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
            # Pandas 2.x safe conversion from datetime to int64 ns since epoch
            tconv = pd.to_datetime(ts, errors="coerce")
            if hasattr(tconv.dt, "tz") and tconv.dt.tz is not None:
                tconv = tconv.dt.tz_convert("UTC").dt.tz_localize(None)
            tvals = tconv.dropna().astype("int64").to_numpy()
            if tvals.size == 0:
                return f'<svg class="dt-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-label="no data"></svg>'
            tmin = int(np.min(tvals))
            tmax = int(np.max(tvals))
            if tmin == tmax:
                tmax = tmin + 1
            bins = int(max(10, min(bins, max(10, min(ts.size, 180)))))
            counts, edges = np.histogram(tvals, bins=bins, range=(tmin, tmax))
            y_max = int(max(1, counts.max()))

            iw = width - margin_left - margin_right
            ih = height - margin_top - margin_bottom
            def sx(x):
                return margin_left + (x - tmin) / (tmax - tmin) * iw
            def sy(y):
                return margin_top + (1 - y / y_max) * ih

            centers = (edges[:-1] + edges[1:]) / 2.0
            pts = " ".join(f"{sx(x):.2f},{sy(float(c)):.2f}" for x, c in zip(centers, counts))

            # Nice ticks helpers (y-axis)
            def _nice_num(rng, do_round=True):
                import math
                if rng <= 0 or not np.isfinite(rng):
                    return 1.0
                exp = math.floor(math.log10(rng))
                frac = rng / (10 ** exp)
                nice = 1 if (frac < 1.5 if do_round else frac <= 1) else (2 if (frac < 3 if do_round else frac <= 2) else (5 if (frac < 7 if do_round else frac <= 5) else 10))
                return nice * (10 ** exp)
            def _nice_ticks(vmin, vmax, n=5):
                import math
                if vmax < vmin:
                    vmin, vmax = vmax, vmin
                if vmax == vmin:
                    vmax = vmin + 1
                rng = _nice_num(vmax - vmin, do_round=False)
                step = _nice_num(rng / max(1, n - 1), do_round=True)
                nice_min = math.floor(vmin / step) * step
                nice_max = math.ceil(vmax / step) * step
                ticks = []
                t = nice_min
                while t <= nice_max + step * 1e-9 and len(ticks) < 50:
                    ticks.append(t)
                    t += step
                return ticks, step

            y_ticks, y_step = _nice_ticks(0, y_max, 5)

            # X ticks: spread ~5 labels across span
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
            # grid lines (y)
            for yt in y_ticks:
                parts.append(f'<line class="grid" x1="{margin_left}" y1="{sy(yt):.2f}" x2="{margin_left + iw}" y2="{sy(yt):.2f}"></line>')
            # polyline
            parts.append(f'<polyline class="line" points="{pts}"></polyline>')

            # hover hotspots per bin (native title tooltips)
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

            # axes
            x_axis_y = margin_top + ih
            parts.append(f'<line class="axis" x1="{margin_left}" y1="{x_axis_y}" x2="{margin_left+iw}" y2="{x_axis_y}"></line>')
            parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}"></line>')

            # y ticks & labels
            for yt in y_ticks:
                py = sy(yt)
                parts.append(f'<line class="tick" x1="{margin_left - 4}" y1="{py:.2f}" x2="{margin_left}" y2="{py:.2f}"></line>')
                lab = int(round(yt))
                parts.append(f'<text class="tick-label" x="{margin_left - 6}" y="{py + 3:.2f}" text-anchor="end">{lab}</text>')

            # x ticks & labels
            for xv in xt_vals:
                px = sx(xv)
                parts.append(f'<line class="tick" x1="{px:.2f}" y1="{x_axis_y}" x2="{px:.2f}" y2="{x_axis_y + 4}"></line>')
                parts.append(f'<text class="tick-label" x="{px:.2f}" y="{x_axis_y + 14}" text-anchor="middle">{_fmt_xt(xv)}</text>')

            # axis titles
            parts.append(f'<text class="axis-title x" x="{margin_left + iw/2:.2f}" y="{x_axis_y + 28}" text-anchor="middle">Time</text>')
            parts.append(f'<text class="axis-title y" transform="translate({margin_left - 36},{margin_top + ih/2:.2f}) rotate(-90)" text-anchor="middle">Count</text>')

            parts.append('</svg>')
            return ''.join(parts)

        dt_line_svg = _dt_line_svg(s_nn)

        # Summaries for lightweight charts (to be drawn by JS later)
        try:
            hour_counts = (s_nn.dt.hour.value_counts().reindex(range(24), fill_value=0).astype(int).tolist()) if cnt else [0]*24
        except Exception:
            hour_counts = [0]*24
        try:
            dow_counts = (s_nn.dt.dayofweek.value_counts().reindex(range(7), fill_value=0).astype(int).tolist()) if cnt else [0]*7
        except Exception:
            dow_counts = [0]*7
        try:
            month_counts = (s_nn.dt.month.value_counts().reindex(range(1,13), fill_value=0).astype(int).tolist()) if cnt else [0]*12
        except Exception:
            month_counts = [0]*12
        # Year distribution (dynamic labels)
        try:
            years = s_nn.dt.year
            if years.empty:
                year_labels, year_counts = [], []
            else:
                y_min, y_max = int(years.min()), int(years.max())
                year_labels = list(range(y_min, y_max + 1))
                vc_year = years.value_counts().sort_index()
                year_counts = [int(vc_year.get(y, 0)) for y in year_labels]
        except Exception:
            year_labels, year_counts = [], []

        # Hidden JSON payload for JS renderers
        try:
            span_seconds = int(span.total_seconds()) if span is not None else None
        except Exception:
            span_seconds = None
        try:
            delta_seconds = int(delta.total_seconds()) if isinstance(delta, pd.Timedelta) else None
        except Exception:
            delta_seconds = None

        meta = {
            "min": _fmt_dt(dt_min),
            "max": _fmt_dt(dt_max),
            "span_seconds": span_seconds,
            "delta_median_seconds": delta_seconds,
            "tz": tz_name,
            "n": cnt,
            "missing": miss,
            "counts": {
                "hour": hour_counts,
                "dow": dow_counts,
                "month": month_counts,
                "year": {"labels": year_labels, "values": year_counts}
            }
        }
        meta_json = json.dumps(meta)
        meta_blob = f'<script type="application/json" id="{col_id}-dt-meta">{meta_json}</script>'

        # Span breakdown in multiple units
        try:
            span_seconds = int(span.total_seconds()) if span is not None else None
        except Exception:
            span_seconds = None
        span_minutes = (span_seconds / 60.0) if span_seconds is not None else None
        span_hours = (span_seconds / 3600.0) if span_seconds is not None else None
        span_days = (span_seconds / 86400.0) if span_seconds is not None else None
        span_months = (span_days / 30.4375) if span_days is not None else None
        span_years = (span_days / 365.25) if span_days is not None else None

        # --- FULL statistics tables (for Details > Statistics) ---
        full_left_tbl_html = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Count</th><td class=\"num\">{cnt:,}</td></tr>
          <tr><th>Missing</th><td class=\"num {miss_cls}\">{miss:,} ({miss_pct:.1f}%)</td></tr>
          <tr><th>Unique</th><td class=\"num\">{uniq:,}</td></tr>
          <tr><th>Duplicate timestamps</th><td class=\"num\">{dup_ts:,}</td></tr>
          <tr><th>Timezone</th><td><code>{tz_name or '—'}</code></td></tr>
          <tr><th>Memory</th><td class=\"num\">{col_mem_display}</td></tr>
          <tr><th>Monotonic</th><td>{'Yes' if monotonic_inc else 'No'}</td></tr>
          <tr><th>Non-decreasing %</th><td class=\"num\">{inc_ratio*100:.1f}%</td></tr>
          <tr><th>Cadence CV (Δ)</th><td class=\"num\">{('—' if cv_delta is None else f'{cv_delta:.2f}')}</td></tr>
          <tr><th>Large gaps (&gt;1.5×Δ)</th><td class=\"num\">{gaps_n:,}{f' (max { _fmt_td_small(max_gap) })' if gaps_n else ''}</td></tr>
        </tbody></table>
        """

        full_right_tbl_html = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Min</th><td>{_fmt_dt(dt_min)}</td></tr>
          <tr><th>Max</th><td>{_fmt_dt(dt_max)}</td></tr>
          <tr><th>Span (s)</th><td class=\"num\">{span_seconds if span_seconds is not None else '—'}</td></tr>
          <tr><th>Span (min)</th><td class=\"num\">{f'{span_minutes:.1f}' if span_minutes is not None else '—'}</td></tr>
          <tr><th>Span (h)</th><td class=\"num\">{f'{span_hours:.1f}' if span_hours is not None else '—'}</td></tr>
          <tr><th>Span (d)</th><td class=\"num\">{f'{span_days:.1f}' if span_days is not None else '—'}</td></tr>
          <tr><th>Span (mo)</th><td class=\"num\">{f'{span_months:.2f}' if span_months is not None else '—'}</td></tr>
          <tr><th>Span (yr)</th><td class=\"num\">{f'{span_years:.2f}' if span_years is not None else '—'}</td></tr>
          <tr><th>Median Δ (s)</th><td class=\"num\">{delta_seconds if delta_seconds is not None else '—'}</td></tr>
          <tr><th>Weekend share</th><td class=\"num\">{(s_nn.dt.dayofweek>=5).mean()*100:.1f}%</td></tr>
        </tbody></table>
        """

        # --- MAIN compact tables (for the card’s triple-row) ---
        main_left_tbl_html = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Count</th><td class=\"num\">{cnt:,}</td></tr>
          <tr><th>Missing</th><td class=\"num {miss_cls}\">{miss:,} ({miss_pct:.1f}%)</td></tr>
          <tr><th>Unique</th><td class=\"num\">{uniq:,}</td></tr>
          <tr><th>Memory</th><td class=\"num\">{col_mem_display}</td></tr>
        </tbody></table>
        """

        main_right_tbl_html = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Min</th><td>{_fmt_dt(dt_min)}</td></tr>
          <tr><th>Max</th><td>{_fmt_dt(dt_max)}</td></tr>
          <tr><th>Span (d)</th><td class=\"num\">{f'{span_days:.1f}' if span_days is not None else '—'}</td></tr>
          <tr><th>Median Δ (s)</th><td class=\"num\">{delta_seconds if delta_seconds is not None else '—'}</td></tr>
          <tr><th>Monotonic</th><td>{'Yes' if monotonic_inc else 'No'}</td></tr>
        </tbody></table>
        """

        suggestions = []
        if miss > 0:
            suggestions.append("Missing timestamps present → consider imputing or filtering before resampling.")
        if not tz_name:
            suggestions.append("Timezone‑naive → localize to your timezone or convert to UTC.")
        if monotonic_inc:
            suggestions.append("Timestamps are monotonic increasing → safe to use as an ordering key.")
        elif inc_ratio < 0.9:
            suggestions.append(f"Unsorted timestamps ({inc_ratio*100:.1f}% non‑decreasing) → sort by time before time‑based ops.")
        if cv_delta is not None and cv_delta > 1:
            suggestions.append("Irregular/bursty cadence (CV of Δ > 1) → consider resampling to a regular grid.")
        if gaps_n > 0:
            suggestions.append(f"Detected {gaps_n} large gaps (max {_fmt_td_small(max_gap)}) → investigate outages or missing periods.")
        wknd_share = (s_nn.dt.dayofweek>=5).mean()*100 if cnt else 0.0
        if wknd_share < 5:
            suggestions.append("Primarily weekday activity → align analyses to business days.")
        elif wknd_share > 40:
            suggestions.append("Strong weekend activity → consider weekly seasonality.")
        biz_share = (((s_nn.dt.hour>=9)&(s_nn.dt.hour<18)).mean()*100) if cnt else 0.0
        if biz_share > 80:
            suggestions.append("Mostly business‑hours events → consider local working‑hour calendars.")
        suggestions_html = "<ul class=\"suggestions\">" + "".join(f"<li>{s}</li>" for s in suggestions) + "</ul>" if suggestions else "<p class=\"muted small\">No specific suggestions.</p>"

        chart_placeholder = (
            f'<div class="box chart">'
            f'<div id="{col_id}-dt-chart" class="dt-chart">{dt_line_svg}</div>'
            f'</div>'
        )

        card_html = f"""
        <article class=\"var-card\" id=\"{col_id}\">
          <header class=\"var-card__header\">
            <div class=\"title\"><span class=\"colname\" title=\"{s.name}\">{s.name}</span>
            <span class=\"badge\">Datetime</span>
            <span class=\"dtype chip\">{str(s.dtype)}</span>
            {quality_flags_html}
            </div>
          </header>
          <div class=\"var-card__body\">
            <div class=\"triple-row\">
              <div class=\"box stats-left\">{main_left_tbl_html}</div>
              <div class=\"box stats-right\">{main_right_tbl_html}</div>
              {chart_placeholder}
            </div>
            <div class=\"card-controls\" role=\"group\" aria-label=\"Column controls\">
              <div class=\"details-slot\">
                <button type=\"button\" class=\"details-toggle btn-soft\" aria-controls=\"{col_id}-details\" aria-expanded=\"false\">Details</button>
              </div>
              <div class=\"controls-slot\"></div>
            </div>
            <section id=\"{col_id}-details\" class=\"details-section\" hidden>
              <nav class=\"tabs\" role=\"tablist\" aria-label=\"More details\">
                <button role=\"tab\" class=\"active\" data-tab=\"stats\">Statistics</button>
                <button role=\"tab\" data-tab=\"dist\">Distributions</button>
                <button role=\"tab\" data-tab=\"suggestions\">Suggestions</button>
              </nav>
              <div class=\"tab-panes\">
                <section class=\"tab-pane active\" data-tab=\"stats\">
                  <div class=\"grid-2col\">{full_left_tbl_html}{full_right_tbl_html}</div>
                </section>
                <section class=\"tab-pane\" data-tab=\"dist\">
                  <div class=\"grid-2col\">
                    <div class=\"box\"><h4 class=\"small muted\">Hour of day</h4><div id=\"{col_id}-dt-hour\" class=\"dt-chart dt-hour\"></div></div>
                    <div class=\"box\"><h4 class=\"small muted\">Day of week</h4><div id=\"{col_id}-dt-dow\" class=\"dt-chart dt-dow\"></div></div>
                    <div class=\"box\"><h4 class=\"small muted\">Month</h4><div id=\"{col_id}-dt-month\" class=\"dt-chart dt-month\"></div></div>
                    <div class=\"box\"><h4 class=\"small muted\">Year</h4><div id=\"{col_id}-dt-year\" class=\"dt-chart dt-year\"></div></div>
                  </div>
                </section>
                <section class=\"tab-pane\" data-tab=\"suggestions\">
                  {suggestions_html}
                </section>
              </div>
            </section>
            {meta_blob}
          </div>
        </article>
        """
        return card_html

    datetime_cards = {}
    for _col in datetime_cols_list:
        s = df[_col]
        col_id = _safe_col_id(_col)
        datetime_cards[_col] = _datetime_card(s, col_id)

    # =============================
    # PER-COLUMN ANALYSIS (Boolean) – CARD VIEW (minimal)
    # =============================
    boolean_cols_list = df.select_dtypes(include=["bool"]).columns.tolist()

    def _bool_stack_svg(true_n: int, false_n: int, miss: int,
                        width: int = 420, height: int = 48, margin: int = 4) -> str:
        total = max(1, int(true_n + false_n + miss))
        inner_w = width - 2 * margin
        seg_h = height - 2 * margin
        # segment widths
        fw = (false_n / total) * inner_w
        tw = (true_n  / total) * inner_w
        mw = (miss    / total) * inner_w

        def fmt_label(n: int) -> str:
            pct = (n / total * 100.0) if total else 0.0
            return f"{n:,} ({pct:.1f}%)", f"{pct:.1f}%"

        parts = [
            f'<svg class="bool-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Boolean distribution">'
        ]

        x = float(margin)
        # Helper to add a segment + label
        def seg(css: str, n: int, w: float, label_text: str):
            nonlocal x
            if w <= 0 or n <= 0:
                return
            parts.append(
                f'<rect class="seg {css}" x="{x:.2f}" y="{margin}" width="{w:.2f}" height="{seg_h:.2f}" rx="2" ry="2" '
                f'data-count="{n}" data-pct="{(n/total*100.0):.1f}" data-label="{css}">'
                f'<title>{label_text}</title>'
                f'</rect>'
            )
            # Decide label verbosity by width
            full, pct_only = fmt_label(n)
            if w >= 80:
                label = full
            elif w >= 28:
                label = pct_only
            else:
                label = ''
            if label:
                cx = x + w / 2.0
                cy = margin + seg_h / 2.0 + 4  # vertically centered (approx baseline)
                parts.append(
                    f'<text class="seg-label" x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" '
                    f'style="fill:#fff;font-size:11px;font-weight:600;">{label}</text>'
                )
            x += w

        # Build segments in False | True | Missing order
        seg('false', false_n, fw, f'{false_n:,} False ({false_n/total*100:.1f}%)')
        seg('true',  true_n,  tw, f'{true_n:,} True ({true_n/total*100:.1f}%)')
        seg('missing', miss,   mw, f'{miss:,} Missing ({miss/total*100:.1f}%)')

        parts.append('</svg>')
        return ''.join(parts)

    def _boolean_card(s: pd.Series, col_id: str) -> str:
        n_total = int(s.size)
        # pandas bool dtype does not carry NaN; keep logic generic anyway
        miss = int(s.isna().sum()) if n_total else 0
        miss_pct = (miss / n_total * 100.0) if n_total else 0.0
        s_nn = s.dropna()
        cnt = int(s_nn.size)
        # Prevalence (among non-missing)
        try:
            true_n = int((s_nn == True).sum())
        except Exception:
            true_n = int(pd.Series(s_nn).astype(bool).sum()) if cnt else 0
        false_n = int(max(0, cnt - true_n))
        # Percentages over TOTAL to match the stacked bar semantics
        true_pct_total = (true_n / n_total * 100.0) if n_total else 0.0
        false_pct_total = (false_n / n_total * 100.0) if n_total else 0.0
        uniq = int(s.nunique(dropna=False)) if n_total else 0
        try:
            col_mem_bytes = int(s.memory_usage(index=False, deep=True))
        except TypeError:
            try:
                col_mem_bytes = int(s.memory_usage(deep=True))
            except Exception:
                col_mem_bytes = 0
        except Exception:
            col_mem_bytes = 0
        col_mem_display = _human_bytes(col_mem_bytes)

        # Flags
        flags = []
        if miss_pct > 0:
            flags.append(f'<li class="flag {"bad" if miss_pct > 20 else "warn"}">Missing</li>')
        if cnt > 0 and (true_n == 0 or false_n == 0):
            flags.append('<li class="flag bad">Constant</li>')
        if cnt > 0:
            p = true_n / cnt
            if p <= 0.05 or p >= 0.95:
                flags.append('<li class="flag warn">Imbalanced</li>')
        quality_flags_html = f"<ul class=\"quality-flags\">{''.join(flags)}</ul>" if flags else ""

        miss_cls = 'crit' if miss_pct > 20 else ('warn' if miss_pct > 0 else '')

        left_tbl = f"""
        <table class=\"kv\"><tbody>
          <tr><th>Count</th><td class=\"num\">{cnt:,}</td></tr>
          <tr><th>Missing</th><td class=\"num {miss_cls}\">{miss:,} ({miss_pct:.1f}%)</td></tr>
          <tr><th>Unique</th><td class=\"num\">{uniq:,}</td></tr>
          <tr><th>Memory</th><td class=\"num\">{col_mem_display}</td></tr>
        </tbody></table>
        """

        right_tbl = f"""
        <table class=\"kv\"><tbody>
          <tr><th>True</th><td class=\"num\">{true_n:,} ({true_pct_total:.1f}%)</td></tr>
          <tr><th>False</th><td class=\"num\">{false_n:,} ({false_pct_total:.1f}%)</td></tr>
        </tbody></table>
        """

        chart_html = _bool_stack_svg(true_n, false_n, miss)

        # Basic suggestions
        suggestions = []
        if cnt > 0 and (true_n == 0 or false_n == 0):
            suggestions.append("Constant boolean → consider dropping (no variance).")
        if miss_pct > 0:
            suggestions.append("Missing values present → consider explicit missing indicator.")
        if cnt > 0:
            p = true_n / cnt
            if p <= 0.05 or p >= 0.95:
                suggestions.append("Severely imbalanced → consider class weighting or focal loss if used as target.")
        suggestions_html = "<ul class=\"suggestions\">" + "".join(f"<li>{s}</li>" for s in suggestions) + "</ul>" if suggestions else "<p class=\"muted small\">No specific suggestions.</p>"

        card_html = f"""
        <article class=\"var-card\" id=\"{col_id}\">
          <header class=\"var-card__header\">
            <div class=\"title\"><span class=\"colname\" title=\"{s.name}\">{s.name}</span>
            <span class=\"badge\">Boolean</span>
            <span class=\"dtype chip\">{str(s.dtype)}</span>
            {quality_flags_html}
            </div>
          </header>
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
            <section id=\"{col_id}-details\" class=\"details-section\" hidden>
              <nav class=\"tabs\" role=\"tablist\" aria-label=\"More details\">
                <button role=\"tab\" class=\"active\" data-tab=\"stats\">Statistics</button>
                <button role=\"tab\" data-tab=\"suggestions\">Suggestions</button>
              </nav>
              <div class=\"tab-panes\">
                <section class=\"tab-pane active\" data-tab=\"stats\">
                  <div class=\"grid-2col\">{left_tbl}{right_tbl}</div>
                </section>
                <section class=\"tab-pane\" data-tab=\"suggestions\">{suggestions_html}</section>
              </div>
            </section>
          </div>
        </article>
        """
        return card_html

    boolean_cards = {}
    for _col in boolean_cols_list:
        s = df[_col]
        col_id = _safe_col_id(_col)
        boolean_cards[_col] = _boolean_card(s, col_id)

    # Build a single "Variables" section (original column order)
    all_cards_list = []
    for __col in df.columns:
        if __col in numeric_cards:
            all_cards_list.append(numeric_cards[__col])
        elif __col in categorical_cards:
            all_cards_list.append(categorical_cards[__col])
        elif __col in boolean_cards:
            all_cards_list.append(boolean_cards[__col])
        elif __col in datetime_cards:
            all_cards_list.append(datetime_cards[__col])
    if all_cards_list:
        variables_section_html = f"""
        <section id=\"vars\">  
          <span id=\"numeric-vars\" class=\"anchor-alias\"></span>
          <h2 class=\"section-title\">Variables</h2>
          <p class=\"muted small\">Analyzing {len(numeric_cols_list)+len(categorical_cols_list)+len(datetime_cols_list)+len(boolean_cols_list)} variables ({len(numeric_cols_list)} numeric, {len(categorical_cols_list)} categorical, {len(datetime_cols_list)} datetime, {len(boolean_cols_list)} boolean).</p>
          <div class=\"cards-grid\">{''.join(all_cards_list)}</div>
        </section>
        """
    else:
        variables_section_html = ""

    # Combine sections so the template needs only the existing placeholder
    sections_html = (dataset_sample_section or "") + (variables_section_html or "")

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
    script_path = os.path.join(static_dir, "js", "functionality.js")
    script_content = load_script(script_path)

    # Load and embed PNG logos (light/dark) and auto-switch via CSS
    logo_light_path = os.path.join(static_dir, "images", "logo_suricata_transparent.png")
    logo_dark_path  = os.path.join(static_dir, "images", "logo_suricata_transparent_dark_mode.png")

    logo_light_img = embed_image(logo_light_path, element_id="logo-light", alt_text="Logo", mime_type="image/png")
    logo_dark_img  = embed_image(logo_dark_path,  element_id="logo-dark",  alt_text="Logo (dark)", mime_type="image/png")

    logo_html = f'<span id="logo">{logo_light_img}{logo_dark_img}</span>'

    # Load and embed the favicon.
    favicon_path = os.path.join(static_dir, "images", "favicon.ico")
    favicon_tag = embed_favicon(favicon_path)

    # Compute how long it took to generate the report.
    end_time = time.time()
    duration_seconds = end_time - start_time

    # Set defaults for new information.
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pysuricata_version = _resolve_pysuricata_version()
    repo_url = "https://github.com/alvarodiez20/pysuricata"

    # Replace placeholders in the template.
    html = template.format(
      favicon=favicon_tag,
      css=css_tag,
      script=script_content,
      logo=logo_html,
      report_title=report_title,
      report_date=report_date,
      pysuricata_version=pysuricata_version,
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
      dataset_sample_section=sections_html,
  )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    return html
