from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


class StreamingCorr:
    """Lightweight streaming correlation estimator for numeric columns.

    Maintains pairwise running sums sufficient to compute Pearson correlation
    at the end without holding full data in memory.
    """

    def __init__(self, columns: Sequence[str]):
        self.cols = list(columns)
        self.pairs: Dict[Tuple[str, str], Dict[str, float]] = {}

    def update_from_pandas(self, df: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return
        use_cols = [c for c in self.cols if c in df.columns]
        if len(use_cols) < 2:
            return
        arrs: Dict[str, np.ndarray] = {}
        for c in use_cols:
            try:
                a = pd.to_numeric(df[c], errors="coerce").to_numpy(
                    dtype="float64", copy=False
                )
            except Exception:
                a = np.asarray(df[c].to_numpy(), dtype=float)
            arrs[c] = a
        for i in range(len(use_cols)):
            ci = use_cols[i]
            xi = arrs[ci]
            for j in range(i + 1, len(use_cols)):
                cj = use_cols[j]
                yj = arrs[cj]
                m = np.isfinite(xi) & np.isfinite(yj)
                if not m.any():
                    continue
                x = xi[m]
                y = yj[m]
                n = float(x.size)
                sx = float(np.sum(x))
                sy = float(np.sum(y))
                sx2 = float(np.sum(x * x))
                sy2 = float(np.sum(y * y))
                sxy = float(np.sum(x * y))
                key = (ci, cj)
                if key not in self.pairs:
                    self.pairs[key] = {
                        "n": 0.0,
                        "sx": 0.0,
                        "sy": 0.0,
                        "sx2": 0.0,
                        "sy2": 0.0,
                        "sxy": 0.0,
                    }
                st = self.pairs[key]
                st["n"] += n
                st["sx"] += sx
                st["sy"] += sy
                st["sx2"] += sx2
                st["sy2"] += sy2
                st["sxy"] += sxy

    def update_from_polars(self, df: "pl.DataFrame") -> None:  # type: ignore[name-defined]
        try:
            import polars as pl  # type: ignore
        except Exception:
            return
        use_cols = [c for c in self.cols if c in df.columns]
        if len(use_cols) < 2:
            return
        arrs: Dict[str, np.ndarray] = {}
        for c in use_cols:
            try:
                # Optimized correlation processing - add fast path for numeric types
                if df[c].dtype in [
                    pl.Float64,
                    pl.Float32,
                    pl.Int64,
                    pl.Int32,
                    pl.UInt64,
                    pl.UInt32,
                ]:
                    a = df[c].to_numpy()
                else:
                    a = df[c].cast(pl.Float64, strict=False).to_numpy()
            except Exception:
                a = np.asarray(df[c].to_list(), dtype=float)
            arrs[c] = a
        for i in range(len(use_cols)):
            ci = use_cols[i]
            xi = arrs[ci]
            for j in range(i + 1, len(use_cols)):
                cj = use_cols[j]
                yj = arrs[cj]
                m = np.isfinite(xi) & np.isfinite(yj)
                if not m.any():
                    continue
                x = xi[m]
                y = yj[m]
                n = float(x.size)
                sx = float(np.sum(x))
                sy = float(np.sum(y))
                sx2 = float(np.sum(x * x))
                sy2 = float(np.sum(y * y))
                sxy = float(np.sum(x * y))
                key = (ci, cj)
                if key not in self.pairs:
                    self.pairs[key] = {
                        "n": 0.0,
                        "sx": 0.0,
                        "sy": 0.0,
                        "sx2": 0.0,
                        "sy2": 0.0,
                        "sxy": 0.0,
                    }
                st = self.pairs[key]
                st["n"] += n
                st["sx"] += sx
                st["sy"] += sy
                st["sx2"] += sx2
                st["sy2"] += sy2
                st["sxy"] += sxy

    def top_map(self, *, threshold: float = 0.5, max_per_col: int = 10):
        def corr_from(st):
            n = st["n"]
            sx, sy = st["sx"], st["sy"]
            sx2, sy2, sxy = st["sx2"], st["sy2"], st["sxy"]
            vx = max(0.0, sx2 - sx * sx / n)
            vy = max(0.0, sy2 - sy * sy / n)
            cov = sxy - sx * sy / n
            denom = (vx * vy) ** 0.5
            if denom <= 0:
                return 0.0
            return float(cov / denom)

        col_map: Dict[str, list[tuple[str, float]]] = {c: [] for c in self.cols}
        for (a, b), st in self.pairs.items():
            r = corr_from(st)
            if abs(r) < float(threshold):
                continue
            col_map[a].append((b, r))
            col_map[b].append((a, r))
        for c in list(col_map.keys()):
            col_map[c].sort(key=lambda t: abs(t[1]), reverse=True)
            col_map[c] = col_map[c][: int(max_per_col)]
        return col_map
