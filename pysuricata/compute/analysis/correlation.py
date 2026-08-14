from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class StreamingCorr:
    """Lightweight streaming correlation estimator for numeric columns.

    Maintains pairwise co-moments (Welford/Chan) rather than raw power sums, so
    Pearson correlation can be computed at the end without holding the data.

    Raw power sums are the obvious encoding and the wrong one: recovering the
    variance as ``sx2 - sx*sx/n`` subtracts two nearly equal large numbers. For a
    column of timestamps-as-int, IDs, or prices near 1e6, those two terms agree
    to within the last bits of a float64 and the variance collapses to zero or
    below -- reporting "no correlation" for perfectly correlated data. Centring
    each batch on its own mean keeps every quantity small and well-conditioned.
    """

    def __init__(self, columns: Sequence[str]):
        self.cols = list(columns)
        self.pairs: dict[tuple[str, str], dict[str, float]] = {}

    def _accumulate_pair(
        self, key: tuple[str, str], x: np.ndarray, y: np.ndarray
    ) -> None:
        """Fold one batch of paired observations into the running co-moments."""
        n_b = float(x.size)
        if n_b <= 0:
            return

        mean_x_b = float(np.mean(x))
        mean_y_b = float(np.mean(y))
        dx_b = x - mean_x_b
        dy_b = y - mean_y_b
        m2x_b = float(np.dot(dx_b, dx_b))
        m2y_b = float(np.dot(dy_b, dy_b))
        cxy_b = float(np.dot(dx_b, dy_b))

        st = self.pairs.get(key)
        if st is None:
            self.pairs[key] = {
                "n": n_b,
                "mean_x": mean_x_b,
                "mean_y": mean_y_b,
                "m2x": m2x_b,
                "m2y": m2y_b,
                "cxy": cxy_b,
            }
            return

        # Chan's pairwise merge, applied to both variances and the covariance.
        n_a = st["n"]
        n = n_a + n_b
        delta_x = mean_x_b - st["mean_x"]
        delta_y = mean_y_b - st["mean_y"]
        scale = n_a * n_b / n

        st["m2x"] += m2x_b + delta_x * delta_x * scale
        st["m2y"] += m2y_b + delta_y * delta_y * scale
        st["cxy"] += cxy_b + delta_x * delta_y * scale
        st["mean_x"] += delta_x * n_b / n
        st["mean_y"] += delta_y * n_b / n
        st["n"] = n

    def update_from_pandas(self, df: pd.DataFrame) -> None:  # type: ignore[name-defined]  # noqa: F821
        try:
            import pandas as pd  # type: ignore
            from pandas.api import types as pdt  # type: ignore
        except Exception:
            return
        use_cols = [c for c in self.cols if c in df.columns]
        if len(use_cols) < 2:
            return
        arrs: dict[str, np.ndarray] = {}
        for c in use_cols:
            try:
                # Fast path: skip pd.to_numeric for already-numeric columns
                dt = df[c].dtype
                if pdt.is_numeric_dtype(dt) and not pdt.is_bool_dtype(dt):
                    a = df[c].to_numpy(dtype="float64", copy=False)
                else:
                    a = pd.to_numeric(df[c], errors="coerce").to_numpy(
                        dtype="float64", copy=False
                    )
            except Exception:
                a = np.asarray(df[c].to_numpy(), dtype=float)
            arrs[c] = a
        # Pre-compute finite masks once per column to avoid redundant recomputation
        finite_masks: dict[str, np.ndarray] = {
            c: np.isfinite(arrs[c]) for c in use_cols
        }
        for i in range(len(use_cols)):
            ci = use_cols[i]
            xi = arrs[ci]
            fi = finite_masks[ci]
            for j in range(i + 1, len(use_cols)):
                cj = use_cols[j]
                yj = arrs[cj]
                m = fi & finite_masks[cj]
                if not m.any():
                    continue
                self._accumulate_pair((ci, cj), xi[m], yj[m])

    def update_from_polars(self, df: pl.DataFrame) -> None:  # type: ignore[name-defined]  # noqa: F821
        try:
            import polars as pl  # type: ignore
        except Exception:
            return
        use_cols = [c for c in self.cols if c in df.columns]
        if len(use_cols) < 2:
            return
        arrs: dict[str, np.ndarray] = {}
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
        # Pre-compute finite masks once per column to avoid redundant recomputation
        finite_masks_pl: dict[str, np.ndarray] = {
            c: np.isfinite(arrs[c]) for c in use_cols
        }
        for i in range(len(use_cols)):
            ci = use_cols[i]
            xi = arrs[ci]
            fi = finite_masks_pl[ci]
            for j in range(i + 1, len(use_cols)):
                cj = use_cols[j]
                yj = arrs[cj]
                m = fi & finite_masks_pl[cj]
                if not m.any():
                    continue
                self._accumulate_pair((ci, cj), xi[m], yj[m])

    def top_map(self, *, threshold: float = 0.5, max_per_col: int = 10):
        def corr_from(st):
            if st["n"] < 2:
                return 0.0
            denom = (st["m2x"] * st["m2y"]) ** 0.5
            if denom <= 0:
                # A genuinely constant column, not a cancellation artefact:
                # correlation is undefined, and 0.0 is the honest report.
                return 0.0
            return float(max(-1.0, min(1.0, st["cxy"] / denom)))

        col_map: dict[str, list[tuple[str, float]]] = {c: [] for c in self.cols}
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
