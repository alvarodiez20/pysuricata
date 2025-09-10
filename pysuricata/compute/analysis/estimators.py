from __future__ import annotations

from typing import Tuple

from ...accumulators.sketches import KMV


class RowKMV:
    """Approximate row-duplicate estimator using a KMV distinct sketch.

    Maintains an approximate count of distinct rows by hashing each row into a
    64-bit signature and feeding it to a KMV (K-Minimum Values) sketch.
    """

    def __init__(self, k: int = 8192) -> None:
        self.kmv = KMV(k)
        self.rows = 0

    def update_from_pandas(self, df: "pd.DataFrame") -> None:  # type: ignore[name-defined]
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return
        try:
            # Fast row-hash: xor column hashes (uint64) to produce a row signature
            h = None
            for c in df.columns:
                hc = pd.util.hash_pandas_object(df[c], index=False).to_numpy(
                    dtype="uint64", copy=False
                )
                h = hc if h is None else (h ^ hc)
            if h is None:
                return
            self.rows += int(len(h))
            for v in h:
                self.kmv.add(int(v))
        except Exception:
            # Conservative fallback: sample a few stringified rows
            n = min(2000, len(df))
            sample = df.head(n).astype(str).agg("|".join, axis=1)
            for s in sample:
                self.kmv.add(s)
            self.rows += n

    def update_from_polars(self, df: "pl.DataFrame") -> None:  # type: ignore[name-defined]
        try:
            import polars as pl  # type: ignore
        except Exception:
            return
        try:
            # Optimized row hashing - use Polars' built-in row hashing if available
            if hasattr(df, "hash_rows"):
                h = df.hash_rows().to_numpy()
                self.rows += int(h.size)
                for v in h:
                    self.kmv.add(int(v))
                return

            # Fallback to optimized column-wise hashing
            h = None
            for c in df.columns:
                hc = df[c].hash().to_numpy()
                h = hc if h is None else (h ^ hc)
            if h is None:
                return
            self.rows += int(h.size)
            for v in h:
                self.kmv.add(int(v))
        except Exception:
            # Fallback: sample small head and reuse pandas-based path for hashing
            try:
                sample = df.head(min(2000, df.height)).to_pandas()
                self.update_from_pandas(sample)
            except Exception:
                self.rows += min(2000, df.height)

    def approx_duplicates(self) -> Tuple[int, float]:
        uniq = self.kmv.estimate()
        d = max(0, self.rows - uniq)
        pct = (d / self.rows * 100.0) if self.rows else 0.0
        return d, pct
