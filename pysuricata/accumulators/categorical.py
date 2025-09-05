from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .sketches import KMV, MisraGries, ReservoirSampler


@dataclass
class CategoricalSummary:
    name: str
    count: int
    missing: int
    unique_est: int
    top_items: List[Tuple[str, int]]
    approx: bool
    # extras for alignment
    mem_bytes: int = 0
    avg_len: Optional[float] = None
    len_p90: Optional[int] = None
    empty_zero: int = 0
    case_variants_est: int = 0
    trim_variants_est: int = 0
    dtype_str: str = "categorical"


class CategoricalAccumulator:
    def __init__(self, name: str, topk_k: int = 50, uniques_k: int = 2048) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self._uniques = KMV(uniques_k)
        self._uniques_lower = KMV(uniques_k)
        self._uniques_strip = KMV(uniques_k)
        self.topk = MisraGries(topk_k)
        self._len_sum = 0
        self._len_n = 0
        self._len_sample = ReservoirSampler(5000)
        self._empty_zero = 0
        self._bytes_seen = 0
        self._dtype_str: str = "categorical"

    def update(self, arr: Sequence[Any]) -> None:
        for v in arr:
            if v is None:
                self.missing += 1
                continue
            if isinstance(v, float):
                try:
                    import math
                    if math.isnan(v):
                        self.missing += 1
                        continue
                except Exception:
                    pass
            self.count += 1
            s = str(v)
            self.topk.add(s, 1)
            self._uniques.add(s)
            try:
                self._uniques_lower.add(s.lower())
                self._uniques_strip.add(s.strip())
            except Exception:
                pass
            L = len(s)
            self._len_sum += L
            self._len_n += 1
            self._len_sample.add(float(L))
            if L == 0:
                self._empty_zero += 1
            self._bytes_seen += L  # approximate

    def finalize(self) -> CategoricalSummary:
        items = self.topk.items()[:10]
        # cast keys to str
        top_items = [(str(k), int(c)) for k, c in items]
        return CategoricalSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            unique_est=self._uniques.estimate(),
            top_items=top_items,
            approx=not self._uniques.is_exact,
            mem_bytes=int(self._bytes_seen),
            avg_len=(self._len_sum / self._len_n) if self._len_n else None,
            len_p90=(int(np.quantile(self._len_sample.values(), 0.90)) if self._len_sample.values() else None),
            empty_zero=int(self._empty_zero),
            case_variants_est=max(0, int(self._uniques.estimate() - self._uniques_lower.estimate())),
            trim_variants_est=max(0, int(self._uniques.estimate() - self._uniques_strip.estimate())),
            dtype_str=self._dtype_str,
        )

    @property
    def avg_len(self) -> Optional[float]:
        return (self._len_sum / self._len_n) if self._len_n else None

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "categorical"

