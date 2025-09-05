from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
import math


@dataclass
class BooleanSummary:
    name: str
    count: int
    missing: int
    true_n: int
    false_n: int
    mem_bytes: int = 0
    dtype_str: str = "boolean"


class BooleanAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self.true_n = 0
        self.false_n = 0
        self._dtype_str: str = "boolean"
        self._mem_bytes: int = 0

    def update(self, arr: Sequence[Any]) -> None:
        for v in arr:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                self.missing += 1
                continue
            if bool(v):
                self.true_n += 1
            else:
                self.false_n += 1
            self.count += 1

    def finalize(self) -> BooleanSummary:
        # Approximate memory: ~1 byte per boolean cell (very rough; marked as ≈ in UI)
        mem_bytes = int(self._mem_bytes if self._mem_bytes > 0 else (self.count + self.missing))
        return BooleanSummary(self.name, self.count, self.missing, self.true_n, self.false_n, mem_bytes, self._dtype_str)

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "boolean"

    def add_mem(self, n: int) -> None:
        try:
            self._mem_bytes += int(n)
        except Exception:
            pass

