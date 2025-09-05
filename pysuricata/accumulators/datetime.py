from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
from datetime import datetime

from .sketches import KMV, ReservoirSampler


@dataclass
class DatetimeSummary:
    name: str
    count: int
    missing: int
    min_ts: Optional[int]
    max_ts: Optional[int]
    by_hour: List[int]  # 24 counts
    by_dow: List[int]   # 7 counts, Monday=0
    by_month: List[int] # 12 counts, Jan=1 index => store 12-length
    # v2 additions
    dtype_str: str = "datetime"
    mono_inc: bool = False
    mono_dec: bool = False
    mem_bytes: int = 0
    sample_ts: Optional[List[int]] = None
    sample_scale: float = 1.0


class DatetimeAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.missing = 0
        self._min_ts: Optional[int] = None
        self._max_ts: Optional[int] = None
        self.by_hour = [0] * 24
        self.by_dow = [0] * 7
        self.by_month = [0] * 12
        self._uniques = KMV(2048)
        self._dtype_str: str = "datetime"
        self._mem_bytes: int = 0
        self._sample = ReservoirSampler(20_000)
        self._last_ts: Optional[int] = None
        self._mono_inc = True
        self._mono_dec = True

    def update(self, arr_ns: Sequence[Optional[int]]) -> None:
        # arr_ns: integer nanoseconds since epoch; None/NaN for missing
        for v in arr_ns:
            if v is None or (isinstance(v, float) and (v != v)):
                self.missing += 1
                continue
            ts = int(v)
            self.count += 1
            if self._min_ts is None or ts < self._min_ts:
                self._min_ts = ts
            if self._max_ts is None or ts > self._max_ts:
                self._max_ts = ts
            self._uniques.add(ts)
            # sample and monotonicity
            self._sample.add(float(ts))
            if self._last_ts is not None:
                if ts < self._last_ts:
                    self._mono_inc = False
                if ts > self._last_ts:
                    self._mono_dec = False
            self._last_ts = ts
            # breakdowns (UTC)
            try:
                dt = datetime.utcfromtimestamp(ts / 1_000_000_000)
                self.by_hour[dt.hour] += 1
                self.by_dow[dt.weekday()] += 1
                self.by_month[dt.month - 1] += 1
            except Exception:
                # ignore ill-formed
                pass

    def finalize(self) -> DatetimeSummary:
        svals = [int(x) for x in self._sample.values()]
        sample_scale = (self.count / len(svals)) if svals else 1.0
        return DatetimeSummary(
            name=self.name,
            count=self.count,
            missing=self.missing,
            min_ts=self._min_ts,
            max_ts=self._max_ts,
            by_hour=self.by_hour,
            by_dow=self.by_dow,
            by_month=self.by_month,
            dtype_str=self._dtype_str,
            mono_inc=bool(self._mono_inc),
            mono_dec=bool(self._mono_dec),
            mem_bytes=int(self._mem_bytes),
            sample_ts=svals,
            sample_scale=float(sample_scale),
        )

    @property
    def unique_est(self) -> int:
        return self._uniques.estimate()

    def set_dtype(self, dtype_str: str) -> None:
        try:
            self._dtype_str = str(dtype_str)
        except Exception:
            self._dtype_str = "datetime"

    def add_mem(self, n: int) -> None:
        try:
            self._mem_bytes += int(n)
        except Exception:
            pass

