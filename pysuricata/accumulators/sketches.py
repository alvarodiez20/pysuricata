from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import hashlib
import random


def _u64(x: bytes) -> int:
    """Return a 64-bit unsigned integer hash from bytes using SHA1.

    Fast enough and avoids external dependencies. Uses the first 8 bytes
    of the sha1 digest to build an unsigned 64-bit integer.
    """
    return int.from_bytes(hashlib.sha1(x).digest()[:8], "big", signed=False)


class KMV:
    """K-Minimum Values distinct counter (approximate uniques) without extra deps.

    Keep the k smallest 64-bit hashes of the observed values. If fewer than k items
    have been seen, |S| is exact uniques. Otherwise, estimate uniques as (k-1)/t,
    where t is the kth smallest hash normalized to (0,1].
    """

    __slots__ = ("k", "_values")

    def __init__(self, k: int = 2048) -> None:
        self.k = int(k)
        self._values: List[int] = []  # store as integers in [0, 2^64)

    def add(self, v: Any) -> None:
        if v is None:
            v = b"__NULL__"
        elif isinstance(v, bytes):
            pass
        else:
            v = str(v).encode("utf-8", "ignore")
        h = _u64(v)
        if len(self._values) < self.k:
            self._values.append(h)
            if len(self._values) == self.k:
                self._values.sort()
        else:
            # maintain k-smallest set (max-heap simulation via last element after sort)
            if h < self._values[-1]:
                # insert in sorted order (k is small)
                lo, hi = 0, self.k - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if self._values[mid] < h:
                        lo = mid + 1
                    else:
                        hi = mid
                self._values.insert(lo, h)
                # trim to size
                del self._values[self.k]

    @property
    def is_exact(self) -> bool:
        return len(self._values) < self.k

    def estimate(self) -> int:
        n = len(self._values)
        if n == 0:
            return 0
        if n < self.k:
            # exact
            return n
        # normalize kth smallest to (0,1]
        kth = self._values[-1]
        t = (kth + 1) / 2**64
        if t <= 0:
            return n
        return max(n, int(round((self.k - 1) / t)))


class ReservoirSampler:
    """Reservoir sampler for numeric/datetime values to approximate quantiles/histograms."""

    __slots__ = ("k", "_buf", "_seen")

    def __init__(self, k: int = 20_000) -> None:
        self.k = int(k)
        self._buf: List[float] = []
        self._seen: int = 0

    def add_many(self, arr: Sequence[float]) -> None:
        for x in arr:
            self.add(float(x))

    def add(self, x: float) -> None:
        self._seen += 1
        if len(self._buf) < self.k:
            self._buf.append(x)
        else:
            j = random.randint(1, self._seen)
            if j <= self.k:
                self._buf[j - 1] = x

    def values(self) -> List[float]:
        return self._buf


class MisraGries:
    """Heavy hitters (top-K) with deterministic memory.

    Maintains up to k counters. Good for approximate top categories.
    """

    __slots__ = ("k", "counters")

    def __init__(self, k: int = 50) -> None:
        self.k = int(k)
        self.counters: Dict[Any, int] = {}

    def add(self, x: Any, w: int = 1) -> None:
        if x in self.counters:
            self.counters[x] += w
            return
        if len(self.counters) < self.k:
            self.counters[x] = w
            return
        # decrement all
        to_del = []
        for key in list(self.counters.keys()):
            self.counters[key] -= w
            if self.counters[key] <= 0:
                to_del.append(key)
        for key in to_del:
            del self.counters[key]

    def items(self) -> List[Tuple[Any, int]]:
        # items are approximate; a second pass could refine if needed
        return sorted(self.counters.items(), key=lambda kv: (-kv[1], str(kv[0])[:64]))

