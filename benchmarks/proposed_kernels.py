"""Two verified drop-in replacements for the hottest kernels, plus their proofs.

Both were measured on 1M float64 values in 5 chunks, against the shipped 0.0.21
implementations, and both are behaviour-preserving:

    KMV.add_many            639 ns/value  ->   87 ns/value   (7.4x, identical estimate)
    ReservoirSampler        146 ns/value  ->   32 ns/value   (4.6x, bit-identical sample)

Run ``python -m benchmarks.proposed_kernels`` to re-verify and re-time both on
your machine before adopting either.

Neither needs Rust. Together they address ~60% of the numeric accumulator's
cost. Copy the bodies into ``pysuricata/accumulators/sketches.py``; the classes
here exist only so the proofs below can diff them against the shipped versions.
"""

from __future__ import annotations

import math

import numpy as np

_TINY = 5e-324
_NO_MORE = 1 << 62
_DRAW_BLOCK = 4096
_SCHED_BLOCK = 512


# ---------------------------------------------------------------------------
# 1. KMV: reject against the threshold before sorting
# ---------------------------------------------------------------------------


def batch_add_hashes_fast(values: np.ndarray, k: int, hashes: np.ndarray) -> np.ndarray:
    """Replacement body for ``KMV._batch_add_hashes``.

    The shipped version does ``np.unique(hashes)`` then ``np.union1d`` on every
    batch -- an O(m log m) sort of the *whole* batch, every chunk, forever.

    But once the sketch is full, the k-th smallest hash it already holds is a
    hard admission threshold: nothing at or above it can ever enter. Testing
    that first is a single vectorised compare that discards, for a warm sketch
    on high-cardinality data, well over 99.9% of the batch. Only the survivors
    get sorted.

    This is the same trick the native crate uses (one compare against the heap
    root); it just turns out to be worth 7x in plain NumPy too.

    Args:
        values: current sketch contents, sorted ascending, at most ``k`` long.
        k: sketch size.
        hashes: batch of 64-bit hashes to offer.

    Returns:
        The new sketch contents.
    """
    if hashes.size == 0:
        return values

    if values.size >= k:
        threshold = values[-1]
        hashes = hashes[hashes < threshold]
        if hashes.size == 0:
            return values

    incoming = np.unique(np.asarray(hashes, dtype=np.uint64))
    combined = np.union1d(values, incoming) if values.size else incoming
    return combined[:k]


class FastKMV:
    """Minimal KMV built on ``batch_add_hashes_fast``, for the proofs below.

    Note the other change worth carrying over: ``_values`` stays a NumPy array.
    The shipped version ends every batch with ``.tolist()``, re-boxing k
    integers into Python objects on each chunk.
    """

    __slots__ = ("k", "_values")

    def __init__(self, k: int = 2048) -> None:
        self.k = int(k)
        self._values = np.empty(0, dtype=np.uint64)

    def offer_hashes(self, hashes: np.ndarray) -> None:
        self._values = batch_add_hashes_fast(self._values, self.k, hashes)

    def estimate(self) -> int:
        n = self._values.size
        if n < self.k:
            return int(n)
        t = (int(self._values[-1]) + 1) / 2**64
        return int(round((self.k - 1) / t)) if t > 0 else int(n)


# ---------------------------------------------------------------------------
# 2. Reservoir: vectorise Algorithm L without changing a single draw
# ---------------------------------------------------------------------------


class VecReservoirL:
    """Algorithm L with the acceptance schedule computed in NumPy.

    The shipped implementation is correct and has a property worth protecting:
    the sample is *bit-identical* regardless of how the stream was chunked,
    because the skip sequence depends only on the draw sequence. The accuracy
    oracle asserts exactly that (``test_sample_is_independent_of_chunking``).

    So the naive speedup -- a vectorised Algorithm R with one draw per element
    -- is not acceptable: its draw count depends on batch sizes, which breaks
    that invariant even though the sample stays unbiased.

    This version keeps the invariant. Acceptances are generated ``_SCHED_BLOCK``
    at a time from the same uniform stream, in the same order, with the running
    ``w`` computed by a single ``cumsum`` of logs instead of one multiply per
    acceptance. Unconsumed acceptances are cached, so no draw is ever taken
    twice or skipped. Output is bit-identical to the shipped class for the same
    seed -- verified below across k, n and chunk count.

    The cost per acceptance drops from four Python calls to a few array ops
    amortised over 512 acceptances.
    """

    __slots__ = (
        "k",
        "_buf",
        "_seen",
        "_w",
        "_next",
        "_draws",
        "_draw_i",
        "_pend_pos",
        "_pend_slot",
        "_pend_i",
        "_exhausted",
    )

    def __init__(self, k: int = 20_000) -> None:
        self.k = int(k)
        self._buf: list[float] = []
        self._seen = 0
        self._w = 1.0
        self._next = -1
        self._draws = np.empty(0, dtype=float)
        self._draw_i = 0
        self._pend_pos = np.empty(0, dtype=np.int64)
        self._pend_slot = np.empty(0, dtype=np.int64)
        self._pend_i = 0
        self._exhausted = False

    def _uniforms(self, n: int) -> np.ndarray:
        """The same draw sequence the scalar ``_uniform()`` produces, n at a time."""
        out = np.empty(n, dtype=float)
        filled = 0
        while filled < n:
            if self._draw_i >= self._draws.size:
                self._draws = np.random.random(_DRAW_BLOCK)
                self._draw_i = 0
            take = min(n - filled, self._draws.size - self._draw_i)
            out[filled : filled + take] = self._draws[
                self._draw_i : self._draw_i + take
            ]
            self._draw_i += take
            filled += take
        return np.maximum(out, _TINY)  # == `u if u > 0.0 else _TINY`

    def _refill_schedule(self) -> None:
        """Generate the next block of (position, slot) acceptances in one go."""
        m = _SCHED_BLOCK
        u = self._uniforms(3 * m)
        u_slot, u_w, u_skip = u[0::3], u[1::3], u[2::3]

        # w_j = w * prod_{i<=j} exp(log(u_w_i)/k). One cumsum, not m multiplies.
        w = self._w * np.exp(np.cumsum(np.log(u_w)) / self.k)
        denom = np.log1p(-w)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(u_skip) / denom
        # denom >= 0 means w underflowed: no further acceptances are possible.
        skips = np.where(denom < 0.0, np.floor(raw), float(_NO_MORE)).astype(np.int64)
        skips = np.minimum(skips, _NO_MORE)

        pos = self._next + np.concatenate(([0], np.cumsum(skips[:-1] + 1)))

        dead = np.flatnonzero(skips >= _NO_MORE)
        if dead.size:
            cut = int(dead[0]) + 1  # that acceptance still happens, then we stop
            pos, skips, w = pos[:cut], skips[:cut], w[:cut]
            self._exhausted = True

        self._pend_pos = pos
        self._pend_slot = (u_slot[: pos.size] * self.k).astype(np.int64)
        self._pend_i = 0
        self._w = float(w[-1])
        self._next = _NO_MORE if self._exhausted else int(pos[-1] + skips[-1] + 1)

    def _start_skipping(self) -> None:
        self._w = 1.0
        u = self._uniforms(2)
        self._w *= math.exp(math.log(u[0]) / self.k)
        denom = math.log1p(-self._w) if self._w < 1.0 else 0.0
        self._next = (
            _NO_MORE if denom >= 0.0 else self._seen + int(math.log(u[1]) / denom)
        )

    def add_many(self, arr) -> None:
        if len(arr) == 0:
            return
        arr = np.asarray(arr, dtype=float)

        if len(self._buf) < self.k:
            needed = min(self.k - len(self._buf), len(arr))
            self._buf.extend(arr[:needed].tolist())
            self._seen += needed
            arr = arr[needed:]
            if len(self._buf) >= self.k:
                self._start_skipping()
            if len(arr) == 0:
                return

        end = self._seen + len(arr)
        buf = self._buf
        while True:
            if self._pend_i >= self._pend_pos.size:
                if self._exhausted and self._pend_pos.size:
                    break
                self._refill_schedule()
            pend = self._pend_pos[self._pend_i :]
            take = int(np.searchsorted(pend, end))
            if take:
                idx = pend[:take] - self._seen
                slots = self._pend_slot[self._pend_i : self._pend_i + take]
                vals = arr[idx]
                for s, v in zip(slots.tolist(), vals.tolist(), strict=False):
                    buf[s] = v
                self._pend_i += take
            if take < pend.size or self._exhausted:
                break
        self._seen = end

    def add(self, x: float) -> None:
        if len(self._buf) < self.k:
            self._buf.append(float(x))
            self._seen += 1
            if len(self._buf) >= self.k:
                self._start_skipping()
            return
        self.add_many(np.array([x], dtype=float))

    def values(self) -> list[float]:
        return self._buf


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------


def _verify_reservoir() -> bool:
    """Bit-identity against the shipped class, across k, n and chunk count."""
    from pysuricata.accumulators.sketches import ReservoirSampler

    cases = [
        (100, 5_000, 1),
        (100, 5_000, 7),
        (2_000, 200_000, 1),
        (2_000, 200_000, 13),
        (2_000, 200_000, 500),
        (20_000, 1_000_000, 5),
    ]
    ok = True
    for k, n, nchunks in cases:
        arr = np.arange(n, dtype=float)
        outs = []
        for cls in (ReservoirSampler, VecReservoirL):
            np.random.seed(42)
            r = cls(k)
            for c in np.array_split(arr, nchunks):
                r.add_many(np.ascontiguousarray(c))
            outs.append(list(r.values()))
        same = outs[0] == outs[1]
        ok &= same
        print(f"  k={k:>6} n={n:>9,} chunks={nchunks:>4}  bit-identical: {same}")
    return ok


def _verify_kmv() -> bool:
    """The pre-filter must not change the estimate."""
    from pysuricata.accumulators.sketches import KMV, _hash_numeric_array

    ok = True
    for n, k in [(50_000, 1024), (1_000_000, 2048), (1_000_000, 4096)]:
        arr = np.random.default_rng(0).standard_normal(n)
        chunks = np.array_split(arr, 5)

        ref = KMV(k)
        fast = FastKMV(k)
        for c in chunks:
            c = np.ascontiguousarray(c)
            ref.add_many(c)
            fast.offer_hashes(_hash_numeric_array(c))

        a, b = ref.estimate(), fast.estimate()
        # The pre-filter changes nothing about which hashes survive, so the
        # estimates must agree to the integer, not merely be close.
        same = a == b
        ok &= same
        print(f"  n={n:>9,} k={k:>5}  ref={a:>9,}  fast={b:>9,}  identical: {same}")
    return ok


def _time() -> None:
    import gc
    import time

    from pysuricata.accumulators.sketches import (
        KMV,
        ReservoirSampler,
        _hash_numeric_array,
    )

    n, ch = 1_000_000, 200_000
    arr = np.random.default_rng(0).standard_normal(n)
    chunks = [arr[i : i + ch] for i in range(0, n, ch)]

    def bench(fn, reps=3):
        fn()
        gc.disable()
        best = float("inf")
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - s)
        gc.enable()
        return best

    def kmv_ref():
        s = KMV(2048)
        for c in chunks:
            s.add_many(c)

    def kmv_fast():
        s = FastKMV(2048)
        for c in chunks:
            s.offer_hashes(_hash_numeric_array(c))

    def res_ref():
        r = ReservoirSampler(20_000)
        for c in chunks:
            r.add_many(c)

    def res_fast():
        r = VecReservoirL(20_000)
        for c in chunks:
            r.add_many(c)

    for label, ref, fast in [
        ("KMV.add_many", kmv_ref, kmv_fast),
        ("ReservoirSampler.add_many", res_ref, res_fast),
    ]:
        a, b = bench(ref), bench(fast)
        print(
            f"  {label:<28}{a * 1e9 / n:7.0f} -> {b * 1e9 / n:5.0f} ns/value   {a / b:5.2f}x"
        )


def main() -> int:
    print("KMV pre-filter -- estimates must be identical")
    kmv_ok = _verify_kmv()
    print("\nVectorised Algorithm L -- samples must be bit-identical")
    res_ok = _verify_reservoir()
    print("\nTimings (1M values, 5 chunks, best of 3)")
    _time()
    print(
        "\n"
        + ("ALL CHECKS PASS" if (kmv_ok and res_ok) else "MISMATCH -- do not adopt")
    )
    return 0 if (kmv_ok and res_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
