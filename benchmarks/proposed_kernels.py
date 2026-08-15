"""Measured, behaviour-checked proposals for the two remaining hot kernels.

Re-verified against 0.0.26. Run ``python -m benchmarks.proposed_kernels``.

Status of the proposals this file has carried:

    KMV pre-filter        LANDED in 0.0.27. 51 -> 17 ns/value on this kernel,
                                 estimates identical, chunk-invariance intact.
    KMV ndarray _values   REJECTED -- wins its own benchmark by 2.7x and loses
                                 35% end to end. See the note below.
    Misra-Gries gate      LANDED in 0.0.27, together with the removal of a
                                 finalize() fallback that was worse than the
                                 thing this gate was written to remove.
    Vectorised Alg L      RETIRED -- superseded by the bulk scheduler that
                                 shipped in 0.0.26 (PR #49). See the note below.

The two verifiers below are kept because they are still the cheapest check that
the shipped behaviour is right. ``FastKMV`` now measures the *residual* gap to
the array-backed variant, which is why that gap is a rejection and not a TODO.

Numbers are from 1M float64 values in 5 chunks, best of 3, GC disabled.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# RETIRED: vectorised Algorithm L
# ---------------------------------------------------------------------------
#
# 0.0.26 shipped its own bulk acceptance scheduler (`_SCHEDULE_BLOCK`, a cumsum
# of logs) which took the reservoir from 154 to 57 ns/value -- 2.7x, and enough
# to move it from 10.2% of the numeric path down to 4.5%.
#
# The version this file used to propose is now BOTH obsolete and wrong to adopt:
# it was built to reproduce the *old* scalar draw sequence bit-for-bit, and
# 0.0.26 draws in a different order, so it no longer matches. It still measures
# ~2.6x faster than the shipped scheduler (21 vs 54 ns/value), but that is 4.5%
# of one column kind, and buying it would mean re-establishing bit-identity
# against the new sequence from scratch. Not worth it. Left here only so the
# history of the decision is legible.


# ---------------------------------------------------------------------------
# REJECTED: hold KMV._values as a uint64 array
# ---------------------------------------------------------------------------
#
# The other half of the KMV proposal was to drop the `.tolist()` at the end of
# every batch merge and keep `_values` as a NumPy array. In isolation it works:
# `FastKMV` below still measures ~2.7x faster than the shipped `KMV.add_many`
# even after the pre-filter landed, and that is the entire remaining gap.
#
# End to end it is a 35% regression -- mixed 200k x 14 goes from 1173 ms to
# 1590 ms. The reason is a path this benchmark never calls: `_add_hash_to_kmv`,
# which the categorical accumulator uses once per distinct value. With a list
# that is `bisect` plus an in-place memmove; with an array it is `searchsorted`
# plus `np.insert`, which allocates and copies the whole array every time.
#
# This is the second time on this codebase that a kernel benchmark has ranked
# something the wall clock disagrees with. The rule stands: confirm against
# end-to-end wall clock before adopting, and make sure the benchmark exercises
# the call sites that actually exist.


# ---------------------------------------------------------------------------
# 1. KMV: reject against the threshold before sorting             [LANDED]
# ---------------------------------------------------------------------------


def batch_add_hashes_fast(values: np.ndarray, k: int, hashes: np.ndarray) -> np.ndarray:
    """Replacement body for ``KMV._batch_add_hashes``.

    Landed in 0.0.27 as ``KMV._batch_add_hashes``. Kept here because the
    verifier below is the cheapest check that the shipped estimates are right.

    The pre-0.0.27 version ran ``np.unique(hashes)`` then ``np.union1d`` on every
    batch -- an O(m log m) sort of the whole batch, every chunk, forever. But
    once the sketch is full, the k-th smallest hash it already holds is a hard
    admission threshold: nothing at or above it can ever enter. Testing that
    first is one vectorised compare that discards, for a warm sketch on
    high-cardinality data, well over 99.9% of the batch. Only survivors get
    sorted.

    Same fast-reject the native crate does against its heap root.

    This copy additionally holds ``_values`` as a NumPy array, which the shipped
    version deliberately does not -- see the REJECTED note above. That is the
    whole of the residual gap the timings below still report.
    """
    if hashes.size == 0:
        return values

    if values.size >= k:
        hashes = hashes[hashes < values[-1]]
        if hashes.size == 0:
            return values

    incoming = np.unique(np.asarray(hashes, dtype=np.uint64))
    combined = np.union1d(values, incoming) if values.size else incoming
    return combined[:k]


class FastKMV:
    """Minimal KMV over ``batch_add_hashes_fast``, for the proof below."""

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
# 2. Misra-Gries: only run it where its answer means something      [LANDED]
# ---------------------------------------------------------------------------
#
# `numeric.py` feeds every finite value to a top-k sketch, unconditionally, for
# every numeric column. That is **34% of the numeric accumulator**.
#
# On a discrete column the answer is worth having. On a high-cardinality one it
# is not merely wasted -- it is actively misleading. Measured on 200k rows:
#
#     column                    unique     top_values   coverage   counts
#     discrete int (12)             12       12 rows      100.0%   16893, 16813, ...
#     float w/ repeats (500)       500       50 rows        6.7%   414, 409, ...
#     high-card float (47,810)  47,810       28 rows        0.0%   2, 1, 1
#     continuous float         200,923       29 rows        0.0%   1, 1, 1
#
# The last two render a "Common values" table in the numeric card listing values
# that occurred **once**. So gating this does two things at once: it removes a
# third of the numeric accumulator's cost, and it removes a table that presents
# sampling noise as a finding.


def should_track_top_k(unique_est: float, count: int, top_k: int = 50) -> bool:
    """Whether a numeric column's top-k answer will carry information.

    The rule: a top-k table is worth building when the k tracked values could
    plausibly cover a meaningful share of the column. If the column holds many
    more distinct values than the sketch has counters, the table degenerates to
    a list of singletons.

    ``unique_est`` is already available -- ``NumericAccumulator`` keeps a KMV
    sketch and the estimate is O(1) to read. Evaluate this once per chunk and
    latch it off; do not re-enable, since counts accumulated before the cutoff
    would be partial.

    Args:
        unique_est: current distinct-count estimate for the column.
        count: number of finite values seen so far.
        top_k: number of Misra-Gries counters.

    Returns:
        True while top-k should keep being fed.
    """
    if count <= 0:
        return True
    if unique_est <= top_k:
        return True  # exact and complete: always worth it
    # Expected coverage of the k most frequent values under a flat distribution.
    # Below a few percent the table is noise. Skewed columns beat this bound,
    # which is the safe direction: they stay enabled.
    return (top_k / max(unique_est, 1.0)) >= 0.02


# ---------------------------------------------------------------------------
# Proofs and timings
# ---------------------------------------------------------------------------


def _verify_kmv() -> bool:
    """The pre-filter must not change the estimate, at all."""
    from pysuricata.accumulators.sketches import KMV, _hash_numeric_array

    ok = True
    for n, k in [(50_000, 1024), (1_000_000, 2048), (1_000_000, 4096)]:
        arr = np.random.default_rng(0).standard_normal(n)
        chunks = np.array_split(arr, 5)
        ref, fast = KMV(k), FastKMV(k)
        for c in chunks:
            c = np.ascontiguousarray(c)
            ref.add_many(c)
            fast.offer_hashes(_hash_numeric_array(c))
        a, b = ref.estimate(), fast.estimate()
        same = a == b
        ok &= same
        print(f"  n={n:>9,} k={k:>5}  ref={a:>9,}  fast={b:>9,}  identical: {same}")
    return ok


def _verify_topk_gate() -> bool:
    """The gate must keep top-k exactly where it is informative."""
    g = np.random.default_rng(0)
    n = 200_000
    cases = [
        ("discrete int (12)", g.integers(0, 12, n).astype(float), True),
        ("float w/ repeats (500)", g.integers(0, 500, n) / 4.0, True),
        ("high-card float", g.integers(0, 50_000, n) / 8.0, False),
        ("continuous float", g.standard_normal(n), False),
    ]
    from pysuricata.accumulators.numeric import NumericAccumulator

    ok = True
    for name, vals, expect_useful in cases:
        acc = NumericAccumulator("x")
        v = np.ascontiguousarray(np.asarray(vals, dtype=float))
        try:
            acc.update(v, row_offset=0)
        except TypeError:
            acc.update(v)
        s = acc.finalize()
        tv = s.top_values or []
        coverage = sum(int(c) for _, c in tv) / n
        decision = should_track_top_k(float(s.unique_est), int(s.count))
        agrees = decision == expect_useful
        ok &= agrees
        print(
            f"  {name:<24} unique~{s.unique_est:>7,.0f}  coverage {coverage:6.1%}"
            f"  gate={'keep' if decision else 'skip':<4}  as expected: {agrees}"
        )
    return ok


def _time() -> None:
    import gc
    import time

    from pysuricata.accumulators.sketches import KMV, MisraGries, _hash_numeric_array

    n, ch = 1_000_000, 200_000
    arr = np.random.default_rng(0).standard_normal(n)
    chunks = [np.ascontiguousarray(arr[i : i + ch]) for i in range(0, n, ch)]

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

    def mg_on():
        s = MisraGries(50)
        for c in chunks:
            s.add_many(c.tolist())

    def mg_off():
        for _ in chunks:
            pass

    a, b = bench(kmv_ref), bench(kmv_fast)
    print(
        f"  {'KMV.add_many':<26}{a * 1e9 / n:6.0f} -> {b * 1e9 / n:5.0f} ns/value   {a / b:5.2f}x"
    )
    c, d = bench(mg_on), bench(mg_off)
    print(
        f"  {'MisraGries.add_many':<26}{c * 1e9 / n:6.0f} -> {d * 1e9 / n:5.0f} ns/value   "
        f"(gated off on this column shape)"
    )


def main() -> int:
    print("KMV pre-filter -- estimates must be identical")
    kmv_ok = _verify_kmv()
    print("\nTop-k gate -- must keep top-k exactly where it is informative")
    gate_ok = _verify_topk_gate()
    print("\nTimings (1M values, 5 chunks, best of 3)")
    _time()
    ok = kmv_ok and gate_ok
    print("\n" + ("ALL CHECKS PASS" if ok else "MISMATCH -- do not adopt"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
