# PySuricata roadmap v2 — re-audit at 0.0.21

Supersedes the 0.0.16 plan. Every number was measured with the harness in
`benchmarks/`, against this repo at `da33604` and `pysuricata==0.0.16` from PyPI,
in separate venvs, interleaved, best of 3, GC disabled. Python 3.11.15,
pandas 2.3.3, numpy 2.4.4, ydata-profiling 4.18.4, 2-core x86-64 Linux container.

Absolute times on that box are noisier than in the first pass, so every
version-to-version comparison here is interleaved on the same machine within the
same minute. **Re-run before publishing anything.**

A rendered version with charts is the `pysuricata-roadmap` artifact.

---

## Where things stand

| | |
|---|---|
| Flagged items closed | **6 of 11** — every one that had a failing test |
| 0.0.16 → 0.0.21, 200k × 14 | **1.24×** (7.583 s → 6.118 s) |
| vs ydata-profiling, 25k × 14 | **9.1×** (1.83 s vs 16.72 s); was 7.9× at 0.0.16 |
| Peak RSS | 173 MB vs 459 MB; **13×** less above the ~150 MB interpreter floor |
| Hashing share of profiled self time | **13.4% → 0.3%** |
| Still available in pure Python | **~2.5×**, measured |

### What went well

Two things stand out beyond the fixes themselves.

You didn't just clear the `xfail` markers — the oracle went from 464 to 578
lines, gaining `TestReservoirInvariants`, `TestRngIsolation`, and three generator
tests, including one asserting the RNG is restored *even when profiling raises*.

And the reservoir fix went past the brief. Algorithm L gives a sample that is
**bit-identical across chunk sizes**, not merely unbiased, and
`test_sample_is_independent_of_chunking` now pins that. It is a stronger property
than I specified — and it is why the obvious speedup for it is wrong (see below).

---

## Scorecard

### Closed

| Item | How |
|---|---|
| **Generator drops chunk 0** — `engine.py` | Peek-and-splice with `itertools.chain([first_chunk_peek], stream)`, sniffing from the peeked chunk. Three new tests including the one-chunk "Empty source" case and a generator-vs-DataFrame equivalence check. |
| **Reservoir sampling biased** — `sketches.py` | Full Algorithm L rewrite with geometric skips; five new invariant tests. *But* the implementation is a per-acceptance Python loop with four calls each — 5× slower than needed. Fix below. |
| **M3/M4 batch merge wrong** — `algorithms.py` | The batch now builds a `StreamingMoments` and delegates to the correct Pébay `merge()`, with a comment explaining why the old inline formula was wrong. |
| **Global RNG reset** — `report.py` | Solved with a save/restore context manager rather than per-accumulator generators. Verified: NumPy and stdlib state both survive, including on failure. *Residual:* the reservoir still draws from global `np.random`, so column-level threading remains blocked. |
| **Correlation cancellation** — `correlation.py` | Chan pairwise co-moments (`mean_x`, `delta_x`, merged per batch). |
| **Skewness denominator** — `algorithms.py` | `m2_pop = self._m2 / self.count`, then `skew = (m3/n) / m2_pop**1.5`. Matches scipy. |

### Open

| Item | State |
|---|---|
| **Types decided from the first chunk** — `inference.py:550` | Unchanged: `unique_count < 10 or unique_ratio < 0.05` over a single chunk. |
| **Row count truncates to 2,000** — `sketches.py:534,555,563` | Unchanged. Four `min(2000, ...)` sites in the `RowKMV` fallback still cap the contribution to `rows`. |
| **Extreme indices chunk-local** — `consume.py:241` | Unchanged, including `_extreme_update_counter % 5`. Note the tracker is only **0.8%** of the numeric path, so the throttle buys nothing — drop it and make min/max exact. |
| **Dead options** — `config.py:137`, `chunking.py:240` | `corr_max_cols` is still declared, validated, copied into config, and never read by anything in `compute/`. The `0.7×optimal + 0.3×requested` chunk-size blend is unchanged. |
| **Pre-1906 timestamps as missing** — `datetime.py:230,316` | Unchanged; the `-2e18` bound is still 1906-05-13. |

### Phase 1 performance items

| Item | State |
|---|---|
| SHA-1 in the sketch | **Done.** blake2b-64 for bytes plus vectorised `_mix64_array`/`_hash_numeric_array`, and `RowKMV`'s uint64 row hashes go straight through it. |
| KMV extend-then-sort | **Done** — now `np.unique` + `np.union1d`. Still sorts the whole batch every chunk; see the drop-in. |
| Date sniff with `format="mixed"` | **Open** at `inference.py:384` and `consume.py:162`. Still 19% of profiled self time. |
| Monotonicity Python loop | **Open.** 8.5% of the numeric path; `np.diff` is 61× faster. |
| ExtremeTracker heap | **Deprioritise.** Measured at 0.8%. I over-weighted this in August. |
| Report v2 / native wiring | **Not started** — both are later phases, on schedule. |

**The pattern worth naming: all six items with tests are closed; all five without
are open.** If you want the rest fixed, write the test first.

---

## New evidence: where the time actually is

In the first pass I ranked hot spots with `cProfile`. That was a mistake, now
caught the hard way: the profiler put the new reservoir at ~30% of self time, but
swapping in a 5×-faster one moved end-to-end wall clock by **4%**. cProfile
charges per Python call and the reservoir makes millions of tiny ones.

So this pass measures two ways, both on wall clock with the profiler off.

### Cost per column kind, 200,000 rows

| Column kind | ms/column | ns/value |
|---|---:|---:|
| datetime | **1067** | 5,335 |
| numeric | 483 | 2,415 |
| categorical | 452 | 2,260 |
| boolean | 70 | 350 |
| *render, all columns* | *26 total* | *3.3% of wall clock* |

**Render is 3% of wall clock.** Report v2 is a size and UX project, not a
performance one — worth knowing before spending a weekend on it expecting a
speedup.

### Inside the numeric accumulator

1M values in 5 chunks. `update()` totals 1,504 ns/value.

| Component | Share | ns/value |
|---|---:|---:|
| KMV distinct count | **50.2%** | 754 |
| Misra-Gries top-k | **35.3%** | 530 |
| ReservoirSampler | 10.2% | 154 |
| MonotonicityDetector | 8.5% | 128 |
| OutlierDetector | 6.3% | 94 |
| StreamingHistogram | 2.0% | 30 |
| **StreamingMoments** | **1.3%** | 19 |
| ExtremeTracker | 0.8% | 12 |

### Misra-Gries is 35% of the numeric path, computing something meaningless

`numeric.py:336` feeds every finite value to a top-k sketch. On a continuous
float column every value is distinct, so the sketch does nothing but evict
counters — and `numeric_card.py:462` discards the result when it comes back
empty. That is 530 ns/value, more than a third of the whole numeric accumulator,
for a top-k that is thrown away.

Gate it on the KMV cardinality estimate (or simply on `int_like`) and that third
disappears with no output change on the columns where it was never useful.

---

## The native core is aimed at the wrong kernel

`native/` was built around a fused numeric scan — Pébay moments, 32 KiB tiling,
four accumulator chains. It works and the engineering stands. But the
decomposition says **moments are 1.3% of the numeric path**. Making them
infinitely fast buys 1.3%.

The part that matters is the one treated as secondary: `kmv.rs`.

| Kernel | Share of numeric path | Current | Native |
|---|---:|---:|---:|
| **KMV distinct count** | 50.2% | 754 ns | **8.7 ns** |
| Reservoir | 10.2% | 154 ns | 7.2 ns |
| Moments | 1.3% | 19 ns | 6.5 ns |

This does not make the crate a waste. The `moments.rs` tiling story is still the
best post in the series — a measured 1.7× from layout alone, and an honest "my
first attempt was slower than NumPy" — and its value is pedagogical, which is
what you wanted from the CS:APP study. And `kmv.rs` plus `hash_arrow_utf8` are
now the highest-value native work in the repo, because they attack 50% of the
numeric path and (wired to polars' Arrow buffers) the categorical path too.

What changes is the *order inside Phase 3*: ship the KMV kernel first, moments
last.

---

## Two verified drop-ins, no Rust

Both live in `benchmarks/proposed_kernels.py`. Run
`python -m benchmarks.proposed_kernels` to re-verify and re-time on your machine.

| Kernel | Before | After | Ratio | Behaviour |
|---|---:|---:|---:|---|
| `KMV.add_many` | 633 ns/value | 72 ns/value | **8.8×** | estimates *identical* |
| `ReservoirSampler.add_many` | 144 ns/value | 29 ns/value | **4.9×** | sample *bit-identical* |

### 1. KMV: reject before you sort

`_batch_add_hashes` runs `np.unique` over the whole batch and `np.union1d`
against the sketch, every chunk. But once the sketch is full, its k-th smallest
hash is a hard admission threshold — nothing at or above it can ever enter. One
vectorised compare discards over 99.9% of a batch before any sorting happens:

```python
if values.size >= k:
    hashes = hashes[hashes < values[-1]]      # the whole idea
    if hashes.size == 0:
        return values
incoming = np.unique(hashes)
return (np.union1d(values, incoming) if values.size else incoming)[:k]
```

Same fast-reject the native crate does against its heap root; worth 8.8× in plain
NumPy too. Estimates are identical, not merely close — the pre-filter cannot
change which hashes survive. Also carry over: keep `_values` as a NumPy array.
The current code ends every batch with `.tolist()`, re-boxing k integers into
Python objects per chunk.

### 2. Reservoir: vectorise Algorithm L without changing a single draw

**The obvious fix is wrong, and your own test catches it.** My first attempt was
a vectorised Algorithm R — one draw per element, accept with probability k/n.
Unbiased, 5× faster, and it fails `test_sample_is_independent_of_chunking`,
because its draw *count* depends on batch sizes, so the same seed gives different
samples at different chunk sizes. Your test caught a regression I was about to
recommend.

Algorithm L's acceptance schedule depends only on the draw sequence, never on
chunking. So it can be generated 512 acceptances at a time in NumPy — the running
`w` becomes a single `cumsum` of logs instead of one multiply per acceptance —
with unconsumed acceptances cached so no draw is ever taken twice. Same uniforms,
same order, same sample. Verified bit-identical at k ∈ {100, 2k, 20k}, n up to
1M, chunk counts from 1 to 500.

**Honest caveat:** at 200k rows the reservoir swap is within noise end-to-end,
because the reservoir is ~10% of one of four column kinds. The KMV one is the
real end-to-end win. Both grow with row count while render and setup stay fixed,
so measure at 1M+ rows — the size the streaming claim is about anyway.

---

## The datetime accumulator is now the worst thing in the codebase

**1,067 ms per column at 200,000 rows — 4.7 µs per value**, more than twice the
per-value cost of a numeric column. Conversion is only 33 ms of that; the rest is
the accumulator, which contains **four separate per-row Python loops**:

| Line | What it does per row |
|---|---|
| `datetime.py:54-56` | `for ts in ts_array:` then `self._uniques.add(ts)` *and* `self._sample.add(float(ts))` — two scalar sketch calls per row, when both classes have batch entry points right there. |
| `datetime.py:76` | `for i, ts in enumerate(timestamps)` to build a validity mask that `np.isfinite` plus a range compare would produce in one pass. |
| `datetime.py:120` | `[datetime.fromtimestamp(ts) for ts in ts_seconds]` — one Python `datetime` object per row, allocated and discarded. |
| `datetime.py:123` | `for dt in datetimes:` to tally hour/weekday patterns, which `Series.dt.hour.value_counts()` does vectorised. |

Add `consume.py:162`, which still parses with `format="mixed"`, and the datetime
path is the least optimised in the library while being the most expensive per
value. Entirely fixable with NumPy and pandas `.dt` accessors — no new
algorithms, no Rust. **Best value per hour on the whole list.**

---

## Revised roadmap

### Phase 0 — Trust ✅ complete

Six bugs fixed, oracle at 578 lines with 51 passing tests and zero xfails,
running in CI on every PR.

### Phase 1 — Finish free performance (1–2 weekends)

In measured value order:

1. **Gate Misra-Gries** off high-cardinality numeric columns — 35% of the numeric
   path for output that is discarded.
2. **KMV pre-filter** — 8.8×, three lines, identical estimates.
3. **Vectorise the datetime accumulator's four loops** — the most expensive
   column kind, the least optimised code.
4. **Drop `format="mixed"`** for try-explicit-formats-first on a 200-row sample.
5. **Vectorised Algorithm L** — 4.9×, bit-identical.
6. **`np.diff` for monotonicity** — 61×.
7. **Share the reservoir with `OutlierDetector`** instead of keeping a second 10k
   one per numeric column.

**Exit:** `mixed` at 1M rows ≥2.5× faster than 0.0.21 with byte-identical report
output, and the accuracy oracle passing unchanged.

### Phase 2 — Publish (1 weekend)

Two numbers worth leading with: **9.1×** faster than ydata-profiling and **13×**
less marginal memory, plus a version-over-version story. Put
`end_to_end.py --markdown` output in the README and `docs/benchmarks.md`, and
write post #1 — now stronger than planned, because you can end on the fix rather
than the finding.

**Exit:** a benchmarks page with the environment block, one post live, and a
comment on ydata-profiling
[#1129](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1129)
linking your polars docs.

### Phase 3 — Native core, reordered (3–5 weekends)

**KMV first**, not moments. Then `hash_arrow_utf8` wired to polars' Arrow buffers
for the categorical path. Moments last, as the demo it turned out to be.

Prerequisites unchanged: properties instead of private attribute reads in the
engine (`engine.py:371`, `render/html.py:80,100`); a `kind` tag instead of
`isinstance` dispatch (`consume_polars.py:170,193,200`); `__reduce__` on the
native types for checkpointing (`engine.py:377`).

Distribution decisions unchanged: separate `pysuricata-core` maturin wheel joined
by a `pysuricata[fast]` extra, abi3-py310, and the pure-Python path kept forever
as the reference the oracle diffs against.

**Exit:** `pip install pysuricata[fast]` works across the platform matrix and the
oracle passes identically with both backends.

### Phase 4 — Report v2, reframed (3–4 weekends)

Content unchanged, but go in knowing render is **3%** of wall clock: this is a
size, shareability and theming project.

Report is 1,180,196 bytes for 891 rows × 12 columns: **592 KB base64 PNGs**,
210 KB CSS, 182 KB inline SVG, 139 KB markup, 71 KB JS, and no JSON payload.

1. Drop the base64 PNGs for inline SVG. −592 KB, one afternoon.
2. Emit `<script type="application/json">` — `report.py::_build_summary` already
   produces the JSON-safe mapping; nothing in `render/` imports `json` today.
3. Stop pre-rendering six SVGs per column (`card_config.py:23`).
4. Real `prefers-color-scheme` theming; collapse the paired
   `--x-light`/`--x-dark` tokens; sweep the ~310 stray hex values.
5. Chart colours in CSS, not baked into `fill=` in `histogram_svg.py:33-40`.
6. Replace the sequential `str.replace` templating in `render/html.py:301-339`.

**Constraint:** the self-download button (`functionality.js:25-65`) re-serialises
the live DOM, finding styles by regex. Any restructuring breaks it *silently*.
Write a download-and-reopen test first.

**Exit:** the Titanic report under 250 KB.

### Phase 5 — Differentiate (ongoing)

`pysuricata check` as a CI gate; `compare(df_a, df_b)` for drift; direct
Arrow/Parquet/DuckDB input. Column-level threading is still blocked by the
reservoir's use of global `np.random` — the save/restore fix solved the
user-visible symptom, not this. Per-accumulator `Generator` is still the
prerequisite.

---

## Unchanged from v1

The market analysis, positioning and writing plan have not changed materially.
Summary, with the parts that got *stronger*:

**Positioning.** *The data profiler that fits in CI.* Single-pass streaming
algorithms, bounded memory regardless of dataset size, pandas and polars native,
four dependencies, one self-contained HTML file — or JSON and an exit code.
Lead with memory, not speed: it follows from the architecture rather than from
tuning, the incumbent's most-reported failure is a MemoryError they document as
unfixed, and the gap held up under re-measurement at **13×**.

**Market.** ydata-profiling (13.7k stars, ~1.76M downloads/mo) renamed to
fg-data-profiling and transferred orgs; Great Expectations' OSS stewardship moved
to Fivetran; Soda Core relicensed to the Elastic License. Open gaps: bounded-memory
profiling (asked for since 2016), polars-native end-to-end (ydata #1129 open since
Oct 2022), profiler-as-CI-gate (every gate tool requires authoring expectations
first), PII detection in OSS (ydata's is paywalled), dataset comparison.

**Writing plan.** Eight posts, each publishable when the matching work lands.
Post #1 ("I wrote one test file and found six statistical bugs in my own
library") is now stronger, and has a twist it didn't have before: the reservoir
test you wrote caught a regression I was about to recommend to you.

Show HN only for the posts with a number in the title. Publish the failures, not
just the wins. Never compare a `minimal=True` incumbent run against a full
PySuricata run without labelling it.

**CS:APP mapping.** Unchanged, with one addition: the reservoir episode is a
clean chapter-5 example in Python — Algorithm L is asymptotically better
(O(k·ln(n/k)) draws instead of O(n)) and empirically 5× slower than the
vectorised form, because at k=20,000 and n=200,000 the asymptotics never get a
chance to matter and per-acceptance interpreter overhead dominates. Asymptotic
improvement and measured improvement are different claims.

---

## This weekend

1. `python -m benchmarks.proposed_kernels` — watch both drop-ins verify and time
   on your machine. Then paste the KMV pre-filter into `_batch_add_hashes`.
2. Gate Misra-Gries. One condition removes 35% of the numeric accumulator.
3. Vectorise `DatetimeAccumulator.update`. Four Python loops, 4.7 µs/value.
4. Re-run the A/B at 1M rows and put the number in the README. You are sitting on
   9.1× and 13× and still have not told anyone.
5. Write post #1.
