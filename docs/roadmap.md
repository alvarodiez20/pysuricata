# PySuricata roadmap v4 — re-audit at 0.0.30

Supersedes v3 (0.0.26). Phase 1 is complete: all seven items landed, plus eight
bugs that only surfaced while landing them.

## Two environments, and why it matters

v3's numbers came from a 2-core x86-64 Linux container. Everything below marked
**(dev)** was re-measured on the development machine instead — Python 3.13.6,
arm64 Darwin, 63.8 GB/s sequential read — because that is where the work was
done. The two are not interchangeable, and the difference is not a detail:

| | reference container | dev machine |
|---|---:|---:|
| mixed 200k × 14 at 0.0.26 | 4,768 ms | 1,517 ms |

A change worth 1.33× on the container measured 1.13× here. **Nothing in this
document may be published until it is re-run on the reference environment**
(issue #68); the incumbent comparison in particular has *not* been re-measured
since 0.0.26 and its 13.3× is now stale in the understated direction.

---

## Where things stand

| | |
|---|---|
| Original audit items | **11 of 11 closed** |
| Phase 1 performance | **7 of 7 closed** |
| mixed 200k × 14, 0.0.26 → 0.0.30 **(dev)** | **2.54×** (1,517 ms → 597 ms) |
| `NumericAccumulator.update` **(dev)** | 1,278 → **83 ns/value**, 15.4× |
| Datetime column, 200k rows **(dev)** | ~308 ms → **33 ms**, 9.3× |
| vs ydata-profiling | **stale** — 13.3× measured at 0.0.26, not re-run |
| Peak RSS | **stale** — 175 MB vs 457 MB at 0.0.26, not re-run |

### What actually happened

v3 projected 1.88× from three changes. The three changes landed and the
measured result was **2.54×** — but almost none of the difference came from
where v3 said it would.

The chunk-size cap was projected at 1.33× and delivered 1.13×, because fixing
KMV flattened the curve that made big chunks expensive. The KMV pre-filter was
projected at 8.7× on its kernel and delivered 3× on the same kernel, because
half the proposal turned out to be a regression. Meanwhile the datetime
accumulator — listed under "loose ends" — turned out to be worth more than all
three Phase 1 items combined.

The pattern worth keeping: **every one of these changes was found by measuring,
and every projection made without measuring was wrong.**

---

## Scorecard

### Phase 1 performance — 7 of 7 closed

| Item | Result |
|---|---|
| **Date sniff** | Explicit formats first, `format="mixed"` demoted to a last-resort probe. Categorical columns 45% faster end to end. |
| **Reservoir per-acceptance loop** | Bulk scheduler (`_SCHEDULE_BLOCK`, cumsum of logs). 154 → 57 ns/value. |
| **Misra-Gries gate** | Gated on the KMV estimate, latching off and discarding. Removed a third of the numeric accumulator *and* a misleading table. |
| **KMV pre-filter** | Threshold reject before the sort. 51 → 17 ns/value on the kernel; estimates provably identical. |
| **Chunk-size default** | 200,000 → 50,000. 1.13× **(dev)** once KMV was fixed. |
| **Datetime accumulator** | Fully vectorised. 308 → 33 ms/column, **9.3×** — the single biggest win of the phase. |
| **`np.diff` monotonicity / outlier sample** | 45.2 → 0.6 ns/value in situ. The second reservoir was deleted rather than shared: nothing ever read it. |

### Bugs found while landing Phase 1

None of these were on the roadmap. All were found by writing the test *before*
believing the change was safe.

| Bug | Severity |
|---|---|
| `finalize()` fabricated "common values" — a continuous column reported values that occurred once as occurring *sample_scale* times, and the fallback overrode the exact counters on columns with <5 distinct values | **High**: invented data rendered as measured data |
| Reported min/max came from the 20,000-value reservoir while the exact extremes sat in the tracker beside them | **High**: two figures on one card that could disagree |
| Hour/weekday tallies used the machine's local timezone against UTC-stored timestamps | **High**: same file, different report in London and Tokyo |
| `update()` raised `ValueError` on numpy arrays and pandas Series in three accumulators (`if not arr`) | Medium: the categorical path converts to a Series on the next line |
| `NumericAccumulator.reset()` raised `AttributeError` on the **default** config | Medium |
| One out-of-range timestamp discarded a whole chunk's temporal patterns | Medium |
| `datetime64[ns] → [D]` overflows at the window floor: 1677-09-21 reported as day *+106750* | Medium, latent |
| The sample-preview table drew from the global RNG and ignored `random_seed` | Low |

---

## Where the time is now (dev)

1M values in 5 chunks, wall clock, GC disabled, profiler off.
`NumericAccumulator.update` totals **83 ns/value**, was 1,278 at 0.0.26.

| Component | ns/value | share | 0.0.26 |
|---|---:|---:|---:|
| StreamingHistogram | 18.4 | 22.2% | 29 |
| KMV distinct count | 18.2 | 21.9% | 668 |
| ReservoirSampler | 17.7 | 21.3% | 57 |
| ExtremeTracker | 16.2 | 19.5% | 9 |
| StreamingMoments | 7.6 | 9.1% | 18 |
| MonotonicityDetector | 0.6 | 0.7% | 89 |
| Misra-Gries top-k | *gated off* | — | 434 |

Two things changed shape entirely. **KMV is no longer the story** — it went from
52.3% to 21.9% and is now merely one of four components of similar size. And
**there is no longer a dominant kernel at all**: the top four are within 2
ns/value of each other. That is what the end of easy wins looks like. Further
Python-level work here is sharpening a knife that is already sharp; the next
real step change is the native core, and it now has a much less impressive
baseline to beat.

`ExtremeTracker` rising from 9 to 16.2 ns/value is not a regression in the
tracker — it is the same absolute cost against a total that shrank 15×.

---

## Loose ends

| Item | Detail |
|---|---|
| **`merge()` drops the top-k counters** (#67) | It merges moments, extremes and byte counts, rebuilds uniques and the reservoir from the other side's sample, and never touches `_topk`. Latent — nothing calls `merge()` today — but it goes live the moment column-level threading does. The same method reconstructs `_uniques` from a sample rather than merging the sketches, which underestimates for the same reason. |
| **Forced and reclassified columns lose their config** (#61) | The adapters replace accumulators with `NumericAccumulator(col_name)` and no config, so `uniques_k` and `topk_k` silently revert to defaults on exactly those columns. Same class as the dead-options bug. |
| **`summarize()` omits numeric top values** (#59) | The HTML shows a table the programmatic API cannot see. Now interacts with the top-k gate: the payload needs to distinguish *not tracked* from *tracked and empty*. |
| **KMV `_values` as an array — rejected** | 2.7× on its own kernel, 35% slower end to end. `_add_hash_to_kmv` runs per distinct value on categorical columns, where `np.insert` allocates and copies what `list.insert` memmoves. Recorded in `proposed_kernels.py` so it is not re-proposed. |

---

## Revised roadmap

### Phase 0 — Trust ✅ complete

### Phase 1 — Performance ✅ complete

2.54× **(dev)**, oracle green, and a test pinning the chunk-size band. The exit
criterion was "mixed 200k × 14 under 2.6 s by default" on the container; the
dev machine is at 597 ms and the container has not been re-run.

### Phase 2 — Publish (next, blocked on measurement)

**#68 first, then #38.** Every headline figure is now stale, all of them
understated. Re-run `end_to_end.py --markdown` and `kernels.py` on the reference
container, record the environment block alongside, and extend the version curve
through 0.0.30. Only then does anything go in the README.

Two anecdotes are worth writing up, both now with better endings than v3 had:

- *"Honouring a config option made my library slower."* The blend hid a bad
  default for months; fixing the option exposed it.
- *"My profiler showed you the most common values. They all occurred once."*
  And the follow-up nobody expects: the fallback underneath was **multiplying
  those counts by the sampling ratio**, so singletons were reported as having
  occurred ten times. A measurement bug, a UX bug, and a fabrication, in one
  function.

**Exit:** benchmarks page with the environment block, one post live, a comment
on ydata-profiling [#1129](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1129).

### Phase 3 — Native core (3–5 weekends)

Datetime is done, so this phase is just the crate now. **KMV first, moments
last** still holds by share — but the argument for the whole phase is weaker
than it was, because the Python path is 15× faster than when the crate was
proposed and no single kernel dominates any more. Do the prerequisites (#64)
first; they are worth having regardless, and they are what makes the crate
swappable rather than bolted on.

**Exit:** `pip install pysuricata[fast]` working across the platform matrix with
the oracle passing on both backends.

### Phase 4 — Report v2 (3–4 weekends)

Unchanged. Still 1.18 MB with ~592 KB of base64 PNGs and no JSON payload. Render
is now a *larger* share of wall clock than it was, simply because compute
shrank 2.5× underneath it — worth re-measuring before assuming it is still ~3%.

**Exit:** Titanic report under 250 KB.

### Phase 5 — Differentiate (ongoing)

`pysuricata check` as a CI gate (#42); `compare(df_a, df_b)` for drift (#65);
direct Arrow/Parquet/DuckDB input (#66). **Column-level threading is unblocked**
— every sketch now owns its generator, seeded per column from the run seed, so
concurrent accumulators are reproducible. Fix #67 before threading, since a
broken `merge()` is exactly what threading will start exercising.

---

## Unchanged from earlier versions

**Positioning.** *The data profiler that fits in CI.* Single-pass streaming
algorithms, bounded memory regardless of dataset size, pandas and polars native,
four dependencies, one self-contained HTML file — or JSON and an exit code. Lead
with memory, not speed: it follows from the architecture rather than from tuning,
and the incumbent's most-reported failure is a MemoryError they document as
unfixed. Note the memory claim also needs re-measuring (#68) — nothing in Phase 1
was expected to change it, but "was not expected to" is not a measurement.

**Market.** ydata-profiling (13.7k stars, ~1.76M downloads/mo) renamed to
fg-data-profiling and transferred orgs; Great Expectations' OSS stewardship moved
to Fivetran; Soda Core relicensed to the Elastic License. Open gaps:
bounded-memory profiling (asked for since 2016), polars-native end-to-end (ydata
#1129 open since Oct 2022), profiler-as-CI-gate, PII detection in OSS, dataset
comparison.

**Measurement discipline.** Three lessons now, all learned here.

1. `cProfile` charges per Python call and over-weights kernels that make many
   small ones. Confirm rankings against wall clock with the profiler off.
2. When checking whether a value reaches the report, search for the **formatted**
   string. An audit wrongly concluded top-k output was discarded because it
   searched for `4248` in a report that renders `4,248`.
3. **A kernel benchmark only measures the call sites it calls.** The KMV
   array-backed `_values` won its benchmark by 2.7× and lost 35% end to end,
   because the benchmark never touched the scalar insert path that categorical
   columns hammer. Confirm every kernel win against end-to-end wall clock before
   adopting it.
