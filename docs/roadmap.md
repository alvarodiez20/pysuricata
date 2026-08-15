# PySuricata roadmap v6 — re-audit at 0.0.38

Supersedes v5 (0.0.31). Two phases closed since: **all twelve user-experience
findings are addressed**, and `pysuricata check` — the item v5 called the
differentiator — has shipped. The ranked list below therefore contains, for the
first time, no item that is purely about the library's surface.

| | |
|---|---|
| 0.0.16 → 0.0.31 | **2.48×** (mixed 200k × 14, round-robin, best of 5) |
| `NumericAccumulator.update` | **152 ns/value**, from 828 at 0.0.16 |
| Phase 1 items | **7 / 7** closed |
| UX findings | **12 / 12** closed |
| Tests | **1,089**, from 770 at 0.0.26 |

The headline ratio is unchanged because it has **not been re-measured since
0.0.31**. Nothing since then touched a hot path — the work was contracts,
presentation and a new command — so the number is stale rather than wrong. It
should still be re-run before anything is published, in one round-robin, per the
rule below.

## What landed since v5

| | Verified in the source |
|---|---|
| **#84** UX-1, UX-2, UX-9, UX-10, UX-12 | The unique-*ratio* arm is gone, replaced by a cardinality ceiling at `_MAX_CATEGORICAL_LEVELS = 50` in `compute/processing/inference.py`. `py.typed`, `__all__`, a one-line `__repr__`, one exception hierarchy, `profile("data.csv")`. |
| **#86** UX-2, UX-3 | `render/identifier.py` — a column that is monotonic, integer-like and has a KMV estimate equal to the row count renders as **Identifier** rather than as a number with a mean. `render/triage.py` — a "needs attention" block built from the quality chips that were already computed, with chip filtering. |
| **#87** UX-4, UX-6, UX-11 | The log-scale chip now drives the chart default. `schema_version` on the payload. Keyword options and `fast`/`thorough` presets. Numeric `top_values` reach `summarize()`, with `None` meaning *not tracked* — distinct from an empty list. |
| **#88** UX-7 | `pysuricata/progress.py`. `True`, `"auto"`, or a callable; everything on stderr, nothing on stdout, asserted in all four modes; an ETA only when the row total is knowable. |
| **#90** UX-5 | `pysuricata check` with an exit code. See below. |

`check` is worth describing in more detail, because the interesting part is not
the command.

The first version gated on the distinct count and failed its own test:
appending rows doubles a continuous column's distinct count. Gating on the
distinct *rate* instead fails the mirror case — a three-level enum keeps its
count and halves its rate. Neither is stable alone; requiring **both** to move
is, because growth moves exactly one of them. The cost of that rule is in the
docstring and pinned by a test rather than left to be discovered: while the row
count is also moving a lot, a small change in levels is not reported.

The same discipline runs through the rest of the defaults — distribution drift
measured in σ rather than percent, a cardinality threshold an order of magnitude
above the KMV error, a warning printed when a threshold is set inside the noise
floor. A gate that fires on sketch noise gets switched off within a week, which
would be a worse outcome than not shipping one.

---

## The measurement rules, unchanged

Three of these were each nearly a published claim. They are restated because
every one of them cost a real retraction.

1. **A ratio is only quotable when both sides were measured in the same
   round-robin**, on the same machine, within the same run. `benchmarks/end_to_end.py`
   and `benchmarks/versions.py` now interleave every tool and version across
   rounds and label anything under three rounds *Not quotable*, so this is
   enforced rather than remembered.
2. **`cProfile` over-weights kernels that make many small calls.** It ranked the
   reservoir at ~30% of self time when replacing it with a 5× faster one moved
   wall clock by 4%.
3. **A kernel benchmark only measures the call sites it calls.** Holding
   `KMV._values` as an array won its own benchmark by 2.7× and lost 35% end to
   end, because the benchmark never touched the scalar insert path that
   categorical columns hammer.

The incumbent comparison — 13.3× against ydata-profiling — has not been
re-measured since 0.0.26 and is stale in the understated direction.

---

## What to do next

**1 · Publish (1 weekend).** This is now the highest-value item and it has no
dependencies left. The library is at 2.48×, the docs are checked in CI, the UX
findings are closed and there is a differentiating command. Re-run the incumbent
round-robin first so the number quoted is one measurement rather than two — see
#38.

**2 · Prove the memory claim, then ship the budget (#92, then #79).** "Bounded
memory, so it fits in CI" is the reason to prefer this over a profiler that
loads the frame, and it has never been measured on a constrained runner. #92
runs `check` against a file larger than RAM in a 512 MB container; #68 is the
same container for the benchmark suite, and they should be done in one sitting.

Only then #79. The model in `docs/adr/memory-budget.md` is fitted on **numeric
columns only**, and a text-heavy frame is the shape most likely to be large —
so the budget would be optimistic exactly where it matters. It must be a
planner that derives settings, never a cap that raises `MemoryError`, and it
must report the accuracy consequence of what it chose: quantile error is
`1/√k`, ±0.7% at k=20,000 and ±3.2% at k=1,000.

**3 · Finish the contract (#43).** `schema_version` exists and `check` now
depends on it, which raises the cost of drift. Still missing: a documented
schema page, a stability policy where a consumer will read it, and the general
form of the test that #87 wrote for one field — *every* statistic in the HTML is
present in the JSON. That test is what stops the next `rows` → `rows_est`.

**4 · The native core (3–5 weekends).** Unchanged in ordering: **KMV first,
moments last**, because moments were 1.3–5% of the numeric path while KMV was
half of it. The crate is vendored with 20 passing tests, nothing imports it, and
there is still no `[fast]` extra. #64 prepares the accumulator boundary. The
32 KiB tiling result — 19.1 → 11.2 ns/row, after the first naive version came in
*slower* than NumPy — is the best technical post in the project.

**5 · Then the reporting features that now have foundations.** `compare()` (#65)
should be built on `check.compare` rather than beside it, so the diff and the
gate cannot disagree about what counts as a change. Freshness gating (#91)
catches the most common scheduled-pipeline failure, which is not drifted data
but yesterday's data served again.

### Open correctness items, none blocking

#67 (`NumericAccumulator.merge()` drops the top-k counters), #61 (forced and
reclassified columns revert to default sketch sizes), #60 (the outlier detector
keeps a second reservoir), #36 (a duplicated missing-cells pass), #41 (report
uncertainty rather than presenting estimates as facts), #89 (a config value that
fails validation is discarded rather than reported).

#89 deserves a note: `_to_engine_config` catches `Exception` and falls back to a
partial mapping, so a value that fails validation does not produce an error — it
produces a **different configuration**, silently, and the fallback never sets
`columns`, the correlation options, `progress`, or the boolean-detection
options. It was found because a misplaced validation rule made `progress=`
vanish for callables while working for `True`.

The report's own presentation — theming and chart colours (#40), and cutting the
example report below 250 KB (#39) — is tracked but out of scope for this list.

---

*Measurements: 0.0.31 at `4831a36`, plus 0.0.27, 0.0.26, 0.0.21 and 0.0.16 each
on `sys.path` in its own interpreter; five round-robin rounds, best of five per
version; `mixed` suite at 200,000 × 14, seed 0; 2-core x86-64 Linux container.
Accumulator timings over 1,000,000 float64 values in 20 chunks of 50,000. Memory
figures from `getrusage` peak RSS in fresh subprocesses at 0.0.27. Absolute
times are not comparable across sessions; ratios within a single round-robin
are.*
