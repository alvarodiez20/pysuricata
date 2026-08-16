# PySuricata roadmap v7 — re-audit at 0.0.50

Supersedes v6 (0.0.38). Two things changed shape since.

**The correctness backlog is empty.** Every item v6 listed as open — #67, #61,
#60, #36, #41, #89 — is closed, along with #43, #64, #65, #68, #91 and #105.
There is no longer a "none blocking" section, because there is nothing in it.

**The report's presentation is now the main body of work**, not a footnote. v6
tracked it in one line at the end as out of scope. It is #110–#125: a
fourteen-issue migration with its own testing strategy, and it is the largest
open item by some distance.

| | |
|---|---|
| 0.0.16 → 0.0.31 | **2.48×** (mixed 200k × 14, round-robin, best of 5) |
| `NumericAccumulator.update` | **152 ns/value**, from 828 at 0.0.16 |
| Report size | **543,577 B**, from 1,110,756 — measured in one run at 0.0.49 |
| Tests | **1,473**, from 1,089 at 0.0.38 |
| Open correctness items | **0** |

The headline ratio is unchanged because it has **still not been re-measured
since 0.0.31**. Nothing since then has touched a hot path — the work has been
contracts, comparison, streaming readers and presentation — so the number is
stale rather than wrong. Re-run it before anything is published, per the rules
below.

## What landed since v6

| | Verified in the source |
|---|---|
| **#93, #94, #95** | `merge()` merges rather than discarding; the `summarize()` payload is a versioned contract that gates on stale data; text memory is flat in rows. |
| **#96** | `sources.py` — Arrow, Parquet and DuckDB read in batches without materialising. A Codecov report caught the DuckDB branch being unreachable because a relation also exports `__arrow_c_stream__`. |
| **#105, #64** | `ComputeOptions.checkpoint` groups five settings behind one name. `accumulators/protocols.py` describes the surface the engine may use, tested against a fake accumulator that **inherits nothing** — as close as Python gets to proving the boundary would hold for a type from another language. |
| **#65** | `compare(a, b)`, built **on** the gate rather than beside it, so a diff and a gate cannot disagree about what counts as a change. |
| **#41** | The distinct count no longer exceeds the row count, and no longer claims to be exact. This also fixed a second bug: the identifier check required 0.98 of the row count, *inside* the sketch's own 2.2% error, so a perfect key was profiled as a measurement with a mean. |
| **#110** | Phase 1 of the redesign — tokens, typography, structural motif. |
| **logo** | 592 KB of base64 PNG replaced by a 10.8 KB inline SVG: **51% of the report**, for a mark drawn at 30 CSS pixels. |

## The measurement rules, unchanged

Four now. Each cost a real retraction or a near miss.

1. **A ratio is only quotable when both sides were measured in the same
   round-robin**, on the same machine, within the same run. `benchmarks/end_to_end.py`
   and `benchmarks/versions.py` interleave every tool and version across rounds
   and label anything under three rounds *Not quotable*.
2. **`cProfile` over-weights kernels that make many small calls.** It ranked the
   reservoir at ~30% of self time when replacing it with a 5× faster one moved
   wall clock by 4%.
3. **A kernel benchmark only measures the call sites it calls.** Holding
   `KMV._values` as an array won its own benchmark by 2.7× and lost 35% end to
   end, because the benchmark never touched the scalar insert path.
4. **A guard is worth only as much as the guarantee that what it reads is what
   runs.** New at 0.0.50. `test_contrast.py` asserted the axis colour cleared
   3:1 and passed for as long as it existed, while a second stylesheet
   redefined that token and the report drew a different colour entirely. The
   test was reading the definition; the page was rendering the override. Found
   by inspecting computed styles in a browser, not by reading CSS.

The incumbent comparison — 13.3× against ydata-profiling — has not been
re-measured since 0.0.26 and is stale in the understated direction.

---

## What to do next

**1 · Finish the redesign (#110–#125).** The largest open item, and the one with
the most leverage: the library's output *is* the product, and every claim about
correctness is read through a page that currently looks like a template.

Phase 1 has landed. The order after it is fixed by dependency — the header
(#111) and summary (#112) sit on the tokens, the cards (#114–#118) sit on both.
Three things are worth holding onto while it runs:

- **The facts must not change while the presentation does.** `scripts/report_fingerprint.py`
  reduces a report to the set of numbers it asserts and discards everything
  about how they look. Phase 1 and the logo change are both byte-identical
  under it, 598 facts. #123 makes that a test rather than a habit.
- **#118 is a data issue wearing a presentation costume.** Three real bugs found
  while designing. It should be treated as correctness work and not deferred
  behind the visual phases.
- **The compatibility shim is scaffolding**, and scaffolding gets left up. 195
  colour literals still sit in components that phases 2–9 own; the shim comes
  out in #122, which is what makes that count reach zero.

**2 · Publish (#38, one weekend).** Unchanged from v6 in everything except its
rank, and it drops one place only because the report is now the thing a reader
sees first. Re-run the incumbent round-robin so the quoted number is one
measurement rather than two.

**3 · Prove the memory claim, then ship the budget (#92, then #79).** "Bounded
memory, so it fits in CI" is the reason to prefer this over a profiler that
loads the frame, and it has still never been measured on a constrained runner.

Only then #79. The model in `docs/adr/memory-budget.md` is fitted on **numeric
columns only**, and a text-heavy frame is the shape most likely to be large — so
the budget would be optimistic exactly where it matters. It must be a planner
that derives settings, never a cap that raises `MemoryError`, and it must report
the accuracy consequence of what it chose: quantile error is `1/√k`, ±0.7% at
k=20,000 and ±3.2% at k=1,000.

**4 · The native core (#44, 3–5 weekends).** Unchanged in ordering: **KMV first,
moments last**, because moments were 1.3–5% of the numeric path while KMV was
half of it. The crate is vendored with 20 passing tests, nothing imports it, and
there is still no `[fast]` extra. #64 has since prepared the boundary, so the
first real obstacle is gone.

**5 · The rest of the report's weight (#39).** The logo was the first item on
that list and the largest; it is done, and the report is at 543 KB against a
250 KB target. What remains is structural — a JSON payload block, and not
pre-rendering six histograms per column. Both are best done *after* the
redesign, since the redesign rewrites the markup they would restructure.

### One piece of infrastructure

**#125 — CI has never run on `main`.** `ci.yml` triggers on `pull_request` only.
Every check the project relies on has only ever run against a proposed merge,
never against the result. The fix needs care rather than effort: `version-check`
and `changelog-check` compare against `origin/main` and would fail on every push
to `main` by construction, so the push trigger has to be scoped to `lint`,
`test` and `accuracy`.

---

*Measurements: performance figures unchanged from v6 — 0.0.31 at `4831a36`, plus
0.0.27, 0.0.26, 0.0.21 and 0.0.16 each on `sys.path` in its own interpreter;
five round-robin rounds, best of five per version; `mixed` suite at 200,000 × 14,
seed 0; 2-core x86-64 Linux container. Report sizes measured at 0.0.49 on an
891 × 8 frame, before and after in a single run against a `main` worktree. Test
count from `pytest -m "not benchmark"` at 0.0.50. Absolute times are not
comparable across sessions; ratios within a single round-robin are.*
