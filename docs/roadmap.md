# PySuricata roadmap v8 — re-audit at 0.0.62

Supersedes v7 (0.0.50). Two things changed shape since.

**The redesign is finished as a body of work.** v7 called #110–#125 "the largest
open item by some distance." Fifteen of its sixteen issues are closed. What is
left is not more design: it is an accessibility close-out (#122), a `compare()`
view (#121), and turning the acceptance criteria into tests (#124).

**A new class of defect surfaced, and it is the interesting finding of this
audit.** `+ add a note` in the summary section did nothing for eleven versions.
Not slowly, not partially — nothing, with a clean console. The redesign renamed
the block from `.description-value` to `.description-block`; the JavaScript
still looked for the old name; and every entry point in that file guards on a
null container and returns quietly. **A test suite of 1,735 tests did not
notice, because not one of them asserted that a selector resolves.**

That is not a bug in a control. It is a gap in what the suite was capable of
seeing, and it was created by exactly the migration v7 was most confident
about — the one with a fingerprint proving the *facts* never changed. The facts
never did. Nothing in the fingerprint has an opinion about whether a button
works.

| | |
|---|---|
| 0.0.16 → 0.0.31 | **2.48×** (mixed 200k × 14, round-robin, best of 5) |
| `NumericAccumulator.update` | **152 ns/value**, from 828 at 0.0.16 |
| Report size, 891 × 12 | **600,028 B**, from 629,117 at 0.0.50 — **−4.6%** |
| Report size, 891 × 8 | **538,660 B**, from 551,198 at 0.0.50 — **−2.3%** |
| Tests | **1,759**, from 1,473 at 0.0.50 |
| Open correctness items | **1** (#139, reopened — see below) |

The headline ratio is **still** unchanged and **still** has not been re-measured
since 0.0.31. v7 said re-run it before publishing anything; that remains true
and is now two audits old. Nothing since has touched a hot path, so it is stale
rather than wrong — but "stale rather than wrong" is a claim that decays.

## What landed since v7

| | Verified in the source |
|---|---|
| **#111–#120** | Header, summary, sample, all four card kinds, correlations, missing values. Every phase byte-identical under the fingerprint except the two that were meant to change facts. |
| **#123** | The invariance harness: fingerprint (598 facts), golden `summarize()` payload on three frames, fact coverage (154 of 154 statistics reach the page). Each guard verified to **fail** on a real regression rather than assumed to work. |
| **#103, #104, #41** | Sample open by default; the donut replaced by a stacked composition bar; the distinct count no longer exceeds the row count or claims to be exact. |
| **logo** | 592 KB of base64 PNG replaced by a 10.8 KB inline SVG — 51% of the report, for a mark drawn at 30 CSS pixels. |
| **the note button** | Fixed, and generalised: every `getElementById` and class selector in the bundled JS is now checked against real rendered markup. |

## The measurement rules

Five now. Each cost a real retraction or a near miss.

1. **A ratio is only quotable when both sides were measured in the same
   round-robin**, on the same machine, within the same run.
2. **`cProfile` over-weights kernels that make many small calls.** It ranked the
   reservoir at ~30% of self time when replacing it with a 5× faster one moved
   wall clock by 4%.
3. **A benchmark only measures the call sites it calls.** Holding `KMV._values`
   as an array won its own benchmark by 2.7× and lost 35% end to end.
4. **A guard is worth only as much as the guarantee that what it reads is what
   runs.** `test_contrast.py` asserted the axis colour cleared 3:1 and passed
   for as long as it existed, while a second stylesheet redefined that token and
   the page drew a different colour.
5. **A check over rendered output is only as good as the markup the fixture
   reaches.** New at 0.0.62, and rule 3 restated for the render layer. Writing
   the selector check produced two confident false positives:

   | reported dead | why it wasn't |
   |---|---|
   | the numeric Linear/Log toggle | `[1.0, 2, 3, 4, 5] * 40` has five distinct values, so it profiles as **categorical** — the fixture had no numeric card at all |
   | the flag filter | a frame with no quality problems renders no `.needs-attention` block |

   Both controls work; both were checked in a browser before anything was
   changed. A fixture that misses a branch does not report "unknown", it reports
   "absent", and absent reads as broken.

**The report-size series is not reproducible, and that is a sixth rule waiting
to be written.** v7 quotes 543,577 B at 0.0.49 on "an 891 × 8 frame." The frame
was never pinned, so the figure cannot be reproduced — the closest
reconstruction gives 551,198 B at the adjacent commit. The numbers in this
document's table were measured across three interleaved rounds on two pinned
frames and are internally consistent; they are **not** comparable to v7's. Pin
the frames the way `tests/fixtures/` pins the invariance inputs.

---

## What to do next

**1 · Close out the redesign (#122, #124, #121).** In that order.

**#122 is the one that matters**, and it carries a decision that has been open
across six separate hand-offs and is now blocking: **the height criteria on
#112 (599px measured against a ≤560 target) and #114 (775px against ≤600)**.

#114's is not a matter of effort. It is a conflict between two of its own
acceptance criteria: six controls at the required 44×44 wrap to 148px, and
fourteen stats in the specified two-column mobile grid are seven rows. One
decision resolves both, and it can be applied retroactively — but it has to be
made rather than built past, which is what happened five times.

#122 also removes the compatibility shim. That is scaffolding, and scaffolding
gets left up: 195 colour literals still sit in components the visual phases own,
and the shim coming out is what makes that count reach zero.

**#124 should absorb the selector check.** The acceptance criteria it exists to
run are mostly geometric — heights, contrast, target sizes — and the note button
is the proof that geometry is not the only thing that silently stops being true.

**2 · #139, reopened.** Per-column per-chunk missing counts are never produced.
It was closed by a **keyword in a PR body, not by a fix** — the write-up of the
gap sat in the commit message that GitHub read as a closing reference. Three
verified findings stand: the renderer reads `chunk_metadata` off an object that
does not carry it, only the numeric accumulator tracks it at all, and
`mark_chunk_boundary()` is called only from `finalize()`, so there is exactly
one boundary no matter how many chunks ran.

The strip renderer is written and tested in isolation, and
`TestTheDataGapIsRealAndDetected` fails the day the engine starts marking
boundaries — so the view turns on by someone noticing a red test.

**3 · Regenerate the example report.** `docs/assets/titanic_report.html` is
1,180,196 B and was last written by PR #23, at **0.0.17**. Every reader of the
documentation is looking at a pre-redesign report roughly twice the size of what
the library now emits — including the base64 logo the release notes say was
removed. This is the cheapest credibility fix available and it is currently the
largest gap between what the project claims and what a visitor sees.

**4 · Publish (#38, one weekend).** Unchanged except that it now has a
precondition it did not have in v7: re-run the incumbent round-robin *and* the
version series, so the two headline numbers are each one measurement rather than
two. The 13.3× against ydata-profiling has not been re-measured since 0.0.26.

**5 · Prove the memory claim, then ship the budget (#92, then #79).** "Bounded
memory, so it fits in CI" is the reason to prefer this over a profiler that
loads the frame, and it has still never been measured on a constrained runner.

Only then #79. The model in `docs/adr/memory-budget.md` is fitted on **numeric
columns only**, and a text-heavy frame is the shape most likely to be large — so
the budget would be optimistic exactly where it matters. It must be a planner
that derives settings, never a cap that raises `MemoryError`, and it must report
the accuracy consequence of what it chose: quantile error is `1/√k`, ±0.7% at
k=20,000 and ±3.2% at k=1,000.

**6 · The native core (#44, 3–5 weekends).** Unchanged: **KMV first, moments
last**, because moments were 1.3–5% of the numeric path while KMV was half of
it. The crate is vendored with 20 passing tests, nothing imports it, and there
is still no `[fast]` extra.

**7 · The rest of the report's weight (#39).** The redesign took 891 × 12 from
629 KB to 600 KB — real, but a rounding error against the 250 KB target. The
logo was the last easy win. What remains is structural: a JSON payload block,
and not pre-rendering six histograms per column. Both are now unblocked, since
the markup they would restructure has stopped moving.

### One piece of infrastructure

**#125 — CI has never run on `main`.** `ci.yml` triggers on `pull_request` only,
so every check the project relies on has only ever run against a proposed merge,
never against the result. The fix needs care rather than effort: `version-check`
and `changelog-check` compare against `origin/main` and would fail on every push
to `main` by construction, so the push trigger has to be scoped to `lint`,
`test` and `accuracy`.

It has a second half found during the redesign: every workflow keys on
`pull_request: branches: [main]`, so a **stacked** PR gets zero checks. Sixteen
of the redesign PRs were stacked, and the ones that ran green did so only
because they were retargeted at `main` before merge.

---

*Measurements: performance figures unchanged from v6 — 0.0.31 at `4831a36`, plus
0.0.27, 0.0.26, 0.0.21 and 0.0.16 each on `sys.path` in its own interpreter;
five round-robin rounds, best of five per version; `mixed` suite at 200,000 × 14,
seed 0; 2-core x86-64 Linux container. **Report sizes re-measured for this
audit**: 0.0.50 at `70e9701` against 0.0.62 at `de36425`, each on `sys.path` in
its own interpreter, three interleaved rounds, byte length of the emitted HTML;
frames are `docs/assets/titanic.csv` (891 × 12) and a seeded 891 × 8 frame
covering all four card kinds. Output is deterministic — all three rounds agreed
exactly — so best-of is not meaningful here and the single value is quoted. Test
count from `pytest -m "not benchmark"` at 0.0.62. Absolute times are not
comparable across sessions; ratios within a single round-robin are.*
