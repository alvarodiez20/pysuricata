# PySuricata roadmap v5 — re-audit at 0.0.31

Supersedes v4 (0.0.30). Phase 1 is finished: all seven performance items are closed
and verified in the source at `4831a36`. **Not one of the twelve user-experience
findings has been addressed** — every one was re-tested against a live 0.0.31 import
while writing this — so for the first time the ranked list of next work contains no
compute item at the top.

| | |
|---|---|
| 0.0.16 → 0.0.31 | **2.48×** (mixed 200k × 14, round-robin, best of 5) |
| `NumericAccumulator.update` | **152 ns/value**, from 828 at 0.0.16 |
| Phase 1 items | **7 / 7** closed |
| UX findings open | **12 / 12** |

## What landed since v4

| Commit | Verified in the source |
|---|---|
| **#62** gate top-k, pre-filter KMV, fix the chunk-size default | `should_track_top_k()` at `accumulators/numeric.py:35`, switching `_track_top_k` off mid-stream once the KMV estimate says the table would be singletons; the merge path clears it too, so a chunked run cannot resurrect it. KMV admission threshold at `accumulators/sketches.py:203`, with the correct strict `<` and the monotonicity argument written into the docstring. |
| **#63** vectorise the datetime accumulator | Loop count in `accumulators/datetime.py` down from 41 to 36; the hot paths take arrays. This was the most expensive column kind. |
| **#69** finish Phase 1 and re-measure | Closes the last of the seven. See the correction below — the numbers recorded there were assembled across sessions. |
| **#70** six interactive figures, and the pre-commit hooks | `docs/javascripts/figures.js`, 66 lines, syncing with mkdocs-material's colour scheme rather than hard-coding a palette. First of the documentation plan to land. |

---

## Re-measured today — and the previous headline was wrong

v4 reported 2.27× and the note after #69 reported 3.56×. **Both were assembled from
measurements taken at different times**, on a container whose available CPU varies
between sessions. The table below comes from a single round-robin: five rounds, all
five versions measured in every round, best-of-five per version, each in its own
subprocess.

| version | ms | × vs 0.0.16 | accumulator ns/value |
|---|---:|---:|---:|
| 0.0.16 | 3,822.6 | 1.00× | 828.3 |
| 0.0.21 | 4,352.3 | **0.88×** | 951.2 |
| 0.0.26 | 3,874.6 | 0.99× | — |
| 0.0.27 | 2,361.1 | 1.62× | 203.8 |
| **0.0.31** | **1,542.5** | **2.48×** | **152.1** |

### Two claims to retract before anything is published

1. **0.0.21 was not 1.24× faster than 0.0.16.** On this workload, measured properly,
   it is **0.88×** — a regression — and the accumulator got slower too, 828 → 951 ns.
   The real inflection point is 0.0.27. If you publish a version-over-version curve
   that dip is in it, and it is better to explain it yourself than to have a reader
   find it.
2. **The headline is 2.48×, not 3.56×.** The larger number came from pairing a slow
   0.0.16 run with a fast 0.0.31 run taken at a different time. It was flattering and
   it was wrong.

### The rule this establishes

A ratio is only quotable when both sides were measured in the same round-robin, on
the same machine, within the same run. This is the third time in this audit that a
measurement artefact nearly became a published claim — the first was cProfile ranking
the reservoir at 30% of self time when replacing it moved wall clock by 4%, the
second was the dev-machine-versus-container gap in v4. **Bake the round-robin into
`benchmarks/` so the discipline is not a habit you have to remember.**

The incumbent comparison — 13.3× against ydata-profiling — has **not** been
re-measured since 0.0.26 and is stale in the understated direction, since two of the
three Phase 1 wins landed after it.

---

## The twelve UX findings — 0 of 12

Re-tested against a live 0.0.31 import, not read from the previous review.

| Check | Result at 0.0.31 |
|---|---|
| `summarize(df)["columns"]["age"]["type"]` | `categorical` (67 distinct integers in 20,000 rows) |
| `cid`, a monotonic unique key | numeric, with a mean |
| `pysuricata/py.typed` exists | `False` |
| `hasattr(pysuricata, "__all__")` | `False` |
| `ReportConfig is ProfileConfig` | `True` |
| `"schema_version" in summarize(df)` | `False` |
| `summarize(df)["columns"]["revenue"]["top_values"]` | key absent, while the HTML renders from the same accumulator |
| `profile("data.csv")` | `TypeError` (the CLI accepts the same path) |
| `ProfileConfig(chunk_size=50_000)` | `TypeError` |
| `profile(df, preset="fast")` | `TypeError` |
| `profile(df, progress=True)` | `TypeError` |
| `len(repr(report))` | 1,108,306 |
| stdout bytes during `summarize()` | **0** — already fixed |

### Five users, and where each one stops

**1 · The evaluator** — ten minutes, comparing tools. `age` comes back categorical and
`customer_id` gets a mean of 1e+04 with a flat histogram. Two of eight columns wrong,
with defaults, in the first thirty seconds. They do not file a bug; they close the
tab. *Unblocked by UX-1, UX-2.*

**2 · The analyst** — has a 60-column frame open. Gets 60 identically sized cards in
source order, no triage anywhere, although the quality chips already compute exactly
the signal needed. Then the `revenue` card detects *Log-scale?* and draws a linear
axis anyway. *Unblocked by UX-3, UX-4.*

**3 · The data engineer** — wants it in the pipeline. `profile` and `summarize` both
exit 0 regardless of what they found. Every existing gate tool makes you author
expectations first; a profiler can gate on shape drift with no configuration at all,
and nobody occupies that position. *Unblocked by UX-5.*

**4 · The integrator** — building something on top. No `schema_version`, and the
payload has already drifted once. No `py.typed`, so every annotation infers as `Any`.
`dir()` is half internal modules. Three exception types for the same user error.
*Unblocked by UX-6, 9, 10, 11, 12.*

**5 · The big-data user** — the one the pitch is aimed at. A 1.8-million-cell profile
produced 46 bytes of output, none of it progress: a hung process and a working one
look identical. And no way to say how much memory it may use. *Unblocked by UX-7,
UX-8.*

**The pattern:** every one of the twelve is presentation, contract or ergonomics. The
accumulators compute the right answers — the oracle proves it at 51 tests — and then
the surface hides them, mislabels them, or refuses the input. That is the cheap half
of the work, and it is the half that decides adoption.

Full text with reproduction, cause, fix and acceptance criteria: `docs/UX_ISSUES.md`.
File them with `python scripts/create_ux_issues.py --create` (after `brew install gh && gh auth
login`); `--dry-run` prints the exact commands without running any of them.

---

## A user-specified memory budget

Yes, and it is a good idea — **as a planner that derives settings, not a cap that
enforces them**. Full reasoning and measurements in `docs/adr/memory-budget.md`.

The model was fitted from measured peak RSS, not asserted:

```
peak_MB ≈ 75 + n_cols × (0.5 + k×37B + chunk_size×48B)
```

Memory really is flat in rows — 200k → 5M rows moved peak RSS from 32 to 35 MB above
the import floor — and linear in everything the library controls, so the budget is
**invertible**. The inversion was implemented and verified across five shapes; every
case landed under budget and the model over-predicts, which is the correct direction
for a budget to be wrong in.

It must not be a hard cap: ~75 MB is gone before the first line runs, the library
does not control pandas' allocations or GC timing, and a cap that raises `MemoryError`
would reproduce the incumbent failure mode this project is positioned against.

The real danger is silent accuracy loss — a tight budget shrinks `k`, and quantile
error is `1/√k` (±0.7% at 20,000, ±3.2% at 1,000). Always report the chosen plan and
its accuracy consequence, and error below the floor rather than degrading.

Before it ships: the model is fitted on **numeric columns only**. Refit for
categorical and datetime or the budget will be optimistic on a text-heavy frame —
which is the frame most likely to be large.

---

## Documentation

`benchmarks/check_docs.py` found 87 errors across 21 of 31 pages, and you fixed them
while this document was being written: `e95b483`, *"docs: fix the 90 errors from the
documentation audit"*, on `docs/fix-audit-errors` — 31 files, +762/−115, including
`mkdocs.yml` and `pyproject.toml`, and three more errors than the audit had found.

**The next move is the one that keeps it fixed.** Eighty-seven errors accumulated
because nothing was checking. Put `check_docs --strict` in CI in the same pull request
that lands these fixes, or the count starts climbing again from zero the next time a
key is renamed. The checker is the deliverable, not the fixes. Worth confirming in the
diff: that `algorithms/sampling.md` now documents Algorithm L rather than Algorithm R,
and that `architecture-diagrams.md` no longer claims extremes run every 5th chunk.

UX-6 is what stops the `rows` → `rows_est` class of error recurring — the docs were the
symptom, an unversioned payload is the cause.

Six figures shipped in #70. The remaining assets are ranked in `docs/DOCS_PLAN.md`
with ready-to-paste prompts in `docs/DIAGRAM_PROMPTS.md`: reservoir R-vs-L,
Misra-Gries eviction, the memory curve, annotated cards, the chunk lifecycle, the
Pébay merge.

Keep the rule that separates the two kinds. Anything depicting *what the code does*
belongs in `scripts/build_docs_assets.py`, generated from a real run with a fixed seed
and checked by `--check` in CI, the way `kmv-unit-interval.svg` already is. Anything
depicting a *concept* can be hand-authored. A picture a script regenerates cannot
quietly start lying.

CI cost remains none: the repository is public, every workflow runs on
`ubuntu-latest`, and GitHub Actions is free with no minute cap for public
repositories on standard runners.

---

## What to do next

**0 · Trust and Phase 1 — complete.** Eleven audit items closed, seven performance
items closed, oracle green at 51 tests, 2.48× end to end.

**1 · The one-day batch — five issues, two users unblocked (1 weekend).**
UX-9 `py.typed`; UX-10 `__all__`, a one-line `__repr__`, one exception type, deprecate
the `ReportConfig` alias; UX-12 accept `str`/`PathLike`; UX-4 let the log-scale chip
drive the chart default; UX-1 replace the unique-*ratio* arm with a cardinality
*ceiling*.
*Exit:* `age` profiles as numeric at 1k/100k/10M rows with a test that pins it;
`reveal_type(profile(df))` is `Report`; `profile("data.csv")` works; a lognormal
column opens on a log axis.

**2 · Ship the docs branch with the checker, then publish (1 weekend).**
The 90 fixes are committed at `e95b483`; open the PR and add `check_docs --strict` to
CI in the same one.
*Exit:* zero `check_docs` errors with CI enforcing it, benchmarks page with an
environment block, post #1 live, a comment on ydata #1129.

**3 · Make the report answer the reader's question (1–2 weekends).**
UX-2 the Identifier badge — `mono_inc`, a KMV estimate equal to the row count and
`int_like` are all already tracked, so this is a card swap. UX-3 a "needs attention"
block driven by the chips you already compute, plus chip filtering.
*Exit:* a 60-column report opens with "3 of 60 columns have issues", each linked.

**4 · Freeze the contract (1 weekend).**
UX-6 `schema_version` and a documented compatibility promise, then reconcile
`summarize()` with the HTML. UX-11 keyword passthrough and `fast`/`thorough` presets,
purely additive.

**5 · Make the streaming claim visible and controllable (1–2 weekends).**
UX-7 `progress="auto"` on stderr. UX-8 `memory_budget=` as a planner, with the model
refitted for categorical and datetime first and a CI test asserting measured peak RSS
stays under the target.
*Exit:* a documented "asked for 512 MB, used 480" produced by a test rather than a
blog post.

**6 · `pysuricata check` — the differentiator (2–3 weekends).**
UX-5. A baseline written from `summarize()`, a comparison, thresholds, a non-zero
exit. Gating on shape drift with zero configuration is a position no competitor
occupies, and it is the natural home for `memory_budget`.

**7 · The native core (3–5 weekends).**
Unchanged in ordering and now unchanged in urgency: **KMV first, moments last**,
because moments were 1.3–5% of the numeric path while KMV was half of it. The crate
is vendored with 20 passing tests and nothing imports it; there is still no `[fast]`
extra. The 32 KiB tiling result — 19.1 → 11.2 ns/row after the first naive version
came in *slower* than NumPy — is the best technical post in the project.

---

## This weekend

1. **File the twelve issues.** `gh auth login`, then
   `python scripts/create_ux_issues.py --create`.
2. **Add `pysuricata/py.typed`** plus the `package-data` line and a mypy step in CI.
3. **Replace the unique-ratio arm with a cardinality ceiling**, and add the test that
   profiles the same column at 1k/100k/10M rows and asserts the classification does
   not move. That test is the real fix; the rule change is one line.
4. **`__all__`, a one-line `__repr__`, one exception type.** Two hours.
5. **Open the PR for `docs/fix-audit-errors`, and add `check_docs --strict` to CI in
   the same PR.** The 90 fixes are committed at `e95b483`; the check is what stops them
   coming back.
6. **Then re-run the incumbent benchmark round-robin and publish.**

---

*Measurements: 0.0.31 at `4831a36`, plus 0.0.27, 0.0.26, 0.0.21 and 0.0.16 each on
`sys.path` in its own interpreter; five round-robin rounds, best of five per version;
`mixed` suite at 200,000 × 14, seed 0; 2-core x86-64 Linux container. Accumulator
timings over 1,000,000 float64 values in 20 chunks of 50,000. Memory figures from
`getrusage` peak RSS in fresh subprocesses at 0.0.27. UX assertions from a live 0.0.31
import. Absolute times are not comparable across sessions; ratios within a single
round-robin are.*
