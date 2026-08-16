# PySuricata roadmap v10 — re-audit at 0.0.62

Supersedes v8 (0.0.62), which was a narrow re-audit written the same day; the
performance and coverage work here is new, and the two findings v8 recorded — the
height decision and the dead-selector class of bug — are folded in below.

Three things changed shape since v7.

**The report is finished enough to publish.** Phases 1–7 of the migration landed
in thirteen commits — #128–#141 — and every tell catalogued at 0.0.38 is gone
from what a reader sees. The one acceptance criterion that was not met is the
histogram's width.

**The surface around the report is now the weakest part of the project.** The
README describes a library from two hundred commits ago, and the demo dataset
cannot exercise the features the redesign added — Titanic has no datetime column
and no correlation pair above 0.5, so the datetime card and both populated
correlation views never render in the one example anybody looks at. None of that
is visible from inside the repository, which is why it survived.

**A control can be correct, well-typed, well-tested and dead.** `+ add a note`
did nothing for eleven versions: the redesign renamed the block's class, the
JavaScript kept looking for the old one, and every entry point in that file
guards on a null container and returns quietly. 1,735 tests did not notice,
because not one of them asserted that a selector *resolves*. Fixed and
generalised in #142. It is the counterweight to #141: the fingerprint proved the
facts never changed, and it was right — it has no opinion about whether a button
works. A migration's confidence is bounded by the axis it instrumented.

| | |
|---|---|
| 0.0.16 → 0.0.61 | **3.01×** (mixed 200k × 14, clean round-robin, best of 5) |
| UX findings closed | **18 / 22** |
| Report, 20,000 × 5 | **519,418 B**, no external assets, no raster images |
| Coverage | **81.00%**, 1,759 tests, 9,335 statements |
| Open correctness items | **1** — quantiles printed as exact (#146) |

The headline ratio **has been re-measured**, which v7 noted it had not: 3.01× at
0.0.61, and 0.0.61 is 3.4% *faster* than 0.0.42, so thirteen commits of
render-layer rewriting cost nothing. `summarize()` does not render, which is why
that is the expected result rather than a lucky one.

## What landed since v7

| | |
|---|---|
| **#128** | the logo was 47% of the report — inline SVG, and the fixed cost halved |
| **#129–#136** | phases 1–5: tokens and type, the 52 px header, the stat row and stacked bar, the borderless sample, the restacked numeric card, axis units, the high-cardinality branch, boolean contrast |
| **#137** | quality chips show the number they already had in the DOM |
| **#138, #140** | correlations that report a weak result; missing values routed on chunk count |
| **#141** | the test that proves the facts do not change while the presentation does |
| **#126** | the impossible unique count — clamped, and `approx: True` on both sides |
| **#107** | UX‑17, UX‑18, UX‑19 in one commit |
| **#108** | the accumulator boundary, prepared for a second implementation — and measured today at **0.97–1.01× of 0.0.42**, so the preparation cost nothing |
| **#109** | `compare(a, b)` |

#141 is the one worth calling out. Thirteen consecutive commits rewrote every
template and stylesheet in the project; the only reason that was reviewable is
that a test asserted, on each of them, that the numbers had not moved.

## The measurement rules, with one clause added

A ratio is quotable only when both sides were measured in the same round-robin,
on the same machine, within the same run — **and nothing else was running.**

That clause is new and it was earned today. The first round-robin put 0.0.61 at
1,599 ms against 0.0.42's 1,448 — a 10.5% regression, well outside the ±1% this
harness reproduces to. The hypothesis was ready-made: #108 introduced an
abstraction boundary in the accumulator hot path, which is exactly where a few
percent goes. Bisecting seven commits refused it — 1,203 to 1,271 ms, no
monotonic trend, HEAD at 1.008× — and the cause was mine: the coverage suite was
running in parallel, so a four-and-a-half-minute pytest run was competing for two
cores with the benchmark measuring against it.

Interleaving cancels drift between versions. It does not cancel contention from a
neighbour. Worth asserting in `benchmarks/versions.py`: refuse to run above a load
threshold, or at minimum print the load average beside the numbers.

Fourth time in this audit that a measurement artefact nearly became a claim, and
the first time it was caught before it was written down.

## What is still open

| # | Finding | State |
|---|---|---|
| UX‑21 | `Processed bytes (≈)` in the primary stat row | half — the donut is gone, this is not |
| UX‑22 | `ComputeOptions` at 22 fields; `numeric_sample_size` not a keyword | open — and it is the knob UX‑8 drives |
| UX‑8 | `memory_budget=` | open, deliberately post-launch |
| — | `ReportConfig is ProfileConfig` | open, no deprecation warning |

## Found in this audit

### The histogram never got its width

```html
<svg class="hist-svg" width="420" height="200" viewBox="0 0 420 200">
```

A fixed 420 px canvas, centred in a card that is now around 1,900 px wide.
Phase 5.1's argument for the restack was precise about why it mattered — *"it
gains ~550px of width, which is what finally makes 50 bins legible and the log
toggle worth using"* — and that did not happen. The chart moved rather than grew.
Cheapest visual win remaining, and "the histogram is at least 900 px at a 1240 px
viewport" is exactly the kind of assertion the layout tests already run.

### The CSS grew, and the ban list was too narrow to notice

| | v6 audit | today |
|---|---:|---:|
| lines of CSS | 8,561 | **8,990** |
| distinct hex values | 90 | **100** |
| `linear-gradient` | 67 | 55 |
| `box-shadow` | 104 | 89 |
| `border-radius` | 158 | 147 |

Phase 1's acceptance was that none of nine named hexes appears in `static/css/`
or `render/`. **All nine are clear** — and there are a hundred, including
Tailwind's amber/red/green and Material's orange/green/red. The list named the
old *accent* colours and missed the semantic ramp, so the check passes while both
frameworks' defaults are still in the file.

Replace it with the inverse assertion: **every hex in `static/css/` must appear
in `_00-tokens.css`.** One test, cannot be outgrown, and it turns commits 15 and
17 from a tidy-up into a ratchet.

### The report is 519 KB only while it is narrow

20,000 rows, columns cycling through all five kinds:

| columns | bytes |
|---:|---:|
| 2 | 464,582 |
| 5 | 519,418 |
| 10 | 702,185 |
| 20 | **1,062,649** |

Fit: **~363 KB fixed + ~33 KB per column**, residuals within 5%.

578 KB came off the *fixed* cost, which is why a narrow report halved. Nothing
came off the per-column term — so a 20-column report is back over a megabyte and
a 60-column one would be around 2.3 MB. Say which shape a size refers to before
quoting it. The next win is per-column: six pre-rendered histograms per numeric
column (three bin counts × two scales) is the obvious place to look.

### `finalize()` does not consume randomness — but the quantiles are unlabelled estimates

This was written up as the project's one open correctness item, on the grounds
that finalising at chunk 3 of 6 diverges from an uninterrupted run on eleven
fields including `median`, `q1`, `q3`, `iqr` and `mad`. **Re-checked at 0.0.62
with a control, and it does not hold.**

```
unseeded, neither finalized mid-stream : 9 fields differ   <- the control
unseeded, one finalized mid-stream     : 11 fields differ
seeded,   neither finalized mid-stream : none
seeded,   one finalized mid-stream     : chunk_metadata
```

The eleven were two **unseeded** runs being compared. A control with no
`finalize()` anywhere already differs on nine of them, so the comparison never
isolated the thing it named. With a seed the median is bit-identical either way
(`0.007186`), and `ReservoirSampler.values()` returns `self._buf` — it consumes
no randomness, so the stated two-line mechanism does not exist.

What survives is smaller and belongs to #139: `mark_chunk_boundary()` is called
only from `finalize()`, so `chunk_metadata` counts *renders*, not chunks.

**The real finding is the one the bad comparison was standing on top of (#146).**
The quantiles genuinely are reservoir estimates, they genuinely do move — eight
unseeded runs spread `1.86 × 10⁻²` on a true median of `2.8 × 10⁻³` — and the
card prints them to four significant figures with no approximation marker, three
orders of magnitude finer than the estimate supports. `CLAUDE.md` has a standing
rule that approximate values must be labelled; the unique count follows it and
the quantiles do not.

It is invisible to the suite because **every test seeds**. `profile(df, seed=0)`
is bit-reproducible; `profile(df)` — what a user writes — is not.

Fourth measurement artefact in two audits, and the first one that was caught by
running a control rather than by someone re-deriving it later.

## The surface

### The committed example report is the design you replaced — the published one is not

`docs/assets/titanic_report.html` is **1,178,450 bytes**, of which 578,276 are the
old logo PNG. 464 uses of `#3b82f6`. Zero occurrences of `--paper`. Last
regenerated at `663ed24` — **PR #23**.

The obvious conclusion is that the README's link serves the design the redesign
replaced. **It does not**, and this was written down twice before anyone checked
the URL rather than the file:

```
$ curl -sI https://alvarodiez20.github.io/pysuricata/assets/titanic_report.html
last-modified: Sun, 16 Aug 2026 09:41:41 GMT      # minutes after #142 merged
$ curl -s ... | wc -c
600049                                             # current output
```

`docs.yml` runs the regeneration script *before* `mkdocs build` and then deploys,
on every push to `main`. So the published report has never been stale. What is
stale is the copy committed to the repository, which nothing serves and no CI job
reads.

That makes this repo hygiene rather than the highest-priority item, and it
reframes the fix (#143). The question is not *regenerate it* but **why a build
artefact is committed at all**: three workflows already produce it, the only
local consumer is `mkdocs serve`, and committing it means every rendering change
either produces a megabyte diff or drifts silently. Forty-five versions of drift
is the evidence that nobody wants that diff. Ignore it and delete it from the
tree, rather than resetting it once and re-arming the same trap.

### Titanic cannot demonstrate what the library does

No datetime column at all, so the datetime card and the four temporal small
multiples never render in the one example anybody looks at. No correlation pair
above 0.5 — the design handoff had to use illustrative numbers for the ranked list
and the matrix for exactly this reason. No identifier column. 891 rows.

| Candidate | Rows | What it adds |
|---|---:|---|
| **NYC Yellow Taxi**, one month of Parquet | ~3M | Two datetimes; strong real correlations (`trip_distance` ↔ `fare_amount` ↔ `total_amount`); categoricals; real missing values; ~50 MB, so **the streaming and DuckDB story is visible in the same example** |
| **UCI Bike Sharing** `hour.csv` | 17,379 | Datetime, `temp`↔`atemp` ≈ 0.99, two booleans. No missing values, no identifier |
| seaborn `taxis.csv` | 6,433 | Pickup and dropoff datetimes, distance↔fare. The low-risk quick-start swap |

Use two: something instant in the quick start, taxi data for the linked showcase.
*(Recommendations from known properties, not measurements — the sandbox could not
reach the hosts to profile them. Verify before committing.)*

### The README describes a project from two hundred commits ago

Wrong: sketch `k` says 1024 and is **2048**; sample `s` says 10 000 and is
**20 000**; the CLI section documents two of three subcommands; the configuration
example teaches `ReportConfig` and the two-constructor ceremony **#87 removed**;
`chunk_size = 250_000` is 5× the default.

Missing: `pysuricata check` — the differentiator is not in the README — DuckDB,
Parquet and Arrow, `compare()`, keyword options, `preset=`, `progress=`,
`py.typed`, `schema_version`, and a screenshot.

192 lines, and one number in them, for a project whose thesis is measured
performance and bounded memory. A draft rewrite is in hand; every code example in
it was executed against 0.0.61, which caught one error in the draft itself —
`Comparison` has no `save_html`, only `to_dict()`.

### The triage block lists rather than triages

```html
<a href="#col_d">d</a>
   <span class="flag bad"  data-flag="87-99-heavy-tailed">87.99 heavy-tailed</span>
   <span class="flag bad"  data-flag="3-0-many-outliers">3.0% many outliers</span>
<a href="#col_gappy">gappy</a>
   <span class="flag bad"  data-flag="37-8-missing">37.8% missing</span>
<a href="#col_const">const</a>
   <span class="flag warn" data-flag="dominant-category">Dominant category</span>
```

1. **It is not ranked.** `bad` and `warn` are in the class and unused, and 37.8%
   missing is listed *below* a 3.0% outlier flag. Rank by severity, then by
   `value / threshold` — the one quantity comparable across flag types. 37.8%
   missing against a 20% threshold is 1.9×; 3.0% outliers against 1% is 3.0×.
2. **Half the rows carry no number.** `Dominant category` is a bare word beside
   two rows that quantify themselves; `const` is 99.8% one value, so say that.
   Same for `Quasi-constant` → `0.3% unique`.
3. **The thresholds were dropped in this layer.** Card chips carry
   `data-threshold` and `data-value` — #137 added them. The attention chips have
   only `data-flag`, so `37.8% missing` says *what* and not *why it is here*.
4. **Nothing aligns.** `grid-template-columns: max-content 1fr`, the same row grid
   used everywhere else in the redesign.
5. **"3 of 4 columns need a look" is a list.** Above roughly half the columns
   flagged the framing stops working: rank and cap at the worst five to ten. And
   the inverse is missing — when nothing is flagged the block disappears, which
   reads as a broken feature. It should say *"All 12 columns look fine — none
   crossed a quality threshold."* That is the empty-state argument #138 already
   accepted for correlations.

None of it needs a new statistic. Severity, value and threshold are all computed
and all present somewhere in the document; the fix is ordering, formatting, and
carrying two attributes one layer further.

## The height decision, made

Open across seven hand-offs and blocking #122. Both targets are raised to their
measured values, and #114's internal conflict is resolved in favour of its
accessibility criteria over its compactness estimate:

| | was | now |
|---|---|---|
| #112 summary | ≤560px | **≤627px** |
| #114 numeric card | ≤600px | **≤820px** |

Pinned to `docs/assets/titanic.csv` at 390 × 844, details collapsed. Set *at* the
measured value, as ratchets.

Neither original number reproduced, and the reason generalises: **neither
criterion named a dataset or a viewport height.** Same failure as the report-size
series in v7, in a different part of the project, found the same way — by trying
to reproduce it. Pin the conditions whenever a number goes in an acceptance box.

Setting them surfaced #145: three of the four card kinds have no height criterion
at all, and the tallest card in the report is a categorical one at **923px**,
above the numeric card that has the only target.

## What to do next

**1 · The pre-launch day.** Make the histogram responsive (#147) — a fixed 420px
canvas in a ~1,900px card, so #114's stated reason for the restack was never
delivered. Label the quantiles approximate (#146). Round the sample table's
figures. Move `Processed bytes` into the details panel. Stop committing the
example report (#143). Install the [Codecov GitHub App](https://github.com/apps/codecov).

**1b · The surface, one day.** Land the rewritten README; swap the demo dataset
for one with a datetime column and a real correlation; rank the triage block and
give every row a number.

**2 · Publish.** All three posts are unblocked at once, which was not true at the
last audit — the retraction, DuckDB plus the memory model, and a third that is
better than it sounds: thirteen commits rewrote every stylesheet in the project
and the reason it was reviewable is a test.

**3 · The deletion pass — commits 15 and 17.** The inverse-hex assertion, then
delete until it passes: 100 hexes to ~12, 55 gradients, 89 shadows, the shim.
Target 8,990 → ~2,500 lines. Then the accessibility pass.

**4 · Phase 5b, the partial report.** Unblocked once #139 makes `chunk_metadata`
count chunks rather than renders.

**5 · The memory budget**, preceded by adding `numeric_sample_size` to the
passthrough keywords.

**6 · Resume, then the native core.** #108 already prepared the boundary, and
today's measurement says that preparation cost nothing — which is the result you
want before building on it. KMV first, moments last.

---

*Measurements: 0.0.61 at `0191b80`, plus 0.0.42, 0.0.27 and 0.0.16, each on
`sys.path` in its own interpreter; five round-robin rounds, best per version;
`mixed` suite at 200,000 × 14, seed 0; 2-core x86-64 Linux container with nothing
else running. Bisect over seven commits, four rounds. Coverage from
`pytest --cov=pysuricata` at the same commit. Report sizes from frames of 20,000
rows with columns cycling through all five kinds. UX assertions from a live 0.0.61
import. Absolute times are not comparable across sessions; ratios within a single
round-robin are, provided nothing else is running.*
