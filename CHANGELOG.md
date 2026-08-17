# Changelog

All notable changes to PySuricata are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) —
which at `0.0.x` means the public API may still change between releases. The
`summarize()` payload is the exception: it carries a `schema_version` and is
treated as a contract.

Each entry states what changed and, where it matters, **what was measured**.
Numbers here come from the benchmark harness in `benchmarks/`; ratios are only
quoted when both sides were measured in the same round-robin run.

## [Unreleased]

### Added

- **The benchmark harnesses refuse to measure on a busy machine** ([#212]). The
  rule this project measures by — *both sides in the same round-robin, on the
  same machine, within the same run* — cancels drift between the things being
  compared. **It does not cancel a neighbour**, because the neighbour is not in
  the round-robin, and that nearly published a claim: a run put 0.0.61 at
  1,599 ms against 0.0.42's 1,448, a **10.5% regression** on a harness that
  reproduces to ±1%, with a ready-made culprit in the abstraction boundary #108
  had just added to the accumulator hot path. Bisecting seven commits refused it
  — 1,203 to 1,271 ms, no trend, HEAD at 1.008× — and the cause was the coverage
  suite running in parallel, competing for two cores with the benchmark
  measuring against it.

  `load_guard()` reads the one-minute load average and refuses above one per
  core, with `--force` for anyone who knows what the neighbour is. The load is
  recorded at **both ends** and exported with the results, because the reading
  taken before a run cannot see a job that starts during it — a suite already
  running is caught by the opening check, one launched a minute later is only
  visible in the closing one. A run that ends busy prints a warning saying not
  to quote a ratio from it.

  `versions.py` imports the guard rather than reimplementing it, so there is one
  threshold rather than two that drift. Where the OS has no `getloadavg`
  (Windows), the check is skipped and says so rather than inventing a number.

  The clause is now in `CLAUDE.md` beside the rule it amends, and
  `tests/test_benchmark_load_guard.py` asserts it is there — a rule that lives
  only in a document is what let this happen, so the test fails if the note is
  removed. All four mutations of the guard are caught, including that one.

### Fixed

- **Categorical and boolean columns track their own chunks, so the Missing
  Values pane is gated the same way on all four card kinds** ([#193]). #154's
  5b.7 set the rule — render the pane only when **missing > 0 and chunks > 1**,
  the one condition under which it knows something the card face does not,
  namely *where in the read* the gaps fall. It could only land for numeric and
  datetime, because `render/html.py` finalized the other two without chunk
  metadata and neither summary had a field to hold it.

  A single-chunk report was consistent by accident: the numeric and datetime
  panes dropped and the other two had nothing to drop against. A multi-chunk
  report was not — `Age` got a strip showing where its gaps fell and `Embarked`
  got a Present/Missing pair restating its header.

  `CategoricalAccumulator` and `BooleanAccumulator` now count rows and missing
  values per chunk through a shared `ChunkTracker`, carry `chunk_metadata` on
  their summaries, and implement `mark_chunk_boundary()`. The engine needed no
  change: it has always been duck-typed, with a comment saying the other kinds
  "should start working the moment they do".

  **Boolean earned its details section back.** It had none — a documented
  decision (#155, 5c.6) rather than an omission, on the grounds that its two
  panes restated the card face. That reasoning named this issue as the release
  condition: *"boolean accumulators are finalized without chunk metadata, so
  the pane has no such fact to offer and cannot acquire one. When #193 lands it
  may earn its tab back under the same rule."* It now has one pane, Missing
  Values, appearing only when the rule opens.

  Merging is handled rather than assumed: a merged column's chunks are the two
  runs' chunks in order, so the second side's boundaries are offset by the
  first's row count instead of restarting at zero halfway through.

  `tests/test_missing_pane_gate.py` asserts the gate **open and closed on every
  kind**, which is the point the issue makes: `getattr(stats, "chunk_metadata",
  None)` returns `None` rather than raising, so a gate applied to a kind with no
  such field *looks* applied while hiding the pane permanently — and a test that
  only checks the closed side passes just as happily against a pane that can
  never appear. All four mutations of the fix are caught, including that one.

  Two fixture traps hit while writing it, both recorded because they report
  "absent" when the truth is "the fixture missed the branch":
  `np.where(mask, pd.NaT, dates)` yields an *object* column that never infers as
  datetime, and a bool column with `None` punched into it is object too and
  infers as categorical. A nullable `"boolean"` dtype with `pd.NA` is what
  produces a boolean card.

  One thing deliberately not changed: feeding a nullable `"boolean"` Series
  straight into the accumulator raises *"boolean value of NA is ambiguous"*.
  That is a shape the pipeline never produces — `_to_bool_array_pandas` hands
  over `[bool | None]`, having already converted `pd.NA` — so the crash was a
  test inventing an input, not a defect.

### Changed

- **`Processed bytes (≈)` left the primary stat row on the numeric and datetime
  cards** ([#209]). UX-21 asked for this; #104 dropped the donut and the stat-row
  half never landed. The numeric card's right-hand table read Min, Q1, Median,
  Mean, Q3, Max — six facts about the distribution — and then one about the
  profiler's own bookkeeping, in the position of highest attention on the card.
  It answers a question about PySuricata, not about the data. It is not useless,
  so it moved to the Statistics pane rather than going away.

  **Two of four kinds, and the other two are recorded rather than waived.**
  Categorical has details panes but no Statistics pane, and every pane it has is
  conditional — filing a fact that must always be in the document inside a pane
  that renders only sometimes would "move" it by making it vanish, which
  `test_report_data_invariance.py` would rightly catch. Boolean has no details
  section at all, and that is a documented decision (#155, 5c.6) rather than an
  omission: two values, two counts, both on the card face, no second level of
  disclosure to offer. Giving it one to house a byte count would be the tail
  wagging the dog.

  `tests/test_processed_bytes_placement.py` pins both halves as a ratchet — move
  one of the remaining two and it fails, telling you to shrink the set. Its
  fixture carries all four card kinds on purpose: Titanic has no datetime column
  (#150), so the example report cannot exercise that branch at all, and a
  fixture that misses a branch reports "absent", which reads as passing.

  Worth recording for the next person: deciding "primary row or details pane" by
  splitting the card's markup at the `details-toggle` button gets **categorical
  backwards**, because that card emits the toggle ahead of its stat row. The
  test decides by which container opened most recently instead, and the first
  version of this measurement reported categorical as already done when it was
  not.

- **A histogram bar stopped paying for things nothing reads** ([#206], first
  pass). A bar is the most repeated element in the report — 50 of them in each
  of 6 variants of every numeric column, 300 per column — so anything constant
  on one is multiplied by 300. Two things were:

  `vector-effect="non-scaling-stroke"`, **41 bytes on every mark**, moved into
  the `.bar`, `.grid` and `.axis` rules. It belongs there: the stylesheet's own
  comment beside `stroke: var(--paper)` already explained why the stroke must
  not scale. And a third decimal on four coordinates — the viewBox is 0..100, so
  a unit is a percent of the plot and at 1,100px the third decimal is a
  ten-thousandth of a pixel.

  | | Before | After |
  |---|---:|---:|
  | one bar | 184 B | **131 B** |
  | marginal bytes per numeric column | 73,204 | **63,596** |
  | Titanic report | 600,491 | **573,809** |

  `vector-effect` as a *CSS property* rather than an attribute is SVG2, so it
  was verified rather than assumed: computed style reports `non-scaling-stroke`
  on bar, grid and axis at 1240px and 390px, and the rendered histogram is
  **pixel-identical** before and after — `getbbox()` on the difference returns
  `None`. Isolating that mattered, because the coordinate rounding *does* move
  264 of 1.1M pixels by at most 47/255, all of it antialiasing on bar edges.

  **`data-col` was measured, removed, and put back.** It reads as pure
  redundancy — the column is on the `.hist-variants` parent, and neither the
  tooltip handler nor any stylesheet touches it. But `scripts/report_fingerprint.py`
  takes an element's scope from the *same tag*, so dropping it turned every
  `attr::col_age::count` into `attr::::count` and collided the bar counts of
  every numeric column under one key. `tests/test_report_data_invariance.py`
  caught it. A weaker invariance guard is the wrong thing to buy with 19 bytes a
  bar, so the 5,700 bytes per column stay spent, and the reason is now written
  next to the attribute.

  This is the cheap half of #206 and does not close it: the six variants are
  still all rendered. The remaining half — emit one and build the other five on
  toggle — needs a JS port of a 179-line SVG renderer, which is a second
  implementation of the chart and wants a decision rather than a commit.

  Guarded by `TestABarPaysOnlyForWhatIsRead`, which asserts both directions:
  nothing constant creeps back on, and everything with a reader stays. All four
  mutations of it fail as they should.

### Added

- **The redesign's acceptance criteria run as tests** ([#124]). Every redesign
  issue ends with numeric acceptance lines, already phrased as assertions and
  until now only read. `tests/test_report_layout.py` executes them: 9 cases with
  no browser, 31 in Chromium across 390/768/1240 × light/dark.

  **The criteria turned out to be a specification, not a description.** #124
  quotes `len(html) < 400_000` and `elements_per_card < 400`. The Titanic report
  is **600,491 bytes** and its widest card holds **843** elements — those are the
  numbers #39 and #206 are open to deliver. Asserting them now would ship a red
  suite that gets disabled on Monday, so they land as **ratchets** against a
  recorded baseline, the idiom `test_colour_tokens.py` already uses: growth
  fails, and shrinking fails too, asking for the baseline to come down.

  **Three criteria appear to fail and do not**, all three because the obvious
  measurement reads the wrong box. The header measures 53px against a ≤52px
  budget until you notice it computes to exactly `height: 52px` and carries a 1px
  bottom border that `getBoundingClientRect()` counts. `.icon-btn` measures 30×30
  against a 44×44 minimum until you notice the hit area is an absolutely
  positioned 44×44 `::after` — `elementFromPoint` six pixels outside the visible
  box still returns the button. And `scrollWidth > clientWidth` names nine
  elements at 1240px, none of which scrolls: `sr-only` clips, `icon-btn`'s
  `::after` overflows, an SVG returns an animated string for `className`. Scored
  properly — content wider than the box *and* a scrollable `overflow-x` — there
  is exactly **one** scroll pane, the sample table, and the document never
  overflows at any breakpoint. All three budgets are met.

  Two criteria genuinely miss and are recorded rather than waived: the summary is
  **620px** at 390px against #112's 560px, and desktop nav links are 31px tall
  against #111's 44px. The two remaining sub-44px targets are inline links inside
  a sentence, which WCAG 2.5.8 exempts.

  The theme axis had to be rebuilt to mean anything. The report does not use
  `prefers-color-scheme` — dark is the *absence* of a `light` class — so
  Playwright's `color_scheme=` did nothing, and the first contact sheet came out
  byte-identical in pairs while six "theme" cases measured one state twice.
  Toggling the class is also not enough on its own: `transition:
  background-color 0.3s` means an immediate read still returns the old paper.
  `assert_theme()` now sets the class, waits out the transition, and **requires
  the two themes to compute different backgrounds**, so the axis cannot go inert
  again. With it working, dark mode provably changes no geometry.

  Each of the seven gates was verified by breaking it on purpose; all seven fail
  when they should.

- **#119's correlation criterion, in the shape that survived the redesign.**
  #124 asks that the matrix emit `n(n-1)/2` cells; **there is no matrix** — #122
  removed the heatmap and #154's 5b.6 replaced it with a per-column partners
  pane, so a test written to the criterion would search for a `corr-cell`, find
  nothing, and pass by being vacuous. The invariant behind it survives and is
  stronger in the new shape: a matrix names each pair once, the panes name it
  from **both** sides, so the count is `n(n-1)`. Measured at n = 3, 4, 5, 6 →
  6, 12, 20, 30. A dropped pair, or one column's pane missing a partner, breaks
  the identity.

  The other four layout criteria #124 lists turned out to be covered already —
  the 12 month slots in `test_accumulators_datetime.py` and
  `test_boolean_and_temporal.py`, the high-cardinality no-chart rule in
  `test_high_cardinality_branch.py`, the frozen index in `test_sample_table.py`,
  and the nav rail by the scroll allow-list plus the recorded 31px target gap.

- **A contact sheet for reviewing a phase** (`scripts/contact_sheet.py`, #124).
  Six full-page captures uploaded by CI as an artifact, and deliberately
  **never a gate**: thirteen redesign issues are *supposed* to change every
  pixel, so a pixel-equality check would be switched off during the first phase
  and stay off. The structural assertions are the gate; the images are how a
  human reviews a phase in thirty seconds instead of thirty minutes.

- Browser work sits in its own `browser` dependency group and its own `layout`
  CI job, so the other six jobs do not pull a ~300 MB Chromium, and the cases
  skip themselves anywhere it is absent.

[#124]: https://github.com/alvarodiez20/pysuricata/issues/124
[#193]: https://github.com/alvarodiez20/pysuricata/issues/193
[#206]: https://github.com/alvarodiez20/pysuricata/issues/206
[#209]: https://github.com/alvarodiez20/pysuricata/issues/209
[#212]: https://github.com/alvarodiez20/pysuricata/issues/212

## [0.1.1] - 2026-08-17

**The contract written at 0.1.0, held.** Nothing covered by `docs/versioning.md`
broke: the `summarize()` payload gained `duplicate_rows_uncertainty` and lost
nothing, so `schema_version` stays at `1`; no public name was removed;
`ReportConfig` warns rather than disappearing, with 0.3.0 as its date. Verified
by diffing the payload's key set against the 0.1.0 tag rather than by reading
this file — zero removed, one added.

The largest changes are in the report's HTML, which that page deliberately does
not cover, and in the documentation, which is now checked against the live
library instead of trusted.


### Fixed

- **The README documented a project several releases behind** ([#151]). Every
  number in it was wrong: the sketch size was given as 1024 against an actual
  `max_uniques` of **2048**, the sample size as 10,000 against a
  `numeric_sample_size` of **20,000**, and the configuration example set a
  `chunk_size` of 250,000 while describing it as ordinary — it is 5× the
  default of 50,000. It taught the two-constructor ceremony removed in #87,
  described `ProfileConfig` as "aliased as `ProfileConfig`", and documented two
  of the three CLI subcommands.

  The ~2.2% KMV error was the one figure that survived, and only by accident:
  the relative standard error is `1/sqrt(k - 2)`, which is 2.2% at k=2048 and
  3.1% at the 1024 the same sentence claimed. The number had quietly followed
  the code while its own `k` did not.

  Missing entirely: `pysuricata check` — the differentiator — plus `compare()`,
  the Parquet/DuckDB/Arrow readers, keyword options, `preset=`, `schema_version`
  and `py.typed`. All now documented.

  **The permanent part is that the README is now checked.** `benchmarks.check_docs`
  globbed `docs/` only, so the one page every reader sees first was the one page
  outside the checker that exists to stop exactly this. It now executes the
  README's fences and resolves its config options and summary keys against the
  live API, and `docs-check.yml` triggers on `README.md`. Two real defects
  surfaced on the first run — a `KeyError: 'mean'` and a missing `profile`
  import — and one of them is the trap the house rules warn about: the setup
  frame was small enough that `age` profiled as *categorical*, so the column
  genuinely had no mean.

  The screenshot is regenerated too, by `scripts/capture_report_screenshot.py`.
  The previous one was captured at **0.0.26** and sat unchanged through the
  entire report redesign, so the picture advertising the project showed a design
  it no longer had. That script is deliberately not wired into
  `build_docs_assets.py --check`: a browser screenshot is not byte-reproducible
  across platforms and font sets, so pinning it there buys a flaky job rather
  than a guarantee.

  **Two gaps closed afterwards**, both about what the new check does *not*
  reach. `docs-check.yml` triggered on `README.md` but not on
  `benchmarks/check_docs.py`, so a pull request that narrowed or broke the
  checker was the one pull request the checker did not run on; it now triggers
  on every script the job runs — `check_docs.py`, `build_docs_assets.py`,
  `regenerate_example_report.py` — and on the workflow file itself, since
  editing the trigger list is exactly the edit that most needs the job to run. And every claim #151
  was actually filed about — sketch `k`, numeric sample size, subcommand count
  — is *prose*, not a fence: `k = sketch size (default 2048)` is italic text
  under a table, and the CLI section is a `bash` block, which `check_docs`
  cannot execute. The guard that closed #151 would not have caught what #151
  reported. `tests/test_readme_is_checked.py` reads those figures from
  `ComputeOptions()` and the subcommand list from the parser's own source, so a
  renamed default fails a test rather than drifting. Digit grouping is
  normalised away — `20 000`, `20,000` and `20_000` are one claim, and a test
  that insists on one spelling fails the next person to restyle a sentence.
- **The datetime timeline no longer scales its own labels** ([#217]). It drew
  every label inside an SVG carrying `preserveAspectRatio="none"` at
  `width: 100%`, so nothing in it had a size of its own: the viewBox mapped
  onto whatever box CSS handed it. In a 1,146px column a 420-unit box scaled
  everything by **2.73** — an 11px tick rendered at ~37px, three times the stat
  row beside it — and the card stood 844px tall for what is often a flat line.

  Widening the viewBox is not the fix, and finding that out is most of the
  work: authoring at ~1,100 units makes a wide column right and a 470px one
  render the same label at **5px**. There is no viewBox that is correct at both
  widths.

  The timeline is now a `figure.hist`, the structure the numeric histogram
  already uses and the reason it uses it: the SVG holds only marks, and every
  label is HTML positioned by percentage. Labels measure 11px at every width
  from 1600px down to 600px. Reusing the histogram's classes rather than
  styling a second chart also means the timeline inherits the gutter, the
  tiered labels that thin 5 → 3 on narrow screens, the caption, and the
  axis-label nudges — all of which already exist and are already tested, and
  none of which can now drift apart from the histogram's.

  The card falls from 844px to 601px, and the column name is no longer drawn a
  second time inside the chart when the card header already carries it.

  `tests/test_chart_layout.py` gains the general rule this came down to: **no
  `<text>` inside a non-uniformly stretched SVG**, anywhere in the report.

- **A numeric card drew every histogram variant at once.** The bins and scale
  toggles offer six combinations, and all six were on screen simultaneously —
  stacked, overlapping their own captions, in a card **1,671px tall instead of
  ~570px**. The toggles appeared to do nothing because every option was already
  displayed.

  A vestigial `display: block` was the whole cause. The rule that carried it
  once forced `height: 100%` onto a child `<svg>`; when that was removed in
  #167 the declaration was left behind, and at one id and five classes it
  outranks both `.hist-variants .variant { display: none }` (one id, two
  classes) and `.hist-variants .variant.active { display: block }` (one id,
  three). Only numeric was affected — categorical variants are `.cat.variant`,
  which that selector never matched, which is why one card kind showed a single
  chart and the other showed all of them.
- **The extreme y-axis labels were nudged the wrong way.** The stylesheet pulls
  the top and bottom labels inward, "or the top label floats above the plot and
  the `0` hangs below the axis" — its own words. It keyed that on
  `:first-of-type` and `:last-of-type`, but ticks are emitted in *ascending*
  order, so the first span is the bottom of the axis and the last is the top.
  The corrections were therefore applied to the opposite ends and produced
  exactly the two defects they exist to prevent. The renderer now tags each
  extreme with `data-edge`, so the nudge no longer depends on DOM order.
- **The caption was drawn on top of the x-axis labels.** `.hist__area` was
  pinned to `--hist-height` and the tick row lives inside it, so that row
  overflowed the box without contributing any layout height and the caption
  below was placed over it — measured at ticks `y[2026..2044]` against caption
  `y[2030..2044]`. The height moved to the plot itself.
- **Categorical bar thickness was inversely proportional to cardinality**
  ([#145] in part). The chart divided a fixed height among its bars, so a
  two-level column drew two **218px** slabs where a five-level column drew
  87px, and the same chart read as a different chart for every column. A bar
  now has a height and the chart is however tall that makes it: 35px per bar
  at any level count, and cards fall from a uniform 743px to 405–542px.

  The chart is also authored at 1,100 units wide rather than 420. It was being
  stretched to fill a ~1,150px column, which multiplied everything inside it by
  2.7 — so an 11px bar label rendered at ~30px, three times the size of the
  stat row beneath it.
- **The count printed inside a bar was unreadable.** It used `--muted`, a grey
  chosen against the page, and on the bar fill that measures **1.20:1** against
  the 4.5:1 AA asks of text this size. The value sits inside the bar when the
  bar is wide enough and past its end when it is not — two different
  backgrounds that cannot share one colour — so the renderer now says which,
  and the inside case takes `--paper`: 6.12:1 in light, 6.28:1 in dark.

  `tests/test_contrast.py` already declared `("paper", "data-2", "count printed
  inside a default bar")` as the correct pairing. The palette was right and the
  chart was not using it.

  All five were found by rendering a report in Chromium and measuring the
  boxes, which is the only way any of them are visible: the fingerprint
  deliberately discards presentation, and every other check reads values rather
  than geometry. `tests/test_chart_layout.py` guards them from the markup and
  the stylesheet, and twelve of its fifteen cases fail against the previous
  behaviour.
- **A polars column of timestamp strings no longer loses every value**
  ([#214]). 200 valid ISO-8601 timestamps profiled through polars came back
  `count=0, missing=200` — the column still labelled `datetime`, so nothing
  looked structurally wrong; the card simply asserted the data was entirely
  absent. The same values through pandas were correct.

  `Series.cast()` from a String yields nulls rather than raising, so a
  `strict=False` cast reports success while producing nothing, and the
  `except Exception` fallback written around it is unreachable. The conversion
  path tried `Date` first and kept it; inference tried `Date` then `Datetime`
  and took the first that looked good. Where the two disagreed, a column was
  typed `datetime` by one and emptied by the other:

  | input | `cast(Date)` | `cast(Datetime)` | `str.to_datetime` |
  |---|---|---|---|
  | `2020-01-01` | ok | all null | ok |
  | `2020-01-01 12:00:00` | all null | all null | ok |
  | `2020-01-01T12:00:00` | all null | ok | ok |

  Both paths now go through one shared parser, so they cannot disagree again.
  Space-separated timestamps also stop being profiled as `categorical` by
  polars and `datetime` by pandas.

  The same change removes a **Polars 2.0** break: casting String → Date/Datetime
  is deprecated from polars 1.43 and removed in 2.0, and `polars>=1.34.0` has no
  upper bound, so an upgrade would have taken these paths out from under the
  library. The repository lockfile pins 1.34.0, which is why CI never saw the
  warning.

  The bug needed **both backends** to be visible at all — one backend alone is
  self-consistent and looks right — and the existing polars fixtures pass
  already-typed `pl.Datetime` columns, which take the fast path and never reach
  the cast. `tests/test_polars_datetime_strings.py` compares the two backends
  field for field across all three string shapes.

### Added

- **An oracle case pinning that `finalize()` is idempotent** ([#205]). The issue
  reported that finalising mid-stream consumed the reservoir's randomness, so
  `checkpoint_write_html=True` changed the median across eleven fields. **It
  does not reproduce**: with `random_seed` set, profiling with partial renders
  on is field-for-field identical to profiling with them off, and no statistic
  in any of the four accumulators moves when `finalize()` is called mid-stream.

  The eleven-field divergence appears to have been a measurement artifact. An
  accumulator built with `seed=None` seeds itself non-deterministically, so two
  runs of the *same* uninterrupted stream already disagree on exactly those
  quantile-and-sample fields — reproduced here by accident while trying to
  confirm the report, which is how it was identified. The only genuine residue
  is `chunk_metadata`, and only when the engine is *not* marking boundaries;
  since #139 it always is, so the real pipeline is unaffected.

  The oracle is added anyway, because the invariant matters more than the bug
  report: it is the precondition for the progressive report, and it now has a
  control case (two uninterrupted runs must agree) so it cannot go green for
  the reason the original measurement went red. Verified against an injected
  RNG draw in `finalize()`, which it catches.

### Fixed

- **`chunk_size` below 1,000 is honoured** ([#173]). Anything smaller was
  silently raised to `min_chunk_size`, so a documented public option — one
  `docs/versioning.md` puts in the covered surface — never produced the
  behaviour it documented. Asking for 100 rows on a 5,000-row frame gave five
  chunks of 1,000. A request above `max_chunk_size` was lowered just as
  quietly; both bounds now constrain only the size the chunker picks for
  itself. A pathological `chunk_size=1` is the caller's choice and the caller's
  cost.

  This was a testing-surface bug as much as an API one, and it had already cost
  the project twice. Small deterministic fixtures are exactly where a small
  chunk size is wanted, so **two separate guards were passing for free**:
  #139's per-chunk guard asked for 150 rows on a 900-row frame, and
  `test_chunking_does_not_change_the_facts` asked for 100 on 891 rows. Both got
  a single chunk and neither could reach the condition it guarded.
- **The chunking-invariance test now actually chunks** ([#201]). It profiled a
  fixture whole, then again with `chunk_size=100`, and asserted no fact
  vanished between the two — comparing a run against itself for its entire
  life, green for a reason that had nothing to do with the invariant the
  accumulators are built on. It now counts the chunks the engine consumes and
  asserts the count, so it fails if it ever silently stops chunking again.
- **The report fingerprint no longer keys facts on sampled row indices**
  ([#201]). With the chunk size honoured, the invariance test reported
  `booked 311` removed and `booked 33` added. Classified: **artifact, not a
  datetime bug.** `_pairs_from_kv` matches a label cell followed by a value
  cell, and the non-greedy group backtracks across closing tags, so the sample
  table's `<th>booked</th></tr></thead><tbody><tr><td>311</td>` matched as one
  label of `booked 311` with `56.0` as its value. The key was a *sampled row
  index*; chunking changes which rows the reservoir keeps, so the key moved
  with the value and the fact read as removed-plus-added rather than changed.
  Labels may no longer span a cell boundary. Exactly one fact is dropped —
  1,081 collected becomes 1,080 — and it is that one.

### Deprecated

- **`ReportConfig` warns, and goes in 0.3.0** ([#210]). It was a bare alias for
  `ProfileConfig`, exported in `__all__`, with no signal that it was going away
  — a reader of `__init__.py` could not tell whether it was deprecated or simply
  a second spelling intended to stay. #82 removed the two-constructor ceremony
  and this alias is what was left holding the door open; the door now has a
  closing date. The clock starts at 0.1.0, so by 0.3.0 a full minor has passed
  with a warning in place: the deprecation policy run rather than described.

  The warning fires on **use**, through a module-level `__getattr__` rather than
  an eager alias, so `import pysuricata` stays silent for the users who never
  touch the old name and `dir()` still lists it without firing. All 127 uses of
  the old name across the documentation were migrated to `ProfileConfig`, since
  a deprecation the docs keep teaching is not one.

### Fixed

- **Datetime columns are read at their own resolution** ([#203]). The pandas
  converter cast straight to `int64` and called the result nanoseconds, which
  held only because pandas 2 stored every datetime as `datetime64[ns]`. pandas
  3 defaults to `datetime64[us]`, so the same cast returned microseconds and
  **every datetime statistic came out a factor of 1,000 wrong while still
  looking plausible** — a 2020 timestamp read as 1970, and a freshness check
  reporting data 18,264 days old. The unit is now read from the column and
  scaled; dates outside what `datetime64[ns]` can represent (before 1677-09-21
  or after 2262-04-11) saturate to NaT, the sentinel the accumulator's validity
  window already rejects, rather than wrapping into a plausible wrong date.

  Not a pandas 3 bug, only a pandas 3 *default*: non-nanosecond dtypes are
  constructible on pandas 2 and arrive on their own from parquet and pyarrow.
  `tests/test_datetime_resolution.py` runs on both, and eleven of its cases
  fail against the old conversion under pandas 2.
- **An identifier column is no longer inferred as datetime** ([#203]). The
  datetime sniff counted a successful parse as a date, and pandas 3 parses
  `"T1"` as year 1 — `T` is the ISO 8601 time designator, so a bare identifier
  parses instead of failing and the digits are taken as a year (`T32` → 2032,
  `T123` → year 123). A ticket column of `T0..T680` scored 99.5% dates under
  pandas 3 against 34% under pandas 2. The gate now requires a plausible year
  as well as a parse, set at 1000 — far below the accumulator's own validity
  window, so it excludes parser artifacts without narrowing which historical
  dates count. (The `-2e18` bound made exactly that mistake once.)
- **An all-missing text column no longer fails the whole profile** ([#204]).
  The per-row memory estimate averaged the string lengths of a sample, and
  `Series.mean()` of an all-NA series is NaN, which reached
  `int(estimate * len(s))` and raised `ValueError: cannot convert float NaN to
  integer`. A memory *estimate* failed a run whose statistics were fine. Under
  pandas 2 this was unreachable because `astype(str)` rendered `None` as the
  literal `"None"` and measured four characters; pandas 3 yields NaN. An
  all-missing column now measures zero bytes of text per row, which is also the
  more honest number.

### Changed

- **A datetime column leads with how regular it is** ([#155], 5c.5). The
  strongest fact about a generated series was a table row reading `Interval
  std dev — 0.0 seconds`, filed alphabetically between timezone and weekend
  ratio. A deviation of zero means every gap is identical, and the pane now
  opens with *Every gap is identical: one record every 17.0 minutes, with no
  irregularity at all. That is a generated series rather than observed
  events.* An irregular column reads *this is an event stream, not a
  schedule.*
- **The temporal panels say what they are pictures of** ([#155], 5c.4). Each
  carries its own peak inline, so a 211-record hour and a 2,626-record month
  no longer draw identically with the peaks in a different tab. The year chart
  is dropped when every record falls in one year — `by_year` is a dict, so
  that rendered a single bar at full height, a chart whose only reading is
  "all of it". The per-chart scaling is stated, since heights compare within a
  chart and not between them.
- **High-cardinality columns get a shape pane, and lose the control they
  could not use** ([#155], 5c.3). Phase 5.4 replaced the meaningless
  top-values *chart* on the card; the details pane still opened on `Common
  values` — the same ten bars of one row each. It is now a `Shape` pane:
  distinct against rows, whether anything repeats, the length range, empty
  strings. The Top-N control renders only when it offers a real choice — `Sex`
  rendered three buttons every one of which read `2`, `Cabin` rendered two both
  reading `1`, and `Name` and `Ticket` rendered five above a sentence where no
  chart exists.
- **`psutil` is no longer a runtime dependency** ([#204]). It was declared in
  `dependencies` and imported by no code path under `pysuricata/` — only by the
  memory tests and the recipes in `docs/performance.md`. It is now the
  `pysuricata[system]` extra. Anyone relying on it transitively will need to
  install it explicitly.

  It was not free: psutil publishes no WASM wheel, so `micropip.install(
  "pysuricata")` could not resolve at all, and the browser demo carries a
  hand-written mock distribution purely to get past it. That shim stays until
  0.1.1 is published, because the demo installs from PyPI and the immutable
  0.1.0 metadata still requires psutil.
- **The pandas ceiling admits pandas 3** ([#203]). `pandas~=2.0` and
  `pandas>=2.2.3,<3.0` both excluded it, so installing into a pandas 3
  environment silently pulled pandas back to 2.3.3 — a downgrade discovered
  only when something else in the user's project broke. Now `<4` on both
  requirement lines. The `python_version` split stays: it exists for the
  *floor*, since 2.2.3 is the first pandas publishing cp313 wheels, and
  collapsing it would let a constrained resolver build 2.2.0 from source
  against a Python it never supported.

  The cap turned out to be defending two real incompatibilities after all, both
  fixed above and neither caught by the audit that profiled one clean frame. CI
  now runs a pandas 3 leg so the claim is checked rather than assumed.
- **The label-length reservoir is spent** ([#155], 5c.2). `categorical.py` has
  kept a 5,000-value reservoir of label lengths all along and the report spent
  it on two numbers, `avg_len` and `len_p90`. The distribution is now drawn,
  one bar per distinct length, and on an identifier column the shape is the
  finding: `Ticket` reads *Lengths run 3 to 18 characters, with nothing between
  13 and 15: 851 labels fall below the gap and 40 above it. Two formats in one
  column look like this.* Suppressed below three distinct lengths, where a
  chart of one full-height bar says only "all of them".
- **Boolean cards have no details section** ([#155], 5c.6) — a decision, not an
  omission. Two values, two counts, one bar, all on the card face: nothing is
  withheld, so there is no second level of disclosure to offer. `Breakdown` was
  a two-row table restating the card's own split; `Missing Values` restated one
  fact under a header already carrying it, and unlike the numeric and datetime
  cards it cannot earn its tab back, because boolean accumulators are finalized
  without chunk metadata ([#193]).
- **Normalisation reports collisions, not transformations** ([#155], 5c.1). The
  pane printed original / `lower()` / `strip()` per level, so for `Embarked` it
  said `S → s → S` — a transformation nobody asked about. It now answers the
  question it exists for: whether normalising would **merge** levels. `5 tracked
  levels become 3 under normalisation: 2 groups merge.` When nothing merges the
  tab does not render. The verdict is hedged to the tracked levels, since only
  the top-k are held.
- **The browser demo adopts the minimal redesign** ([#196]). One 600px reading
  column, the report's own paper/ink and blue tokens from
  `docs/design/tokens.css` in place of the demo's separate green palette, mono
  micro-labels, and figures as a label/value ledger rather than boxed tiles. The
  report frame breaks out of the reading column to 1120px, since at 600px the
  report's own cards wrap one per row. Nothing that worked was dropped — the
  log, the streamed 5M-row demo, the JSON download, the version line and the
  sandboxed report frame all survive, restyled.
- **Sub-threshold correlation pairs are now kept per column** ([#154], 5b.6) —
  a deliberate change to the `summarize()` payload, and one of the two the
  invariance harness names as permitted. `corr_top` previously held only pairs
  at or above `corr_threshold` (default 0.50), so on most frames it was empty
  and every per-column pane rendered an empty state. It now holds every numeric
  partner, strongest first, bounded by `corr_max_cols` (default 50). No other
  field changed: every differing key in the golden payload is `corr_top`, and
  every one goes from `[]` to a populated list.

- **The statistics pane opens with the distribution's shape** ([#154], 5b.1).
  Twenty-six key–value rows across two tables, with nothing in the layout
  saying which to read — `Jarque–Bera χ²` carried the same weight as `Median`,
  and **`Std Dev` was printed twice**, once in each table. The nine percentiles
  now sit on the axis the Outliers and Min/Max panes use, as a box with the IQR
  band, whiskers terminating at the band edges, the median protruding past both
  and the mean as a caret above it. Two prose lines spend thresholds the report
  already held and never showed: `Jarque–Bera is 18.79 against a 5.99 critical
  value — far enough from normal to reject it`, and the confidence interval as
  a width (`±1.066`) rather than two endpoints. `Std Dev` is printed once, with
  the moments where it belongs.
- **The per-column correlations pane shows every partner** ([#154], 5b.6).
  It repeated the section-level empty state inside a card — `No significant
  correlations found`, on a column that has partners and simply has no strong
  ones. `Age` has exactly two numeric partners in the Titanic frame, so listing
  both is *complete* information in two rows: **the strongest partner is Fare
  at +0.096**, and the pane says so rather than shrugging. Capped at five with
  a `6 more, all below 0.04` line, on the same diverging bar as the
  section-level list so sign stays position and never colour.
- **Common values rank visibly** ([#154], 5b.3). Five columns become three —
  the ordinals `1ˢᵗ 2ⁿᵈ 3ʳᵈ` were decoration on a list that is already ordered,
  and count and percent are one fact about one value rather than two. **The bar
  is scaled to the most common value, not to 100%**: at 3.2% of 714 rows every
  bar was 3% of its track and all ten looked identical, so the ranking could
  not be seen. Relative scaling hides absolute rarity in exchange, which the
  caption now carries. The pane also says the finding out loud — *All 10 are
  whole numbers, though the column stores 3 decimals. 22.3% of values end in a
  0 or a 5* — two numbers the report already computed and never put next to
  each other. A column where nothing repeats loses the tab entirely rather than
  drawing ten full-width bars over ten equally-common values.
- **Min/Max plots both tails on the fence's axis** ([#154], 5b.5). It was two
  tables headed `Min values` and `Max values`, five rows each of index and
  value — ten numbers, no context. A reader could not tell that **every one of
  `Age`'s five maxima crosses the IQR fence and not one of its five minima
  does**, which is the whole story of that column's tails and was already
  computable. The pane now reads *The low tail is ordinary — all 5 sit inside
  the fence. Every one of the 5 highest crosses it.* above the same figure the
  Outliers pane draws, and gives each row its position (`high · 2.3× IQR`,
  `inside the fence`). Repeated values are marked, so `Age`'s two 0.75s and two
  71s stop looking like four findings.
- **The Outliers pane draws the fence** ([#154], 5b.2). It opened with roughly
  60px announcing `Low Outliers — 0 outliers (0.0%)` over three severity chips
  all reading zero, said the same again for the high side, then listed the
  values in a `rowspan` table with no picture of what they had crossed. An
  outlier is *defined* by a threshold, so the threshold is now the graphic:
  an IQR band with the fence marked and the values beyond it as capsules,
  amber for moderate and rust for high or extreme.
  - **The empty low side became a sentence.** `Age`'s lower fence sits at
    −6.7 years and its minimum is 0.42, so the column cannot have a low
    outlier — a fact from two numbers already on `stats`. The verdict branches
    four ways over it, and the impossibility claim is decided against the
    *exact* minimum rather than the sample, so it is never a confident guess.
  - **One row per value, both verdicts side by side.** The `rowspan` gave a
    value flagged by two methods two rows and a value flagged by one a single
    row, so the table's shape encoded something other than the data. The
    methods' disagreement — `IQR flags 7; MAD flags 1` on Titanic's `Age` — is
    now stated in prose instead of left as two sets of chips to reconcile.
  - Values closer together than one mark collapse into a capsule carrying its
    count, with the values in its `title`.
- **The closed `Details` row names what is behind it** ([#154], 5b.8) —
  `statistics · 48 common values · 5 lowest and highest · 11 outliers`. The tab
  set was known at render time and never printed, so the word "Details"
  promised nothing and a reader had to open every card to learn whether opening
  was worth it. `11 outliers` is the reason to open it; `no outliers` is the
  reason not to. Each tab carries its count too, so the right one can be picked
  first time.
- **The Missing Values pane renders only when it knows something** (5b.7):
  missing > 0 **and** more than one chunk. With a single chunk it stated one
  fact four times — a Present stat, a Missing stat, a two-segment bar and a
  one-segment chunk strip — under a header already flagging the percentage. The
  only thing it knows that the card face does not is *where in the read* the
  gaps fall. Applies to numeric and datetime; categorical and boolean are never
  handed chunk metadata to gate on, which is filed as [#193].
- **The active tab's underline moved onto the label.** The button is 44px tall
  because it is a tap target, so a `border-bottom` on it painted the rule ~29px
  below the word — a second hairline floating under the strip. The tinted
  background went with it: a filled tab competing with an underline says the
  same thing twice, and the tint was a colour on neither scale.

### Fixed

- **A second file in the same session profiles again** ([#196]). WORKERFS leaves
  the mount point's child nodes behind on unmount, so mounting the second file
  onto `/data` failed with an opaque `ErrnoError`. Every run now mounts under
  its own directory and releases the previous one.
- **A failed run no longer inflates the next run's heap figure.** The previous
  report stayed pinned in the Python heap because cleanup ran only on the
  success path, so the next run's "peak heap" was partly measuring the last one.
  Cleanup now runs on the error path too.
- **The page no longer shows one run's figures while the next is still going**,
  and a late message from a superseded run can no longer repaint a panel a newer
  run already owns. Runs carry an id, and every run begins by clearing what the
  last one left.
- **Picking the same file twice works.** Re-selecting an identical file fires no
  `change` event, which read as a dead button; the input is reset before the
  file is handled.
- **A worker that cannot be constructed says so.** It previously threw past the
  rest of the script, leaving three disabled buttons and no explanation.
- **The report reaches the frame through a blob URL rather than `srcdoc`.**
  Chrome silently drops a `srcdoc` document past roughly 700 KB — no error, no
  console warning, just a blank frame. Measured: 684 KB renders, 721 KB does
  not, which any wide or high-cardinality frame clears.
- **The invariance harness was comparing 45% of the facts it collected.**
  `report_fingerprint.diff()` read both fingerprints into a `dict`, and the
  keys were not unique — `age` and `fare` both emit a `Median` row, one
  histogram emits 64 `data-count` attributes under one key. Of 559 facts
  collected, **308 were never compared**, and two dead entries had been sitting
  in the fixture because no run could see them. `kv::` keys are now scoped to
  their column card, and `diff()` compares the multiset under each key. The
  fixture grows to 1,000 lines of previously-uncompared facts; no value
  changed. This is the check that guards the whole redesign, so it is worth
  saying plainly that it was half blind.
- **The histogram fills its card** ([#147]). At a 1240px viewport the `<svg>`
  element was already 1,099px wide and the bars occupied **356px** of it — 68%
  blank — because `preserveAspectRatio` defaults to `xMidYMid meet` and the
  container's fixed height was the limiting dimension. Bars now cover 100% of
  the plot at 1240px **and** at 390px.

  The fix is a split: bars, gridlines and axis rules go in an SVG stretched
  with `preserveAspectRatio="none"` and `vector-effect="non-scaling-stroke"`,
  and every label is HTML at a percentage offset. That is the only arrangement
  that gives a full-width chart *and* 11px labels at every viewport — uniform
  scaling makes them 28.8px on a desktop and 3.5px on a phone.
- **Bars no longer merge on a narrow screen.** The gap was `bar_width - 1`, a
  1-unit gap in viewBox space that scales with x: 1.1px at a 1,100px plot and
  **0.28px at 284px**. Bars are drawn edge to edge with a non-scaling `--paper`
  stroke, which is 1px by construction.
- The y gutter is fixed at 44px, so the plot's left edge is identical on every
  numeric card and bars line up down the page.
- Nine x tick labels are written and tiered by importance; CSS drops them to
  five and then three. Tier 1 is the two ends **and the midpoint** — a range
  with no middle says nothing about whether a distribution is centred.
- The x unit moved from the right end of the axis into a caption line carrying
  the bin count and the exact peak (`years · 25 bins · peak 83 rows at
  25.9–29.1`). At 1,100px the unit and `ROWS` were a hand-span apart and had
  stopped reading as a pair; the peak matters more now that the y labels
  abbreviate to four glyphs.
- The empty state is a sentence rather than `No data` centred in a blank
  420×200 canvas, which read as a chart that had failed.

[#147]: https://github.com/alvarodiez20/pysuricata/issues/147

- **The top-missing list says what it counted** ([#182]). It showed five
  columns and stopped, with no indication there were more — `total_significant`
  was computed, stored on the result object and printed nowhere. A frame with
  23 partially-missing columns now ends the list with `+ 18 more columns with
  missing values`. A list that truncates in silence is worse than a shorter
  list, because the reader cannot tell they are seeing part of the answer.
- **The missing threshold is defined once.** It was in three places —
  `MissingColumnsAnalyzer.MIN_THRESHOLD_PCT` at 0.0, the factory at 0.5, and
  `ProfileConfig` at 0.0 — and since the render path reads the config, the
  factory's 0.5 had never applied to a report. The value stays **0.0**, which
  is what every shipped report has used; raising it would quietly drop columns
  from people's summaries.
- **The no-missing state is a sentence, not a fake row.** It rendered a list
  item with a `<code>` reading `No missing data`, a `0 (0.0%)` figure and a
  zero-width bar — a table row impersonating data, with an element drawn to
  represent nothing.

- **Quality chips print the number they already carry** ([#184]). Every chip
  emitted `data-value` and `data-threshold` and displayed neither on the
  categorical, boolean and datetime cards, so a card said `Missing` where it
  could say `19.9% missing`. `Dominant category` and `Empty or zero` gained
  theirs too. `High cardinality`, `Monotonic ↑` and `Positive-only` stay bare
  on purpose — those need a phrase rather than a numeric prefix, and inventing
  one would read worse than the word.
- **`Empty strings` was the wrong name and is now `Empty or zero`.** The
  accumulator counts `value == "" or value == "0"`. Putting the number on the
  chip is what made that visible: titanic's `SibSp` and `Parch` profile as
  categorical and rendered `608 empty strings` and `678 empty strings`, when
  they have 608 and 678 *zeros* and not one empty string between them. The
  vague label had been hiding a false one.

- **A histogram y-axis count label is now guaranteed at most four glyphs**
  ([#183]). It used to *prefer* short and not guarantee it: `12,500` came out
  as six characters and `12.5M` as five. That matters because the redesigned
  chart fixes the y gutter at 44px so the plot's left edge does not move
  between columns — a wider label either overflows it or forces the gutter to
  breathe, and a breathing gutter loses the alignment the fixed one buys.

- **A column whose values never repeat no longer reports zeros it cannot
  know** ([#181]). `Cabin` printed `Entropy NaN` — the only `NaN` in the whole
  report — alongside `Rare levels 0 (0.0%)`, `Top 5 coverage 0.0%` and
  `Mode % 0.0%`. All four come from the top-k sketch, which was empty and
  correctly so: Misra-Gries only guarantees a survivor above `n/(k+1)`, and
  `Cabin`'s most frequent value appears 4 times in 204 against a threshold of
  exactly 4. The four cells now render an em dash carrying the reason. `NaN`
  announced itself; `0.0%` did not, which is why it is the more dangerous of
  the two.

### Changed

- **No emoji anywhere in the report** ([#180]). Eight went: a `🔗` before a
  heading reading *Correlations*, a `📋` on every header tooltip, three `✓`
  in empty states whose sentence already said it, and `❓ 0️⃣ ➖` on the
  quality indicators. `∞` stays — it names the thing it counts and is
  mathematical notation, not an emoji.

  Two reasons. They render at a different weight and baseline on every
  platform, which shows in a report that otherwise sets every figure in one
  mono face; and inline glyphs are **announced by screen readers**, so `✓` in
  front of "No missing values detected" was read out as "check mark no missing
  values detected".

### Changed

- The `pypi` environment no longer requires a reviewer, so **pushing a version
  tag publishes to PyPI with no confirmation step**. `cd.yml` and
  `docs/versioning.md` said otherwise and now say this; a comment asserting a
  safety property that has been removed is worse than no comment, because it is
  what someone reads to decide whether a push is reversible. The `guard` and
  `smoke` jobs stand in for the reviewer.
- **The PyPI summary now says what the tool does.** It read "A lightweight EDA
  tool inspired by the curious nature of suricates. Built just for fun 🔬",
  which describes the project's origin rather than the package someone is
  deciding whether to install. It is now "Streaming EDA profiler: one pass over
  pandas or polars, bounded memory, a self-contained HTML report."
- The PyPI sidebar gains Homepage, Documentation, Changelog and Issues links —
  it carried only a repository link — and the package declares keywords, so it
  is findable by search rather than only by name.

  Note that release metadata on PyPI is immutable per version, so none of this
  changes the 0.1.0 page. It takes effect with the next release.

### Fixed

- **Every interactive target in the report is now at least 44×44** at 390px
  ([#122]). Ten kinds were smaller — the card info links at 24×24, the filter
  tabs at 29px tall, the search field at 33, pagination at 39×29, the
  needs-attention rows at 16. The two footer links stay small on purpose: they
  sit inline in a run of metadata text, which is WCAG 2.5.8's own exception.
- **The pagination page numbers are buttons**, not `<span>`s with click
  listeners. They had no role, could not be reached by keyboard, and announced
  "2" as their entire accessible name.
- **204 contrast failures in light and 93 in dark**, found by measuring every
  text node against the background actually painted behind it rather than
  against the page. They came from eight framework colours printed on a 10%
  wash of themselves — outlier severity, correlation strength, correlation
  sign. All eight now read the design tokens; the sign reads no colour at all,
  since `_00-tokens.css` already encodes it by which side of centre the bar
  sits on. Both themes are now clean over 2,334 text nodes.
- **The quality flags carry a shape as well as a hue** — circle, triangle,
  square. `--q-good` and `--q-bad` are 1.05:1 apart in luminance, so in
  greyscale a chip reading "Positive-only" and one reading "24.28 heavy-tailed"
  were the same chip. The marks are drawn rather than typed so screen readers
  do not read out a check mark before a label that already says what it is.
- **`prefers-reduced-motion` is honoured**, against 49 transitions and
  animations that previously ignored it. The state change is kept; only the
  travel is removed.

### Changed

- **The compatibility shim in `_00-tokens.css` is gone** ([#122]), along with
  the 285 dead selectors it was holding up — the pre-redesign missing-values
  section, the old correlation heatmap, and the datetime and boolean markup the
  redesign replaced. Verified by computed style across 143,068 elements in four
  reports and both themes: removing them changes nothing.
- `--accent-color` used to map to `--q-good`, so the olive that means "passes a
  check" was also drawing focus rings, hover borders and the active tab. It now
  reads `--data-1`, which is what the header and the details toggle already use
  for `:focus-visible`.
- The untokenised-colour ratchet drops from 88 to 70. Eleven of those needed no
  decision at all — they were sitting in rules for markup nobody renders.

[#122]: https://github.com/alvarodiez20/pysuricata/issues/122
[#180]: https://github.com/alvarodiez20/pysuricata/issues/180
[#181]: https://github.com/alvarodiez20/pysuricata/issues/181
[#182]: https://github.com/alvarodiez20/pysuricata/issues/182
[#183]: https://github.com/alvarodiez20/pysuricata/issues/183
[#184]: https://github.com/alvarodiez20/pysuricata/issues/184

## [0.1.0] - 2026-08-16

**The first release that promises anything.**

The sixty-one releases before this one were not decisions. `version-check`
required a bump on every pull request and `cd.yml` published on every push to
`main`, so one merged PR was exactly one PyPI release, unconditionally — a
rewritten kernel and a fixed typo the same size of event. `0.0.71 → 0.0.72`
could not describe a change, because the version was incremented by the *act of
merging* rather than by a judgement about what merged.

That is fixed, and this is the first version to arrive on purpose:

- **Publishing happens on a pushed tag**, through an ordered pipeline that
  smoke-tests the built wheel in a clean virtualenv before anything reaches
  PyPI, and creates the GitHub release only after PyPI has the package.
- **`docs/versioning.md` states the contract.** At 0.x, a minor bump is what a
  major bump becomes at 1.0 — it is the one allowed to break you, and a patch
  never is. With an enumerated covered surface and an explicit not-covered list.
- **`pysuricata~=0.1.0` is now a real guarantee.**

Why 0.1.0 and not 1.0.0: the API is demonstrably still moving. `ReportConfig`
is waiting to be deprecated, `numeric_sample_size` is not a passthrough keyword,
the five checkpoint options are due to collapse into `progress_report=`, and the
triage block is about to be reshaped. `1.0.0` is a promise you cannot withdraw.
The five gates for it are listed in `docs/versioning.md`.

### Honesty about approximation
Three figures stopped claiming more precision than they have. The quantiles are
reservoir estimates and now say so; the duplicate count carries the bound that
actually applies to it rather than the sketch's own, which understated the error
by up to two orders of magnitude; and the label-length statistics, which had
been reading from a dict that never carried them and printing `NaN` for every
categorical column in every report, now print the values the accumulator had all
along.


### Fixed
- **`--data-3` could not legally carry a standalone mark (#156).** At `#7FA0B5` it was **2.63:1** on the paper — below the 3:1 non-text minimum (WCAG 1.4.11), which is the entire job a third step exists for. It is now `#5C7F99`.

  | | paper | track |
  |---|---:|---:|
  | old `#7FA0B5` | **2.63** ✗ | 2.24 ✗ |
  | new `#5C7F99` light | 4.03 ✓ | 3.42 ✓ |
  | new `#5C7F99` dark | 4.09 ✓ | 3.51 ✓ |

- **`--data-3` now carries no text.** At 4.03:1 against the paper and 3.83:1 against the ink it reaches neither text minimum, so it is a fill and never a label background. `--on-data-3` is removed, and a composition-bar segment on that step sends its count to the legend — the mechanism a too-narrow segment already uses.

### Added
- `test_data_3_carries_no_text` asserts a **failure** on purpose: raising `--data-3` far enough to carry text breaks the build and forces the conversation, rather than quietly reintroducing a label nobody measured. Paired with `test_data_3_can_stand_alone`, which asserts the property the old value failed.

### Note
The handoff also proposed `#4A6E8A` for the **dark** `--data-4`. **Not taken.** It separates from `--data-2` by only **1.95:1**, and those two carry the two-dataset comparison, which needs 2:1; it is also 1.27:1 from `--data-3`, so two adjacent steps would be indistinguishable. The repo's `#2C4A62` gives 3.36 and 2.19 and is kept. The handoff labels its dark section *"PROPOSED, NOT YET VERIFIED as a whole"* — it measured each step against the surfaces, not the steps against each other. Every ratio above was recomputed here rather than copied.

### Fixed
- **Per-column per-chunk missing counts are now produced (#139).** `mark_chunk_boundary()` was only ever called from `finalize()`, so the recorded boundaries counted **renders**, not chunks — one for an uninterrupted run, two for a checkpointed one, never the chunk count. The engine now marks a boundary after every chunk it consumes.

  This is the root cause of the impossible figure found in #154: a segment reading `data-missing="1563"` on an 891-row frame — 175.4% — because a single boundary accumulated every chunk's counter while being sized as one chunk.

  Verified on 5,000 rows at `chunk_size=1000`: five boundaries, tiling `[0,999] … [4000,4999]` with no gap or overlap, counts summing to the column's 1,200 missing, and no chunk reporting more missing than it has rows.

- **#120's by-chunk view is unblocked** and renders — five segments for five chunks. Small frames still degrade to the one-row-per-column view, which is the common case.

### Changed
- **A details tab renders only when it has something to say (#154, 5b.4).** The Missing Values pane rendered on **every** column, including ones with no missing values, where it drew a 100%-present bar and a one-segment chunk strip reading `0.0%`. A click to learn nothing. All four card types now drop it when the column is complete; the order of the remaining tabs is unchanged, so a tab appears or does not but never moves.
- The **Correlations** pane no longer repeats the section-level empty state inside a card. It renders only when the column has a correlation above the threshold.

### Fixed
- **The report was rendering 175.4% missing.** Dropping the empty panes made the invariance fingerprint lose four facts, which is what sent me to look at them:

  ```
  attr::::missing  0        a pane reporting nothing
  attr::::missing  1563     on an 891-row frame
  attr::::pct      175.4    ...that is 175.4% missing
  ```

  The second pair is #139's chunk metadata — which counts *renders* rather than chunks — drawn as a severity-coloured segment inside a pane on a column that had **no missing values at all**. The harness flagged that facts had disappeared; the facts turned out to be impossible. A test now asserts no segment claims more missing rows than the frame has, and no percentage exceeds 100.

### Added
- **A ratchet on colour (#148).** #110's acceptance was that four retired column-type hues do not appear in `static/css/`. All four are clear — and there are **94 distinct hex values** in the stylesheets, including Tailwind's blue-600, green-500 and amber-500 and Material's red-700. The ban list named the accents being replaced and missed everything that walked in behind them.

  A ban list can always be outgrown by a colour nobody thought to ban, so the assertion runs the other way: **every hex outside `_00-tokens.css` must equal a value the token file defines.** That cannot be satisfied today — 88 file/value pairs do not — so it ships as a ratchet against a recorded baseline. A new literal fails immediately; tokenising one also fails, loudly, telling you to shrink the baseline. The number only goes down.

  Two exclusions, both narrow: `var(--paper, #FBF9F5)` is a fallback *for a token that exists*, not an untokenised colour; and `body.suricata-standalone` sits outside `#pysuricata-report` and so cannot read tokens scoped there — the only place in the stylesheets that has to repeat a palette value, and the file already said so.

  The first draft of this test asserted that "the nine named colours" from #110 are clear. **There is no such list** — `test_design_system.py` carries four. The invented list went red on `#93c5fd`, which is how I found out it was mine rather than the project's. It now reads the real list from the test that owns it.

  Deleting the 88 is deliberately not in this change. `_12-missing.css` alone holds 27, and 212 of its 405 rules reference no class in a rendered report — but `.miss-strip` is the by-chunk view blocked by #139 and `.corr-scale` is the ranked-list correlation route, so both are *ahead of their data* rather than dead. Mass-deleting on one fixture is the mistake #158 made with the chart rules, twice. That needs the multi-route fixture corpus #124 is meant to establish.

### Changed
- **Publishing is triggered by pushing a version tag, not by merging (#159).**

  `cd.yml` triggered on `push: branches: [main]` while `version-check` required a bump on every pull request. Each rule is defensible alone; together they made **one merged pull request exactly one PyPI release, unconditionally**. A rewritten kernel and a fixed typo were the same size of event, so `0.0.71 → 0.0.72` carried no information and nobody could pin against anything. Semantic versioning was not un-adopted — it was unreachable, because the version was incremented by the *act of merging* rather than by a judgement about what merged.

  ```bash
  git tag v0.1.0 && git push origin v0.1.0
  ```

- **`version-check` validates a bump instead of demanding one.** A pull request need not bump — **this one does not, and that is the first demonstration of the change**. If it does bump, `scripts/check_version.py` asserts the step is legal (one component raised, the ones below reset, nothing skipped, no downgrade) and that a matching changelog section exists.

- **The pipeline is ordered: `guard → build → smoke → publish → release`.** `release` and `publish` previously ran in parallel with no `needs:` between them, so **PyPI could receive a package whose tag step had failed**.

- **`actions/create-release@v1`** — archived by GitHub in 2021, still running Node 12 — replaced with `gh release create`. `checkout@v3` and `setup-python@v4` raised to v4/v5.

### Added
- **`smoke`** — the built **wheel** is installed into an empty virtualenv on 3.10 and 3.14, asked for its version, then made to profile a frame and check `schema_version`. CI tests the repository; this tests the artifact, which is not the same object: a missing package-data glob ships a wheel with no templates and a green test suite.
- **`scripts/release_notes.py`** lifts the `## [X.Y.Z]` section out of `CHANGELOG.md` and **refuses to release a version that has none**. The changelog already enforced on every PR becomes the release page for free.
- **`docs/versioning.md` (#160)** — the contract. At 0.x a minor bump is what a major becomes at 1.0, with an enumerated covered surface and an explicit not-covered list. The exclusion for *the exact value of any approximate figure* is already load-bearing: recent releases changed how quantiles are presented and what bound the duplicate count carries, and neither should force a minor bump.
- `tests/test_release_tooling.py` (33 tests).

- **The note editor was styled before the redesign and never revisited.** It asked for `--text-primary` on `--bg-secondary` — **neither of which is defined anywhere in the stylesheets** — so the textarea had no colour and no background of its own and fell back to the browser's. Around that: a 2px border, a 4px radius, and a `rgba(59, 130, 246, .1)` focus glow, which is Tailwind's blue-500 and appears nowhere else in the report. A `padding: 0` override then set the text flush against the border.

  It now looks like the note it is about to become — same type, same measure, the same `--q-good` left rule, in the same column — so committing an edit changes the text and nothing else. Previously the text jumped ~100px right on save, because the empty block is a two-track grid and the filled one is three. Focus changes the weight of the rule that is already there instead of drawing a glow around a box that is not.

  The styling was also split across `_03-summary.css` and `_13-utilities.css` at different specificities, so the focus rule in the second file silently lost to the base rule in the first. It lives in one place now.

### Note on the Python floor
`check_version.py` first used `tomllib`, which is standard library only from **3.11**, while this project's floor is **3.10**. It did not break the script — it broke the whole *test module*, because importing the tests imports the script. Nothing local caught it (every interpreter here is newer); **CI on 3.10 did**, and only because #166 had just made the matrix run on this branch at all. The version is now parsed with a regex scoped to the `[project]` table, and a test asserts neither script imports anything newer than the declared floor.

### Note
`check_step` shipped its central rule wrong in the first draft: it asked that only one component *change*, which **rejects `0.0.72 → 0.1.0`** — the exact release this reform exists to enable — because bumping minor forces patch from 72 to 0. A reset is part of the bump, not a second decision. Caught by running the rule over real cases before trusting it, and kept as a test.

### Requires a repository setting
Trusted Publishing needs a one-time configuration on PyPI (publisher: this repo, workflow `cd.yml`, environment `pypi`) and a `pypi` environment in GitHub settings with a required reviewer. Until that exists the `publish` job fails — deliberately, rather than falling back to the long-lived `PYPI_TOKEN`.

Planned work is tracked in
[`docs/roadmap.md`](https://github.com/alvarodiez20/pysuricata/blob/main/docs/roadmap.md),
[`docs/UX_ISSUES.md`](https://github.com/alvarodiez20/pysuricata/blob/main/docs/UX_ISSUES.md) and
[`docs/integration.md`](https://github.com/alvarodiez20/pysuricata/blob/main/docs/integration.md).

## [0.0.73] - 2026-08-16

### Fixed
- **Label length printed as `NaN` for every categorical column (#155).** `_right_stats` read `avg_len` and `len_p90` out of `cat_stats` — the dict `_compute_categorical_stats` builds — and that dict has never carried either key, so `.get(..., float("nan"))` and `.get(..., "—")` returned their defaults every time.

  | column | was | is |
  |---|---|---:|
  | `Embarked` | `NaN` | 1 |
  | `Name` | `NaN` | **26.97** |
  | `Ticket` | `NaN` | 6.75 |
  | `Sex` | `NaN` | 4.71 |

  The design handoff reported this as an `Embarked` quirk about one-character labels. It was neither: `Name`, whose labels average 27 characters, printed `NaN` just the same. The accumulator had the right answers throughout — same shape as #139, a field read off an object that does not carry it, failing quietly because the call site supplied a plausible default.

- The em dash now means *absent*. It was standing in for *read from the wrong place*, which is a different thing and is what hid this.

### Added
- `tests/test_label_length.py` (18 tests), including a guard aimed at the cause rather than the symptom: it asserts the derived dict still does **not** carry those keys, so reintroducing the old read fails even on a day the dict happens to have them.

## [0.0.72] - 2026-08-16

### Changed
- **The categorical, datetime and boolean cards are restacked (#158).** #114 did this for the numeric card and left the other three on `.triple-row`, so a report mixing column types showed two different card architectures side by side — which is more jarring than either alone.

  All four now emit a full-width chart above a single `vstat-row`. Measured at 390 × 844 on `docs/assets/titanic.csv`, details collapsed:

  | card kind | before | after |
  |---|---:|---:|
  | categorical (`SibSp`) | 923px | **648px** (−30%) |
  | boolean (`Survived`) | 475px | **395px** |
  | numeric (`Fare`) | 820px | **820px** — unchanged, as intended |

  `_build_stat_row` moves to `CardRenderer`; four copies would drift.

- **170 lines of dead CSS removed** — the `.triple-row` grid and the `.stats-left` / `.stats-right` table rules, which styled boxes nothing emits any more.

### Fixed
- The boolean split bar was `height: 100%` inside a box that set a height. `.var-chart` does not, so it resolved against nothing and collapsed the bar to 29px. The height is now stated rather than inherited from a parent that stopped providing one.

### Note
Three mistakes were made getting here and are worth recording, because each was caught by measurement rather than by review.

1. Rescoping `.triple-row .chart` → `.var-chart` in `_06-cards.css` made the *old* layout's rules apply on top of the ones #114 wrote for the new one. The numeric card grew 40px. That file keeps #114's rules and simply drops the dead blocks.
2. A comment three lines above `.card-controls` contained the words `.triple-row`, and the block-remover's `[^{}]*` crossed newlines — so it deleted the comment **and the rule below it**. Comments are masked before matching now.
3. Type-specific chart rules were stripped before being rescoped, so datetime lost its chart sizing. Order matters: rescope, then strip.

## [0.0.71] - 2026-08-16

### Fixed
- **`lint` and `test` had never run on `main` (#125).** `ci.yml` triggered on `pull_request` only, so every check the project relies on ran against a *proposed* merge and never against the result. A PR is tested against `main` as it stood when the branch opened, so two individually green PRs can be red together and nothing would notice.

  `ci.yml` now also triggers on push to `main`. `version-check` and `changelog-check` are excluded there by an explicit `if` — both diff against `origin/main`, so on a push to `main` they would compare a commit with itself and fail by construction. They are pre-merge gates by nature, not checks that were forgotten.

- **Stacked PRs ran zero checks.** Every workflow keyed on `pull_request: branches: [main]`, so a PR targeting another branch matched no workflow at all — which is what happened to sixteen of the redesign PRs until they were retargeted at `main` before merge. The branch filter is removed from `ci.yml`, `accuracy.yml` and `docs-check.yml`.

- **`codecov/patch` failed on pull requests with nothing to measure.** A docs-only or config-only change touches no measurable lines, and Codecov reports "no coverage data" as a failure by default — so a red check appeared on pull requests that could not possibly have a coverage problem. That teaches the reader to ignore the check that *does* have something to say. `if_not_found: success` now says what was meant.

- **`.gitignore` covers the local scratch directory and `README.draft.md`.** A `git add -A` during a conflict resolution swept four scratch files into a commit; ignoring them makes that impossible rather than careful.

### Note on the original report
The issue title said *CI has never run on main*. That was imprecise: `accuracy.yml` and `docs.yml` already carried `push: branches: [main]`. What had never run on `main` was `ci.yml` — which is where `lint` and `test` live, so the substance of the finding holds and its scope was smaller than stated.
## [0.0.70] - 2026-08-16
## [0.0.68] - 2026-08-16

### Removed
- **`docs/assets/titanic_report.html` is no longer committed (#143).** It is generated by `scripts/regenerate_example_report.py`, which all three docs workflows run *before* `mkdocs build`, so the published site has always carried a current copy.

  Committing it as well meant every rendering change either produced a megabyte diff or produced none — and for 45 versions it produced none. The committed file was **1,180,196 bytes from 0.0.17**, against 600,028 bytes of real output, still carrying the base64 logo that was replaced at 0.0.48.

  Regenerating once would have reset that and re-armed the same trap. Ignoring it fixes it permanently.

### Note on the original report of this
The issue was filed claiming that *readers of the documentation see a pre-redesign report*. **That was wrong**, and it was my error: I read the file's git history rather than the URL the README links to. `docs.yml` regenerates and deploys on every push to `main`, so the published report was current the whole time — `last-modified` minutes after the previous merge, 600,049 bytes, with the inline-SVG logo. What was stale was a file nothing serves and no CI job reads, which makes this repo hygiene rather than the credibility problem it was filed as.

### Changed
- `docs/contributing.md` documents the one-line regeneration step for local `mkdocs serve`. Verified that `mkdocs build --strict` passes with the file absent.

## [0.0.68] - 2026-08-16

### Removed
- **`docs/assets/titanic_report.html` is no longer committed (#143).** It is generated by `scripts/regenerate_example_report.py`, which all three docs workflows run *before* `mkdocs build`, so the published site has always carried a current copy.

  Committing it as well meant every rendering change either produced a megabyte diff or produced none — and for 45 versions it produced none. The committed file was **1,180,196 bytes from 0.0.17**, against 600,028 bytes of real output, still carrying the base64 logo that was replaced at 0.0.48.

  Regenerating once would have reset that and re-armed the same trap. Ignoring it fixes it permanently.

### Note on the original report of this
The issue was filed claiming that *readers of the documentation see a pre-redesign report*. **That was wrong**, and it was my error: I read the file's git history rather than the URL the README links to. `docs.yml` regenerates and deploys on every push to `main`, so the published report was current the whole time — `last-modified` minutes after the previous merge, 600,049 bytes, with the inline-SVG logo. What was stale was a file nothing serves and no CI job reads, which makes this repo hygiene rather than the credibility problem it was filed as.

### Changed
- `docs/contributing.md` documents the one-line regeneration step for local `mkdocs serve`. Verified that `mkdocs build --strict` passes with the file absent.

## [0.0.67] - 2026-08-16

### Removed
- **Every emoji in the report (#157), and the dead JavaScript behind four selectors (#144).**

  #119 removed three glyphs from the correlations section and guarded them *there*. The guard only ever looked at one section, so the rest survived. A sweep of the whole render layer found them in five more places, and two of those could not have been found by searching for the glyph:

  | where | how it hid |
  |---|---|
  | numeric outlier and context notes | `💡`, `ℹ️` — plain text, listed in the design handoff |
  | severity headers in the categorical, datetime and numeric cards | `🚨 ⚠️ ⚡ ✅` interpolated from a `severity_icon` variable, so only reachable on a branch a simple fixture never took |
  | **`boolean_card.py`** | written as **HTML entities** — `&#128680;`, `&#9888;`, `&#9889;`, `&#10004;` — invisible to any search for the character |
  | `_09-datetime.css`, `_12-missing.css` | `content: '💡'` on a pseudo-element, so present in no Python source at all |
  | `pagination.js` no-results block | `🔍` |

  The severity class already carries the severity; the glyph was redundant colour. They also render inconsistently — `ℹ️` takes a coloured emoji presentation on some platforms, so a note styled as quiet grey text acquires a blue box.

- **The missing-values tab switcher** (~1,750 characters). #120 replaced the tabs with a chunk-count route and `test_missing_section_views.py` asserts the markup stays gone, so this listened for clicks on elements no report emits.
- **A `.details-panel` fallback** for a pre-refactor layout — a second code path nobody exercised, reachable only when the primary branch failed to find its target, which is exactly when a silent `return` is wrong.

### Added
- A repo-wide emoji guard covering the rendered report *and* the render sources, plus a separate assertion that **no numeric character reference above U+2000** appears anywhere in `pysuricata/` — the check that would have caught the boolean card.

- **`tests/test_missing_spectrum_severity.py`** (252 tests) drives the per-chunk missing spectrum in all four card renderers directly, across all four severity bands. Those blocks return early without `chunk_metadata`, which #139 shows is never produced — so in a real report they never draw, and that is precisely why four emoji survived a sweep that believed it was complete. Covering them proves the removal held on branches no fixture reaches, and asserts the severity still lands in the markup now that the glyph does not carry it.

### Changed
- `_KNOWN_DEAD` in `tests/test_description_editor.py` is down to one entry. `compact-row` stays: it is not stale but ahead of its data (#139). An exemption is a promise to come back, not a place to leave things.

## [0.0.66] - 2026-08-16

### Fixed
- **The duplicate-row figure carried the wrong error bound (#161).** `approx_duplicates()` returns `rows - distinct`. `rows` is exact and `distinct` is a sketch estimate, so the entire *absolute* error of the distinct estimate lands on a quantity that is usually far smaller — the relative error is multiplied by `distinct / duplicates`.

  On 200,000 rows containing exactly 2,000 duplicates, the distinct estimate was **0.48% low** (well inside spec) and the reported duplicate count came back **2,942 — 47% high**. The amplification factor is 99×, and 0.48% × 99 is 47%, so the model and the observation agree to the digit. At a 0.1% duplicate rate the error on the reported figure is around 1,100%.

  The card was not silent: it printed `≈ KMV sketch`, which reads as ±1–2%. That is the error on **distinct**, not on the number being shown. An approximation marker implying the wrong order of magnitude is worse than none, because it turns a naked estimate into a confident one.

  The figure now carries the bound that actually applies to it:

  | case | shown |
  |---|---|
  | distinct count exact (any frame under `k` distinct rows) | `100` · **exact** |
  | estimated, count above its uncertainty | `2,942` · **± 2,178 · KMV sketch** |
  | estimated, count below its uncertainty | `< 2,199` · **below sketch resolution** |

- **Small frames were being told their exact count was approximate.** KMV counts exactly until it has seen `k` distinct values, so most frames have no estimation error here at all. `≈ KMV sketch` claimed otherwise on every one of them.

### Added
- `RowKMV.duplicates_uncertainty()`, `.duplicates_are_resolvable()` and `.kmv_is_exact()`.
- `tests/test_duplicate_uncertainty.py` (17 tests), including the property that matters: across duplicate rates of 0.1%, 1%, 10% and 50%, the truth lies inside the stated interval.

### Note
The first version of this change reported `< 10 (below sketch resolution)` for an 891-row frame whose duplicate count was known **exactly** — understating what the sketch knew, which is the opposite of the intent. `tests/test_report_data_invariance.py` caught it by losing the `duplicates` fact. The fix restored the fact, so no fixture was regenerated.

## [0.0.65] - 2026-08-16

Documentation only.

### Changed
- **`docs/integration.md`** replaced with the updated design handoff, which adds phases **5b** (numeric details pane) and **5c** (categorical, datetime and boolean details panes) and a revised commit sequence.
- **Its status table was rewritten.** The handoff audited a snapshot from before phases 2–7 landed, so four rows read *not applied* for work that has since shipped — flag chip values (#137), the correlations empty state (#138), the correlations emoji (#138) and the missing-values tabs (#140). Every row was re-verified against `main` by rendering a report and reading the markup. Where the handoff was still right, an issue is linked.

### Added
- `docs/design/` — the handoff README, the proposed `tokens.css`, and the updated contrast test kept as `contrast_test.reference.py` (deliberately not named `test_*`, so pytest does not collect a file that asserts token values the repo does not have yet).

### Issues filed from this handoff and from roadmap v11
#154 numeric details pane · #155 categorical/datetime/boolean details panes · #156 `--data-3` value change and the three untestable rules · #157 surviving emoji · #158 three card types still on `.triple-row` · #159 the release pipeline · #160 versioning and `0.1.0` · #161 the duplicate count's propagated error

## [0.0.64] - 2026-08-16

### Fixed
- **The quantiles were sampled estimates printed as if exact (#146).** `Q1`, `Median`, `Q3`, `IQR` and `MAD` come from a reservoir holding `numeric_sample_k` values; on a 60,000-row column that is a third of the data. They rendered to four significant figures in the same typography as `Min` and `Max`, which **are** exact — #118 made the extremes come from every value precisely so they would stop being sampled. On a standard normal column the report printed a median of `0.003684` where the true median is `-0.00252`: the right order of magnitude, the wrong sign, and four digits of implied precision.

  The five sampled statistics now carry `(≈)`, the marker the distinct count has used since #41. `Min`, `Max` and `Mean` deliberately do not — they see every value, and the difference should be visible.

- **The pin button's tooltip did not follow its state.** `setPinned` updated `aria-label` and not `title`, so the tooltip read *Unpin header* on a header that was already unpinned — a sighted user and a screen-reader user were told different things about the same control.

### Changed
- **Header controls reordered to pin → download → theme.** The pin governs the bar it sits in, so it leads; the theme toggle is set once and left, so it trails.

### Added
- `pysuricata/render/sampling.py` — one predicate, `quantiles_are_sampled()`, answering whether the reservoir saw the whole column. It returns `False` when the sample is absent or empty: those quantiles were not drawn from a reservoir at all, and marking them would attach a warning to the numbers that least need one.
- `tests/test_sampled_quantiles.py` and `tests/test_header_actions.py` (34 tests).

### Note on the justification
The first write-up of #146 argued from run-to-run variance and was **wrong**. `profile()` defaults to seed 0, so unseeded runs are bit-identical; the variance came from driving `NumericAccumulator` directly with no seed, which is a configuration the public API never uses — the same error as measuring a kernel through a call site nothing calls. Corrected before it reached a decision. The argument that holds is accuracy, not stability: the estimate is perfectly reproducible **and** still an estimate, and changing the seed moves the median from `0.003684` to `0.01293`.

## [0.0.63] - 2026-08-16

Documentation only. Roadmap re-audited at 0.0.62 (v10), superseding v8.

### Changed
- **`docs/roadmap.md`** — v10. The redesign is finished as a body of work; the performance headline is re-measured (**3.01×**, which v7 and v8 both flagged as overdue); and the surface around the report is now the weakest part of the project.

### Fixed
Two findings that did not survive being checked, both corrected before they reached a decision:

- **The published example report was never stale.** `docs.yml` regenerates it before `mkdocs build` and deploys on every push to `main`, so the README's link serves current output (600,049 B, inline-SVG logo) — not the 1,178,450 B artifact committed at 0.0.17. The conclusion had been drawn from the committed file rather than the served URL, twice. What remains is repo hygiene, and the question is why a build artifact is committed at all (#143).
- **`finalize()` does not consume randomness.** It had been recorded as the one open correctness item, on the basis that a mid-stream `finalize()` diverges on eleven fields including the median. Running a control refuted it: two *unseeded* runs with no `finalize()` anywhere already differ on nine of them. With a seed the median is bit-identical either way, and `ReservoirSampler.values()` returns the buffer — it consumes no randomness, so the stated mechanism does not exist.

  The real bug underneath is **#146**: the quantiles are reservoir estimates printed to four significant figures with no approximation marker, while eight unseeded runs spread `1.86 × 10⁻²` on a true median of `2.8 × 10⁻³`. Invisible to the suite because every test seeds.

### Added
- **The height decision**, open across seven hand-offs and blocking #122. #112 moves from ≤560px to **≤627px** and #114 from ≤600px to **≤820px**, both pinned to a named dataset and viewport and set at the measured value as ratchets. Neither original number reproduced, because neither criterion had named a dataset or a viewport — the same failure as v7's report-size series.
- Seven issues for the audit's verified findings: #145 card heights, #146 quantiles, #147 histogram width, #148 hex ratchet, #149 triage block, #150 demo dataset, #151 README.

## [0.0.62] - 2026-08-16

### Fixed
- **`+ add a note` did nothing.** Not slowly, not partially — nothing, with a clean console. The redesign renamed the description block from `.description-value` to `.description-block`, and `description-editor.js` was never updated. Every entry point in that file guards on a null container and returns quietly, so the rename turned a working control into an inert one and left no trace. It now finds the block by its `id`, which is what the template guarantees; the class is presentation and has moved once already.
- **Saving a note would have left it invisible.** `.is-empty` sets `display: none` on `.description-content`, so fixing only the selector would have stored the note, escaped it, inserted it — and shown nothing. The class, the label (*Description* → *Note*) and the invitation (*+ add a note* → *edit*) now move together, on save and when a note is restored from `localStorage`.
- **The editor had nowhere to type.** The textarea is appended into a three-column grid; without an explicit span it landed in an 88px track. It now takes its own full-width row.
- **The invitation was under the minimum pointer target.** Now 24px (WCAG 2.5.8) in its own right, rather than relying on the whole row being clickable.

### Added
- **`tests/test_description_editor.py`** (23 tests). Most of it is not about the description block: it checks **every** `getElementById` and class selector in the bundled JS against real rendered markup, so the next rename is caught in whichever module it happens. Verified by reintroducing the bug — three tests fail.

  Two false positives were found writing it, and both were the same mistake — a selector check is only as good as the markup the fixture reaches. `[1.0, 2, 3, 4, 5] * 40` has five distinct values and profiles as *categorical*, so the frame had no numeric card and the Linear/Log control was reported dead when it is not; and a frame with no quality problems renders no `.needs-attention` block, so the flag filter was reported dead too. Both were checked in a browser and work. The fixtures now cover every card kind and both correlation routes.

### Removed
- Five `.description-value` rules in `_13-utilities.css` that had styled nothing since the rename, and a client-side placeholder string (`Click to add description...`) that sat inside an element the empty state hides.

### Known
- Four selectors in `functionality.js` and `tooltips.js` still point at markup the redesign removed (`missing-tabs`, `missing-tab-content`, `details-panel`, `compact-row`). These are dead code, not dead controls — nothing renders a button that reaches them — and are listed explicitly in the new test with a guard that fails if any starts resolving again.

## [0.0.61] - 2026-08-16

The redesign's testing harness (#123), turning fourteen commits of manual
checking into a guard.

### Added
- **`tests/test_report_data_invariance.py`**, with committed fixtures. Three tests for the sentence the whole migration rests on — *presentation changes on every commit, the facts change on exactly two*:
  - **The fingerprint.** 598 facts, with colours, class names, element order, tag names, whitespace and SVG geometry discarded. An HTML snapshot would be 100% churn on every one of these commits, so nobody would read it, so a real regression would ride in unnoticed.
  - **The golden `summarize()` payload**, on three frames. The cheapest test in the set and the one that matters most: milliseconds to run, and the only thing between *a CSS refactor* and *a CSS refactor that quietly changed the median*.
  - **Fact coverage** — 154 of 154 statistics appear on the page, floored at 90%. The real risk of restacking a card is not a wrong number, it is a **missing** one, and no snapshot diff shows that in a document where every line changed.

  Each guard was verified to fail: a changed value, a removed fact, a mutated payload, and a report showing nothing all trip it.

### Fixed
Two things that would have made the harness flaky, both found by running it rather than by reading it:

- **The fingerprint captured wall-clock duration.** `elapsed` moved from 0.02s to 0.04s simply by running the suite under load. A guard that fails for reasons nobody can act on trains its reader to re-baseline on red, which is the one habit this file exists to prevent.
- **Memory figures depend on the state of the process, not the data.** A column of a few repeated short strings — `male`/`female`, `C85`/`B42` — measures differently depending on whether those exact string objects are already alive, because an object array stores pointers and the accounting walks unique objects. Two runs of the same frame in one suite disagreed by 160 bytes. Excluded from both guards, with the reason recorded and a test that keeps the finding.

The allow-list of statistics that never render verbatim is now self-verifying: two of its entries named fields that do not exist (`ts_min` for `min_ts`, and a `sample_scale` that was never in the payload), so they exempted nothing while reading as though they did.

## [0.0.60] - 2026-08-16

Phase 7 of the report redesign (#120): missing values, routed on chunk count
rather than on tabs.

### Changed
- **The tabs are gone.** `Data Completeness` and `Missing per Chunk` over three rows was two clicks for one screen of content — and with a single chunk the second tab held one full-width block per column, a tab that hid nothing. The view is chosen by chunk count instead, the same shape of conditional the correlations section uses.
- **One row per column**: name, a bar on the warm severity scale, count and percent in the matching severity colour, with the bands stated in a legend. Same row shape as the summary's missing list, so a reader learns it once, and it fits a phone unchanged.
- **Nothing missing says so in one line** rather than rendering an empty grid, and complete columns are **summarised**, not listed — sixty column names is not a summary.

### Found while doing this — #139
The by-chunk half of #120 **cannot be built**, and the tab it replaces has been empty since it was written. Three things, each verified:

- The section read `chunk_metadata` off the *accumulator*. No accumulator has that attribute; it lives on the **summary** `finalize()` returns. It was always `None`.
- Only the numeric accumulator tracks per-chunk missing counts at all — categorical, datetime and boolean keep none.
- Even the numeric one records a single boundary: `mark_chunk_boundary()` is called only from its own `finalize()`, never per chunk. On 900 rows at `chunk_size=150` — six chunks — it reports `boundaries=[900]`.

The route degrades to the single-column view automatically, which is what #120's own edge case asks for when the metadata is unavailable. The strip renderer is written, tested directly, and will draw as soon as there is something to draw. A test asserts the gap still exists, so whoever fixes #139 is told the view can be switched on.

### Note on the fingerprint
One fact was removed: `data-chunk="1"`. That is the chunk *index* from the single full-width block the old tab drew, not a statistic — the figure it accompanied, `683 (76.7%)`, is still on the page in the new row.

## [0.0.59] - 2026-08-16

Phase 6 of the report redesign (#119): correlations. Nothing removed or changed
in the fact fingerprint.

### Changed
- **"No significant correlations found" now reports what was found.** This is the common case — both example reports hit it — and nothing was actually missing: the pairs *were* computed, and every one came back weak. That is a finding, not an absence.

  > All **3** numeric pairs are weakly related. The strongest is **0.025**, under the **0.50** reporting threshold.

  The pairs below threshold are listed with their real values, capped at ten and stating how many were checked — with forty numeric columns there are 780 pairs, and the cap is not the count.
- **Sign is position, not colour.** One diverging bar per row: zero at the centre, negative running left, positive running right, with the scale in the header. Colouring a negative correlation red reads as *bad*, and a negative correlation is often the interesting one. This also survives greyscale and needs no legend.
- **The three strength bands are three steps of one blue** rather than three hues, and the rank badges are gone — the list is ordered, so `#1` beside the first row states what its position already says.
- **The matrix is a lower triangle.** The full square printed every pair twice and spent a diagonal saying `1.00` once per column: half the ink for none of the information. Its ceiling drops from 15 columns to **10**, where the cells stop being wide enough to label, and two numeric columns now take the list — one pair reads better as a sentence than as a single cell.
- **Weak cells stay visible and go quiet** rather than being left blank. A blank cell is indistinguishable from a pair that could not be computed, and an all-weak row is information.
- **The count says what was checked**: `7 pairs above 0.50, of 190 checked`.

### Removed
- The `📊`, `📈` and `📉` emoji, from the correlations section and from the numeric card — not part of this brand, and they render inconsistently across platforms. The card's direction indicators become arrows, which carry the same meaning in one glyph that renders everywhere.

### Verified
#119 notes that no correlation above 0.5 had ever been seen in a real report here, so both populated views were designed against illustrative numbers. Both were checked against datasets with genuine structure before merging: a five-column frame for the matrix and a twelve-column one with six positive and six negative relationships for the list.

## [0.0.58] - 2026-08-16

Phase 5.7 of the report redesign (#118): three defects found while designing,
and easy to lose in a long plan. Nothing removed or changed in the fact
fingerprint.

### Changed
- **The quality chips show the number they already had.** Every chip carried `data-threshold` and `data-value` in the DOM and displayed neither, so a card said `Missing` where it could have said `19.9% missing` — and a reader had to open the details pane, or the inspector, to learn whether that meant two rows or two hundred. The value now leads (it is the fact; the label says what the fact is about) and the threshold moves into a `title`, because it answers a different question.

  Done as one transform over the markup contract rather than at each of the **forty-two** places a chip is emitted. Those attributes are a contract every one of them already satisfied — which is exactly why the information was there and unused.
- **The chips are outlined, not filled.** Each severity carried a tinted background in a colour left over from the old palette — `rgba(241, 94, 78)` and friends, on neither scale — so a row of chips read as a row of coloured blocks with the severity competing against the text inside it. The border states the severity; the text states the fact. The warning chip takes `--q-warn-text`, not `--q-warn-fill`: the fill step sits deliberately below the text minimum so a bar can be lighter than a word, and a chip label is a word.

### Removed
- **The `.stat-badges` CSS.** The renderer that emitted it had already gone; what remained was a block of rules plus one more rule to hide the markup they styled. Both are still bytes in a single-file report (#39).

### Already fixed
The distinct count no longer exceeds the row count — that was closed in 0.0.47, before any of the redesign was screenshotted, so the baselines this migration measures against were taken from correct output. It is asserted again here because the acceptance list belongs to this issue, and because the clamp has to hold for every kind that publishes `unique_est`. A boolean column does not publish one, and should not: its distinct count is 2 by definition, so an estimate of it would be an approximation of something exactly known.

### Fixed along the way
`tests/test_api_honesty.py` extracted chips with `<li class="flag[^"]*"[^>]*>`, which ends the tag early on `data-threshold="|kurtosis| > 3"` and silently returned a fragment of the attribute as a chip label. It now uses the render layer's own parser, which has handled that case since #86.

## [0.0.57] - 2026-08-16

Phases 5.5 and 5.6 of the report redesign (#117). Nothing removed or changed in
the fact fingerprint.

### Fixed
- **The boolean bar's labels were illegible on the pale segment.** Every one was `fill="white"`, which is fine on `--data-2` and about **1.8:1** on `--data-4` — against the 4.5:1 a label needs. `--on-data-*` exists to state which ink goes with which step, and it has to be used, not merely defined.
- **The bar was a 52px band** for a column with two values. It is 38px now, in the 36–40 the design asks for. The height was a default argument in the renderer, so the 48 in the chart config had never taken effect.

### Added
- **The temporal charts name their unit**: `RECORDS`, once, above the y axis. Nothing said these counted records per bucket — a reader could as easily have taken the axis for a share, or for the column's own values.

### Verified rather than assumed
Most of what this issue asks for was already true by the time it was reached, and checking that was the work:

- The boolean segments are already two steps of one hue, with no red-and-green. `Survived` reads as two values of one column, not as good versus bad.
- `temporal_charts.py` already allocates a **fixed** slot per bucket — 24 hours, 7 days, 12 months — rather than only the populated ones. Confirmed on a column spanning six months: the month chart still draws twelve slots and leaves six empty. Two populated months as two half-width slabs would read as *spread evenly across the timeline* instead of *2 of 12*.
- The tick labels survive on all four small multiples, and `temporal_charts.py` carries no legacy hex.

## [0.0.56] - 2026-08-16

Phase 5.3 of the report redesign (#116): a chart with nothing to say now says
so. Nothing removed or changed in the fact fingerprint.

### Added
- **High-cardinality columns get a sentence instead of a chart.** `Name`, `Ticket` and `Cabin` rendered ten bars of one row each — a chart drawn at full size, carrying no information, indistinguishable at a glance from one that carries plenty. **No chart element is emitted at all**, and no empty box: a container the height of a chart with nothing in it reads as a failed render rather than as *there is nothing to draw*.

  The sentence tells the truth about which case it is. A column where every value is distinct says so. `Cabin` — 147 values in 204 rows — cannot, so it reports the numbers instead: `147 distinct values in 204 rows, and the five most common cover 8.8% of them.` An `identifier-like` flag appears only where uniqueness actually holds.
- **Ordinary columns state their coverage**: `2 of 2 levels shown · covers 100% of non-missing rows`. A top-N chart is a sample of the levels, and without this there was nothing to say whether the bars were the whole column or a tenth of it.

### On the rule
Two arms, deliberately overlapping, because both inputs are approximate.

Coverage comes from Misra-Gries counts, which are **lower bounds** — so the test can only under-state coverage, and errs towards replacing the chart. That is the safe direction: a sentence about a chartable column is a lesser failure than a chart of slivers.

The cardinality arm sits at 0.5, the same ceiling the summary already uses for *high-cardinality categorical*, so the card and the summary agree about which columns those are. Far enough from the KMV error that a 2.2% wobble cannot move a column across it — a card that changes shape between runs of the same data would be worse than either shape.

`top_items` can be **empty** rather than full of singletons, because Misra-Gries is gated off entirely on high-cardinality columns (#62). The absence is the signal — but not on its own, since an all-missing column has no top values either.

## [0.0.55] - 2026-08-16

Phase 5.2 of the report redesign (#115): the histogram says what its axes mean.
Nothing removed or changed in the fact fingerprint.

### Changed
- **The column name is no longer printed inside the chart.** The card header carries it, so the plot was spending a line on a word the reader had just read — while still leaving nothing to say which axis was years and which was rows.
- **The axes are labelled**: `ROWS` above the y column, and the x unit at the right end of its axis. Both mono 10px in `--axis`.
- **The x unit is derived only when the column name states one.** `age` and `age_years` are years, `elapsed_ms` is milliseconds, `size_bytes` is bytes. **A column called `score` gets no label at all** — inventing `SCORE` would add a word and no information while looking like a unit, which is worse than a bare axis. Nine name families are recognised; everything else is unitless.
- **The first and last x labels sit inside the plot.** Centred on their tick, the first ran under the y-axis labels and the last ran off the SVG; they anchor to the plot edge instead.
- **Bars carry no stroke, no `fill-opacity` and no rounded corners.** Each of those changes a bar's apparent length, which is the one thing it encodes.
- Axis figures are monospace, like every other figure in the report.

### Fixed
- **Very large magnitudes no longer print in full.** A tick read `-2,000,000,000,000,000` — 22 characters, wide enough to collide with its neighbours and too long to take in. Values past a million are compacted to `2M` / `2B` / `2T`, which is shorter than both that and `-2.0e+15`.

### Measured
The chart is **959px wide** on a 1,013px card, against 420px before — a gain of **539px**, which is what makes 50 bins legible and the log toggle worth using.

## [0.0.54] - 2026-08-16

Phase 5.1 of the report redesign (#114): the numeric card restacked. No number
on the page changes — nothing removed, nothing changed in the fact fingerprint.

### Changed
- **The histogram takes the card's full width**, above the stats rather than squeezed beside them. It was one third of a `240px 240px 1fr` row; full width it gains about 550px, which is what makes 50 bins legible and the log toggle worth using.
- **The two key/value tables became one stat row.** `minmax(0, 1fr)`, not `1fr`: a grid track's default minimum is its content, so one long value — `-1.2345678e+18` is the case that does it — widens its own column and pushes the others out of alignment.

### Fixed
- **The page no longer scrolls sideways at 390px.** `.card-controls` was a grid sized `var(--triple-left) var(--triple-right) 1fr` "to match `.triple-row`" — a layout the card no longer uses — so its centre track measured 361px inside 358px. A page that scrolls sideways makes every horizontal gesture ambiguous, including the one inside the sample table's own scroll pane.
- **Every control is a 44×44 target.** The scale and bins toggles were inline links, so the target was the line box — about 20px, under even the 24px minimum of WCAG 2.5.8, on the controls a reader touches most. The Details toggle came along at 27px.
- **The mobile stat grid actually applies.** Its rule sat *before* the desktop rule it overrides, and with equal specificity the later one wins — so the four-column desktop grid was in force at 390px. It measured shorter that way, which would have read as the height target being met.

### Note
The generic `.triple-row` grid is deliberately retained. The categorical, boolean and datetime cards still emit it — they are phases 5.3 and 5.4 — and removing it with the numeric restack flattened all three at once. There is now a test per card type.

### Not met
The acceptance target was a card ≤600px at 390px; the numeric card is **775px**. The parts are a 180px chart, 148px of controls and a 306px stat row, under a 73px header. Two acceptance criteria are in direct tension here: six controls at the required 44×44 wrap to 148px at 390px, and fourteen statistics in the specified two-column mobile grid are seven rows. Meeting the height means dropping controls, dropping statistics, or abandoning the 44px target — each a design decision rather than a tightening.

### On the fingerprint
This is the first commit of the migration where the fingerprint moved, and it was **the extractor at fault, not the report**: it read label/value pairs only out of table cells, so statistics that moved into the stat row read as *removed* from a report that still displayed them. `scripts/report_fingerprint.py` now matches both shapes under one key — which is the loosening its own docstring prescribes for exactly this — and picks up six summary facts it had never covered.

## [0.0.53] - 2026-08-16

Phase 4 of the report redesign (#113), and it closes #103. The fact fingerprint
is byte-identical, 598 facts, across six consecutive commits now.

### Changed
- **A missing value renders as an em dash, not the string `nan`.** The literal is what pandas prints, and it reads as a value: a column of `nan` looks like text data rather than absence. The real value is kept in a `title`, and the glyph takes a text-grade colour because it *is* data.

  The distinction that matters here is between a null and a string that spells one. **A column named `nan`, the three characters `n`, `a`, `n`, and a zero are all real data** and render as themselves. Dashing them would not be tidying up — it would be corrupting a value and calling it missing.
- **The cell borders are gone.** The table drew one on every cell — about 300 for a 10 × 13 sample — and striped alternate rows on top, so the grid competed with the data it held. What is left is a rule above the header, one under it, and a hairline under each row, which is the job the striping was doing.
- **The row index is frozen** when the table scrolls sideways, so a wide frame never loses your place in the row.
- **The table says what it is showing**: `12 cols · scroll →` when there is something off-screen, and `10 rows drawn at random from the first chunk` — because they are not the head of the file, and a reader who assumes they are will misread every value in the table.
- **Long values clamp** at 260px with the full string in a `title`. One 500-character cell used to widen the pane until nothing else fit.
- **The sample opens on load** (#103). It is the fastest way to see whether the profile matches the data, and behind a click most readers never opened it. Still collapsible.

### Removed
- The pandas `to_html` table builder, which was dead: both backends already funnelled through the same simple builder, and the two used to emit different markup for the same table.
- The `<span class="num">` wrapper on every numeric cell. Right-alignment is a class on the cell now — the span was an element per numeric value to do what one attribute does.

### Note on the frozen index
The design specifies two tables side by side, and names the risk itself: rows of differing height drift out of step. This uses `position: sticky` on the index cell instead. A sticky cell is part of the row it belongs to, so the alignment cannot come apart — the failure mode is removed rather than tested for.

## [0.0.52] - 2026-08-16

Phase 3 of the report redesign (#112), and it closes #104. **The summary is
599px tall on a phone, from 1,330** — measured at 390px, before and after, on
the same frame. No number on the page changes; the fact fingerprint is
byte-identical, 598 facts, now across five consecutive commits of the migration.

### Changed
- **The donut is now a 100% stacked bar.** A donut cannot be read to exact proportion — comparing two arcs is a harder perceptual task than comparing two lengths against a shared baseline — and it stops working below about 200px wide, which is every phone. The bar reads at any width, reflows for free, and prints each count inside its own segment, so nothing has to be estimated.

  **The segment widths sum to exactly 100.** Rounding each share independently leaves a gap at the right edge — a third, three times, rounds to 99.9 — which reads as a rendering bug. The largest-remainder method fixes the total first and hands the leftover tenths to whichever shares lost most to the floor.

  **A type with no columns gets no segment**, because a zero-width segment is an artifact rather than information, and the palest step of the data scale sits close enough to `--track` that a hairline of it reads as a seam. Those types keep a muted legend entry with their zero, which is the thing a reader actually wants to know.
- **Five bordered cards became one stat row**: a rule above, hairlines between, six cells on mobile. The `min-height: 280px` on the second-row cards is gone — it held three boxes open at 280px each on a phone for content needing a fraction of it.
- **Five "quick insight" pills became one mono run.** Five borders to state five short facts, and the borders were doing none of the work.
- **The description is a margin note.** Empty, it costs exactly one 45px hairline row: a report generated in a loop never has a description and must not be disfigured by an invitation nobody will accept. Filled, it takes a `--q-good` left rule and a `NOTE` label. `description-editor.js` and the `data-report-id` / `data-original-markdown` contract are untouched.
- **Top-missing rows are one line each** — name, bar, figure — instead of a stacked pair.

### Removed
- `render/donut_chart.py`, `static/css/_04-donut.css`, and the donut's hover tooltip. The tooltip existed because an arc cannot be read to a value; the bar prints every count on screen already, so it was a hover target that repeated what was visible.
- The column-type key in `_03-summary.css`. It now comes from the renderer, so the swatch and the segment it labels take the same value from the same place and cannot disagree.

### Not met
The acceptance target was ≤560px and this lands at **599px**. The mobile stat row is 245px of that on its own — six cells in three rows at the specified 23px value — so closing the last 39px means either dropping a cell or shrinking the figure, both of which are departures from the design rather than tightening. Flagged rather than fudged.

## [0.0.51] - 2026-08-16

Phase 2 of the report redesign (#111): one header bar, and metadata that says
what it is. **No number on the page changes** — the fact fingerprint is
byte-identical, 598 facts, and has now stayed so across four consecutive
commits of the migration.

### Changed
- **The header is one 52px bar**, down from about 96px. It was a two-row grid with a 78px logo column holding a vertical lockup — and a vertical lockup inside a horizontal bar is exactly what forces a tall header. The mark and the name now sit side by side at 30px. Mobile is 48px.
- **The sections are plain text.** The active one is marked by colour and a 2px rule beneath it rather than a filled pill: there is nothing here to press.
- **The metadata left the bar and got labels.** It used to be a bare timestamp, a bare duration, a bare version, and a bare `891 × 12` whose meaning lived in a `title` attribute — and a tooltip survives neither printing, nor a screenshot, nor PDF export, which are three of the four ways anyone reads a report they did not just generate. It now reads `Generated … │ Profiled in … │ Shape 891 rows × 8 columns │ pysuricata 0.0.51`, under the report title, and the shape is spelled out so the tooltip is not needed.
- **The bar names what was profiled**, when there is a name to give. `profile("passengers.csv")` shows the filename; an in-memory frame shows nothing, and the separator is emitted with the name or not at all, so the common case leaves no rule dangling. Available as `dataset_name` for callers who want to set it.
- **The pin joined the icon group.** It used to sit alone at `margin-left: auto` in a metadata row that no longer exists, and `functionality.js` injects one into the nav when it finds none — which would have dropped an icon into the text sections and into the mobile rail.

### Removed
- The indigo, amber, sky and emerald header icon colours. None of them was on either scale; they were simply picked. Header icons take `--ink-2` and draw with `currentColor`.
- The compact-mode block in `_06-cards.css` that shrank the old header by trimming its padding and font sizes. Every selector in it named an element the template no longer emits.

### Accessibility
- **Every action is a 44×44 target** while still painting at 30px, because a 52px bar has no room for a 44px box. The hit area is extended past the paint — the two are not the same rectangle, and only one is visible.
- **Section links clear the 24×24 minimum** in WCAG 2.5.8. A 13.5px line box is about 21px tall, so the box was grown rather than the type.
- **The mobile rail fits all five sections at 390px** without swiping, measured: 325px of content in 334px of rail. It uses `min-height`, never `height` — with `overflow-x` a fixed height has the scrollbar subtracted from it, which silently leaves a 29px rail that fails the target while measuring as though it passed.

### Verified
The self-download still round-trips: bar, mark, five section links, three actions, the metadata line and all eight cards survive re-serialisation. That button finds its content by regex and fails silently, so it is checked whenever the header markup moves.

## [0.0.50] - 2026-08-16

Phase 1 of the report redesign (#110): the token layer, the typography, and the
structural motif every later phase builds on. **No number on the page changes** —
the fact fingerprint is byte-identical across it, 598 facts.

### Changed
- **Type is no longer a colour.** The old palette gave each column type a hue and the per-column cards inherited it. Inside a card the badge already names the type, so the hue carried nothing — and it collided: olive meant both *categorical* and *passes*, rust meant both *boolean* and *fails*, so a rust bar and a rust warning chip could sit in the same card meaning unrelated things.

  There are now two scales that never mix: **`--data-*` blue for every chart, bar and segment**, and **`--q-*` warm for data quality alone**. Missing-value bars are the one deliberate exception, because there the encoding *is* severity — 77% missing should look worse than 0.2%.
- **`Survived` is no longer red-and-green.** Colouring `false` rust and `true` olive read as bad-versus-good — the report passing judgement on someone's data. Two values of one column now get two steps of one hue.
- **Typography.** `font-family: Arial, sans-serif` is gone. Prose and UI take `--font-sans`; **every figure, column name, dtype and axis label takes `--font-mono`**, so columns of digits align without `font-variant-numeric` hacks. The monospace stack had been spelled out as a literal in 30 places across 8 files, and one of them had already drifted from the others.
- **The structural motif**: hairline rules and whitespace instead of bordered boxes inside a bordered page. Radius is gone from data containers and clamped to 6px on chips and buttons.

### Removed
- The decorative shadows and gradients — `--chart-shadow-*`, `--segment-shadow-*`, `--label-shadow-*`, `--legend-shadow-*` and the `--chart-bg-*` / `--svg-bg-*` gradient pairs. They were **45 of the 80 lines** of the old token file and none of them encoded anything.

### Fixed
Three things found by inspecting a rendered report rather than the stylesheet, each of which a colour-literal grep passes straight over:

- **`_06-cards.css` was redefining `--axis` and `--axis-text`.** The stylesheets are concatenated in filename order, so it loaded after the token file and silently won: the report drew its axes in `rgba(0, 0, 0, 0.45)` while the token file — and the contrast test reading it — said `#8F8474`. A contrast guard that reads only token definitions is worth exactly as much as the guarantee that those definitions are the ones in force, so a test now fails if any other stylesheet reassigns a token.
- **The column-type key was on the data-quality scale.** The four swatches sat on `--q-good`, `--q-warn-fill` and `--q-bad`, because the palette swap had replaced each legacy hue with whichever new token looked closest. The old hexes were gone, so every literal check passed — while the report still said olive for both *categorical* and *passes*. That is the exact collision this palette exists to remove, reintroduced by the change meant to remove it.
- **Chart axis labels were unreachable by the token.** `font-family: inherit` on the SVG text pulled in the body sans and beat the presentation attribute on the element.

### Documentation
- **`docs/roadmap.md` is re-audited at v7.** v6 was written at 0.0.38, twelve releases ago, and predates the redesign entirely. Two things changed shape: the **correctness backlog is empty** — every item v6 listed as open is closed — and the report's presentation, which v6 tracked in one line as out of scope, is now the largest open item. A fourth measurement rule is added, from the token bug found in this release: *a guard is worth only as much as the guarantee that what it reads is what runs*.

### Note
`--rule-strong` was specified as "container edge, axis line", and the contrast test failed on it immediately: those two want different things. A container edge is decorative structure with no minimum — a hairline at 3:1 is a heavy black line, which is the boxed look this palette removes. A chart axis is part of a graphic required to understand the content, so WCAG 1.4.11 applies. They are now two tokens: `--rule-strong` stays a hairline, `--axis` clears 3:1 in both themes.

Dark-mode values ship as proposed and are covered by the contrast test, but the full dark pass belongs to the closing phase. The compatibility shim mapping legacy variable names onto the new scale is scaffolding and comes out then too.

## [0.0.49] - 2026-08-16

**The report is now half the size it was.** 1,110,756 bytes → 543,577 bytes on an
891 × 8 frame, measured before and after in the same run.

### Changed
- **The logo is inline SVG instead of two base64 PNGs.** They were **592 KB of a
  1.11 MB report** — more bytes than the data, the CSS, the JavaScript and the
  markup put together — to draw a mark 30 CSS pixels tall.

  There were two because the artwork had the wordmark baked into it, and a drawn
  wordmark has to be recoloured for dark mode, so the report shipped both copies
  and hid one with CSS. Setting the name as **type** beside the mark removes the
  duplicate outright: text follows `currentColor`, so there is nothing to swap
  and nothing to keep in sync. It also reads better, because the drawn wordmark
  is a display face whose letters land about eight pixels tall at header size.

  The traced mark is **10,814 bytes**, 16× smaller than the PNG it came from. No
  point on it moves further than one source pixel — 1/20 of a rendered pixel at
  header size — from the artwork's true boundary.

### Added
- **`scripts/trace_logo.py`**, which produced that asset and can reproduce it.
  It is committed so the mark can be regenerated if the artwork changes, and so
  the tolerance is a recorded decision with its measurements attached rather
  than a number someone once picked. A test asserts the committed SVG is exactly
  what the tracer produces from the artwork, so the two cannot drift apart.
- **A size guard on the report's embedded payload.** This is the failure mode
  that had gone unnoticed for a year: nothing breaks when an inlined image gets
  big, so the report simply got heavier every release until somebody measured
  it. The test now fails if the embedded payload exceeds 64 KB.

### Removed
- `logo_suricata_transparent_dark_mode.png`, which existed only for the swap
  described above and is no longer referenced anywhere.

### Verified
The report's **fact fingerprint is byte-identical** across this change — 598
facts, checked with `scripts/report_fingerprint.py`. The self-download button
still round-trips: the inline mark survives re-serialisation with its SVG
namespace and all three paths intact, which was worth checking because that
button finds its content by regex and fails silently.

## [0.0.48] - 2026-08-16

Repository housekeeping ahead of the report redesign. No library code changed.

### Changed
- **The changelog moved to `CHANGELOG.md` at the repository root**, which is where GitHub, PyPI and release tooling look for it. `docs/changelog.md` now includes that file verbatim through `pymdownx.snippets`, so the two cannot drift apart, and the CI freshness check follows the file. The root copy had been left behind at 0.0.42 while the docs copy carried five more releases; they are merged, and there is now one place to edit.

### Added
- **`docs/integration.md` and `docs/MIGRATION_TESTING.md`** — the redesign plan and its testing strategy, referenced by every issue in the #110–#122 series.
- **`scripts/report_fingerprint.py`** — reduces a rendered report to the set of *facts* it asserts, discarding colours, class names, element order and geometry. The redesign rewrites the render layer across seventeen commits, which makes an HTML snapshot test worthless: the diff is 100% churn on every commit, so nobody reads it, so a real regression rides in unnoticed. This holds the line that presentation may change on every commit while the numbers may not.
- **`codecov.yml`**, so a docs-only or config-only pull request stops drawing a coverage comment that has nothing to report.

### Fixed
- **`check_docs` read filenames out of YAML comments.** `check_nav` regex-scans `mkdocs.yml` as raw text, so a comment mentioning `CHANGELOG.md` was taken for a nav entry and reported as a page missing from disk. Comments are stripped first now.
- **`docs/integration.md` and `docs/MIGRATION_TESTING.md` are marked as planning documents**, alongside `DOCS_PLAN.md`. They live in `docs/` for convenience but are not published, and their code fences illustrate test scaffolding rather than the public API — so the checker was executing `pytest` and `polars` snippets and reporting the resulting `NameError`s as documentation defects.

## [0.0.47] - 2026-08-16

The first of three things to fix before the redesign takes a screenshot of
anything. Commit 0 of the report migration: correctness before presentation, so
every baseline the migration measures against is taken from correct output.

### Fixed
- **The distinct count could exceed the row count, and claimed to be exact.** 20,000 standard normals reported **20,197 distinct**, and a 20,000-row primary key reported 19,478 — both with `approx: False`.

  The sketch is not at fault: KMV at k=2048 has a relative standard error near `1/√(k−2)`, about 2.2%, and both figures sit inside it. The reporting was at fault. More distinct values than rows is arithmetically impossible, and a reader who notices it does not conclude *sketch tolerance* — they conclude the numbers cannot be trusted, and that judgement lands on every other statistic on the page.

  `unique_est` is now clamped to the row count on numeric, categorical and datetime columns, and `approx` is true whenever the value came from the sketch rather than from KMV's exact counter — it previously meant *sampling was involved*, so a column small enough to hold every value in the reservoir reported `False` while publishing a sketched distinct count. The card already prints `Unique (≈)` when `approx` is set, so the page stops claiming a precision it does not have.
- **A perfect key was not recognised as one, for the same reason.** The identifier check required the distinct estimate to reach 0.98 of the row count — *inside* the estimator's own 2.2% error — so `np.arange(20_000)` came back at 0.974 and was profiled as a measurement, with a mean. The tolerance is now 0.95, two standard errors out, which is where a threshold has to sit relative to the error of the thing it tests.

### Not changed
The distinct figure is marked `(≈)` but does not yet print its bound. Showing `20,000 ± 440` is more honest still, and needs the sketch to expose its own error — that goes with the quality-flag threshold work in #118.

## [0.0.46] - 2026-08-16

#65. Dataset comparison as a first-class output.

### Added
- **`compare(a, b)`**, and [a page about it](comparing.md). It reports what moved between two datasets — every delta, whether or not it crosses any threshold. Both sides accept anything `profile()` does, or a `summarize()` payload you already have, which is not re-profiled.

  ```python
  from pysuricata import compare

  diff = compare(january, february)
  diff.schema.added                          # {"extra": "numeric"}
  diff.columns["amount"].median_shift_sigma  # 0.24
  diff.columns["region"].categories_added    # ("west",)
  ```

  `Comparison.to_dict()` is the JSON contract; a text rendering is available for a terminal.

### How it relates to `check`
It is **built on** the gate rather than beside it: `check` is thresholds applied to the deltas `compare` computes, and both read the payload through the same functions. A gate and a diff that disagreed about what counts as a change would be worse than either alone, so there are tests asserting the two report the same numbers and share the same sketch-noise floor.

Two things are in the diff and deliberately not in the gate. **Category churn** — which values entered and left the top-k — because top-k membership is not a census and its tail reshuffles on counting noise, making it a poor thing to fail a build on and the most legible thing to show a reader. And **every quartile**, not just the median: a gate wants one number, a diff wants the shape.

Cardinality is reported as two numbers, the distinct **count** change and the distinct **rate** change, because neither answers alone — doubling the rows doubles a continuous column's count while leaving its rate alone, and does the opposite to a three-level enum. `check` resolves that with a rule; `compare` hands the reader both and lets them see it.

### Honesty
Sketch-derived deltas are marked `approximate`, and the text rendering **suppresses a distinct-count change smaller than the sketch's own error** (~2.2% at the default `uniques_k`) — printing a 1% move as a finding is the same mistake as printing an estimate as an exact integer. The structured delta still carries the number.

`compare` profiles both sides with `seed=0` by default, so comparing a dataset against itself is a no-op rather than a set of sampling wobbles.

### Not included
No HTML view. The JSON contract and the text rendering are what this ships with; the report side belongs with the wider work on the report's presentation rather than bolted on beside it.

## [0.0.45] - 2026-08-16

#64 and #105. No behaviour change: the accumulator boundary becomes something a
second implementation could satisfy, and the options grow a shape.

### Added
- **`pysuricata.accumulators.protocols`** now describes the surface the engine may use: a `StreamingAccumulator` protocol, an `AccumulatorKind` tag, and an explicit pickle protocol. This is the prerequisite work for the native core (#44), and it is not tidiness — a PyO3 accumulator cannot satisfy `isinstance(acc, NumericAccumulator)`, cannot expose a `_uniques` attribute holding a Python `KMV`, and cannot be pickled by copying `__dict__`. Each of those was how something outside the accumulator package reached inside it.
  - **Dispatch reads `acc.kind`** instead of testing types. The old `isinstance` chain's *order* was load-bearing without saying so anywhere.
  - **`tracks_top_values`** replaces the report layer reading `acc._track_top_k`, and the render layer reads `unique_est` — a property on every kind — instead of `acc._uniques.estimate()`.
  - **`__reduce__`, `__getstate__` and `__setstate__`** on all four accumulators, round-tripped in tests. Checkpointing pickles the accumulator dict, so a native type without an explicit reduce breaks checkpointing at the *end* of a long run.

  The tests are written against a fake accumulator that implements the protocol and **inherits nothing**, which is as close as a Python test gets to proving the boundary would hold for a type from another language.
- **`ComputeOptions.checkpoint`** groups the five checkpointing settings under shorter names — `options.checkpoint.every_n_chunks` — as a lens on the same fields rather than a copy, so nothing that sets them directly breaks and the two cannot disagree.

### Changed
- **An unknown keyword now points at the one that works.** Someone who read `ComputeOptions` and typed the field name they found there was told it was unknown, which is true and useless: `numeric_sample_size=5000` now answers *"the keyword for it is `sample=`"*, and a field with no keyword form says to set it through `config=`.

## [0.0.44] - 2026-08-16

#98, #100, #101 and #102. None of these is a wrong number; all of them cost
trust in the numbers.

### Fixed
- **A healthy column was flagged as a data-quality problem** (#98). `age` — 68 distinct integers between 18 and 85 — was unflagged at 1,000 rows and **Quasi-constant** at 20,000, because the flag compared `unique_est / count` against 2%. That is the same unique-*ratio* reasoning the type classifier dropped in #84, left behind in the flag layer; and since #86 put the quality chips in a triage block at the top of the report, the false alarm had become the first thing a reader sees.

  Quasi-constant is now a claim about **concentration** — the most common value covers 95% or more of the rows — which is what the words mean and does not move with the row count. Misra-Gries counts are lower bounds, so a share computed from them can understate dominance but never invent it, which is the right direction for a warning. **Discrete** now uses the classifier's own cardinality ceiling, imported rather than repeated, so the flag and the classification cannot disagree.
- **Bad input escaped the exception hierarchy** (#100). `profile([1, 2, 3])` raised `RuntimeError: Adapter selection failed: Unsupported input type: <class 'int'>` — outside `PySuricataError`, in vocabulary about our module layout, naming the type of the *first element* rather than of the argument. It now raises `UnsupportedDataError: Cannot profile list of int`, and a generator yielding the wrong thing — which cannot be inspected without consuming it — arrives as the same kind of error from the engine. `config="oops"` raised `AttributeError: 'str' object has no attribute 'compute'`; it now raises `ConfigurationError` naming the keyword form.
- **Validation disagreed with itself** (#101). `ComputeOptions(chunk_size=0)` was rejected while building one and setting `chunk_size = 0` afterwards was accepted — and since the options are mutable, the permissive path is the one people take. `ComputeOptions.validate()` is now one rule called from both places, checked again when the options reach the engine.
- **A successful call announced that it had failed** (#102). `profile(pd.DataFrame())` logged `Stream processing failed: Empty source` and then returned a usable report. An empty input is a valid, boring case: it is now silent at default verbosity. In CI — where `pysuricata check` puts this library on purpose — a line containing "failed" on a green run is exactly what gets grepped for.

`MAX_CATEGORICAL_LEVELS` is now a public name in `compute.processing.inference`, since the render layer imports it.

## [0.0.43] - 2026-08-15

### Removed
- **`scripts/create_ux_issues.py` and `docs/UX_ISSUES.md`.** The twenty-two UX findings they carried are now GitHub issues, which is where a backlog belongs — an issue can be closed, commented on, and linked from a commit; a Python file holding issue bodies can only go stale. Everything in them was re-checked against 0.0.42 before filing: two were folded into existing issues (#39, #41) rather than duplicated, and one had to be corrected, because it claimed the sample-size knob was unreachable when it is reachable under a different name.

No library change. `benchmarks/check_docs.py` drops `UX_ISSUES.md` from its
list of planning documents to skip, and repository lint is clean again — the
generator was the one file failing `ruff check`.

## [0.0.42] - 2026-08-15

#95 and #33. The bounded-memory claim now holds for text columns, and row
signatures stop being re-hashed through a lossy conversion.

### Fixed
- **A text column's memory grew with the row count** (#95), which contradicted the one claim the library is positioned on — and text-heavy frames are the ones people reach for this tool with. A single string column holding **four distinct values** reached 339 MB of peak RSS at 8.4M rows.

  The cause was `Series.str.len()`. Nothing was retained — `sys.getallocatedblocks()` stayed flat and the sketch state stayed at four counters — so it was allocator churn inside the accessor rather than a leak, but RSS is what a CI runner limits. Taking the length of each *distinct* value and gathering it back through the factorisation codes is flat in rows and returns exactly the same lengths in the same order, so no statistic moves.

  It is **4× faster on that kernel** for a low-cardinality column, because `len()` runs once per category instead of once per row — and that is worth **nothing end to end**: a text-heavy 200,000 × 3 profile measures 1.00× either way, because the length computation was never on the critical path. This is a memory fix, not a speed one.

  | rows | before | after |
  |---:|---:|---:|
  | 524,288 | 39 MB | 7 MB |
  | 2,097,152 | 78 MB | 7 MB |
  | 8,388,608 | **339 MB** | **7 MB** |

  A test now measures this in subprocesses on every run, because peak RSS cannot be checked honestly from inside the process doing the allocating.
- **Distinct row signatures could collide into one** (#33). `RowKMV` computes a vectorised uint64 row hash and handed it to `KMV.add_many`, which canonicalises integers through float64 so that `1` and `1.0` count as one value. That is right for a data column and wrong for a signature: float64 has 53 bits of mantissa, so a uint64 loses its low 11 bits. **1,000 hashes differing only in those bits were counted as 1 distinct value.** `KMV.offer_u64` skips the conversion, and the duplicate-row estimate no longer depends on which bits happen to differ.

### Not changed, and why
- **#32 (a bounded heap for the KMV sorted list) is closed on measurement.** The full sort it describes no longer happens: the admission-threshold pre-filter added in #62 rejects a 50,000-hash batch in 127 µs where sorting it costs 2,659 µs. Full KMV ingest is 29.7 ns/value, dominated by hashing rather than by sketch maintenance — so the target is the native core, where KMV is already ranked first.

## [0.0.41] - 2026-08-15

#66. Every source used to arrive as a pandas or polars frame, so profiling a
Parquet file meant reading the whole thing into memory first — which
contradicts the one claim the library is positioned on, for exactly the inputs
where the claim matters most.

### Added
- **[Arrow, Parquet and DuckDB read without being materialised](https://alvarodiez20.github.io/pysuricata/data-sources/).** `profile()` and `summarize()` now accept a `pyarrow` Table, RecordBatch, RecordBatchReader or Dataset, a DuckDB relation, and anything exporting the Arrow PyCapsule interface. A DuckDB relation is a query that has not run yet, so a filtered join across several Parquet files can be profiled without any of it ever existing as a frame.
- **`pysuricata.sources`** with `stream_parquet()`, `stream_arrow()` and `stream_duckdb()`, for when you want the batches rather than a profile. `stream_parquet(..., columns=[...])` never decodes the columns it does not read.

### Changed
- **A Parquet path is read a batch at a time** rather than loaded whole, by both `profile()` and the CLI. Measured on a 4,000,000 × 6 frame in a 180 MB file, peak RSS above the import floor drops from **581 MB to 307 MB**, and stops rising with file size the way loading does.

  A file that fits in one batch (under 65,536 rows) is still handed to the engine as a frame, so small files keep whole-frame type inference and classify exactly as before. Above that, a numeric column with few distinct values stays numeric rather than being reclassified as categorical — a stream cannot see distinct counts, and a leading run of one value would mislabel the column permanently. `profile(pd.read_parquet(path))` remains available when you want the old behaviour on a file that fits.

Neither `pyarrow` nor `duckdb` is a runtime dependency. DuckDB is duck-typed on `fetch_record_batch`, so nothing imports it; the Parquet and Arrow readers need `pyarrow` and say so plainly if it is missing.

### Measured, and open
Memory is **exactly flat in rows for numeric columns** — 19 MB above the import floor at 500,000 rows and at 8,400,000 — and **not flat for string columns**, which reach 339 MB at 8.4M rows on a column with four distinct values. That is not a leak (`sys.getallocatedblocks()` is flat and the sketch state is constant); it is allocator churn from materialising fresh Python strings per batch. Filed with the measurements as [#95](https://github.com/alvarodiez20/pysuricata/issues/95), and it blocks the memory budget in #79.

## [0.0.40] - 2026-08-15

#43 and #91. The payload becomes a contract that is checked rather than
described, and the gate learns the failure every other check passes.

### Added
- **[A documented `summarize()` schema](https://alvarodiez20.github.io/pysuricata/summary-schema/)** — every key, its type, which ones are estimates and with what error, and the stability policy. Adding a key does not change `schema_version`; renaming, removing, or changing the meaning or units of one does.
- **The payload now carries what the HTML shows.** It was a strictly poorer view and nothing said so — a gap only findable by reading the renderer, which is how it happened twice already (#24 correlations, #59 numeric top values). Numeric columns gained `skew`, `kurtosis`, `variance`, `cv`, `se`, `gmean`, `iqr`, `mad`, `ci_lo`/`ci_hi`, `jb_chi2`, `inf`, `outliers_mod_zscore`, `heap_pct`, `bimodal`, the granularity pair, the extreme values with their row indices, and the histogram. Datetime columns gained fifteen fields, having previously published six. Categorical gained the entropy and diversity measures, the length statistics, and the case/whitespace variant estimates behind the quality flags. Boolean gained its ratios and entropy. Every kind gained `dtype`.
- **A test that keeps it that way.** It walks the accumulators' own summary dataclasses and fails if a computed statistic is neither published nor listed in `SUMMARY_FIELDS_WITHHELD` **with a reason**. Adding a statistic now forces a decision about the contract.
- **Freshness gating** (#91) — `--require-fresh` fails when a datetime column's newest timestamp did not advance past the baseline's, and `--max-age 26h` fails when it is older than a duration, needing no baseline at all. This catches the most common failure of a scheduled pipeline: the job produced *yesterday's data again*, where every distribution matches and every other check passes because the data is literally the same. Both are off by default — a datetime column can be a birth date rather than an event time. Comparison is in UTC, so the gate does not depend on where CI runs.

### Changed
- **The payload is JSON-serialisable without a custom encoder.** Numpy scalars were leaking into `mean`, `missing` and `outliers_iqr_est`; a payload every consumer has to re-encode is not a contract.

`schema_version` stays **1**: this release only adds keys, which is exactly what the policy says is safe.

## [0.0.39] - 2026-08-15

Six issues (#36, #60, #61, #67, #89, and a bug found while fixing #61) with one
thing in common: a code path nothing ran. `merge()` exists for distributed use
and the pipeline never calls it; the adapters replace an accumulator only for
forced or reclassified columns; the config fallback fires only when validation
fails.

### Fixed
- **`merge()` lost most of what it was merging** (#67). It replayed one side's *reservoir buffer* through `add()`, treating 20,000 retained values as a 20,000-value stream. Merging a 90,000-row shard into a 60,000-row one reported a **median of 0.17 where the true value was 4.03**, and a distinct count of 83,514 against a true 154,923. The top-k counters were not merged at all, so a merged column reported only the left-hand side's common values.

  Every sketch involved composes, and now does so directly: KMV merges **exactly** (the k smallest hashes of the union are always a subset of the two sides' k-smallest sets), Misra-Gries merges by summing counters and subtracting the (k+1)-th largest, which preserves the frequency-error bound, and the reservoir merges by weight so each side appears in proportion to what it *saw* rather than what it retained. Monotonicity is deliberately not merged — it is a claim about arrival order, and two shards say nothing about how they would interleave.
- **The categorical merge was worse**, on the stated belief that "KMV sketches cannot be easily merged". It replayed one `add()` call per counted occurrence — merging a value seen ten million times ran ten million Python calls — and seeded the distinct estimate from at most 100 top-k keys. The case- and whitespace-folded sketches behind the variant flags were not merged at all.
- **`topk_k` never reached numeric columns** (found while fixing #61). `AccumulatorConfig.from_legacy_config` set `top_k_size` on the categorical config and omitted it from the numeric one, so the "Common values" table on a numeric card always kept 50 counters regardless of what the caller asked for.
- **Forced and reclassified columns fell back to library defaults** (#61). The twelve sites that replace an accumulator constructed it with no config, so `numeric_sample_k`, `uniques_k` and `topk_k` were silently discarded for exactly those columns — a user asking for `uniques_k=8192` got 2,048, with nothing in the report saying this column was measured to a different accuracy than its neighbours. They now go through one `build_accumulator()` in the factory.
- **A config value that failed validation was discarded rather than reported** (#89). `_to_engine_config` wrapped `from_options` in a bare `except Exception` with a fallback that mapped a subset of the fields by hand, so a bad value produced not an error but a **different configuration** — one that never set `columns`, the correlation options, `progress`, `engine` or any boolean-detection option. A caller asking for one column got the whole frame and a successful-looking run. The fallback is gone; the failure now reaches the caller as `ConfigurationError`.
- **`outlier_methods` did nothing** (#60). It was read by a detector that was never called, while `finalize()` always computed both IQR and MAD. It is now honoured.

### Changed
- **Missing cells come from the accumulators** (#36) rather than from an `isnull().sum().sum()` over every chunk — a second pass over every cell for a number the accumulators had just counted. The first chunk paid for it twice. Totals are unchanged, and asserted equal to a full pass at three chunk sizes.

## [0.0.38] - 2026-08-15

UX-5 (#76). The roadmap's differentiator: every existing gate — Great
Expectations, Soda, pointblank — asks you to author expectations first. A
profiler already knows the shape of yesterday's data, so it can gate with no
configuration at all.

### Added
- **`pysuricata check <data> --baseline baseline.json`** — compares a dataset against a stored baseline and **exits non-zero** when a threshold is crossed. `profile` and `summarize` both exit 0 no matter what they found, which is what made them unusable in a pipeline. Exit codes are 0 pass, 1 threshold crossed, 2 the check could not run, so a build can tell drift from an outage. `--write-baseline` creates the baseline, `--json` emits a machine-readable result on stdout while progress stays on stderr, and `--warn-only` reports without failing.
- **Thresholds in a file or on the command line.** `--thresholds` reads JSON or TOML, including a `[tool.pysuricata.check]` table in `pyproject.toml`; `--max-missing-pct` and `--min-rows` are absolute gates that need no baseline at all. A misspelled threshold is an error rather than a silent no-op — a typo that quietly loosens a gate is the worst failure mode a gate has.
- **`pysuricata.check`**, the comparison as an importable module: `compare()`, `Thresholds`, `Finding`, `CheckResult`, `make_baseline()`, `read_baseline()`, `write_baseline()`.
- **[Gating CI on drift](https://alvarodiez20.github.io/pysuricata/data-checks/)**, with a GitHub Actions job.

### Notes on the defaults
Three choices are what keep the gate from crying wolf, and all three are documented where they are made:

- **Growth is not drift.** Row-count drift is off by default. For the same reason the cardinality check requires both the distinct *count* and the distinct *rate* to move: doubling the rows doubles a continuous column's distinct count while leaving its rate alone, and leaves a three-level enum's count alone while halving its rate — so gating on either one alone fails every build that appends data. The cost, stated rather than left to be discovered: while the row count is also moving a lot, a small change in levels sits inside the band growth could explain and is not reported.
- **Distribution drift is measured in standard deviations, not percent.** A relative change in the mean is meaningless when the mean is near zero and incomparable across columns with different units.
- **Approximate quantities get loose thresholds.** `unique_est` is a KMV estimate with relative error near `1/√k` — about 2.2% at the default `uniques_k`. The default threshold sits an order of magnitude above that, any threshold set inside the noise floor is called out in the output, and findings resting on an estimate are labelled approximate.

`check` defaults to `--seed 0` rather than to no seed, so re-running it on unchanged data is a no-op rather than a coin flip. A baseline records the version and the payload's `schema_version`; reading one that does not match is an error telling you to regenerate it, not a comparison that silently succeeds against fields that moved.

## [0.0.37] - 2026-08-15

UX-7 (#78).

### Added
- **`progress=` on `profile()` and `summarize()`.** A 1.8-million-cell profile produced 46 bytes of output, none of it progress — for the use case this library is positioned on, a hung process and a working one looked identical. `log_every_n_chunks` existed but routes to a logger that is off by default, so it is invisible unless you configure logging first, which is not what you think to do while waiting to find out whether anything is happening.

  `True` reports; `"auto"` reports only when stderr is a terminal, so a redirect or a cron job stays quiet without being configured; a callable receives `chunks`, `rows` and `elapsed`. **Everything goes to stderr and nothing to stdout**, so a profile written to a pipe stays parseable. The line is throttled to stay readable and carries an ETA only when the row total is knowable — a generator source gets a counter and a rate, not an invented estimate.

### Fixed
- **A bad `progress` value could be silently discarded.** `_to_engine_config` falls back to a direct mapping inside a bare `except Exception`, so a value that fails validation deeper in becomes a *different configuration* rather than an error. `progress` is now validated at the public boundary, where the caller sees it.

## [0.0.36] - 2026-08-15

UX-4, UX-6 and UX-11 (#75, #77, #82). Three findings with one shape: the
library already had the answer and either made the caller work to reach it or
did not expose it at all.

### Added
- **Keyword options on `profile()` and `summarize()`.** Setting one integer took three imports and two nested constructors, because the nesting modelled the module layout rather than intent — nobody thinks *"I would like to configure the compute subsystem"*, they think *"smaller chunks"*. The six most-reached-for settings are now keywords: `chunk_size`, `columns`, `sample`, `correlations`, `seed`, `title`.
- **`preset="fast"` and `preset="thorough"`** — one word for an intent, rather than working out which of twenty-one knobs to turn. `config=` remains the full escape hatch, and combining it with a preset or a keyword is refused rather than silently ignored.
- **`schema_version` on the `summarize()` payload.** It had already drifted once — `dataset["rows"]` became `dataset["rows_est"]`, which silently broke every documented example and would have broken every downstream consumer. The promise: adding a key changes nothing, renaming or removing one bumps the major.
- **Numeric `top_values` reach the payload.** The HTML rendered a "Common values" table from the Misra-Gries counters while `summarize()` omitted them, so a tool built on the payload saw strictly less than a reader of the report. `None` means *not tracked* — the sketch is gated off on columns too high-cardinality for the answer to mean anything — which is a different statement from an empty list.

### Fixed
- **The histogram ignored the log-scale flag the card itself computed.** A lognormal column was correctly labelled *Positive-only · Skewed Right · Heavy-tailed · Log-scale?* and then drawn on a linear axis, where the whole distribution renders as one bar at the left edge. When the heuristic fires the chart now opens on a log axis; the toggle still switches both ways. Computing the right answer and displaying the wrong picture is worse than not detecting it, because it teaches the reader that the chips are cosmetic.

## [0.0.35] - 2026-08-15

UX-2 and UX-3 (#73, #74). Neither computes anything new: the signals were
already tracked and the chips were already rendered. Both findings were that
the report had the answer and did not use it.

### Added
- **Identifier columns are recognised and presented as keys.** A monotonic, fully distinct, integral column with no nulls now gets an **Identifier** badge and a card answering what a key raises — rows, distinct, duplicates, gaps in the sequence, order — instead of a mean, a standard deviation, a flat uniform histogram and `Zeros: 1 (0.0%)`, which is true and meaningless. `summarize()` reports `"type": "identifier"` and carries `mono_inc`, `mono_dec` and `int_like`, so the payload is not poorer than the HTML.
- **A "needs attention" block opens the Variables section**, naming the columns with real defects and linking to each card. Clicking one of its chips filters the list to those columns; clicking it again, or the All tab, restores source order in one click.

### Changed
- **Monotonicity detection is on by default.** It was off "for performance" when the detector looped over every value at 89 ns/value. As a sign test on `np.diff` it is 0.6 ns/value, and it is what lets the report recognise a key. The detector no longer re-filters an array the caller has already filtered — that redundant `isfinite` pass and copy, per numeric column per chunk, was the entire cost of turning it on (636 ms → 570 ms on mixed 200,000 × 14, against 573 ms with it off).

### Fixed
- **Search and the type filters did nothing on any report with ten columns or fewer.** The pagination module returned early when a single page was enough and hid the whole controls row, so the search box and the Numeric/Categorical/Datetime/Boolean tabs were rendered but never wired. Only the page buttons are hidden now.
- **The distinct-count estimate on an identifier card is clamped to the row count.** KMV carries about 2.2% error at k=2048, so a real key could report more distinct values than rows — arithmetically impossible, and it reads as a bug.

## [0.0.34] - 2026-08-15

Benchmark tooling only; no library changes.

### Added
- **`benchmarks/versions.py`** — version-over-version timing in one interleaved round-robin. Each version is installed into its own throwaway virtualenv and timed in its own subprocess, so import cost, allocator state and garbage from one version cannot leak into the next.

### Changed
- **`benchmarks/end_to_end.py` schedules a round-robin by default** (`--rounds 5`). Every tool is measured in every round and each one's best is reported, so a slow patch of machine time penalises everything in that round and cancels in the ratio. Running one tool to completion and then the next compares them across two different stretches of machine time, which on a shared runner is not the same machine.
- **Both harnesses refuse to imply a quotable ratio below three rounds**, in the terminal and in the generated markdown. The generated tables also carry the round count and the per-tool spread, and state that ratios are only comparable within the table they appear in.

Two published claims came from cross-session pairing, which is what this exists
to prevent: *"0.0.21 is 1.24x faster than 0.0.16"* is really **0.88x** — a
regression reported as an improvement — and a *3.56x* headline is really
**2.48x**, from pairing a slow baseline run with a fast recent one.

## [0.0.33] - 2026-08-15

The first four of the twelve user-experience findings (#72, #79, #81, #83).

### Fixed
- **A numeric column's classification changed as the table grew.** `age` with 67 distinct values profiled as *numeric* in a 1,000-row frame and *categorical* in a 20,000-row one, because the rule fired on `unique_ratio < 0.05`. Every bounded integer — age, year, rating, day-of-month, HTTP status, state code — crossed the line purely by adding rows. The rule is now a cardinality **ceiling** (50 distinct, integral values only), which is stable under row count. For a profiler whose pitch is large data, a heuristic that degraded with scale was backwards.
- **Whether reclassification ran at all depended on `chunk_size`.** The streaming guard asked whether the *first chunk* held every row, so an in-memory frame larger than one chunk was treated as a stream and skipped reclassification entirely — the same column came back categorical at 50,000 rows and numeric at 200,000. The question is about the source, not the chunk: an in-memory frame is fully known however the engine splits it. Streams are unaffected, and still stay numeric.
- **`repr(report)` returned the whole document.** The dataclass default rendered every byte of `html`, so a bare `report` in a REPL printed over a megabyte and any traceback carried the report inline. It is now one line naming the shape and size.
- **A column of nothing but infinities raised `UnboundLocalError`** in `finalize()`.

### Added
- **`profile()` and `summarize()` accept a file path**, `str` or `PathLike`, for `.csv`, `.parquet` and `.json` — the same formats the CLI has always read. `profile("data.csv")` raised `TypeError` while `pysuricata profile data.csv` worked.
- **`py.typed`**, so annotations are visible to type checkers rather than inferring as `Any`.
- **`__all__`**, so `dir(pysuricata)` is the public API rather than a list of internal submodules.
- **`PySuricataError`**, one base for everything the library raises deliberately. `UnsupportedDataError` and `ConfigurationError` subclass it *and* the builtin they used to raise, so existing `except TypeError` / `except ValueError` handlers keep working.

### Changed
- A string argument is now read as a path, so passing an unusable one reports `File not found` rather than an unsupported-type error.

## [0.0.32] - 2026-08-15

Documentation only; no library changes. **90 documented errors down to zero**,
and CI now fails a PR that reintroduces one.

### Fixed
- **Seven pages documented two configuration options that do not exist.** `config.compute.uniques_sketch_size` and `config.compute.top_k_size` were never real names — they are `max_uniques` and `top_k` on `ComputeOptions`. A reader following the docs got a silently ignored setting rather than an error, which is worse. 27 occurrences across `configuration.md`, `api.md`, `performance.md`, `why-pysuricata.md`, `architecture.md`, `complexity-analysis.md`, `faq.md` and `quickstart.md`. The internal accumulator configs, where those names *are* real, are untouched.
- **`summarize()` field names that do not exist in the payload.** The docs promised `skewness`, `true_count`, `true_pct`, `balance_score`, `distinct`, `top_values`, `gini`, `entropy`, `hour_distribution` and `["dataset"]["rows"]`. The real fields are `unique_est`, `top_items`, `true`/`false`, `min_ts`/`max_ts` and `rows_est`; skew and entropy are not exposed at all. Every example now prints something the payload actually contains.
- **Examples that could not run.** 25 fences called `profile()`, `summarize()` or `ReportConfig()` with no import; others referenced frames and columns that were never defined. 97 of the 98 runnable fences now execute end to end exactly as pasted — the one exception needs `hypothesis`, a test-only dependency.
- **Fifty-three snippets silently assumed a DataFrame named `df`.** Pages that share one now say so, with a paste-able block at the top that the checker executes as the page's stated setup.
- **Prose describing behaviour that was removed.** `architecture-diagrams.md` still said extremes were tracked "every 5th chunk only" (removed in 0.0.26, they are exact) and that type inference used the "first chunk only" (gated on `first_chunk_is_whole` since 0.0.24). `sketches.md` taught KMV with `hashlib.md5`, where the library uses blake2b and a vectorised splitmix64, and presented Algorithm R as the reservoir sampler without noting that Algorithm L is what ships.
- Dropped the hand-written `Last updated: 2025-10-12` footer from seven pages. A date nobody remembers to change is worse than no date.
- `docs/roadmap.md` was on disk but missing from the nav, so it was never rendered.

### Added
- **`check_docs --strict` and the generated-asset check run in CI.** A renamed option or a moved summary key now fails the PR that did it. The checker had been papering over the largest defect class by injecting a `df` into every snippet, so every fence passed while a reader pasting the same code got `NameError`; it now runs each page's own declared setup instead. Three of its own false-positive classes are fixed too: fences nested in tabbed blocks are dedented before parsing, column names inside `summarize()["columns"][...]` are no longer checked against a synthetic frame, and filenames in badge URLs are no longer read as attribute access.

## [0.0.31] - 2026-08-15

Documentation and tooling only; no library changes.

### Added
- **Six interactive figures**, embedded on the pages they explain: reservoir sampling (Algorithm R against Algorithm L), Misra-Gries eviction, the memory curve, the chunk lifecycle, the Welford-to-Pébay merge, and an annotated report card. Each runs the real algorithm in the browser with a fixed seed, so they are simulations rather than illustrations. They ship as plain HTML, SVG and vanilla JavaScript — no React, no CDN, no third-party requests, and they work offline.

### Fixed
- **`docs/algorithms/sampling.md` documented an algorithm the library does not run.** The page described Algorithm R — one random draw per element, testing every arrival — while `ReservoirSampler` has used Algorithm L with a bulk acceptance schedule since 0.0.23. The class name was the only accurate thing on the page; the constructor signature, the field names and the method were all wrong for anyone reading it to understand the code. Rewritten against the implementation, keeping Algorithm R as a labelled contrast.
- **The pre-commit test hook rebuilt the project on every commit.** `uv run` re-resolved and reinstalled the package against a tree pre-commit had partially stashed, which hung commits and made a test-only hook report "files were modified by this hook". It also ran a three-file subset chosen for having no optional dependencies — 25 tests that could not catch an accumulator regression. It now runs the statistical core and the invariants most likely to break, and the whole hook suite takes 2.8 seconds.
- **Generated documentation assets were rewritten by `end-of-file-fixer`** because the generator omitted a trailing newline, after which the generator's own `--check` mode reported drift against itself.

## [0.0.30] - 2026-08-15

Closes Phase 1. Mixed 200,000 x 14 is **597 ms**, down from 1,517 ms at 0.0.26
on the same machine — **2.54x**. `NumericAccumulator.update` is **83 ns/value**,
down from 1,278.

### Fixed
- **The reported minimum and maximum were sampled, not measured.** They came from the reservoir, which holds 20,000 values, while the exact extremes sat in the tracker right beside them — so a numeric card could print a "Maximum" that disagreed with the first row of its own extreme-values table, and whether it did came down to whether the true extreme happened to be sampled. Both now come from the tracker. (0.0.26 made the extremes exact and their indices global; it did not connect them to these two fields.)

### Changed
- **Monotonicity detection is a sign test on `np.diff`** rather than a Python loop over every value: 45.2 -> 0.6 ns/value in situ, 64x on the isolated kernel. The pair straddling a chunk boundary is compared against the carried last value, so chunked and unchunked results still agree.
- **The extreme-value heaps are the right way round.** Keeping the k smallest values means evicting the largest, which is a max-heap's job; the code used a min-heap and made up the difference with an O(k) `max()` scan, a linear search for the matching entry and a full `heapify` on every insert — O(k log k) per value on a structure whose purpose is O(log k) inserts. Now `heappushpop`. Measured flat at the default k=5, where the scan was over five items; the point is that it no longer degrades as `max_extremes` rises.

### Removed
- **The second reservoir in `OutlierDetector`.** Every numeric column built one and fed it 10,000 sampled values on every chunk — and nothing ever read it: `detect_outliers()` has no caller, and the outlier counts in the report are computed in `finalize()` from the accumulator's own sample. The class stays (it is exported, and its detection methods work); it is simply no longer wired into the accumulator. Worth 2.5% of the numeric path and 10,000 floats per numeric column.

## [0.0.29] - 2026-08-15

**The datetime accumulator is 9.3x faster** (308 ms -> 33 ms per 200,000-row
column), which takes mixed 200,000 x 14 from 1,175 ms to **656 ms**. Cumulative
since 0.0.26 on the same machine: **2.31x**.

### Changed
- **`DatetimeAccumulator.update` is vectorised.** It was the most expensive column kind by a factor of two and the only accumulator never touched. Four per-row Python loops are gone: the validity mask, the `int()` conversion, the sketch/reservoir feed, and — the expensive one — a `datetime.fromtimestamp` object constructed per row to read four calendar fields off it. Calendar fields now come from integer division and `np.bincount`. The consume layer also stops building a `list[int | None]` per column: it hands over the int64 array it already had, since NaT's sentinel is outside the validity window anyway and needed no translation.

### Fixed
- **Hour and weekday tallies were computed in the machine's local timezone.** `datetime.fromtimestamp()` without a `tz` argument uses the local zone, while the timestamps themselves are stored as UTC — so profiling the same file in London and in Tokyo produced different "peak hour" and weekend-ratio figures, with nothing to indicate the report depended on where it ran. Tallies are now UTC, matching the data as stored.
- **A single out-of-range timestamp discarded a whole chunk's temporal patterns.** `fromtimestamp` raises `OSError` for some values on some platforms, and the handler caught it around the entire loop and moved on — so one bad row could empty the hour, weekday, month and year histograms for every row beside it.
- **Timestamps at the bottom of the validity window were decomposed wrongly.** Casting `datetime64[ns]` to a coarser unit overflows there: numpy reports 1677-09-21 as day *+106750*, sign flipped, which yields hour 46. The decomposition now divides in integers, which is exact and floors correctly for pre-1970 instants.
- **Values rejected by the validity mask were not counted as missing** on the element-wise path when a later conversion failed.

## [0.0.28] - 2026-08-15

**1.29x faster on mixed 200,000 x 14** (1,517 ms -> 1,175 ms), and two numeric
cards now say nothing where they used to say something untrue.

### Removed
- **The "Common values" table no longer appears on high-cardinality numeric columns.** Misra-Gries ran on every numeric column unconditionally, so a column of 200,923 distinct floats rendered a ranked table of values that had occurred *once*. Top-k is now gated on the distinct estimate and fed only while its answer could carry information; the gate latches off and discards its partial counts, so the table a column gets does not depend on how it was chunked. Columns with fewer distinct values than counters, and columns the counters can meaningfully cover, are unaffected. This is also 34% of the numeric accumulator.
- **A fallback in `NumericAccumulator.finalize()` that invented common values.** When the sketch returned fewer than five entries it recomputed them from the reservoir sample and multiplied the counts by the sampling ratio "to represent the full dataset" — reporting a value that occurred once as having occurred `sample_scale` times, formatted in the report exactly like a measured count. It also overrode the *exact* counters on any column with fewer than five distinct values, replacing a correct answer with an estimate. An absent table is the honest output when nothing is common.

### Changed
- **`chunk_size` now defaults to 50,000 rows, down from 200,000.** The old value was never exercised: until 0.0.25 the option was blended away, so nothing depended on it being right. Bigger is not faster — the sketch merges are superlinear in batch size, so one 200,000-row batch costs more than four 50,000-row ones. Measured optimum is 50,000, worth 1.13x on its own once the KMV pre-filter is in. A test now pins the chosen size to a band so the default cannot drift back.
- **KMV rejects hashes against its admission threshold before sorting them.** Once the sketch is full, the kth smallest hash it holds is a hard bound — nothing at or above it can enter, now or later. Testing that first with one vectorised compare discards over 99.9% of a batch from a high-cardinality column, leaving `np.unique` and `np.union1d` to sort the survivors instead of the whole chunk. 51 -> 17 ns/value; the retained set, and therefore every estimate, is identical by construction.

### Fixed
- **Pre-1906 timestamps were still dropped on the fallback path.** The window widened in 0.0.26 missed `_update_fallback`, which kept the old `-2e18` bound. Same symptom as before — historical dates counted as missing — on the path taken when a timestamp resists array conversion.

## [0.0.27] - 2026-08-15

### Changed
- **Sampling draws from per-column generators instead of the process-global RNG.** `random_seed` used to be applied by calling `np.random.seed()` and `random.seed()`, which meant profiling reset the caller's generators; 0.0.18 papered over that by snapshotting and restoring them around each run. The sketches now each own a `numpy.random.Generator`, seeded per column as `blake2b(f"{run_seed}:{column}")`, and the snapshot/restore wrapper is gone — `profile()` neither reads nor writes global RNG state, seeded or not. Two consequences worth knowing: the same seed gives a *different* sample than it did in 0.0.26 (PCG64 rather than the legacy Mersenne Twister, and a per-column seed rather than one shared stream), and a column's sample no longer depends on which other columns are present, so profiling a subset now reproduces the numbers from profiling the whole frame. This is what per-column threading needs to be reproducible.
- **The sample-preview table is reproducible.** It called `df.sample()` with no `random_state`, so the preview rows were drawn from the global RNG and ignored `random_seed` entirely. Both backends now take an explicit seed derived the same way.

### Fixed
- **`Accumulator.update()` crashed on numpy arrays and pandas Series.** The categorical, datetime and boolean accumulators guarded with `if not arr`, which raises `ValueError: truth value of an array ... is ambiguous` for exactly the array types the library passes internally — the categorical path even converts its input to a Series on the next line. Now guarded on length, as the numeric accumulator already was.
- **`NumericAccumulator.reset()` raised `AttributeError` on the default configuration.** It called `reset()` on three components that had none (`StreamingMoments`, `OutlierDetector`, `PerformanceMetrics`), two of which are enabled by default. Those methods now exist, and `reset()` also clears chunk metadata, which it was leaving in place for the next run to append to.

### Removed
- **`NumericCardRenderer._simulate_chunk_distribution`** — dead code with no caller in the render path that fabricated plausible-looking chunk sizes and missing-value counts from a global-RNG draw. Invented data has no place in a report, and this was the last thing in the package touching the stdlib `random` module.

## [0.0.26] - 2026-08-15

### Fixed
- **Timestamps before 1906-05-13 were counted as missing.** The validity window's lower bound was `-2e18` ns, commented as "roughly 1900-2100". Birthdates and historical records fell outside it and were reclassified as nulls, so a column of 19th-century dates looked almost entirely missing rather than old — with the count, the missing percentage and the reported date range all wrong together, and nothing to indicate why. The window is now the range `datetime64[ns]` can actually represent: 1677-09-21 to 2262-04-11.
- **Extreme-value row indices were chunk-local.** `NumericAccumulator.update` numbered rows with `np.arange(len(chunk))`, so "row 4,182 had the maximum" named a position inside whichever chunk the value arrived in — wrong for every chunk after the first. The engine already tracked a global row offset for chunk metadata; it now passes it down.
- **The reported minimum and maximum could miss the true ones.** A second extreme-tracking pass in the consume layer ran only on every fifth chunk. It was also redundant, feeding the same tracker a duplicate chunk-local copy of each extreme — which is why one extreme value could appear twice under two different indices. That pass is removed; extremes come from the accumulator's own pass, on every chunk.

## [0.0.25] - 2026-08-14

### Fixed
- **`ComputeOptions.columns` now restricts what is profiled.** It was documented and validated, but never reached the engine — asking for three columns of a hundred profiled all hundred. Applied per chunk, so it works for streaming sources too. Names that are not present are ignored rather than raising, since a stream may legitimately vary.
- **`corr_max_cols` now caps correlation analysis.** It was declared, documented, validated and copied into the config, then never read: a 1,000-column frame built 499,500 pairs despite a documented cap of 50. The cap is applied before pair construction, which is the quadratic part.
- **`chunk_size` is now the size you asked for.** It was blended as `0.7*optimal + 0.3*requested`, so the caller never got the requested size — which quietly defeats any attempt to reason about or test chunk-dependent behaviour. An explicit request is now honoured, clamped only to the chunker's bounds; adaptive sizing applies only when no size is given.

## [0.0.24] - 2026-08-14

### Changed (behavioral — streaming sources)
- **Numeric columns are no longer reclassified as categorical from the first chunk of a stream.** The heuristic reads the distinct-value ratio of the first chunk, which is evidence about the column only when the chunk *is* the column. On a stream it is not: a sorted column, or one with a leading run of a single value, presents a prefix that looks low-cardinality while the column is not — and nothing revisited the decision. A 285,000-row column with 244,255 distinct values was profiled as categorical because its first 45,000 rows held nine. Reclassification now runs only when the first chunk provably contains every row. The trade-off, stated plainly: a genuinely low-cardinality *streamed* column now renders a numeric card rather than a categorical one. Little is lost, since the numeric accumulator already tracks top values via Misra-Gries. In-memory frames are unaffected.

### Fixed
- **The row count silently truncated to 2,000 per chunk when row hashing failed** — the fallback path stringified a 2,000-row sample to feed the duplicate sketch, and then counted *the sample* rather than the chunk. A 50,000-row chunk contributed 2,000 rows. That figure is what the report prints as "Rows" and what `missing_cells_pct` divides by, so a single unhashable column (one holding lists, say) corrupted the headline row count and every missing-value percentage in the report. The sample now bounds only what the sketch sees; every row is counted. Affected the pandas path and all three polars fallbacks.
- **The duplicate estimate is now marked as degraded** when the sketch has seen less than the full data, via `RowKMV.duplicates_degraded`, and clamped so it can never exceed the row count.

## [0.0.23] - 2026-08-14

**2.30x faster than 0.0.20** on the mixed benchmark suite (200,000 x 14: 3.190s
-> 1.384s), with the sampling guarantees from 0.0.18 intact.

### Changed
- **Reservoir acceptances are scheduled in bulk** — Algorithm L's schedule depends only on the random generator and the reservoir size, never on the data, so there is no reason to derive it one acceptance at a time in Python. Writing the recurrence as a cumulative sum makes every term a vectorised array operation: `log W = cumsum(log u)/k`, `skip = floor(log v / log(1-W))`, `index = base + cumsum(skip) + i`. Accumulation drops from 59.1% to 49.9% of self time. The schedule is still generated from the draw sequence alone, so the sample remains identical however the stream is chunked.

### Added
- Tests covering the schedule's block boundary — a stream long enough to force several refills must still give an identical sample for 1, 13 and 977 chunks — plus strictly-increasing acceptance indices, in-range slot choices, and an implementation-independent check that the sample mean tracks the population.

## [0.0.22] - 2026-08-14

Completes Phase 1 of `docs/roadmap.md`. **Report generation is 1.99x faster than
0.0.20** on the mixed benchmark suite (200,000 x 14: 3.190s -> 1.601s), with no
new dependencies and no API change. The phase's exit criterion was hashing and
date parsing each under 5% of self time; they are now 0.7% and 0.3%.

### Changed
- **Date sniffing no longer parses columns row by row** — deciding whether an object column holds dates ran up to 10,000 rows through `pd.to_datetime(format="mixed")`, which disables pandas' vectorised parser and falls back to `dateutil` one row at a time: 166,302 `get_token` calls in a single 50,000-row profile, 20.7% of total runtime. It now probes 200 rows against a list of explicit formats, each of which takes the fast path, and only reaches for `mixed` when every fixed format has failed. Classification is unchanged, including for formats outside the fixed list.

### Fixed
- **Empty and all-null object columns no longer compute 0/0** while sniffing, which produced a `RuntimeWarning` and a `nan` success rate.

### Added
- 12 tests covering date-format classification, the `dateutil` fallback for unusual formats, all-null columns, and a bound on how much of a large column is parsed.

## [0.0.21] - 2026-08-14

Phase 1 of `docs/roadmap.md` — pure-Python performance, no new dependencies and
no API change. **Report generation is 1.83x faster** on the mixed benchmark
suite (200,000 x 14: 3.190s -> 1.739s).

### Changed
- **The distinct-count sketch no longer uses SHA-1** — a cryptographic hash resisting preimage attacks was doing a job that needs uniformity and avalanche and nothing else, at roughly a third of total runtime. Numeric columns are now hashed by bit pattern with a vectorised splitmix64 finaliser, costing no Python object per row; byte input uses blake2b with an 8-byte digest, so there is no wider digest to slice. An avalanche test asserts a single input bit flip changes about half the output bits.
- **Reservoir sampling draws its uniforms in blocks** — Algorithm L needs two draws per acceptance, and a scalar `np.random` call is dominated by interpreter overhead. Slot selection reuses the same buffer instead of a separate `randint` call.

### Fixed
- **Repeated values inflated the distinct count** — the sketch kept its k retained hashes in a list that was never de-duplicated, so a value seen twice could occupy two slots. Since the estimator reads the retained count as a distinct count below k, 5,000 distinct values repeated 20x estimated as ~101,800. It now estimates 4,928 — 1.4% error, well inside the sketch's bound.
- **Leaving exact-counting mode double-counted the spill** — every value counted so far was moved into the sketch and then offered to it a second time, so a 1,000-distinct-value column reported 1,100.
- **`KMV.add()` disagreed with the batch path** — it keyed the exact counter by the value's bytes while batches keyed by hash, so the same value could count twice depending on which path it arrived through; and on the branch that crossed the exact-tracking limit it inserted the current hash, then fell through and inserted it again.

### Fixed
- **Granularity detection crashed on very small numbers under numpy >= 2.5** — the step-size histogram guarded only that the spread was strictly positive. For values around 1e-15 the gaps between them differ by ~1e-31, which is narrower than one float64 ULP at that magnitude, so every computed bin edge rounds to the same value. numpy 2.5 rejects that outright (`Too many bins for data range`) where 2.1 silently returned degenerate bins, so profiling any column at that scale raised. Differences that are equal to within floating-point resolution now skip the histogram entirely — the granularity simply is that difference.
- **numpy floor for Python 3.14** — numpy 2.1.3 publishes no cp314 wheels, so the resolver picked it on 3.14 and built it from source against a Python it never supported. The resulting binary computed `uint64` arithmetic wrongly for large arrays, collapsing every hash in the distinct-count sketch to the same value and reporting 300,000 distinct values as 1. Floored to `numpy>=2.3.3` on 3.14, the first release with cp314 wheels.

### Added
- `tests/test_accumulators_core.py` — 48 unit tests for the statistical core, asserting the mergeability and chunk-invariance properties the whole streaming design rests on. Coverage of `sketches.py` rises 84% → 85% and `algorithms.py` 79% → 85%.

## [0.0.20] - 2026-08-14

### Added
- **Vendored the native core crate** (`native/`) — the optional Rust kernels (`pysuricata-core`: hashing, KMV, moments, reservoir) are now tracked in git. Storing the source is not the same as starting Phase 3: nothing imports it, no build runs it, and the 37 native agreement tests in `benchmarks/accuracy.py` stay skipped until someone runs `maturin develop`. It was previously untracked working-tree state, one `git clean` from being lost.
- `.gitignore` rules for Rust build artifacts. `Cargo.lock` is deliberately *not* ignored, since this crate ships wheels and pinning the versions that built them is what makes a release reproducible.

## [0.0.19] - 2026-08-14

### Added
- **The accuracy oracle now runs in CI** — a new `Accuracy` workflow runs `benchmarks/accuracy.py` on every pull request. The six statistical bugs fixed in 0.0.18 were only findable because that suite exists; nothing ran it automatically, so they could have regressed silently. This is the Phase 0 exit criterion from `docs/roadmap.md`.
- **Slow end-to-end invariants run on every push to `main`** — the chunked-vs-unchunked checks take tens of seconds per case, so they stay off the pull-request path but gate the branch.

### Changed
- **`xfail_strict` is enabled** — an `xfail`-marked test that starts passing (XPASS) now fails the build instead of passing quietly, so a fixed bug cannot leave a stale marker behind claiming it is still broken.

## [0.0.18] - 2026-08-14

Correctness release. `benchmarks/accuracy.py` — a new statistical oracle that
checks chunked results against unchunked ones and against NumPy — shipped six
`xfail`-marked tests, each naming a live bug. All six are fixed.

### Fixed
- **Generator sources silently dropped the first chunk** *(critical)* — adapter sniffing consumed the first chunk of a generator, so the documented "stream chunks larger than RAM" API omitted chunk 0 from every statistic, and a single-chunk generator reported `Empty source`. Chunk counts, `min`/`max`, means and every sketch were wrong for streaming input.
- **Reservoir sampling was biased toward late elements** *(critical)* — `add_many` used one uniform draw over the post-batch count instead of a denominator that grows within the batch, and the bias grew with chunk size. Replaced with Algorithm L (Li, 1994). Every quantile, the median, IQR, MAD, outlier count and the histogram derive from this reservoir; for a fixed seed the sample is now identical regardless of chunking, and Algorithm L also reduces random draws from one per row to roughly `k·ln(n/k)`.
- **Skewness and kurtosis were wrong for multi-chunk data** *(critical)* — the M3/M4 batch merge was "simplified" and not Pébay's formula, so it disagreed with the correct `merge()` in the same class. Results were right only for single-chunk input. Now exact across any chunking.
- **`profile()` reset the caller's global RNG** — seeding for reproducibility wrote to the process-global NumPy and stdlib generators, silently resetting a caller's own seeded state. The state is now snapshotted and restored, including when report generation raises.
- **Correlations collapsed to 0.00 on large-mean columns** — the naive `sx2 - sx*sx/n` variance cancels catastrophically for timestamps-as-int, IDs or prices near 1e6, and `max(0.0, …)` hid it. Switched to Welford/Chan pairwise co-moments.
- **Skewness used the sample variance in its denominator** — g1 is defined against the population second moment; the n−1 form biased it by ((n−1)/n)^1.5 and never converged away.

### Added
- **`corr_top` in `summarize()` output** — correlations were computed and rendered into the HTML report but never emitted in the JSON summary, so the programmatic contract was strictly weaker than the visual one.
- **`benchmarks/` accuracy oracle and performance harness**, plus `docs/roadmap.md`.
- **`no-commit-to-branch` pre-commit hook** — work reaches `main` only through a pull request.

### Changed
- The pre-commit `ruff` pin and the dev-group `ruff` had drifted either side of the `UP038` rule's removal, so pre-commit rejected code that CI accepted. Both now pin the same version.

## [0.0.17] - 2026-08-14

### Changed (behavioral — review before upgrading)
- **Automatic boolean detection is now more aggressive.** Integer 0/1 columns without a
  boolean-sounding name were previously profiled as **categorical** and are now profiled as
  **boolean**, changing which card type is rendered. (Titanic's `Survived` is a typical
  example.) Columns that already had a boolean-sounding name are unaffected. Two defaults
  changed, in both `EngineConfig` and `ComputeOptions`:
    - `boolean_detection_require_name_pattern`: `True` → `False` — a column no longer needs a
      boolean-sounding name (`is_*`, `has_*`, …) to be promoted; values alone are enough.
    - `boolean_detection_max_zero_ratio`: `0.95` → `0.80` — columns more than 80% zeros are no
      longer promoted (previously 95%), so heavily-skewed indicator columns stay numeric.

  To restore the previous behavior:

  ```python
  from pysuricata.api import ComputeOptions, ProfileConfig, profile

  profile(df, config=ProfileConfig(compute=ComputeOptions(
      boolean_detection_require_name_pattern=True,
      boolean_detection_max_zero_ratio=0.95,
  )))
  ```

  Set `enable_auto_boolean_detection=False` to turn the promotion off entirely.

### Added
- **CSS integrity test suite** — `test_css_integrity.py` with 9 automated checks (file presence, selector coverage, `!important` budget, breakpoint standardization, inline handler removal)
- **Pre-commit hooks** — `.pre-commit-config.yaml` with trailing-whitespace, end-of-file-fixer, check-yaml/toml, ruff lint+format, and fast pytest
- **Colored header icons** — Sun/moon toggle, calendar, clock, download, and pin SVG icons now use theme-appropriate colors instead of monochrome
- **Extended dtype inference** — pyarrow-backed columns (`pd.ArrowDtype`) are now classified by their underlying Arrow type instead of falling through to categorical; pandas `timedelta64` and polars `Duration`/`Time` are treated as numeric; polars `String` is recognized alongside the legacy `Utf8` alias
- **Vendored Titanic dataset** — `docs/assets/titanic.csv` is now committed, so `scripts/regenerate_example_report.py` and the docs CI jobs no longer depend on network access

### Changed
- **CSS modularization** — Replaced monolithic `style.css` (8,742 lines) with 14 scoped partials (`_00-tokens.css` through `_13-utilities.css`), loaded via `load_css_dir()` with caching
- **Inline event handler removal** — Replaced all inline `onclick`/`onchange` handlers with `data-action` attributes and delegated event listeners
- **Ruff lint cleanup** — Auto-fixed 656 issues, reformatted 36 files, manually fixed F821/E711/B007/F401 across 7 source files
- **Report size reduction** — HTML output is ~15% smaller (1.17MB vs 1.38MB for Titanic dataset)
- **Polars string→boolean inference** — Under the `AGGRESSIVE` strategy, string columns are now matched against an explicit token set (`true`/`false`/`1`/`0`/`yes`/`no`) rather than `cast(pl.Boolean, strict=False)`, aligning polars behavior with pandas
- **Quality flags** — `case_variants` and `trim_variants` are now raised only when lowercasing or stripping actually reduces the unique count, so a disabled estimator no longer reports phantom variants

### Fixed
- **Duplicate column names no longer crash profiling** — pandas frames with repeated column names are renamed with numeric suffixes (with a `UserWarning`). Suffix generation now skips names already present in the frame, so `["a", "a", "a_1"]` renames to `["a", "a_2", "a_1"]` instead of producing another duplicate and failing with `'DataFrame' object has no attribute 'dtype'`
- **Boolean columns misclassified as numeric** — pandas `is_numeric_dtype()` returns `True` for `bool` dtype, so the bool check now runs first in `_infer_pandas_series_type`
- **Report titles and descriptions containing braces are no longer corrupted** — template substitution is now a single regex pass, so a value such as `title="My {report_date} report"` is emitted verbatim instead of having its placeholder expanded by a later substitution

### Performance
- **No regression** — Report generation is ~15% faster (0.045s vs 0.053s avg on Titanic dataset, 891 rows × 12 cols)
- Template substitution makes one pass over the document instead of 34, and duplicate-column renaming uses a shallow copy rather than deep-copying the frame

### Removed
- `style.css` (monolithic), `style.css.backup`, `style_updated.css`, `chart.min.js`, `functionality.js.backup`, `cards_new.py`

## [0.0.16] - 2026-02-15

### Added
- **Polars nested type support** — Structs and Lists are now gracefully handled as categorical variables (with debug warnings) instead of causing inference errors

### Changed
- **Performance optimization** — optimized `_safe_compute` to use NumPy arrays for type checks, reducing overhead in large datasets

## [0.0.15] - 2026-02-14

### Added
- **Python 3.14 CI testing** — Added Python 3.14 to CI test matrix
- **Changelog CI check** — PRs now require a changelog entry
- **Mermaid architecture diagrams** — Replaced ASCII art with 5 interactive diagrams

### Fixed
- **MathJax formula rendering** — Fixed `ignoreHtmlClass` regex that prevented all formula rendering
- **Code/equation styling** — Changed code and math colors from green to standard gray
- **Memory stress test** — Bumped threshold from 200→250 MB for Python 3.14 compatibility

### Changed
- **Dropped Python 3.9** — Minimum version is now Python 3.10
- **CI runs on PR only** — Tests no longer run on push to main (CD handles releases)
- **Cleaned dev dependencies** — Removed `ydata-profiling` and `ipykernel` (not 3.14-compatible)
- **Cleaned examples/** — Removed benchmark scripts, generated reports, and ydata comparisons
- **Removed `.claude/skills`** — Cleaned up unused skill symlinks
- **Documentation improvements** — Rewrote API reference, complexity analysis, quality flags (tables), stats overview

### Removed
- **`report_preview.png`** — Replaced with link to live interactive report on GitHub Pages
- Stale dates from stats documentation pages

## [0.0.14] - 2026-02-14

### Added
- **Polars LazyFrame support** — LazyFrames are now automatically collected before profiling
- **ReportConfig alias** — Added `ReportConfig` as an alias for `ProfileConfig` for better API discoverability

### Fixed
- **Self-contained HTML reports** — HTML reports no longer depend on external CDN (Chart.js is now inlined)

### Changed
- **Lighter dependencies** — Removed unused dependencies: `matplotlib`, `seaborn`, `ipywidgets`

## [0.0.13] - 2026-01-02

### Added
- **CLI tool** — New command-line interface with `pysuricata profile` and `pysuricata summarize` commands
- **Comprehensive stress tests** — New `test_complexity_analysis.py` with time/space profiling
- **Python 3.14 support** — Officially supported in package metadata

### Fixed
- **Memory leak fixes** — Resolved memory leaks in KMV sketch, ExtremeTracker, and chunk metadata

### Changed
- **Realistic benchmarks** — Updated README and docs with measured performance figures

## [0.0.12] - 2025-10-25

### Added
- Continued feature work on the report ([#17](https://github.com/alvarodiez20/pysuricata/pull/17)).

## [0.0.11] - 2025-10-12

### Added
- Enhanced documentation with mathematical formulas
- Comprehensive examples gallery
- Detailed algorithm documentation (Welford, Pébay, KMV, Misra-Gries)

## [0.0.10] - 2025-10-03

### Added
- **Missing-values display** — a dedicated section summarising missing data per column ([#13](https://github.com/alvarodiez20/pysuricata/pull/13)).

## [0.0.9] - 2025-09-11

### Changed
- Stabilisation pass across the code base ([#11](https://github.com/alvarodiez20/pysuricata/pull/11)).

## [0.0.8] - 2025-09-05

### Added
- **Logging** throughout the profiling run ([#9](https://github.com/alvarodiez20/pysuricata/pull/9)).

### Fixed
- Documentation build.

## [0.0.7] - 2025-09-03

### Added
- **Chunked processing** — the first streaming pass over a file, and the point at which the library became a streaming profiler rather than a whole-frame one ([#8](https://github.com/alvarodiez20/pysuricata/pull/8)).

### Changed
- Report layout reformatted.

## [0.0.6] - 2025-08-27

### Added
- A first technical document, and a rewritten README ([#6](https://github.com/alvarodiez20/pysuricata/pull/6)).

## [0.0.5] - 2025-08-27

### Added
- **Overview, Sample and Variables sections** in the HTML report — the three-part structure the report still uses.

## [0.0.4] - 2025-03-29

### Fixed
- Images were missing from the published package.

## [0.0.3] - 2025-03-29

### Fixed
- Packaging of the HTML report template.

## [0.0.2] - 2025-03-29

### Added
- `report_template.html`, and the first examples ([#2](https://github.com/alvarodiez20/pysuricata/pull/2)).

### Fixed
- The HTML report was not included in the distributed package.

## [0.0.1] - 2025-03-26

First release to PyPI.

*Entries for 0.0.1 – 0.0.12 were reconstructed from the git history in August 2026
and are deliberately brief; the releases predate this changelog.*

[Unreleased]: https://github.com/alvarodiez20/pysuricata/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/alvarodiez20/pysuricata/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alvarodiez20/pysuricata/compare/0.0.73...v0.1.0
[0.0.61]: https://github.com/alvarodiez20/pysuricata/compare/0.0.60...0.0.61
[0.0.60]: https://github.com/alvarodiez20/pysuricata/compare/0.0.59...0.0.60
[0.0.59]: https://github.com/alvarodiez20/pysuricata/compare/0.0.58...0.0.59
[0.0.58]: https://github.com/alvarodiez20/pysuricata/compare/0.0.57...0.0.58
[0.0.57]: https://github.com/alvarodiez20/pysuricata/compare/0.0.56...0.0.57
[0.0.56]: https://github.com/alvarodiez20/pysuricata/compare/0.0.55...0.0.56
[0.0.55]: https://github.com/alvarodiez20/pysuricata/compare/0.0.54...0.0.55
[0.0.54]: https://github.com/alvarodiez20/pysuricata/compare/0.0.53...0.0.54
[0.0.53]: https://github.com/alvarodiez20/pysuricata/compare/0.0.52...0.0.53
[0.0.52]: https://github.com/alvarodiez20/pysuricata/compare/0.0.51...0.0.52
[0.0.51]: https://github.com/alvarodiez20/pysuricata/compare/0.0.50...0.0.51
[0.0.50]: https://github.com/alvarodiez20/pysuricata/compare/0.0.49...0.0.50
[0.0.49]: https://github.com/alvarodiez20/pysuricata/compare/0.0.48...0.0.49
[0.0.48]: https://github.com/alvarodiez20/pysuricata/compare/0.0.47...0.0.48
[0.0.47]: https://github.com/alvarodiez20/pysuricata/compare/0.0.46...0.0.47
[0.0.46]: https://github.com/alvarodiez20/pysuricata/compare/0.0.45...0.0.46
[0.0.45]: https://github.com/alvarodiez20/pysuricata/compare/0.0.44...0.0.45
[0.0.44]: https://github.com/alvarodiez20/pysuricata/compare/0.0.43...0.0.44
[0.0.43]: https://github.com/alvarodiez20/pysuricata/compare/0.0.42...0.0.43
[0.0.42]: https://github.com/alvarodiez20/pysuricata/compare/0.0.41...0.0.42
[0.0.41]: https://github.com/alvarodiez20/pysuricata/compare/0.0.40...0.0.41
[0.0.40]: https://github.com/alvarodiez20/pysuricata/compare/0.0.39...0.0.40
[0.0.39]: https://github.com/alvarodiez20/pysuricata/compare/0.0.38...0.0.39
[0.0.38]: https://github.com/alvarodiez20/pysuricata/compare/0.0.37...0.0.38
[0.0.37]: https://github.com/alvarodiez20/pysuricata/compare/0.0.36...0.0.37
[0.0.36]: https://github.com/alvarodiez20/pysuricata/compare/0.0.35...0.0.36
[0.0.35]: https://github.com/alvarodiez20/pysuricata/compare/0.0.34...0.0.35
[0.0.34]: https://github.com/alvarodiez20/pysuricata/compare/0.0.33...0.0.34
[0.0.33]: https://github.com/alvarodiez20/pysuricata/compare/0.0.32...0.0.33
[0.0.32]: https://github.com/alvarodiez20/pysuricata/compare/0.0.31...0.0.32
[0.0.31]: https://github.com/alvarodiez20/pysuricata/compare/0.0.30...0.0.31
[0.0.30]: https://github.com/alvarodiez20/pysuricata/compare/0.0.29...0.0.30
[0.0.29]: https://github.com/alvarodiez20/pysuricata/compare/0.0.28...0.0.29
[0.0.28]: https://github.com/alvarodiez20/pysuricata/compare/0.0.27...0.0.28
[0.0.27]: https://github.com/alvarodiez20/pysuricata/compare/0.0.26...0.0.27
[0.0.26]: https://github.com/alvarodiez20/pysuricata/compare/0.0.25...0.0.26
[0.0.25]: https://github.com/alvarodiez20/pysuricata/compare/0.0.24...0.0.25
[0.0.24]: https://github.com/alvarodiez20/pysuricata/compare/0.0.23...0.0.24
[0.0.23]: https://github.com/alvarodiez20/pysuricata/compare/0.0.22...0.0.23
[0.0.22]: https://github.com/alvarodiez20/pysuricata/compare/0.0.21...0.0.22
[0.0.21]: https://github.com/alvarodiez20/pysuricata/compare/0.0.20...0.0.21
[0.0.20]: https://github.com/alvarodiez20/pysuricata/compare/0.0.19...0.0.20
[0.0.19]: https://github.com/alvarodiez20/pysuricata/compare/0.0.18...0.0.19
[0.0.18]: https://github.com/alvarodiez20/pysuricata/compare/0.0.17...0.0.18
[0.0.17]: https://github.com/alvarodiez20/pysuricata/compare/0.0.16...0.0.17
[0.0.16]: https://github.com/alvarodiez20/pysuricata/compare/0.0.15...0.0.16
[0.0.15]: https://github.com/alvarodiez20/pysuricata/compare/0.0.14...0.0.15
[0.0.14]: https://github.com/alvarodiez20/pysuricata/compare/0.0.13...0.0.14
[0.0.13]: https://github.com/alvarodiez20/pysuricata/compare/0.0.12...0.0.13
[0.0.12]: https://github.com/alvarodiez20/pysuricata/compare/0.0.11...0.0.12
[0.0.11]: https://github.com/alvarodiez20/pysuricata/compare/0.0.10...0.0.11
[0.0.10]: https://github.com/alvarodiez20/pysuricata/compare/0.0.9...0.0.10
[0.0.9]: https://github.com/alvarodiez20/pysuricata/compare/0.0.8...0.0.9
[0.0.8]: https://github.com/alvarodiez20/pysuricata/compare/0.0.7...0.0.8
[0.0.7]: https://github.com/alvarodiez20/pysuricata/compare/0.0.6...0.0.7
[0.0.6]: https://github.com/alvarodiez20/pysuricata/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/alvarodiez20/pysuricata/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/alvarodiez20/pysuricata/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/alvarodiez20/pysuricata/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/alvarodiez20/pysuricata/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/alvarodiez20/pysuricata/releases/tag/0.0.1
