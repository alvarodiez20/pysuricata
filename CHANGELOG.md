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

Nothing yet. Planned work is tracked in
[`docs/roadmap.md`](https://github.com/alvarodiez20/pysuricata/blob/main/docs/roadmap.md),
[`docs/UX_ISSUES.md`](https://github.com/alvarodiez20/pysuricata/blob/main/docs/UX_ISSUES.md) and
[`docs/integration.md`](https://github.com/alvarodiez20/pysuricata/blob/main/docs/integration.md).

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

## [0.0.11] - 2025-10-12

### Added
- Enhanced documentation with mathematical formulas
- Comprehensive examples gallery
- Detailed algorithm documentation (Welford, Pébay, KMV, Misra-Gries)

## [0.0.12] - 2025-10-25

### Added
- Continued feature work on the report ([#17](https://github.com/alvarodiez20/pysuricata/pull/17)).

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

[Unreleased]: https://github.com/alvarodiez20/pysuricata/compare/0.0.54...HEAD
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
