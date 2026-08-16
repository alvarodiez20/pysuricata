# pysuricata report UI — integration plan

Design decisions settled with the designer, written for implementation in the
`pysuricata` repo. Each phase is independently shippable and independently
reviewable. Work them in order — phase 1 establishes tokens every later phase
depends on.

> **Amended 2026-08-16.** Three additions to the plan as handed over, marked
> *(amendment)* throughout: a **commit 0** carrying the data fixes that were at
> commit 12, a **phase 5b** for the partial report, and a **phase 10** for resume.
> The rationale for each is at the point of insertion. Testing for the whole
> migration is in [`docs/MIGRATION_TESTING.md`](MIGRATION_TESTING.md), which
> replaces the "Tests to update" list at the end of this file.

**Constraints that hold throughout**

- Reports stay **single-file**: inline CSS, inline JS, inline SVG. No external assets, no web fonts.
- Dark mode keeps working (`#pysuricata-report:not(.light)`).
- No new JS dependencies. CSS-only toggles stay CSS-only where they already are.
- Charts stay hand-rolled SVG (`render/histogram_svg.py`, `temporal_charts.py`, `donut_chart.py`).

## What's in this package

| File | Use |
| --- | --- |
| `integration.md` | this plan |
| `assets/tokens.css` | drop-in replacement for the token layer of `_00-tokens.css`, with a compatibility shim so the existing CSS keeps working while phases land |
| `assets/test_contrast.py` | drop into `tests/`. Parses the tokens and asserts every pair the design uses clears its WCAG minimum, in both themes |
| `assets/logo_mark.png` | the meerkat alone, cropped from `logo_suricata_transparent.png`. **Re-export from the original vector before shipping** — ideally as inline SVG, which removes the dark-mode duplicate |
| `assets/logo_wordmark.png` | the wordmark alone, for reference |
| `*.dc.html` | the design files. Open directly in a browser |
| `source/` | the two generated reports every number in the designs was taken from |

**Design files, and what each decides**

| File | Covers |
| --- | --- |
| `Report Baseline.dc.html` | faithful recreation of today's header / summary / sample, for before-and-after |
| `Report Screen.dc.html` | chosen header + summary + sample assembled, 1240px and 390px |
| `Variables 4b.dc.html` | variables layout for all five column types, plus the axis treatment |
| `Palette.dc.html` | the colour system, and the two-dataset comparison |
| `Correlations and Missing.dc.html` | correlations (three states) and missing values (two options) |
| `Variables.dc.html` | superseded exploration of three variables layouts. Kept for the reasoning; 4b won |

Every figure in the designs comes from your own output: the Titanic report for
everything except the datetime card, which uses `daily_2026-08-16`. Two
exceptions are labelled in the files themselves — the correlation values above
0.5 (Titanic has none) and the second series in the comparison charts.

---

## Phase 0 — Correct the numbers first *(amendment)*

Files: `accumulators/`, `render/card_base.py`, `compute/`

Nothing in this phase is design work, and all of it must land **before** phase 1.
The reason is baselines: every golden payload, fact list and screenshot taken
during this migration becomes the reference the next sixteen commits are checked
against, and three of the numbers in today's output are wrong. A baseline taken
from a wrong report certifies the wrongness.

### 0.1 Clamp the distinct estimate

Phase 5.7 already carries this: `Name` reports **892 distinct against 891 rows**.
It is here instead because it touches `accumulators/`, so it needs the accuracy
oracle rather than a snapshot review — a different kind of review from every
other commit in the sequence, and one that should not be buried at position 12.

Clamp to `min(estimate, count)`, set `approx: True` for any value that came from
the sketch rather than the exact counter, and keep the `(≈)` marker.

### 0.2 The quasi-constant rule *(not in the plan as handed over)*

`age` — 67 distinct integers over 18–85 in 20,000 rows — is flagged
**Quasi-Constant**, severity `bad`, from `data-value="0.3%"`: a unique *ratio*.
The type classifier was fixed for this case; the same reasoning survives in the
quality-flag layer.

This matters here because **phase 5.7 makes it worse**. Showing a flag's value
against its threshold, outlined in its severity colour, renders this false alarm
more legibly — and phase 3's triage block puts it at the top of the page. On a
two-column frame the report currently opens with *"1 of 2 columns need a look"*
and names the healthy column.

Replace the ratio test with the cardinality ceiling used by the classifier, and
add the test that profiles the same column at 1k / 100k / 10M rows and asserts
the flag set does not change.

### 0.3 `finalize()` must not consume randomness

`finalize()` advances `ReservoirSampler._rng`. Every call therefore changes every
subsequent sampling decision, so **`checkpoint_write_html=True` silently changes
the report's quantiles, median, IQR, MAD and histogram**. Verified: finalising at
chunk 3 of 6 and continuing diverges from an uninterrupted run on `sample_vals`
and four dependent fields.

It is a correctness bug on its own, and it is also the precondition for phase 5b
— a partial report is a mid-stream `finalize()`, so until this is fixed, any
progressive rendering corrupts the run it is reporting on.

**Acceptance:** `unique <= count` for every column; `age` raises no flag at 1k,
100k and 10M rows; `finalize()` mutates nothing, and a mid-stream `finalize()`
followed by more chunks equals an uninterrupted run field for field; the accuracy
oracle gains a case for **profiling with partial renders on equals profiling with
them off**; all 51 existing oracle tests still pass.

---

## Phase 1 — Tokens and typography

Files: `static/css/_00-tokens.css`, `_01-base.css`

Replace the token layer with `assets/tokens.css`. It carries the full palette,
the measured contrast ratios, the dark-mode proposal, and a compatibility shim
mapping the legacy variable names onto the new scale so nothing breaks mid-way.
**Delete the shim once phase 12 lands** — it is scaffolding.

Two scales, and they never mix: **blue means data, warm means data quality.**

Two rules worth a comment in the CSS, because they are why the palette looks like this:

1. **Type is not a colour.** The donut assigned numeric/categorical/datetime/boolean
   their own hue and the per-column cards inherited it. Inside a card the badge already
   names the type, so the hue carried nothing — and it collided: olive meant both
   "categorical" and "passes", rust meant both "boolean" and "fails". A rust bar and a
   rust warning chip in the same card meant unrelated things.
2. **`Survived` must not be red-and-green.** Colouring `false` rust and `true` olive
   reads as bad-versus-good, which is the report passing judgement on someone's data.
   Two values of one column get two steps of one hue.

One exception: **missing-value bars keep the warm scale**, because their encoding *is*
severity. `Cabin` at 77% should look worse than `Embarked` at 0.2%. Thresholds are in
`tokens.css`.

### Typography

Replace `font-family: Arial, sans-serif`.

- Prose, headings, UI labels: `var(--font-sans)`
- **Every figure, column name, dtype, axis label and code token**: `var(--font-mono)`

Monospace-for-all-numbers does real work: columns of figures align without
`font-variant-numeric` hacks, and it is the single biggest reason the report stops
reading like a generic dashboard.

Micro-label convention for every section header and stat caption: mono, 10–11px,
`letter-spacing: 0.13em`, uppercase, `--muted`. There is a `.micro-label` class in
`tokens.css`.

### Structural motif

Today the summary is eight bordered boxes inside a bordered page. That stacking is the
strongest "template" signal in the report. Replace with hairline rules and whitespace:

- Section and card edges: one `1px solid var(--rule-strong)`.
- Group headers: `1px solid var(--ink)` above the group.
- Row separators: `1px solid var(--rule)`.
- Drop `border-radius` on data containers; keep it only on chips and buttons, ≤6px.
- Delete the decorative shadows and gradients in `_00-tokens.css`: `--chart-shadow-*`,
  `--segment-shadow-*`, `--label-shadow-*`, `--legend-shadow-*`, and the
  `--chart-bg-*` / `--svg-bg-*` gradient pair.

### Two contrast failures to settle before this lands *(amendment)*

`test_contrast.py` run against `tokens.css` as shipped gives **36 pass, 2 fail**,
the same pair in both themes:

```
[light] --rule-strong (#CFC7B8) on --paper (#FBF9F5) is 1.60:1, needs 3.0:1
[dark]  --rule-strong (#46413A) on --paper (#1C1A17) is 1.72:1, needs 3.0:1
```

Resolve it by **splitting the token**, not by relaxing the assertion — the test is
only worth having while it is non-negotiable. `--rule-strong` is doing two jobs
with different requirements: a decorative row divider, which WCAG 1.4.11 does not
cover, and a **chart axis line**, which it does, as "part of a graphic required to
understand the content". Keep `--rule-strong` as the hairline and add `--axis` at
≥3:1 for axes and gridlines.

Four more pairs the design uses and the test does not check:

| Pair | Used by | Light | Dark |
| --- | --- | ---: | ---: |
| `data-4` on `paper` | below-threshold correlation bars (6.1) | **1.83** | **1.87** |
| `data-4` on `track` | palest step inside its own track | **1.55** | **1.60** |
| `data-3` on `paper` | second series | **2.63** | 3.21 |
| adjacent `data-*` segments | stacked composition bar (3.2) | **1.44–2.33** | **1.44–1.95** |

The adjacency has a fix already written elsewhere in this document: the matrix in
6.3 calls for "a 2px `--paper` gutter, so the grid reads as tiles rather than a
table". **Apply the same gutter between segments of the composition bar** and the
requirement is met without touching the palette. `--data-4` at 1.83:1 on the paper
needs a palette decision, since as a below-threshold bar fill it is a graphic
carrying information: darken it, or outline those bars in `--axis`.

**Acceptance:** `test_css_integrity.py` and the new `test_contrast.py` pass —
including the added pairs, in **both** themes, from this commit rather than from
commit 16; none of
`#3b82f6 #8ac926 #ffca3a #ff595e #4ea3f1 #f15e4e #f1c54e #60a5fa #1d4ed8` appears
anywhere in `static/css/` or `render/`.

---

## Phase 2 — Header

Files: `templates/report_template.html`, `static/css/_02-header.css`, `render/html.py`, `static/images/`

Target: one **52px** bar, down from ~96px with a 78px logo column.

### 2.1 Logo — two lockups

Use the **mark alone** in the bar at `height: 30px`, and set the product name in
**type** beside it. Keep the stacked lockup for the docs site, README and any PDF cover.

Reasons: a vertical lockup inside a horizontal bar is what forces today's tall header;
the wordmark is a hand-drawn display face whose letters land ~8px tall at bar size and
go soft as a raster; and type follows `currentColor` into dark mode, so the
`#logo-light` / `#logo-dark` swap in `_02-header.css` disappears with it.

### 2.2 Bar contents

```
[mark 30px] pysuricata │ <dataset name, mono 13px, --muted, ellipsis>
… <nav links 13.5px> │ [download] [dark]
```

- Nav: plain text, no pills, no background. Active = `--ink` with a `2px solid
  var(--q-good)` bottom border. Hover = `--ink` only.
- Download and dark-mode are **icon-only**, 30×30 desktop / 32×32 mobile, reusing the
  two SVGs already in `report_template.html` verbatim.
- The pin button either joins the icon group or goes. It currently sits alone at
  `margin-left: auto` in the meta row, and that row no longer exists.

### 2.3 Metadata gets labels

Today: a bare timestamp, a bare duration, a bare version, context only in a `title`.
Nobody can tell what `891 × 12` means, and a tooltip does not survive PDF export.
Move metadata **out of the bar**, into a hairline line under the report title:

```
Generated 2026-03-01 00:15:48 │ Profiled in 0.06 s │ Shape 891 rows × 12 columns │ pysuricata 0.0.43
```

Mono 12px, labels `--muted`, values `--ink`, 1px vertical rules between. Spelling out
"891 rows × 12 columns" retires the tooltip. On mobile, wrap to three lines.

### 2.4 Mobile

- Bar 48px: mark, name, two icons.
- Nav becomes a second row at `min-height: var(--tap-min)`, horizontally scrollable.
  **`min-height`, not `height`** — with `overflow-x: auto` a fixed height has the
  scrollbar subtracted from it, silently giving a 29px rail. Hide the scrollbar
  (`scrollbar-width: none` plus `::-webkit-scrollbar { display: none }`).
- Either fit all five labels at 390px or add a right-edge fade, so a clip reads as
  intentional rather than as a truncation bug.

**Acceptance:** header ≤52px desktop, ≤48px mobile; version from `_version.py`; every
target ≥44×44; nav rail `clientHeight ≥ 44` and `scrollWidth == clientWidth` at 390px.

---

## Phase 3 — Summary

Files: `render/sections.py`, `render/donut_chart.py`, `templates/report_template.html`, `_03-summary.css`, `_04-donut.css`

### 3.1 Stat row replaces five cards

`1px solid var(--ink)` above, `--rule` below and between cells. Per cell: mono uppercase
caption 10px `--muted`; value mono 29px `letter-spacing: -0.02em`; sub-line mono 11.5px.

| Caption | Value | Sub |
| --- | --- | --- |
| ROWS | `891` | `single pass` |
| COLUMNS | `12` | `3 num · 8 cat · 1 bool` |
| MISSING | `8.1%` | `866 cells` — `--q-warn-text` past threshold |
| DUPLICATES | `0` | `≈ KMV sketch` |
| PROCESSED | `121 KB` | `0.06 s` |

Mobile: `1fr 1fr`, six cells (add ELAPSED), value 23px. **Delete `min-height: 280px`**
on `.second-row .summary-card` — it forces three 280px boxes on a phone for content
needing a fraction of that.

### 3.2 The donut becomes a 100% stacked bar

Replace the 135px pie with one horizontal 100% bar, 32–34px tall, segments descending by
size, count printed inside each, inline legend below.

Why: a donut cannot be read to exact proportion and stops working below ~200px wide —
which is every phone. A stacked bar reads at any width and reflows for free.

Fills from the data scale by size (`--data-1`, `--data-3`, `--data-4`); segment text uses
`--on-data-*`. A type with **zero** columns gets **no segment** — render it as a muted
legend entry (`datetime 0`), because the palest blue is close to `--track` and a
zero-width segment is an artifact, not information. Keep the accessible `<desc>` text
from `donut_chart.py`; it is good.

### 3.3 Missing, and quick facts

- Top-missing rows: `grid-template-columns: 104px 1fr 100px` — name (mono), 6px bar on
  `--track` using the warm scale, count and percent right-aligned. Below: a link stating
  how many columns are complete.
- "Quick insights" five pills become one mono run:
  `12 unique · 0 constant · 2 high-cardinality · 8 text (avg len 5.8) · no date range`.
  Five bordered pills is five borders to say five short facts.

### 3.4 Description — a margin note

- **Empty:** one 44px hairline row: `DESCRIPTION` left, `+ add a note` right. That is
  the whole cost. Reports generated in a loop never have a description and must not be
  disfigured by an invitation nobody will accept.
- **Filled:** renders inline with a `2px solid var(--q-good)` left rule, a `NOTE`
  micro-label in the margin, prose at 14.5px / 1.6, and a small `edit` link. The olive
  rule marks the one human voice on the page without a card around it.

`description-editor.js` and the `data-report-id` / `data-original-markdown` contract
are unchanged; only the container moves.

**Acceptance:** summary height at 390px drops from ~1,100px to ≤560px; segment widths sum
to 100 and match `numeric_cols`/`categorical_cols`/`datetime_cols`/`bool_cols`; an empty
description costs exactly one row.

---

## Phase 4 — Sample

Files: `render/sections.py` (`dataset_sample_section`), `_05-sample.css`

Keep the table. Remove the decoration.

- Delete every cell border — the current table draws ~300 of them for 10×13 cells. Keep
  `1px solid var(--ink)` above the header, `--rule-strong` under the header, `--rule`
  under each row.
- Remove zebra striping; the row rule does that job. Keep the hover tint.
- Mono 12.5px; numeric columns right-aligned; header captions mono 10.5px uppercase `--muted`.
- **`nan` renders as an em dash in `--muted`**, not the literal string. Keep the real
  value in a `title`. That glyph is data, so it must clear 4.5:1.
- Cells clamp with `max-width: 260px; overflow: hidden; text-overflow: ellipsis`, full
  value in a `title`.
- State the overflow instead of hiding it: `12 cols · scroll →` above the table, plus
  `10 rows drawn at random from the first chunk`.

### Mobile

Freeze the row-index column: two tables in a flex row, index table `flex-shrink: 0`,
wrapper `align-items: flex-start` (otherwise the index table stretches past the scroll
pane), remainder in an `overflow-x: auto` pane that bleeds to the card edges with a
right-edge fade.

Frozen index is what makes sideways scrolling survivable on a phone: you never lose your
place in the row.

**Acceptance:** frozen index rows align with scroll-pane rows to 0px; no cell borders in
the output; `nan` never appears as a literal in a rendered cell.

---

## Phase 5 — Variables

Files: `render/card_base.py`, `numeric_card.py`, `categorical_card.py`, `boolean_card.py`,
`datetime_card.py`, `histogram_svg.py`, `temporal_charts.py`, `_06-cards.css` … `_10-boolean.css`

Chosen: **keep one card per column, restack the triple row.** Nothing collapses; every
distribution stays visible while scrolling.

### 5.1 Numeric card

Today `grid-template-columns: 240px 240px 1fr` — two narrow stat tables and a squeezed
histogram, collapsing to one column under 1024px. New order:

1. Header: column name (mono 16px), type word, dtype chip, quality flags.
2. **Histogram, full card width**, 200–240px tall. It gains ~550px of width, which is
   what finally makes 50 bins legible and the log toggle worth using.
3. Controls: `Scale linear|log`, `Bins 10|25|50`.
4. Stat row: `repeat(4, 1fr)`, one label/value pair per cell, `--rule` under each.
   Replaces both `.kv` tables.
5. Detail tabs, same content, restyled as text links with an olive underline on active.

Mobile: histogram full width at 180px, stat row `1fr 1fr`, tabs a 44px scrollable rail.

### 5.2 Axis — ruled, with units

`histogram_svg.py` prints the column name inside the chart and bare numbers on both axes.
The name is redundant once the header carries it; the **units are what's missing** —
nothing says which axis is years and which is rows.

- Drop the in-chart title.
- Y: gridlines at quarters of the rounded max, labels at each, mono 11px `--muted`.
- X: ticks at round values across the range.
- Unit labels **inside the chart**: `ROWS` above the y column, the x unit at the right
  end of the axis. Mono 10px, `letter-spacing: 0.12em`, `--muted`.
- The x unit comes from the column name and dtype. **Add the unitless branch** — omit
  the label rather than invent one.
- Clamp the first and last x label inside the plot instead of letting them overhang.

Bars `--data-2`: no stroke, no `fill-opacity`, no rounded corners.

### 5.3 Categorical

Horizontal bars, one row per level: 96px name / bar / 120px count and percent, `--data-2`
on `--track`. Below, state coverage honestly:
`3 of 3 levels shown · covers 100% of non-missing rows`.

### 5.4 High-cardinality columns — new branch

`Name`, `Ticket` and `Cabin` currently render ten bars of one row each, 0.1% apiece. That
chart says nothing. When top-5 coverage is under ~2%, or distinct approaches the row
count, **replace the chart with a sentence**:

> Every value is different. A top-values chart would be ten bars of one row each, so
> there is nothing to plot.

plus longest and shortest value, and an `identifier-like` flag. In any overview this
becomes a short note (`892 distinct, no repeats`) where other columns show a chart — and
**do not reserve the chart box** when there is no chart, or the row reads as a failed render.

### 5.5 Boolean

One split bar, 36–40px, labelled in place (`false 61.6%` / `true 38.4%`) with counts
below. **Two steps of one hue** — phase 1, rule 2.

### 5.6 Datetime

- Main chart: records-per-bucket timeline in `--data-2`, same as every other chart.
  Y unit `RECORDS`, x labelled with real dates.
- The four temporal small multiples keep their **fixed categorical axis** — 24 hours,
  7 days, **12** months — which `temporal_charts.py` already does correctly. Do not
  switch to "populated buckets only": two populated months drawn as two half-width slabs
  reads as "spread evenly across the timeline" instead of "2 of 12". Keep the tick
  labels; `Mon…Sun` is what tells a reader whether the week starts Monday.

### 5.7 Fixes found while designing

- **`Name` reports 892 distinct against 891 rows.** KMV carries ~2.2% error. Clamp the
  estimate to the row count and keep the `(≈)` marker.
- Quality flags already carry `data-threshold` and `data-value` in the DOM but display
  only the word `Missing`. Show the value against its threshold — `19.9% missing` — put
  the threshold in the `title`, and outline the chip in its severity colour rather than
  filling it.
- `.stat-badges` is `display: none` in `_06-cards.css` while still being rendered.
  Delete the renderer, not just the rule.

**Acceptance:** one card per column at every breakpoint; card height at 390px ≤600px; no
chart element emitted for high-cardinality columns; month chart has 12 slots;
`unique <= count` for every column.

---

## Phase 5b — The partial report *(amendment)*

Files: `checkpoint.py`, `compute/orchestration/engine.py`, `config.py`, `render/html.py`

Belongs here because it is a rendering concern and the renderer is already open.
It depends on **0.3**.

### What exists today

`pysuricata/checkpoint.py` writes gzipped pickles of the accumulator state every N
chunks, optionally with an HTML snapshot, and rotates them. Five configuration
options drive it. **Nothing can read them**: there is no `load`, `resume` or
`restore` anywhere in the package, and all ten tests in `test_checkpoint.py` test
the writer. Measured on 300,000 × 4 with `checkpoint_every_n_chunks=2`: three
pickles at ~335 KB and three HTML snapshots at ~1,008 KB — about **4 MB of disk
for a 1.2-second profile**, of which 57% is the logo, written again per snapshot.

### What to build instead

A partial report is worth more than the ability to resume, and costs far less. A
crash at 90% that leaves a 90% report gives value without re-running anything, and
it survives an OOM kill, a CI timeout and a Ctrl-C — none of which resume handles
gracefully. It needs no source-identity check, no offset tracking and no format
stability.

- **One file, atomically replaced.** Write to a temporary path, `fsync`, rename
  over the target. One `report.html` that keeps getting better, not one file per
  chunk. Delete the rotation logic with the per-chunk files.
- **Label it in the report itself**, in the metadata line from 2.3:
  `Partial · 6 of 40 chunks · 15% of rows`. A partial report that does not say so
  is a correctness hazard, not a feature.
- **Collapse the five options into one.** `progress_report="path.html"` plus an
  interval. `checkpoint_write_html`, `checkpoint_max_to_keep` and
  `checkpoint_prefix` all disappear; the interval should be **time-based**, since
  chunk duration varies by two orders of magnitude between a narrow numeric frame
  and a wide text one.
- **Reuse the logo decision from 2.1.** Once the mark is inline SVG, a snapshot
  costs kilobytes rather than a megabyte, which is what makes frequent snapshots
  reasonable at all.

This is also what makes the Arrow/Parquet/DuckDB work pay off: a DuckDB relation
over a directory of Parquet files is exactly the run long enough to want watching.

**Acceptance:** a long run writes a readable report from the first interval
onward; killing the process at any point leaves that file valid HTML; the partial
banner states chunk and row coverage; **a run with progressive reporting on
produces byte-identical output to the same run with it off** (this is the 0.3
oracle case, and it is the one that matters).

---

## Phase 6 — Correlations

Files: `render/correlations_section.py`, `_11-correlations.css`

Your renderer already branches on column count: matrix at ≤15 numeric columns, ranked
list above. Keep the branch, change the thresholds and add a third state.

**Routing**

| Condition | View |
| --- | --- |
| fewer than 2 numeric columns | 6.1 empty state, reason: "needs at least 2 numeric columns" |
| no pair above threshold | **6.1 empty state** |
| ≤10 numeric columns **and** viewport ≥640px | 6.3 matrix |
| otherwise | 6.2 ranked list |

Two changes to the existing thresholds: the matrix ceiling drops from 15 to **10** (at 15
the cells are 30px and the labels stop fitting), and the matrix is **width-gated** —
under 640px it falls back to the list.

### 6.1 Empty state — the common case

Both of your example reports hit this, and today it is `📊` plus
"No significant correlations found (threshold: 0.50)". But nothing was missing: three
pairs were computed and all three came back weak. That is a finding, not an absence.

Report the result:

> All **3** numeric pairs are weakly related. The strongest is **0.096**, well under the
> **0.50** reporting threshold.

then list the pairs below threshold with their real values, using the same diverging bar
as 6.2 in `--data-4`. It costs about 80px, and the numbers were computed anyway.

**Cap it.** With 40 numeric columns there are 780 pairs. Show the top 10 below threshold
and state how many were checked. `_collect_correlations` currently filters by threshold
before returning, so this needs a second call — or return everything and filter at the
render site.

### 6.2 Ranked list

Structure as today, with three substantive changes:

- **Sign is position, not colour.** One diverging bar per row: zero at 50%, negative runs
  left, positive runs right. Header reads `− 1.0 ← 0 → + 1.0`. This survives greyscale,
  needs no legend, and stops the report using red for "negative" — which reads as "bad"
  when a negative correlation is often the interesting one.
- **Three strength bands become three steps of one blue**, not three hues:
  `|r| ≥ 0.9 → --data-1`, `≥ 0.7 → --data-2`, `≥ 0.5 → --data-3`.
- **Drop the rank badges.** The list is ordered; `#1` next to the first row is noise.

Row grid `340px 1fr 84px`: pair (mono, both names with `↔`, ellipsis plus `title`),
diverging bar 14px, value `±0.000` mono 13px right-aligned. Count in the header:
`7 pairs above 0.50, of 190 checked`. Mobile stacks pair over bar.

### 6.3 Matrix

**Lower triangle only.** The full square repeats every pair twice and spends a diagonal
saying 1.00 six times — half the ink for none of the information. Row labels on the left,
column labels along the bottom.

Cell tint is `|r|` in steps of the data scale; the **sign is the printed number**, not a
colour. Cell text switches to `--ink` on the two pale steps. Cells 74px × 42px with a 2px
`--paper` gutter, so the grid reads as tiles rather than a table. Legend: the four steps
under `|r| 0 → 1.0`, with `sign printed in the cell`.

Keep `weak` cells visible but faded (`--track`) — an all-weak row is information.

### 6.4 Remove the emoji

`📊` in the empty state, `📈`/`📉` per row. Not part of this brand, and they render
inconsistently across platforms.

**Acceptance:** empty state names the pair count and the strongest value; no emoji in
`correlations_section.py`; matrix emits `n*(n-1)/2` cells, not `n²`; sign never encoded
by colour; every band colour comes from `--data-*`.

---

## Phase 7 — Missing values

Files: `render/missing_section.py`, `missing_columns.py`, `missing_values_heatmap.py`, `_12-missing.css`

**Drop the tabs.** `Data Completeness` and `Missing per Chunk` over three rows is two
clicks for one screen of content, and with a single chunk the second tab is one
full-width block per column — a tab that hides nothing.

Route on chunk count instead:

| Condition | View |
| --- | --- |
| 1 chunk | 7.1 |
| more than 1 chunk | 7.2 |

Same shape of conditional you already have for correlations, and it keeps each layout out
of the case that makes it look broken.

### 7.1 One chunk — one row per column

`grid-template-columns: 92px 1fr 110px`: column name (mono), a 14px bar on `--track`
using the warm severity scale, count and percent right-aligned in the matching severity
text colour. Legend `≤5% / 5–20% / >20%` below, and a link to the complete columns.

Same row shape as the Summary missing list, so the reader learns it once. Fits a phone
unchanged.

### 7.2 More than one chunk — split the two questions

`grid-template-columns: 92px 1fr 110px 1fr`: name, **share missing** bar, count, then a
**by-chunk strip** of equal-width segments coloured by that chunk's severity.

One bar cannot carry both meanings at forty chunks — a reader will read length as a total
when it is a sequence. Splitting them keeps each encoding answering one question.

The chunk strip is where streaming earns its keep: a column that is fine early and empty
later is a pipeline problem, and this is the only place in the report that shows it. Keep
the `data-chunk` / `data-start` / `data-end` / `data-missing` / `data-pct` attributes —
they are already right, and they drive the `title`.

Mobile stacks the strip under its row (four columns will not fit 390px).

**Acceptance:** no tab markup in the output; view chosen by chunk count; severity colours
come from `--q-*`; complete columns are summarised, not listed.

---

## Phase 8 — Comparison of two datasets

Not built yet; the colour decision is settled and should be honoured when it is.

Both datasets stay **inside the blue scale**: `--data-2` for one, `--data-4` for the
other, side-by-side bars within each bin so neither hides the other. No second hue enters
the report — the warm range stays entirely with data quality. `test_contrast.py` asserts
those two steps stay far enough apart to separate.

**Deltas are not verdicts.** Do not colour increases green and decreases red. More rows
is neutral; more missing values is not. Direction goes in a glyph, colour is reserved for
crossing a threshold:

```
ROWS       1,047    ↑ 156 · 17.5%        --ink-2
COLUMNS    12       no change            --muted
MISSING    11.3%    ↑ 3.2pp · over 10%   --q-warn-text
DUPLICATES 0        no change            --muted
```

Otherwise the page shouts about the row count and whispers about the only number that
matters.

---

## Phase 9 — Mobile and accessibility pass

Once phases 2–7 land:

1. **Every interactive target ≥44×44.** The scale/bins toggles are the easy miss — they
   are inline links whose box is just the line box. `display: inline-flex; align-items:
   center; min-height: var(--tap-min); min-width: var(--tap-min); padding: 0 8px`. Where a
   whole row is tappable, make the row the link and leave the marker `aria-hidden`.
2. **Every scrollable rail** uses `min-height`, not `height`, and either fits its labels or
   shows an edge fade.
3. **Contrast** in both themes: text ≥4.5:1, fills and borders ≥3:1. Check against the
   actual ancestor background, not the page background — a token that passes on `--paper`
   can fail on a tinted surface. `test_contrast.py` covers the token pairs; spot-check
   anything rendered on a coloured segment.
4. **Greyscale check.** Print one report to greyscale. Because type is a word and the data
   scale is one hue in steps, it should survive. Anything that becomes ambiguous is a
   colour-carrying-meaning bug.
5. No horizontal scroll anywhere except the sample scroll pane and the nav rails.

---

## Phase 10 — Resume *(amendment, and not before launch)*

Files: `accumulators/*`, `checkpoint.py`, `compute/orchestration/engine.py`, `benchmarks/accuracy.py`

Separated from 5b deliberately. The difficulty here is the project's strongest
claim: the accuracy oracle proves `chunked == unchunked` across 51 tests, and
resume introduces a **third path** that must produce identical results across a
process restart. A resume that is *nearly* right quietly breaks that claim, which
is worse than not having resume.

The good news is that the hard half is already solved in principle: `_rng` and
`_seed` live on the accumulator, so the sampling path can be reproduced.

- **Do not pickle.** Give each accumulator an explicit `to_state()` /
  `from_state()` returning plain dicts and typed arrays. Today the snapshot
  pickles live instances — `NumericAccumulator` has 30 attributes and no
  `__slots__` — so any refactor invalidates every checkpoint on disk and fails at
  *unpickling* with an `AttributeError` rather than a version error. A
  `"version": 1` field is written and never checked. It is also arbitrary code
  execution on a file that may sit in a CI cache.
- **Check identity before resuming.** Source fingerprint (path, size, mtime,
  schema hash) plus row offset. Refuse to resume against different data rather
  than emitting a plausible, wrong profile — that is the worst failure available
  here and the one to design against first.
- **Route it through `merge()`.** Resume is state plus more chunks, which is
  nearly what merge already does, and merge was fixed with 392 lines of tests.
- **Write atomically**, for the same reason as 5b: `save()` currently writes
  straight to the final path, and crashes are likeliest under memory pressure,
  which is exactly when checkpointing runs.

**Acceptance:** the oracle gains a third path and
`resumed == chunked == unchunked` holds across every fixture; a checkpoint from a
different source is refused with a clear message; the format carries a version
that is checked on load; `test_checkpoint.py` tests a round trip rather than only
the writer.

**Until this ships**, the five checkpoint options should either be wired to 5b or
removed. Options that appear to work, produce artifacts nothing can consume, and
change your results are the worst available API surface — and they are five of the
twenty-two fields that make `ComputeOptions` a cognitive-load problem.

---

## Commit sequence

| # | Commit | Touches |
| --- | --- | --- |
| **0a** | **clamp the distinct estimate; `approx: True` for sketch values** | `accumulators/`, `render/card_base.py` |
| **0b** | **replace the quasi-constant ratio rule with the cardinality ceiling** | `compute/`, `render/card_base.py` |
| **0c** | **stop `finalize()` consuming the sampler's RNG**, + the oracle case | `accumulators/`, `benchmarks/accuracy.py` |
| **0d** | **take every baseline** — golden payload, rendered-fact list, screenshots | `tests/fixtures/` |
| 1 | tokens, typography, drop decorative shadows and gradients; **split `--rule-strong` from `--axis`** | `_00-tokens.css`, `_01-base.css` |
| 2 | contrast test, **extended pairs, both themes** | `tests/test_contrast.py` |
| 3 | header: mark asset, 52px bar, icon buttons, labelled metadata | `report_template.html`, `_02-header.css`, `render/html.py`, `static/images/` |
| 4 | summary: stat row, stacked composition bar (**2px gutter**), missing rows, quick facts | `render/sections.py`, `render/donut_chart.py`, `_03-summary.css`, `_04-donut.css` |
| 5 | summary: description as margin note | `report_template.html`, `_03-summary.css` |
| 6 | sample: borderless table, `nan` as dash, overflow notice | `render/sections.py`, `_05-sample.css` |
| 7 | sample: frozen index on mobile | `render/sections.py`, `_05-sample.css` |
| 8 | numeric card: restacked layout | `numeric_card.py`, `card_base.py`, `_06-cards.css` |
| 9 | histogram: units on axes, drop in-chart title, unitless branch | `histogram_svg.py`, `_07-histogram.css` |
| 10 | categorical + high-cardinality branch | `categorical_card.py`, `_08-categorical.css` |
| 11 | boolean + datetime cards | `boolean_card.py`, `datetime_card.py`, `temporal_charts.py`, `_09-datetime.css`, `_10-boolean.css` |
| **11b** | **partial report: one atomic file, partial banner, drop rotation** | `checkpoint.py`, `engine.py`, `config.py` |
| | ▲ **cut line for launch** — everything above is what a visitor sees | |
| ~~12~~ | ~~data fixes~~ — *moved to 0a–0c*; delete the dead `.stat-badges` renderer here | `card_base.py`, `_06-cards.css` |
| 13 | correlations: routing, empty state, diverging list, triangle matrix, no emoji | `correlations_section.py`, `_11-correlations.css` |
| 14 | missing values: drop tabs, route on chunk count | `missing_section.py`, `missing_columns.py`, `_12-missing.css` |
| 15 | remove the compatibility shim — **shrink it per phase, not all here** | `_00-tokens.css` |
| 16 | dark mode contrast pass — **confirmation, since commit 2 already covers dark** | `_00-tokens.css` |
| 17 | accessibility pass: targets, rails, contrast, greyscale | across |
| **18+** | **resume** (phase 10) — after launch, with the oracle's third path | `accumulators/*`, `checkpoint.py`, `benchmarks/accuracy.py` |

**The cut line matters.** Commits 0–11b are the header, summary, sample and the
five card types: everything a visitor sees in a screenshot. Correlations, missing
values and the comparison view sit below the fold and can land after publishing.

## Tests to update

The full strategy — what this migration can break, and the harness that catches
it — is in [`docs/MIGRATION_TESTING.md`](MIGRATION_TESTING.md). Two pieces of it
carry most of the weight and are worth naming here, because they are what make a
seventeen-commit diff reviewable at all:

- **A golden `summarize()` payload.** Sixteen of the eighteen commits must leave it
  byte-identical. The two that may not are 0a and 13.
- **A rendered-fact list.** 117 of the 125 numeric statistics in `summarize()`
  appear somewhere in today's HTML; that set must not shrink. Commit 8 — two
  fourteen-row tables becoming a four-cell stat row — is where one gets dropped,
  and no snapshot diff would show it.

Beyond those:

- **`tests/test_contrast.py`** — new, provided. Add it early (commit 2); it is what
  catches a palette regression before review does.
- `test_css_integrity.py` — token names change. Add an assertion that no legacy hex
  appears in `static/css/`.
- `test_html_templating.py` — metadata chips become a metadata line; the description
  container changes.
- `test_profile_html_snapshot.py` — regenerate **once per phase**, not once at the end,
  so each diff stays reviewable.
- `test_donut_chart.py` — the donut becomes a stacked bar. Rename and rewrite for widths
  summing to 100 and zero-count types emitting no segment.
- `test_correlation.py` — add cases for the empty state text, the below-threshold cap,
  and triangle cell count.
- `test_missing_columns.py`, `test_missing_context.py`, `test_sections.py` — markup
  assertions.

## Still open

0. **`--data-4` at 1.83:1 on the paper** needs a palette decision before commit 13,
   since it is the fill for below-threshold correlation bars *(amendment)*.

1. **Dark mode values are proposed, not verified** (`tokens.css`, and phase 16).
2. **Correlation values above 0.5 have never been seen in a real report here** — both
   example reports return none, so 6.2 and 6.3 have been designed against illustrative
   numbers. Generate a report from a dataset with genuine correlations and check the two
   views against it before commit 13.
3. Whether the comparison is asymmetric (baseline vs current) or symmetric (train vs
   test). It no longer changes the colour, but it changes whether one series should be
   visually subordinate.
4. Whether `Age` should carry units in the sample table, or only in the variable card.
