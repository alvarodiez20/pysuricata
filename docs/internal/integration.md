# pysuricata report UI — integration plan

Design decisions settled with the designer, written for implementation in the
`pysuricata` repo. Each phase is independently shippable and independently
reviewable. Work them in order — phase 1 establishes tokens every later phase
depends on.

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
| `Details.dc.html` | the details panes for all four types, plus an audit of what landed. Newest turn at the top |
| `Histogram.dc.html` | histogram geometry: the width problem, tick density, the gutter, captions, bins on a phone |
| `Datetime Card.dc.html` | the datetime card face — timezone, the two unjudgeable ratios, the line-vs-bars question |
| `Categorical Card.dc.html` | the categorical card face — one face asked to serve four kinds of column |
| `Variables Section.dc.html` | everything between the Variables heading and the first card, plus the pagination defect |
| `Variables.dc.html` | superseded exploration of three variables layouts. Kept for the reasoning; 4b won |

Every figure in the designs comes from your own output: the Titanic report for
everything except the datetime card, which uses `daily_2026-08-16`. Two
exceptions are labelled in the files themselves — the correlation values above
0.5 (Titanic has none) and the second series in the comparison charts.

---

## Status, as of the last audit

Read against the source, not the example reports — `examples/titanic_report.html` and
`reports/daily_2026-08-16.html` both predate the numeric restack, so they disagree with
the code. **Regenerate them**; an audit should be able to read the report.

| Phase | State |
| --- | --- |
| 1 · tokens, typography, micro-label | **landed.** Verbatim, plus an `--axis` token split out of `--rule-strong` — a better call than the original spec, since a hairline and a chart axis want different minimums |
| 5.1 · numeric card restack | **landed** (`numeric_card.py` emits `vstat-row`) |
| 5.1 · categorical, datetime, boolean cards | **not applied.** `categorical_card.py:672`, `datetime_card.py:842`, `boolean_card.py:539` still emit `.triple-row`, so three of four types keep the squeezed chart |
| 5.7 · flag chips show their value | **not applied.** Still renders `>Missing</li>` with `data-value="19.9%"` beside it. The test exists; the renderer does not use it |
| 6.1 · correlations empty state | **not applied** (`correlations_section.py:375`) |
| 6.4 · remove emoji | **not applied.** `📊` in correlations, `📈` `📉` in the outlier pane headers, `💡` and `ℹ️` in its notes |
| 7 · missing values, drop the tabs | **not applied.** Still `.missing-tabs` |
| — · details behind a toggle | **new, not designed.** Every card now has a Details button collapsing the whole pane. Reasonable, but the tab set is invisible until clicked — see 5b.8 |
| 5c.4, 5c.5 · datetime details | **landed.** `_interval_sentence` and `_build_temporal_distributions` are both in, and the sentence is better written than the design specified |
| 5c.2 · the `NaN` length bug | **landed**, and the diagnosis was better than mine: `avg_len` and `len_p90` were being read off the wrong object, so **every** categorical column printed `NaN`, not just `Embarked`. `_unknown_cell` — a dash that carries its reason in a `title` — is a better pattern than the bare dash I asked for |
| 5c.3 · the one-option Top-N chooser | **landed.** `Sex` was shipping three buttons all reading `2` |
| 5d.1, 5d.7 · histogram architecture | **landed** in both `histogram_svg.py` and `datetime_card.py`, including `preserveAspectRatio="none"` and `vector-effect="non-scaling-stroke"` throughout |
| 6.2, 6.3 · correlations list and matrix | **landed** — diverging bar, lower triangle only |
| 7 · missing values routing | **landed.** Routes on chunk count; the legend lives once. Two dead methods left behind — see the code findings |
| 1 · tokens | **landed with one violation in shipped code** — see the code findings |

---

## Phase 1 — Tokens and typography

Files: `static/css/_00-tokens.css`, `_01-base.css`

Replace the token layer with `assets/tokens.css`. It carries the full palette,
the measured contrast ratios, the dark-mode proposal, and a compatibility shim
mapping the legacy variable names onto the new scale so nothing breaks mid-way.
**Delete the shim once phase 12 lands** — it is scaffolding.

Two scales, and they never mix: **blue means data, warm means data quality.**

`tokens.css` carries three rules a contrast test cannot check, because they are
about which surface a mark lands on, what order marks paint in, and what a zero
value draws. Each was a real defect caught in review, twice in one case:

1. **Only `--data-1`, `--data-2` and `--data-3` may sit on the paper or on an empty
   `--track`.** `--data-4` is stack-internal only. It reads as a reasonable "quiet"
   choice for a subordinate chart and at 1.83:1 it is a ghost — if a chart should be
   subordinate, demote it one step and promote the mark above it.
2. **A quality-coloured mark crossing a data fill must protrude onto the paper or
   paint underneath it.** `--q-bad` on `--data-2` is 1.08:1; no warm colour is visible
   on a blue fill. A reference line goes *before* the bars in DOM order so they occlude
   it, and carries its value as a tick in the axis gutter. A marker inside a band must
   protrude past the edge and differ from its neighbour by **shape**, not colour. A
   legend swatch must be the colour actually drawn.
3. **A zero count draws nothing.** `v === 0 ? 0 : Math.max(1, …)`. A 1px floor is right
   for a small non-zero value and wrong for zero — the month chart is where it matters,
   since ten empty months drawn as ten 1px bars assert data that is not there.

**`--data-3` changes value**, from `#7FA0B5` to `#5C7F99`. At 2.63:1 on the paper the
old value could not legally carry a standalone mark, which is the job a third step
exists for. `#5C7F99` clears 3:1 on both surfaces in both themes, and is too mid-tone
to carry text either way — so it is a fill, never a label background.

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
severity. `Cabin` at 77% should look worse than `Embarked` at 0.2%. Thresholds are in `tokens.css`, and the break is at **20%**, not 50 —
`missing_columns.py` already classifies at `<=5 / <=20 / >20` and the missing section
prints that split in a visible legend, so the code is self-consistent and the first draft
of the token file was the outlier.

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

**Acceptance:** `test_css_integrity.py` and the new `test_contrast.py` pass; none of
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

## Phase 4b — The variables section, above the cards

Files: `render/triage.py`, `render/html.py`, `static/js/pagination.js`, `_13-utilities.css`

Design: `Variables Section.dc.html`, options 15a–15d.

Between the `Variables` heading and the first card there are four things: the
needs-attention block, a count sentence, a search-and-filter row, and a page control. The
attention block is the best-designed piece in the report; its **chips** are the problem.

### 4b.1 The `display: none` pagination defect — fix this first

`pagination.js:163` hides off-page cards with `card.style.display = 'none'`. On a 60-column
frame that removes 50 cards from the document, and four things break at once:

| Breaks | Why it matters |
| --- | --- |
| **Ctrl+F** | A browser find cannot match text in a `display: none` subtree, so searching for a column name silently fails for 50 of 60 columns — the primary action in a profiling report |
| **Anchor links** | The attention block and the summary missing list both link to `#col_<name>`. If the target is off-page the jump lands nowhere and the reader sees no change |
| **Print / PDF** | Only the current page prints. A 60-column profile exports as 10 columns with nothing saying so — the page control does not print either |
| **Deep links** | A URL ending `#col_Fare` opens on page 1 regardless, so a shared link does not land on the column it names |

The print case is the worst because it is **silent**: the reader has no way to tell that
fifty columns are missing.

**Fix: collapse, do not hide** (15d). Every column keeps a row in the document; beyond the
first ten the row is the card header — `+`, name, type, flags — at 44px, and the body folds.
Find works, anchors land, print expands everything. Fifty collapsed rows is a readable index
of the rest of the frame, which a page number never was.

Cost to state honestly: fifty rows is ~2,200px of scroll where the page control was 40px. It
changes nothing about file size — the charts were always in the file.

### 4b.2 Make the chips mean something (15a chips + 15b reference)

The block currently renders, for `Fare`:

```
Fare    33.20 heavy-tailed 13.0% many outliers
```

Six problems in one row:

1. **Two chips read as one string.** They are adjacent `<li>` with no border, background or
   separator.
2. **The threshold is in a `title`.** `annotate_flags` puts the value on the face and the
   limit in a tooltip — invisible on a phone, absent from a PDF. So `33.20` has nothing to be
   judged against.
3. **`33.20` of what.** The value carries no unit and the label names a conclusion rather
   than a measure. Nobody can tell it is kurtosis, or that normal is near 0.
4. **"Need a look" for what.** Two of the five flagged columns are identifiers and want
   *excluding*, not looking at.
5. **The column name sits below its chips**, on a different baseline, so the eye reads the
   flags then travels back to find the column.
6. **An instruction, not an affordance** — "Click a column to jump to its card" tells rather
   than shows, and is untrue in print.

**Do:** border the chips and put the threshold on the face —
`33.20 kurtosis · limit 10`, `77.1% missing · limit 20%`. That is a change to
`annotate_flags` plus CSS, and it closes the meaning gap without the profiler giving advice.

**Then:** a flag reference rendered from the flags the report actually raised (15b) — flag,
what is measured, what fires it, what it means. Four rows on Titanic, nothing on a clean
frame. It is the tooltip content, printed, plus the sentence the tooltip never had.

**Hold:** 15a's *grouping by action* ("drop or impute", "exclude — identifiers, not
variables", "transform before use") until you decide whether pysuricata should recommend
actions at all. It is the highest-value option in that file and the only one making a claim
beyond the data — "drop before modelling" is wrong for a reader who is not modelling.

### 4b.3 The 20% cliff — decide this regardless

`Age` is missing **19.9%** of its values and **does not appear in the attention block at
all**; its row lists only outliers. `actionable_chips` keeps `bad` plus selected `warn`
slugs, and the missing chip only reaches `bad` above 20%. A fifth of the passengers have no
age — arguably the most consequential fact in the dataset — and it misses the cut by one
tenth of a point.

Either the missing chip is actionable at `warn` too, or the block grows a quieter tier for
values just under a limit. A threshold artefact should not read as a judgement.

### 4b.4 The toolbar says what it is showing (15c)

- **Drop the count sentence.** "Analyzing 12 variables (3 numeric, 8 categorical, 1 datetime,
  0 boolean)" duplicates the composition bar in Summary, and prints `0 boolean` for a type
  with no columns — the same zero-segment problem as phase 3.2.
- **A tab per type that exists, with its count**: `All 12 · Numeric 3 · Categorical 8 ·
  Boolean 1`. Titanic has no datetime columns and currently gets a Datetime tab that filters
  to an empty grid with no explanation. Same rule as the zero-width donut segment and the
  one-option Top-N chooser.
- **One result line covering all three mechanisms**: `3 numeric columns · 10 expanded, 2
  collapsed`, with `clear filter` beside it. `Showing 1-10 of 12` cannot describe search +
  filter + page.
- **Sort**: dataset order (default), most missing, most flagged, name. The attention block
  already ranks worst-first internally; this exposes it for the full list. Keep dataset order
  the default — it is what someone reading alongside their dataframe expects.

**Acceptance:** no card is ever `display: none`; `#col_<name>` resolves for every column;
printing a 60-column report yields 60 cards; every chip shows its threshold without a hover;
no tab for a type with zero columns; `Age` at 19.9% missing appears in the attention block.

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

## Phase 5b — Details panes: numeric

Files: `render/numeric_card.py`, `render/card_base.py`, `_06-cards.css`

Design: `Details.dc.html`, options 9a–9e.

The details pane is where every number that did not fit the card went. For `Age` that is
26 key–value rows across two tables, with nothing in the layout saying which to read:
`Jarque–Bera χ²` carries the same weight as `Median`, and `Std Dev` is printed twice,
once in each table. Three problems run through it, and each fix costs no new statistics.

### 5b.1 Statistics — quantile strip aligned to the histogram (9a)

The nine percentiles are a shape. Printed as a column of numbers directly under a
histogram drawing that shape, they cannot be read against it. Put them on the same axis:

- The card histogram repeats at `--data-2`, subordinate to the strip.
- Below it, a box on the same scale: IQR band `--data-1`, whiskers `--data-2` **split
  into two spans terminating at the band edges** (one span painting across the band is
  wrong, and it is what a single P1–P99 line does).
- Median as a full-height rule in `--ink` **protruding past both band edges**; mean as a
  `--q-bad` caret sitting entirely **above** the band, on the paper. They land ~24px apart
  inside a dark fill, so they are differentiated by shape — rule 2 in `tokens.css`.
- A percentile ladder underneath, ticks at true position, labels on **two alternating rows**
  so P1 and P5 do not collide. **Drop rule:** if two ticks fall within ~4% of each other,
  print only the outer one and leave the rest to the table. On `Fare`, P1 through Q3 all
  land in the first fifth of the axis.
- Two prose lines spending thresholds you already hold: `data-threshold="JB χ² < 5.99"` is
  in the DOM today and never shown, so a reader is handed 18.63 with no way to judge it.

Fall back to **9b** — the same two `kv` tables regrouped by what they answer, plus one bar
width per percentile row — if a second chart per numeric column is not worth maintaining.
It is a real improvement on its own at about a fifth of the work, and nothing collides
whatever the distribution.

### 5b.2 Outliers — draw the fence (9c)

The worst pane in the report. It opens with a block announcing `Low Outliers — 0 outliers
(0.0%)` and three severity chips all reading zero, then the same again for the high side, a
`rowspan` table, and two notes led by `💡` and `ℹ️`. Roughly 60px says nothing happened,
and the values are listed with no picture of what they crossed.

- Draw the IQR fence and mark the points beyond it. An outlier is *defined* by a threshold,
  so the threshold is the one graphic that explains the number.
- **Replace the empty low block with a sentence.** `Age`'s lower fence sits at **−6.7**
  years — no age can be below it, so this column cannot have a low outlier. Branch four
  ways: no low fence possible, low outliers present, none present but possible, none at all.
- Flatten the `rowspan` table to one row per value with both verdicts side by side, and
  state the disagreement in prose: IQR flags seven, MAD flags one, they ask different
  questions.
- Points overlap when several outliers share a value — needs jitter or a count label.

### 5b.3 Common values (9d)

Five columns become three. The ordinals `1ˢᵗ 2ⁿᵈ 3ʳᵈ` are decoration on an ordered list,
and count plus percent belong together.

**Scale the bar to the top value, not to 100%.** At 3.2% of 714 rows the current bars are
3% of their track and all ten look identical. State the scaling in the caption, since
relative scaling hides absolute rarity.

Then say the finding: all ten most common `Age` values are whole numbers though the column
stores three decimals, and `Heaping %` is 22.27 — two numbers you already compute and
never put next to each other.

### 5b.4 A tab has to earn itself (9e)

Render a tab only when it has something to say, and never when it repeats the card face.
Keep the **order fixed** so a tab never moves, only appears or not.

| Pane | Today | Do |
| --- | --- | --- |
| Numeric · Correlations | the section-level empty state, repeated inside a card | name the strongest pair and its value — `Age`'s is 0.096 against `Fare` |
| Every type · Missing Values | rendered at 0%, as a 100%-present bar and a one-segment chunk strip reading 0.0% | render only when the column has missing values |
| Datetime · Statistics | seven rows repeat the card's own tables word for word | keep only the four peak lines and the two ratios; rename it **Patterns** |
| Boolean · Breakdown | a two-row table under a card already showing the same split | drop it (see 5c.4) |

### 5b.5 Min/Max — draw the fence, cluster the marks (12a)

Two tables of index and value, ten numbers, no context. A reader cannot see that **every
one of the five maxima is an outlier and not one of the five minima is** — the whole story
of this column's tails, and already computable from the fence.

- Same value axis as 5b.1 and 5b.2, so a reader who has opened one pane can read this one.
- A **position** column per row (`moderate`, `high · 2.0× IQR`, `below P1`) replacing the
  bare index/value pair. Severity words and colours must match the Outliers pane exactly —
  a value that is `moderate` there cannot be `high` here.
- **Mark ties.** `Age` has 0.75 twice and 71 twice; the current pane lists them as separate
  rows without comment.
- Rule 2(e): all five low values fall inside 0.41 years, narrower than one mark, so they
  collapse to one capsule labelled `×5` with the row ids in its `title`.

Open question for the designer: on `Age` this pane and Outliers plot the same points,
because its five highest values all happen to be outliers. On a column with no outliers
they are entirely different panes. Folding them into one **Tails** tab removes the overlap
and costs the plain "what are the biggest values" question, which is asked more often.

### 5b.6 Correlations — show every partner (12b)

The per-column pane repeats the section-level empty state inside a card. But `Age` has
exactly two numeric partners in this frame, so listing both is *complete* information in two
rows — nothing is withheld. "Both partners are weak, the stronger is Fare at +0.096" is a
finding; "no significant correlations" is a shrug.

Same diverging bar as phase 6.2, so sign stays position and never colour. **Cap at 5** with
a `35 more below 0.50` line, or a 40-column frame renders 39 rows. Needs the sub-threshold
pairs kept per column, which `_collect_correlations` filters away before returning.

### 5b.7 Missing — one bar, and only when chunks exist (12c)

The pane states one fact four times: a `Present` stat, a `Missing` stat, a two-segment bar,
and a one-segment chunk spectrum — under a header already flagging `19.9% missing`. Then a
three-item legend, on every card with any missing at all.

Tighten 5b.4's rule: render the tab when **missing > 0 AND chunks > 1**. That is the only
condition under which this pane knows something the card face does not — where in the read
the gaps fall. On Titanic every numeric card loses a tab and nothing goes with it.

When it does render: one chunk strip, severity per chunk on the warm scale, and a spread
line (`18.4 – 21.5% per chunk`, `Spread 3.1 pp`) so "steady" is a number rather than an
impression. Pick the threshold for calling it steady; nobody has.

### 5b.8 The strip — say what is behind the button (12d)

A `Details` button toggles `hidden` on the whole section, and inside it six tabs. Two levels
of disclosure, and the word "Details" promises nothing — so a reader opens every card to
learn whether opening was worth it. **The tab set is known at render time. Print it.**

```
Details ▾   statistics · 10 common values · 5 lowest and highest · 11 outliers · 2 correlations
```

- `11 outliers` beside the button is the reason to open; its absence is the reason not to.
- Open, each tab carries its count, so a reader picks the right one first time.
- Tabs never reorder — they appear or they do not.
- Mobile: the rail uses `min-height`, not `height`, with a right-edge mask.
- **The active-tab underline goes on an inner span wrapping the label**, not on the 44px tap
  box, or the rule paints ~29px below the text and reads as a doubled hairline.

**Acceptance:** no pane repeats a value shown on its card face; a column with 0 missing has
no Missing tab; a column with 1 chunk has no Missing tab; the low-outlier sentence is
generated for all four cases; severity words agree between Min/Max and Outliers; no emoji in
`numeric_card.py` or its panes; the closed `Details` row names its panes.

---

## Phase 5c — Details panes: categorical, datetime, boolean

Files: `render/categorical_card.py`, `datetime_card.py`, `boolean_card.py`,
`accumulators/categorical.py`, `accumulators/datetime.py`, `_08-categorical.css`,
`_09-datetime.css`, `_10-boolean.css`

Design: `Details.dc.html`, options 10a–10f. Each notes whether the data already exists.

### 5c.1 Normalization: report collisions, not transformations (10a)

The pane prints original / `lower()` / `strip()`, so for `Embarked` it says `S → s → S`.
That teaches nothing. The question it exists to answer is whether normalising would
**merge** levels — the difference between three categories and two.

Report a verdict (`3 levels stay 3 under lower() and 3 under strip(). Nothing merges.`)
and list the merging groups only when there are any. Free: it is a
`len(set(map(str.lower, levels)))` over data you already hold. Hedge honestly — only the
top-K levels are tracked, so the verdict is "no collisions among the 10 tracked levels".

**Bug this surfaces:** `Embarked` carries `Case variants` and `Trim variants` flags while
the pane finds no collisions to justify either. One of the two is wrong.

### 5c.2 Spend the length reservoir (10b)

`categorical.py` already keeps a 5,000-value `ReservoirSampler` of label lengths
(`self._len_sample`) and the report spends it on two numbers, `avg_len` and `len_p90`. The
whole distribution is sitting there.

Draw it as a small histogram plus the longest and shortest value. On an identifier column
the shape is the finding: `Ticket` clusters at 4–7 characters and 10–14 with a tail to 18,
which is two ticket formats in one column — a cleaning finding available no other way.

Two requirements: the reservoir must **expose its sample** (currently private), and add a
**suppression rule** — draw the chart only when more than two distinct lengths appear, and
print the single length as a sentence otherwise, or `Embarked` renders one bar at 1.

**Bug to fix here:** `Embarked` prints `Label length (avg)` as **`NaN`** and `Length p90`
as an em dash, for a column whose three labels are all one character long.

### 5c.3 High-cardinality columns need a details pane too (10c)

Phase 5.4 replaced the meaningless top-values *chart* on the card. The details pane still
opens on `Common values` — the same ten bars of one row each, 0.1% apiece. On the card the
fix was to say there is nothing to plot; in the pane there **is** something to plot, just
not that.

- Drop `Common values` for these columns; there is no ranking, and rendering one implies
  a frequency that does not exist.
- Show the shape of the values: distinct against row count, length range, longest and
  shortest, empty strings. Length and both extremes are already computed.
- A sample of ten arbitrary values tells a reader more about an identifier column than ten
  equally-rare "most common" ones. This is **new state** — a small reservoir per
  high-cardinality column — and putting raw values in a shared HTML file is a privacy
  question the sample table already faces. Decide it deliberately rather than inheriting it.

### 5c.4 Datetime: label the axis, drop the empty chart (10d)

Four small multiples with an `<h4>` each and no y axis on any of them, so a 211-record hour
and a 2,626-record month draw identically — and the peaks that would resolve it live in a
different tab.

- Give each chart its own zero-based y axis and print its peak inline in the header.
- `by_year` is a `dict`, so a dataset inside one year renders **a single bar at full
  height** — a chart whose only reading is "all of it". State the span in a line instead.
- Keep the month chart's **12 fixed slots**. `temporal_charts.py` already gets this right
  and must not lose it: two populated months drawn as two half-width slabs reads as
  "spread evenly across the timeline" instead of "2 of 12".
- Per-chart scales mean the three cannot be compared to each other by height. A shared
  scale would fix that and flatten the hour chart to nothing — a real trade, taken on the
  readable side. Say so in the caption.

### 5c.5 Datetime: say the series is regular (10e)

The strongest fact about `signed_up` is a table row reading `Interval std dev — 0.0
seconds`, filed alphabetically between timezone and weekend ratio. A standard deviation of
zero means every gap is identical: a record every 17 minutes, no gaps, machine-generated.

Lead the pane with that sentence. On a real event stream the same line reads "median 4
minutes, longest gap 3.2 days on 14 Feb", which is what anyone opens a datetime column
to ask.

**The one proposal needing new state.** `_calculate_interval_stats` builds an interval
array and returns two floats from it — a max and a p50 come free from the same array. "Longest
gap and when" needs a timestamp kept alongside; the gap strip needs per-bucket counts you
may not want to store. Take the free half first.

### 5c.6 Boolean: no details pane (10f)

Dropping `Breakdown` leaves boolean with no details section at all. That is correct — two
values, two counts, one bar, nothing withheld. Worth a comment in `boolean_card.py` so it
reads as a decision rather than an omission.

The one thing a boolean pane could add that the card cannot is **true rate by chunk**: a
flag that is 12% early and 60% late is a pipeline change, and a single 38.4% hides it. Two
counts per chunk is the cheapest new state in this document and would serve the categorical
mode share too. But chunks are an artifact of how the file was read — reorder the input and
the chart changes — so it needs a caveat line, and I would not build it before someone asks.

**Acceptance:** no normalization row per level when nothing merges; length chart suppressed
below three distinct lengths; `avg_len` never `NaN`; no year chart when the span is inside
one year; month chart has 12 slots and **zero counts draw nothing** (rule 3); boolean has no
details section.

---

## Phase 5d — Histogram geometry

Files: `render/histogram_svg.py`, `static/css/_07-histogram.css`

Design: `Histogram.dc.html`, options 13a–13g. **Supersedes the axis half of phase 5.2** —
build 5.2's units and tick logic on this geometry, not on the 420×200 canvas.

### The measured problem

The card was restacked so the chart could be full width. The chart did not get it. At a
1240px viewport the `<svg>` element is 1,099px and the bars occupy **356px** — 68% of the
element is blank, because `preserveAspectRatio` defaults to `xMidYMid meet`, the container
is `height: 210px`, and height is therefore the limiting dimension.

The trap, stated plainly:

> Uniform scale ⇒ text size varies with viewport.
> Fixed text size ⇒ the canvas must be ~1:1 with its display size.
> One static SVG cannot be 1:1 at both 1,099px and 284px.

### 5d.1 Split the coordinate systems

**Rule 4 in `tokens.css`.** Bars, gridlines and axis lines in an SVG with
`preserveAspectRatio="none"` and `vector-effect="non-scaling-stroke"`; every label, caption,
unit and tooltip in HTML at **percentage offsets**. Nothing else in this phase works without
it, and it takes all text out of the SVGs — 23% of report bytes today, mostly label markup.

### 5d.2 Composition — cap the plot (13a)

Width past ~800px buys almost nothing: at 800px, 50 bins is 16px each; at 1,100px, 22px.
Nobody reads a 22px bar more accurately.

- Plot area `max-width: 820px`, left-aligned. Ratio settles near 4:1 instead of 5.2:1.
- The freed ~386px takes the **scale and bin controls**, beside what they change rather than
  below it. Card height drops ~40px per numeric column — 1,600px of scroll on a 40-column frame.
- Second breakpoint at 1180px, where the controls return under the chart.

Alternative **13b**: full bleed with `height: clamp(170px, 21vw, 290px)`. Cheaper (one
`clamp`, no breakpoint, no second column), answers the filed issue literally, and costs
60–80px of height per numeric column on the axis a profiling report is already long in.

### 5d.3 Tick density, decided by CSS (13c)

The renderer cannot know the viewport, so **write every tick you would ever want and tag it
by importance.** Nine x-ticks, alternates `data-tier="2"`, the next drop `data-tier="3"`:

```css
@media (max-width: 760px) { .hist-x [data-tier="2"] { display: none } }  /* → 5 */
@media (max-width: 440px) { .hist-x [data-tier="3"] { display: none } }  /* → 3 */
```

No variants, no JS; the cost is four short strings per column. Tiering by **importance**
rather than index means the first and last tick — the range — are always tier 1 and never
drop. Container queries are the exact tool and fix the case where a wide viewport has a
narrow card; use them if a 2023+ browser is acceptable.

### 5d.4 The gutter — 44px, because counts abbreviate (13d)

A y tick is a row count and nobody reads seven significant figures off an axis.
`1,234,567` must render `1.2M`. **Cap the label at four glyphs** and the gutter is a
constant: 27px of 11px mono + 5px tick + 8px air = `--hist-gutter: 44px`. Fixed at every
width, so the plot's left edge never moves between columns and bars line up down the page.

`_format_tick_label_standardized(v, is_count=True)` must **guarantee** four glyphs, not
prefer them — it can currently emit `12.5K`, which is five. The exact peak stays in the
caption, so abbreviating loses nothing.

### 5d.5 Where the captions live (13e)

`ROWS` stays in the gutter — anchored 44px from its axis at any width. The x unit does
**not** stay at the right end of the axis: at 1,100px the two captions are a metre apart and
stop reading as a pair. Move it into one caption under the axis, left-aligned to the plot
origin, carrying the bin count and the peak:

```
age in years · 25 bins · peak 83 rows at 26–29
```

`derive_x_unit` returning `None` must read gracefully — `25 bins · peak 83`, no unit clause.

### 5d.6 Bins on a phone (13f)

At 284px, 50 bins is ~5.7px each. Hide the option under 560px and **print the reason**
(`50 needs a wider screen`) in the space the two remaining buttons are not using — otherwise
the mobile and desktop reports look like different products. Saves no bytes: the variant is
still in the file. Both remaining targets stay 44×44.

### 5d.7 The bar gap will vanish (13g)

**Rule 5 in `tokens.css`, and this one is not optional.** `_render_bars` sets
`bar_w = max(1, bar_width - 1)` — a 1-unit gap in viewBox space, which under
`preserveAspectRatio="none"` scales with x:

| Plot width | x scale | 1-unit gap becomes |
| --- | --- | --- |
| 1,100px | ×1.10 | 1.1px — fine |
| 560px | ×0.56 | 0.56px — thin |
| 284px | ×0.28 | 0.28px — bars merge |

Draw bars edge to edge and separate them with a `--paper` non-scaling stroke.

### 5d.8 Derive the axis max per chart

`render_histogram_from_bins` already does this right at lines 263–266 — `nice_ticks(0,
actual_max, 5)` per render — and it matters more once the plot fills its width. Changing the
bin count changes the peak: 25 bins peaks at 83 (axis 100, 83% fill), 50 bins peaks near 50
(axis 50, full fill). A shared max across variants would draw the 50-bin chart half empty,
which is the same defect as the letterbox. **Do not hoist the max out of the per-render path.**

### Bytes

The six variants per numeric column (10/25/50 × lin/log) share an identical **x range**, so
with text out of the SVG the x-tick HTML layer can be written **twice per column** instead of
six times. Only the y layer changes with bin count, because `y_max` does. At ~7.1 KB per
histogram SVG today, most of it label markup, this phase moves toward the 250 KB target
rather than against it.

**Acceptance:** no `<text>` in any histogram SVG; bars occupy ≥95% of the plot width at
1240px and at 390px; tick labels measure 11px at every viewport; bar separators measure 1px
at 1240px and at 390px; a 50-bin chart's tallest bar reaches its top gridline; no y label
exceeds four glyphs.

---

## Phase 5e — Datetime card face

Files: `render/datetime_card.py`, `accumulators/datetime.py`, `_09-datetime.css`

Design: `Datetime Card.dc.html`, options 14a–14c. The restack, the histogram architecture
and the interval sentence have all landed here — this phase is what is left, and none of it
is layout.

### 5e.1 The timezone is a literal — correctness, fix first

`_left_stats` emits `("Timezone", "UTC", None)`: a hardcoded string, never read from the
column. `DateTimeStats` **already carries `source_timezone`**, and the accumulator populates
it by parsing `datetime64[ns, US/Eastern]` out of the dtype string
(`accumulators/datetime.py:191–209`). A column stored in Eastern is labelled UTC on the
card, and `_format_timestamp` appends `UTC` to min and max as well.

Three wrong labels from one unused field. One line in `_left_stats`, one in
`_format_timestamp`. A naive column has no timezone and should say so rather than claim UTC.

### 5e.2 Two percentages nobody can judge (14a)

`Weekend % 27.0` and `Business hrs % 24.3` print bare. A flat calendar gives:

| Ratio | Flat baseline | Arithmetic |
| --- | --- | --- |
| Weekend share | **28.6%** | 2 of 7 days |
| Business hours | **23.8%** | 8 of 24 hours on 5 of 7 days |

So both are noise — and the renderer knows it: the flag threshold beside them carries the
comment *"expected ~28.5%"*. The baseline is in a code comment instead of on the card, which
is the Jarque–Bera problem exactly.

Draw each share as a bar with a **rule at the flat value**, and read the verdict off it:
`flat · −1.6pp vs 28.6%`. Both baselines are constants, so this costs no new statistics. The
rule stays at 390px even when its label goes — position against a mark is readable without a
caption.

### 5e.3 Thirteen statistics become eight (14a)

- **`Avg interval` and `Interval std` move to the pane.** `_interval_sentence` already
  interprets them, and it is better than they are. Promote the sentence to the card face,
  above the chart, where it is read before the conclusions.
- **`Min` and `Max` lose the `<br>`.** Two double-height cells in a 4-column grid make
  every row in the grid taller.
- **`Processed bytes (≈)` goes.** Engineering telemetry among data statistics.
- Keep: count, missing, unique (≈), time span, min, max, density, timezone.

Cost to state honestly: the lede is prose on a card face, so a frame with thirty datetime
columns gains thirty paragraphs, and on the irregular branch the sentence is two lines. The
baseline panel also takes ~300px that the chart could use, which conflicts with the 820px cap
in 5d.2 — below the second breakpoint it moves under the chart.

### 5e.4 A line asserts values the buckets do not hold (14b)

The timeline is a `<polyline>` through bucket centres, so the slope between "84 records on 8
Jan" and "83 on 9 Jan" is **drawn rather than measured**. The card's own temporal panes draw
the same quantity as bars, so one card carries two encodings for counts.

Proposal: bars while a bucket is at least 4px, which at the 820px cap is about 180 buckets —
the renderer's own `min(bins, 180)` ceiling. A line above that, where bars would be
sub-pixel. The existing hotspot rects become the bars rather than sitting invisibly on top of
them.

**This is the one genuine trade in the phase.** A line reads a trend better, and trend is
often what a datetime column is for; on 180 near-equal buckets the bars are a grey slab where
the line still shows drift. Two encodings behind one threshold is also a branch and a test.
Worth looking at both charts in `Datetime Card.dc.html` before deciding.

### 5e.5 The missing pane (14c)

Phase 5b.7's condition applies unchanged: render when `missing > 0` **and** `chunks > 1`.
When it does render, delete the per-card three-item legend (it lives once, in the Missing
values section), the title-case headings, and `Hover over segments to see chunk details` —
an instruction that is untrue on a phone and gone in a PDF. Lead with the finding a chunk
strip exists to reveal: `the last two chunks hold 71% of them`.

**Acceptance:** no hardcoded `"UTC"` anywhere in `datetime_card.py`; a naive column does not
claim a timezone; both ratios render with their flat baseline; eight stats on the face; no
per-card severity legend.

---

## Phase 5f — Categorical card face

Files: `render/categorical_card.py`, `_08-categorical.css`

Design: `Categorical Card.dc.html`, options 16a–16b. This is the most common column type —
eight of Titanic's twelve — and the one card asked to be four different things.

### The problem

| Column | Levels | What the 12 stats do with it | What it is |
| --- | --- | --- | --- |
| `Sex` | 2 | Entropy 0.936, rare levels 0, top-5 coverage 100% — three statistics describing the spread of a distribution with two members and no spread | a boolean in a string |
| `Embarked` | 3 | The one kind the twelve were written for. Even here, 72.4% mode share has nothing to be judged against | a true category |
| `Cabin` | 147 | A top-10 bar chart covering 3.4% of rows, under a column that is 77.1% empty | a sparse identifier |
| `Name` | 891 | Ten bars of one row each, 0.1% apiece. Entropy computed over values that never repeat | a primary key |

`Entropy`, `Rare levels` and `Top 5 coverage` describe how a distribution spreads across
levels. They are meaningful for **one** of those four kinds.

### 5f.1 Suppress what cannot be true (16b) — take this

One face, one code path. A statistic that cannot be true for this column is **not rendered**
— the row closes up rather than printing a dash. Same principle as `_unknown_cell`, one step
further: *absent* rather than *unknown*. `Sex` drops from twelve slots to six.

Suppression is per statistic, so there is no arbitrary level boundary to defend and no
argument about where a 12-level column belongs.

Cost: a varying slot count makes the stat row a different height per column, so the page
loses its rhythm — more subtly than three layouts would, but it does.

### 5f.2 The even-split rule (from 16a) — take this too

Beside each level bar, a rule at `100 / n_levels`. `Embarked`'s S at 72.4% against a 33.3%
mark says *dominated by one port* with no arithmetic asked of the reader, and it is the same
device as 5e.2's flat-calendar rule and 5b.1's fence. Nothing new is computed.

State coverage honestly under the list: `3 of 3 levels shown · covers 100% of non-missing
rows`, and for `Cabin`, `3 of 147 levels shown · covers 5.9% of the 204 non-missing rows`.

### 5f.3 The many-level chart (from 16a) — a chart decision, not a face decision

`Cabin`'s 147 levels are not a bar chart at any width: the top ten cover 16.7% of 204
non-missing rows in a column that is 77.1% empty, so the chart describes a twentieth of the
data. Report concentration instead — rows per level, singleton count, top-10 coverage — and
say what the chart cannot: `119 of 147 levels occur exactly once`.

This is phase 5.4's high-cardinality branch extended: 5.4 handled *levels ≈ rows*
(`Name`), and this handles *many levels, few rows each* (`Cabin`), which the current code
treats as an ordinary category.

### 5f.4 Routing (16a) — hold

Three faces routed on level count is the fuller answer and reads better per column, but it
means eight categorical columns present three layouts, the boundaries are arbitrary (a
12-level and a 22-level column are the same kind of thing), and it is three code paths where
there is one. 5f.1 plus 5f.2 get the two real wins without that. Revisit if the suppression
rule turns out to produce cards that still feel wrong.

**Acceptance:** no statistic rendered that cannot be true for its column; every level bar
carries its even-split rule; a column whose top-10 coverage is under ~20% gets concentration
figures rather than a bar chart; coverage is stated as a share of non-missing rows.

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
- **Drop the strength bands.** The first cut specified three tints. On a diverging bar
  the length already encodes `|r|` — it is `|r|` × half the track — so the bands restated
  what the bar was already saying, and two of the three then failed rule 1. Every bar is
  `--data-2`.
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

Both datasets stay **inside the blue scale**: `--data-1` for one, `--data-3` for the
other, side-by-side bars within each bin so neither hides the other. (The first cut
paired `--data-2` with `--data-4`; `--data-4` cannot sit on the paper — rule 1.) No second hue enters
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

## Code findings, independent of the phases

Three things found while reading the source for turns 14–16. None is a design change; each is
about ten minutes.

| Where | What | Kind |
| --- | --- | --- |
| `correlations_section.py:169` | A below-threshold pair's diverging bar is filled with `var(--data-4)`. That step is **1.83:1 on the paper** and `tokens.css` documents it as stack-internal only — rule 1. The bar is invisible in print and close to invisible on screen. Use `--data-3`. | **rule violation** |
| `correlations_section.py:364` | `_render_no_correlations_state` still prints a bare message in a div. Phase 6.1's enriched empty state landed on the path where pairs exist but not on this one, so a frame with fewer than two numeric columns still gets one line. | half-applied |
| `missing_section.py:245–395` | `_build_completeness_tab` and `_build_chunk_tab` are the old two-tab implementation, unreachable since the chunk-count routing replaced them. ~150 lines, carrying their own `chunk-legend` with hardcoded severity colours that no longer match the tokens. | dead code |

Take the `--data-4` one first: it violates a rule that is written down and shipped, and it
stays invisible until someone looks at a below-threshold pair on paper.

## Commit sequence

| # | Commit | Touches |
| --- | --- | --- |
| 1 | tokens, typography, drop decorative shadows and gradients | `_00-tokens.css`, `_01-base.css` |
| 2 | contrast test | `tests/test_contrast.py` |
| 3 | header: mark asset, 52px bar, icon buttons, labelled metadata | `report_template.html`, `_02-header.css`, `render/html.py`, `static/images/` |
| 4 | summary: stat row, stacked composition bar, missing rows, quick facts | `render/sections.py`, `render/donut_chart.py`, `_03-summary.css`, `_04-donut.css` |
| 5 | summary: description as margin note | `report_template.html`, `_03-summary.css` |
| 6 | sample: borderless table, `nan` as dash, overflow notice | `render/sections.py`, `_05-sample.css` |
| 7 | sample: frozen index on mobile | `render/sections.py`, `_05-sample.css` |
| 8 | numeric card: restacked layout | `numeric_card.py`, `card_base.py`, `_06-cards.css` |
| 9 | histogram: units on axes, drop in-chart title, unitless branch | `histogram_svg.py`, `_07-histogram.css` |
| 10 | categorical + high-cardinality branch | `categorical_card.py`, `_08-categorical.css` |
| 11 | boolean + datetime cards | `boolean_card.py`, `datetime_card.py`, `temporal_charts.py`, `_09-datetime.css`, `_10-boolean.css` |
| 12 | data fixes: clamp KMV distinct, flag thresholds, delete dead `.stat-badges` | `accumulators/`, `card_base.py`, `_06-cards.css` |
| 12a | details: numeric statistics pane — quantile strip (5b.1) | `numeric_card.py`, `_06-cards.css` |
| 12b | details: outliers pane — fence, low-side sentence, flat table, no emoji (5b.2) | `numeric_card.py`, `_06-cards.css` |
| 12c | details: common values, three columns, bar scaled to top (5b.3) | `card_base.py`, `_06-cards.css` |
| 12d | details: a tab renders only when it has something to say (5b.4) | `card_base.py`, all four card renderers |
| 12e | details: normalization reports collisions; fix the flag contradiction (5c.1) | `categorical_card.py`, `triage.py` |
| 12f | details: length distribution — expose the reservoir, fix `NaN` (5c.2) | `accumulators/categorical.py`, `categorical_card.py` |
| 12g | details: high-cardinality pane, length + extremes half only (5c.3) | `categorical_card.py`, `identifier.py` |
| 12h | details: temporal axes, drop the single-year chart, zero draws nothing (5c.4) | `temporal_charts.py`, `datetime_card.py` |
| 12i | details: regularity line, max and median interval (5c.5) | `accumulators/datetime.py`, `datetime_card.py` |
| 12j | details: drop the boolean pane (5c.6) | `boolean_card.py`, `_10-boolean.css` |
| 12k | details: min/max pane — fence, position column, ties, clustering (5b.5) | `numeric_card.py`, `_06-cards.css` |
| 12l | details: per-column correlations with every partner + cap (5b.6) | `numeric_card.py`, `correlations_section.py` |
| 12m | details: missing pane only when chunks > 1 (5b.7) | `numeric_card.py`, `card_base.py` |
| 12n | details: strip prints its panes; underline on an inner span (5b.8) | `card_base.py`, `_06-cards.css` |
| 12o | histogram: split SVG bars from HTML labels (5d.1) | `histogram_svg.py`, `_07-histogram.css` |
| 12p | histogram: cap the plot, controls beside it (5d.2) | `_07-histogram.css`, `numeric_card.py` |
| 12q | histogram: tiered ticks, 44px gutter, caption line (5d.3–5d.5) | `histogram_svg.py`, `_07-histogram.css` |
| 12r | histogram: non-scaling bar separator, hide 50 bins under 560px (5d.6–5d.7) | `histogram_svg.py`, `_07-histogram.css` |
| 13a | **code finding:** `--data-4` → `--data-3` on the below-threshold bar | `correlations_section.py` |
| 13b | **code finding:** enrich the remaining correlations empty state | `correlations_section.py` |
| 13c | **code finding:** delete the two dead missing-section methods | `missing_section.py` |
| 14a | datetime: read `source_timezone`; stop claiming UTC (5e.1) | `datetime_card.py` |
| 14b | datetime: flat baselines on the two ratios (5e.2) | `datetime_card.py`, `_09-datetime.css` |
| 14c | datetime: promote the interval sentence, 13 stats → 8 (5e.3) | `datetime_card.py` |
| 14d | datetime: bars under 180 buckets, line above (5e.4) | `datetime_card.py` |
| 14e | datetime: missing pane only when chunks > 1; drop the per-card legend (5e.5) | `datetime_card.py`, `_09-datetime.css` |
| 15a | categorical: suppress statistics that cannot be true (5f.1) | `categorical_card.py` |
| 15b | categorical: even-split rule on every level bar (5f.2) | `categorical_card.py`, `_08-categorical.css` |
| 15c | categorical: concentration figures for many-level columns (5f.3) | `categorical_card.py` |
| 16a | **variables: collapse instead of `display: none`** (4b.1) | `pagination.js`, `html.py`, `_13-utilities.css` |
| 16b | variables: chips carry their threshold on the face (4b.2) | `triage.py`, `_13-utilities.css` |
| 16c | variables: flag reference, rendered from flags raised (4b.2) | `triage.py` |
| 16d | variables: the 20% cliff — `Age` at 19.9% must appear (4b.3) | `triage.py` |
| 16e | variables: toolbar counts, no tab for absent types, one result line, sort (4b.4) | `html.py`, `functionality.js`, `_13-utilities.css` |
| 13 | correlations: routing, empty state, diverging list, triangle matrix, no emoji | `correlations_section.py`, `_11-correlations.css` |
| 14 | missing values: drop tabs, route on chunk count | `missing_section.py`, `missing_columns.py`, `_12-missing.css` |
| 15 | remove the compatibility shim from `tokens.css` | `_00-tokens.css` |
| 16 | dark mode contrast pass | `_00-tokens.css` |
| 17 | accessibility pass: targets, rails, contrast, greyscale | across |

## Tests to update

- **`tests/test_contrast.py`** — provided, and updated since the first handoff. Add it
  early (commit 2); it is what catches a palette regression before review does. It now also
  asserts the two roles that are easy to lose: `--data-3` carries no text, and `--data-4`
  is *expected to fail* against the paper because it is stack-internal only.
- The three rules at the bottom of `tokens.css` are **not** testable as token pairs — they
  are about surface, paint order and zero handling. A DOM-level check is possible and worth
  it for rule 3: assert no `<rect>` in a generated report has `height="1"` where its
  `data-count` is `0`.
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
- `test_histogram_svg.py` — assert no `<text>` element is emitted, that bar rects are
  edge-to-edge (`x[i+1] == x[i] + width[i]`), and that the axis max comes from the chart's
  own peak so a 50-bin chart's tallest bar reaches the top tick.
- A DOM-level check is worth having for rule 3: assert no `<rect>` in a generated report has
  `height="1"` where its `data-count` is `0`.
- **`test_pagination.py`** — assert no `.var-card` ends up with `display: none`, and that
  every `#col_<name>` anchor in the attention block resolves to an element in the document.
  That is the test 4b.1 exists for.
- `test_datetime_card.py` — assert the timezone cell reads `source_timezone` (parametrise a
  naive column, a UTC column and an `US/Eastern` column), and that no literal `"UTC"` string
  is emitted for a naive one.
- `test_categorical_card.py` — assert a 2-level column emits no entropy, rare-levels or
  top-5-coverage cell, and that a 147-level column emits concentration figures rather than a
  top-N chart.
- `test_triage.py` — assert every chip carries its threshold in its rendered text, not only
  in a `title`, and that a column at 19.9% missing appears in the attention block.
- `test_missing_columns.py`, `test_missing_context.py`, `test_sections.py` — markup
  assertions.

## Still open

1. **Dark mode values are proposed, not verified** (`tokens.css`, and phase 16) — though the
   dark data scale itself was measured and is in the file.
2. **The details toggle** added since the first handoff is not designed. It collapses the
   whole pane behind one button, so the tab set is invisible until clicked. Worth revisiting
   once phases 5b and 5c make the panes worth opening.
3. **Whether raw values may appear in a details pane** (5c.3). The sample table already puts
   real rows in the file, so the precedent exists, but a value reservoir per high-cardinality
   column should be a deliberate decision.
4. **Correlation values above 0.5 have never been seen in a real report here** — both
   example reports return none, so 6.2 and 6.3 have been designed against illustrative
   numbers. Generate a report from a dataset with genuine correlations and check the two
   views against it before commit 13.
5. Whether the comparison is asymmetric (baseline vs current) or symmetric (train vs
   test). It no longer changes the colour, but it changes whether one series should be
   visually subordinate.
6. Whether `Age` should carry units in the sample table, or only in the variable card.
7. **Should pysuricata recommend actions?** 15a groups flagged columns by what to do — "drop
   or impute", "exclude — identifiers, not variables", "transform before use". It is the
   highest-value option in the whole set and the only one that makes a claim beyond the data.
   Everything else in this plan reports facts; that one gives advice. Decide it deliberately.
8. **Bars or a line for the datetime timeline** (5e.4). The only genuine visual trade left.
9. **Never designed at all**, in the order I would take them: dark mode (values proposed, never
   looked at in situ), print/PDF as a whole (interacts with 4b.1), degenerate frames (one
   column, zero rows, all one type), and the section-header system — Summary uses a bare
   `<h2>` where the other four use `.section-title`.
