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
| 1 · tokens, typography, micro-label | **landed.** Verbatim, plus an `--axis` token split out of `--rule-strong` |
| 5.1 · numeric card restack | **landed** (`numeric_card.py` emits `vstat-row`) |
| 5.1 · categorical, datetime, boolean cards | **still open** — `categorical_card.py:672`, `datetime_card.py:842`, `boolean_card.py:539` still emit `.triple-row`. Tracked as #158 |
| 5.7 · flag chips show their value | **landed** since this audit (#137). Chips render `20.0% missing`, with the threshold in a `title` |
| 6.1 · correlations empty state | **landed** since this audit (#138) |
| 6.4 · remove emoji | **partly landed.** Gone from `correlations_section.py` (#138). `💡` and `ℹ️` remain in the numeric outlier and context notes — #157 |
| 7 · missing values, drop the tabs | **landed** since this audit (#140). The by-chunk half is blocked on #139 |
| — · details behind a toggle | **kept.** Still worth revisiting once the panes are worth opening — phases 5b (#154) and 5c (#155) |

> **Note on this table.** The handoff audited a snapshot from before phases 2–7 landed, so
> four of its rows read "not applied" for work that has since shipped. Every row above was
> re-verified against `main` at `ae9d2e8` by rendering a report and reading the markup,
> not by reading the handoff. Where the handoff was still right — the three unrestacked
> card types, the surviving emoji — an issue is linked.

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

**Acceptance:** no pane repeats a value shown on its card face; a column with 0 missing has
no Missing tab; the low-outlier sentence is generated for all four cases; no emoji in
`numeric_card.py` or its panes.

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
