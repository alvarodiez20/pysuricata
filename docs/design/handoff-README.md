# pysuricata report redesign — handoff

Start with **`integration.md`**. It opens with a status table saying what already landed
and what drifted, then nine phases mapped to files in your repo, each with acceptance
criteria, and a commit sequence.

**Read the status table first.** It was built by reading the source, not the example
reports — `examples/titanic_report.html` and `reports/daily_2026-08-16.html` both predate
the numeric restack, so they disagree with the code. Regenerate them.

## Ready to use as-is

| File | What to do with it |
| --- | --- |
| `assets/tokens.css` | replace the token layer of `pysuricata/static/css/_00-tokens.css`. Keeps the `--axis` token you added. **`--data-3` changes value** — see below. Includes a compatibility shim so the rest of the CSS keeps working while phases land; delete it at commit 15 |
| `assets/test_contrast.py` | copy into `tests/`. Updated since the first handoff: it now also asserts that `--data-3` carries no text and that `--data-4` *fails* against the paper by design |
| `assets/logo_mark.png` | the meerkat alone, for the app bar and favicon. **Re-export from vector before shipping** — this is a crop of the existing raster. Inline SVG is better still: it follows `currentColor` into dark mode, so the `#logo-light` / `#logo-dark` swap disappears |
| `assets/logo_wordmark.png` | the wordmark alone, for reference |

## Three rules a contrast test cannot check

These are at the bottom of `tokens.css` with the measurements. They are about which surface
a mark lands on, what order marks paint in, and what a zero value draws — none of which is
a token pair. Each was a real defect caught in review, and one of them twice.

1. **Only `--data-1`, `--data-2` and `--data-3` may sit on the paper or on an empty
   `--track`.** `--data-4` is stack-internal only — legal beside another segment, never
   alone on a background. It reads as a reasonable "quiet" choice for a subordinate chart
   and at 1.83:1 it is a ghost.
2. **A quality-coloured mark crossing a data fill must protrude onto the paper or paint
   underneath it.** `--q-bad` on `--data-2` is 1.08:1; no warm colour is visible on a blue
   fill. A reference line goes *before* the bars so they occlude it, and carries its value
   as a tick in the axis gutter. A marker inside a band must protrude past the edge and
   differ from its neighbour by **shape**, not colour.
3. **A zero count draws nothing.** `v === 0 ? 0 : Math.max(1, …)`. A 1px floor is right for
   a small non-zero value and wrong for zero — ten empty months drawn as ten 1px bars
   assert data that is not there.

**`--data-3` changes from `#7FA0B5` to `#5C7F99`.** At 2.63:1 on the paper the old value
could not legally carry a standalone mark, which is the job a third step exists for.
`#5C7F99` clears 3:1 on both surfaces in both themes, and is too mid-tone for text either
way — so it is a fill, never a label background.

## The designs

Open any `.dc.html` directly in a browser. Each shows desktop and mobile side by side, with
the reasoning and trade-offs written next to each option. Newest work is at the top of each
file.

| File | Covers |
| --- | --- |
| `Details.dc.html` | **the details panes for all four column types**, plus the audit that produced the status table |
| `Report Screen.dc.html` | the chosen header, summary and sample assembled |
| `Variables 4b.dc.html` | variables for all five column types, and the axis treatment |
| `Correlations and Missing.dc.html` | correlations (three states) and missing values (two options) |
| `Palette.dc.html` | the colour system, and comparing two datasets |
| `Report Baseline.dc.html` | faithful recreation of the original UI, for before-and-after |
| `Report Redesign.dc.html` | the header / summary / sample options that led to the choices |
| `Variables.dc.html` | superseded exploration of three variables layouts — kept for the reasoning |

## Where the numbers came from

`source/` holds the generated reports every figure was taken from. `current_titanic.html`
and `current_daily.html` are the latest pair; `titanic_report.html` and `daily_report.html`
are the originals the baseline recreation was measured against.

Bin counts, quantiles, percentiles, outlier severities, top-N values, entropy, temporal
peaks and chunk percentages are all real output. Four things are illustrative and each is
labelled in the file that uses it: correlation values above 0.5 (Titanic has none), the
second series in the comparison charts, the `Ticket` length counts, and the boolean
true-rate-by-chunk chart.

## Bugs, not design

Spread across phases 5b, 5c, 6 and 7, and easy to lose in a long document:

1. `Embarked` prints `Label length (avg)` as **`NaN`** and `Length p90` as an em dash, for a
   column whose three labels are all one character long.
2. `Embarked` carries `Case variants` and `Trim variants` flags while its normalization pane
   finds no collisions to justify either. One of the two is wrong.
3. Quality flags carry `data-threshold` and `data-value` in the DOM and display neither, so a
   card says `Missing` where it could say `19.9% missing`.
4. The categorical Top-N control renders a single button reading `3` — a chooser with one
   choice. Hide it when the level count is below the smallest step.
5. High-cardinality columns (`Name`, `Ticket`, `Cabin`) render ten bars of one row each, on
   the card *and* in the details pane.
6. `📊` `📈` `📉` `💡` `ℹ️` in `correlations_section.py` and the numeric outlier pane.
7. `Age`'s outlier pane spends ~60px announcing `Low Outliers — 0 (0.0%)`. The reason is
   worth stating instead: the lower IQR fence sits at **−6.7 years**, so the column cannot
   have a low outlier.

## Two things already fixed upstream

Noted so they are not re-reported: the KMV distinct estimate exceeding the row count, and
`.stat-badges` being rendered then hidden, are both covered by `tests/test_flag_chips.py`
in your repo.
