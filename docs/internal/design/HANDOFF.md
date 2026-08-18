# pysuricata report redesign — handoff

Start with **`integration.md`**. It opens with a status table saying what already landed and
what drifted, then fifteen phases mapped to files in your repo, each with acceptance criteria,
then three code findings, then a commit sequence.

**Read the status table first.** It was built by reading the source, not the example reports —
those are stale and disagree with the code. Regenerate them.

## Ready to use as-is

| File | What to do with it |
| --- | --- |
| `tokens.css` | replace the token layer of `pysuricata/static/css/_00-tokens.css`. Keeps the `--axis` token you added. Carries **five rules a contrast test cannot check**. Includes a compatibility shim so the rest of the CSS keeps working while phases land; delete it once they have |
| `contrast_test.reference.py` | copy into `tests/`. Asserts every pair the design uses clears its WCAG minimum in both themes, that `--data-3` carries no text, and that `--data-4` *fails* against the paper by design. Add it early — it caught four real failures |
| `flag_reference.reference.py` | copy into `pysuricata/render/`. The flag glossary — what each chip measures, the threshold that fires it, and what it means for the data — plus the flat-calendar baselines and the even-split helper. Phases 4b.2, 5e.2 and 5f.2 all read from it |

## Five rules a contrast test cannot check

In `tokens.css` with the measurements. They are about which surface a mark lands on, what
order marks paint in, what a zero value draws, where text lives, and how a gap is made — none
of which is a token pair. Each was a real defect caught in review, two of them twice.

1. **Only `--data-1`, `--data-2` and `--data-3` may sit on the paper or on an empty `--track`.**
   `--data-4` is stack-internal only. It reads as a reasonable "quiet" choice for a
   subordinate chart and at 1.83:1 it is a ghost. *(There is one live violation of this in
   shipped code — see the code findings in the plan.)*
2. **A quality-coloured mark crossing a data fill must protrude onto the paper or paint
   underneath it** — `--q-bad` on `--data-2` is 1.08:1. A reference line goes *before* the bars
   so they occlude it, and carries its value as a tick in the axis gutter. A marker inside a
   band must protrude and differ by **shape**, not colour. A point mark that may land on
   another carries a paper halo. Marks closer than one diameter collapse into a capsule
   carrying its count.
3. **A zero count draws nothing.** `v === 0 ? 0 : Math.max(1, …)`. A 1px floor is right for a
   small non-zero value and wrong for zero — ten empty months drawn as ten 1px bars assert
   data that is not there.
4. **Text never lives inside a scaled SVG.** Uniform scale makes text size track the viewport;
   non-uniform scale stretches glyphs. Bars and rules in SVG with
   `preserveAspectRatio="none"`; all labels in HTML at percentage offsets.
5. **A bar gap is not geometry.** A 1-unit gap becomes 0.28px at 284px and bars merge. Draw
   bars edge to edge with a `--paper` non-scaling stroke.

And one nothing catches: `test_contrast.py` checks tokens against `--paper` only. A foreground
on a **tinted** surface has to be measured by hand.

**`--data-3` is `#5C7F99`, not `#7FA0B5`.** At 2.63:1 on the paper the old value could not
carry a standalone mark, which is the job a third step exists for. `#5C7F99` clears 3:1 on
both surfaces in both themes, and is too mid-tone for text either way — a fill, never a label
background.

## The designs

Open any `.dc.html` directly in a browser. Each shows desktop and mobile side by side with the
reasoning and trade-offs beside each option. Newest work is at the top of each file.

| File | Covers |
| --- | --- |
| `Report Screen.dc.html` | header, summary and sample assembled, plus what happens when many columns are missing |
| `Variables Section.dc.html` | everything between the Variables heading and the first card — the attention block's chips, the flag reference, and the `display: none` pagination defect |
| `Details.dc.html` | the details panes for all four column types — six numeric panes plus the tab strip — and the audit behind the status table |
| `Histogram.dc.html` | histogram geometry: the 68%-blank-space problem, tick density, the gutter, captions, 50 bins on a phone |
| `Datetime Card.dc.html` | the datetime card face — the invented timezone, two unjudgeable ratios, bars vs line |
| `Categorical Card.dc.html` | the categorical card face — one face asked to serve four kinds of column |
| `Correlations and Missing.dc.html` | correlations (three states) and missing values (two options) |
| `Palette.dc.html` | the colour system, and comparing two datasets |
| `Variables 4b.dc.html` | variables for all five column types, and the axis treatment |
| `Report Baseline.dc.html` | faithful recreation of the original UI, for before-and-after |
| `Report Redesign.dc.html` | the header / summary / sample options that led to the choices |
| `Variables.dc.html` | superseded exploration of three variables layouts — kept for the reasoning |

## Where the numbers came from

`source/` holds the generated reports every figure was taken from. `current_titanic.html` and
`current_daily.html` are the latest pair; `titanic_report.html` and `daily_report.html` are
the originals the baseline recreation was measured against.

Bin counts, quantiles, percentiles, outlier severities and their IQR/MAD multiples, extreme
row indices, level counts and coverage, entropy, label lengths, temporal peaks and chunk
percentages are all real output. Five things are derived or illustrative, each labelled in the
file that uses it: correlation values above 0.5 (Titanic has none), the second series in the
comparison charts, the `Ticket` length counts, the boolean true-rate-by-chunk chart, and the
50-bin histogram in `Histogram.dc.html` 13b.

## Fix these first, in this order

Everything below is in the plan with a phase number. This is the order I would take it in.

1. **`pagination.js:163` — `card.style.display = 'none'`** (phase 4b.1). Breaks Ctrl+F, the
   anchor links from the attention block, deep links and print. A 60-column profile exports to
   PDF as 10 columns with nothing saying so. Silent, and the primary action in a profiling
   report is finding a column by name.
2. **`--data-4` as a standalone bar fill** in `correlations_section.py:169`. Violates rule 1,
   which is written down and shipped.
3. **The hardcoded `"UTC"`** in `datetime_card.py` (phase 5e.1). `DateTimeStats` already
   carries `source_timezone` and the accumulator populates it; the card ignores it and labels
   an Eastern column UTC, in three places.
4. **The 20% cliff** (phase 4b.3). `Age` is missing 19.9% of its values and does not appear in
   the needs-attention block at all — it misses `bad` by one tenth of a point.

## Bugs, not design

Spread across the phases, and easy to lose in a long document:

1. `pagination.js` hides cards with `display: none` (as above).
2. `correlations_section.py:169` fills a standalone diverging bar with `--data-4` (1.83:1).
3. `correlations_section.py:364` — the enriched correlations empty state landed on one path
   only; a frame with fewer than two numeric columns still gets one bare line.
4. `missing_section.py:245–395` — ~150 unreachable lines from the old two-tab implementation,
   carrying their own legend with hardcoded severity colours.
5. `datetime_card.py` — `("Timezone", "UTC", None)` is a literal, and `_format_timestamp`
   appends `UTC` regardless.
6. `datetime_card.py` — `Weekend %` and `Business hrs %` print bare; their flat baselines
   (28.6% and 23.8%) exist only in a code comment beside the threshold.
7. `categorical_card.py` — entropy, rare levels and top-5 coverage are computed and shown for
   2-level and fully-unique columns, where they describe nothing.
8. `triage.py` — chips carry their threshold in a `title` only, so it is invisible on a phone
   and absent from print; and two adjacent chips render as one run-together string.
9. `html.py` — the "Analyzing N variables" sentence duplicates the Summary composition bar and
   prints `0 boolean` for an absent type; the filter tabs render for types with no columns.
10. `_format_tick_label_standardized(v, is_count=True)` can emit `12.5K` — five glyphs, which
    breaks the fixed 44px y gutter.

## Already fixed upstream

Noted so they are not re-reported: the KMV distinct estimate exceeding the row count,
`.stat-badges` rendered then hidden, the `NaN` label length (`avg_len` read off the wrong
object — which affected *every* categorical column, not just `Embarked`), and the Top-N
chooser rendering three buttons all reading `2`. All covered by tests in your repo, and the
`_unknown_cell` pattern that came out of the length fix is better than what the design asked
for.

## Still undesigned

In the order I would take them: **dark mode** (values proposed and measured for the data scale,
never looked at in situ), **print/PDF** as a whole (interacts with fix 1), **degenerate frames**
(one column, zero rows, all one type), and the **section-header system** — Summary uses a bare
`<h2>` where the other four sections use `.section-title`.

One open question worth deciding before phase 4b.2 lands: **should pysuricata recommend
actions?** `Variables Section.dc.html` 15a groups flagged columns by what to do — drop or
impute, exclude as identifiers, transform. It is the highest-value option in the whole set and
the only one that makes a claim beyond the data. Everything else reports facts.
