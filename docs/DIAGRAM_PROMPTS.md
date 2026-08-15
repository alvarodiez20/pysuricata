# Prompts for generating PySuricata's diagrams and animations

Ready to paste into Claude. One prompt per visual, in the order I would build them.

---

## Read this first: two kinds of visual, two workflows

A diagram that asserts something about the code can go stale, exactly like the
prose did. So these split into two groups, and the workflow differs.

**Group A — must stay truthful.** Reservoir sampling, KMV, Misra-Gries, the
memory curve, report screenshots. These depict *what the code actually does*.
Use Claude to design the visual language once — layout, motion, labelling — then
**port the result into `scripts/build_docs_assets.py`** so it is regenerated from
a real run with a fixed seed and `--check` fails CI when it drifts. Claude gives
you the design; the script gives you the guarantee. `kmv-unit-interval.svg` is
already built this way and is the reference.

**Group B — stable by nature.** The chunk lifecycle, the Pébay merge, the
bounded-memory architecture. These depict *concepts*, not measurements. Hand-authored
Mermaid or SVG is fine; they change only when the architecture does.

Tell Claude which group it is — the prompts below already say.

## The house constraints (paste with every prompt)

```
Constraints for every asset:
- A single self-contained SVG. No external CSS, no JS, no web fonts, no images.
- Under 25 KB. Under 12 KB if it is static.
- Theme through CSS custom properties with literal fallbacks, so it works inside
  mkdocs-material in both light and dark mode:
    fill:  var(--md-default-fg-color, #0b0b0b)
    muted: var(--md-default-fg-color--light, #52514e)
    hair:  var(--md-default-fg-color--lighter, #b6b5ae)
    accent:var(--md-primary-fg-color, #2a78d6)
  Second accent, when one is genuinely needed: #eb6834 (orange). Never more than
  two accents. Never red/green as the only distinction.
- Animation: SMIL <animate> or CSS keyframes, looping, 8-14 s per cycle, and
  wrapped so motion stops under reduced-motion:
    @media (prefers-reduced-motion: reduce){ animate{ display:none } }
  The still first frame must make the point on its own — assume print, a screen
  reader, and a reader who is skimming.
- role="img" plus an aria-label that states the conclusion in one sentence, not
  a description of the shapes.
- Type: system-ui stack, 11 px labels, 13 px for the one number that matters.
- Do not put a title inside the SVG; the page supplies the heading.
```

---

## 1. Reservoir sampling — Algorithm R vs Algorithm L

**Group A** · for `docs/algorithms/sampling.md` · animated · the highest-value asset

```
Design an animated SVG for a data-profiling library's documentation, explaining
reservoir sampling. It goes on a page that currently has 68 lines of prose and no
picture.

The idea to convey: a reservoir sampler keeps a fixed-size uniform random sample
from a stream of unknown length. Every element seen ends up in the sample with
probability k/n regardless of when it arrived. Algorithm R tests every element.
Algorithm L computes a geometric skip and jumps straight over the elements it
will never select, which is the same sample for far less work.

Layout: two horizontal tracks stacked, sharing one stream. Label the top
"Algorithm R — tests every element" and the bottom "Algorithm L — skips to the
next acceptance".

- The stream is a row of small ticks flowing right to left, endless.
- Below each track, k=8 reservoir slots drawn as rounded rectangles.
- Top track: a marker touches every arriving tick; occasionally a slot flashes
  and swaps its value. Early on this happens often, then visibly rarer.
- Bottom track: a pointer sits idle, then leaps forward over a run of ticks and
  lands on exactly one — a slot swaps. The ticks it skipped are never touched.
  Draw the leap as an arc so the jump reads as deliberate.
- A small counter on each track: "elements examined". The gap between the two
  numbers widening over the loop IS the point of the animation — make that the
  thing the eye lands on.

The still first frame should show both tracks with partly filled reservoirs and
the two counters already differing, so a static reader sees the comparison.

aria-label: "Algorithm R examines every element of a stream; Algorithm L skips
directly to each element it will accept, producing the same uniform sample with
far fewer operations."

[paste the house constraints]
```

**Then:** port into `build_docs_assets.py` driven by a real `ReservoirSampler`
run with a fixed seed, so the skip pattern is the library's own.

---

## 2. Misra-Gries eviction

**Group A** · for `docs/algorithms/sketches.md` · animated

```
Design an animated SVG explaining the Misra-Gries heavy-hitters sketch, for a
data-profiling library's documentation.

The idea: the sketch keeps at most k counters. A value already counted increments
its counter. A new value with a free slot takes it. A new value with no free slot
causes every counter to decrement, and any counter reaching zero is dropped —
which is how a bounded structure survives an unbounded number of distinct values.

Layout: k=6 counter slots as a row of labelled bars, each showing a key and a
count. A stream of keys arrives from the right, one at a time.

Three cases, cycling, each clearly distinguished:
1. Known key arrives — its bar grows. Quiet, no other movement.
2. New key, free slot — the empty slot fills. Quiet.
3. New key, all slots full — THE moment. Every bar drops by one simultaneously,
   any bar hitting zero fades out, and the new key takes the freed slot. Give
   this beat extra time and a brief label: "all counters decrement".

Case 3 is what readers cannot follow in prose, so it should get the most screen
time and the clearest treatment. The other two are context.

Still first frame: mid-stream, slots partly filled, one bar low enough that the
reader can see it is about to be evicted.

aria-label: "A Misra-Gries sketch with six counters. When a new key arrives and
no slot is free, every counter decrements and any counter reaching zero is
dropped, which keeps memory bounded however many distinct values arrive."

[paste the house constraints]
```

---

## 3. The memory curve

**Group A** · for `docs/index.md` and `docs/why-pysuricata.md` · static · the marketing asset

```
Design a static SVG line chart for the front page of a data-profiling library.

The claim: PySuricata's memory use is flat as the dataset grows, because it makes
one streaming pass with fixed-size sketches. The incumbent's grows linearly and
eventually fails.

Two lines, rows on x (log scale, 10k to 100M), peak memory on y (linear, MB):
- PySuricata: essentially flat, a slight rise. Accent colour, thicker, direct-labelled
  at the right end. No legend box.
- ydata-profiling: rising steadily, then STOPPING at a marked point with a small
  cross or break and the label "MemoryError". Muted grey.

The ending is the story — the flat line continuing past the point where the other
one stops. Give the flat line room to keep going after the failure point so the
contrast is spatial, not just colour.

Annotate one measured point on each line rather than every point. Put the
measurement conditions in small muted text at the bottom: machine, versions, date.
Leave those as clearly marked placeholders — I will fill them from a real run.

Do not use a bar chart. Do not add a gridline for every tick; two or three
horizontal hairlines is enough.

aria-label: "Peak memory against dataset size. PySuricata stays flat as rows grow;
ydata-profiling rises linearly and fails with a MemoryError."

[paste the house constraints, but skip the animation clause]
```

**Then:** generate it from `benchmarks/end_to_end.py` output so it is a
measurement, not an illustration. Never ship this one hand-drawn.

---

## 4. The chunk lifecycle

**Group B** · for the merged architecture page · static Mermaid

```
Write a Mermaid diagram for a streaming data-profiling library's architecture page.
It replaces three overlapping pages, two of which describe behaviour that has been
removed, so accuracy matters more than elegance.

The flow, once per chunk:
  source (DataFrame | LazyFrame | iterator of chunks)
    -> adapter selection (peek one chunk, splice it back)
    -> chunk loop
        -> per column: convert to array
        -> per column: accumulator.update(array, row_offset)
             numeric | categorical | datetime | boolean
    -> after the last chunk: finalize() per column
    -> render: HTML report, or summarize(): JSON

The one thing the diagram must communicate that a call graph would not: **which
state is bounded**. Draw a boundary around the accumulator state and label it
"O(1) in rows — fixed-size sketches". Mark the chunk itself as "O(chunk_size),
released after each iteration". Everything outside that boundary should read as
transient.

Use a subgraph for the per-chunk loop so the repetition is visible. Keep node
labels to four words. No colours beyond Mermaid's defaults — the page theme
handles that.

Return the Mermaid source only, in a fenced block, ready to paste into a
mkdocs-material page.
```

---

## 5. Welford → Pébay merge

**Group B** · for `docs/algorithms/streaming.md` · static SVG

```
Design a static SVG explaining why streaming moments can be computed in parallel
chunks and combined exactly.

The idea: two partitions each hold their own count, mean and central moments.
Pébay's formulas combine them into the moments of the union exactly — which is why
profiling data in 1 chunk and in 50 chunks gives identical answers.

Layout: two boxes side by side, each labelled with n, mean and M2. Below them, an
arrow into a single combined box. The interesting part is the correction term:
draw the horizontal distance between the two means as a bracketed span labelled
delta, and show that the combined M2 is M2_a + M2_b PLUS a term built from delta
and the two counts. Make the correction term visually separate from the simple
sum — it is the part a reader will not guess.

Include the formula for M2 only, set in the diagram:
  M2 = M2a + M2b + delta^2 * na * nb / n
Mention in a one-line caption that M3 and M4 follow the same shape with larger
correction terms. Do not draw those.

No animation — there is no time axis here, only a decomposition.

aria-label: "Two partitions, each with its own count, mean and second moment,
combine exactly into the moments of their union; the correction term is built from
the distance between the two means."

[paste the house constraints, but skip the animation clause]
```

---

## 6. Annotated report card

**Group A** · for the four `docs/stats/*.md` pages · static, screenshot-based

```
I have a screenshot of a numeric column card from a data-profiling report. Design
an SVG annotation layer to sit over it: callout lines from labels in the margin to
specific regions of the card, explaining what each part means.

Regions to annotate on a numeric card:
- the column name, type badge and dtype badge
- the quality chips ("Skewed Right", "Heavy-tailed", "Many outliers") — this is
  the most valuable part and deserves the most prominent callout
- the left statistics table (count, unique, missing, outliers, zeros)
- the right statistics table (quartiles, min, max, mean)
- the histogram, and the scale/bins controls beneath it
- the Details disclosure

Rules: callout lines are thin, use one accent colour, and never cross each other.
Labels sit in the left and right margins, not on top of the card. Numbers on the
labels (1..7) with a short legend beneath, so the labels stay short.

Give me the annotation layer as a separate SVG positioned over a background image
placeholder, so the screenshot can be regenerated without redrawing the
annotations.

aria-label: "An annotated numeric column card, labelling the quality chips, the
two statistics tables, the histogram and its controls."

[paste the house constraints, but skip the animation clause]
```

**Then:** script the screenshot with Playwright against a freshly generated report
so it refreshes on every docs build, and keep the annotation layer's coordinates
in the same script.

---

## 7. The KMV sketch — already built

**Group A** · `docs/assets/generated/kmv-unit-interval.svg` · animated · 18 KB

Already generated by `scripts/build_docs_assets.py` from a real `KMV` run: hashes
land on the unit interval, the k smallest are kept, the view zooms to follow the
shrinking threshold, and the estimate `(k-1)/t` converges on the true count.

Only worth re-prompting if you want the visual language redesigned to match
whatever comes out of prompts 1 and 2 — in which case:

```
Here is an existing animated SVG explaining a K-Minimum-Values distinct-count
sketch [attach the file]. It is correct and I want to keep its content exactly:
the unit interval, the k retained marks, the threshold gate, the zooming window,
and the running estimate. Redesign only its visual language so it matches these
other two animations [attach 1 and 2] — same type scale, same accent usage, same
motion timing, same caption placement. Return SVG only.
```

---

## Order, and how to keep them honest

Build 3 first — it is the one asset that also works outside the docs, in the
README and in a launch post. Then 1 and 2, which are the ones that make the
algorithms pages worth reading. Then 4, which lets you delete two stale pages.
Then 5 and 6 as you rewrite the pages they belong to.

For every Group A asset, the last step is the same and is not optional: move it
into `scripts/build_docs_assets.py`, driven by the real class with a fixed seed,
and add it to the `--check` set. A picture that a script regenerates cannot
quietly start lying, which is the whole reason this document exists.
