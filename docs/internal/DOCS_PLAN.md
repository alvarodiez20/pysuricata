> **Superseded.** This audit is a snapshot at 0.0.26 and its headline — "87
> mechanical errors across 21 pages" — no longer describes the tree;
> `python -m benchmarks.check_docs --strict` currently reports **0 errors** over
> 39 pages. The findings it did not cover were re-audited and filed as #266–#281.
> Kept as a record of the method. Not documentation, not published.

# Documentation audit and visual plan

Generated from a mechanical pass over all 31 pages at 0.0.26 (`52e13cd`), plus a
read of the pages a checker cannot judge. Re-run the audit with:

```bash
python -m benchmarks.check_docs --quiet-info      # report
python -m benchmarks.check_docs --strict          # exit 1 on any ERROR (for CI)
python -m benchmarks.check_docs --json out.json   # every finding with a line number
```

---

## Where things stand

| | |
|---|---|
| Pages | 31, ~8,500 lines |
| Mechanical errors | **87** across 21 pages |
| Broken examples | 33 fences raise when pasted |
| Conceptual pages with a diagram | **0 of 17** |
| Pages stamped "Last updated: 2025-10-12" | 8 |

`performance.md` has 16 errors — the most of any page, and it is the page that
carries the positioning claim you are about to link from a blog post. Every
configuration example on it is broken and every timing predates three rounds of
optimisation. Fix it before publishing anything.

---

## Five failure classes

### 1. Config options that do not exist — 17 occurrences, real API drift

Seven pages document `config.compute.uniques_sketch_size` and
`config.compute.top_k_size`. **Neither exists.** The real names are `uniques_k`
and `topk_k` on the config object (`max_uniques` / `top_k` on `ComputeOptions`).
A reader following the docs gets a silently ignored setting rather than an error,
which is worse.

Affected: `configuration.md`, `api.md`, `performance.md`, `stats/numeric.md`,
`stats/categorical.md`, `why-pysuricata.md`.

### 2. Renamed summary keys — 10 occurrences, real API drift

`summarize()["dataset"]["rows"]` is now `rows_est`. Nine pages show the old key
and the example raises `KeyError`. The accuracy oracle caught this rename; the
tests were updated and the docs were not.

### 3. Fences missing their imports — 25 occurrences

Blocks calling `profile()`, `summarize()` or `ProfileConfig(...)` with no import
line. A reader who copies the block gets a `NameError`. Every fence should stand
alone or be explicitly marked as a fragment.

(The docs used to teach `ReportConfig`, which was a live alias. It now warns on
use and goes in 0.3.0 (#210), so every fence was migrated to `ProfileConfig`.)

### 4. Output tagged as `python` — 9 occurrences

Console output and shell transcripts inside ```` ```python ```` fences.
`quickstart.md`, `usage.md` and `why-pysuricata.md` have three each. Re-tag as
`text` or `console`.

### 5. Prose describing behaviour you removed

Needs reading, not a checker. Confirmed:

- `architecture-diagrams.md:109` — "ExtremeTracker … Every 5th chunk only". That
  throttle was removed in 0.0.26; extremes are exact now.
- `architecture-diagrams.md:190` — "Type inference … first chunk only". Now gated
  on `first_chunk_is_whole`.
- `algorithms/sampling.md` — documents **Algorithm R** in full, with a proof. The
  library implements **Algorithm L** with a bulk scheduler. 68 lines teaching the
  wrong algorithm.
- `algorithms/sketches.md:63` — teaches KMV using `hashlib.md5`. The library uses
  blake2b for bytes and a vectorised splitmix64 for numeric arrays.

### Also: nothing documents the last three months

No page mentions the accuracy oracle, the chunked-equals-unchunked invariant, the
vendored native crate, or the CI gate. `changelog.md` is excellent and current and
is the only place any of it exists. "We assert chunked == unchunked in CI" is a
selling point, and it is invisible.

---

## Per-page triage

| Page | errors | Dominant problem | Action |
|---|---:|---|---|
| `performance.md` | 16 | 10 broken examples, 6 bad config names, 5 stale timings | rewrite |
| `configuration.md` | 10 | 7 config names that do not exist | rewrite |
| `quickstart.md` | 8 | 4 broken examples, 3 mis-tagged fences | rewrite |
| `why-pysuricata.md` | 8 | mis-tagged fences, bad config, stale keys | rewrite |
| `stats/categorical.md` | 7 | 4 bad config names | patch |
| `api.md` | 6 | examples + config + `rows` key | patch |
| `examples.md` | 6 | examples needing files; `rows` key | patch |
| `usage.md` | 6 | mis-tagged fences, missing imports | patch |
| `faq.md`, `stats/numeric.md` | 4 | examples, config names | patch |
| `analytics/correlations.md`, `stats/boolean.md` | 3 | examples, `rows` key | patch |
| `advanced.md`, `stats/datetime.md` | 2 | examples | patch |
| `changelog.md`, `install.md` | 1 | one mis-tagged fence each | trivial |
| `architecture-diagrams.md` | 0 | mechanically clean, **semantically stale** | merge & redraw |
| `algorithms/*.md` | 0 | clean; wrong algorithm, no visuals | rewrite + illustrate |

A page can be mechanically clean and still be the most wrong page in the set:
`algorithms/sampling.md` has zero errors and documents an algorithm you do not
use. The checker buys you the boring half so your attention goes to this half.

Also: `docs/roadmap.md` is on disk but not in the mkdocs nav, so it is never
rendered. Add it or move it out of `docs/`.

---

## The plan

### Phase 0 — Make drift impossible to reintroduce (1 hour)

Land `benchmarks/check_docs.py` in CI with `--strict`. From then on a renamed
config option or a moved summary key fails a PR instead of rotting for ten
months. Fix the nav orphan.

**Exit:** CI red on the current tree, which is the point.

### Phase 1 — Mechanical sweep (2–3 hours)

Three global replacements are most of the count:

- `uniques_sketch_size` → `uniques_k`
- `top_k_size` → `topk_k`
- `["dataset"]["rows"]` → `["dataset"]["rows_est"]`

Then re-tag the nine output-as-python fences and add imports to the 25 orphan
fences. Re-run until only INFO remains.

**Exit:** `check_docs --strict` green.

### Phase 2 — The four pages that carry the story (1 weekend)

`index.md`, `why-pysuricata.md`, `quickstart.md`, `performance.md`. Rewrite with
current numbers (13.3× vs ydata-profiling, 12× less marginal memory), the
bounded-memory claim stated plainly, and the chunked-equals-unchunked invariant
named. Delete every hardcoded timing you cannot regenerate.

**Exit:** a stranger reads `index.md` and can say what the library is for.

### Phase 3 — Rewrite the algorithms trilogy, with visuals (1 weekend)

`sampling.md` to Algorithm L (and say why: bit-identical across chunk sizes).
`sketches.md` to the real hashing plus the threshold idea. `streaming.md` to
Pébay's merge and the reason it exists. This is where the animations go. It is
also the most defensible content you have — nobody else's docs explain this.

### Phase 4 — Collapse three architecture pages into one (half weekend)

`architecture.md`, `architecture-diagrams.md` and
`sequence-diagrams-complexity.md` overlap heavily and two carry removed claims.
One page, one chunk-lifecycle diagram, one complexity table.

**Exit:** no diagram asserts anything the code does not do.

### Phase 5 — Reference pages (ongoing)

`stats/*` and `analytics/*` — 2,900 lines, mechanically patched in phase 1,
illustrated with annotated card screenshots as you go. Drop the "Last updated"
footers on eight pages: generate them from git or remove them, because a wrong
date is worse than no date.

---

## What would benefit most from diagrams and animations

The distribution today is exactly inverted. Pages that describe *processes* have
no visuals; the only pages with diagrams describe module layout, which a
directory listing already conveys.

| Section | Pages | Lines | Diagrams |
|---|---:|---:|---:|
| `algorithms/` | 3 | 857 | **0** |
| `stats/` | 5 | 2,184 | **0** |
| `analytics/` | 4 | 910 | **0** |
| `sequence-diagrams-complexity.md` | 1 | 297 | 6 |
| `architecture.md` | 1 | 219 | 5 |
| `architecture-diagrams.md` | 1 | 197 | 4 (2 stale) |

**The test for "does this need an animation?" is: does the idea have a time
axis?** A streaming algorithm is a state machine walking a sequence — a moving
picture by nature, and prose is the wrong medium. A formula is not.

### Ranked

**1. Reservoir sampling — `algorithms/sampling.md` — animate.**
The best candidate in the set, and you are rewriting the page anyway. Two panels,
same stream. Left, Algorithm R: every element tested, the reservoir flickering
constantly early then almost never. Right, Algorithm L: a pointer leaping over
the elements it will never select, landing only on acceptances. The viewer *sees*
the work disappear — which is the whole point of Algorithm L and something the
current proof sketch cannot convey. Plus a static chart: retention probability vs
stream position, a flat line at k/n. That is what the proof is trying to say.

**2. KMV on the unit interval — `algorithms/sketches.md` — animate.**
Possibly the most elegant animation in streaming statistics, currently two pages
of prose. Hashes land on `[0,1)`; the k smallest stay lit; the k-th smallest *t*
creeps toward zero; the estimate `(k-1)/t` converges. Duplicates land on existing
marks and change nothing — the intuition for why a distinct-count sketch works.
Draw the admission threshold as a gate and everything beyond it as excluded: that
*is* the pre-filter optimisation on the roadmap, so the diagram documents the code
and motivates it at once. **A working version of this ships in
`scripts/build_docs_assets.py` — 18 KB, generated from a real `KMV` run.**
Second animation on the same page: Misra-Gries eviction — counters fill, a new key
arrives, everything decrements, the zeros drop out. Hard to follow in text,
trivial as a picture.

**3. The memory curve — `index.md` and `why-pysuricata.md` — static chart.**
Not a diagram, an *asset*. Two lines, rows on x, peak RSS on y: yours flat, the
incumbent's linear until it stops at a MemoryError. The single picture that makes
"bounded memory" concrete, the one that travels on social media, and your front
page currently has no chart at all. Generate it from `benchmarks/end_to_end.py` so
it is a measurement rather than an illustration.

**4. Annotated report cards — `stats/*.md` — screenshots.**
2,184 lines explaining what the numbers on a card mean, with no picture of the
card. A reader of these pages has the report open in another tab. One annotated
screenshot per column type replaces several screens of prose. Cheapest high-value
item here — you already generate the report; script the capture with Playwright.

**5. The chunk lifecycle — merged architecture page — mermaid.**
One sequence diagram: source → adapter → chunk → consume → per-column accumulators
→ merge → finalize → render, with the bounded-memory boundary drawn explicitly
(what is O(1), what is O(n)). Replaces three overlapping pages, two of which
currently lie.

**6. Welford → Pébay merge — `algorithms/streaming.md` — static.**
Two partitions with their own means and moments combining into one, with the
`δ²·n_a·n_b/n` correction drawn as the thing accounting for the gap between the
means. Static is right; there is no time axis, just a decomposition. This is also
the diagram that explains *why* chunked equals unchunked.

**7. Missingness matrix — `analytics/missing-values.md` — screenshot.**
309 lines describing a visualisation without showing it.

### What not to animate

Formulas, config tables, the FAQ, the API reference. Motion on a static idea is
decoration, costs bytes, and fights the reader. `stats/overview.md` already uses
MathJax for ten formulas and is better for it.

---

## How to build them without creating new drift

**The trap:** you have just spent three rounds fixing documentation that went
stale because nothing checked it. Twenty hand-drawn diagrams would recreate that
problem in a medium nobody can diff. **Generate the visuals, do not draw them.**

| Need | Use | Not |
|---|---|---|
| Flow, sequence, state | **Mermaid** — already enabled, stays text in git, diffable in review | an exported image |
| Algorithm animation | **Inline SVG with CSS/SMIL** — a few KB, scales, inherits theme through CSS variables so dark mode works free, honours `prefers-reduced-motion` | **GIF** — megabytes, fixed palette, wrong on dark backgrounds, invisible to screen readers, unreviewable in a PR |
| Charts from measurements | **SVG emitted by a script** reading `benchmarks/*.json` | a screenshot of a notebook |
| Report cards | **Playwright screenshots** of a freshly generated report, cropped and annotated by the same script | a manual screen grab |

### The shape of it

```
scripts/build_docs_assets.py          # one entry point, regenerates everything
docs/assets/generated/
    kmv-unit-interval.svg             # SHIPPED — animated, from the real KMV
    reservoir-r-vs-l.svg              # next
    misra-gries-eviction.svg
    memory-curve.svg                  # from end_to_end.py results
    card-numeric.png                  # Playwright, annotated
    MANIFEST.json                     # digest per asset
```

`python scripts/build_docs_assets.py --check` fails if any committed asset no
longer matches what the code generates — the same trick as a snapshot test. A
change to `ReservoirSampler` that invalidates the animation shows up as a red
build, not as a diagram that quietly starts lying.

### Two details worth getting right

- **Animate from the real thing.** The KMV asset replays an actual `KMV` run with
  a fixed seed, using the library's own hash. The picture is evidence, and it
  changes when the algorithm changes.
- **Give every animation a still first frame and a caption that stands alone.**
  Print, screen readers, `prefers-reduced-motion` and the skimming reader all need
  the static version to carry the point.

### Order

Checker into CI → mechanical sweep → the four story pages with the memory curve →
the algorithms trilogy with the two animations → everything else. The first two
are hours, not weekends, and they stop the bleeding. The memory curve and the KMV
animation are the two assets that will do work for you outside the docs — in the
README, in post #3, and in the Show HN thumbnail.
