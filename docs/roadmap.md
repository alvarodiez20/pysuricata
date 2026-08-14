# PySuricata roadmap — performance, correctness, positioning

Written 14 Aug 2026. Every number here was measured with the harness in
`benchmarks/`, against `pysuricata 0.0.16`, Python 3.11.15, pandas 2.3.3,
numpy 2.4.4, ydata-profiling 4.18.4, on a 2-core x86-64 Linux container with a
measured 26.0 GB/s streaming read. **Re-run before publishing anything** — your
machine will give different absolute numbers and the same ratios.

A rendered version of this document with charts exists as the
`pysuricata-roadmap` artifact.

---

## The short version

PySuricata already beats ydata-profiling by a wide margin and nobody knows,
because it has never been measured or published. Meanwhile a third of its runtime
goes to SHA-1 hashing that no sketch needs, a fifth goes to `dateutil` parsing
dates one row at a time, and six statistical bugs make chunked results disagree
with unchunked ones.

A native Rust core is the right project. It is not the *next* project.

| Measured today | |
|---|---|
| vs ydata-profiling, 25k x 14 mixed frame | **7.9x** faster wall clock (1.77 s vs 13.98 s) |
| vs ydata-profiling, marginal memory | **11x** less (28 MB vs 313 MB above the ~150 MB interpreter floor) |
| Runtime in SHA-1 | **~36%** of profiled self time |
| Runtime in `dateutil` | **20.7%** |
| Correctness bugs with a reproducing test | **6** |

---

## Phase 0 — Trust (2 weekends)

Nothing else matters if the numbers are wrong, and an optimisation you cannot
verify is a liability.

`benchmarks/accuracy.py` ships six `xfail`-marked tests. Each names a file and
line. Fix, delete the marker, move on. When one starts passing while the marker
is still there, pytest reports XPASS and tells you.

**Exit criterion:** the accuracy suite passes with zero xfails, and CI runs it on
every PR.

### The bugs

#### CRITICAL — generator sources silently drop chunk 0
`compute/orchestration/engine.py:127`

`next(iter(data))` is used to sniff which adapter to build, consuming the first
chunk. `chunks_from_source` then iterates the already-advanced generator. The
documented "stream chunks larger than RAM" API — the headline feature — silently
omits the first chunk from every statistic, and a one-chunk generator reports
"Empty source".

Fix: wrap in `itertools.chain([first], rest)`, or sniff from a peeked copy.
Half an hour.

#### CRITICAL — reservoir sampling is biased
`accumulators/sketches.py:311-324`

`add_many` draws `np.random.randint(1, seen + len(arr) + 1, size=len(arr))` — one
uniform over the *post-batch* count for every element. True reservoir sampling
requires each element `i` to be accepted with probability `k/(seen+i)`, a
denominator that grows *within* the batch. Early elements are under-weighted,
late ones over-weighted, and the bias grows with chunk size.

Every quantile, the median, IQR, MAD, outlier counts and the histogram come from
this reservoir, so accuracy silently degrades as chunk size grows — the opposite
of what the docs imply.

Fix: Algorithm L (Li 1994). Implemented in `native/src/reservoir.rs`; the
pure-Python version is ~25 lines. It is also much cheaper: ~145k random draws for
10M rows into k=20k, instead of 10M.

#### CRITICAL — skewness and kurtosis are wrong for multi-chunk data
`accumulators/algorithms.py:138-162`

`StreamingMoments._update_vectorized`'s M3/M4 batch merge is commented "simplified
for performance". It is not Pébay's formula. The M3 cross term is
`3*delta*M2_B/n_B` where it should be `3*delta*(n_A*M2_B - n_B*M2_A)/n`, and M4
has the same defect.

Results are correct for single-chunk data and wrong for everything else — which is
exactly why it survived. The `merge()` method twelve lines below has the correct
formula, so the two code paths in the same class disagree with each other.

Fix: build a `StreamingMoments` from the batch and call the existing correct
`merge()`. Net deletion of code. Reference implementation in
`native/src/moments.rs`.

#### SERIOUS — `profile()` resets the caller's global RNG
`report.py:88-89`

`np.random.seed(seed)` and `random.seed(seed)` are process-global, and the default
seed is 0. A notebook that seeds its own experiment and then profiles a frame has
its RNG state silently reset. It also makes per-column threading impossible to do
reproducibly, so this blocks the parallelism work.

Fix: per-accumulator `np.random.Generator`. Mechanical, ~6 call sites.

#### SERIOUS — correlations collapse to 0.00 on large-mean columns
`compute/analysis/correlation.py:153-158`

`vx = max(0.0, sx2 - sx*sx/n)` is the textbook catastrophic-cancellation form. For
timestamps-as-int, IDs, or prices around 1e6, the two terms differ only in the
last bits of float64 and the variance collapses to zero — then the function
returns `0.0`, silently reporting "no correlation". The `max(0.0, ...)` hides the
symptom instead of surfacing it.

Fix: Welford/Chan pairwise co-moments — the same merge already needed for M2.

#### MODERATE — skewness uses the sample variance in the denominator
`accumulators/algorithms.py:192`

g1 is `(m3/n) / (m2/n)^1.5` using the *population* second moment. The code divides
by `variance**1.5` with the n-1 form, a systematic scale error of `((n-1)/n)^1.5`.
Small, but it never converges away, and it means the numbers do not match scipy.

### Bugs without a test yet

- **Column types decided from the first chunk alone** — `engine.py:294`,
  `inference.py:545`. `should_reclassify_numeric_as_categorical` uses
  `unique_count / total_count < 0.05` over one chunk. A 100M-row numeric column
  with 10M distinct values can be permanently classified categorical because its
  first 200k rows had nine. Fix: defer the decision, or use the KMV estimate
  rather than a chunk-local ratio.
- **Row count silently truncates to 2,000 on the hash fallback** —
  `accumulators/sketches.py:483-488`. If vectorised row hashing raises for any
  chunk, the `except` path contributes at most 2,000 rows to `row_kmv.rows`. That
  value is what the report displays as "Rows" (`render/html.py:62`) and what
  `missing_cells_pct` divides by. Fix: always add `len(df)`; let only the
  duplicate *estimate* degrade, and mark it approximate.
- **Extreme-value row indices are chunk-local** — `consume.py:240-254`,
  `consume_polars.py:186-190`. The engine tracks a global row offset but never
  passes it down, so "row 4,182 had the maximum" is wrong for every chunk after
  the first. Extremes are also only sampled every 5th chunk
  (`_extreme_update_counter % 5`), so the reported min/max can miss the true
  extremes entirely. Fix: pass the offset; make min/max exact (O(1) per chunk) and
  keep the throttle only for top-k *with indices*.
- **Documented options that do nothing** — `ComputeOptions.columns`
  (`api.py:208`) is validated but never reaches the engine. `corr_max_cols`
  (`api.py:277`) is declared, validated, copied into config, and never read — a
  1,000-column frame builds 499,500 pairs despite a documented cap of 50.
  `chunk_size` is blended (`0.7*optimal + 0.3*requested`, `chunking.py:240`) so
  the user never gets what they asked for. Fix: wire them up or delete them from
  the docs.
- **Timestamps before 1906 counted as missing** — `accumulators/datetime.py:230`.
  The validity window is `[-2e18, 1e20]` ns; the lower bound is 1906-05-13.
  Birthdates and historical records are silently reclassified as nulls. Fix:
  widen to the int64 datetime64[ns] range (1677-2262).

### Why these were not caught

The test suite is 41 files and ~304 KB — but roughly **2.3 KB of it covers the
accumulators**, which are ~2,600 lines of the statistical core. There is no test
anywhere that chunked results equal unchunked results, and none that any statistic
matches NumPy. Five of the six bugs are in the numerical core.

---

## Phase 1 — Free performance (2-3 weekends)

No Rust, no new dependencies, no API change.

`python -m benchmarks.hotspots --suite mixed` on a 50,000 x 14 frame, 2.4 s
unprofiled, self time by subsystem:

| Subsystem | Share |
|---|---|
| accumulate | 28.8% |
| date parsing | 20.7% |
| unclassified | 17.3% |
| hashing (SHA-1) | 13.4% |
| pandas / numpy | 7.1% |
| list sorting | 5.5% |
| render | 4.9% |
| type inference | 1.6% |

`accumulate` is dominated by `sketches.py` `add_many` (0.589 + 0.234 + 0.135 s
self), which is the KMV ingest path. Adding its SHA-1 cost (`_u64` 0.365 s,
`digest` 0.174 s) and the list sorts it triggers (0.264 s), **the distinct-count
sketch alone is ~36% of profiled self time.**

### The work

1. **Replace SHA-1 with a 64-bit mixer** — `accumulators/sketches.py:11`.
   A cryptographic hash resisting preimage attacks is doing a job that needs
   uniformity and avalanche and nothing else. Measured **755x** on the hash alone.
2. **Replace the sorted list with a bounded heap** — `sketches.py:124`.
   `_batch_add_hashes` does `list.extend(...)` then `list.sort()` on every chunk,
   over boxed Python ints. Measured **12.3x** on the full ingest.
3. **Give `RowKMV` a direct hash path** — `sketches.py:454`. It computes a
   perfectly good vectorised `uint64` row hash with
   `pd.util.hash_pandas_object`, then hands it to `KMV.add_many`, which calls
   `str(v).encode()` and `hashlib.sha1` on every one of those uint64s. For a
   10M-row frame that is 10M string formats and 10M SHA-1 digests to re-hash
   values that were already uniformly distributed.
4. **Fix the date sniff** — `compute/processing/inference.py:383`. Every object
   column runs up to 10,000 rows through
   `pd.to_datetime(..., format="mixed")` as an "is this a date?" test.
   `format="mixed"` disables pandas' vectorised parser and falls back to
   `dateutil`, one row at a time, in Python: 166,302 `get_token` calls and 34,737
   `_strptime` calls in a single 50k-row profile. Try a small set of explicit
   formats first, sample 200 rows not 10,000, short-circuit on dtype.
5. **`np.diff` for monotonicity** — `accumulators/algorithms.py:422`.
   `MonotonicityDetector.update` iterates every value in Python. Measured **61x**
   from three lines of NumPy. No Rust required.
6. **`heapq.heappushpop` for extremes** — `algorithms.py:332`.
   `_add_to_min_heap` does an O(k) `max()` scan plus a full `heapify` per insert,
   on a data structure whose entire purpose is O(log k) inserts.
7. **Delete duplicated passes** — `adapters/pandas.py:217` `missing_cells` runs a
   full `isnull().sum().sum()` per chunk, duplicating work the accumulators
   already did.

**Exit criterion:** `hotspots.py` shows hashing and date parsing under 5% each,
and the `mixed` suite is >= 2x faster than v0.0.17 with byte-identical output.

---

## Phase 2 — Publish (1-2 weekends)

Run `python -m benchmarks.end_to_end --markdown results.md`. Put the table and the
environment block in the README and in `docs/benchmarks.md`.

Rules for anything published:

- Same DataFrame object for every tool, generated from a seeded RNG.
- Separate subprocess per tool (the harness already does this).
- Report peak RSS and output size next to wall time.
- Never compare a `minimal=True` incumbent run against a full PySuricata run, or
  the reverse, without labelling it. Someone will check.
- State the version of every package involved.
- **Publish the failures.** "ydata-profiling raised MemoryError at 5M x 40 on a
  16 GB machine" is more persuasive, and more honest, than a bar chart — and their
  own issue tracker corroborates it.

Also: comment on ydata-profiling issue
[#1129](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1129),
open since October 2022 asking for polars support that PySuricata already has.

**Exit criterion:** a benchmarks page exists, one post is live, and the polars
issue has a link to the docs.

---

## Phase 3 — Native core (4-6 weekends)

The crate in `native/` already builds, ships an abi3 wheel, passes 20 Rust unit
tests and 35 Python agreement tests, and releases the GIL (measured 91% parallel
efficiency on 2 threads).

| Module | Replaces | Design |
|---|---|---|
| `hashing.rs` | SHA-1 in `_u64` | splitmix64 for already-64-bit values; wyhash-style multiply-fold for bytes, three independent accumulator chains in the bulk loop. Includes an avalanche test. |
| `kmv.rs` | the sorted Python list | Fixed-capacity binary max-heap over a flat `Vec<u64>`. 8 KiB at k=1024, cache-resident; the common case rejects in one compare against the root. Reports its own relative standard error. |
| `moments.rs` | the "simplified" merge | Pébay's pairwise merge + a 32 KiB-tiled two-pass scan. Two passes per tile, both hitting L1, so DRAM traffic is paid once. |
| `reservoir.rs` | the biased sampler | Algorithm L with geometric skips, xoshiro256++ seeded per instance. |

### Measured kernel costs (1M rows, ns/row, best of 5-7)

| Kernel | Before | After | Ratio |
|---|---:|---:|---:|
| hash a value | 1138 | 1.5 | 755x |
| KMV ingest | 107 | 8.7 | 12.3x |
| monotonicity (NumPy, not Rust) | 71 | 1.2 | 61x |
| full numeric column | 32 | 6.5 | 4.9x |
| reservoir sampling | 13 | 7.2 | 1.8x + unbiased |

### The instructive failure

The first native numeric kernel — one fused single pass, Welford per value — came
out at **0.7x the speed of NumPy's multi-pass version**. Fewer passes over memory,
and it lost. Welford's update has a division and a loop-carried dependency on
every element, so the scalar loop stalls on latency while NumPy's passes vectorise
cleanly.

Blocking the array into 32 KiB L1-resident tiles, computing moments about the
*tile* mean in a second L1-only pass, splitting the accumulators into four
independent chains, and replacing the per-value `ln()` with a renormalised running
product took it from 19.1 to 11.2 ns/row — same arithmetic, 1.7x from layout
alone. At that point it does the whole column's work 2.9x faster than NumPy needs
for the same set of quantities, and 4.9x if the geometric mean is skipped.

### Distribution decisions worth keeping

- **Separate package, not a separate build backend.** `pysuricata` stays a
  pure-Python setuptools wheel; `pysuricata-core` is a maturin wheel;
  `pysuricata[fast]` joins them. Making the root project maturin-built would force
  a per-platform matrix on every release and break the current one-step CD. This
  is what Polars does — its top-level PyPI package is a 0.8 MB pure-Python
  dispatcher over separate `polars-runtime-*` packages.
- **abi3 while you can.** One wheel per platform instead of one per Python minor.
  For reference: pydantic-core ships 137 wheels totalling 277 MB because it needs
  version-specific CPython internals; tokenizers ships 17 abi3 wheels at 75 MB.
  Caveat: abi3 does not yet exist for free-threaded builds — 3.13t/3.14t need
  version-specific wheels today.
- **The Python path stays forever**, as the reference implementation the oracle
  diffs against. The moment it is deleted, "is the Rust right?" becomes
  unanswerable.

### Prerequisites in the Python side

1. Move private attribute reads out of the engine. It reaches into
   `acc._bytes_seen`, `acc._uniques.estimate()`, `acc._min_ts`
   (`engine.py:371`, `render/html.py:80,100`). Put them behind properties first.
2. Replace the `isinstance` dispatch in the consume loop
   (`consume_polars.py:170,193,200`) with a `kind` tag, so a native accumulator
   does not have to subclass the Python class.
3. Give the native types `__reduce__` — `KmvSketch` has one, `NumericKernel` needs
   one, because `engine.py:377` pickles accumulators for checkpointing.
4. Add the categorical kernel. Biggest remaining win: the current path calls
   `Series.tolist()`, allocating one Python object per row, before Misra-Gries
   sees anything. `hash_arrow_utf8` in the crate already hashes an Arrow UTF-8
   buffer in place.

**Exit criterion:** `pip install pysuricata[fast]` works on
Linux/macOS/Windows x x86-64/arm64, and the accuracy suite passes identically with
both backends.

---

## Phase 4 — Report v2 (3-4 weekends)

The `titanic_report.html` example is **1.18 MB for 891 rows x 12 columns**, of
which ~855 KB is fixed cost independent of the data:

| Component | Bytes | Share |
|---|---:|---:|
| base64 PNG logos + favicon | 578,278 | 49% |
| CSS | 210,218 | 18% |
| inline SVG | 181,627 | 15% |
| markup | 138,500 | 12% |
| JS | 67,257 | 6% |

In order of value per hour:

1. **Drop the base64 PNGs** for inline SVG. -576 KB, one afternoon, zero risk.
2. **Emit a JSON payload.** `report.py::_build_summary` already produces a
   JSON-safe per-column mapping — it is what `summarize()` returns. Nothing in
   `render/` imports `json` today. Adding
   `<script type="application/json">` unlocks everything below.
3. **Stop pre-rendering six histograms per column.** `card_config.py:23` emits 3
   bin options x 2 scales, all hidden with CSS. 100 numeric columns is 600 SVGs.
   Draw one from the payload and re-bin client-side.
4. **Real theming.** Dark is currently the default with `.light` as the override,
   using manually paired `--x-light`/`--x-dark` tokens and no
   `prefers-color-scheme` anywhere. Move to `[data-theme]` + a media-query
   default, collapse the pairs into one semantic scale, sweep the ~310 stray hex
   values (`_12-missing.css` alone has 121).
5. **Chart colours in CSS, not Python.** `histogram_svg.py:33-40` bakes `#3b82f6`
   into `fill=` attributes, so themes never reach the charts.
6. **Replace `str.replace` templating.** `render/html.py:301-339` runs sequential
   replacements over the whole document, so a replacement value containing another
   placeholder gets re-expanded, and a column named `{n_rows}` collides.

**Constraint to design around:** the self-download button
(`functionality.js:25-65`) re-serialises the live DOM, finding styles by the regex
`/#pysuricata-report|suricata-standalone/` and the script by `/toggleDarkMode/`.
Any restructuring — external assets, ES modules, a renamed function, scoped styles
— breaks it *silently*. Write a test that downloads and re-opens the report before
touching the render layer.

**Two additions that show statistical taste:**

- **Show the uncertainty.** "~12,400 distinct (+/-3%)" is more trustworthy than
  "12,437", and the native `KmvSketch` already returns its relative standard
  error. No competitor does this, because none of them have an error bound to
  report.
- **Make the histogram honest.** The streaming histogram redistributes counts by
  bin centre whenever the range expands (`sketches.py:658`), which is lossy in a
  way the reader cannot see. Either switch to a KLL/t-digest sketch with a real
  error bound, or label the chart as estimated from an n-value sample.

**Exit criterion:** the Titanic report is under 250 KB, charts re-render on bin
change without a page of hidden SVG, and the report respects OS theme.

---

## Phase 5 — Differentiate (ongoing)

In priority order:

1. **`pysuricata check`** — a CI gate with an exit code and a thresholds file.
2. **`compare(df_a, df_b)`** — drift. The accumulators are mergeable, which makes
   this cheap.
3. **Direct Parquet / Arrow / DuckDB input** without a pandas round-trip.
4. **Column-level threading**, once the RNG is per-accumulator.

**Exit criterion:** at least one thing PySuricata does that no incumbent does,
documented on the front page.

---

## Market, as of August 2026

The incumbent layer is unusually unsettled. In the last twelve months:
ydata-profiling was renamed to **fg-data-profiling** and transferred to a new
GitHub org with an EOL notice on the old package; Great Expectations' OSS
stewardship moved to **Fivetran** (May 2026); and Soda Core **relicensed from
Apache-2.0 to the source-available Elastic License 2.0** (January 2026),
triggering a Canonical fork.

| Tool | Stars | Downloads/mo | Backend | State |
|---|---:|---:|---|---|
| ydata / fg-data-profiling | 13.7k | ~1.76M | pandas, Spark | renamed + transferred; ~20 runtime deps incl. numba |
| great_expectations | 11.5k | ~26.9M | pandas, Spark, SQL | Fivetran stewardship; validation, not profiling |
| evidently | 7.5k | ~1.26M | pandas | active — ML/LLM drift |
| dtale | 5.2k | ~53.6k | pandas | active — interactive GUI, different product |
| lux | 5.4k | ~1.4k | pandas | abandoned (no release since Feb 2022) |
| pandasgui | 3.3k | ~3.2k | pandas | likely abandoned |
| sweetviz | 3.1k | ~154k | pandas | active but narrow — owns dataset *comparison* |
| soda-core | 2.4k | ~3.5M | 12 warehouses | no longer OSI open source |
| dataprep | 2.2k | ~9.2k | pandas, Dask | abandoned (no release since Aug 2022) |
| skimpy | 517 | ~12.7k | pandas + polars | active — console only, no HTML report |
| pointblank (Posit) | 455 | ~28.9k | Narwhals + Ibis | very active; CI-first CLI |
| **pysuricata** | 7 | ~59 | pandas + polars | the only streaming/bounded-memory profiler |

### What users complain about

- **"Pandas profiling becoming too slow : un-usable"** — ydata #743, open since
  March 2021. "With 10k rows and 30 columns, it takes more than 2mins."
- **"MemoryError: Unable to allocate 20.0 PiB"** — ydata #1435, on a *10-row*
  DataFrame. And #1597: 1.72 TiB requested for a 5.6M x 9 frame on a 256 GB
  machine. Their own docs list this as a known, unfixed class of bug with "filter
  out large outliers first" as the workaround.
- **Dependency collisions** — yfinance #2464: "[0.2.59] breaks ydata-profiling
  dependency (numba)". An unrelated package's release broke installs for everyone
  with ydata-profiling in the same requirements file.
- **Polars support** — ydata #1129, opened October 2022, still labelled
  `needs-triage`.

### Gaps

| Gap | Status elsewhere | PySuricata |
|---|---|---|
| Bounded-memory / out-of-core profiling | Asked for since 2016 (pandas-profiling #26, #57). No incumbent does single-pass streaming. | Architecture already there; needs the generator bug fixed and a headline number |
| Polars-native end to end | Only skimpy (console) and pointblank (validation) | Shipped — DataFrame + LazyFrame |
| Profiler as a CI gate | GX / Soda / pointblank all gate CI, but all require authoring expectations first. No profiler gates on shape alone. | Half-built: `summarize()` exists; no CLI exit code, no thresholds file |
| PII detection in OSS | ydata has it, paywalled behind YData Fabric. OSS core has none. | Open |
| Dataset comparison / drift | sweetviz for pandas; evidently at the monitoring layer. The leader has no built-in compare. | Open — accumulators are mergeable, so this is cheap |
| Install weight | ydata pulls ~20 packages including numba/llvmlite | 4 deps, and `psutil` appears unused — drop it |

---

## Positioning

Current PyPI description: "A lightweight EDA tool inspired by the curious nature
of suricates. Built just for fun." Charming, and why nobody installs it.

> **The data profiler that fits in CI.** Single-pass streaming algorithms, bounded
> memory regardless of dataset size, pandas and polars native, four dependencies,
> one self-contained HTML file — or JSON and an exit code.

Every clause there is something an incumbent cannot say. "Faster than
ydata-profiling" is not — anyone can claim that, and it invites a benchmark
argument you win narrowly. "Profiles a dataset larger than your RAM in constant
memory" is a categorical difference.

**Lead with memory, not speed:**

- It follows from the architecture rather than from tuning, so it cannot be closed
  by the incumbent shipping a patch.
- The incumbent's most-reported failure is a MemoryError, which they document as
  unfixed.
- It makes the CI story credible: a profiler that runs in a 512 MB runner on a
  40 GB Parquet file is a genuinely new thing.
- Speed follows anyway. It is the second sentence.

### Two adoption paths, both real

uv (753 points on HN), DuckDB (926), marimo (448) and Polars (238) all had loud
moments built on one benchmarkable number stated everywhere. But **narwhals never
cleared 4 points on HN** and is now a declared dependency of 28 projects including
pandera, plotly, altair and scikit-lego — adoption through direct maintainer
outreach and being useful infrastructure.

Worth knowing: **no EDA or profiling tool has ever cleared 50 points on Hacker
News.** Either the category has never had a benchmark-led launch and you would be
first, or people install profilers and forget them rather than evangelising them.
Plan for both. The narwhals move here is to make `summarize()` a stable,
documented, versioned JSON contract other tools can build on.

---

## Writing plan

Eight posts, each publishable the weekend the corresponding work lands, each with
a concrete artefact behind it.

| # | Post | Angle | After phase |
|---|---|---|---|
| 1 | I wrote one test file and found six statistical bugs in my own library | The chunked-vs-unchunked invariant, the "simplified for performance" comment, the reservoir bias. Builds trust before it makes a claim. | 0 |
| 2 | Your profiler is spending a third of its time on SHA-1 | Why a cryptographic hash ended up in a distinct-count sketch, and a 755x measurement. Generalises past this library. | 1 |
| 3 | Profiling a dataset larger than RAM, in constant memory | The architecture piece. Welford/Pébay, KMV, Misra-Gries, Algorithm L, and a flat memory curve. The positioning post. | 2 |
| 4 | My one-pass Rust kernel was slower than NumPy's eleven passes | The best one. Dependency chains, division latency, and how 32 KiB tiling turned it into a win. CS:APP ch. 5-6 with a measurement at each step. | 3 |
| 5 | Shipping a Rust extension without breaking `pip install` | Separate `-core` package, abi3, the wheel matrix. Evergreen, and what people actually get stuck on. | 3 |
| 6 | A 1.2 MB HTML report for 891 rows — where it all went | Byte-level teardown and diet. | 4 |
| 7 | Data profiling as a CI check | Gate a PR on shape drift without authoring an expectation. Aim at the dbt/Airflow audience. | 5 |
| 8 | Reporting uncertainty in a data profile | Sketch error bounds; why every other tool reports estimates as facts. | 4-5 |

**Where:** own site first (canonical), then cross-post — never Medium as primary.
Show HN only for posts 3 and 4, the ones with a number in the title,
Tue-Thu 8-10am ET, one shot each. r/Python (1, 2, 4, 5), r/datascience (3, 7),
r/dataengineering (3, 7), Lobsters (2, 4, 5). LinkedIn as a technical summary with
the chart, not an announcement — the chart is what travels.

---

## CS:APP mapping

| Chapter | Task | The measurement |
|---|---|---|
| 2 — Information | The geometric mean in `moments.rs` accumulates a running *product* and renormalises with a hand-written `frexp` reading the exponent field out of the bit pattern, instead of one `ln()` per value. | `scan_numeric(arr, None, False)` vs `True`: 6.5 vs 11.2 ns/row |
| 3 — Machine code | Read the disassembly of the tiled scan (`cargo asm` / `objdump -d`). Confirm pass A vectorised to `addpd`/`maxpd` and pass B didn't spill. | Packed vs scalar instructions in the inner loop |
| 5 — Optimising | Why the one-pass kernel was slower. Welford has a loop-carried dependency with a division; four independent accumulator chains break it. | 19.1 -> 11.2 ns/row with identical arithmetic. Compute the CPE by hand and compare to the latency bound of `divsd`. |
| 6 — Memory hierarchy | The 32 KiB tile. Sweep `TILE` from 256 to 262144 and plot ns/row. | The sweep shows L1, L2 and L3 as inflections. Best figure for post #4, twenty minutes to produce. |
| 6 — roofline | `kernels.py`'s `% roofline` column. | 26.0 GB/s here. NumPy's multi-pass column scan at 21%, the tiled scan at 4.7% — and knowing which means "optimise the code" vs "reduce the traffic". |
| 12 — Concurrency | `py.detach()` and per-column accumulation on a thread pool. The blocker is the global RNG, which is why the seeding bug is on the critical path. | Already measured: 91% parallel efficiency on 2 threads, `gil_released=True` |

Study habit worth adopting: for each chapter, add *one* benchmark to `kernels.py`
that demonstrates the chapter's central claim on this codebase. By the end you
have a benchmark suite, a set of figures, your study notes, and post #4.

---

## Reading the roofline column

`benchmarks/kernels.py` measures how fast the machine streams a float64 array,
then reports every kernel against that ceiling.

| Reading | Means | Do |
|---|---|---|
| **< 20% of roofline** | The instruction stream is the bottleneck. Nowhere near memory-limited. | Better code: a native kernel, fewer branches, more ILP. |
| **> 70% of roofline** | Saturating memory bandwidth for the traffic generated. | Generate less traffic. Fuse passes; nothing else will help. |

NumPy's multi-pass column scan sits at 21% — not badly written, just reading the
column many times over. The tiled native scan sits at 4.7%, doing the same work in
a fifth of the time with headroom left. That contrast *is* the argument for a
native core, stated as a number rather than a vibe — and it also says when to
stop.
