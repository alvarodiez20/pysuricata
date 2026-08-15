---
title: Changelog
description: Version history and release notes for PySuricata
---

# Changelog

All notable changes to PySuricata are documented here.

## [0.0.40] - 2026-08-15

#43 and #91. The payload becomes a contract that is checked rather than
described, and the gate learns the failure every other check passes.

### Added
- **[A documented `summarize()` schema](summary-schema.md)** — every key, its type, which ones are estimates and with what error, and the stability policy. Adding a key does not change `schema_version`; renaming, removing, or changing the meaning or units of one does.
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
- **[Gating CI on drift](data-checks.md)**, with a GitHub Actions job.

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

## [0.0.26] - 2026-08-14

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

## [0.0.17] - 2026-02-28

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

## [0.0.14] - 2026-01-14

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

## [0.0.11] - 2025-12-XX

### Added
- Enhanced documentation with mathematical formulas
- Comprehensive examples gallery
- Detailed algorithm documentation (Welford, Pébay, KMV, Misra-Gries)

## Earlier Versions

See [GitHub Releases](https://github.com/alvarodiez20/pysuricata/releases) for complete history.
