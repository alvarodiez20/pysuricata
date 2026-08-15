---
title: Changelog
description: Version history and release notes for PySuricata
---

# Changelog

All notable changes to PySuricata are documented here.

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
