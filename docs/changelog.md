---
title: Changelog
description: Version history and release notes for PySuricata
---

# Changelog

All notable changes to PySuricata are documented here.

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
