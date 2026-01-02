---
title: Changelog
description: Version history and release notes for PySuricata
---

# Changelog

All notable changes to PySuricata are documented here.

## [0.0.13] - 2026-01-02

### Added
- **CLI tool** - New command-line interface with `pysuricata profile` and `pysuricata summarize` commands
- **Comprehensive stress tests** - New `test_complexity_analysis.py` with time/space profiling

### Fixed
- **Memory leak fixes** - Resolved memory leaks in KMV sketch, ExtremeTracker, and chunk metadata
- **Documentation accuracy** - Updated all performance claims to reflect actual benchmark results

### Changed
- **Realistic benchmarks** - Updated README and docs with measured performance figures:
  - 1M rows × 10 columns: ~3 minutes (was incorrectly claimed as 15 seconds)
  - Peak memory: ~50MB (verified accurate)
  - Throughput: ~5,500 rows/second

## [0.0.11] - 2025-12-XX

### Added
- Enhanced documentation with mathematical formulas for all variable types
- Comprehensive examples gallery
- Detailed algorithm documentation (Welford, Pébay, KMV, Misra-Gries)
- CI/CD documentation checks

### Improved
- Documentation structure with logical navigation
- Missing values analysis documentation
- Configuration guide with all parameters

## [0.0.10] - Previous Release

### Features
- Streaming correlation computation
- Missing values chunk-level tracking
- Enhanced datetime analysis
- Boolean variable profiling

## Earlier Versions

See [GitHub Releases](https://github.com/alvarodiez20/pysuricata/releases) for complete history.

---

*Last updated: 2026-01-02*
