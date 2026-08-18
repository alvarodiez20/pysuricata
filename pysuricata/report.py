"""Report orchestration for streaming EDA.

This module coordinates the end-to-end generation of a self-contained HTML EDA
report from in-memory data. It supports both pandas and polars through a small
engine adapter layer, enabling:

- Streaming computation over single DataFrames or iterables of chunks.
- Optional in-memory chunking for large DataFrames to control peak memory.
- Lightweight checkpointing (periodic pickle/HTML) for long-running jobs.
- Optional correlation chips for numeric columns (thresholded, top-k).

The core computation is handled by compact accumulator objects; rendering is
performed by the HTML renderer. This file focuses on orchestration: selecting
the engine adapter, wiring chunks, checkpointing, and delegating to metrics and
renderers.

Example:
  >>> import pandas as pd
  >>> from pysuricata.report import build_report
  >>> df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
  >>> html = build_report(df)
"""

from __future__ import annotations

import logging
import time
from typing import Any

# Checkpointing imports
# Processing imports
from .compute.analysis import RowKMV
from .compute.manifest import duplicate_fields
from .compute.orchestration.engine import StreamingEngine

# Core imports
from .config import EngineConfig
from .logger import SectionTimer as _SectionTimer

# Rendering imports
from .render.format_utils import human_bytes as _human_bytes
from .render.html import render_empty_html as _render_empty_html
from .render.html import render_html_snapshot as _render_html_snapshot
from .render.identifier import looks_like_identifier as _looks_like_identifier

# Version of the mapping returned by summarize() and carried on Report.stats.
#
# The payload has already drifted once without one: dataset["rows"] became
# dataset["rows_est"], which silently broke every documented example and would
# have broken every downstream consumer. Version it before anyone depends on it.
#
# The promise: adding a key does not change this number, because a consumer
# reading known keys is unaffected. Renaming or removing one bumps the major.
SUMMARY_SCHEMA_VERSION = 1

# The payload key each summary-dataclass field is published under, where the two
# names differ. Every other field is published under its own name.
#
# This lives in the source rather than only in the docs because a test walks it:
# `tests/test_summary_contract.py` asserts that every field of every summary
# dataclass is either published or listed below as deliberately withheld. Adding
# a statistic to an accumulator therefore forces a decision about the contract
# instead of quietly widening the gap between the HTML and the JSON -- which is
# how #24 (correlations) and #59 (numeric top values) both happened.
SUMMARY_FIELD_ALIASES = {
    "dtype_str": "dtype",
    "outliers_iqr": "outliers_iqr_est",
    "true_n": "true",
    "false_n": "false",
}

# Fields that are deliberately not published, with the reason. These are not
# statistics: they are the raw material a statistic was computed from, or a
# rendering detail.
SUMMARY_FIELDS_WITHHELD = {
    "name": "the column name is the key in the columns mapping",
    "sample_vals": "the reservoir itself, up to 20,000 values, not a statistic",
    "sample_ts": "the datetime reservoir, same reasoning",
    "sample_scale": "a scale factor the renderer applies to sampled counts",
    "chunk_metadata": "per-chunk bookkeeping used to draw the chunk strip",
    "corr_threshold": "an echo of the caller's own configuration",
    "hist_counts": "legacy field, superseded by true_histogram_counts",
    "len_hist": (
        "a render-shaped binning of the length reservoir, capped at a bin "
        "count chosen for how many bars a reader can take in. `avg_len` and "
        "`len_p90` are the statistics; publishing this would pin a "
        "presentation choice into a versioned payload"
    ),
}


def _as_public_error(message: str) -> Exception:
    """Turn an engine failure into an exception the caller can catch.

    Engine failures used to surface as `RuntimeError`, which escapes the
    `PySuricataError` hierarchy the public API promises -- so
    `except PySuricataError` did not catch the most common way to get an input
    wrong. The internal vocabulary goes too: "Adapter selection failed" is a
    sentence about our module layout, not about the caller's data.

    Args:
        message: The engine's error string.

    Returns:
        The exception to raise. Left as `PySuricataError` for anything not
        recognised, rather than guessed at.
    """
    from .api import PySuricataError, UnsupportedDataError

    if "Unsupported input type" in message or "Adapter selection failed" in message:
        _, _, detail = message.rpartition(":")
        # The engine reports `<class 'int'>`; say `int`.
        named = detail.strip().removeprefix("<class '").removesuffix("'>")
        return UnsupportedDataError(
            f"Cannot profile a source yielding {named}. Provide a pandas "
            "DataFrame, a polars DataFrame/LazyFrame, an Arrow table or reader, "
            "a DuckDB relation, a path to a .csv/.parquet/.json/.arrow/.xlsx "
            "file, or an iterable of DataFrame chunks."
        )
    return PySuricataError(message)


def _f(value: Any) -> float | None:
    """Coerce a statistic to a plain float, keeping None as None.

    The accumulators return numpy scalars in places, which are not JSON
    serialisable and compare unequal to their Python counterparts under `is`.
    A payload that has to be re-encoded by every consumer is not a contract.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN and the infinities are not values, and this payload is documented as
    # JSON-safe: `json.dumps(..., allow_nan=False)` rejects all three, and the
    # `NaN` Python emits by default is not JSON any other language will read.
    # A statistic that could not be computed is `null`, which every consumer
    # already has to handle.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


#: Keys that still mean something for a column holding no values.
#:
#: The rule is one line: **a count over an empty set is zero, a statistic over
#: an empty set is undefined** (#315). `count`, `missing` and `zeros` are
#: genuinely 0 for a column with nothing in it; `mean`, `entropy` and
#: `time_span_days` are not 0, they are unanswerable, and reporting 0.0 for
#: them invents a reading rather than declining to give one.
#:
#: The same argument covers the booleans. `mono_inc` came back `True` for an
#: empty column -- vacuously true, and a claim no reader wants to be handed.
#:
#: `dtype` and `source_timezone` stay because they are properties of the
#: *schema*, which a zero-row frame still has; that schema is the whole reason
#: this payload is worth producing. `approx` stays because it describes the
#: sketch rather than the data. The collections stay because empty is the
#: correct answer for them, and is different from null.
_DEFINED_WHEN_EMPTY = frozenset(
    {
        # identity
        "type",
        "dtype",
        "source_timezone",
        # counts, all legitimately zero
        "count",
        "missing",
        "unique_est",
        "inf",
        "zeros",
        "negatives",
        "true",
        "false",
        "mem_bytes",
        # a property of the sketch, not of the data
        "approx",
        # collections whose correct value is empty
        "corr_top",
        "top_values",
        "top_items",
        "min_items",
        "max_items",
        "true_histogram_counts",
        "true_histogram_edges",
        "by_hour",
        "by_dow",
        "by_month",
        "by_year",
    }
)


def _null_undefined_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    """Blank the statistics a column with no values cannot have (#315).

    Applies to every column kind, because all four fabricated the same way: an
    empty numeric column reported `min` and `mean` of `0.0`, an empty
    categorical one an `entropy` of `0.0`, an empty datetime one a
    `time_span_days` of `0.0`.

    A no-op for any column that saw a value, so the ordinary path is untouched.
    """
    if payload.get("count"):
        return payload
    return {
        key: (value if key in _DEFINED_WHEN_EMPTY else None)
        for key, value in payload.items()
    }


def _i(value: Any) -> int | None:
    """Coerce a statistic to a plain int, keeping None as None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ReportOrchestrator:
    """Orchestrates the end-to-end EDA report generation process.

    This class encapsulates the complex logic of building streaming EDA reports,
    breaking it down into focused, testable methods.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
    ):
        """Initialize the report orchestrator.

        Args:
            config: Engine configuration. If None, uses default configuration.
        """
        self.config = config or EngineConfig()
        self.logger = self._setup_logger()
        self.start_time = time.time()

        # Initialize services
        self.streaming_engine = StreamingEngine(logger=self.logger)

        # Processing state
        self.row_kmv = RowKMV()

    def _setup_logger(self) -> logging.Logger:
        """Configure and return the logger for this report generation."""
        logger = self.config.logger or logging.getLogger(__name__)
        logger.setLevel(self.config.log_level)
        return logger

    def _log_startup_info(self, source: Any) -> None:
        """Log startup information about the report generation."""
        source_info = (
            source
            if isinstance(source, str)
            else f"DataFrame{getattr(source, 'shape', '')}"
        )

        self.logger.info("Starting report generation: source=%s", source_info)
        self.logger.info(
            "chunk_size=%d, uniques_k=%d, numeric_sample_k=%d, topk_k=%d",
            self.config.chunk_size,
            self.config.uniques_k,
            self.config.numeric_sample_k,
            self.config.topk_k,
        )

    def _build_manifest_inputs(
        self, kinds, accs, first_columns
    ) -> tuple[Any, Any, int, int, Any]:
        """Build the manifest for final processing."""
        with _SectionTimer(
            self.logger, "Compute top-missing, duplicates & quick metrics"
        ):
            kinds_map = self._build_kinds_map(kinds, accs)
            col_order = self._compute_col_order(first_columns, kinds)
            n_rows, n_cols = self._compute_dataset_shape(kinds_map, self.row_kmv)
            miss_list = self._compute_top_missing(kinds_map)
            return kinds_map, col_order, n_rows, n_cols, miss_list

    def _build_kinds_map(self, kinds, accs) -> dict[str, tuple[str, Any]]:
        """Return name -> (kind, accumulator) map for all known columns."""
        return {
            **{name: ("numeric", accs[name]) for name in kinds.numeric if name in accs},
            **{
                name: ("categorical", accs[name])
                for name in kinds.categorical
                if name in accs
            },
            **{
                name: ("datetime", accs[name])
                for name in kinds.datetime
                if name in accs
            },
            **{name: ("boolean", accs[name]) for name in kinds.boolean if name in accs},
        }

    def _compute_top_missing(self, kinds_map) -> list[tuple[str, float, int]]:
        """Compute per-column missing percentage and counts, sorted descending by pct."""
        miss_list: list[tuple[str, float, int]] = []
        for name, (_kind, acc) in kinds_map.items():
            miss = int(getattr(acc, "missing", 0))
            cnt = int(getattr(acc, "count", 0)) + miss
            pct = (miss / cnt * 100.0) if cnt else 0.0
            miss_list.append((name, pct, miss))
        miss_list.sort(key=lambda t: t[1], reverse=True)
        return miss_list

    def _compute_col_order(self, first_columns, kinds) -> list[str]:
        """Prefer the original first chunk order when available; otherwise by kinds."""
        prefer = list(first_columns) if first_columns else []
        valid = set(kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean)
        return [c for c in prefer if c in valid] or (
            kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean
        )

    def _compute_dataset_shape(self, kinds_map, row_kmv) -> tuple[int, int]:
        """Return (n_rows, n_cols) for the dataset used by manifest/reporting.

        Rows are estimated from the row-KMV tracker; columns from the kinds map.
        """
        n_rows = int(getattr(row_kmv, "rows", 0))
        n_cols = int(len(kinds_map))
        return n_rows, n_cols

    def _apply_correlation_chips(self, accs, kinds, corr_est) -> None:
        """Process correlation chips and attach to numeric accumulators."""
        if corr_est is not None:
            # Every partner, not only the strong ones (#154, 5b.6).
            #
            # The per-column pane used to repeat the section-level empty state
            # inside a card -- `No significant correlations found` on a column
            # that has partners and simply has no strong ones. `Age` has
            # exactly two numeric partners in the Titanic frame, so listing
            # both is *complete* information in two rows.
            #
            # This is one of the two deliberate fact changes the invariance
            # harness names in `scripts/report_fingerprint.py`: below-threshold
            # pairs become visible. `corr_max_cols` still bounds the work, so
            # the list is at most one entry per numeric column tracked.
            top_map = corr_est.top_map(
                threshold=0.0,
                max_per_col=self.config.corr_max_cols,
            )
            for name in kinds.numeric:
                acc = accs.get(name)
                if acc is None:
                    continue
                if hasattr(acc, "set_corr_top"):
                    try:
                        acc.set_corr_top(top_map.get(name, []))
                    except Exception:
                        pass
                if hasattr(acc, "set_corr_threshold"):
                    try:
                        acc.set_corr_threshold(self.config.corr_threshold)
                    except Exception:
                        pass

    def _render_html(
        self,
        kinds,
        accs,
        first_columns,
        total_missing_cells,
        approx_mem_bytes,
        sample_section_html,
        report_title: str | None = None,
        chunk_metadata: list[tuple[int, int, int]] | None = None,
        corr_est: Any | None = None,
    ) -> str:
        """Render the final HTML report."""
        with _SectionTimer(self.logger, "Render final HTML"):
            return _render_html_snapshot(
                kinds=kinds,
                accs=accs,
                first_columns=first_columns,
                row_kmv=self.row_kmv,
                total_missing_cells=total_missing_cells,
                approx_mem_bytes=approx_mem_bytes,
                start_time=self.start_time,
                cfg=self.config,
                report_title=report_title,
                sample_section_html=sample_section_html,
                chunk_metadata=chunk_metadata,
                corr_est=corr_est,
            )

    def _build_summary(
        self,
        kinds_map: Any,
        col_order: Any,
        miss_list: Any,
        n_rows: int,
        n_cols: int,
        total_missing_cells: int,
        approx_mem_bytes: int = 0,
    ) -> dict | None:
        """Build the programmatic summary."""
        dataset_summary = {
            "rows_est": int(n_rows),
            "cols": int(n_cols),
            "missing_cells": int(total_missing_cells),
            "missing_cells_pct": (total_missing_cells / max(1, n_rows * n_cols) * 100.0)
            if (n_rows and n_cols)
            else 0.0,
            **duplicate_fields(self.row_kmv),
            "memory_bytes": int(approx_mem_bytes),
            "top_missing": [
                {"column": str(col), "pct": float(pct), "count": int(cnt)}
                for col, pct, cnt in (list(miss_list)[:5] if miss_list else [])
            ],
        }

        columns_summary: dict[str, dict[str, Any]] = {}
        for name in col_order:
            kind, acc = kinds_map[name]
            if kind == "numeric":
                s = acc.finalize()
                # The HTML names a key column; the payload has to as well, or a
                # tool built on summarize() sees a strictly poorer view than a
                # reader of the report does.
                is_identifier = _looks_like_identifier(s)
                columns_summary[name] = {
                    "type": "identifier" if is_identifier else "numeric",
                    "count": _i(s.count),
                    "missing": _i(s.missing),
                    "unique_est": _i(s.unique_est),
                    "mean": _f(s.mean),
                    "std": _f(s.std),
                    "min": _f(s.min),
                    "q1": _f(s.q1),
                    "median": _f(s.median),
                    "q3": _f(s.q3),
                    "max": _f(s.max),
                    "zeros": _i(s.zeros),
                    "negatives": _i(s.negatives),
                    "outliers_iqr_est": _i(s.outliers_iqr),
                    "approx": bool(s.approx),
                    "mem_bytes": _i(s.mem_bytes),
                    # Correlations are computed and applied before this point, but
                    # used to reach the HTML report only -- summarize() dropped
                    # them, so the JSON contract was strictly weaker than the HTML.
                    "corr_top": [
                        (str(other), float(r)) for other, r in (s.corr_top or [])
                    ],
                    "mono_inc": bool(s.mono_inc),
                    "mono_dec": bool(s.mono_dec),
                    "int_like": bool(s.int_like),
                    # The HTML renders a "Common values" table from these; the
                    # payload used to omit them, so a tool built on summarize()
                    # saw strictly less than a reader of the report. None means
                    # "not tracked" -- the top-k sketch is gated off on columns
                    # too high-cardinality for the answer to mean anything --
                    # which is a different statement from an empty list.
                    "top_values": (
                        [(float(v), int(c)) for v, c in s.top_values]
                        if s.top_values
                        else ([] if acc.tracks_top_values else None)
                    ),
                    # Everything below is shown on the numeric card and used to
                    # be reachable only by reading the HTML.
                    "dtype": str(s.dtype_str),
                    "inf": int(s.inf),
                    "variance": _f(s.variance),
                    "skew": _f(s.skew),
                    "kurtosis": _f(s.kurtosis),
                    "cv": _f(s.cv),
                    "se": _f(s.se),
                    "gmean": _f(s.gmean),
                    # None when the column has no positive value at all, which
                    # is a different statement from 0.0 and the reason this is
                    # not passed through `_f`.
                    "min_positive": (
                        _f(s.min_positive) if s.min_positive is not None else None
                    ),
                    "iqr": _f(s.iqr),
                    "mad": _f(s.mad),
                    "ci_lo": _f(s.ci_lo),
                    "ci_hi": _f(s.ci_hi),
                    "jb_chi2": _f(s.jb_chi2),
                    "outliers_mod_zscore": int(s.outliers_mod_zscore),
                    "heap_pct": _f(s.heap_pct),
                    "bimodal": bool(s.bimodal),
                    "gran_decimals": _i(s.gran_decimals),
                    "gran_step": _f(s.gran_step),
                    "unique_ratio_approx": _f(s.unique_ratio_approx),
                    # The card lists these as "Extreme values", with the row
                    # each one sits at.
                    "min_items": [
                        (int(idx), float(val)) for idx, val in (s.min_items or [])
                    ],
                    "max_items": [
                        (int(idx), float(val)) for idx, val in (s.max_items or [])
                    ],
                    "true_histogram_edges": [
                        float(e) for e in (s.true_histogram_edges or [])
                    ],
                    "true_histogram_counts": [
                        int(c) for c in (s.true_histogram_counts or [])
                    ],
                }
            elif kind == "categorical":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "categorical",
                    "count": _i(s.count),
                    "missing": _i(s.missing),
                    "unique_est": _i(s.unique_est),
                    "top_items": [(str(v), _i(c)) for v, c in (s.top_items or [])],
                    "approx": bool(s.approx),
                    "mem_bytes": _i(s.mem_bytes),
                    "dtype": str(s.dtype_str),
                    "entropy": _f(s.entropy),
                    "avg_len": _f(s.avg_len),
                    "len_p90": _i(s.len_p90),
                    "empty_zero": int(s.empty_zero),
                    # The two variant estimates drive the "looks like a case or
                    # whitespace variant of another value" quality flags.
                    "case_variants_est": _i(s.case_variants_est),
                    "trim_variants_est": _i(s.trim_variants_est),
                    "most_common_ratio": _f(s.most_common_ratio),
                    "diversity_ratio": _f(s.diversity_ratio),
                    "gini_impurity": _f(s.gini_impurity),
                }
            elif kind == "datetime":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "datetime",
                    "count": _i(s.count),
                    "missing": _i(s.missing),
                    "min_ts": _f(s.min_ts),
                    "max_ts": _f(s.max_ts),
                    "mem_bytes": _i(s.mem_bytes),
                    "dtype": str(s.dtype_str),
                    "unique_est": _i(s.unique_est),
                    "mono_inc": bool(s.mono_inc),
                    "mono_dec": bool(s.mono_dec),
                    "time_span_days": _f(s.time_span_days),
                    "avg_interval_seconds": _f(s.avg_interval_seconds),
                    "interval_std_seconds": _f(s.interval_std_seconds),
                    "weekend_ratio": _f(s.weekend_ratio),
                    "business_hours_ratio": _f(s.business_hours_ratio),
                    "seasonal_pattern": (
                        str(s.seasonal_pattern) if s.seasonal_pattern else None
                    ),
                    "source_timezone": (
                        str(s.source_timezone) if s.source_timezone else None
                    ),
                    # The tallies behind the hour/day/month/year charts. Small
                    # and fixed in size -- 24, 7, 12, and one entry per year.
                    "by_hour": [int(v) for v in (s.by_hour or [])],
                    "by_dow": [int(v) for v in (s.by_dow or [])],
                    "by_month": [int(v) for v in (s.by_month or [])],
                    "by_year": {int(y): int(c) for y, c in (s.by_year or {}).items()},
                }
            else:  # boolean
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "boolean",
                    "count": _i(s.count),
                    "missing": _i(s.missing),
                    "true": _i(s.true_n),
                    "false": _i(s.false_n),
                    "mem_bytes": _i(s.mem_bytes),
                    "dtype": str(s.dtype_str),
                    "true_ratio": _f(s.true_ratio),
                    "false_ratio": _f(s.false_ratio),
                    "entropy": _f(s.entropy),
                }

        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "dataset": dataset_summary,
            "columns": {
                name: _null_undefined_statistics(payload)
                for name, payload in columns_summary.items()
            },
        }

    def _write_output_file(self, html: str, output_file: str) -> None:
        """Write the HTML report to a file."""
        with _SectionTimer(self.logger, f"Write HTML to {output_file}"):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html)

            self.logger.info(
                "report written: %s (%s)",
                output_file,
                _human_bytes(len(html.encode("utf-8"))),
            )

    def _log_completion(self) -> None:
        """Log completion information."""
        elapsed_time = time.time() - self.start_time
        self.logger.info("Report generation complete in %.2fs", elapsed_time)

    def build_report(
        self,
        source: Any,
        *,
        output_file: str | None = None,
        report_title: str | None = None,
        return_summary: bool = False,
        compute_only: bool = False,
    ) -> str | tuple[str, dict]:
        """Build a streaming EDA report from in-memory data.

        This method orchestrates the complete report generation process:
        1. Setup and configuration
        2. First chunk processing and pipeline setup
        3. Stream processing of remaining chunks
        4. Manifest building and correlation processing
        5. HTML rendering and summary generation
        6. Output handling

        Args:
            source: Input data (pandas/polars DataFrame or iterable of chunks)
            output_file: Optional path to write the final HTML document
            report_title: Optional title for the HTML report
            return_summary: If True, returns tuple (html, summary)
            compute_only: If True, skips HTML rendering

        Returns:
            HTML string or tuple (html, summary) if return_summary is True

        Raises:
            TypeError: If source is not a supported type
        """
        # Phase 1: Setup and configuration
        # No global seeding here: ``config.random_seed`` reaches the sketches as
        # a per-column generator built in the accumulator factory, so profiling
        # neither reads nor perturbs the caller's RNG.
        self._log_startup_info(source)

        # Phase 2: Process stream
        stream_result = self.streaming_engine.process_stream(
            source, self.config, self.row_kmv
        )

        if not stream_result.success:
            if "Empty source" in stream_result.error:
                # An empty input is a valid, boring case: the call succeeds and
                # returns a usable report. Announcing "Stream processing failed"
                # on it told the caller their successful call had failed -- and
                # in CI, where `pysuricata check` now puts this library on
                # purpose, a line containing "failed" on a green run is exactly
                # what gets grepped for.
                self.logger.debug("Empty source; rendering an empty report")
                html = _render_empty_html(self.config.title)
                if return_summary:
                    return html, {}
                return html
            self.logger.error("Stream processing failed: %s", stream_result.error)
            raise _as_public_error(stream_result.error)

        # Extract results from stream processing
        (
            kinds,
            accs,
            n_rows,
            n_cols,
            total_missing_cells,
            approx_mem_bytes,
            first_columns,
            sample_section_html,
            corr_est,
            chunk_metadata,
        ) = stream_result.data

        # Phase 3: Build manifest and process correlations
        kinds_map, col_order, n_rows, n_cols, miss_list = self._build_manifest_inputs(
            kinds, accs, first_columns
        )
        self._apply_correlation_chips(accs, kinds, corr_est)

        # Log top-missing columns
        self.logger.info(
            "top-missing columns: %s",
            ", ".join([c for c, _, _ in miss_list[:5]]) or "(none)",
        )

        # Phase 4: Render HTML and build summary
        html = ""
        if not compute_only:
            html = self._render_html(
                kinds,
                accs,
                first_columns,
                total_missing_cells,
                approx_mem_bytes,
                sample_section_html,
                report_title,
                chunk_metadata,
                corr_est,
            )

        summary_obj = self._build_summary(
            kinds_map,
            col_order,
            miss_list,
            n_rows,
            n_cols,
            total_missing_cells,
            approx_mem_bytes,
        )

        # Phase 5: Handle output
        if output_file and not compute_only:
            self._write_output_file(html, output_file)

        self._log_completion()

        # Return results
        if return_summary:
            return html, (summary_obj or {})
        return html


def build_report(
    source: Any,
    *,
    config: EngineConfig | None = None,
    output_file: str | None = None,
    report_title: str | None = None,
    return_summary: bool = False,
    compute_only: bool = False,
) -> str | tuple[str, dict]:
    """Build a streaming EDA report from in-memory data.

    This function orchestrates the complete report generation process:
    1. Setup and configuration
    2. First chunk processing and pipeline setup
    3. Stream processing of remaining chunks
    4. Manifest building and correlation processing
    5. HTML rendering and summary generation
    6. Output handling

    Args:
        source: Input data (pandas/polars DataFrame or iterable of chunks)
        config: Engine configuration. If None, uses default configuration.
        output_file: Optional path to write the final HTML document
        report_title: Optional title for the HTML report
        return_summary: If True, returns tuple (html, summary)
        compute_only: If True, skips HTML rendering

    Returns:
        HTML string or tuple (html, summary) if return_summary is True

    Raises:
        TypeError: If source is not a supported type

    Examples:
        Basic usage with pandas::

            >>> import pandas as pd
            >>> from pysuricata.report import build_report
            >>> df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
            >>> html = build_report(df)
            >>> assert "<html" in html.lower()

        Custom configuration::

            >>> from pysuricata.config import EngineConfig
            >>> config = EngineConfig(chunk_size=100_000, numeric_sample_k=10_000)
            >>> html = build_report(df, config=config)
    """
    orchestrator = ReportOrchestrator(config)
    return orchestrator.build_report(
        source=source,
        output_file=output_file,
        report_title=report_title,
        return_summary=return_summary,
        compute_only=compute_only,
    )
