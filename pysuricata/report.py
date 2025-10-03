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
import random as _py_random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Checkpointing imports
from .accumulators.protocols import FinalizableAccumulator
from .checkpoint import maybe_make_manager as _maybe_ckpt

# Processing imports
from .compute.analysis import RowKMV
from .compute.orchestration.engine import StreamingEngine

# Core imports
from .config import EngineConfig
from .logger import SectionTimer as _SectionTimer

# Rendering imports
from .render.format_utils import human_bytes as _human_bytes
from .render.html import (
    render_empty_html as _render_empty_html,
)
from .render.html import (
    render_html_snapshot as _render_html_snapshot,
)

# Module-level RNG seed used by public SVG helpers
_REPORT_RANDOM_SEED: int = 0


class ReportOrchestrator:
    """Orchestrates the end-to-end EDA report generation process.

    This class encapsulates the complex logic of building streaming EDA reports,
    breaking it down into focused, testable methods.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
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

    def _setup_random_seeds(self) -> None:
        """Configure random seeds for reproducible results."""
        if self.config.random_seed is not None:
            try:
                seed = int(self.config.random_seed)
                np.random.seed(seed)
                _py_random.seed(seed)
                global _REPORT_RANDOM_SEED
                _REPORT_RANDOM_SEED = seed
            except Exception as e:
                self.logger.warning("Failed to set random seed: %s", e)

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
    ) -> Tuple[Any, Any, int, int, Any]:
        """Build the manifest for final processing."""
        with _SectionTimer(
            self.logger, "Compute top-missing, duplicates & quick metrics"
        ):
            kinds_map = self._build_kinds_map(kinds, accs)
            col_order = self._compute_col_order(first_columns, kinds)
            n_rows, n_cols = self._compute_dataset_shape(kinds_map, self.row_kmv)
            miss_list = self._compute_top_missing(kinds_map)
            return kinds_map, col_order, n_rows, n_cols, miss_list

    def _build_kinds_map(self, kinds, accs) -> Dict[str, Tuple[str, Any]]:
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

    def _compute_top_missing(self, kinds_map) -> List[Tuple[str, float, int]]:
        """Compute per-column missing percentage and counts, sorted descending by pct."""
        miss_list: List[Tuple[str, float, int]] = []
        for name, (_kind, acc) in kinds_map.items():
            miss = int(getattr(acc, "missing", 0))
            cnt = int(getattr(acc, "count", 0)) + miss
            pct = (miss / cnt * 100.0) if cnt else 0.0
            miss_list.append((name, pct, miss))
        miss_list.sort(key=lambda t: t[1], reverse=True)
        return miss_list

    def _compute_col_order(self, first_columns, kinds) -> List[str]:
        """Prefer the original first chunk order when available; otherwise by kinds."""
        prefer = list(first_columns) if first_columns else []
        valid = set(kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean)
        return [c for c in prefer if c in valid] or (
            kinds.numeric + kinds.categorical + kinds.datetime + kinds.boolean
        )

    def _compute_dataset_shape(self, kinds_map, row_kmv) -> Tuple[int, int]:
        """Return (n_rows, n_cols) for the dataset used by manifest/reporting.

        Rows are estimated from the row-KMV tracker; columns from the kinds map.
        """
        n_rows = int(getattr(row_kmv, "rows", 0))
        n_cols = int(len(kinds_map))
        return n_rows, n_cols

    def _apply_correlation_chips(self, accs, kinds, corr_est) -> None:
        """Process correlation chips and attach to numeric accumulators."""
        if corr_est is not None:
            top_map = corr_est.top_map(
                threshold=self.config.corr_threshold,
                max_per_col=self.config.corr_max_per_col,
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

    def _render_html(
        self,
        kinds,
        accs,
        first_columns,
        total_missing_cells,
        approx_mem_bytes,
        sample_section_html,
        report_title: Optional[str] = None,
        chunk_metadata: Optional[List[Tuple[int, int, int]]] = None,
    ) -> str:
        """Render the final HTML report."""
        with _SectionTimer(self.logger, f"Render final HTML"):
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
            )

    def _build_summary(
        self,
        kinds_map: Any,
        col_order: Any,
        miss_list: Any,
        n_rows: int,
        n_cols: int,
        total_missing_cells: int,
    ) -> Optional[dict]:
        """Build the programmatic summary."""
        dataset_summary = {
            "rows_est": int(n_rows),
            "cols": int(n_cols),
            "missing_cells": int(total_missing_cells),
            "missing_cells_pct": (total_missing_cells / max(1, n_rows * n_cols) * 100.0)
            if (n_rows and n_cols)
            else 0.0,
            "duplicate_rows_est": int(self.row_kmv.approx_duplicates()[0])
            if hasattr(self.row_kmv, "approx_duplicates")
            else 0,
            "duplicate_rows_pct_est": float(self.row_kmv.approx_duplicates()[1])
            if hasattr(self.row_kmv, "approx_duplicates")
            else 0.0,
            "top_missing": [
                {"column": str(col), "pct": float(pct), "count": int(cnt)}
                for col, pct, cnt in (list(miss_list)[:5] if miss_list else [])
            ],
        }

        columns_summary: Dict[str, Dict[str, Any]] = {}
        for name in col_order:
            kind, acc = kinds_map[name]
            if kind == "numeric":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "numeric",
                    "count": s.count,
                    "missing": s.missing,
                    "unique_est": s.unique_est,
                    "mean": s.mean,
                    "std": s.std,
                    "min": s.min,
                    "q1": s.q1,
                    "median": s.median,
                    "q3": s.q3,
                    "max": s.max,
                    "zeros": s.zeros,
                    "negatives": s.negatives,
                    "outliers_iqr_est": s.outliers_iqr,
                    "approx": bool(s.approx),
                }
            elif kind == "categorical":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "categorical",
                    "count": s.count,
                    "missing": s.missing,
                    "unique_est": s.unique_est,
                    "top_items": s.top_items,
                    "approx": bool(s.approx),
                }
            elif kind == "datetime":
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "datetime",
                    "count": s.count,
                    "missing": s.missing,
                    "min_ts": s.min_ts,
                    "max_ts": s.max_ts,
                }
            else:  # boolean
                s = acc.finalize()
                columns_summary[name] = {
                    "type": "boolean",
                    "count": s.count,
                    "missing": s.missing,
                    "true": s.true_n,
                    "false": s.false_n,
                }

        return {"dataset": dataset_summary, "columns": columns_summary}

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
        output_file: Optional[str] = None,
        report_title: Optional[str] = None,
        return_summary: bool = False,
        compute_only: bool = False,
    ) -> str | Tuple[str, dict]:
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
        self._setup_random_seeds()
        self._log_startup_info(source)

        # Phase 2: Process stream
        stream_result = self.streaming_engine.process_stream(
            source, self.config, self.row_kmv
        )

        if not stream_result.success:
            self.logger.error("Stream processing failed: %s", stream_result.error)
            if "Empty source" in stream_result.error:
                html = _render_empty_html(self.config.title)
                if return_summary:
                    return html, {}
                return html
            raise RuntimeError(stream_result.error)

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
            )

        summary_obj = self._build_summary(
            kinds_map, col_order, miss_list, n_rows, n_cols, total_missing_cells
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
    config: Optional[EngineConfig] = None,
    output_file: Optional[str] = None,
    report_title: Optional[str] = None,
    return_summary: bool = False,
    compute_only: bool = False,
) -> str | Tuple[str, dict]:
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
