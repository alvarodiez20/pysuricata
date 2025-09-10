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
from typing import Any, Optional, Tuple

import numpy as np

# Checkpointing imports
from .checkpoint import maybe_make_manager as _maybe_ckpt

# Processing imports
from .compute.analysis import RowKMV
from .compute.orchestration.engine import (
    consume_stream,
    get_adapter_tag,
    maybe_corr_estimator,
)
from .compute.orchestration.services import (
    ChunkingService,
    EngineService,
    MetricsService,
    ProcessingService,
    ResourceManager,
)

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
        # Dependency injection for testability
        chunking_service: Optional[ChunkingService] = None,
        engine_service: Optional[EngineService] = None,
        metrics_service: Optional[MetricsService] = None,
        resource_manager: Optional[ResourceManager] = None,
    ):
        """Initialize the report orchestrator.

        Args:
            config: Engine configuration. If None, uses default configuration.
            chunking_service: Chunking service instance for dependency injection
            engine_service: Engine service instance for dependency injection
            metrics_service: Metrics service instance for dependency injection
            resource_manager: Resource manager instance for dependency injection
        """
        self.config = config or EngineConfig()
        self.logger = self._setup_logger()
        self.start_time = time.time()

        # Initialize services with dependency injection
        self.chunking_service = chunking_service or ChunkingService(self.config)
        self.engine_service = engine_service or EngineService(self.config)
        self.metrics_service = metrics_service or MetricsService(self.config)
        self.resource_manager = resource_manager or ResourceManager(self.config)

        # Processing state
        self.adapter = None
        self.kinds = None
        self.accs = None
        self.corr_est = None
        self.row_kmv = RowKMV()

        # Metrics tracking
        self.total_missing_cells = 0
        self.approx_mem_bytes = 0
        self.n_rows = 0
        self.n_cols = 0
        self.chunk_idx = 0

        # Data state
        self.first_columns = []
        self.sample_section_html = ""

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

    def _process_first_chunk(self, first_chunk: Any) -> None:
        """Process the first chunk to set up the processing pipeline.

        Args:
            first_chunk: The first data chunk to process.

        Raises:
            TypeError: If the input type is not supported.
        """
        # Choose engine adapter using service
        adapter_result = self.engine_service.select_adapter(first_chunk)
        if not adapter_result.success:
            raise TypeError(adapter_result.error)
        self.adapter = adapter_result.data

        # Extract basic information from first chunk
        self._extract_chunk_info(first_chunk)

        # Infer column types and build accumulators
        with _SectionTimer(
            self.logger,
            f"{get_adapter_tag(self.adapter)} Infer kinds & build accumulators",
        ):
            self.kinds, self.accs = self.adapter.infer_and_build(
                first_chunk, self.config
            )

        # Set up correlation estimator
        self.corr_est = maybe_corr_estimator(self.kinds, self.config)

        # Process the first chunk
        with _SectionTimer(
            self.logger, f"{get_adapter_tag(self.adapter)} Consume first chunk"
        ):
            self.adapter.consume_chunk(first_chunk, self.accs, self.kinds, self.logger)
            if self.corr_est is not None:
                self.adapter.update_corr(first_chunk, self.corr_est, self.logger)

        # Log column type information
        self._log_column_types()

    def _extract_chunk_info(self, chunk: Any) -> None:
        """Extract basic information from a data chunk."""
        self.approx_mem_bytes += int(self.adapter.estimate_mem(chunk) or 0)
        self.adapter.update_row_kmv(chunk, self.row_kmv)
        self.total_missing_cells += int(self.adapter.missing_cells(chunk) or 0)
        self.first_columns = list(getattr(chunk, "columns", []))
        self.sample_section_html = self.adapter.sample_section_html(chunk, self.config)

        try:
            self.n_rows = int(getattr(chunk, "height", len(chunk)))
        except Exception:
            self.n_rows = 0
        self.n_cols = len(getattr(chunk, "columns", []))
        self.chunk_idx = 1

    def _log_column_types(self) -> None:
        """Log information about detected column types."""
        tag = get_adapter_tag(self.adapter)
        self.logger.info(
            f"{tag} kinds: %d numeric, %d categorical, %d datetime, %d boolean",
            len(self.kinds.numeric),
            len(self.kinds.categorical),
            len(self.kinds.datetime),
            len(self.kinds.boolean),
        )

    def _process_remaining_chunks(self, chunks) -> None:
        """Process all remaining chunks in the stream."""
        ckpt_mgr = _maybe_ckpt(self.config, None)  # output_file handled separately

        # Use engine service for stream consumption
        result = self.engine_service.consume_stream(
            adapter=self.adapter,
            chunks=chunks,
            accs=self.accs,
            kinds=self.kinds,
            logger=self.logger,
            corr_est=self.corr_est,
            row_kmv=self.row_kmv,
            approx_mem_bytes=self.approx_mem_bytes,
            total_missing_cells=self.total_missing_cells,
            first_columns=self.first_columns,
            sample_section_html=self.sample_section_html,
            cfg=self.config,
            ckpt_mgr=ckpt_mgr,
            start_time=self.start_time,
            report_title=None,  # handled separately
            n_rows=self.n_rows,
            chunk_idx=self.chunk_idx,
        )

        if not result.success:
            self.logger.error("Stream processing failed: %s", result.error)
            raise RuntimeError(result.error)

        self.n_rows, self.approx_mem_bytes, self.total_missing_cells, self.chunk_idx = (
            result.data
        )

    def _build_manifest(self) -> Tuple[Any, Any, int, int, Any]:
        """Build the manifest for final processing."""
        with _SectionTimer(
            self.logger, "Compute top-missing, duplicates & quick metrics"
        ):
            result = self.metrics_service.build_manifest_inputs(
                kinds=self.kinds,
                accs=self.accs,
                row_kmv=self.row_kmv,
                first_columns=self.first_columns,
            )

            if not result.success:
                self.logger.error("Manifest building failed: %s", result.error)
                raise RuntimeError(result.error)

            return result.data

    def _process_correlations(self) -> None:
        """Process correlation chips and attach to numeric accumulators."""
        if self.corr_est is not None:
            top_map = self.corr_est.top_map(
                threshold=self.config.corr_threshold,
                max_per_col=self.config.corr_max_per_col,
            )

            result = self.metrics_service.apply_correlation_chips(
                accs=self.accs,
                kinds=self.kinds,
                top_map=top_map,
            )

            if not result.success:
                self.logger.warning("Correlation processing failed: %s", result.error)

    def _render_html(self, report_title: Optional[str] = None) -> str:
        """Render the final HTML report."""
        with _SectionTimer(
            self.logger, f"{get_adapter_tag(self.adapter)} Render final HTML"
        ):
            return _render_html_snapshot(
                kinds=self.kinds,
                accs=self.accs,
                first_columns=self.first_columns,
                row_kmv=self.row_kmv,
                total_missing_cells=self.total_missing_cells,
                approx_mem_bytes=self.approx_mem_bytes,
                start_time=self.start_time,
                cfg=self.config,
                report_title=report_title,
                sample_section_html=self.sample_section_html,
            )

    def _build_summary(
        self, kinds_map: Any, col_order: Any, miss_list: Any
    ) -> Optional[dict]:
        """Build the programmatic summary."""
        result = self.metrics_service.build_summary(
            kinds_map=kinds_map,
            col_order=col_order,
            row_kmv=self.row_kmv,
            total_missing_cells=self.total_missing_cells,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            miss_list=miss_list,
        )

        if not result.success:
            self.logger.warning("Failed to build summary: %s", result.error)
            return None

        return result.data

    def _write_output_file(self, html: str, output_file: str) -> None:
        """Write the HTML report to a file."""
        with _SectionTimer(
            self.logger, f"{get_adapter_tag(self.adapter)} Write HTML to {output_file}"
        ):
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

        # Phase 2: Build chunk iterator and process first chunk
        with _SectionTimer(self.logger, "Build chunk iterator"):
            chunk_result = self.chunking_service.chunks_from_source(
                source,
                int(self.config.chunk_size),
                bool(getattr(self.config, "force_chunk_in_memory", False)),
            )

            if not chunk_result.success:
                self.logger.error("Chunking failed: %s", chunk_result.error)
                raise RuntimeError(chunk_result.error)

            chunks = chunk_result.data

        with _SectionTimer(self.logger, "Read first chunk"):
            try:
                first_chunk = next(chunks)
            except StopIteration:
                self.logger.warning("Empty source; nothing to report")
                html = _render_empty_html(self.config.title)
                if return_summary:
                    return html, {}
                return html

        self._process_first_chunk(first_chunk)

        # Phase 3: Process remaining chunks
        self._process_remaining_chunks(chunks)

        # Phase 4: Build manifest and process correlations
        kinds_map, col_order, self.n_rows, self.n_cols, miss_list = (
            self._build_manifest()
        )
        self._process_correlations()

        # Log top-missing columns
        self.logger.info(
            "top-missing columns: %s",
            ", ".join([c for c, _, _ in miss_list[:5]]) or "(none)",
        )

        # Phase 5: Render HTML and build summary
        html = ""
        if not compute_only:
            html = self._render_html(report_title)

        summary_obj = self._build_summary(kinds_map, col_order, miss_list)

        # Phase 6: Handle output
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
