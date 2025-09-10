from __future__ import annotations

"""Engine adapters and streaming loop for the report orchestrator.

This module abstracts backend-specific logic (pandas vs polars) behind a small
adapter protocol so the orchestration in ``report.build_report`` can remain
backend-agnostic and easy to test. It also provides a unified streaming loop
with optional checkpointing and lightweight correlation updates.

Design goals:
  - Avoid importing heavy, optional dependencies at module import time.
  - Keep per-backend logic localized to small adapters.
  - Prefer structural typing (Protocols) over concrete types for flexibility.
  - Ensure small, testable pure helpers (e.g., ``prep_first_chunk``).
"""

from typing import Any, Dict, Iterator, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ..accumulators import (
    BooleanAccumulatorV2 as BooleanAccumulator,
)
from ..accumulators import (
    CategoricalAccumulatorV2 as CategoricalAccumulator,
)
from ..accumulators import (
    DatetimeAccumulatorV2 as DatetimeAccumulator,
)
from ..accumulators import (
    NumericAccumulatorV2 as NumericAccumulator,
)
from ..accumulators.factory import build_accumulators as _build_accumulators
from ..checkpoint import make_state_snapshot as _make_state_snapshot
from ..render.html import render_html_snapshot as _render_html_snapshot
from ..render.sections import render_sample_section as _render_sample_section
from .consume import consume_chunk_pandas as _consume_chunk_pandas
from .consume_polars import consume_chunk_polars as _consume_chunk_polars
from .core.types import ColumnKinds
from .processing.inference import UnifiedTypeInferrer


class EngineAdapter(Protocol):
    """Backend adapter interface for pandas/polars implementations.

    Methods are intentionally narrow to keep coordination logic in the
    orchestrator while allowing backends to optimize ingestion and updates.
    """

    def infer_and_build(
        self, first: Any, cfg: Any
    ) -> Tuple[ColumnKinds, Dict[str, Any]]:
        """Infer column kinds and create accumulators for the first chunk."""

    def estimate_mem(self, frame: Any) -> int:
        """Return an estimated memory footprint for a chunk/frame in bytes."""

    def update_row_kmv(self, frame: Any, row_kmv: Any) -> None:
        """Update the row-level distinct estimator from a chunk/frame."""

    def missing_cells(self, frame: Any) -> int:
        """Return the number of missing cells in a chunk/frame."""

    def consume_chunk(
        self, frame: Any, accs: Dict[str, Any], kinds: ColumnKinds, logger: Any
    ) -> None:
        """Wire the chunk into the appropriate accumulators."""

    def update_corr(self, frame: Any, corr_est: Any, logger: Any) -> None:
        """Update streaming correlations, if enabled."""

    def sample_section_html(self, first: Any, cfg: Any) -> str:
        """Return HTML for the sample section built from the first chunk."""


class PandasAdapter(EngineAdapter):
    """Engine adapter for pandas frames and chunks."""

    def infer_and_build(
        self, first: Any, cfg: Any
    ) -> Tuple[ColumnKinds, Dict[str, Any]]:
        inferrer = UnifiedTypeInferrer()
        result = inferrer.infer_kinds(first)
        if not result.success:
            raise RuntimeError(f"Type inference failed: {result.error}")
        kinds = result.data
        accs = _build_accumulators(kinds, cfg)
        try:
            dtypes_map = {c: str(first[c].dtype) for c in first.columns}
            for name in kinds.numeric:
                if name in accs and isinstance(accs[name], NumericAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "numeric"))
            for name in kinds.categorical:
                if name in accs and isinstance(accs[name], CategoricalAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "category"))
            for name in kinds.boolean:
                if name in accs and isinstance(accs[name], BooleanAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "boolean"))
            for name in kinds.datetime:
                if name in accs and isinstance(accs[name], DatetimeAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "datetime64[ns]"))
        except Exception:
            pass
        return kinds, accs

    def estimate_mem(self, frame: Any) -> int:
        try:
            return int(frame.memory_usage(deep=True).sum())
        except Exception:
            return 0

    def update_row_kmv(self, frame: Any, row_kmv: Any) -> None:
        row_kmv.update_from_pandas(frame)

    def missing_cells(self, frame: Any) -> int:
        try:
            return int(frame.isna().sum().sum())
        except Exception:
            return 0

    def consume_chunk(
        self, frame: Any, accs: Dict[str, Any], kinds: ColumnKinds, logger: Any
    ) -> None:
        _consume_chunk_pandas(frame, accs, kinds, logger)

    def update_corr(self, frame: Any, corr_est: Any, logger: Any) -> None:
        if corr_est is None:
            return
        try:
            corr_est.update_from_pandas(frame)
        except Exception:
            logger.exception("Correlation update failed on pandas chunk")

    def sample_section_html(self, first: Any, cfg: Any) -> str:
        if not getattr(cfg, "include_sample", True):
            return ""
        return _render_sample_section(first, getattr(cfg, "sample_rows", 10))


class PolarsAdapter(EngineAdapter):
    """Engine adapter for polars frames and chunks."""

    def infer_and_build(
        self, first: Any, cfg: Any
    ) -> Tuple[ColumnKinds, Dict[str, Any]]:
        inferrer = UnifiedTypeInferrer()
        result = inferrer.infer_kinds(first)
        if not result.success:
            raise RuntimeError(f"Type inference failed: {result.error}")
        kinds = result.data
        accs = _build_accumulators(kinds, cfg)
        try:
            dtypes_map = {c: str(first.schema[c]) for c in first.columns}
            for name in kinds.numeric:
                if name in accs and isinstance(accs[name], NumericAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "numeric"))
            for name in kinds.categorical:
                if name in accs and isinstance(accs[name], CategoricalAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "categorical"))
            for name in kinds.boolean:
                if name in accs and isinstance(accs[name], BooleanAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "boolean"))
            for name in kinds.datetime:
                if name in accs and isinstance(accs[name], DatetimeAccumulator):
                    accs[name].set_dtype(dtypes_map.get(name, "datetime"))
        except Exception:
            pass
        return kinds, accs

    def estimate_mem(self, frame: Any) -> int:
        try:
            return int(frame.estimated_size())
        except Exception:
            return 0

    def update_row_kmv(self, frame: Any, row_kmv: Any) -> None:
        row_kmv.update_from_polars(frame)

    def missing_cells(self, frame: Any) -> int:
        try:
            import polars as pl  # type: ignore

            # Optimized missing value counting - use sum_horizontal() instead of complex aggregation
            return int(frame.null_count().sum_horizontal().item())
        except Exception:
            try:
                # Fallback to the original method if sum_horizontal() is not available
                return int(frame.null_count().select(pl.sum(pl.all())).to_numpy()[0][0])
            except Exception:
                return 0

    def consume_chunk(
        self, frame: Any, accs: Dict[str, Any], kinds: ColumnKinds, logger: Any
    ) -> None:
        _consume_chunk_polars(frame, accs, kinds, logger)

    def update_corr(self, frame: Any, corr_est: Any, logger: Any) -> None:
        if corr_est is None:
            return
        try:
            corr_est.update_from_polars(frame)
        except Exception:
            logger.exception("Correlation update failed on polars chunk")

    def sample_section_html(self, first: Any, cfg: Any) -> str:
        if not getattr(cfg, "include_sample", True):
            return ""
        try:
            # Delegate to sections renderer; it dispatches and avoids pandas if desired
            return _render_sample_section(first, getattr(cfg, "sample_rows", 10))
        except Exception:
            return ""


def consume_stream(
    adapter: EngineAdapter,
    chunks: Iterator[Any],
    *,
    accs: Dict[str, Any],
    kinds: ColumnKinds,
    logger: Any,
    corr_est: Any,
    row_kmv: Any,
    approx_mem_bytes: int,
    total_missing_cells: int,
    first_columns: list[str],
    sample_section_html: str,
    cfg: "EngineConfig",
    ckpt_mgr: Optional[Any],
    start_time: float,
    report_title: Optional[str],
    n_rows: int,
    chunk_idx: int,
) -> tuple[int, int, int, int]:
    """Consume all chunks through the adapter while tracking progress.

    Args:
      adapter: The engine adapter (pandas or polars).
      chunks: Iterator of chunks/frames to process.
      accs: Accumulator mapping per column.
      kinds: Inferred kinds for columns.
      logger: Logger for progress and diagnostics.
      corr_est: Optional streaming correlation estimator.
      row_kmv: Row-level distinct estimator for duplicate estimation.
      approx_mem_bytes: Running memory estimate (bytes).
      total_missing_cells: Running missing cell count.
      first_columns: Column order from first chunk (for checkpoints).
      sample_section_html: Sample section HTML (for checkpoints).
      cfg: Engine configuration (see EngineConfig Protocol).
      ckpt_mgr: Optional checkpoint manager.
      start_time: Report start timestamp (for elapsed time in HTML ckpt).
      report_title: Optional title used in checkpoint HTML.
      n_rows: Running row count.
      chunk_idx: Chunk index (1-based, first chunk processed outside).

    Returns:
      Tuple of updated ``(n_rows, approx_mem_bytes, total_missing_cells, chunk_idx)``.
    """
    # Adapter log tag for clarity
    tag = get_adapter_tag(adapter)
    for ch in chunks:
        chunk_idx += 1
        if (chunk_idx - 1) % max(1, int(getattr(cfg, "log_every_n_chunks", 1))) == 0:
            try:
                size = getattr(ch, "height", None)
                size = int(size) if size is not None else len(ch)
                logger.info("%s processing chunk %d: %d rows", tag, chunk_idx, size)
            except Exception:
                logger.info("%s processing chunk %d", tag, chunk_idx)
        adapter.consume_chunk(ch, accs, kinds, logger)
        approx_mem_bytes += int(adapter.estimate_mem(ch) or 0)
        if corr_est is not None:
            adapter.update_corr(ch, corr_est, logger)
        try:
            n_rows += int(getattr(ch, "height", len(ch)))
        except Exception:
            pass
        adapter.update_row_kmv(ch, row_kmv)
        total_missing_cells += int(adapter.missing_cells(ch) or 0)
        if ckpt_mgr and (
            chunk_idx % int(getattr(cfg, "checkpoint_every_n_chunks", 0)) == 0
        ):
            try:
                snapshot = _make_state_snapshot(
                    kinds=kinds,
                    accs=accs,
                    row_kmv=row_kmv,
                    total_missing_cells=total_missing_cells,
                    approx_mem_bytes=approx_mem_bytes,
                    chunk_idx=chunk_idx,
                    first_columns=first_columns,
                    sample_section_html=sample_section_html,
                    cfg=cfg,
                )
                html_ckpt = None
                if bool(getattr(cfg, "checkpoint_write_html", False)):
                    html_ckpt = _render_html_snapshot(
                        kinds=kinds,
                        accs=accs,
                        first_columns=first_columns,
                        row_kmv=row_kmv,
                        total_missing_cells=total_missing_cells,
                        approx_mem_bytes=approx_mem_bytes,
                        start_time=start_time,
                        cfg=cfg,
                        report_title=report_title,
                        sample_section_html=sample_section_html,
                    )
                pkl_path, html_path = ckpt_mgr.save(chunk_idx, snapshot, html=html_ckpt)  # type: ignore[attr-defined]
                logger.info(
                    "Checkpoint saved at %s%s",
                    pkl_path,
                    f" and {html_path}" if html_path else "",
                )
            except Exception:
                logger.exception("Failed to write checkpoint at chunk %d", chunk_idx)
    return n_rows, approx_mem_bytes, total_missing_cells, chunk_idx


@runtime_checkable
class EngineConfig(Protocol):
    """Typed view of configuration fields used by the streaming engine."""

    log_every_n_chunks: int
    checkpoint_every_n_chunks: int
    checkpoint_write_html: bool
    compute_correlations: bool
    corr_max_cols: int


def maybe_corr_estimator(kinds: ColumnKinds, cfg: "EngineConfig"):
    """Create a streaming correlation estimator if config allows.

    Returns ``None`` if correlations are disabled or the number of numeric
    columns exceeds the configured maximum.
    """
    try:
        compute = bool(getattr(cfg, "compute_correlations", False))
        max_cols = int(getattr(cfg, "corr_max_cols", 0))
    except Exception:
        compute = False
        max_cols = 0
    if compute and len(kinds.numeric) > 1 and (len(kinds.numeric) <= max_cols):
        from .analysis.correlation import StreamingCorr  # local import to avoid cycles

        return StreamingCorr(kinds.numeric)
    return None


def prep_first_chunk(adapter: EngineAdapter, first: Any, cfg: Any, row_kmv: Any):
    """Prepare first-chunk derived values in one place.

    Args:
      adapter: Engine adapter.
      first: First frame/chunk.
      cfg: Engine configuration (sampling options for sample HTML).
      row_kmv: Row distinct estimator to be updated.

    Returns:
      Tuple[int, int, list[str], str]:
        (approx_mem_delta, missing_cells_delta, first_columns, sample_section_html).
    """
    approx_mem = int(adapter.estimate_mem(first) or 0)
    adapter.update_row_kmv(first, row_kmv)
    missing_delta = int(adapter.missing_cells(first) or 0)
    first_columns = list(getattr(first, "columns", []))
    sample_section_html = adapter.sample_section_html(first, cfg)
    return approx_mem, missing_delta, first_columns, sample_section_html


def select_adapter_for(obj: Any) -> Optional[EngineAdapter]:
    """Return the appropriate engine adapter for a given frame-like object."""
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.DataFrame):
            return PandasAdapter()
    except Exception:
        pass
    try:
        import polars as pl  # type: ignore

        if isinstance(obj, pl.DataFrame):
            return PolarsAdapter()
    except Exception:
        pass
    return None


def get_adapter_tag(adapter: EngineAdapter) -> str:
    """Return a short tag identifying the adapter for logging (e.g., "[pd]")."""
    if isinstance(adapter, PandasAdapter):
        return "[pd]"
    if isinstance(adapter, PolarsAdapter):
        return "[pl]"
    return "[eng]"
