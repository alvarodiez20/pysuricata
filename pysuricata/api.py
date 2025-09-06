"""High-level public API for PySuricata.

This module exposes two primary entry points that are safe to use from
applications and notebooks:

- `profile`: Computes streaming statistics over a dataset and renders a
  self-contained HTML report alongside a JSON-friendly summary.
- `summarize`: Computes the same statistics but returns only the
  machine-readable summary mapping (no HTML).

Both functions are intentionally lightweight wrappers around the internal
streaming engine implemented in `pysuricata.report`. They accept
in-memory data (pandas or polars) or an iterable of pandas DataFrame chunks.
"""

from __future__ import annotations

import collections.abc as cabc
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from . import report
from .config import ReportConfig as _EngineReportConfig
from .io import iter_chunks as _iter_chunks

# Type-only imports so pandas/polars/pyarrow remain optional
if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore
    import polars as pl  # type: ignore

# Public data-like union: in-memory only (no file paths).
# Accept single frames (pandas/polars), polars LazyFrame, or iterables of frames.
DataLike = Union[
    "pd.DataFrame",  # pandas
    "pl.DataFrame",  # polars eager
    "pl.LazyFrame",  # polars lazy
    cabc.Iterable,  # iterator/generator yielding pandas or polars DataFrames
]


# Thin wrapper Report object with convenience methods
@dataclass
class Report:
    html: str
    stats: Mapping[str, Any]

    """Container for a rendered report and its computed statistics.

    Attributes:
        html: The full HTML document for the report (self‑contained).
        stats: JSON‑serializable mapping with dataset‑level and per‑column
            statistics, suitable for programmatic consumption (e.g., CI checks).
    """

    def save_html(self, path: str) -> None:
        """Write the HTML report to disk.

        Args:
            path: Destination file path. Parent directories must exist.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)

    def save_json(self, path: str) -> None:
        """Write the statistics mapping to a JSON file.

        Args:
            path: Destination file path. Parent directories must exist.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

    def save(self, path: str) -> None:
        """Save the report based on the file extension.

        If the extension is ``.html``, the HTML is written. If it is ``.json``,
        the stats mapping is written as JSON.

        Args:
            path: Destination file path.

        Raises:
            ValueError: If the extension is not one of ``.html`` or ``.json``.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".html":
            self.save_html(path)
        elif ext == ".json":
            self.save_json(path)
        else:
            raise ValueError(f"Unknown extension for Report.save(): {ext}")

    # Jupyter-friendly inline display
    def _repr_html_(self) -> str:  # pragma: no cover - visual
        return self.html


@dataclass
class ComputeOptions:
    """Compute options for the profiling engine.

    These options control how data is streamed and how approximations are
    performed during computation. They are intentionally conservative by
    default to provide stable results for small to medium datasets, while still
    scaling to larger ones.

    Attributes:
        chunk_size: Suggested chunk size when chunking is needed (rows).
        columns: Optional subset of columns to include during computation.
        numeric_sample_size: Reservoir sample size used for numeric summaries
            (quantiles, histogram preview).
        max_uniques: Sketch size used to estimate number of unique values.
        top_k: Maximum number of top categories tracked for categorical columns.
        detect_pii: Reserved for future use (PII detection disabled by default).
        engine: Reserved selector for future backends ("auto" today).
        dtypes: Reserved per-column dtype overrides for future ingestion layer.
        random_seed: Seed for deterministic sampling; ``None`` keeps RNG state.
    """

    # General compute knobs
    chunk_size: Optional[int] = 200_000
    columns: Optional[Sequence[str]] = None
    numeric_sample_size: int = 20_000
    max_uniques: int = 2_048
    top_k: int = 50
    detect_pii: bool = False  # reserved for future integration
    engine: str = "auto"  # reserved
    # Parsing control (reserved)
    dtypes: Optional[Mapping[str, str]] = None
    random_seed: Optional[int] = 0

    # --- Engine-aligned accessors (no breaking change) ---
    @property
    def numeric_sample_k(self) -> int:
        """Alias used by the engine; backed by ``numeric_sample_size``."""
        return int(self.numeric_sample_size)

    @property
    def uniques_k(self) -> int:
        """Alias used by the engine; backed by ``max_uniques``."""
        return int(self.max_uniques)

    @property
    def topk_k(self) -> int:
        """Alias used by the engine; backed by ``top_k``."""
        return int(self.top_k)


@dataclass
class RenderOptions:
    """Render options for the HTML output (mostly reserved).

    The current HTML report is self-contained and styled with the built-in
    assets. These options are left for forward compatibility.

    Attributes:
        theme: Desired theme name; only ``"light"`` is used at the moment.
        embed_assets: Whether to inline assets into the HTML document.
        show_quality_flags: Whether to include quality flags chips.
    """

    theme: str = "light"  # reserved; v2 uses built-in theme
    embed_assets: bool = True  # reserved; v2 HTML is self-contained
    show_quality_flags: bool = True  # reserved; chips already rendered


@dataclass
class ReportConfig:
    """High-level configuration passed to :func:`profile` and :func:`summarize`.

    Attributes:
        compute: Compute related knobs; see :class:`ComputeOptions`.
        render: Render related knobs; see :class:`RenderOptions`.
    """

    compute: ComputeOptions = field(default_factory=ComputeOptions)
    render: RenderOptions = field(default_factory=RenderOptions)


def _coerce_input(data: DataLike) -> Union["pd.DataFrame", cabc.Iterable]:
    """Normalize supported inputs into a form the engine can consume.

    The API is intentionally strict about accepted inputs to keep the
    orchestration layer lightweight and dependency‑optional. File paths and
    on‑disk loaders are out of scope for this function.

    Args:
        data: One of the supported in‑memory data forms:
            - a pandas ``DataFrame``;
            - a polars eager or lazy frame (handled upstream by the caller);
            - an iterable (generator, list, tuple, etc.) yielding pandas or
              polars ``DataFrame`` chunks.

    Returns:
        Either a pandas ``DataFrame`` or an iterable of ``DataFrame`` objects.

    Raises:
        TypeError: If the object is not one of the supported forms.
    """
    # pandas DataFrame
    try:
        import pandas as pd  # type: ignore

        if isinstance(data, pd.DataFrame):
            return data  # type: ignore[return-value]
    except Exception:
        pass
    # polars eager/lazy frames: let the caller decide how to iterate
    try:
        import polars as pl  # type: ignore

        if (
            isinstance(data, (pl.DataFrame,))
            or getattr(data, "__class__", None).__name__ == "LazyFrame"
        ):
            return data  # type: ignore[return-value]
    except Exception:
        pass
    # Iterator/generator of DataFrames (duck-typed); let report validate on consumption
    try:
        # Accept any non-string, non-mapping iterable (lists/tuples allowed)
        if isinstance(data, cabc.Iterable) and not isinstance(
            data, (str, bytes, bytearray, cabc.Mapping)
        ):
            return data  # type: ignore[return-value]
    except Exception:
        pass
    raise TypeError(
        "Unsupported data type for this API. Provide a pandas DataFrame, a polars DataFrame/LazyFrame, or an iterable of pandas/polars DataFrames."
    )


def _to_engine_config(cfg: ReportConfig):
    """Translate high-level config into the internal engine configuration.

    Leverages engine-compatible accessors on ``ComputeOptions`` to avoid field
    duplication and brittle mappings.
    """
    compute = cfg.compute
    # Prefer constructor if available (newer engine)
    try:
        ctor = getattr(_EngineReportConfig, "from_options", None)
        if callable(ctor):
            return ctor(compute)  # type: ignore[arg-type]
    except Exception:
        pass
    # Fallback mapping for older engine versions without `from_options`
    try:
        return _EngineReportConfig(
            chunk_size=compute.chunk_size or 200_000,
            numeric_sample_k=int(
                getattr(compute, "numeric_sample_k", compute.numeric_sample_size)
            ),
            uniques_k=int(getattr(compute, "uniques_k", compute.max_uniques)),
            topk_k=int(getattr(compute, "topk_k", compute.top_k)),
            engine=str(compute.engine),
            random_seed=compute.random_seed,
        )
    except Exception:
        # Last resort: minimal defaults
        return _EngineReportConfig()


def profile(
    data: DataLike,
    config: Optional[ReportConfig] = None,
) -> Report:
    """Compute statistics and render a self‑contained HTML report.

    The function accepts in‑memory data (pandas or polars) or an iterable of
    pandas or polars chunks. For polars input, native polars chunks are used
    (no conversion to pandas), and the engine consumes them via its polars path. All heavy
    lifting (streaming, sketches, HTML composition) is handled by the internal
    engine.

    Args:
        data: Dataset to analyze. Supported:
            - ``pandas.DataFrame``
            - ``polars.DataFrame`` or ``polars.LazyFrame``
            - Iterable yielding ``pandas.DataFrame`` or ``polars.DataFrame`` chunks
        config: Optional configuration overriding compute/render defaults.

    Returns:
        A :class:`Report` object containing the HTML and the computed stats
        mapping.

    Raises:
        TypeError: If ``data`` is not of a supported type.
    """

    cfg = config or ReportConfig()
    inp_raw = _coerce_input(data)
    wrapped = None
    try:
        import polars as pl  # type: ignore

        if (
            isinstance(inp_raw, (pl.DataFrame,))
            or getattr(inp_raw, "__class__", None).__name__ == "LazyFrame"
        ):
            wrapped = _iter_chunks(
                inp_raw, chunk_size=cfg.compute.chunk_size, columns=cfg.compute.columns
            )
    except Exception:
        pass
    inp = wrapped if wrapped is not None else inp_raw
    v2cfg = _to_engine_config(cfg)

    # Always compute stats to return machine-readable mapping
    html, summary = report.build_report(inp, config=v2cfg, return_summary=True)  # type: ignore[misc]

    try:
        stats = dict(summary or {})
    except Exception:
        stats = {"dataset": {}, "columns": {}}
    return Report(html=html, stats=stats)


def summarize(
    data: DataLike,
    config: Optional[ReportConfig] = None,
) -> Mapping[str, Any]:
    """Compute statistics only and return a JSON‑safe mapping.

    This is the programmatic counterpart to :func:`profile` for code paths that
    do not need the HTML report (e.g., CI checks and data quality gates).

    Args:
        data: Dataset to analyze. Same accepted types as :func:`profile`.
        config: Optional configuration overriding compute/render defaults.

    Returns:
        A nested mapping with dataset‑level and per‑column statistics. The
        result is safe to serialize to JSON.

    Raises:
        TypeError: If ``data`` is not of a supported type.
    """

    cfg = config or ReportConfig()
    inp_raw = _coerce_input(data)
    wrapped = None
    try:
        import polars as pl  # type: ignore

        if (
            isinstance(inp_raw, (pl.DataFrame,))
            or getattr(inp_raw, "__class__", None).__name__ == "LazyFrame"
        ):
            wrapped = _iter_chunks(
                inp_raw, chunk_size=cfg.compute.chunk_size, columns=cfg.compute.columns
            )
    except Exception:
        pass
    inp = wrapped if wrapped is not None else inp_raw
    v2cfg = _to_engine_config(cfg)
    # compute-only to skip HTML render
    _html, summary = report.build_report(
        inp, config=v2cfg, return_summary=True, compute_only=True
    )  # type: ignore[misc]
    stats = dict(summary or {})
    return stats
