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
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Union

from . import report
from .config import EngineConfig as _EngineConfig
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

        def _convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: _convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert numpy types to native Python types
        converted_stats = _convert_numpy_types(self.stats)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(converted_stats, f, ensure_ascii=False, indent=2)

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
    """Configuration for data processing and analysis.

    These options control how data is streamed and how approximations are
    performed during computation. They are intentionally conservative by
    default to provide stable results for small to medium datasets, while still
    scaling to larger ones.

    Examples:
        # For large datasets (memory constrained)
        ComputeOptions(chunk_size=50_000, numeric_sample_size=5_000)

        # For high-quality analysis
        ComputeOptions(chunk_size=500_000, numeric_sample_size=50_000)

        # For reproducible results
        ComputeOptions(random_seed=42)

        # For specific columns only
        ComputeOptions(columns=["age", "income", "education"])

        # With checkpointing for large datasets
        ComputeOptions(
            chunk_size=100_000,
            checkpoint_every_n_chunks=10,
            checkpoint_dir="./checkpoints",
            checkpoint_write_html=True
        )

    Attributes:
        chunk_size: Number of rows to process in each chunk. Larger chunks
            use more memory but may be faster. Default: 200,000
        columns: Optional subset of columns to analyze. If None, all columns
            are analyzed. Default: None (all columns)
        numeric_sample_size: Reservoir sample size for numeric statistics like
            quantiles and histograms. Larger samples give better accuracy but
            use more memory. Default: 20,000
        max_uniques: Sketch size for approximate unique value counting.
            Larger sketches give better accuracy but use more memory.
            Default: 2,048
        top_k: Maximum number of top categories to track for categorical
            columns. Default: 50
        random_seed: Seed for reproducible sampling. Set to None for
            non-deterministic results. Default: 0
        log_every_n_chunks: Log progress every N chunks. Set to 1 to log every
            chunk, higher values for less frequent logging. Default: 1
        checkpoint_every_n_chunks: Create checkpoint every N chunks. Set to 0
            to disable checkpointing. Default: 0 (disabled)
        checkpoint_dir: Directory for checkpoint files. If None, uses current
            working directory. Default: None
        checkpoint_prefix: Prefix for checkpoint filenames. Default: "pysuricata_ckpt"
        checkpoint_write_html: Whether to include HTML in checkpoints.
            Default: False
        checkpoint_max_to_keep: Maximum number of checkpoints to retain.
            Default: 3
        enable_auto_boolean_detection: Whether to automatically detect 0/1 numeric
            columns as boolean. Default: True
        boolean_detection_min_samples: Minimum number of samples required for
            boolean detection. Default: 100
        boolean_detection_max_zero_ratio: Maximum ratio of zeros allowed for
            boolean detection (to avoid classifying mostly-zero columns as boolean).
            Default: 0.95
        boolean_detection_require_name_pattern: Whether to require boolean-like
            column names (e.g., 'is_', 'has_', 'can_') for detection. Default: True
        force_column_types: Optional dictionary mapping column names to their
            forced types. Overrides automatic type inference. Default: None
    """

    chunk_size: Optional[int] = 200_000
    columns: Optional[Sequence[str]] = None
    numeric_sample_size: int = 20_000
    max_uniques: int = 2_048
    top_k: int = 50
    random_seed: Optional[int] = 0

    # Logging and checkpointing
    log_every_n_chunks: int = 1
    checkpoint_every_n_chunks: int = 0  # 0 disables checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_prefix: str = "pysuricata_ckpt"
    checkpoint_write_html: bool = False
    checkpoint_max_to_keep: int = 3

    # Boolean detection options
    enable_auto_boolean_detection: bool = True
    boolean_detection_min_samples: int = 100
    boolean_detection_max_zero_ratio: float = 0.95
    boolean_detection_require_name_pattern: bool = True
    force_column_types: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.numeric_sample_size <= 0:
            raise ValueError("numeric_sample_size must be positive")
        if self.max_uniques <= 0:
            raise ValueError("max_uniques must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.log_every_n_chunks <= 0:
            raise ValueError("log_every_n_chunks must be positive")
        if self.checkpoint_every_n_chunks < 0:
            raise ValueError("checkpoint_every_n_chunks must be non-negative")
        if self.checkpoint_max_to_keep <= 0:
            raise ValueError("checkpoint_max_to_keep must be positive")
        if self.boolean_detection_min_samples <= 0:
            raise ValueError("boolean_detection_min_samples must be positive")
        if not 0 <= self.boolean_detection_max_zero_ratio <= 1:
            raise ValueError("boolean_detection_max_zero_ratio must be between 0 and 1")
        if self.force_column_types is not None:
            valid_types = {"numeric", "categorical", "datetime", "boolean"}
            for col_name, col_type in self.force_column_types.items():
                if col_type not in valid_types:
                    raise ValueError(
                        f"Invalid column type '{col_type}' for column '{col_name}'. Must be one of: {valid_types}"
                    )

    # --- Engine-aligned accessors (for backward compatibility) ---
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
    """Render options for the HTML output.

    The current HTML report is self-contained and styled with built-in assets.
    This class is kept for future extensibility but currently has no effect
    on the output.
    """

    pass  # No options currently implemented


@dataclass
class ProfileConfig:
    """High-level configuration for data profiling.

    This is the main configuration class used by the public API functions
    `profile()` and `summarize()`. It contains compute and render options
    that control how data is processed and how the output is generated.

    Examples:
        # Basic usage with defaults
        config = ProfileConfig()

        # Custom compute settings
        config = ProfileConfig(
            compute=ComputeOptions(
                chunk_size=100_000,
                numeric_sample_size=10_000,
                random_seed=42
            )
        )

    Attributes:
        compute: Compute-related options; see :class:`ComputeOptions`.
        render: Render-related options; see :class:`RenderOptions`.
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
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data
    except ImportError:
        pass

    try:
        import polars as pl

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return data
    except ImportError:
        pass

    if isinstance(data, cabc.Iterable) and not isinstance(
        data, (str, bytes, bytearray, cabc.Mapping)
    ):
        return data

    raise TypeError(
        "Unsupported data type for this API. Provide a pandas DataFrame, a polars DataFrame/LazyFrame, or an iterable of pandas/polars DataFrames."
    )


def _to_engine_config(cfg: ProfileConfig) -> _EngineConfig:
    """Convert public configuration to internal engine configuration.

    This function translates the user-friendly public configuration into the
    internal engine configuration format.
    """
    compute = cfg.compute

    # Use the from_options method if available
    try:
        return _EngineConfig.from_options(compute)
    except Exception:
        # Fallback: direct mapping with checkpointing support
        # Only include checkpointing parameters if they exist in the config
        # Handle chunk_size=None to disable chunking (pass 0 to engine)
        engine_chunk_size = 0 if compute.chunk_size is None else compute.chunk_size

        config_kwargs = {
            "chunk_size": engine_chunk_size,
            "numeric_sample_k": compute.numeric_sample_k,
            "uniques_k": compute.uniques_k,
            "topk_k": compute.topk_k,
            "random_seed": compute.random_seed,
        }

        # Add checkpointing parameters only if they exist in both compute and _EngineConfig
        checkpoint_params = [
            "log_every_n_chunks",
            "checkpoint_every_n_chunks",
            "checkpoint_dir",
            "checkpoint_prefix",
            "checkpoint_write_html",
            "checkpoint_max_to_keep",
        ]

        # Get the constructor signature to check which parameters are supported
        import inspect

        try:
            sig = inspect.signature(_EngineConfig.__init__)
            supported_params = set(sig.parameters.keys()) - {"self"}

            for param in checkpoint_params:
                if hasattr(compute, param) and param in supported_params:
                    config_kwargs[param] = getattr(compute, param)
        except Exception:
            # If we can't inspect the signature, just use the basic parameters
            pass

        return _EngineConfig(**config_kwargs)


def profile(
    data: DataLike,
    config: Optional[ProfileConfig] = None,
) -> Report:
    """Compute statistics and render a self‑contained HTML report.

    The function accepts in‑memory data (pandas or polars) or an iterable of
    pandas or polars chunks. Both pandas and polars DataFrames are processed
    consistently - chunking is handled by the engine based on the chunk_size
    configuration.

    Args:
        data: Dataset to analyze. Supported:
            - ``pandas.DataFrame``
            - ``polars.DataFrame`` or ``polars.LazyFrame``
            - Iterable yielding ``pandas.DataFrame`` or ``polars.DataFrame`` chunks
        config: Optional configuration overriding compute/render defaults.
            Set chunk_size=None to disable chunking for both pandas and polars.

    Returns:
        A :class:`Report` object containing the HTML and the computed stats
        mapping.

    Raises:
        TypeError: If ``data`` is not of a supported type.
        ValueError: If ``data`` is None.
    """
    if data is None:
        raise ValueError("Input data cannot be None")

    cfg = config or ProfileConfig()
    inp = _coerce_input(data)  # No more polars-specific wrapping!
    cfg = _to_engine_config(cfg)

    # Always compute stats to return machine-readable mapping
    html, summary = report.build_report(inp, config=cfg, return_summary=True)  # type: ignore[misc]

    try:
        stats = dict(summary or {})
    except Exception:
        stats = {"dataset": {}, "columns": {}}
    return Report(html=html, stats=stats)


def summarize(
    data: DataLike,
    config: Optional[ProfileConfig] = None,
) -> Mapping[str, Any]:
    """Compute statistics only and return a JSON‑safe mapping.

    This is the programmatic counterpart to :func:`profile` for code paths that
    do not need the HTML report (e.g., CI checks and data quality gates).
    Both pandas and polars DataFrames are processed consistently.

    Args:
        data: Dataset to analyze. Same accepted types as :func:`profile`.
        config: Optional configuration overriding compute/render defaults.
            Set chunk_size=None to disable chunking for both pandas and polars.

    Returns:
        A nested mapping with dataset‑level and per‑column statistics. The
        result is safe to serialize to JSON.

    Raises:
        TypeError: If ``data`` is not of a supported type.
        ValueError: If ``data`` is None.
    """
    if data is None:
        raise ValueError("Input data cannot be None")

    cfg = config or ProfileConfig()
    inp = _coerce_input(data)  # No more polars-specific wrapping!
    cfg = _to_engine_config(cfg)
    # compute-only to skip HTML render
    _html, summary = report.build_report(
        inp, config=cfg, return_summary=True, compute_only=True
    )  # type: ignore[misc]
    stats = dict(summary or {})
    return stats
