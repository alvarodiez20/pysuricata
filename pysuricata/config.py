from __future__ import annotations

"""Internal engine configuration for the streaming report.

Separated from `pysuricata.report` to avoid circular imports and to keep the
engine's configuration distinct from the public API config in
`pysuricata.api`.
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EngineConfig:
    """Internal engine configuration for the streaming report.

    This configuration is used internally by the streaming engine and contains
    low-level settings for performance tuning, logging, and checkpointing.

    Attributes:
        title: Report title for HTML output
        description: Optional user description for the summary section
        chunk_size: Number of rows to process in each chunk
        numeric_sample_k: Reservoir sample size for numeric statistics
        uniques_k: Sketch size for approximate unique counting
        topk_k: Maximum number of top categories to track
        engine: Engine selector (currently only "auto")
        logger: Optional logger instance
        log_level: Logging level
        log_every_n_chunks: Log every N chunks (reduce verbosity)
        include_sample: Whether to include sample data in output
        sample_rows: Number of sample rows to include
        compute_correlations: Whether to compute column correlations
        corr_threshold: Minimum correlation threshold to report
        corr_max_cols: Maximum number of columns for correlation analysis
        corr_max_per_col: Maximum correlations to report per column
        random_seed: Seed for reproducible results (None for random)
        checkpoint_every_n_chunks: Checkpoint frequency (0 to disable)
        checkpoint_dir: Directory for checkpoint files
        checkpoint_prefix: Prefix for checkpoint filenames
        checkpoint_write_html: Whether to write HTML with checkpoints
        checkpoint_max_to_keep: Maximum number of checkpoints to retain
        force_chunk_in_memory: Force in-memory chunking
        missing_columns_threshold_pct: Minimum missing percentage to display
        missing_columns_max_initial: Maximum columns shown initially
        missing_columns_max_expanded: Maximum columns shown when expanded
    """

    title: str = "PySuricata EDA Report"
    description: str | None = None
    chunk_size: int = 200_000
    numeric_sample_k: int = 20_000
    uniques_k: int = 2048
    topk_k: int = 50
    engine: str = "auto"  # reserved for future (e.g., force polars)
    # Logging
    logger: logging.Logger | None = None
    log_level: int = logging.INFO
    log_every_n_chunks: int = 1  # set >1 to reduce verbosity on huge runs
    include_sample: bool = True
    sample_rows: int = 10
    # Correlations (optional, lightweight)
    compute_correlations: bool = True
    corr_threshold: float = 0.6
    corr_max_cols: int = 50
    corr_max_per_col: int = 2
    # Randomness control (None = nondeterministic; set an int for reproducibility)
    random_seed: int | None = None

    # Checkpointing
    checkpoint_every_n_chunks: int = 0  # 0 disables
    checkpoint_dir: str | None = (
        None  # if None, the engine decides (usually CWD or an engine-defined default)
    )
    checkpoint_prefix: str = "pysuricata_ckpt"
    checkpoint_write_html: bool = False  # also dump partial HTML next to pickle
    checkpoint_max_to_keep: int = 3  # rotate old checkpoints

    # In-memory chunking control
    force_chunk_in_memory: bool = False

    # Boolean detection options
    enable_auto_boolean_detection: bool = True
    boolean_detection_min_samples: int = 100
    boolean_detection_max_zero_ratio: float = 0.95
    boolean_detection_require_name_pattern: bool = True
    force_column_types: dict[str, str] | None = None

    # Missing columns display options
    missing_columns_threshold_pct: float = 0.0  # Minimum missing percentage to display
    missing_columns_max_initial: int = 5  # Maximum columns shown initially
    missing_columns_max_expanded: int = 10  # Maximum columns shown when expanded

    @classmethod
    def from_options(cls, opts: EngineOptions) -> EngineConfig:
        """Build engine config from any EngineOptions-compatible object.

        Uses duck-typing to avoid import cycles and keep public/internal models
        decoupled.
        """
        # Handle chunk_size=None to disable chunking (pass 0 to engine)
        engine_chunk_size = 0 if opts.chunk_size is None else opts.chunk_size

        return cls(
            chunk_size=engine_chunk_size,
            numeric_sample_k=int(opts.numeric_sample_k),
            uniques_k=int(opts.uniques_k),
            topk_k=int(opts.topk_k),
            engine=getattr(opts, "engine", "auto"),
            random_seed=opts.random_seed,
            # Add checkpointing parameters
            log_every_n_chunks=getattr(opts, "log_every_n_chunks", 1),
            checkpoint_every_n_chunks=getattr(opts, "checkpoint_every_n_chunks", 0),
            checkpoint_dir=getattr(opts, "checkpoint_dir", None),
            checkpoint_prefix=getattr(opts, "checkpoint_prefix", "pysuricata_ckpt"),
            checkpoint_write_html=getattr(opts, "checkpoint_write_html", False),
            checkpoint_max_to_keep=getattr(opts, "checkpoint_max_to_keep", 3),
            # Boolean detection parameters
            enable_auto_boolean_detection=getattr(
                opts, "enable_auto_boolean_detection", True
            ),
            boolean_detection_min_samples=getattr(
                opts, "boolean_detection_min_samples", 100
            ),
            boolean_detection_max_zero_ratio=getattr(
                opts, "boolean_detection_max_zero_ratio", 0.95
            ),
            boolean_detection_require_name_pattern=getattr(
                opts, "boolean_detection_require_name_pattern", True
            ),
            force_column_types=getattr(opts, "force_column_types", None),
        )

    def __post_init__(self) -> None:
        """Normalize and validate configuration invariants."""
        # Normalize logger level if a logger is provided
        if self.logger is not None and self.log_level is not None:
            try:
                self.logger.setLevel(self.log_level)
            except Exception:
                pass  # leave logger as-is if setting level fails

        # Validate non-negative integers
        for name in (
            "chunk_size",
            "numeric_sample_k",
            "uniques_k",
            "topk_k",
            "sample_rows",
            "log_every_n_chunks",
            "checkpoint_every_n_chunks",
            "checkpoint_max_to_keep",
        ):
            val = getattr(self, name, None)
            if not (isinstance(val, int) and val >= 0):
                raise ValueError(f"{name} must be a non-negative integer, got {val!r}")

        # Correlation threshold must be within [0.0, 1.0]
        if not (0.0 <= float(self.corr_threshold) <= 1.0):
            raise ValueError(
                f"corr_threshold must be between 0.0 and 1.0, got {self.corr_threshold}"
            )

        # Clamp topk_k to uniques_k to avoid nonsensical configuration
        if self.topk_k > self.uniques_k:
            self.topk_k = self.uniques_k

        # Checkpointing settings: when enabled, keep at least one checkpoint
        if self.checkpoint_every_n_chunks > 0 and self.checkpoint_max_to_keep < 1:
            raise ValueError(
                "checkpoint_max_to_keep must be >= 1 when checkpointing is enabled"
            )


@runtime_checkable
class EngineOptions(Protocol):
    """Typed view of the options the engine cares about.

    Any object exposing these attributes can be used to build an engine
    ``EngineConfig``. This enables decoupling while avoiding field duplication
    across public and internal configs.
    """

    chunk_size: int | None
    numeric_sample_k: int
    uniques_k: int
    topk_k: int
    engine: str
    random_seed: int | None
