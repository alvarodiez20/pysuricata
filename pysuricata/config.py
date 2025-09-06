from __future__ import annotations

"""Internal engine configuration for the streaming report.

Separated from `pysuricata.report` to avoid circular imports and to keep the
engine's configuration distinct from the public API config in
`pysuricata.api`.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class ReportConfig:
    """Configuration understood by the streaming engine (internal).

    Attributes mirror the engine's needs: chunk sizes, sketch/sample sizes,
    correlation toggles and thresholds, logging, and checkpointing settings.
    """

    title: str = "PySuricata EDA Report (streaming)"
    chunk_size: int = 200_000
    numeric_sample_k: int = 20_000
    uniques_k: int = 2048
    topk_k: int = 50
    engine: str = "auto"  # reserved for future (e.g., force polars)
    # Logging
    logger: Optional[logging.Logger] = None
    log_level: int = logging.INFO
    log_every_n_chunks: int = 1  # set >1 to reduce verbosity on huge runs
    include_sample: bool = True
    sample_rows: int = 10
    # Correlations (optional, lightweight)
    compute_correlations: bool = True
    corr_threshold: float = 0.6
    corr_max_cols: int = 50
    corr_max_per_col: int = 2
    # Randomness control
    random_seed: Optional[int] = 0

    # Checkpointing
    checkpoint_every_n_chunks: int = 0  # 0 disables
    checkpoint_dir: Optional[str] = None  # default: dirname(output_file) or CWD
    checkpoint_prefix: str = "pysuricata_ckpt"
    checkpoint_write_html: bool = False  # also dump partial HTML next to pickle
    checkpoint_max_to_keep: int = 3  # rotate old checkpoints


@runtime_checkable
class EngineOptions(Protocol):
    """Typed view of the options the engine cares about.

    Any object exposing these attributes can be used to build an engine
    ``ReportConfig``. This enables decoupling while avoiding field duplication
    across public and internal configs.
    """

    chunk_size: Optional[int]
    numeric_sample_k: int
    uniques_k: int
    topk_k: int
    engine: str
    random_seed: Optional[int]

    @classmethod
    def from_options(cls, opts: EngineOptions) -> "ReportConfig":
        """Build engine config from any ``EngineOptions``-compatible object.

        Uses duck-typing to avoid import cycles and keep public/internal models
        decoupled.
        """
        return cls(
            chunk_size=opts.chunk_size or 200_000,
            numeric_sample_k=int(opts.numeric_sample_k),
            uniques_k=int(opts.uniques_k),
            topk_k=int(opts.topk_k),
            engine=str(opts.engine),
            random_seed=opts.random_seed,
        )
