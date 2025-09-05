from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union, TYPE_CHECKING
import collections.abc as cabc
import os
import json

# Type-only imports so pandas/polars/pyarrow remain optional
if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore
    import polars as pl  # type: ignore

# Public data-like union: in-memory only (no file paths).
DataLike = Union[
    "pd.DataFrame",  # pandas
    "pl.DataFrame",  # polars eager
    "pl.LazyFrame",  # polars lazy
    cabc.Iterable,    # iterator/generator yielding pandas DataFrames
]


# Thin wrapper Report object with convenience methods
@dataclass
class Report:
    html: str
    stats: Mapping[str, Any]

    def save_html(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

    def save(self, path: str) -> None:
        """Save based on extension (.html or .json)."""
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
    # General compute knobs
    chunk_size: Optional[int] = 200_000
    columns: Optional[Sequence[str]] = None
    numeric_sample_size: int = 20_000
    max_uniques: int = 2_048
    top_k: int = 50
    detect_pii: bool = False  # reserved for future integration
    engine: str = "auto"      # reserved
    # Parsing control (reserved)
    dtypes: Optional[Mapping[str, str]] = None
    random_seed: Optional[int] = 0


@dataclass
class RenderOptions:
    theme: str = "light"            # reserved; v2 uses built-in theme
    embed_assets: bool = True        # reserved; v2 HTML is self-contained
    show_quality_flags: bool = True  # reserved; chips already rendered


@dataclass
class ReportConfig:
    compute: ComputeOptions = field(default_factory=ComputeOptions)
    render: RenderOptions = field(default_factory=RenderOptions)


# Path-based loading is intentionally unsupported.


def _coerce_input(data: DataLike) -> Union["pd.DataFrame", cabc.Iterable]:
    """Resolve the actual input to feed into report_v2.

    Supports:
    - pandas.DataFrame
    - Iterator/generator yielding pandas DataFrames
    """
    # Path-like not supported
    # pandas DataFrame
    try:
        import pandas as pd  # type: ignore
        if isinstance(data, pd.DataFrame):
            return data  # type: ignore[return-value]
    except Exception:
        pass
    # Iterator/generator of DataFrames (duck-typed); let report_v2 validate on consumption
    try:
        if isinstance(data, cabc.Iterable) and not isinstance(data, (dict, list, tuple)):
            return data  # type: ignore[return-value]
    except Exception:
        pass
    raise TypeError(
        "Unsupported data type for this API. Provide a pandas DataFrame, a polars DataFrame, or an iterable of pandas DataFrames."
    )


def _to_engine_config(path_or_df: Union["pd.DataFrame", cabc.Iterable], cfg: ReportConfig) -> "report_v2.ReportConfig":  # type: ignore[name-defined]
    """Translate our high-level ReportConfig into engine config (internal)."""
    from . import report_v2

    compute = cfg.compute
    # Base mapping
    v2 = report_v2.ReportConfig(
        chunk_size=compute.chunk_size or 200_000,
        numeric_sample_k=compute.numeric_sample_size,
        uniques_k=compute.max_uniques,
        topk_k=compute.top_k,
        engine=compute.engine,
    )

    return v2


def profile(
    data: DataLike,
    config: Optional[ReportConfig] = None,
) -> Report:
    """Compute stats and render HTML. Returns a Report(html, stats).

    - data: In-memory DataFrame or iterable of DataFrames.
    - config: High-level config with compute/render options.
    - loader: deprecated (not supported)
    """
    from . import report_v2

    cfg = config or ReportConfig()
    inp_raw = _coerce_input(data)
    # Special-case: in-memory polars -> chunked pandas frames
    wrapped = None
    try:
        import polars as pl  # type: ignore
        if isinstance(inp_raw, (pl.DataFrame,)) or getattr(inp_raw, "__class__", None).__name__ == "LazyFrame":
            from .io import iter_chunks as _iter_chunks
            wrapped = _iter_chunks(
                inp_raw, chunk_size=cfg.compute.chunk_size, columns=cfg.compute.columns
            )
    except Exception:
        pass
    inp = wrapped if wrapped is not None else inp_raw
    v2cfg = _to_engine_config(inp, cfg)

    # Always compute stats to return machine-readable mapping
    html, summary = report_v2.build_report(inp, config=v2cfg, return_summary=True)  # type: ignore[misc]

    try:
        stats = dict(summary or {})
    except Exception:
        stats = {"dataset": {}, "columns": {}}
    return Report(html=html, stats=stats)


def summarize(
    data: DataLike,
    config: Optional[ReportConfig] = None,
) -> Mapping[str, Any]:
    """Stats-only manifest. JSON-safe mapping suitable for CI/data-quality checks."""
    from . import report_v2
    cfg = config or ReportConfig()
    inp_raw = _coerce_input(data)
    # Special-case: in-memory polars -> chunked pandas frames
    wrapped = None
    try:
        import polars as pl  # type: ignore
        if isinstance(inp_raw, (pl.DataFrame,)) or getattr(inp_raw, "__class__", None).__name__ == "LazyFrame":
            from .io import iter_chunks as _iter_chunks
            wrapped = _iter_chunks(
                inp_raw, chunk_size=cfg.compute.chunk_size, columns=cfg.compute.columns
            )
    except Exception:
        pass
    inp = wrapped if wrapped is not None else inp_raw
    v2cfg = _to_engine_config(inp, cfg)
    # compute-only to skip HTML render
    _html, summary = report_v2.build_report(inp, config=v2cfg, return_summary=True, compute_only=True)  # type: ignore[misc]
    stats = dict(summary or {})
    return stats
