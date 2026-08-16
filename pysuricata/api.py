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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from . import report
from . import sources as _sources
from .config import EngineConfig as _EngineConfig


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy scalar/array types to native Python types.

    Needed because ``json.dump`` cannot serialise numpy integers, floats, or
    ndarrays out of the box.
    """
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
    return obj


# Type-only imports so pandas/polars/pyarrow remain optional
if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore
    import polars as pl  # type: ignore

# Public data-like union: a frame, a lazy frame, an iterable of chunks, or a
# path to a file this library knows how to read.
DataLike = Union[
    "pd.DataFrame",  # pandas
    "pl.DataFrame",  # polars eager
    "pl.LazyFrame",  # polars lazy
    str,  # path to a .csv / .parquet / .json file
    os.PathLike,
    cabc.Iterable,  # iterator/generator yielding pandas or polars DataFrames
]


class PySuricataError(Exception):
    """Base class for every error this library raises deliberately.

    The public surface used to raise ``TypeError`` for an unsupported input,
    ``ValueError`` for an unsupported file format and ``RuntimeError`` from the
    engine for the same class of mistake, so a caller had to catch three types
    to handle one situation. Everything raised on purpose now derives from this;
    the specific types remain as subclasses so existing ``except TypeError``
    handlers keep working.
    """


class UnsupportedDataError(PySuricataError, TypeError):
    """The input is not something this library can profile."""


class ConfigurationError(PySuricataError, ValueError):
    """A configuration value is outside its documented range."""


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

    def __repr__(self) -> str:
        """One line, not the whole document.

        The dataclass default rendered every byte of ``html``, so a bare
        ``report`` in a REPL printed over a megabyte and an exception
        traceback carried the entire report inline.
        """
        dataset = self.stats.get("dataset", {}) if self.stats else {}
        rows = dataset.get("rows_est")
        cols = len(self.stats.get("columns", {})) if self.stats else 0
        shape = f"{rows:,} rows" if isinstance(rows, int) else "unknown rows"
        return (
            f"<Report {shape} x {cols} columns, {len(self.html) / 1024:.0f} KB of HTML>"
        )

    # Jupyter-friendly inline display
    def _repr_html_(self) -> str:  # pragma: no cover - visual
        return self.html

    def display_in_notebook(self, width: str = "100%", height: str = "600px") -> None:
        """Display the report in a Jupyter notebook using an iframe.

        This method provides better display for large reports in Jupyter notebooks
        by using an iframe instead of inline HTML.

        Args:
            width: Width of the iframe (default: "100%")
            height: Height of the iframe (default: "600px")
        """
        try:
            import os
            import tempfile
            import threading
            import time

            from IPython.display import IFrame, display

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False
            ) as f:
                f.write(self.html)
                temp_path = f.name

            # Get the file URL for the iframe
            file_url = f"file://{temp_path}"

            # Display using iframe
            display(IFrame(file_url, width=width, height=height))

            # Clean up the temporary file after a delay
            def cleanup():
                time.sleep(5)  # Wait 5 seconds before cleanup
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

            cleanup_thread = threading.Thread(target=cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()

        except ImportError:
            # Fallback to regular HTML display if IPython is not available
            return self._repr_html_()

    def show(self, width: str = "100%", height: str = "600px") -> None:
        """Alias for display_in_notebook for convenience."""
        return self.display_in_notebook(width, height)


class CheckpointView:
    """The checkpointing settings of a `ComputeOptions`, under shorter names.

    Reads and writes go straight through to the underlying fields, so this is a
    lens on the same state rather than a copy of it -- a copy would be a second
    place for the settings to live, and they would disagree the first time
    someone set one directly.
    """

    __slots__ = ("_options",)

    _FIELDS = {
        "every_n_chunks": "checkpoint_every_n_chunks",
        "dir": "checkpoint_dir",
        "prefix": "checkpoint_prefix",
        "write_html": "checkpoint_write_html",
        "max_to_keep": "checkpoint_max_to_keep",
    }

    def __init__(self, options: ComputeOptions) -> None:
        object.__setattr__(self, "_options", options)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._options, self._FIELDS[name])
        except KeyError:
            raise AttributeError(
                f"no checkpoint setting {name!r}; "
                f"try one of: {', '.join(sorted(self._FIELDS))}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            field_name = self._FIELDS[name]
        except KeyError:
            raise AttributeError(
                f"no checkpoint setting {name!r}; "
                f"try one of: {', '.join(sorted(self._FIELDS))}"
            ) from None
        setattr(self._options, field_name, value)

    def __repr__(self) -> str:
        settings = ", ".join(
            f"{short}={getattr(self._options, full)!r}"
            for short, full in self._FIELDS.items()
        )
        return f"CheckpointSettings({settings})"


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
        chunk_size: Number of rows to process in each chunk. Default: 50,000.
            Bigger is not faster: the sketch merges are superlinear in batch
            size, so one 200,000-row batch costs more than four 50,000-row
            ones. Raise it only to trade memory for fewer chunk boundaries.
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
        compute_correlations: Whether to compute pairwise correlations between
            numeric columns. Default: True
        corr_threshold: Minimum absolute correlation value to report. Only
            correlations with |r| >= threshold are shown. Default: 0.5
        corr_max_cols: Maximum number of numeric columns for correlation analysis.
            If exceeded, correlations are skipped. Default: 50
        corr_max_per_col: Maximum number of top correlations to show per column.
            Default: 10
    """

    chunk_size: int | None = 50_000
    columns: Sequence[str] | None = None
    numeric_sample_size: int = 20_000
    max_uniques: int = 2_048
    top_k: int = 50
    random_seed: int | None = 0
    # True, False, "auto", or a callable taking chunks/rows/elapsed. Always
    # stderr, never stdout, so piped output stays parseable.
    progress: Any = False

    # Logging and checkpointing
    log_every_n_chunks: int = 1
    # Five of the twenty-two fields serve one concern most users never touch.
    # They stay here as fields for compatibility -- removing them would break
    # existing code for no gain -- and are also reachable as a named group via
    # the `checkpoint` view below, which is what makes the shape legible.
    checkpoint_every_n_chunks: int = 0  # 0 disables checkpointing
    checkpoint_dir: str | None = None
    checkpoint_prefix: str = "pysuricata_ckpt"
    checkpoint_write_html: bool = False
    checkpoint_max_to_keep: int = 3

    # Boolean detection options
    enable_auto_boolean_detection: bool = True
    boolean_detection_min_samples: int = 100
    boolean_detection_max_zero_ratio: float = 0.80
    boolean_detection_require_name_pattern: bool = False
    force_column_types: dict[str, str] | None = None

    # Correlation analysis options
    compute_correlations: bool = True
    corr_threshold: float = 0.5
    corr_max_cols: int = 50
    corr_max_per_col: int = 10

    def __post_init__(self) -> None:
        """Validate at construction. See :meth:`validate`."""
        self.validate()

    def validate(self) -> None:
        """Check every invariant, from wherever the options are about to be used.

        This is called at construction *and* again when the options are handed
        to the engine, because the dataclass is mutable and the second path is
        the one people take:

        ```python
        ComputeOptions(chunk_size=0)     # ValueError: chunk_size must be positive
        c = ComputeOptions()
        c.chunk_size = 0                 # accepted, and profiled happily
        ```

        The constructor guarded a door nobody walks through. Two rules for one
        field is also the class of inconsistency that produces a bug report you
        cannot reproduce, so there is now one rule, called from both places.

        Raises:
            ValueError: If a value is outside its documented range.
            ConfigurationError: If `progress` is not one of its four shapes.
        """
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
        # Validated here as well as on EngineConfig: _to_engine_config falls
        # back to a direct mapping inside a bare `except Exception`, so an error
        # raised further in becomes a silently different configuration rather
        # than a message. Failing at the public boundary is the only place the
        # caller reliably sees it.
        if not (
            self.progress is None
            or isinstance(self.progress, bool)
            or callable(self.progress)
            or self.progress == "auto"
        ):
            raise ConfigurationError(
                "progress must be True, False, 'auto' or a callable, got "
                f"{self.progress!r}"
            )
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
        if not 0 <= self.corr_threshold <= 1:
            raise ValueError("corr_threshold must be between 0 and 1")
        if self.corr_max_cols <= 0:
            raise ValueError("corr_max_cols must be positive")
        if self.corr_max_per_col <= 0:
            raise ValueError("corr_max_per_col must be positive")

    @property
    def checkpoint(self) -> CheckpointView:
        """The five checkpointing settings as one named group.

        A view rather than a nested dataclass: the fields stay where they are,
        so nothing existing breaks, while `options.checkpoint.every_n_chunks`
        gives the concern a name and keeps it out of the way of the settings
        people actually reach for.

        ```python
        options = ComputeOptions()
        options.checkpoint.every_n_chunks = 10
        options.checkpoint.dir = "./checkpoints"
        ```
        """
        return CheckpointView(self)

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
    This class controls various rendering aspects of the report.

    Attributes:
        title: Optional custom title for the HTML report. If not provided,
            defaults to "PySuricata EDA Report". This title appears in both
            the browser tab and the main heading of the report.
        description: Optional user description to display in the summary section.
            If provided, this will be shown below the "Summary" heading with
            consistent styling. Can be used to provide context about the dataset
            or analysis.
    """

    title: str | None = None
    description: str | None = None


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


def _coerce_input(data: DataLike) -> pd.DataFrame | cabc.Iterable:
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
            # Deduplicate column names to prevent engine crash
            # (df[name] returns a DataFrame instead of Series when duplicates exist)
            if data.columns.duplicated().any():
                import warnings

                # Every existing name is already taken, so a generated suffix must
                # never collide with one (e.g. columns ["a", "a", "a_1"] must not
                # rename the second "a" to "a_1").
                taken: set[str] = set(data.columns)
                counters: dict[str, int] = {}
                new_cols: list[str] = []
                emitted: set[str] = set()
                for col in data.columns:
                    if col not in emitted:
                        emitted.add(col)
                        new_cols.append(col)
                        continue
                    suffix = counters.get(col, 0)
                    while True:
                        suffix += 1
                        candidate = f"{col}_{suffix}"
                        if candidate not in taken:
                            break
                    counters[col] = suffix
                    taken.add(candidate)
                    emitted.add(candidate)
                    new_cols.append(candidate)
                # Shallow copy: only the column axis is rewritten, never the blocks.
                data = data.copy(deep=False)
                data.columns = new_cols
                assert not data.columns.duplicated().any(), (
                    "column deduplication failed to produce unique names"
                )
                warnings.warn(
                    "DataFrame has duplicate column names. "
                    "Columns were renamed with numeric suffixes to avoid errors.",
                    stacklevel=3,
                )
            return data
    except ImportError:
        pass

    try:
        import polars as pl

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return data
    except ImportError:
        pass

    # Arrow and DuckDB stream natively, so they are never materialised. Checked
    # by module name rather than by importing pyarrow, which would make every
    # call pay for a dependency the caller may not have.
    #
    # DuckDB first: a relation also exports the Arrow capsule, so the Arrow
    # branch would swallow it and we would lose control of the batch size. That
    # is how this ordering was found -- the DuckDB branch was unreachable, and a
    # coverage report said so before any test did.
    if _sources.is_duckdb_relation(data):
        return _sources.stream_duckdb(data)

    if _sources.is_arrow_source(data):
        return _sources.first_batch_or_stream(_sources.stream_arrow(data))

    # A path is the input people reach for first, and the CLI already accepts
    # one -- `pysuricata profile data.csv` worked while `profile("data.csv")`
    # raised TypeError. Same loader, same formats.
    if isinstance(data, (str, os.PathLike)):
        return _read_path(data)

    if isinstance(data, cabc.Iterable) and not isinstance(
        data, (bytes, bytearray, cabc.Mapping)
    ):
        # A sequence can be looked at without consuming it, so a list of the
        # wrong thing is caught here rather than deep in adapter selection --
        # which reported `Unsupported input type: <class 'int'>` for a `list`
        # argument, describing the first element, as a RuntimeError outside the
        # exception hierarchy.
        if isinstance(data, cabc.Sequence) and data:
            first = data[0]
            if not _is_frame(first):
                raise UnsupportedDataError(
                    f"Cannot profile {type(data).__name__} of "
                    f"{type(first).__name__}. An iterable input must yield "
                    "pandas or polars DataFrame chunks."
                )
        return data

    raise UnsupportedDataError(
        f"Cannot profile {type(data).__name__}. Provide a pandas DataFrame, a "
        "polars DataFrame/LazyFrame, an Arrow table or reader, a DuckDB "
        "relation, a path to a .csv/.parquet/.json file, or an iterable of "
        "DataFrame chunks."
    )


def _is_frame(obj: Any) -> bool:
    """Whether this is a frame the engine can consume, without importing either
    library that is not already loaded."""
    module = type(obj).__module__ or ""
    if module.startswith("pandas"):
        import pandas as pd

        return isinstance(obj, pd.DataFrame)
    if module.startswith("polars"):
        import polars as pl

        return isinstance(obj, (pl.DataFrame, pl.LazyFrame))
    return False


def _read_path(path: str | os.PathLike) -> pd.DataFrame:
    """Load a CSV, Parquet or JSON file into a DataFrame.

    Args:
        path: Path to the file to read.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        PySuricataError: If the file does not exist or the suffix is not one of
            the supported formats.
    """
    import pandas as pd

    resolved = Path(path)
    if not resolved.exists():
        raise PySuricataError(f"File not found: {resolved}")

    suffix = resolved.suffix.lower()

    # Parquet is read a batch at a time rather than loaded whole. It is the one
    # supported format that carries its own row groups, and it is the format
    # people point at large data with -- materialising it would contradict the
    # bounded-memory claim on exactly the input where the claim matters. A file
    # that fits in one batch comes back as a frame, so small files behave
    # exactly as they did.
    if suffix == ".parquet":
        return _sources.first_batch_or_stream(_sources.stream_parquet(resolved))

    readers = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
    }
    reader = readers.get(suffix)
    if reader is None:
        raise UnsupportedDataError(
            f"Cannot read '{resolved.name}': unsupported format "
            f"'{resolved.suffix}'. Use .csv, .parquet or .json, or load it "
            "yourself and pass the DataFrame."
        )
    return reader(resolved)


def _source_name(data: DataLike) -> str:
    """The display name of the input, or empty when it has none.

    Only a path names itself. A DataFrame, an iterable of chunks and a
    generator do not, and inventing something for them ("DataFrame") would be
    worse than showing nothing -- it would look like a real filename.
    """
    if isinstance(data, (str, os.PathLike)):
        try:
            return Path(data).name
        except (TypeError, ValueError):
            return ""
    return ""


def _to_engine_config(cfg: ProfileConfig) -> _EngineConfig:
    """Convert public configuration to internal engine configuration.

    Args:
        cfg: The public configuration.

    Returns:
        The engine configuration.

    Raises:
        ConfigurationError: If a setting is not one the engine can accept. The
            failure has to reach the caller: this used to be a bare
            ``except Exception`` around ``from_options`` with a fallback that
            mapped a *subset* of the fields by hand, so a value that failed
            validation produced not an error but a **different configuration**
            -- silently dropping ``columns``, the correlation options,
            ``progress``, ``engine`` and every boolean-detection option along
            with the offending one. A caller who asked for one column got the
            whole frame and a successful-looking run.
    """
    compute = cfg.compute
    render = cfg.render

    try:
        # The options are mutable, so re-check them here rather than trusting
        # what they were at construction.
        compute.validate()
        engine_config = _EngineConfig.from_options(compute)
    except (TypeError, ValueError) as e:
        raise ConfigurationError(f"invalid compute options: {e}") from e

    engine_config.title = render.title or "PySuricata EDA Report"
    engine_config.description = render.description
    return engine_config


# The six options people actually reach for, mapped to where they live. The
# nesting models the module layout, not intent: nobody thinks "I would like to
# configure the compute subsystem", they think "smaller chunks".
_KEYWORD_OPTIONS = {
    "chunk_size": ("compute", "chunk_size"),
    "columns": ("compute", "columns"),
    "sample": ("compute", "numeric_sample_size"),
    "correlations": ("compute", "compute_correlations"),
    "seed": ("compute", "random_seed"),
    "title": ("render", "title"),
    "progress": ("compute", "progress"),
}

# Field names on the options dataclasses, mapped to the keyword that sets them.
# Somebody who read `ComputeOptions` and typed the field name they found there
# was told it was an unknown option, which is true and useless -- the answer is
# the short name, and the error should say it rather than making them guess
# which of seven keywords corresponds to the field they are looking at.
_OPTION_ALIASES = {
    "numeric_sample_size": "sample",
    "compute_correlations": "correlations",
    "random_seed": "seed",
    "max_uniques": None,
    "top_k": None,
}

# One word for an intent. ydata-profiling's single most-used API feature is
# minimal=True, for exactly this reason.
_PRESETS = {
    "fast": {
        "compute": {
            "numeric_sample_size": 5_000,
            "max_uniques": 1_024,
            "top_k": 20,
            "compute_correlations": False,
        }
    },
    "thorough": {
        "compute": {
            "numeric_sample_size": 50_000,
            "max_uniques": 8_192,
            "top_k": 100,
            "compute_correlations": True,
            "corr_threshold": 0.0,
        }
    },
}


def _unknown_option_message(unknown: list[str]) -> str:
    """Explain a rejected keyword, pointing at the one that works.

    Args:
        unknown: The keywords that were not recognised.

    Returns:
        The error message.
    """
    lines = [f"Unknown option(s): {', '.join(unknown)}."]
    for name in unknown:
        if name in _OPTION_ALIASES:
            keyword = _OPTION_ALIASES[name]
            lines.append(
                f"{name} is a ComputeOptions field; the keyword for it is {keyword}=."
                if keyword
                else f"{name} is a ComputeOptions field with no keyword form; "
                "set it through config=ProfileConfig(compute=ComputeOptions(...))."
            )
    lines.append(f"Available: {', '.join(sorted(_KEYWORD_OPTIONS))}.")
    lines.append("Anything else goes through config=ProfileConfig(...).")
    return " ".join(lines)


def _resolve_config(
    config: ProfileConfig | None, preset: str | None, options: dict[str, Any]
) -> ProfileConfig:
    """Build the effective configuration from a preset and keyword options.

    Precedence, lowest to highest: defaults, preset, keyword options. An
    explicit ``config=`` bypasses all of it -- it is the escape hatch, and a
    caller who built one means it.

    Args:
        config: An explicit configuration, or None.
        preset: ``"fast"``, ``"thorough"`` or None.
        options: Keyword options from :func:`profile` or :func:`summarize`.

    Returns:
        The configuration to run with.

    Raises:
        ConfigurationError: If the preset is unknown or an option is not one of
            the documented keywords.
    """
    if config is not None:
        if not isinstance(config, ProfileConfig):
            # Without this, `profile(df, config="oops")` reached the conversion
            # and surfaced as `AttributeError: 'str' object has no attribute
            # 'compute'` -- an internal detail, outside the exception hierarchy,
            # naming a field the caller has never heard of.
            raise ConfigurationError(
                f"config= must be a ProfileConfig, not {type(config).__name__}. "
                "For individual settings use keyword options, e.g. "
                "profile(df, chunk_size=50_000)."
            )
        if preset or options:
            raise ConfigurationError(
                "Pass either config= or preset=/keyword options, not both. "
                "config= is the full escape hatch and takes everything."
            )
        return config

    if preset is not None and preset not in _PRESETS:
        raise ConfigurationError(
            f"Unknown preset {preset!r}. Available: {', '.join(sorted(_PRESETS))}."
        )

    unknown = set(options) - set(_KEYWORD_OPTIONS)
    if unknown:
        raise ConfigurationError(_unknown_option_message(sorted(unknown)))

    compute: dict[str, Any] = {}
    render: dict[str, Any] = {}
    if preset:
        compute.update(_PRESETS[preset].get("compute", {}))
        render.update(_PRESETS[preset].get("render", {}))
    for name, value in options.items():
        group, field = _KEYWORD_OPTIONS[name]
        (compute if group == "compute" else render)[field] = value

    return ProfileConfig(
        compute=ComputeOptions(**compute),
        render=RenderOptions(**render),
    )


def profile(
    data: DataLike,
    config: ProfileConfig | None = None,
    *,
    preset: str | None = None,
    **options: Any,
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
            This is the full escape hatch and cannot be combined with ``preset``
            or the keyword options below.
        preset: ``"fast"`` or ``"thorough"``. One word for an intent, rather
            than working out which of twenty-one knobs to turn.
        **options: The most-reached-for settings, without the nesting:
            ``chunk_size``, ``columns``, ``sample``, ``correlations``, ``seed``,
            ``title``, ``progress``.

    Returns:
        A :class:`Report` object containing the HTML and the computed stats
        mapping.

    Raises:
        TypeError: If ``data`` is not of a supported type.
        ValueError: If ``data`` is None.
        ConfigurationError: If an option or preset is not recognised, or
            ``config`` is combined with either.

    Examples:
        >>> profile(df)                                        # doctest: +SKIP
        >>> profile(df, chunk_size=50_000, correlations=False)  # doctest: +SKIP
        >>> profile(df, preset="fast")                          # doctest: +SKIP
    """
    if data is None:
        raise ValueError("Input data cannot be None")

    cfg = _resolve_config(config, preset, options)
    inp = _coerce_input(data)  # No more polars-specific wrapping!
    cfg = _to_engine_config(cfg)
    # The header names what was profiled. Only a path carries a name; an
    # in-memory frame has none, and the header renders without one. Set after
    # the conversion so an explicitly configured name is never overwritten.
    if not cfg.dataset_name:
        cfg.dataset_name = _source_name(data)

    # Always compute stats to return machine-readable mapping
    html, summary = report.build_report(inp, config=cfg, return_summary=True)  # type: ignore[misc]

    try:
        stats = dict(summary or {})
    except Exception:
        stats = {"dataset": {}, "columns": {}}
    return Report(html=html, stats=stats)


def summarize(
    data: DataLike,
    config: ProfileConfig | None = None,
    *,
    preset: str | None = None,
    **options: Any,
) -> Mapping[str, Any]:
    """Compute statistics only and return a JSON‑safe mapping.

    This is the programmatic counterpart to :func:`profile` for code paths that
    do not need the HTML report (e.g., CI checks and data quality gates).
    Both pandas and polars DataFrames are processed consistently.

    Args:
        data: Dataset to analyze. Same accepted types as :func:`profile`.
        config: Optional configuration overriding compute/render defaults.
            Cannot be combined with ``preset`` or the keyword options.
        preset: ``"fast"`` or ``"thorough"``.
        **options: Same keyword options as :func:`profile`.

    Returns:
        A nested mapping with dataset‑level and per‑column statistics. The
        result is safe to serialize to JSON.

    Raises:
        TypeError: If ``data`` is not of a supported type.
        ValueError: If ``data`` is None.
        ConfigurationError: If an option or preset is not recognised.
    """
    if data is None:
        raise ValueError("Input data cannot be None")

    cfg = _resolve_config(config, preset, options)
    inp = _coerce_input(data)  # No more polars-specific wrapping!
    cfg = _to_engine_config(cfg)
    # compute-only to skip HTML render
    _html, summary = report.build_report(
        inp, config=cfg, return_summary=True, compute_only=True
    )  # type: ignore[misc]
    stats = dict(summary or {})
    return stats
