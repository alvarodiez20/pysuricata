"""Streaming engine for data processing orchestration.

This module provides the main streaming engine that orchestrates data processing
operations, including adapter selection, chunking, and streaming coordination.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any

from ... import progress as _progress
from ..adapters import PandasAdapter, PolarsAdapter
from ..core.protocols import DataAdapter
from ..core.types import ProcessingResult
from ..processing.chunking import AdaptiveChunker, ChunkingStrategy

# Import pandas and polars for type checking
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


def _is_exhaustible_iterator(source: Any) -> bool:
    """Return True if consuming from ``source`` permanently loses those items.

    Generators and file readers are their own iterator, so anything read out of
    them during adapter sniffing can never be read again. Re-iterable containers
    (lists, tuples) and DataFrames hand out a fresh iterator each time and are
    therefore safe to peek at directly.
    """
    if isinstance(source, (str, bytes)):
        return False
    if pd is not None and isinstance(source, pd.DataFrame):
        return False
    if pl is not None and isinstance(source, (pl.DataFrame, pl.LazyFrame)):
        return False
    try:
        return iter(source) is source
    except TypeError:
        return False


def _missing_so_far(accs: dict[str, Any]) -> int:
    """Total missing cells counted by the accumulators up to now.

    Every accumulator already counts the nulls it skipped while folding a chunk
    in, so the dataset total is a sum over columns rather than a second pass
    over every cell.

    Args:
        accs: The per-column accumulators.

    Returns:
        The running total across all profiled columns.
    """
    return sum(int(getattr(acc, "missing", 0)) for acc in accs.values())


def _mark_chunk_boundary(accs) -> None:
    """Tell every accumulator that a chunk ended.

    `mark_chunk_boundary()` was only ever called from `finalize()`, so the
    per-column boundaries recorded the number of *renders* rather than the
    number of chunks: an uninterrupted run produced one, and a checkpointed one
    produced two. Neither is the chunk count (#139).

    The consequence reached the page. A `Missing Values` pane on a column with
    no missing values drew a severity-coloured segment reading `data-missing`
    1563 on an 891-row frame -- 175.4% -- because a single boundary accumulated
    every chunk's counter into one segment sized as if it were one chunk.

    Duck-typed rather than gated on a type: only the numeric accumulator
    implements it today, and the others should start working the moment they do.
    """
    for acc in accs.values():
        mark = getattr(acc, "mark_chunk_boundary", None)
        if mark is not None:
            mark()


def _schema_only_frame(source: Any) -> Any | None:
    """The source itself when it is a frame with columns and no rows.

    A zero-row frame is **not** an empty source. Its schema is known --
    `pd.DataFrame({"a": pd.Series([], dtype="float64")})` has a column named
    `a` of dtype `float64` -- and `summarize()` used to throw all of it away
    and return `{}`, with no `schema_version`, no `dataset` and no `columns`
    (#315). That is the one part of the surface `docs/versioning.md`
    guarantees, returning silence rather than an error, on the shape a filter
    matching nothing produces routinely.

    Returning the frame here lets the ordinary path run over an empty chunk:
    inference types each column from its dtype, the accumulators fold in zero
    values, and `finalize()` reports counts of zero. Nothing downstream needs
    a special case.

    Returns:
        The frame when it has at least one column and no rows, otherwise
        `None` -- an exhausted generator or a frame with no columns knows
        nothing about a schema, and for those "Empty source" is still the
        honest answer.
    """
    for module in (pd, pl):
        if module is None:
            continue
        frame_type = getattr(module, "DataFrame", None)
        if frame_type is None or not isinstance(source, frame_type):
            continue
        try:
            if len(source) == 0 and len(source.columns) > 0:
                return source
        except Exception:  # pragma: no cover - a frame that cannot be measured
            return None
    return None


def _select_columns(chunk: Any, columns: tuple[str, ...] | None) -> Any:
    """Restrict a chunk to the configured column subset.

    ``ComputeOptions.columns`` was documented and validated but never reached
    the engine, so asking for three columns of a hundred profiled all hundred.
    Applied per chunk rather than once at the source, because a streaming source
    has no single frame to subset.

    Names that are not present are ignored rather than raising: a stream may
    legitimately vary, and refusing to profile anything because one column is
    missing from one chunk would be worse than profiling what is there.
    """
    if not columns:
        return chunk
    available = getattr(chunk, "columns", None)
    if available is None:
        return chunk
    keep = [c for c in columns if c in set(available)]
    if not keep or len(keep) == len(list(available)):
        return chunk
    try:
        return chunk[keep]
    except Exception:
        return chunk


def _dataset_is_fully_known(source: Any) -> bool:
    """Return True when every row is available before processing starts.

    Type inference reclassifies a numeric column as categorical from the
    distinct values it can see. That is sound evidence when the whole column is
    in hand and unsound otherwise: a sorted stream, or one with a leading run of
    a single value, gives a prefix that looks low-cardinality while the column
    is not -- and the decision is never revisited.

    The question is about the *source*, not about the chunk. An in-memory frame
    is fully known however the engine chooses to split it; asking whether the
    first chunk happened to hold every row instead made classification depend on
    `chunk_size`, so the same column came back numeric at 50,000 rows and
    categorical at 200,000. An exhaustible iterator is streaming even if it
    happens to yield a single chunk, because finding that out would consume it.
    """
    try:
        if pd is not None and isinstance(source, pd.DataFrame):
            return True
        if pl is not None and isinstance(source, (pl.DataFrame, pl.LazyFrame)):
            return True
    except Exception:
        return False
    return False


class EngineManager:
    """Manages engine adapters and their selection.

    This class is responsible for discovering and registering available engine
    adapters (e.g., for pandas and polars) and selecting the most appropriate
    one for a given data source. This allows the rest of the system to be
    agnostic to the underlying data-handling library.

    Attributes:
        logger: A logger instance for logging messages related to adapter
            management and selection.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initializes the EngineManager.

        Args:
            logger: An optional logger instance. If not provided, a new logger
                will be created.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._adapters: dict[str, DataAdapter] = {}
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Discovers and registers the default engine adapters.

        This method attempts to import pandas and polars and, if successful,
        registers the corresponding adapters. It logs a warning if a library
        is not found.
        """
        try:
            import pandas as pd  # noqa: F401

            self._adapters["pandas"] = PandasAdapter()
        except ImportError:
            self.logger.warning("pandas not available, skipping pandas adapter")

        try:
            import polars as pl  # noqa: F401

            self._adapters["polars"] = PolarsAdapter()
        except ImportError:
            self.logger.warning("polars not available, skipping polars adapter")

    def select_adapter(self, data: Any) -> ProcessingResult[DataAdapter]:
        """Selects the most appropriate adapter for the given data.

        This method inspects the type of the input data to determine which
        engine adapter to use. It supports pandas DataFrames, polars
        DataFrames, and iterables of DataFrames.

        Args:
            data: The input data to analyze.

        Returns:
            A `ProcessingResult` containing the selected `DataAdapter` on
            success, or an error message on failure.
        """
        start_time = time.time()

        try:
            # Check for pandas DataFrame
            if (
                "pandas" in self._adapters
                and pd is not None
                and isinstance(data, pd.DataFrame)
            ):
                adapter = self._adapters["pandas"]
                duration = time.time() - start_time
                return ProcessingResult.success_result(
                    data=adapter,
                    metrics={
                        "adapter_type": "pandas",
                        "selection_reason": "pandas_dataframe",
                    },
                    duration=duration,
                )

            # Check for polars DataFrame or LazyFrame
            if "polars" in self._adapters and pl is not None:
                # Collect LazyFrame to eager DataFrame
                if isinstance(data, pl.LazyFrame):
                    data = data.collect()
                if isinstance(data, pl.DataFrame):
                    adapter = self._adapters["polars"]
                    duration = time.time() - start_time
                    return ProcessingResult.success_result(
                        data=adapter,
                        metrics={
                            "adapter_type": "polars",
                            "selection_reason": "polars_dataframe",
                        },
                        duration=duration,
                    )

            # Check for iterable of DataFrames
            if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
                try:
                    first_item = next(iter(data))
                    return self.select_adapter(first_item)
                except StopIteration:
                    return ProcessingResult.error_result("Empty source")

            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Unsupported input type: {type(data)}",
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Adapter selection failed: {str(e)}",
                duration=duration,
            )

    def get_adapter_tag(self, adapter: DataAdapter) -> str:
        """Returns a short string tag for the given adapter.

        This is useful for logging and debugging.

        Args:
            adapter: The adapter to get a tag for.

        Returns:
            A short string tag for the adapter (e.g., "pandas", "polars").
        """
        return adapter.__class__.__name__.lower().replace("adapter", "")

    def register_adapter(self, name: str, adapter: DataAdapter) -> None:
        """Registers a custom engine adapter.

        This allows users to extend the library with support for new data
        sources or data-handling libraries.

        Args:
            name: The name to register the adapter under.
            adapter: The adapter instance to register.
        """
        self._adapters[name] = adapter
        self.logger.info(f"Registered adapter: {name}")

    def get_available_adapters(self) -> dict[str, str]:
        """Returns a dictionary of available engine adapters.

        Returns:
            A dictionary mapping adapter names to their class names.
        """
        return {
            name: adapter.__class__.__name__ for name, adapter in self._adapters.items()
        }


class StreamingEngine:
    """Orchestrates the data processing pipeline for streaming data.

    This class is the core of the data processing functionality. It coordinates
    the various components of the system, including the `EngineManager` for
    adapter selection and the `AdaptiveChunker` for data chunking. It is
    responsible for processing a data stream end-to-end and returning the
    results.

    Attributes:
        engine_manager: An `EngineManager` instance for managing adapters.
        chunker: An `AdaptiveChunker` instance for chunking data.
        logger: A logger instance for logging messages.
    """

    def __init__(
        self,
        engine_manager: EngineManager | None = None,
        chunker: AdaptiveChunker | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initializes the StreamingEngine.

        Args:
            engine_manager: An optional `EngineManager` instance. If not
                provided, a new one will be created.
            chunker: An optional `AdaptiveChunker` instance. If not provided,
                a new one will be created.
            logger: An optional logger instance. If not provided, a new logger
                will be created.
        """
        self.engine_manager = engine_manager or EngineManager(logger)
        self.chunker = chunker or AdaptiveChunker(
            strategy=ChunkingStrategy.ADAPTIVE, logger=logger
        )
        self.logger = logger or logging.getLogger(__name__)

    def process_stream(
        self,
        source: Any,
        config: Any,
        row_kmv: Any,
    ) -> ProcessingResult[tuple]:
        """Processes a data stream from end to end.

        This method orchestrates the entire data processing pipeline for a given
        data source. It performs the following steps:

        1.  Selects the appropriate engine adapter for the data source.
        2.  Generates chunks of data from the source.
        3.  Initializes accumulators and other metrics based on the first chunk.
        4.  Processes the remaining chunks, updating the accumulators and
            metrics.
        5.  Returns a `ProcessingResult` containing a tuple of the computed
            statistics and metadata.

        Args:
            source: The data source to process. This can be a pandas DataFrame,
                a polars DataFrame, or an iterable of DataFrames.
            config: A configuration object with settings for the processing.
            row_kmv: A `RowKMV` instance for estimating the number of unique
                rows.

        Returns:
            A `ProcessingResult` containing a tuple with the following items:
            - `kinds`: A `ColumnKinds` object with the inferred types of the
              columns.
            - `accs`: A dictionary of accumulators for each column.
            - `n_rows`: The total number of rows processed.
            - `n_cols`: The total number of columns.
            - `total_missing_cells`: The total number of missing cells.
            - `approx_mem_bytes`: An estimate of the memory usage in bytes.
            - `first_columns`: A list of the column names from the first
              chunk.
            - `sample_section_html`: The HTML for the sample data section.
            - `corr_est`: A streaming correlation estimator.
        """
        start_time = time.time()

        try:
            # Adapter sniffing has to look at a real chunk, but reading one out of
            # a generator would consume it for good -- the chunk loop below would
            # then start at chunk 1 and every statistic would silently omit chunk 0.
            # Peek once, sniff from the peeked chunk, and splice it back onto the
            # front of the stream.
            sniff_target = source
            fully_known = _dataset_is_fully_known(source)
            if _is_exhaustible_iterator(source):
                stream = iter(source)
                try:
                    first_chunk_peek = next(stream)
                except StopIteration:
                    return ProcessingResult.error_result("Empty source")
                sniff_target = first_chunk_peek
                source = itertools.chain([first_chunk_peek], stream)

            # Select appropriate adapter
            adapter_result = self.engine_manager.select_adapter(sniff_target)
            if not adapter_result.success:
                return ProcessingResult.error_result(
                    f"Adapter selection failed: {adapter_result.error}"
                )

            adapter = adapter_result.data

            # Collect LazyFrame to eager DataFrame before processing
            if pl is not None and isinstance(source, pl.LazyFrame):
                source = source.collect()

            # Generate chunks
            chunk_result = self.chunker.chunks_from_source(
                source, config.chunk_size, config.force_chunk_in_memory
            )
            if not chunk_result.success:
                return ProcessingResult.error_result(
                    f"Chunking failed: {chunk_result.error}"
                )

            chunks = chunk_result.data

            # Process first chunk to initialize
            try:
                first_chunk = next(chunks)
            except StopIteration:
                # No chunks is not the same as nothing to say. A frame with
                # columns and zero rows still has a schema, and the payload is
                # a contract (#315) -- run the ordinary path over one empty
                # chunk rather than erroring out with the columns in hand.
                first_chunk = _schema_only_frame(source)
                if first_chunk is None:
                    return ProcessingResult.error_result("Empty source")

            column_subset = getattr(config, "columns", None)
            first_chunk = _select_columns(first_chunk, column_subset)

            # Whether this first chunk is the entire dataset decides how much
            # weight type inference may put on it. If more chunks follow, the
            # chunk's distinct-value ratio says nothing reliable about the
            # column, so reclassification heuristics must not fire.
            # Asked of the original source: a peeked chunk is a DataFrame even
            # when it came off a generator, so sniffing it would call every
            # stream fully known.
            first_chunk_is_whole = fully_known

            kinds, accs = adapter.infer_and_build(
                first_chunk, config, first_chunk_is_whole=first_chunk_is_whole
            )
            corr_est = self.maybe_corr_estimator(kinds, config)

            # Process first chunk
            adapter.consume_chunk(
                first_chunk, accs, kinds, config, self.logger, row_offset=0
            )
            _mark_chunk_boundary(accs)
            if corr_est is not None:
                adapter.update_corr(first_chunk, corr_est, self.logger)
            adapter.update_row_kmv(first_chunk, row_kmv)

            # Initialize metrics
            n_rows = len(first_chunk) if hasattr(first_chunk, "__len__") else 0
            n_cols = len(first_chunk.columns) if hasattr(first_chunk, "columns") else 0
            first_columns = list(getattr(first_chunk, "columns", []))
            sample_section_html = adapter.sample_section_html(first_chunk, config)

            # Missing cells come from the accumulators, which counted them while
            # folding the chunk in. `adapter.missing_cells()` is a full
            # `isnull().sum().sum()` over the frame -- a second pass over every
            # cell, for a number already computed. The first chunk used to pay
            # for it twice over.
            total_missing_cells = _missing_so_far(accs)

            # Initialize chunk metadata collection
            chunk_metadata = []
            current_row = 0

            # Process first chunk
            chunk_size = len(first_chunk) if hasattr(first_chunk, "__len__") else 0
            chunk_metadata.append(
                (current_row, current_row + chunk_size - 1, total_missing_cells)
            )
            current_row += chunk_size

            # Initialize checkpoint manager if configured
            checkpoint_manager = None
            if config.checkpoint_every_n_chunks > 0:
                from ...checkpoint import maybe_make_manager

                checkpoint_manager = maybe_make_manager(config, None)

            # Track chunk index for logging and checkpointing
            chunk_idx = 1  # First chunk already processed

            # A hung process and a working one look identical without this. The
            # row total is only knowable for an in-memory frame; a generator
            # source gets a counter and a rate, and no invented ETA.
            reporter = _progress.resolve(getattr(config, "progress", False))
            total_rows = None
            try:
                if _dataset_is_fully_known(sniff_target):
                    total_rows = len(sniff_target)
            except Exception:
                total_rows = None
            reporter.start(total_rows)
            reporter.advance(chunk_idx, n_rows)

            # Process remaining chunks
            for chunk in chunks:
                chunk = _select_columns(chunk, column_subset)
                # current_row is the global index of this chunk's first row, so
                # extreme-value indices come out global rather than chunk-local.
                adapter.consume_chunk(
                    chunk, accs, kinds, config, self.logger, row_offset=current_row
                )
                _mark_chunk_boundary(accs)
                if corr_est is not None:
                    adapter.update_corr(chunk, corr_est, self.logger)
                adapter.update_row_kmv(chunk, row_kmv)

                chunk_size = len(chunk) if hasattr(chunk, "__len__") else 0
                # The accumulators carry a running total, so this chunk's share
                # is the difference. No second pass over the data.
                missing_so_far = _missing_so_far(accs)
                chunk_missing = missing_so_far - total_missing_cells
                chunk_metadata.append(
                    (current_row, current_row + chunk_size - 1, chunk_missing)
                )
                current_row += chunk_size

                n_rows += chunk_size
                total_missing_cells = missing_so_far

                # Increment chunk counter
                chunk_idx += 1
                reporter.advance(chunk_idx, n_rows)

                # Log progress every N chunks
                if (
                    config.log_every_n_chunks > 0
                    and chunk_idx % config.log_every_n_chunks == 0
                ):
                    self.logger.info(
                        "Processed chunk %d: %d rows total, %d missing cells",
                        chunk_idx,
                        n_rows,
                        total_missing_cells,
                    )

                # Create checkpoint every N chunks
                if checkpoint_manager and config.checkpoint_every_n_chunks > 0:
                    if chunk_idx % config.checkpoint_every_n_chunks == 0:
                        from ...checkpoint import make_state_snapshot

                        # Calculate memory usage on-demand for checkpointing
                        checkpoint_mem_bytes = 0
                        for acc in accs.values():
                            if hasattr(acc, "_bytes_seen"):
                                checkpoint_mem_bytes += acc._bytes_seen
                            elif hasattr(acc, "_mem_bytes"):
                                checkpoint_mem_bytes += acc._mem_bytes

                        state = make_state_snapshot(
                            kinds=kinds,
                            accs=accs,
                            row_kmv=row_kmv,
                            total_missing_cells=total_missing_cells,
                            approx_mem_bytes=checkpoint_mem_bytes,
                            chunk_idx=chunk_idx,
                            first_columns=first_columns,
                            sample_section_html=sample_section_html,
                            cfg=config,
                        )
                        # Generate partial HTML if configured
                        html_snapshot = None
                        if config.checkpoint_write_html:
                            from ...render.html import render_html_snapshot

                            html_snapshot = render_html_snapshot(
                                kinds=kinds,
                                accs=accs,
                                first_columns=first_columns,
                                row_kmv=row_kmv,
                                total_missing_cells=total_missing_cells,
                                approx_mem_bytes=checkpoint_mem_bytes,
                                start_time=start_time,
                                cfg=config,
                                report_title=f"Checkpoint at Chunk {chunk_idx}",
                                sample_section_html=sample_section_html,
                                chunk_metadata=chunk_metadata,
                                corr_est=corr_est,
                            )

                        checkpoint_manager.save(chunk_idx, state, html_snapshot)
                        self.logger.info("Checkpoint created at chunk %d", chunk_idx)

            # Calculate total memory as sum of all column memories
            # This ensures Total Dataset Memory = Sum of All Column Memories
            approx_mem_bytes = 0
            for acc in accs.values():
                if hasattr(acc, "_bytes_seen"):
                    approx_mem_bytes += acc._bytes_seen
                elif hasattr(acc, "_mem_bytes"):
                    approx_mem_bytes += acc._mem_bytes

            duration = time.time() - start_time
            reporter.finish(chunk_idx, n_rows, n_rows * max(1, n_cols))

            return ProcessingResult.success_result(
                data=(
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
                ),
                metrics={
                    "processing_time": duration,
                    "chunks_processed": chunk_result.metrics.get("chunk_size", 0),
                    "adapter_type": adapter.__class__.__name__,
                },
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ProcessingResult.error_result(
                f"Stream processing failed: {str(e)}",
                duration=duration,
            )

    def maybe_corr_estimator(self, kinds, config) -> Any | None:
        """Creates a streaming correlation estimator if conditions are met.

        A correlation estimator is created if the following conditions are met:

        - The `compute_correlations` option is enabled in the configuration.
        - There are at least two numeric columns in the dataset.

        Args:
            kinds: A `ColumnKinds` object with the inferred types of the
                columns.
            config: A configuration object with settings for the processing.

        Returns:
            A `StreamingCorr` instance if the conditions are met, otherwise
            `None`.
        """
        try:
            from ..analysis.correlation import StreamingCorr

            if len(kinds.numeric) < 2:
                return None

            if not config.compute_correlations:
                return None

            # corr_max_cols was declared, documented, validated and copied into
            # the config, then never read -- so a 1,000-column frame built
            # 499,500 pairs despite a documented cap of 50. The cap has to apply
            # here, before pair construction, because that is the quadratic part.
            max_cols = int(getattr(config, "corr_max_cols", 0) or 0)
            columns = list(kinds.numeric)
            if 0 < max_cols < len(columns):
                self.logger.info(
                    "correlations limited to the first %d of %d numeric columns "
                    "(corr_max_cols)",
                    max_cols,
                    len(columns),
                )
                columns = columns[:max_cols]

            return StreamingCorr(columns)

        except Exception as e:
            self.logger.warning(f"Failed to create correlation estimator: {e}")
            return None

    def get_engine_info(self) -> dict[str, Any]:
        """Returns a dictionary with information about the engine.

        This information can be used for debugging and monitoring.

        Returns:
            A dictionary with the following keys:
            - `available_adapters`: A dictionary of available engine adapters.
            - `chunker_strategy`: The chunking strategy being used.
            - `chunker_metrics`: A dictionary of metrics from the chunker.
        """
        return {
            "available_adapters": self.engine_manager.get_available_adapters(),
            "chunker_strategy": self.chunker.strategy.value,
            "chunker_metrics": self.chunker.get_performance_metrics(),
        }
