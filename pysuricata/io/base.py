from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol, Sequence, Union

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# Type aliases for better readability
FrameLike = Any  # engine-native frame (e.g., pandas.DataFrame)


class DataFrameType(Enum):
    """Enumeration of supported DataFrame types."""

    PANDAS = "pandas"
    POLARS = "polars"
    LAZY_POLARS = "lazy_polars"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for chunking operations."""

    chunk_size: int = 200_000
    columns: Optional[Sequence[str]] = None
    force_in_memory: bool = False
    memory_limit_mb: int = 1024

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_size > 10_000_000:
            warnings.warn("Very large chunk_size may cause memory issues", UserWarning)
        if self.columns is not None and not self.columns:
            raise ValueError("columns cannot be empty list")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")


@dataclass(frozen=True)
class ChunkingMetrics:
    """Metrics for chunking operations."""

    total_chunks: int
    total_rows: int
    total_memory_bytes: int
    processing_time_seconds: float
    average_chunk_size: float
    data_type: DataFrameType


class DataFrameProtocol(Protocol):
    """Protocol for DataFrame-like objects."""

    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...


# Custom Exceptions
class ChunkingError(Exception):
    """Base exception for chunking operations."""

    pass


class UnsupportedDataTypeError(ChunkingError):
    """Raised when data type is not supported."""

    pass


class ChunkingConfigurationError(ChunkingError):
    """Raised when configuration is invalid."""

    pass


class MemoryLimitExceededError(ChunkingError):
    """Raised when memory constraints are exceeded."""

    pass


class ChunkingStrategy(ABC):
    """Abstract base class for different chunking strategies."""

    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """Check if this strategy can handle the given data."""
        pass

    @abstractmethod
    def chunk(self, data: Any, config: ChunkingConfig) -> Iterator[DataFrameProtocol]:
        """Generate chunks from the data."""
        pass

    @abstractmethod
    def get_data_type(self) -> DataFrameType:
        """Get the DataFrame type this strategy handles."""
        pass


class IterableChunkingStrategy(ChunkingStrategy):
    """Handles iterables of DataFrames."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is an iterable of DataFrames."""
        if not (
            hasattr(data, "__iter__")
            and not hasattr(data, "__array__")
            and not isinstance(data, (str, bytes, bytearray))
        ):
            return False

        # Peek at first element to validate
        try:
            iterator = iter(data)
            first = next(iterator)
            # Check if first element is a DataFrame
            return self._is_dataframe(first)
        except (StopIteration, Exception):
            return False

    def chunk(self, data: Any, config: ChunkingConfig) -> Iterator[DataFrameProtocol]:
        """Generate chunks from iterable of DataFrames."""
        try:
            iterator = iter(data)
            first = next(iterator)

            # Validate first element
            if not self._is_dataframe(first):
                warnings.warn(
                    "iter_chunks received an iterable whose first element is not a pandas/polars DataFrame; passing through anyway.",
                    RuntimeWarning,
                )

            # Apply column selection to first chunk
            if config.columns is not None:
                first = self._select_columns(first, config.columns)

            yield first

            # Process remaining chunks
            for chunk in iterator:
                if config.columns is not None:
                    chunk = self._select_columns(chunk, config.columns)
                yield chunk

        except StopIteration:
            # Empty iterable
            return
        except Exception as e:
            raise ChunkingError(f"Failed to process iterable: {e}") from e

    def get_data_type(self) -> DataFrameType:
        return DataFrameType.UNKNOWN

    def _is_dataframe(self, obj: Any) -> bool:
        """Check if object is a DataFrame."""
        try:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return True
        except ImportError:
            pass

        try:
            import polars as pl

            if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
                return True
        except ImportError:
            pass

        return False

    def _select_columns(self, df: Any, columns: Sequence[str]) -> Any:
        """Select columns from DataFrame."""
        try:
            # Try pandas-style selection
            if hasattr(df, "iloc"):
                return df[list(columns)]
            # Try polars-style selection
            elif hasattr(df, "select"):
                return df.select(list(columns))
            else:
                return df
        except Exception:
            warnings.warn(
                f"Failed to select columns {columns}, returning original DataFrame"
            )
            return df


class PandasChunkingStrategy(ChunkingStrategy):
    """Handles pandas DataFrames."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a pandas DataFrame."""
        try:
            import pandas as pd

            return isinstance(data, pd.DataFrame)
        except ImportError:
            return False

    def chunk(self, data: Any, config: ChunkingConfig) -> Iterator[DataFrameProtocol]:
        """Generate chunks from pandas DataFrame."""
        try:
            import pandas as pd

            if not isinstance(data, pd.DataFrame):
                raise TypeError("Expected pandas DataFrame")

            n = len(data)
            if n == 0:
                return

            step = config.chunk_size
            for i in range(0, n, step):
                df = data.iloc[i : i + step]

                if config.columns is not None:
                    try:
                        df = df[list(config.columns)]
                    except Exception as e:
                        warnings.warn(f"Failed to select columns {config.columns}: {e}")

                yield df

        except ImportError:
            raise ChunkingError("pandas is required but not installed") from None
        except Exception as e:
            raise ChunkingError(f"Failed to chunk pandas DataFrame: {e}") from e

    def get_data_type(self) -> DataFrameType:
        return DataFrameType.PANDAS


class PolarsChunkingStrategy(ChunkingStrategy):
    """Handles polars DataFrames (both eager and lazy)."""

    def can_handle(self, data: Any) -> bool:
        """Check if data is a polars DataFrame or LazyFrame."""
        try:
            import polars as pl

            return isinstance(data, (pl.DataFrame, pl.LazyFrame))
        except ImportError:
            return False

    def chunk(self, data: Any, config: ChunkingConfig) -> Iterator[DataFrameProtocol]:
        """Generate chunks from polars DataFrame or LazyFrame."""
        try:
            import polars as pl

            if isinstance(data, pl.DataFrame):
                return self._chunk_eager_dataframe(data, config)
            elif isinstance(data, pl.LazyFrame):
                return self._chunk_lazy_dataframe(data, config)
            else:
                raise TypeError("Expected polars DataFrame or LazyFrame")

        except ImportError:
            raise ChunkingError("polars is required but not installed") from None
        except Exception as e:
            raise ChunkingError(f"Failed to chunk polars DataFrame: {e}") from e

    def _chunk_eager_dataframe(
        self, data: Any, config: ChunkingConfig
    ) -> Iterator[DataFrameProtocol]:
        """Chunk eager polars DataFrame."""
        import polars as pl

        step = config.chunk_size
        n = data.height
        use_cols = list(config.columns) if config.columns is not None else None

        for i in range(0, n, step):
            chunk = data.slice(i, min(step, n - i))

            if use_cols is not None:
                try:
                    chunk = chunk.select(use_cols)
                except Exception as e:
                    warnings.warn(f"Failed to select columns {use_cols}: {e}")

            yield chunk

    def _chunk_lazy_dataframe(
        self, data: Any, config: ChunkingConfig
    ) -> Iterator[DataFrameProtocol]:
        """Chunk lazy polars DataFrame."""
        import polars as pl

        lf = data
        if config.columns is not None:
            try:
                lf = lf.select(list(config.columns))
            except Exception as e:
                warnings.warn(f"Failed to select columns {config.columns}: {e}")

        step = config.chunk_size
        offset = 0

        while True:
            try:
                chunk = lf.slice(offset, step).collect()
            except Exception as e:
                warnings.warn(f"Failed to collect chunk at offset {offset}: {e}")
                break

            height = getattr(chunk, "height", None)
            if height is None:
                try:
                    height = len(chunk)
                except Exception:
                    height = 0

            if not height:
                break

            yield chunk

            if height < step:
                break

            offset += step

    def get_data_type(self) -> DataFrameType:
        return DataFrameType.POLARS


class ChunkingEngine:
    """Main engine for chunking different data types."""

    def __init__(self):
        """Initialize the chunking engine with available strategies."""
        self._strategies = [
            IterableChunkingStrategy(),
            PandasChunkingStrategy(),
            PolarsChunkingStrategy(),
        ]

    def chunk_data(
        self, data: Any, config: Optional[ChunkingConfig] = None
    ) -> Iterator[DataFrameProtocol]:
        """
        Main entry point for chunking data.

        Args:
            data: Input data to chunk
            config: Chunking configuration

        Returns:
            Iterator of DataFrame chunks

        Raises:
            UnsupportedDataTypeError: If data type is not supported
            ChunkingConfigurationError: If configuration is invalid
        """
        if config is None:
            config = ChunkingConfig()

        for strategy in self._strategies:
            if strategy.can_handle(data):
                return strategy.chunk(data, config)

        raise UnsupportedDataTypeError(
            f"Unsupported input type: {type(data)}. "
            "Supported types: pandas.DataFrame, polars.DataFrame, polars.LazyFrame, "
            "or iterable of DataFrames."
        )

    def chunk_data_with_metrics(
        self, data: Any, config: Optional[ChunkingConfig] = None
    ) -> tuple[Iterator[DataFrameProtocol], ChunkingMetrics]:
        """
        Chunk data and return metrics.

        Args:
            data: Input data to chunk
            config: Chunking configuration

        Returns:
            Tuple of (chunk iterator, metrics)
        """
        start_time = time.time()

        # Get the appropriate strategy to determine data type
        data_type = DataFrameType.UNKNOWN
        for strategy in self._strategies:
            if strategy.can_handle(data):
                data_type = strategy.get_data_type()
                break

        chunks = self.chunk_data(data, config)
        chunk_list = list(chunks)

        total_rows = sum(len(chunk) for chunk in chunk_list)
        total_memory = sum(self._estimate_memory(chunk) for chunk in chunk_list)

        metrics = ChunkingMetrics(
            total_chunks=len(chunk_list),
            total_rows=total_rows,
            total_memory_bytes=total_memory,
            processing_time_seconds=time.time() - start_time,
            average_chunk_size=total_rows / len(chunk_list) if chunk_list else 0,
            data_type=data_type,
        )

        return iter(chunk_list), metrics

    def _estimate_memory(self, chunk: Any) -> int:
        """Estimate memory usage of a chunk in bytes."""
        try:
            # Try to get memory usage from pandas
            if hasattr(chunk, "memory_usage"):
                return chunk.memory_usage(deep=True).sum()
            # Try to get memory usage from polars
            elif hasattr(chunk, "estimated_size"):
                return chunk.estimated_size()
            # Fallback: rough estimate based on size
            else:
                return len(chunk) * 100  # Rough estimate: 100 bytes per row
        except Exception:
            return len(chunk) * 100


# Global engine instance for convenience
_default_engine = ChunkingEngine()


def iter_chunks(
    data: Union[FrameLike, Iterator[FrameLike]],
    *,
    chunk_size: Optional[int] = 200_000,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[FrameLike]:
    """
    Yield DataFrame chunks from in-memory objects only.

    This is the legacy function for backward compatibility.

    Args:
        data: Input data (DataFrame or iterable of DataFrames)
        chunk_size: Size of each chunk
        columns: Optional columns to select

    Returns:
        Iterator of DataFrame chunks

    Raises:
        UnsupportedDataTypeError: If data type is not supported
    """
    config = ChunkingConfig(chunk_size=chunk_size or 200_000, columns=columns)
    return _default_engine.chunk_data(data, config)


def chunk_data(
    data: Any,
    *,
    chunk_size: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    force_in_memory: bool = False,
    memory_limit_mb: int = 1024,
) -> Iterator[DataFrameProtocol]:
    """
    Modern interface for chunking data with full configuration options.

    Args:
        data: Input data to chunk
        chunk_size: Size of each chunk (default: 200,000)
        columns: Optional columns to select
        force_in_memory: Whether to force in-memory processing
        memory_limit_mb: Memory limit in MB

    Returns:
        Iterator of DataFrame chunks
    """
    config = ChunkingConfig(
        chunk_size=chunk_size or 200_000,
        columns=columns,
        force_in_memory=force_in_memory,
        memory_limit_mb=memory_limit_mb,
    )
    return _default_engine.chunk_data(data, config)


def chunk_data_with_metrics(
    data: Any,
    *,
    chunk_size: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    force_in_memory: bool = False,
    memory_limit_mb: int = 1024,
) -> tuple[Iterator[DataFrameProtocol], ChunkingMetrics]:
    """
    Chunk data and return metrics.

    Args:
        data: Input data to chunk
        chunk_size: Size of each chunk (default: 200,000)
        columns: Optional columns to select
        force_in_memory: Whether to force in-memory processing
        memory_limit_mb: Memory limit in MB

    Returns:
        Tuple of (chunk iterator, metrics)
    """
    config = ChunkingConfig(
        chunk_size=chunk_size or 200_000,
        columns=columns,
        force_in_memory=force_in_memory,
        memory_limit_mb=memory_limit_mb,
    )
    return _default_engine.chunk_data_with_metrics(data, config)
