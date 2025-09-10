"""Tests for the io.base module."""

import warnings
from typing import Iterator, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from pysuricata.io.base import (
    ChunkingConfig,
    ChunkingConfigurationError,
    ChunkingEngine,
    ChunkingError,
    ChunkingMetrics,
    DataFrameType,
    IterableChunkingStrategy,
    MemoryLimitExceededError,
    PandasChunkingStrategy,
    PolarsChunkingStrategy,
    UnsupportedDataTypeError,
    chunk_data,
    chunk_data_with_metrics,
    iter_chunks,
)


class TestChunkingConfig:
    """Test ChunkingConfig validation and behavior."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 200_000
        assert config.columns is None
        assert config.force_in_memory is False
        assert config.memory_limit_mb == 1024

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=1000,
            columns=["A", "B"],
            force_in_memory=True,
            memory_limit_mb=512,
        )
        assert config.chunk_size == 1000
        assert config.columns == ["A", "B"]
        assert config.force_in_memory is True
        assert config.memory_limit_mb == 512

    def test_invalid_chunk_size(self):
        """Test validation of invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-1)

    def test_large_chunk_size_warning(self):
        """Test warning for very large chunk size."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkingConfig(chunk_size=15_000_000)
            assert len(w) == 1
            assert "Very large chunk_size may cause memory issues" in str(w[0].message)

    def test_empty_columns_error(self):
        """Test error for empty columns list."""
        with pytest.raises(ValueError, match="columns cannot be empty list"):
            ChunkingConfig(columns=[])

    def test_invalid_memory_limit(self):
        """Test validation of invalid memory limit."""
        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            ChunkingConfig(memory_limit_mb=0)

        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            ChunkingConfig(memory_limit_mb=-1)


class TestChunkingMetrics:
    """Test ChunkingMetrics data structure."""

    def test_metrics_creation(self):
        """Test creating metrics object."""
        metrics = ChunkingMetrics(
            total_chunks=10,
            total_rows=1000,
            total_memory_bytes=50000,
            processing_time_seconds=1.5,
            average_chunk_size=100.0,
            data_type=DataFrameType.PANDAS,
        )

        assert metrics.total_chunks == 10
        assert metrics.total_rows == 1000
        assert metrics.total_memory_bytes == 50000
        assert metrics.processing_time_seconds == 1.5
        assert metrics.average_chunk_size == 100.0
        assert metrics.data_type == DataFrameType.PANDAS


class TestIterableChunkingStrategy:
    """Test IterableChunkingStrategy."""

    def test_can_handle_iterable(self):
        """Test can_handle for valid iterables."""
        strategy = IterableChunkingStrategy()

        # Test with a simple list
        assert strategy.can_handle([1, 2, 3]) is False  # Not DataFrames
        assert strategy.can_handle("string") is False
        assert strategy.can_handle(123) is False
        assert strategy.can_handle(b"bytes") is False

    def test_can_handle_empty_iterable(self):
        """Test can_handle for empty iterable."""
        strategy = IterableChunkingStrategy()
        assert strategy.can_handle([]) is False

    def test_chunk_empty_iterable(self):
        """Test chunking empty iterable."""
        strategy = IterableChunkingStrategy()
        config = ChunkingConfig()
        chunks = list(strategy.chunk([], config))
        assert len(chunks) == 0


class TestPandasChunkingStrategy:
    """Test PandasChunkingStrategy."""

    def test_can_handle_non_pandas(self):
        """Test can_handle for non-pandas objects."""
        strategy = PandasChunkingStrategy()
        assert strategy.can_handle("not a dataframe") is False
        assert strategy.can_handle(123) is False

    def test_get_data_type(self):
        """Test get_data_type method."""
        strategy = PandasChunkingStrategy()
        assert strategy.get_data_type() == DataFrameType.PANDAS


class TestPolarsChunkingStrategy:
    """Test PolarsChunkingStrategy."""

    def test_can_handle_non_polars(self):
        """Test can_handle for non-polars objects."""
        strategy = PolarsChunkingStrategy()
        assert strategy.can_handle("not a dataframe") is False
        assert strategy.can_handle(123) is False

    def test_get_data_type(self):
        """Test get_data_type method."""
        strategy = PolarsChunkingStrategy()
        assert strategy.get_data_type() == DataFrameType.POLARS


class TestChunkingEngine:
    """Test ChunkingEngine main functionality."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ChunkingEngine()
        assert len(engine._strategies) == 3
        assert isinstance(engine._strategies[0], IterableChunkingStrategy)
        assert isinstance(engine._strategies[1], PandasChunkingStrategy)
        assert isinstance(engine._strategies[2], PolarsChunkingStrategy)

    def test_chunk_data_unsupported_type(self):
        """Test chunk_data with unsupported data type."""
        engine = ChunkingEngine()

        with pytest.raises(UnsupportedDataTypeError):
            list(engine.chunk_data("not a dataframe"))

    def test_memory_estimation_fallback(self):
        """Test memory estimation fallback."""
        engine = ChunkingEngine()

        mock_chunk = Mock()
        mock_chunk.__len__ = Mock(return_value=100)
        # Remove memory_usage and estimated_size methods
        del mock_chunk.memory_usage
        del mock_chunk.estimated_size

        memory = engine._estimate_memory(mock_chunk)
        assert memory == 10000  # 100 * 100 bytes per row


class TestLegacyInterface:
    """Test legacy iter_chunks function for backward compatibility."""

    def test_iter_chunks_unsupported_type(self):
        """Test iter_chunks with unsupported data type."""
        with pytest.raises(UnsupportedDataTypeError):
            list(iter_chunks("not a dataframe"))

    def test_iter_chunks_with_config(self):
        """Test iter_chunks with configuration."""
        with pytest.raises(UnsupportedDataTypeError):
            list(iter_chunks("not a dataframe", chunk_size=100, columns=["A", "B"]))


class TestModernInterface:
    """Test modern chunk_data and chunk_data_with_metrics functions."""

    def test_chunk_data_unsupported_type(self):
        """Test chunk_data with unsupported data type."""
        with pytest.raises(UnsupportedDataTypeError):
            list(chunk_data("not a dataframe"))

    def test_chunk_data_with_metrics_unsupported_type(self):
        """Test chunk_data_with_metrics with unsupported data type."""
        with pytest.raises(UnsupportedDataTypeError):
            chunk_data_with_metrics("not a dataframe")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unsupported_data_type_error(self):
        """Test UnsupportedDataTypeError."""
        error = UnsupportedDataTypeError("Test error")
        assert isinstance(error, ChunkingError)
        assert str(error) == "Test error"

    def test_chunking_configuration_error(self):
        """Test ChunkingConfigurationError."""
        error = ChunkingConfigurationError("Config error")
        assert isinstance(error, ChunkingError)
        assert str(error) == "Config error"

    def test_memory_limit_exceeded_error(self):
        """Test MemoryLimitExceededError."""
        error = MemoryLimitExceededError("Memory error")
        assert isinstance(error, ChunkingError)
        assert str(error) == "Memory error"


class TestDataFrameType:
    """Test DataFrameType enum."""

    def test_dataframe_type_values(self):
        """Test DataFrameType enum values."""
        assert DataFrameType.PANDAS.value == "pandas"
        assert DataFrameType.POLARS.value == "polars"
        assert DataFrameType.LAZY_POLARS.value == "lazy_polars"
        assert DataFrameType.UNKNOWN.value == "unknown"


class TestPerformanceAndMemory:
    """Test performance and memory-related functionality."""

    def test_memory_estimation_pandas(self):
        """Test memory estimation for pandas DataFrames."""
        engine = ChunkingEngine()

        mock_chunk = Mock()
        mock_chunk.memory_usage = Mock(return_value=Mock(sum=Mock(return_value=5000)))

        memory = engine._estimate_memory(mock_chunk)
        assert memory == 5000

    def test_memory_estimation_polars(self):
        """Test memory estimation for polars DataFrames."""
        engine = ChunkingEngine()

        mock_chunk = Mock()
        # Remove memory_usage method to test polars path
        del mock_chunk.memory_usage
        mock_chunk.estimated_size = Mock(return_value=3000)

        memory = engine._estimate_memory(mock_chunk)
        assert memory == 3000


# Integration tests with real pandas/polars (if available)
class TestIntegration:
    """Integration tests with real libraries if available."""

    def test_pandas_integration(self):
        """Test with real pandas if available."""
        try:
            import pandas as pd

            # Create a simple DataFrame
            df = pd.DataFrame({"A": range(100), "B": range(100)})

            # Test chunking
            chunks = list(iter_chunks(df, chunk_size=25))
            assert len(chunks) == 4  # 100 / 25 = 4 chunks

            # Test column selection
            chunks_with_cols = list(iter_chunks(df, columns=["A"]))
            assert len(chunks_with_cols) == 1  # Single chunk for small DataFrame
            assert list(chunks_with_cols[0].columns) == ["A"]

        except ImportError:
            pytest.skip("pandas not available")

    def test_polars_integration(self):
        """Test with real polars if available."""
        try:
            import polars as pl

            # Create a simple DataFrame
            df = pl.DataFrame({"A": range(100), "B": range(100)})

            # Test chunking
            chunks = list(iter_chunks(df, chunk_size=25))
            assert len(chunks) == 4  # 100 / 25 = 4 chunks

            # Test column selection
            chunks_with_cols = list(iter_chunks(df, columns=["A"]))
            assert len(chunks_with_cols) == 1  # Single chunk for small DataFrame
            assert chunks_with_cols[0].columns == ["A"]

        except ImportError:
            pytest.skip("polars not available")

    def test_iterable_integration(self):
        """Test with iterable of DataFrames."""
        try:
            import pandas as pd

            # Create multiple DataFrames
            dfs = [pd.DataFrame({"A": range(10)}) for _ in range(3)]

            # Test chunking
            chunks = list(iter_chunks(dfs))
            assert len(chunks) == 3

        except ImportError:
            pytest.skip("pandas not available")

    def test_modern_interface_integration(self):
        """Test modern interface with real data."""
        try:
            import pandas as pd

            df = pd.DataFrame({"A": range(100), "B": range(100)})

            # Test chunk_data
            chunks = list(chunk_data(df, chunk_size=25))
            assert len(chunks) == 4

            # Test chunk_data_with_metrics
            chunks, metrics = chunk_data_with_metrics(df, chunk_size=25)
            chunk_list = list(chunks)
            assert len(chunk_list) == 4
            assert metrics.total_chunks == 4
            assert metrics.total_rows == 100
            assert metrics.data_type == DataFrameType.PANDAS
            assert metrics.processing_time_seconds > 0

        except ImportError:
            pytest.skip("pandas not available")


if __name__ == "__main__":
    pytest.main([__file__])
