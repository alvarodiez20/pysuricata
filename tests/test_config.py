"""Tests for the configuration module."""

import logging

import pytest

from pysuricata.config import EngineConfig, EngineOptions


class TestEngineConfig:
    """Test the EngineConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EngineConfig()

        # Test default values
        assert config.title == "PySuricata EDA Report"
        assert config.chunk_size == 200_000
        assert config.numeric_sample_k == 20_000
        assert config.uniques_k == 2048
        assert config.topk_k == 50
        assert config.engine == "auto"
        assert config.random_seed is None
        assert config.logger is None
        assert config.log_level == logging.INFO
        assert config.log_every_n_chunks == 1
        assert config.include_sample is True
        assert config.sample_rows == 10
        assert config.compute_correlations is True
        assert config.corr_threshold == 0.6
        assert config.corr_max_cols == 50
        assert config.corr_max_per_col == 2
        assert config.checkpoint_every_n_chunks == 0
        assert config.checkpoint_dir is None
        assert config.checkpoint_prefix == "pysuricata_ckpt"
        assert config.checkpoint_write_html is False
        assert config.checkpoint_max_to_keep == 3
        assert config.force_chunk_in_memory is False

    def test_custom_config(self):
        """Test custom configuration values."""
        logger = logging.getLogger("test")
        config = EngineConfig(
            title="Custom Report",
            chunk_size=100_000,
            numeric_sample_k=10_000,
            uniques_k=1024,
            topk_k=25,
            engine="pandas",
            random_seed=42,
            logger=logger,
            log_level=logging.DEBUG,
            log_every_n_chunks=5,
            include_sample=False,
            sample_rows=20,
            compute_correlations=False,
            corr_threshold=0.2,
            corr_max_cols=50,
            corr_max_per_col=3,
            checkpoint_every_n_chunks=10,
            checkpoint_dir="/tmp/checkpoints",
            checkpoint_prefix="custom_ckpt",
            checkpoint_write_html=True,
            checkpoint_max_to_keep=5,
            force_chunk_in_memory=True,
        )

        assert config.title == "Custom Report"
        assert config.chunk_size == 100_000
        assert config.numeric_sample_k == 10_000
        assert config.uniques_k == 1024
        assert config.topk_k == 25
        assert config.engine == "pandas"
        assert config.random_seed == 42
        assert config.logger is logger
        assert config.log_level == logging.DEBUG
        assert config.log_every_n_chunks == 5
        assert config.include_sample is False
        assert config.sample_rows == 20
        assert config.compute_correlations is False
        assert config.corr_threshold == 0.2
        assert config.corr_max_cols == 50
        assert config.corr_max_per_col == 3
        assert config.checkpoint_every_n_chunks == 10
        assert config.checkpoint_dir == "/tmp/checkpoints"
        assert config.checkpoint_prefix == "custom_ckpt"
        assert config.checkpoint_write_html is True
        assert config.checkpoint_max_to_keep == 5
        assert config.force_chunk_in_memory is True

    def test_validation_negative_values(self):
        """Test validation with negative values."""
        # Test negative chunk_size
        with pytest.raises(
            ValueError, match="chunk_size must be a non-negative integer"
        ):
            EngineConfig(chunk_size=-1)

        # Test negative numeric_sample_k
        with pytest.raises(
            ValueError, match="numeric_sample_k must be a non-negative integer"
        ):
            EngineConfig(numeric_sample_k=-1)

        # Test negative uniques_k
        with pytest.raises(
            ValueError, match="uniques_k must be a non-negative integer"
        ):
            EngineConfig(uniques_k=-1)

        # Test negative topk_k
        with pytest.raises(ValueError, match="topk_k must be a non-negative integer"):
            EngineConfig(topk_k=-1)

        # Test negative sample_rows
        with pytest.raises(
            ValueError, match="sample_rows must be a non-negative integer"
        ):
            EngineConfig(sample_rows=-1)

        # Test negative log_every_n_chunks
        with pytest.raises(
            ValueError, match="log_every_n_chunks must be a non-negative integer"
        ):
            EngineConfig(log_every_n_chunks=-1)

        # Test negative checkpoint_every_n_chunks
        with pytest.raises(
            ValueError, match="checkpoint_every_n_chunks must be a non-negative integer"
        ):
            EngineConfig(checkpoint_every_n_chunks=-1)

        # Test negative checkpoint_max_to_keep
        with pytest.raises(
            ValueError, match="checkpoint_max_to_keep must be a non-negative integer"
        ):
            EngineConfig(checkpoint_max_to_keep=-1)

    def test_validation_correlation_threshold(self):
        """Test correlation threshold validation."""
        # Test threshold below 0.0
        with pytest.raises(
            ValueError, match="corr_threshold must be between 0.0 and 1.0"
        ):
            EngineConfig(corr_threshold=-0.1)

        # Test threshold above 1.0
        with pytest.raises(
            ValueError, match="corr_threshold must be between 0.0 and 1.0"
        ):
            EngineConfig(corr_threshold=1.1)

        # Test valid thresholds
        EngineConfig(corr_threshold=0.0)
        EngineConfig(corr_threshold=0.5)
        EngineConfig(corr_threshold=1.0)

    def test_validation_topk_k_clamping(self):
        """Test topk_k clamping to uniques_k."""
        # Test topk_k > uniques_k gets clamped
        config = EngineConfig(uniques_k=10, topk_k=20)
        assert config.topk_k == 10  # Should be clamped to uniques_k

        # Test topk_k <= uniques_k remains unchanged
        config = EngineConfig(uniques_k=20, topk_k=10)
        assert config.topk_k == 10

    def test_validation_checkpointing(self):
        """Test checkpointing validation."""
        # Test checkpointing enabled but max_to_keep < 1
        with pytest.raises(
            ValueError,
            match="checkpoint_max_to_keep must be >= 1 when checkpointing is enabled",
        ):
            EngineConfig(checkpoint_every_n_chunks=5, checkpoint_max_to_keep=0)

        # Test checkpointing disabled (max_to_keep can be 0)
        config = EngineConfig(checkpoint_every_n_chunks=0, checkpoint_max_to_keep=0)
        assert config.checkpoint_max_to_keep == 0

        # Test checkpointing enabled with valid max_to_keep
        config = EngineConfig(checkpoint_every_n_chunks=5, checkpoint_max_to_keep=3)
        assert config.checkpoint_max_to_keep == 3

    def test_from_options_method(self):
        """Test the from_options class method."""

        class MockOptions:
            def __init__(self):
                self.chunk_size = 50_000
                self.numeric_sample_k = 5_000
                self.uniques_k = 512
                self.topk_k = 10
                self.engine = "polars"
                self.random_seed = 123

        options = MockOptions()
        config = EngineConfig.from_options(options)

        assert config.chunk_size == 50_000
        assert config.numeric_sample_k == 5_000
        assert config.uniques_k == 512
        assert config.topk_k == 10
        assert config.engine == "polars"
        assert config.random_seed == 123

    def test_from_options_with_none_chunk_size(self):
        """Test from_options with None chunk_size."""

        class MockOptions:
            def __init__(self):
                self.chunk_size = None
                self.numeric_sample_k = 5_000
                self.uniques_k = 512
                self.topk_k = 10
                self.engine = "auto"
                self.random_seed = None

        options = MockOptions()
        config = EngineConfig.from_options(options)

        assert config.chunk_size == 0  # Should use default
        assert config.numeric_sample_k == 5_000
        assert config.uniques_k == 512
        assert config.topk_k == 10
        assert config.engine == "auto"
        assert config.random_seed is None

    def test_edge_case_values(self):
        """Test edge case values."""
        # Test zero values (should be valid)
        config = EngineConfig(
            chunk_size=0,
            numeric_sample_k=0,
            uniques_k=0,
            topk_k=0,
            sample_rows=0,
            log_every_n_chunks=0,
            checkpoint_every_n_chunks=0,
            checkpoint_max_to_keep=0,
        )
        assert config.chunk_size == 0
        assert config.numeric_sample_k == 0
        assert config.uniques_k == 0
        assert config.topk_k == 0
        assert config.sample_rows == 0
        assert config.log_every_n_chunks == 0
        assert config.checkpoint_every_n_chunks == 0
        assert config.checkpoint_max_to_keep == 0

    def test_very_large_values(self):
        """Test very large values."""
        config = EngineConfig(
            chunk_size=10_000_000,
            numeric_sample_k=1_000_000,
            uniques_k=100_000,
            topk_k=10_000,
            sample_rows=1000,
            log_every_n_chunks=1000,
            checkpoint_every_n_chunks=1000,
            checkpoint_max_to_keep=100,
        )
        assert config.chunk_size == 10_000_000
        assert config.numeric_sample_k == 1_000_000
        assert config.uniques_k == 100_000
        assert config.topk_k == 10_000
        assert config.sample_rows == 1000
        assert config.log_every_n_chunks == 1000
        assert config.checkpoint_every_n_chunks == 1000
        assert config.checkpoint_max_to_keep == 100

    def test_string_values(self):
        """Test string configuration values."""
        config = EngineConfig(
            title="Test Report",
            engine="pandas",
            checkpoint_dir="/path/to/checkpoints",
            checkpoint_prefix="test_ckpt",
        )
        assert config.title == "Test Report"
        assert config.engine == "pandas"
        assert config.checkpoint_dir == "/path/to/checkpoints"
        assert config.checkpoint_prefix == "test_ckpt"

    def test_boolean_values(self):
        """Test boolean configuration values."""
        config = EngineConfig(
            include_sample=False,
            compute_correlations=False,
            checkpoint_write_html=True,
            force_chunk_in_memory=True,
        )
        assert config.include_sample is False
        assert config.compute_correlations is False
        assert config.checkpoint_write_html is True
        assert config.force_chunk_in_memory is True

    def test_logger_configuration(self):
        """Test logger configuration."""
        logger = logging.getLogger("test_logger")
        config = EngineConfig(logger=logger, log_level=logging.WARNING)

        assert config.logger is logger
        assert config.log_level == logging.WARNING

        # Test post_init logger level setting
        config.__post_init__()
        assert logger.level == logging.WARNING

    def test_random_seed_types(self):
        """Test random seed with different types."""
        # Test with integer
        config = EngineConfig(random_seed=42)
        assert config.random_seed == 42

        # Test with None
        config = EngineConfig(random_seed=None)
        assert config.random_seed is None

        # Test with string (should remain as string - no automatic conversion)
        config = EngineConfig(random_seed="123")
        assert config.random_seed == "123"

    def test_correlation_configuration(self):
        """Test correlation-related configuration."""
        config = EngineConfig(
            compute_correlations=True,
            corr_threshold=0.3,
            corr_max_cols=75,
            corr_max_per_col=7,
        )
        assert config.compute_correlations is True
        assert config.corr_threshold == 0.3
        assert config.corr_max_cols == 75
        assert config.corr_max_per_col == 7

    def test_checkpointing_configuration(self):
        """Test checkpointing configuration."""
        config = EngineConfig(
            checkpoint_every_n_chunks=25,
            checkpoint_dir="/custom/path",
            checkpoint_prefix="custom_prefix",
            checkpoint_write_html=True,
            checkpoint_max_to_keep=7,
        )
        assert config.checkpoint_every_n_chunks == 25
        assert config.checkpoint_dir == "/custom/path"
        assert config.checkpoint_prefix == "custom_prefix"
        assert config.checkpoint_write_html is True
        assert config.checkpoint_max_to_keep == 7


class TestEngineOptions:
    """Test the EngineOptions protocol."""

    def test_engine_options_protocol(self):
        """Test that EngineOptions protocol works correctly."""

        class MockOptions:
            def __init__(self):
                self.chunk_size = 100_000
                self.numeric_sample_k = 10_000
                self.uniques_k = 1024
                self.topk_k = 25
                self.engine = "pandas"
                self.random_seed = 42

        options = MockOptions()

        # Test that it implements the protocol
        assert isinstance(options, EngineOptions)

        # Test that EngineConfig.from_options works with it
        config = EngineConfig.from_options(options)
        assert config.chunk_size == 100_000
        assert config.numeric_sample_k == 10_000
        assert config.uniques_k == 1024
        assert config.topk_k == 25
        assert config.engine == "pandas"
        assert config.random_seed == 42

    def test_engine_options_with_none_values(self):
        """Test EngineOptions with None values."""

        class MockOptions:
            def __init__(self):
                self.chunk_size = None
                self.numeric_sample_k = 10_000
                self.uniques_k = 1024
                self.topk_k = 25
                self.engine = "auto"
                self.random_seed = None

        options = MockOptions()
        assert isinstance(options, EngineOptions)

        config = EngineConfig.from_options(options)
        assert config.chunk_size == 0  # Should use default
        assert config.numeric_sample_k == 10_000
        assert config.uniques_k == 1024
        assert config.topk_k == 25
        assert config.engine == "auto"
        assert config.random_seed is None
