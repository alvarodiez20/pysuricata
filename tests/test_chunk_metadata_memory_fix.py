"""Comprehensive tests for chunk metadata memory optimization.

This module tests the chunk metadata optimization to ensure:
1. Memory usage is bounded when chunk metadata is disabled
2. Memory usage is bounded when chunk metadata is enabled but limited
3. Backward compatibility is maintained
4. Visualization works correctly with and without chunk metadata
"""

import pytest
import tracemalloc
import numpy as np
from typing import List, Tuple

from pysuricata.accumulators.numeric import NumericAccumulator
from pysuricata.accumulators.config import NumericConfig


class TestChunkMetadataMemoryFix:
    """Test chunk metadata memory optimization."""

    def test_chunk_metadata_disabled_saves_memory(self):
        """Test that disabling chunk metadata saves memory."""
        # Create accumulator with chunk metadata disabled
        config = NumericConfig(enable_chunk_metadata=False)
        accumulator = NumericAccumulator("test_col", config)
        
        # Verify chunk metadata is disabled
        assert not accumulator._chunk_metadata_enabled
        assert accumulator._chunk_boundaries is None
        assert accumulator._chunk_missing is None
        
        # Process many chunks
        for i in range(1000):
            values = np.random.randn(1000)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        # Verify no chunk metadata was stored
        assert accumulator._chunk_boundaries is None
        assert accumulator._chunk_missing is None
        assert accumulator._chunk_count == 0

    def test_chunk_metadata_enabled_with_limit(self):
        """Test that chunk metadata is bounded when enabled."""
        # Create accumulator with limited chunk metadata
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=10)
        accumulator = NumericAccumulator("test_col", config)
        
        # Verify chunk metadata is enabled
        assert accumulator._chunk_metadata_enabled
        assert accumulator._chunk_boundaries is not None
        assert accumulator._chunk_missing is not None
        
        # Process more chunks than the limit
        for i in range(20):  # More than max_chunks=10
            values = np.random.randn(1000)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        # Verify chunk metadata is bounded
        assert len(accumulator._chunk_boundaries) <= 10
        assert len(accumulator._chunk_missing) <= 10
        assert accumulator._chunk_count <= 10
        
        # After exceeding limit, metadata should be disabled
        if accumulator._chunk_count >= 10:
            assert not accumulator._chunk_metadata_enabled

    def test_memory_usage_comparison(self):
        """Test memory usage comparison between enabled and disabled chunk metadata."""
        tracemalloc.start()
        
        # Test with chunk metadata enabled
        config_enabled = NumericConfig(enable_chunk_metadata=True, max_chunks=100)
        accumulator_enabled = NumericAccumulator("test_col", config_enabled)
        
        for i in range(100):
            values = np.random.randn(1000)
            accumulator_enabled.update(values)
            accumulator_enabled.mark_chunk_boundary()
        
        current_enabled, peak_enabled = tracemalloc.get_traced_memory()
        
        # Reset memory tracking
        tracemalloc.stop()
        tracemalloc.start()
        
        # Test with chunk metadata disabled
        config_disabled = NumericConfig(enable_chunk_metadata=False)
        accumulator_disabled = NumericAccumulator("test_col", config_disabled)
        
        for i in range(100):
            values = np.random.randn(1000)
            accumulator_disabled.update(values)
            accumulator_disabled.mark_chunk_boundary()
        
        current_disabled, peak_disabled = tracemalloc.get_traced_memory()
        
        tracemalloc.stop()
        
        # Memory usage should be lower when chunk metadata is disabled
        # (though the difference might be small due to other factors)
        print(f"Memory with chunk metadata: {peak_enabled / 1024 / 1024:.2f} MB")
        print(f"Memory without chunk metadata: {peak_disabled / 1024 / 1024:.2f} MB")

    def test_backward_compatibility(self):
        """Test that the API remains backward compatible."""
        # Test with default configuration (chunk metadata enabled)
        accumulator_default = NumericAccumulator("test_col")
        assert accumulator_default._chunk_metadata_enabled
        assert accumulator_default._chunk_boundaries is not None
        assert accumulator_default._chunk_missing is not None
        
        # Test with explicit configuration
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=50)
        accumulator_explicit = NumericAccumulator("test_col", config)
        assert accumulator_explicit._chunk_metadata_enabled
        assert accumulator_explicit.config.max_chunks == 50
        
        # Test that the interface remains the same
        values = np.random.randn(100)
        accumulator_default.update(values)
        accumulator_default.mark_chunk_boundary()
        
        accumulator_explicit.update(values)
        accumulator_explicit.mark_chunk_boundary()

    def test_chunk_boundary_behavior(self):
        """Test chunk boundary marking behavior."""
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=5)
        accumulator = NumericAccumulator("test_col", config)
        
        # Test normal chunk boundary marking
        for i in range(3):
            values = np.random.randn(100)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
            
            assert len(accumulator._chunk_boundaries) == i + 1
            assert len(accumulator._chunk_missing) == i + 1
            assert accumulator._chunk_count == i + 1
        
        # Test exceeding max_chunks
        for i in range(3, 8):  # Process 5 more chunks (total 8, limit is 5)
            values = np.random.randn(100)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
            
            if i < 5:
                assert len(accumulator._chunk_boundaries) == i + 1
                assert accumulator._chunk_count == i + 1
            else:
                # After exceeding limit, should stop tracking
                assert len(accumulator._chunk_boundaries) <= 5
                assert accumulator._chunk_count <= 5

    def test_finalize_with_chunk_metadata(self):
        """Test finalize method with chunk metadata."""
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=10)
        accumulator = NumericAccumulator("test_col", config)
        
        # Process some chunks
        for i in range(5):
            values = np.random.randn(100)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        # Finalize and check metadata
        summary = accumulator.finalize()
        
        # Should have chunk metadata
        assert summary.chunk_metadata is not None
        assert len(summary.chunk_metadata) == 5

    def test_finalize_without_chunk_metadata(self):
        """Test finalize method without chunk metadata."""
        config = NumericConfig(enable_chunk_metadata=False)
        accumulator = NumericAccumulator("test_col", config)
        
        # Process some chunks
        for i in range(5):
            values = np.random.randn(100)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        # Finalize and check metadata
        summary = accumulator.finalize()
        
        # Should not have chunk metadata
        assert summary.chunk_metadata is None

    def test_finalize_with_provided_chunk_metadata(self):
        """Test finalize method with provided chunk metadata."""
        config = NumericConfig(enable_chunk_metadata=False)
        accumulator = NumericAccumulator("test_col", config)
        
        # Process some data
        values = np.random.randn(1000)
        accumulator.update(values)
        
        # Provide external chunk metadata
        external_metadata = [(0, 499, 10), (500, 999, 15)]
        summary = accumulator.finalize(chunk_metadata=external_metadata)
        
        # Should use provided metadata
        assert summary.chunk_metadata == external_metadata

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=1)
        accumulator = NumericAccumulator("test_col", config)
        
        # Test with max_chunks=1
        values = np.random.randn(100)
        accumulator.update(values)
        accumulator.mark_chunk_boundary()
        
        assert len(accumulator._chunk_boundaries) == 1
        assert accumulator._chunk_count == 1
        
        # Add another chunk - should disable metadata tracking
        values = np.random.randn(100)
        accumulator.update(values)
        accumulator.mark_chunk_boundary()
        
        assert len(accumulator._chunk_boundaries) == 1  # Should not grow
        assert not accumulator._chunk_metadata_enabled  # Should be disabled

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large dataset."""
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=100)
        accumulator = NumericAccumulator("test_col", config)
        
        tracemalloc.start()
        
        # Process many chunks
        for i in range(1000):  # 1000 chunks
            values = np.random.randn(1000)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        current, peak = tracemalloc.get_traced_memory()
        
        tracemalloc.stop()
        
        # Memory should be bounded despite processing many chunks
        assert peak < 100 * 1024 * 1024  # Less than 100MB
        
        # Chunk metadata should be limited
        assert len(accumulator._chunk_boundaries) <= 100
        assert len(accumulator._chunk_missing) <= 100

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test valid configuration
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=100)
        assert config.enable_chunk_metadata is True
        assert config.max_chunks == 100
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            NumericConfig(max_chunks=0)  # Should raise ValueError
        
        with pytest.raises(ValueError):
            NumericConfig(max_chunks=-1)  # Should raise ValueError

    def test_performance_impact(self):
        """Test performance impact of chunk metadata optimization."""
        import time
        
        # Test with chunk metadata enabled
        config_enabled = NumericConfig(enable_chunk_metadata=True, max_chunks=100)
        accumulator_enabled = NumericAccumulator("test_col", config_enabled)
        
        start_time = time.perf_counter()
        for i in range(100):
            values = np.random.randn(1000)
            accumulator_enabled.update(values)
            accumulator_enabled.mark_chunk_boundary()
        end_time = time.perf_counter()
        
        time_enabled = end_time - start_time
        
        # Test with chunk metadata disabled
        config_disabled = NumericConfig(enable_chunk_metadata=False)
        accumulator_disabled = NumericAccumulator("test_col", config_disabled)
        
        start_time = time.perf_counter()
        for i in range(100):
            values = np.random.randn(1000)
            accumulator_disabled.update(values)
            accumulator_disabled.mark_chunk_boundary()
        end_time = time.perf_counter()
        
        time_disabled = end_time - start_time
        
        # Performance should be similar (chunk metadata overhead is minimal)
        print(f"Time with chunk metadata: {time_enabled:.4f}s")
        print(f"Time without chunk metadata: {time_disabled:.4f}s")
        
        # Should not be significantly slower
        assert time_enabled < time_disabled * 1.5  # Allow 50% overhead


class TestChunkMetadataVisualization:
    """Test chunk metadata visualization functionality."""

    def test_visualization_with_metadata(self):
        """Test visualization works with chunk metadata."""
        config = NumericConfig(enable_chunk_metadata=True, max_chunks=10)
        accumulator = NumericAccumulator("test_col", config)
        
        # Process chunks with different missing value patterns
        for i in range(5):
            values = np.random.randn(100)
            # Add some missing values
            if i % 2 == 0:
                values[:10] = np.nan
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        summary = accumulator.finalize()
        
        # Should have chunk metadata for visualization
        assert summary.chunk_metadata is not None
        assert len(summary.chunk_metadata) == 5

    def test_visualization_without_metadata(self):
        """Test visualization works without chunk metadata."""
        config = NumericConfig(enable_chunk_metadata=False)
        accumulator = NumericAccumulator("test_col", config)
        
        # Process chunks
        for i in range(5):
            values = np.random.randn(100)
            accumulator.update(values)
            accumulator.mark_chunk_boundary()
        
        summary = accumulator.finalize()
        
        # Should not have chunk metadata
        assert summary.chunk_metadata is None
        
        # But should still have other statistics
        assert summary.count > 0
        assert summary.mean is not None

    def test_visualization_with_external_metadata(self):
        """Test visualization works with external chunk metadata."""
        config = NumericConfig(enable_chunk_metadata=False)
        accumulator = NumericAccumulator("test_col", config)
        
        # Process data
        values = np.random.randn(1000)
        accumulator.update(values)
        
        # Provide external metadata
        external_metadata = [(0, 199, 5), (200, 399, 10), (400, 599, 3), (600, 799, 8), (800, 999, 2)]
        summary = accumulator.finalize(chunk_metadata=external_metadata)
        
        # Should use external metadata
        assert summary.chunk_metadata == external_metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
