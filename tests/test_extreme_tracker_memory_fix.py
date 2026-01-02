"""Comprehensive tests for ExtremeTracker memory leak fixes.

This module tests the fixed ExtremeTracker implementation to ensure:
1. Memory usage is O(k) not O(k × chunks)
2. Correct extreme values are tracked
3. Merge functionality works correctly
4. Performance is maintained or improved
"""

import pytest
import tracemalloc
import numpy as np
from typing import List, Tuple

from pysuricata.accumulators.algorithms import ExtremeTracker


class TestExtremeTrackerMemoryFix:
    """Test ExtremeTracker memory usage fixes."""

    def test_bounded_memory_small_dataset(self):
        """Test memory stays bounded with small dataset."""
        tracker = ExtremeTracker(max_extremes=5)
        
        # Add some values
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        tracker.update(values, indices)
        
        min_pairs, max_pairs = tracker.get_extremes()
        
        # Should have at most 5 extremes each
        assert len(min_pairs) <= 5
        assert len(max_pairs) <= 5
        
        # Check that extremes are correct
        min_values = [pair[1] for pair in min_pairs]
        max_values = [pair[1] for pair in max_pairs]
        
        assert min(min_values) == 1.0
        assert max(max_values) == 10.0

    def test_memory_bounded_many_chunks(self):
        """Test memory stays bounded with many chunks."""
        tracker = ExtremeTracker(max_extremes=5)
        
        tracemalloc.start()
        
        # Simulate processing 1000 chunks
        for chunk in range(1000):
            # Each chunk has 100 values
            values = np.random.randn(100) * 1000  # Large range
            indices = np.arange(chunk * 100, (chunk + 1) * 100)
            
            tracker.update(values, indices)
            
            # Check memory every 100 chunks
            if chunk % 100 == 0:
                current, peak = tracemalloc.get_traced_memory()
                # Memory should not grow linearly with chunks
                assert peak < 10 * 1024 * 1024  # Less than 10MB
        
        tracemalloc.stop()
        
        # Should still have bounded extremes
        min_pairs, max_pairs = tracker.get_extremes()
        assert len(min_pairs) <= 5
        assert len(max_pairs) <= 5

    def test_correct_extremes_tracking(self):
        """Test that correct extreme values are tracked."""
        tracker = ExtremeTracker(max_extremes=3)
        
        # Add values with known extremes
        values = np.array([100.0, 1.0, 50.0, 2.0, 99.0, 3.0, 98.0, 4.0, 97.0, 5.0])
        indices = np.arange(len(values))
        
        tracker.update(values, indices)
        
        min_pairs, max_pairs = tracker.get_extremes()
        
        # Should track the 3 smallest and 3 largest
        min_values = [pair[1] for pair in min_pairs]
        max_values = [pair[1] for pair in max_pairs]
        
        # Check that we have the correct extremes
        assert 1.0 in min_values
        assert 2.0 in min_values
        assert 3.0 in min_values
        
        assert 100.0 in max_values
        assert 99.0 in max_values
        assert 98.0 in max_values

    def test_heap_properties(self):
        """Test that heap properties are maintained."""
        tracker = ExtremeTracker(max_extremes=3)
        
        # Add values one by one and check heap properties
        test_values = [10.0, 5.0, 15.0, 1.0, 20.0, 2.0, 25.0, 3.0]
        
        for i, value in enumerate(test_values):
            tracker.update(np.array([value]), np.array([i]))
            
            # Check that heaps don't exceed max_extremes
            assert len(tracker._min_heap) <= 3
            assert len(tracker._max_heap) <= 3

    def test_merge_functionality(self):
        """Test merge functionality works correctly."""
        tracker1 = ExtremeTracker(max_extremes=3)
        tracker2 = ExtremeTracker(max_extremes=3)
        
        # Add different sets of values
        values1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        values2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        
        tracker1.update(values1, np.arange(len(values1)))
        tracker2.update(values2, np.arange(len(values2)) + 100)  # Different indices
        
        # Merge tracker2 into tracker1
        tracker1.merge(tracker2)
        
        min_pairs, max_pairs = tracker1.get_extremes()
        
        # Should have extremes from both datasets
        min_values = [pair[1] for pair in min_pairs]
        max_values = [pair[1] for pair in max_pairs]
        
        assert 1.0 in min_values  # From tracker1
        assert 2.0 in min_values  # From tracker1
        assert 3.0 in min_values  # From tracker1
        
        assert 8.0 in max_values  # From tracker2
        assert 9.0 in max_values  # From tracker2
        assert 10.0 in max_values  # From tracker2

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        tracker = ExtremeTracker(max_extremes=5)
        
        # Empty array
        tracker.update(np.array([]), np.array([]))
        min_pairs, max_pairs = tracker.get_extremes()
        assert len(min_pairs) == 0
        assert len(max_pairs) == 0
        
        # Single value
        tracker.update(np.array([42.0]), np.array([0]))
        min_pairs, max_pairs = tracker.get_extremes()
        assert len(min_pairs) == 1
        assert len(max_pairs) == 1
        assert min_pairs[0][1] == 42.0
        assert max_pairs[0][1] == 42.0
        
        # All NaN values
        tracker = ExtremeTracker(max_extremes=5)
        tracker.update(np.array([np.nan, np.nan, np.nan]), np.array([0, 1, 2]))
        min_pairs, max_pairs = tracker.get_extremes()
        assert len(min_pairs) == 0
        assert len(max_pairs) == 0
        
        # Mixed finite and infinite values
        tracker = ExtremeTracker(max_extremes=5)
        tracker.update(np.array([1.0, np.inf, 2.0, -np.inf, 3.0]), np.array([0, 1, 2, 3, 4]))
        min_pairs, max_pairs = tracker.get_extremes()
        # Should only track finite values
        assert len(min_pairs) == 3
        assert len(max_pairs) == 3

    def test_performance_comparison(self):
        """Test that performance is maintained or improved."""
        tracker = ExtremeTracker(max_extremes=5)
        
        # Large dataset
        n_values = 100000
        values = np.random.randn(n_values)
        indices = np.arange(n_values)
        
        import time
        start_time = time.perf_counter()
        
        tracker.update(values, indices)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Should process 100k values in reasonable time
        assert processing_time < 1.0  # Less than 1 second
        
        # Check that extremes are reasonable
        min_pairs, max_pairs = tracker.get_extremes()
        assert len(min_pairs) <= 5
        assert len(max_pairs) <= 5

    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        tracker = ExtremeTracker(max_extremes=5)
        
        # Initial memory should be small
        initial_memory = len(tracker._min_heap) + len(tracker._max_heap)
        assert initial_memory == 0
        
        # Add some values
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        tracker.update(values, np.arange(len(values)))
        
        # Memory should be bounded
        final_memory = len(tracker._min_heap) + len(tracker._max_heap)
        assert final_memory <= 10  # At most 5 + 5

    def test_stress_test_extreme_values(self):
        """Stress test with extreme values."""
        tracker = ExtremeTracker(max_extremes=3)
        
        # Add many chunks with extreme values
        for chunk in range(1000):
            # Each chunk has some extreme values
            values = np.array([-1000.0, 0.0, 1000.0] + list(np.random.randn(97)))
            indices = np.arange(chunk * 100, (chunk + 1) * 100)
            
            tracker.update(values, indices)
            
            # Memory should stay bounded
            assert len(tracker._min_heap) <= 3
            assert len(tracker._max_heap) <= 3
        
        # Final extremes should include the extreme values
        min_pairs, max_pairs = tracker.get_extremes()
        min_values = [pair[1] for pair in min_pairs]
        max_values = [pair[1] for pair in max_pairs]
        
        # Should have tracked some of the extreme values
        assert any(v <= -1000.0 for v in min_values)
        assert any(v >= 1000.0 for v in max_values)

    def test_backward_compatibility(self):
        """Test that the API remains backward compatible."""
        # Should work with default parameters
        tracker = ExtremeTracker()
        assert tracker.max_extremes == 5
        
        # Should work with custom max_extremes
        tracker = ExtremeTracker(max_extremes=10)
        assert tracker.max_extremes == 10
        
        # Should have the same interface
        values = np.array([1.0, 2.0, 3.0])
        indices = np.array([0, 1, 2])
        
        tracker.update(values, indices)
        min_pairs, max_pairs = tracker.get_extremes()
        
        # Should return the same format
        assert isinstance(min_pairs, list)
        assert isinstance(max_pairs, list)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in min_pairs)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in max_pairs)


class TestExtremeTrackerCorrectness:
    """Test that statistical correctness is maintained."""

    def test_exact_extremes_small_dataset(self):
        """Test exact extremes for small datasets."""
        tracker = ExtremeTracker(max_extremes=10)
        
        # Small dataset where we can track all extremes
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        indices = np.arange(len(values))
        
        tracker.update(values, indices)
        
        min_pairs, max_pairs = tracker.get_extremes()
        
        # Should have all values since max_extremes >= dataset size
        assert len(min_pairs) == 10
        assert len(max_pairs) == 10
        
        # Check ordering
        min_values = [pair[1] for pair in min_pairs]
        max_values = [pair[1] for pair in max_pairs]
        
        assert min_values == sorted(min_values)
        assert max_values == sorted(max_values, reverse=True)

    def test_reproducibility(self):
        """Test that results are reproducible."""
        tracker1 = ExtremeTracker(max_extremes=5)
        tracker2 = ExtremeTracker(max_extremes=5)
        
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        indices = np.arange(len(values))
        
        # Add same values to both
        tracker1.update(values, indices)
        tracker2.update(values, indices)
        
        min_pairs1, max_pairs1 = tracker1.get_extremes()
        min_pairs2, max_pairs2 = tracker2.get_extremes()
        
        # Results should be the same
        assert min_pairs1 == min_pairs2
        assert max_pairs1 == max_pairs2

    def test_duplicate_values(self):
        """Test handling of duplicate values."""
        tracker = ExtremeTracker(max_extremes=3)
        
        # Add duplicate values
        values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        indices = np.array([0, 1, 2, 3, 4, 5])
        
        tracker.update(values, indices)
        
        min_pairs, max_pairs = tracker.get_extremes()
        
        # Should handle duplicates correctly
        assert len(min_pairs) <= 3
        assert len(max_pairs) <= 3
        
        # Should have the correct extreme values
        min_values = [pair[1] for pair in min_pairs]
        max_values = [pair[1] for pair in max_pairs]
        
        assert 1.0 in min_values
        assert 3.0 in max_values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
