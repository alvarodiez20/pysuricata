"""
Stress tests for the per-column chunk metadata architecture.

These tests validate the system under heavy load and edge cases to ensure
robustness and performance.
"""

import gc
import os
import time

import numpy as np
import pandas as pd
import psutil
import pytest

from pysuricata.compute.adapters.pandas import PandasAdapter
from pysuricata.render.numeric_card import NumericCardRenderer


class TestChunkMetadataStress:
    """Stress tests for chunk metadata system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pandas_adapter = PandasAdapter()
        self.renderer = NumericCardRenderer()

    def test_large_dataset_memory_usage(self):
        """Test memory usage with large dataset."""
        # Create large dataset
        np.random.seed(42)
        n_rows = 1000000  # 1 million rows
        n_cols = 100  # 100 columns

        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create dataset
        data = {}
        for i in range(n_cols):
            col_data = np.random.normal(0, 1, n_rows)
            # Add some missing values
            missing_indices = np.random.choice(
                n_rows, size=n_rows // 100, replace=False
            )
            col_data[missing_indices] = np.nan
            data[f"col_{i}"] = col_data

        df = pd.DataFrame(data)

        # Measure memory after DataFrame creation
        df_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test per-column missing count calculation
        start_time = time.time()
        per_column_missing = self.pandas_adapter.missing_cells_per_column(df)
        end_time = time.time()

        # Measure memory after calculation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Verify results
        assert len(per_column_missing) == n_cols
        for col_name, missing_count in per_column_missing.items():
            assert missing_count == n_rows // 100

        # Performance assertions
        assert end_time - start_time < 10.0  # Should complete within 10 seconds

        # Memory usage should be reasonable (not more than 2x the DataFrame size)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 2000  # Less than 2GB increase

        # Clean up
        del df, per_column_missing
        gc.collect()

    def test_concurrent_chunk_processing(self):
        """Test concurrent processing of multiple chunks."""
        import queue
        import threading

        def process_chunk(chunk_data, result_queue):
            """Process a single chunk."""
            try:
                df = pd.DataFrame(chunk_data)
                per_column_missing = self.pandas_adapter.missing_cells_per_column(df)
                result_queue.put(per_column_missing)
            except Exception as e:
                result_queue.put(f"Error: {e}")

        # Create multiple chunks
        n_chunks = 10
        chunk_size = 10000
        n_cols = 20

        chunks = []
        for i in range(n_chunks):
            chunk_data = {}
            for j in range(n_cols):
                col_data = np.random.normal(0, 1, chunk_size)
                # Add some missing values
                missing_indices = np.random.choice(
                    chunk_size, size=chunk_size // 50, replace=False
                )
                col_data[missing_indices] = np.nan
                chunk_data[f"col_{j}"] = col_data
            chunks.append(chunk_data)

        # Process chunks concurrently
        threads = []
        result_queue = queue.Queue()

        start_time = time.time()
        for chunk_data in chunks:
            thread = threading.Thread(
                target=process_chunk, args=(chunk_data, result_queue)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.time()

        # Collect results
        results = []
        while not result_queue.empty():
            result = result_queue.get()
            results.append(result)

        # Verify results
        assert len(results) == n_chunks
        for result in results:
            assert isinstance(result, dict)
            assert len(result) == n_cols
            for col_name, missing_count in result.items():
                assert missing_count == chunk_size // 50

        # Performance assertion
        assert end_time - start_time < 5.0  # Should complete within 5 seconds

    def test_extreme_missing_value_patterns(self):
        """Test with extreme missing value patterns."""
        # Test with very high missing percentage
        n_rows = 10000
        data_high_missing = {
            "col1": np.full(n_rows, np.nan),  # 100% missing
            "col2": np.random.choice(
                [1, np.nan], n_rows, p=[0.01, 0.99]
            ),  # 99% missing
            "col3": np.random.choice([1, np.nan], n_rows, p=[0.5, 0.5]),  # 50% missing
            "col4": np.random.choice([1, np.nan], n_rows, p=[0.99, 0.01]),  # 1% missing
            "col5": np.ones(n_rows),  # 0% missing
        }

        df = pd.DataFrame(data_high_missing)
        per_column_missing = self.pandas_adapter.missing_cells_per_column(df)

        # Verify extreme patterns
        assert per_column_missing["col1"] == n_rows  # 100% missing
        assert per_column_missing["col2"] > n_rows * 0.95  # >95% missing
        assert (
            per_column_missing["col3"] > n_rows * 0.45
            and per_column_missing["col3"] < n_rows * 0.55
        )  # ~50% missing
        assert per_column_missing["col4"] < n_rows * 0.05  # <5% missing
        assert per_column_missing["col5"] == 0  # 0% missing

    def test_memory_leak_prevention(self):
        """Test that there are no memory leaks in repeated operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform many iterations
        n_iterations = 100
        for i in range(n_iterations):
            # Create dataset
            n_rows = 10000
            n_cols = 10
            data = {}
            for j in range(n_cols):
                col_data = np.random.normal(0, 1, n_rows)
                missing_indices = np.random.choice(
                    n_rows, size=n_rows // 20, replace=False
                )
                col_data[missing_indices] = np.nan
                data[f"col_{j}"] = col_data

            df = pd.DataFrame(data)
            per_column_missing = self.pandas_adapter.missing_cells_per_column(df)

            # Verify results
            assert len(per_column_missing) == n_cols

            # Clean up
            del df, per_column_missing
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 100MB)
        assert memory_increase < 100

    def test_chunk_metadata_consistency(self):
        """Test consistency of chunk metadata across different operations."""
        # Create dataset
        n_rows = 1000
        n_cols = 5
        data = {}
        for i in range(n_cols):
            col_data = np.random.normal(0, 1, n_rows)
            missing_indices = np.random.choice(n_rows, size=n_rows // 10, replace=False)
            col_data[missing_indices] = np.nan
            data[f"col_{i}"] = col_data

        df = pd.DataFrame(data)

        # Test multiple times to ensure consistency
        results = []
        for _ in range(10):
            per_column_missing = self.pandas_adapter.missing_cells_per_column(df)
            results.append(per_column_missing)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0]

    def test_performance_scaling(self):
        """Test performance scaling with dataset size."""
        sizes = [1000, 10000, 100000, 500000]
        times = []

        for size in sizes:
            # Create dataset
            n_cols = 10
            data = {}
            for i in range(n_cols):
                col_data = np.random.normal(0, 1, size)
                missing_indices = np.random.choice(size, size=size // 20, replace=False)
                col_data[missing_indices] = np.nan
                data[f"col_{i}"] = col_data

            df = pd.DataFrame(data)

            # Measure time
            start_time = time.time()
            per_column_missing = self.pandas_adapter.missing_cells_per_column(df)
            end_time = time.time()

            times.append(end_time - start_time)

            # Verify results
            assert len(per_column_missing) == n_cols
            for col_name, missing_count in per_column_missing.items():
                assert missing_count == size // 20

            # Clean up
            del df, per_column_missing
            gc.collect()

        # Performance should scale reasonably (not exponentially)
        # The ratio of times should be roughly proportional to dataset size
        for i in range(1, len(times)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]
            # Time ratio should be less than 2x the size ratio
            assert time_ratio < size_ratio * 2

    def test_edge_case_data_types(self):
        """Test with various edge case data types."""
        # Test with different data types
        data_types = {
            "int8": np.random.randint(-128, 127, 1000, dtype=np.int8),
            "int16": np.random.randint(-32768, 32767, 1000, dtype=np.int16),
            "int32": np.random.randint(-2147483648, 2147483647, 1000, dtype=np.int32),
            "int64": np.random.randint(
                -9223372036854775808, 9223372036854775807, 1000, dtype=np.int64
            ),
            "float32": np.random.normal(0, 1, 1000).astype(np.float32),
            "float64": np.random.normal(0, 1, 1000).astype(np.float64),
            "bool": np.random.choice([True, False], 1000),
            "object": np.random.choice(["a", "b", "c", "d"], 1000),
        }

        for dtype_name, col_data in data_types.items():
            # Add some missing values
            missing_indices = np.random.choice(
                len(col_data), size=len(col_data) // 20, replace=False
            )
            col_data[missing_indices] = np.nan

            df = pd.DataFrame({dtype_name: col_data})
            per_column_missing = self.pandas_adapter.missing_cells_per_column(df)

            # Verify results
            assert per_column_missing[dtype_name] == len(col_data) // 20

    def test_chunk_metadata_accuracy_under_stress(self):
        """Test accuracy of chunk metadata under stress conditions."""
        # Create complex dataset with known patterns
        n_rows = 50000
        n_cols = 50

        # Create dataset with known missing patterns
        data = {}
        expected_missing = {}

        for i in range(n_cols):
            col_data = np.random.normal(0, 1, n_rows)
            # Create specific missing patterns
            if i % 3 == 0:
                # Every 3rd column: 10% missing
                missing_indices = np.random.choice(
                    n_rows, size=n_rows // 10, replace=False
                )
                expected_missing[f"col_{i}"] = n_rows // 10
            elif i % 3 == 1:
                # Every 3rd+1 column: 5% missing
                missing_indices = np.random.choice(
                    n_rows, size=n_rows // 20, replace=False
                )
                expected_missing[f"col_{i}"] = n_rows // 20
            else:
                # Every 3rd+2 column: 20% missing
                missing_indices = np.random.choice(
                    n_rows, size=n_rows // 5, replace=False
                )
                expected_missing[f"col_{i}"] = n_rows // 5

            col_data[missing_indices] = np.nan
            data[f"col_{i}"] = col_data

        df = pd.DataFrame(data)

        # Test under stress (multiple iterations)
        for iteration in range(5):
            per_column_missing = self.pandas_adapter.missing_cells_per_column(df)

            # Verify accuracy
            for col_name, expected_count in expected_missing.items():
                assert per_column_missing[col_name] == expected_count

            # Clean up
            del per_column_missing
            gc.collect()

    def test_system_resources_under_load(self):
        """Test system resource usage under heavy load."""
        process = psutil.Process(os.getpid())

        # Monitor CPU and memory
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create heavy load
        n_iterations = 50
        for i in range(n_iterations):
            # Create large dataset
            n_rows = 50000
            n_cols = 20
            data = {}
            for j in range(n_cols):
                col_data = np.random.normal(0, 1, n_rows)
                missing_indices = np.random.choice(
                    n_rows, size=n_rows // 20, replace=False
                )
                col_data[missing_indices] = np.nan
                data[f"col_{j}"] = col_data

            df = pd.DataFrame(data)
            per_column_missing = self.pandas_adapter.missing_cells_per_column(df)

            # Verify results
            assert len(per_column_missing) == n_cols

            # Clean up
            del df, per_column_missing
            gc.collect()

        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # System should not be overwhelmed
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500  # Less than 500MB increase

        # CPU usage should be reasonable (not 100%)
        assert final_cpu < 90  # Less than 90% CPU usage


if __name__ == "__main__":
    pytest.main([__file__])
