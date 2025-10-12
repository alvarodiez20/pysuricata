"""
Comprehensive test suite for the per-column chunk metadata architecture fix.

This test suite validates the complete implementation of per-column missing count tracking
in chunk metadata, replacing the previous system that incorrectly used total missing counts.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators.numeric import NumericAccumulator, NumericSummary
from pysuricata.compute.adapters.pandas import PandasAdapter
from pysuricata.compute.adapters.polars import PolarsAdapter
from pysuricata.render.card_types import NumericStats
from pysuricata.render.numeric_card import NumericCardRenderer


class TestPandasAdapterChunkMetadata:
    """Test pandas adapter per-column missing count functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = PandasAdapter()

    def test_missing_cells_per_column_accuracy(self):
        """Test accuracy of per-column missing count calculation."""
        # Create test data with known missing patterns
        data = {
            "age": [25, 30, np.nan, 35, 40, np.nan, 45, 50, np.nan, 55],
            "income": [
                50000,
                np.nan,
                60000,
                70000,
                np.nan,
                80000,
                90000,
                np.nan,
                100000,
                110000,
            ],
            "education": [
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
            ],
        }
        df = pd.DataFrame(data)

        result = self.adapter.missing_cells_per_column(df)

        expected = {"age": 3, "income": 3, "education": 0}

        assert result == expected

    def test_missing_cells_per_column_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = self.adapter.missing_cells_per_column(df)
        assert result == {}

    def test_missing_cells_per_column_no_missing(self):
        """Test with no missing values."""
        data = {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
        }
        df = pd.DataFrame(data)

        result = self.adapter.missing_cells_per_column(df)

        expected = {"age": 0, "income": 0}

        assert result == expected

    def test_missing_cells_per_column_all_missing(self):
        """Test with all values missing in some columns."""
        data = {
            "age": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "income": [50000, 60000, 70000, 80000, 90000],
        }
        df = pd.DataFrame(data)

        result = self.adapter.missing_cells_per_column(df)

        expected = {"age": 5, "income": 0}

        assert result == expected

    def test_missing_cells_per_column_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(42)
        n_rows = 10000

        # Create data with known missing patterns
        age_data = np.random.normal(30, 10, n_rows)
        age_missing_indices = np.random.choice(n_rows, size=1000, replace=False)
        age_data[age_missing_indices] = np.nan

        income_data = np.random.normal(50000, 15000, n_rows)
        income_missing_indices = np.random.choice(n_rows, size=500, replace=False)
        income_data[income_missing_indices] = np.nan

        data = {"age": age_data, "income": income_data}
        df = pd.DataFrame(data)

        result = self.adapter.missing_cells_per_column(df)

        assert result["age"] == 1000
        assert result["income"] == 500

    def test_missing_cells_per_column_performance(self):
        """Test performance with large dataset."""
        np.random.seed(42)
        n_rows = 100000
        n_cols = 50

        # Create large dataset
        data = {}
        for i in range(n_cols):
            col_data = np.random.normal(0, 1, n_rows)
            # Add some missing values
            missing_indices = np.random.choice(n_rows, size=n_rows // 10, replace=False)
            col_data[missing_indices] = np.nan
            data[f"col_{i}"] = col_data

        df = pd.DataFrame(data)

        import time

        start_time = time.time()
        result = self.adapter.missing_cells_per_column(df)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(result) == n_cols
        # Each column should have approximately 10% missing values
        for col_name, missing_count in result.items():
            assert missing_count == n_rows // 10

    def test_backward_compatibility_missing_cells(self):
        """Test that existing missing_cells method still works."""
        data = {
            "age": [25, 30, np.nan, 35, 40, np.nan, 45, 50, np.nan, 55],
            "income": [
                50000,
                np.nan,
                60000,
                70000,
                np.nan,
                80000,
                90000,
                np.nan,
                100000,
                110000,
            ],
            "education": [
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
            ],
        }
        df = pd.DataFrame(data)

        total_missing = self.adapter.missing_cells(df)
        per_column_missing = self.adapter.missing_cells_per_column(df)

        # Total should equal sum of per-column missing counts
        assert total_missing == sum(per_column_missing.values())


class TestPolarsAdapterChunkMetadata:
    """Test polars adapter per-column missing count functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            import polars as pl

            self.pl = pl
            self.adapter = PolarsAdapter()
        except ImportError:
            pytest.skip("Polars not available")

    def test_missing_cells_per_column_accuracy(self):
        """Test accuracy of per-column missing count calculation."""
        # Create test data with known missing patterns
        data = {
            "age": [25, 30, None, 35, 40, None, 45, 50, None, 55],
            "income": [
                50000,
                None,
                60000,
                70000,
                None,
                80000,
                90000,
                None,
                100000,
                110000,
            ],
            "education": [
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
            ],
        }
        df = self.pl.DataFrame(data)

        result = self.adapter.missing_cells_per_column(df)

        expected = {"age": 3, "income": 3, "education": 0}

        assert result == expected

    def test_missing_cells_per_column_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = self.pl.DataFrame()
        result = self.adapter.missing_cells_per_column(df)
        assert result == {}

    def test_missing_cells_per_column_no_missing(self):
        """Test with no missing values."""
        data = {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
        }
        df = self.pl.DataFrame(data)

        result = self.adapter.missing_cells_per_column(df)

        expected = {"age": 0, "income": 0}

        assert result == expected

    def test_backward_compatibility_missing_cells(self):
        """Test that existing missing_cells method still works."""
        data = {
            "age": [25, 30, None, 35, 40, None, 45, 50, None, 55],
            "income": [
                50000,
                None,
                60000,
                70000,
                None,
                80000,
                90000,
                None,
                100000,
                110000,
            ],
            "education": [
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
                "Master",
                "PhD",
                "Bachelor",
            ],
        }
        df = self.pl.DataFrame(data)

        total_missing = self.adapter.missing_cells(df)
        per_column_missing = self.adapter.missing_cells_per_column(df)

        # Total should equal sum of per-column missing counts
        assert total_missing == sum(per_column_missing.values())


class TestNumericAccumulatorChunkMetadata:
    """Test numeric accumulator integration with per-column chunk metadata."""

    def setup_method(self):
        """Set up test fixtures."""
        self.accumulator = NumericAccumulator("test_column")

    def test_finalize_with_per_column_chunk_metadata(self):
        """Test finalize method with per-column chunk metadata."""
        # Add some data to the accumulator
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.accumulator.update(data)

        # Create per-column chunk metadata
        per_column_chunk_metadata = {
            "test_column": [(0, 4, 0)]  # No missing values in this chunk
        }

        result = self.accumulator.finalize(
            chunk_metadata=None, per_column_chunk_metadata=per_column_chunk_metadata
        )

        assert isinstance(result, NumericSummary)
        assert result.per_column_chunk_metadata == per_column_chunk_metadata

    def test_finalize_with_both_metadata_types(self):
        """Test finalize method with both old and new metadata types."""
        # Add some data to the accumulator
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.accumulator.update(data)

        # Create both types of metadata
        chunk_metadata = [(0, 4, 0)]
        per_column_chunk_metadata = {"test_column": [(0, 4, 0)]}

        result = self.accumulator.finalize(
            chunk_metadata=chunk_metadata,
            per_column_chunk_metadata=per_column_chunk_metadata,
        )

        assert isinstance(result, NumericSummary)
        assert result.chunk_metadata == chunk_metadata
        assert result.per_column_chunk_metadata == per_column_chunk_metadata

    def test_finalize_backward_compatibility(self):
        """Test backward compatibility with old chunk_metadata only."""
        # Add some data to the accumulator
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.accumulator.update(data)

        # Use only old chunk_metadata
        chunk_metadata = [(0, 4, 0)]

        result = self.accumulator.finalize(chunk_metadata=chunk_metadata)

        assert isinstance(result, NumericSummary)
        assert result.chunk_metadata == chunk_metadata
        assert result.per_column_chunk_metadata is None

    def test_finalize_with_no_metadata(self):
        """Test finalize method with no metadata."""
        # Add some data to the accumulator
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.accumulator.update(data)

        result = self.accumulator.finalize()

        assert isinstance(result, NumericSummary)
        assert result.chunk_metadata is None
        assert result.per_column_chunk_metadata is None


class TestNumericCardRendererChunkMetadata:
    """Test numeric card renderer with per-column chunk metadata."""

    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = NumericCardRenderer()

    def test_build_chunk_distribution_with_per_column_metadata(self):
        """Test chunk distribution with per-column metadata."""
        # Create mock stats with per-column chunk metadata
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = 800
        stats.missing = 200

        # Mock per-column chunk metadata
        per_column_chunk_metadata = {
            "age": [
                (0, 399, 50),
                (400, 799, 100),
                (800, 999, 50),
            ]  # 12.5%, 25%, 12.5% missing
        }
        stats.per_column_chunk_metadata = per_column_chunk_metadata
        stats.chunk_metadata = None

        result = self.renderer._build_chunk_distribution(stats)

        assert "Missing Values Distribution" in result
        assert "3 chunks analyzed" in result
        assert "Peak: 25.0%" in result
        assert "chunk-segment" in result

    def test_build_chunk_distribution_fallback_to_old_metadata(self):
        """Test fallback to old chunk_metadata when per_column_chunk_metadata is not available."""
        # Create mock stats with old chunk metadata
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = 800
        stats.missing = 200

        # Mock old chunk metadata
        chunk_metadata = [(0, 399, 50), (400, 799, 100), (800, 999, 50)]
        stats.chunk_metadata = chunk_metadata
        stats.per_column_chunk_metadata = None

        result = self.renderer._build_chunk_distribution(stats)

        assert "Missing Values Distribution" in result
        assert "3 chunks analyzed" in result

    def test_build_chunk_distribution_no_metadata(self):
        """Test with no chunk metadata available."""
        # Create mock stats with no chunk metadata
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = 800
        stats.missing = 200
        stats.chunk_metadata = None
        stats.per_column_chunk_metadata = None

        result = self.renderer._build_chunk_distribution(stats)

        assert result == ""

    def test_build_chunk_distribution_empty_metadata(self):
        """Test with empty chunk metadata."""
        # Create mock stats with empty chunk metadata
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = 800
        stats.missing = 200
        stats.chunk_metadata = []
        stats.per_column_chunk_metadata = {"age": []}

        result = self.renderer._build_chunk_distribution(stats)

        assert result == ""

    def test_build_chunk_distribution_column_not_in_metadata(self):
        """Test when column name is not in per_column_chunk_metadata."""
        # Create mock stats where column name is not in per_column_chunk_metadata
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = 800
        stats.missing = 200

        # Mock per_column_chunk_metadata without the current column
        per_column_chunk_metadata = {
            "income": [(0, 399, 50), (400, 799, 100), (800, 999, 50)]
        }
        stats.per_column_chunk_metadata = per_column_chunk_metadata
        stats.chunk_metadata = None

        result = self.renderer._build_chunk_distribution(stats)

        assert result == ""


class TestChunkMetadataIntegration:
    """Integration tests for the complete chunk metadata system."""

    def test_titanic_dataset_simulation(self):
        """Test with Titanic-like dataset to verify the fix."""
        # Create Titanic-like dataset
        np.random.seed(42)
        n_rows = 891

        # Age column: ~20% missing (typical for Titanic)
        age_data = np.random.normal(30, 10, n_rows)
        age_missing_indices = np.random.choice(n_rows, size=178, replace=False)  # ~20%
        age_data[age_missing_indices] = np.nan

        # Cabin column: ~77% missing (typical for Titanic)
        cabin_data = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_rows)
        cabin_missing_indices = np.random.choice(
            n_rows, size=687, replace=False
        )  # ~77%
        cabin_data[cabin_missing_indices] = np.nan

        # Embarked column: ~0.2% missing (typical for Titanic)
        embarked_data = np.random.choice(["S", "C", "Q"], n_rows)
        embarked_missing_indices = np.random.choice(
            n_rows, size=2, replace=False
        )  # ~0.2%
        embarked_data[embarked_missing_indices] = np.nan

        df = pd.DataFrame(
            {"age": age_data, "cabin": cabin_data, "embarked": embarked_data}
        )

        # Test pandas adapter
        adapter = PandasAdapter()
        per_column_missing = adapter.missing_cells_per_column(df)

        # Verify the counts are correct
        assert per_column_missing["age"] == 178
        assert per_column_missing["cabin"] == 687
        assert per_column_missing["embarked"] == 2

        # Verify total missing matches sum of per-column missing
        total_missing = adapter.missing_cells(df)
        assert total_missing == sum(per_column_missing.values())

    def test_chunk_metadata_accuracy_verification(self):
        """Test that chunk metadata accurately reflects per-column missing counts."""
        # Create dataset with known missing patterns
        data = {
            "col1": [1, 2, np.nan, 4, 5, np.nan, 7, 8, np.nan, 10],  # 3 missing
            "col2": [np.nan, 2, 3, np.nan, 5, 6, np.nan, 8, 9, np.nan],  # 4 missing
            "col3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0 missing
        }
        df = pd.DataFrame(data)

        adapter = PandasAdapter()
        per_column_missing = adapter.missing_cells_per_column(df)

        # Verify accuracy
        assert per_column_missing["col1"] == 3
        assert per_column_missing["col2"] == 4
        assert per_column_missing["col3"] == 0

        # Test with chunk metadata simulation
        chunk_metadata = {
            "col1": [(0, 9, 3)],  # 3 missing in this chunk
            "col2": [(0, 9, 4)],  # 4 missing in this chunk
            "col3": [(0, 9, 0)],  # 0 missing in this chunk
        }

        # Verify chunk metadata accuracy
        for col_name, missing_count in per_column_missing.items():
            assert chunk_metadata[col_name][0][2] == missing_count

    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        np.random.seed(42)
        n_rows = 100000
        n_cols = 20

        # Create large dataset
        data = {}
        for i in range(n_cols):
            col_data = np.random.normal(0, 1, n_rows)
            # Add some missing values
            missing_indices = np.random.choice(n_rows, size=n_rows // 20, replace=False)
            col_data[missing_indices] = np.nan
            data[f"col_{i}"] = col_data

        df = pd.DataFrame(data)

        adapter = PandasAdapter()

        import time

        start_time = time.time()
        per_column_missing = adapter.missing_cells_per_column(df)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 2.0
        assert len(per_column_missing) == n_cols

        # Verify accuracy
        for col_name, missing_count in per_column_missing.items():
            assert missing_count == n_rows // 20

    def test_edge_cases(self):
        """Test various edge cases."""
        adapter = PandasAdapter()

        # Test with single row
        df_single = pd.DataFrame({"col1": [1], "col2": [np.nan]})
        result = adapter.missing_cells_per_column(df_single)
        assert result == {"col1": 0, "col2": 1}

        # Test with single column
        df_single_col = pd.DataFrame({"col1": [1, 2, np.nan, 4]})
        result = adapter.missing_cells_per_column(df_single_col)
        assert result == {"col1": 1}

        # Test with all NaN values
        df_all_nan = pd.DataFrame({"col1": [np.nan, np.nan, np.nan]})
        result = adapter.missing_cells_per_column(df_all_nan)
        assert result == {"col1": 3}

        # Test with mixed data types
        df_mixed = pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4],
                "string": ["a", "b", None, "d"],
                "boolean": [True, False, np.nan, True],
            }
        )
        result = adapter.missing_cells_per_column(df_mixed)
        assert result == {"numeric": 1, "string": 1, "boolean": 1}


if __name__ == "__main__":
    pytest.main([__file__])
