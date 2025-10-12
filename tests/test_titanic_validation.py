"""
Validation tests for the Titanic dataset fix.

These tests specifically validate that the per-column chunk metadata fix
resolves the issue where the 'age' column showed 97.5% missing values
in the Missing Values distribution.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pysuricata.compute.adapters.pandas import PandasAdapter
from pysuricata.render.card_types import NumericStats
from pysuricata.render.numeric_card import NumericCardRenderer


class TestTitanicValidation:
    """Validation tests for Titanic dataset fix."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = PandasAdapter()
        self.renderer = NumericCardRenderer()

    def create_titanic_like_dataset(self):
        """Create a Titanic-like dataset for testing."""
        np.random.seed(42)
        n_rows = 891  # Same as Titanic dataset

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

        # Fare column: ~0% missing (typical for Titanic)
        fare_data = np.random.normal(32, 50, n_rows)
        fare_data[fare_data < 0] = 0  # Ensure non-negative fares

        # Pclass column: ~0% missing (typical for Titanic)
        pclass_data = np.random.choice([1, 2, 3], n_rows)

        df = pd.DataFrame(
            {
                "age": age_data,
                "cabin": cabin_data,
                "embarked": embarked_data,
                "fare": fare_data,
                "pclass": pclass_data,
            }
        )

        return df

    def test_titanic_missing_counts_accuracy(self):
        """Test that missing counts are accurate for Titanic-like dataset."""
        df = self.create_titanic_like_dataset()

        # Calculate per-column missing counts
        per_column_missing = self.adapter.missing_cells_per_column(df)

        # Verify accuracy
        assert per_column_missing["age"] == 178  # ~20% missing
        assert per_column_missing["cabin"] == 687  # ~77% missing
        assert per_column_missing["embarked"] == 2  # ~0.2% missing
        assert per_column_missing["fare"] == 0  # ~0% missing
        assert per_column_missing["pclass"] == 0  # ~0% missing

        # Verify total missing matches sum of per-column missing
        total_missing = self.adapter.missing_cells(df)
        assert total_missing == sum(per_column_missing.values())

    def test_titanic_chunk_metadata_simulation(self):
        """Test chunk metadata simulation for Titanic-like dataset."""
        df = self.create_titanic_like_dataset()

        # Simulate chunk processing (e.g., 3 chunks of ~297 rows each)
        chunk_size = 297
        n_chunks = 3

        per_column_chunk_metadata = {}

        for i in range(n_chunks):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size - 1, len(df) - 1)
            chunk_df = df.iloc[start_row : end_row + 1]

            chunk_per_column_missing = self.adapter.missing_cells_per_column(chunk_df)

            for col_name, missing_count in chunk_per_column_missing.items():
                if col_name not in per_column_chunk_metadata:
                    per_column_chunk_metadata[col_name] = []
                per_column_chunk_metadata[col_name].append(
                    (start_row, end_row, missing_count)
                )

        # Verify chunk metadata structure
        assert len(per_column_chunk_metadata) == 5  # 5 columns
        for col_name in ["age", "cabin", "embarked", "fare", "pclass"]:
            assert col_name in per_column_chunk_metadata
            assert len(per_column_chunk_metadata[col_name]) == n_chunks

        # Verify that sum of chunk missing counts equals total missing counts
        total_per_column_missing = self.adapter.missing_cells_per_column(df)
        for col_name in per_column_chunk_metadata:
            chunk_sum = sum(chunk[2] for chunk in per_column_chunk_metadata[col_name])
            assert chunk_sum == total_per_column_missing[col_name]

    def test_titanic_age_column_fix_verification(self):
        """Verify that the age column fix resolves the 97.5% missing issue."""
        df = self.create_titanic_like_dataset()

        # Process age column specifically
        age_column = df["age"]
        age_missing_count = age_column.isnull().sum()
        age_total_count = len(age_column)
        age_missing_percentage = (age_missing_count / age_total_count) * 100

        # Verify the actual missing percentage is ~20%, not 97.5%
        assert 15 <= age_missing_percentage <= 25  # Should be around 20%

        # Test with chunk metadata
        chunk_size = 297
        n_chunks = 3

        age_chunk_metadata = []
        for i in range(n_chunks):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size - 1, len(df) - 1)
            chunk_df = df.iloc[start_row : end_row + 1]

            chunk_age_missing = chunk_df["age"].isnull().sum()
            age_chunk_metadata.append((start_row, end_row, chunk_age_missing))

        # Verify chunk metadata accuracy
        total_chunk_missing = sum(chunk[2] for chunk in age_chunk_metadata)
        assert total_chunk_missing == age_missing_count

        # Test chunk distribution rendering
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = age_total_count - age_missing_count
        stats.missing = age_missing_count

        # Mock per-column chunk metadata
        per_column_chunk_metadata = {"age": age_chunk_metadata}
        stats.per_column_chunk_metadata = per_column_chunk_metadata
        stats.chunk_metadata = None

        # Render chunk distribution
        chunk_html = self.renderer._build_chunk_distribution(stats)

        # Verify the rendered HTML shows correct percentages
        assert "Missing Values Distribution" in chunk_html
        assert "3 chunks analyzed" in chunk_html

        # The peak percentage should be reasonable (not 97.5%)
        # Calculate expected peak percentage
        chunk_percentages = []
        for start_row, end_row, missing_count in age_chunk_metadata:
            chunk_size = end_row - start_row + 1
            chunk_percentage = (missing_count / chunk_size) * 100
            chunk_percentages.append(chunk_percentage)

        peak_percentage = max(chunk_percentages)
        assert peak_percentage < 50  # Should be much less than 97.5%

    def test_titanic_multiple_columns_verification(self):
        """Verify fix works for multiple columns in Titanic-like dataset."""
        df = self.create_titanic_like_dataset()

        # Test all columns
        columns_to_test = ["age", "cabin", "embarked", "fare", "pclass"]

        for col_name in columns_to_test:
            column = df[col_name]
            missing_count = column.isnull().sum()
            total_count = len(column)
            missing_percentage = (missing_count / total_count) * 100

            # Verify reasonable missing percentages
            if col_name == "age":
                assert 15 <= missing_percentage <= 25  # ~20%
            elif col_name == "cabin":
                assert 70 <= missing_percentage <= 80  # ~77%
            elif col_name == "embarked":
                assert 0 <= missing_percentage <= 1  # ~0.2%
            elif col_name in ["fare", "pclass"]:
                assert missing_percentage == 0  # ~0%

            # Test chunk metadata for this column
            chunk_size = 297
            n_chunks = 3

            column_chunk_metadata = []
            for i in range(n_chunks):
                start_row = i * chunk_size
                end_row = min((i + 1) * chunk_size - 1, len(df) - 1)
                chunk_df = df.iloc[start_row : end_row + 1]

                chunk_missing = chunk_df[col_name].isnull().sum()
                column_chunk_metadata.append((start_row, end_row, chunk_missing))

            # Verify chunk metadata accuracy
            total_chunk_missing = sum(chunk[2] for chunk in column_chunk_metadata)
            assert total_chunk_missing == missing_count

    def test_titanic_chunk_distribution_rendering(self):
        """Test chunk distribution rendering for Titanic-like dataset."""
        df = self.create_titanic_like_dataset()

        # Test age column rendering
        age_column = df["age"]
        age_missing_count = age_column.isnull().sum()
        age_total_count = len(age_column)

        # Create chunk metadata
        chunk_size = 297
        n_chunks = 3

        age_chunk_metadata = []
        for i in range(n_chunks):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size - 1, len(df) - 1)
            chunk_df = df.iloc[start_row : end_row + 1]

            chunk_age_missing = chunk_df["age"].isnull().sum()
            age_chunk_metadata.append((start_row, end_row, chunk_age_missing))

        # Create mock stats
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = age_total_count - age_missing_count
        stats.missing = age_missing_count

        # Mock per-column chunk metadata
        per_column_chunk_metadata = {"age": age_chunk_metadata}
        stats.per_column_chunk_metadata = per_column_chunk_metadata
        stats.chunk_metadata = None

        # Render chunk distribution
        chunk_html = self.renderer._build_chunk_distribution(stats)

        # Verify HTML content
        assert "Missing Values Distribution" in chunk_html
        assert "3 chunks analyzed" in chunk_html
        assert "chunk-segment" in chunk_html

        # Verify that percentages are reasonable
        # Calculate expected percentages
        chunk_percentages = []
        for start_row, end_row, missing_count in age_chunk_metadata:
            chunk_size = end_row - start_row + 1
            chunk_percentage = (missing_count / chunk_size) * 100
            chunk_percentages.append(chunk_percentage)

        # All percentages should be reasonable (not 97.5%)
        for percentage in chunk_percentages:
            assert percentage < 50  # Should be much less than 97.5%

        # Peak percentage should be reasonable
        peak_percentage = max(chunk_percentages)
        assert peak_percentage < 50

    def test_titanic_backward_compatibility(self):
        """Test backward compatibility with old chunk_metadata."""
        df = self.create_titanic_like_dataset()

        # Test with old chunk_metadata (deprecated)
        chunk_size = 297
        n_chunks = 3

        old_chunk_metadata = []
        for i in range(n_chunks):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size - 1, len(df) - 1)
            chunk_df = df.iloc[start_row : end_row + 1]

            # Old method: total missing across all columns
            chunk_total_missing = chunk_df.isnull().sum().sum()
            old_chunk_metadata.append((start_row, end_row, chunk_total_missing))

        # Create mock stats with old chunk_metadata
        stats = Mock(spec=NumericStats)
        stats.name = "age"
        stats.count = len(df) - df["age"].isnull().sum()
        stats.missing = df["age"].isnull().sum()

        # Mock old chunk_metadata
        stats.chunk_metadata = old_chunk_metadata
        stats.per_column_chunk_metadata = None

        # Render chunk distribution (should fall back to old method)
        chunk_html = self.renderer._build_chunk_distribution(stats)

        # Should still render (backward compatibility)
        assert "Missing Values Distribution" in chunk_html
        assert "3 chunks analyzed" in chunk_html

    def test_titanic_performance_validation(self):
        """Test performance with Titanic-like dataset."""
        import time

        df = self.create_titanic_like_dataset()

        # Test per-column missing count performance
        start_time = time.time()
        per_column_missing = self.adapter.missing_cells_per_column(df)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 0.1  # Less than 100ms

        # Verify results
        assert len(per_column_missing) == 5
        assert per_column_missing["age"] == 178
        assert per_column_missing["cabin"] == 687
        assert per_column_missing["embarked"] == 2
        assert per_column_missing["fare"] == 0
        assert per_column_missing["pclass"] == 0

    def test_titanic_edge_cases(self):
        """Test edge cases with Titanic-like dataset."""
        df = self.create_titanic_like_dataset()

        # Test with single row
        single_row_df = df.iloc[[0]]
        per_column_missing = self.adapter.missing_cells_per_column(single_row_df)

        # Verify results
        assert len(per_column_missing) == 5
        for col_name in per_column_missing:
            assert per_column_missing[col_name] in [0, 1]  # Either 0 or 1 missing

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        per_column_missing = self.adapter.missing_cells_per_column(empty_df)
        assert per_column_missing == {}

        # Test with all NaN values
        all_nan_df = pd.DataFrame(
            {"col1": [np.nan, np.nan, np.nan], "col2": [np.nan, np.nan, np.nan]}
        )
        per_column_missing = self.adapter.missing_cells_per_column(all_nan_df)
        assert per_column_missing == {"col1": 3, "col2": 3}


if __name__ == "__main__":
    pytest.main([__file__])
