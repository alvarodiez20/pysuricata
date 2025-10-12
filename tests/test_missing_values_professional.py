"""
Test suite for professional missing values design in numerical variable cards.

This test suite verifies that the simplified missing values section works correctly
across all edge cases and that other tabs remain unaffected.
"""

from unittest.mock import Mock

import pytest

from pysuricata.accumulators.numeric import NumericStats
from pysuricata.render.numeric_card import NumericCard


class TestMissingValuesProfessional:
    """Test the professional missing values design."""

    def setup_method(self):
        """Set up test fixtures."""
        self.card = NumericCard()

    def create_numeric_stats(self, count: int, missing: int, chunk_metadata=None):
        """Create NumericStats object for testing."""
        stats = Mock(spec=NumericStats)
        stats.count = count
        stats.missing = missing
        stats.zeros = 0
        stats.inf = 0
        stats.negatives = 0
        stats.chunk_metadata = chunk_metadata
        return stats

    def test_no_missing_values(self):
        """Test with 0% missing values."""
        stats = self.create_numeric_stats(count=1000, missing=0)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 1,000 (100.0%)" in html
        assert "Missing: 0 (0.0%)" in html
        assert "bar-fill present" in html
        assert "bar-fill missing" not in html or "width: 0.0%" in html

    def test_low_missing_values(self):
        """Test with < 5% missing values."""
        stats = self.create_numeric_stats(count=950, missing=50)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 950 (95.0%)" in html
        assert "Missing: 50 (5.0%)" in html
        assert "bar-fill present" in html
        assert "bar-fill missing" in html

    def test_medium_missing_values(self):
        """Test with 5-20% missing values."""
        stats = self.create_numeric_stats(count=800, missing=200)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 800 (80.0%)" in html
        assert "Missing: 200 (20.0%)" in html

    def test_high_missing_values(self):
        """Test with > 20% missing values."""
        stats = self.create_numeric_stats(count=500, missing=500)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 500 (50.0%)" in html
        assert "Missing: 500 (50.0%)" in html

    def test_all_missing_values(self):
        """Test with 100% missing values."""
        stats = self.create_numeric_stats(count=0, missing=1000)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 0 (0.0%)" in html
        assert "Missing: 1,000 (100.0%)" in html
        assert "bar-fill present" not in html or "width: 0.0%" in html
        assert "bar-fill missing" in html

    def test_with_chunk_metadata(self):
        """Test with chunk metadata (chunk mode)."""
        chunk_metadata = [
            (0, 199, 10),  # 5% missing
            (200, 399, 40),  # 20% missing
            (400, 599, 5),  # 2.5% missing
        ]
        stats = self.create_numeric_stats(
            count=545, missing=55, chunk_metadata=chunk_metadata
        )
        html = self.card._build_missing_values_table(stats)

        assert "Missing Values Distribution" in html
        assert "3 chunks analyzed" in html
        assert "Peak: 20.0%" in html
        assert "chunk-segment" in html
        assert "chunk-legend" in html

    def test_without_chunk_metadata(self):
        """Test without chunk metadata (non-chunk mode)."""
        stats = self.create_numeric_stats(count=800, missing=200, chunk_metadata=None)
        html = self.card._build_missing_values_table(stats)

        # Should only show completeness, no chunk distribution
        assert "Data Completeness" in html
        assert "Missing Values Distribution" not in html
        assert "chunk-segment" not in html

    def test_empty_chunk_metadata(self):
        """Test with empty chunk metadata."""
        stats = self.create_numeric_stats(count=800, missing=200, chunk_metadata=[])
        html = self.card._build_missing_values_table(stats)

        # Should only show completeness, no chunk distribution
        assert "Data Completeness" in html
        assert "Missing Values Distribution" not in html

    def test_large_dataset_multiple_chunks(self):
        """Test large dataset with multiple chunks."""
        chunk_metadata = [
            (0, 999, 50),  # 5% missing
            (1000, 1999, 200),  # 20% missing
            (2000, 2999, 100),  # 10% missing
            (3000, 3999, 25),  # 2.5% missing
            (4000, 4999, 75),  # 7.5% missing
        ]
        stats = self.create_numeric_stats(
            count=4550, missing=450, chunk_metadata=chunk_metadata
        )
        html = self.card._build_missing_values_table(stats)

        assert "5 chunks analyzed" in html
        assert "Peak: 20.0%" in html
        assert "chunk-segment" in html

    def test_single_chunk_dataset(self):
        """Test single chunk dataset."""
        chunk_metadata = [(0, 999, 100)]  # 10% missing
        stats = self.create_numeric_stats(
            count=900, missing=100, chunk_metadata=chunk_metadata
        )
        html = self.card._build_missing_values_table(stats)

        assert "1 chunks analyzed" in html
        assert "Peak: 10.0%" in html

    def test_edge_case_single_row(self):
        """Test edge case: single row dataset."""
        stats = self.create_numeric_stats(count=1, missing=0)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 1 (100.0%)" in html
        assert "Missing: 0 (0.0%)" in html

    def test_edge_case_single_value_missing(self):
        """Test edge case: single value missing."""
        stats = self.create_numeric_stats(count=0, missing=1)
        html = self.card._build_missing_values_table(stats)

        assert "Present: 0 (0.0%)" in html
        assert "Missing: 1 (100.0%)" in html

    def test_chunk_distribution_colors(self):
        """Test chunk distribution color classes."""
        chunk_metadata = [
            (0, 99, 2),  # 2% missing - should be "low"
            (100, 199, 10),  # 10% missing - should be "medium"
            (200, 299, 50),  # 50% missing - should be "high"
        ]
        stats = self.create_numeric_stats(
            count=240, missing=62, chunk_metadata=chunk_metadata
        )
        html = self.card._build_missing_values_table(stats)

        assert 'class="chunk-segment low"' in html
        assert 'class="chunk-segment medium"' in html
        assert 'class="chunk-segment high"' in html

    def test_chunk_tooltips(self):
        """Test chunk tooltips contain correct information."""
        chunk_metadata = [(0, 199, 20)]  # 10% missing
        stats = self.create_numeric_stats(
            count=180, missing=20, chunk_metadata=chunk_metadata
        )
        html = self.card._build_missing_values_table(stats)

        assert 'title="Rows 0-199: 20 missing (10.0%)"' in html

    def test_professional_styling_classes(self):
        """Test that professional styling classes are present."""
        stats = self.create_numeric_stats(count=800, missing=200)
        html = self.card._build_missing_values_table(stats)

        # Check for professional CSS classes
        assert 'class="missing-analysis-header"' in html
        assert 'class="section-title"' in html
        assert 'class="completeness-container"' in html
        assert 'class="completeness-stats"' in html
        assert 'class="completeness-bar"' in html
        assert 'class="stat-item"' in html
        assert 'class="stat-label"' in html
        assert 'class="stat-value"' in html

    def test_no_emojis_or_casual_language(self):
        """Test that no emojis or casual language are present."""
        stats = self.create_numeric_stats(count=800, missing=200)
        html = self.card._build_missing_values_table(stats)

        # Check for absence of emojis and casual language
        emojis = ["🚨", "⚠️", "⚡", "✅", "❓", "0️⃣", "∞", "➖", "🔍", "💡", "📊"]
        for emoji in emojis:
            assert emoji not in html, f"Emoji {emoji} found in HTML"

        casual_phrases = ["Hover over", "chunk details", "DataPrep", "spectrum"]
        for phrase in casual_phrases:
            assert phrase not in html, f"Casual phrase '{phrase}' found in HTML"

    def test_chunk_metadata_edge_cases(self):
        """Test edge cases in chunk metadata."""
        # Test with zero-sized chunk
        chunk_metadata = [(0, 0, 0)]  # Zero-sized chunk
        stats = self.create_numeric_stats(
            count=100, missing=0, chunk_metadata=chunk_metadata
        )
        html = self.card._build_missing_values_table(stats)

        # Should handle gracefully without errors
        assert "Data Completeness" in html

    def test_percentage_calculations(self):
        """Test percentage calculations are accurate."""
        stats = self.create_numeric_stats(count=750, missing=250)
        html = self.card._build_missing_values_table(stats)

        # Check for accurate percentages
        assert "Present: 750 (75.0%)" in html
        assert "Missing: 250 (25.0%)" in html

    def test_large_numbers_formatting(self):
        """Test formatting of large numbers."""
        stats = self.create_numeric_stats(count=1234567, missing=123456)
        html = self.card._build_missing_values_table(stats)

        # Check for proper number formatting with commas
        assert "Present: 1,234,567" in html
        assert "Missing: 123,456" in html


class TestIsolationVerification:
    """Test that changes don't affect other tabs or variable types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.card = NumericCard()

    def test_statistics_tab_unaffected(self):
        """Test that Statistics tab still works correctly."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.zeros = 50
        stats.inf = 0
        stats.negatives = 25
        stats.mean = 10.5
        stats.std = 2.3
        stats.min = 1.0
        stats.max = 20.0
        stats.quantiles = Mock()
        stats.quantiles.q25 = 5.0
        stats.quantiles.q50 = 10.0
        stats.quantiles.q75 = 15.0

        # Test that statistics table still builds
        html = self.card._build_stats_table(stats)
        assert "Statistics" in html or "Mean" in html or "Std" in html

    def test_common_values_tab_unaffected(self):
        """Test that Common values tab still works correctly."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.common_values = [
            (10.0, 100),
            (20.0, 80),
            (30.0, 60),
        ]

        # Test that common values table still builds
        html = self.card._build_common_values_table(stats)
        assert "Common values" in html or "Value" in html or "Count" in html

    def test_extremes_tab_unaffected(self):
        """Test that Extremes tab still works correctly."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.min = 1.0
        stats.max = 100.0
        stats.min_values = [1.0, 1.1, 1.2]
        stats.max_values = [100.0, 99.9, 99.8]

        # Test that extremes table still builds
        html = self.card._build_extremes_table(stats)
        assert "Min/Max" in html or "Minimum" in html or "Maximum" in html

    def test_outliers_tab_unaffected(self):
        """Test that Outliers tab still works correctly."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.outliers_low = []
        stats.outliers_high = []

        # Test that outliers tables still build
        low_html, high_html = self.card._build_outliers_tables(stats)
        assert "outliers" in low_html.lower() or "outliers" in high_html.lower()

    def test_correlations_tab_unaffected(self):
        """Test that Correlations tab still works correctly."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.correlations = []

        # Test that correlation table still builds
        html = self.card._build_correlation_table(stats)
        assert "Correlations" in html or "correlation" in html.lower()

    def test_details_section_structure(self):
        """Test that details section structure is preserved."""
        html = self.card._build_details_section(
            col_id="test_col",
            stats_quantiles="<div>Stats</div>",
            common_table="<div>Common</div>",
            extremes_table="<div>Extremes</div>",
            outliers_low="<div>Low</div>",
            outliers_high="<div>High</div>",
            corr_table="<div>Corr</div>",
            missing_table="<div>Missing</div>",
        )

        # Check that all tabs are present
        assert 'data-tab="stats"' in html
        assert 'data-tab="common"' in html
        assert 'data-tab="extremes"' in html
        assert 'data-tab="outliers"' in html
        assert 'data-tab="corr"' in html
        assert 'data-tab="missing"' in html


class TestChunkModeCompatibility:
    """Test chunk mode compatibility and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.card = NumericCard()

    def test_chunk_metadata_structure(self):
        """Test that chunk metadata structure is handled correctly."""
        chunk_metadata = [
            (0, 99, 5),  # 5% missing
            (100, 199, 20),  # 20% missing
            (200, 299, 10),  # 10% missing
        ]
        stats = Mock(spec=NumericStats)
        stats.count = 270
        stats.missing = 35
        stats.chunk_metadata = chunk_metadata

        html = self.card._build_chunk_distribution(stats)

        # Verify chunk distribution is built
        assert "Missing Values Distribution" in html
        assert "3 chunks analyzed" in html
        assert "Peak: 20.0%" in html

    def test_chunk_segment_widths(self):
        """Test that chunk segment widths are proportional."""
        chunk_metadata = [
            (0, 199, 10),  # 200 rows, 5% missing
            (200, 299, 20),  # 100 rows, 20% missing
        ]
        stats = Mock(spec=NumericStats)
        stats.count = 270
        stats.missing = 30
        stats.chunk_metadata = chunk_metadata

        html = self.card._build_chunk_distribution(stats)

        # First chunk should be 2/3 width, second chunk 1/3 width
        assert 'style="width: 66.67%"' in html  # 200/300 * 100
        assert 'style="width: 33.33%"' in html  # 100/300 * 100

    def test_chunk_colors_by_missing_percentage(self):
        """Test that chunk colors reflect missing percentages correctly."""
        chunk_metadata = [
            (0, 99, 2),  # 2% missing - low
            (100, 199, 15),  # 15% missing - medium
            (200, 299, 50),  # 50% missing - high
        ]
        stats = Mock(spec=NumericStats)
        stats.count = 250
        stats.missing = 67
        stats.chunk_metadata = chunk_metadata

        html = self.card._build_chunk_distribution(stats)

        # Check color classes
        assert 'class="chunk-segment low"' in html
        assert 'class="chunk-segment medium"' in html
        assert 'class="chunk-segment high"' in html

    def test_chunk_tooltip_accuracy(self):
        """Test that chunk tooltips show accurate information."""
        chunk_metadata = [(0, 149, 15)]  # 150 rows, 10% missing
        stats = Mock(spec=NumericStats)
        stats.count = 135
        stats.missing = 15
        stats.chunk_metadata = chunk_metadata

        html = self.card._build_chunk_distribution(stats)

        # Check tooltip content
        assert 'title="Rows 0-149: 15 missing (10.0%)"' in html

    def test_single_chunk_scenario(self):
        """Test single chunk scenario (non-chunked data)."""
        chunk_metadata = [(0, 999, 50)]  # Single chunk, 5% missing
        stats = Mock(spec=NumericStats)
        stats.count = 950
        stats.missing = 50
        stats.chunk_metadata = chunk_metadata

        html = self.card._build_chunk_distribution(stats)

        assert "1 chunks analyzed" in html
        assert "Peak: 5.0%" in html

    def test_many_chunks_scenario(self):
        """Test many chunks scenario (large dataset)."""
        chunk_metadata = [
            (i * 100, (i + 1) * 100 - 1, i * 2) for i in range(10)
        ]  # 10 chunks
        stats = Mock(spec=NumericStats)
        stats.count = 980
        stats.missing = 90
        stats.chunk_metadata = chunk_metadata

        html = self.card._build_chunk_distribution(stats)

        assert "10 chunks analyzed" in html

    def test_backward_compatibility_no_chunk_metadata(self):
        """Test backward compatibility when chunk_metadata is None."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.chunk_metadata = None

        html = self.card._build_chunk_distribution(stats)

        # Should return empty string
        assert html == ""

    def test_backward_compatibility_empty_chunk_metadata(self):
        """Test backward compatibility when chunk_metadata is empty."""
        stats = Mock(spec=NumericStats)
        stats.count = 1000
        stats.missing = 100
        stats.chunk_metadata = []

        html = self.card._build_chunk_distribution(stats)

        # Should return empty string
        assert html == ""


if __name__ == "__main__":
    pytest.main([__file__])
