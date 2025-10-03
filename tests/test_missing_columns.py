"""Comprehensive tests for intelligent missing columns functionality.

This module tests the new dynamic missing columns analysis and rendering system,
ensuring it works correctly across different dataset sizes and scenarios.
"""

import pytest

from pysuricata.render.missing_columns import (
    MissingColumnsAnalyzer,
    MissingColumnsRenderer,
    create_missing_columns_renderer,
)


class TestMissingColumnsAnalyzer:
    """Test the intelligent missing columns analyzer."""

    def test_small_dataset_all_columns_shown(self):
        """Test that small datasets show all columns."""
        analyzer = MissingColumnsAnalyzer()
        miss_list = [
            ("col1", 5.0, 100),
            ("col2", 3.0, 60),
            ("col3", 1.0, 20),
        ]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=3, n_rows=1000)

        assert len(result.initial_columns) == 3
        assert len(result.expanded_columns) == 3
        assert not result.needs_expandable
        assert result.total_significant == 3
        assert result.total_insignificant == 0

    def test_medium_dataset_limited_initial(self):
        """Test that medium datasets show limited initial columns."""
        analyzer = MissingColumnsAnalyzer()
        miss_list = [(f"col{i}", 10.0 - i, 100) for i in range(15)]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=15, n_rows=1000)

        assert len(result.initial_columns) <= 10
        assert len(result.expanded_columns) <= 20
        assert result.needs_expandable
        assert result.total_significant == 10

    def test_large_dataset_dynamic_limits(self):
        """Test that large datasets use appropriate dynamic limits."""
        analyzer = MissingColumnsAnalyzer()
        miss_list = [(f"col{i}", 15.0 - (i % 10), 100) for i in range(100)]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=100, n_rows=10000)

        assert len(result.initial_columns) <= 15
        assert len(result.expanded_columns) <= 30
        assert result.needs_expandable
        assert result.total_significant == 100

    def test_threshold_filtering(self):
        """Test that columns below threshold are filtered out."""
        analyzer = MissingColumnsAnalyzer(min_threshold_pct=2.0)
        miss_list = [
            ("col1", 5.0, 100),  # Above threshold
            ("col2", 1.5, 30),  # Below threshold
            ("col3", 0.3, 6),  # Below threshold
            ("col4", 8.0, 160),  # Above threshold
        ]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=4, n_rows=1000)

        assert len(result.initial_columns) == 2  # Only col1 and col4
        assert result.total_significant == 2
        assert result.total_insignificant == 2
        assert result.initial_columns[0][0] == "col1"  # Sorted by percentage
        assert result.initial_columns[1][0] == "col4"

    def test_no_significant_missing(self):
        """Test behavior when no columns have significant missing data."""
        analyzer = MissingColumnsAnalyzer(min_threshold_pct=5.0)
        miss_list = [
            ("col1", 1.0, 20),
            ("col2", 0.5, 10),
            ("col3", 2.0, 40),
        ]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=3, n_rows=1000)

        assert len(result.initial_columns) == 0
        assert result.total_significant == 0
        assert result.total_insignificant == 3

    def test_custom_threshold(self):
        """Test custom threshold configuration."""
        analyzer = MissingColumnsAnalyzer(min_threshold_pct=10.0)
        miss_list = [
            ("col1", 15.0, 300),
            ("col2", 8.0, 160),
            ("col3", 12.0, 240),
        ]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=3, n_rows=1000)

        assert len(result.initial_columns) == 2  # Only col1 and col3
        assert result.threshold_used == 10.0


class TestMissingColumnsRenderer:
    """Test the missing columns HTML renderer."""

    def test_render_no_missing_columns(self):
        """Test rendering when no significant missing columns exist."""
        renderer = MissingColumnsRenderer()
        miss_list = [
            ("col1", 0.1, 2),
            ("col2", 0.3, 6),
        ]

        html = renderer.render_missing_columns_html(miss_list, n_cols=2, n_rows=1000)

        assert "No significant missing data" in html
        assert "expand-btn" not in html  # No expand button needed

    def test_render_small_dataset_no_expand(self):
        """Test rendering small dataset without expand functionality."""
        renderer = MissingColumnsRenderer()
        miss_list = [
            ("col1", 5.0, 100),
            ("col2", 3.0, 60),
        ]

        html = renderer.render_missing_columns_html(miss_list, n_cols=2, n_rows=1000)

        assert "col1" in html
        assert "col2" in html
        assert "expand-btn" not in html  # No expand button needed
        assert "toggleMissingColumns" not in html  # No JavaScript needed

    def test_render_large_dataset_with_expand(self):
        """Test rendering large dataset with expand functionality."""
        renderer = MissingColumnsRenderer()
        miss_list = [(f"col{i}", 10.0 - i, 100) for i in range(20)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=20, n_rows=1000)

        assert "col0" in html  # First column should be visible
        assert "expand-btn" in html  # Expand button should be present
        assert "toggleMissingColumns" in html  # JavaScript should be present
        assert "Show " in html and "more..." in html  # Expand text should be present

    def test_severity_classification(self):
        """Test that severity classes are correctly assigned."""
        renderer = MissingColumnsRenderer()
        miss_list = [
            ("low_col", 3.0, 60),  # Should be "low"
            ("medium_col", 15.0, 300),  # Should be "medium"
            ("high_col", 25.0, 500),  # Should be "high"
        ]

        html = renderer.render_missing_columns_html(miss_list, n_cols=3, n_rows=1000)

        assert 'class="missing-fill low"' in html
        assert 'class="missing-fill medium"' in html
        assert 'class="missing-fill high"' in html

    def test_html_escaping(self):
        """Test that column names are properly HTML escaped."""
        renderer = MissingColumnsRenderer()
        miss_list = [
            ("col<with>tags", 5.0, 100),
            ("col&with&ampersands", 3.0, 60),
            ('col"with"quotes', 2.0, 40),
        ]

        html = renderer.render_missing_columns_html(miss_list, n_cols=3, n_rows=1000)

        assert "col&lt;with&gt;tags" in html
        assert "col&amp;with&amp;ampersands" in html
        assert "col&quot;with&quot;quotes" in html


class TestFactoryFunction:
    """Test the factory function for creating renderers."""

    def test_create_renderer_default_config(self):
        """Test creating renderer with default configuration."""
        renderer = create_missing_columns_renderer()

        assert isinstance(renderer, MissingColumnsRenderer)
        assert renderer.analyzer.min_threshold_pct == 0.5

    def test_create_renderer_custom_threshold(self):
        """Test creating renderer with custom threshold."""
        renderer = create_missing_columns_renderer(min_threshold_pct=2.0)

        assert isinstance(renderer, MissingColumnsRenderer)
        assert renderer.analyzer.min_threshold_pct == 2.0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_huge_dataset_scenario(self):
        """Test behavior with a realistic huge dataset scenario."""
        # Simulate a dataset with 500 columns, 1M rows
        miss_list = [(f"column_{i:03d}", 20.0 - (i % 20), 1000) for i in range(500)]

        renderer = create_missing_columns_renderer(min_threshold_pct=0.5)
        result = renderer.analyzer.analyze_missing_columns(
            miss_list, n_cols=500, n_rows=1000000
        )

        # Should show limited initial columns
        assert len(result.initial_columns) <= 15
        # Should have expand functionality
        assert result.needs_expandable
        # Should show many columns when expanded
        assert len(result.expanded_columns) <= 30

        html = renderer.render_missing_columns_html(
            miss_list, n_cols=500, n_rows=1000000
        )
        assert "expand-btn" in html
        assert "toggleMissingColumns" in html

    def test_edge_case_empty_dataset(self):
        """Test behavior with empty dataset."""
        renderer = create_missing_columns_renderer()

        html = renderer.render_missing_columns_html([], n_cols=0, n_rows=0)
        assert "No significant missing data" in html

    def test_edge_case_all_columns_insignificant(self):
        """Test behavior when all columns have insignificant missing data."""
        miss_list = [
            ("col1", 0.1, 1),
            ("col2", 0.2, 2),
            ("col3", 0.3, 3),
        ]

        renderer = create_missing_columns_renderer(min_threshold_pct=1.0)
        html = renderer.render_missing_columns_html(miss_list, n_cols=3, n_rows=1000)

        assert "No significant missing data" in html
        assert "expand-btn" not in html


if __name__ == "__main__":
    pytest.main([__file__])
