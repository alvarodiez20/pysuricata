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
        """Test that small datasets show all columns (up to 5)."""
        analyzer = MissingColumnsAnalyzer()
        miss_list = [
            ("col1", 5.0, 100),
            ("col2", 3.0, 60),
            ("col3", 1.0, 20),
        ]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=3, n_rows=1000)

        assert len(result.columns) == 3
        assert result.total_significant == 3
        assert result.total_insignificant == 0

    def test_medium_dataset_capped_at_five(self):
        """Test that medium datasets are capped at 5 columns."""
        analyzer = MissingColumnsAnalyzer()
        # Create 15 columns but only 10 have > 0% missing (col0-col9: 10.0% - 1.0%)
        miss_list = [(f"col{i}", 10.0 - i, 100) for i in range(15)]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=15, n_rows=1000)

        assert len(result.columns) == 5  # Capped at 5
        assert result.total_significant == 10  # 10 columns have > 0% missing

    def test_large_dataset_capped_at_five(self):
        """Test that large datasets are also capped at 5 columns."""
        analyzer = MissingColumnsAnalyzer()
        miss_list = [(f"col{i}", 15.0 - (i % 10), 100) for i in range(100)]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=100, n_rows=10000)

        assert len(result.columns) == 5
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

        assert len(result.columns) == 2  # Only col1 and col4
        assert result.total_significant == 2
        assert result.total_insignificant == 2
        assert result.columns[0][0] == "col1"  # Sorted by percentage
        assert result.columns[1][0] == "col4"

    def test_no_significant_missing(self):
        """Test behavior when no columns have significant missing data."""
        analyzer = MissingColumnsAnalyzer(min_threshold_pct=5.0)
        miss_list = [
            ("col1", 1.0, 20),
            ("col2", 0.5, 10),
            ("col3", 2.0, 40),
        ]

        result = analyzer.analyze_missing_columns(miss_list, n_cols=3, n_rows=1000)

        assert len(result.columns) == 0
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

        assert len(result.columns) == 2  # Only col1 and col3
        assert result.threshold_used == 10.0


class TestMissingColumnsRenderer:
    """Test the missing columns HTML renderer."""

    def test_render_no_missing_columns(self):
        """Test rendering when no significant missing columns exist."""
        renderer = create_missing_columns_renderer(min_threshold_pct=0.5)
        miss_list = [
            ("col1", 0.1, 2),
            ("col2", 0.3, 6),
        ]

        html = renderer.render_missing_columns_html(miss_list, n_cols=2, n_rows=1000)

        assert "No column has missing values" in html
        assert "expand-btn" not in html  # No expand button

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
        assert "expand-btn" not in html  # No expand button
        assert "toggleMissingColumns" not in html  # No JavaScript

    def test_render_large_dataset_caps_at_five(self):
        """Test rendering large dataset caps at 5 columns (no expand)."""
        renderer = MissingColumnsRenderer()
        miss_list = [(f"col{i}", 10.0 - i, 100) for i in range(20)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=20, n_rows=1000)

        assert "col0" in html  # First column should be visible
        assert "col4" in html  # Fifth column should be visible
        assert "col5" not in html  # Sixth column should NOT be visible
        assert "expand-btn" not in html  # No expand button
        assert "toggleMissingColumns" not in html  # No JavaScript

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
        """The factory used to default to 0.5 while the class said 0.0 and the
        config said 0.0. The config wins in the render path, so the factory
        never actually applied 0.5 to a report -- it only made the code read as
        though it did."""
        renderer = create_missing_columns_renderer()

        assert isinstance(renderer, MissingColumnsRenderer)
        assert (
            renderer.analyzer.min_threshold_pct
            == MissingColumnsAnalyzer.MIN_THRESHOLD_PCT
        )

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

        # Should show max 5 columns
        assert len(result.columns) == 5
        assert result.total_significant == 500

        html = renderer.render_missing_columns_html(
            miss_list, n_cols=500, n_rows=1000000
        )
        # Should not have expand functionality
        assert "expand-btn" not in html
        assert "toggleMissingColumns" not in html

    def test_edge_case_empty_dataset(self):
        """Test behavior with empty dataset."""
        renderer = create_missing_columns_renderer()

        html = renderer.render_missing_columns_html([], n_cols=0, n_rows=0)
        assert "No column has missing values" in html

    def test_edge_case_all_columns_insignificant(self):
        """Test behavior when all columns have insignificant missing data."""
        miss_list = [
            ("col1", 0.1, 1),
            ("col2", 0.2, 2),
            ("col3", 0.3, 3),
        ]

        renderer = create_missing_columns_renderer(min_threshold_pct=1.0)
        html = renderer.render_missing_columns_html(miss_list, n_cols=3, n_rows=1000)

        assert "No column has missing values" in html
        assert "expand-btn" not in html


if __name__ == "__main__":
    pytest.main([__file__])


class TestTheListSaysWhatItCounted:
    """`total_significant` was computed, stored on the result, and printed
    nowhere -- so a frame with 23 partially-missing columns showed five and
    read as though that were all of them.

    A list that truncates in silence is worse than a shorter list: the reader
    has no way to know they are looking at part of the answer.
    """

    def test_the_remainder_is_stated(self):
        renderer = MissingColumnsRenderer()
        miss_list = [(f"col{i}", 20.0 - i * 0.5, 100) for i in range(23)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=23, n_rows=1000)

        assert "18 more columns" in html, html[-300:]

    def test_it_is_singular_for_one(self):
        renderer = MissingColumnsRenderer()
        miss_list = [(f"col{i}", 20.0 - i, 100) for i in range(6)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=6, n_rows=1000)

        assert "1 more column with missing values" in html

    def test_nothing_is_added_when_the_list_is_complete(self):
        renderer = MissingColumnsRenderer()
        miss_list = [("col1", 5.0, 100), ("col2", 3.0, 60)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=2, n_rows=1000)

        assert "more column" not in html

    def test_the_remainder_names_the_threshold_it_counted_against(self):
        """`+ 18 more columns` is ambiguous without it -- more than what?

        At the shipped default of 0 the phrasing changes: "above 0% missing"
        is a strange way to say "has any missing at all".
        """
        renderer = create_missing_columns_renderer(min_threshold_pct=2.0)
        miss_list = [(f"col{i}", 20.0 - i * 0.5, 100) for i in range(23)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=23, n_rows=1000)

        assert "above 2% missing" in html


class TestTheThresholdIsWrittenDownOnce:
    """It was 0.0 on the class and 0.5 in the factory, so the same class
    filtered differently depending on how it was built, and nothing in the
    report said which had happened."""

    def test_the_remainder_reads_naturally_at_the_default(self):
        renderer = create_missing_columns_renderer()
        miss_list = [(f"col{i}", 20.0 - i * 0.5, 100) for i in range(23)]

        html = renderer.render_missing_columns_html(miss_list, n_cols=23, n_rows=1000)

        assert "18 more columns with missing values" in html
        assert "above 0% missing" not in html

    def test_direct_construction_and_the_factory_agree(self):
        assert (
            MissingColumnsAnalyzer().min_threshold_pct
            == create_missing_columns_renderer().analyzer.min_threshold_pct
        )

    def test_the_default_is_the_class_constant(self):
        assert (
            MissingColumnsAnalyzer().min_threshold_pct
            == MissingColumnsAnalyzer.MIN_THRESHOLD_PCT
        )

    def test_an_explicit_value_is_still_honoured(self):
        """`None` means "use the default"; a number means that number, and the
        two must not collapse into each other."""
        assert MissingColumnsAnalyzer(min_threshold_pct=7.5).min_threshold_pct == 7.5
        assert MissingColumnsAnalyzer(min_threshold_pct=0.0).min_threshold_pct == 0.0

    def test_the_config_default_matches_the_analyzer(self):
        """The render path reads `ProfileConfig`, so a disagreement here means
        the value in the code is not the value in the report."""
        from pysuricata.config import EngineConfig

        assert (
            EngineConfig().missing_columns_threshold_pct
            == MissingColumnsAnalyzer.MIN_THRESHOLD_PCT
        )


class TestTheEmptyStateIsNotAFakeRow:
    def test_it_draws_no_bar(self):
        """The old empty state rendered a zero-width `.missing-fill` -- an
        element drawn to represent nothing."""
        renderer = create_missing_columns_renderer()
        html = renderer.render_missing_columns_html([], n_cols=3, n_rows=1000)

        assert "missing-fill" not in html
        assert "width:0%" not in html

    def test_it_quotes_no_zero_statistic(self):
        renderer = create_missing_columns_renderer()
        html = renderer.render_missing_columns_html([], n_cols=3, n_rows=1000)

        assert "0 (0.0%)" not in html
