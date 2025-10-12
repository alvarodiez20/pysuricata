"""Tests for missing values section renderer.

This module tests the MissingValuesSectionRenderer to ensure:
1. Proper rendering of bar chart for variables with missing values
2. Proper rendering of chunk distributions
3. Correct HTML structure and CSS classes
4. Graceful handling of edge cases
"""

from __future__ import annotations

import pytest

from pysuricata.render.missing_section import MissingValuesSectionRenderer


class MockAccumulator:
    """Mock accumulator for testing."""

    def __init__(
        self,
        name: str,
        count: int,
        missing: int,
        chunk_metadata: list[tuple[int, int, int]] | None = None,
        per_column_chunk_metadata: dict[str, list[tuple[int, int, int]]] | None = None,
    ):
        self.name = name
        self.count = count
        self.missing = missing
        self.chunk_metadata = chunk_metadata
        self.per_column_chunk_metadata = per_column_chunk_metadata


class TestRenderSection:
    """Test the render_section method."""

    def test_complete_section_structure(self):
        """Test that the complete section has proper structure with tabs."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "col1": ("numeric", MockAccumulator("col1", 90, 10)),
            "col2": ("categorical", MockAccumulator("col2", 80, 20)),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1], "col2": kinds_map["col2"][1]},
            n_rows=100,
            n_cols=2,
            total_missing_cells=30,
            per_column_chunk_metadata=None,
        )

        # Verify container structure with tabs
        assert "missing-values-container" in html
        assert "missing-section-tabs" in html
        assert 'data-tab="data-completeness"' in html
        assert 'data-tab="missing-distribution"' in html

        # Verify both tab buttons exist
        assert "Data Completeness" in html
        assert "Missing Values Distribution" in html

        # Verify bar chart content in first tab
        assert "missing-bar-chart" in html

    def test_only_variables_with_missing_values(self):
        """Test that only variables with missing values are shown."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "col1": ("numeric", MockAccumulator("col1", 100, 0)),  # No missing
            "col2": ("categorical", MockAccumulator("col2", 90, 10)),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1], "col2": kinds_map["col2"][1]},
            n_rows=100,
            n_cols=2,
            total_missing_cells=10,
            per_column_chunk_metadata=None,
        )

        # Should only show col2
        assert "col2" in html
        assert "col1" not in html or html.count("col1") < 2  # May appear in stats

    def test_no_missing_values(self):
        """Test rendering when no variables have missing values."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "col1": ("numeric", MockAccumulator("col1", 100, 0)),
            "col2": ("categorical", MockAccumulator("col2", 100, 0)),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1], "col2": kinds_map["col2"][1]},
            n_rows=100,
            n_cols=2,
            total_missing_cells=0,
            per_column_chunk_metadata=None,
        )

        # Should show empty state message
        assert "No missing values found" in html

    def test_mixed_variable_types(self):
        """Test rendering with mixed variable types."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "num_col": ("numeric", MockAccumulator("num_col", 90, 10)),
            "cat_col": ("categorical", MockAccumulator("cat_col", 85, 15)),
            "dt_col": ("datetime", MockAccumulator("dt_col", 95, 5)),
            "bool_col": ("boolean", MockAccumulator("bool_col", 80, 20)),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={k: v[1] for k, v in kinds_map.items()},
            n_rows=100,
            n_cols=4,
            total_missing_cells=50,
            per_column_chunk_metadata=None,
        )

        # All variables with missing values should appear
        assert "num_col" in html
        assert "cat_col" in html
        assert "dt_col" in html
        assert "bool_col" in html

    def test_bar_chart_content(self):
        """Test that bar chart shows correct enhanced content."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "col1": ("numeric", MockAccumulator("col1", 90, 10)),
            "col2": ("categorical", MockAccumulator("col2", 70, 30)),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1], "col2": kinds_map["col2"][1]},
            n_rows=100,
            n_cols=2,
            total_missing_cells=40,
            per_column_chunk_metadata=None,
        )

        # Verify enhanced compact card elements
        assert "missing-var-card-compact" in html
        assert "var-name" in html
        assert "var-stats" in html
        assert "completeness-bar-compact" in html
        assert "present-stat" in html
        assert "missing-stat" in html

        # Verify summary
        assert "2 variables with missing data" in html

        # Verify checkmark and x symbols
        assert "✓" in html
        assert "✗" in html

    def test_severity_class_assignment(self):
        """Test that severity classes are correctly assigned to bars."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "low_col": ("numeric", MockAccumulator("low_col", 98, 2)),  # 2% - low
            "med_col": (
                "categorical",
                MockAccumulator("med_col", 85, 15),
            ),  # 15% - medium
            "high_col": ("boolean", MockAccumulator("high_col", 50, 50)),  # 50% - high
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={k: v[1] for k, v in kinds_map.items()},
            n_rows=100,
            n_cols=3,
            total_missing_cells=67,
            per_column_chunk_metadata=None,
        )

        # Verify severity classes are applied to the new compact bars
        assert "bar-fill missing low" in html
        assert "bar-fill missing medium" in html
        assert "bar-fill missing high" in html

        # Verify severity classes in stats text
        assert "missing-stat low" in html
        assert "missing-stat medium" in html
        assert "missing-stat high" in html

    def test_distribution_tab_with_chunk_metadata(self):
        """Test that distribution tab shows chunk visualizations when metadata exists."""
        renderer = MissingValuesSectionRenderer()

        chunk_metadata = [(0, 99, 10), (100, 199, 20), (200, 299, 5)]
        per_column_metadata = {"col1": chunk_metadata}

        kinds_map = {
            "col1": (
                "numeric",
                MockAccumulator("col1", 265, 35, chunk_metadata=chunk_metadata),
            ),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1]},
            n_rows=300,
            n_cols=1,
            total_missing_cells=35,
            per_column_chunk_metadata=per_column_metadata,
        )

        # Verify distribution tab exists
        assert 'data-tab="missing-distribution"' in html

        # Verify ultra-compact chunk visualization elements
        assert "chunk-var-row" in html
        assert "chunk-spectrum-compact" in html
        assert "chunk-segment" in html
        assert "chunk-legend-shared" in html  # Shared legend, not per-variable
        assert "Peak:" in html

    def test_distribution_tab_without_chunk_metadata(self):
        """Test that distribution tab shows message when no chunk metadata."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "col1": ("numeric", MockAccumulator("col1", 90, 10)),  # No chunk metadata
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1]},
            n_rows=100,
            n_cols=1,
            total_missing_cells=10,
            per_column_chunk_metadata=None,
        )

        # Verify distribution tab shows no metadata message
        assert "No chunk metadata available" in html

    def test_dual_color_bars(self):
        """Test that dual-color bars show both present and missing data."""
        renderer = MissingValuesSectionRenderer()

        kinds_map = {
            "col1": ("numeric", MockAccumulator("col1", 80, 20)),  # 20% missing
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1]},
            n_rows=100,
            n_cols=1,
            total_missing_cells=20,
            per_column_chunk_metadata=None,
        )

        # Verify dual-color bar sections
        assert "bar-fill present" in html
        assert "bar-fill missing" in html

        # Verify percentages in stats
        assert "✓ 80.0%" in html  # Present
        assert "✗ 20.0%" in html  # Missing

    def test_compact_summary_text(self):
        """Test that summary text uses correct singular/plural."""
        renderer = MissingValuesSectionRenderer()

        # Test singular
        kinds_map_single = {
            "col1": ("numeric", MockAccumulator("col1", 90, 10)),
        }

        html_single = renderer.render_section(
            kinds_map=kinds_map_single,
            accs={"col1": kinds_map_single["col1"][1]},
            n_rows=100,
            n_cols=1,
            total_missing_cells=10,
            per_column_chunk_metadata=None,
        )

        assert "1 variable with missing data" in html_single

        # Test plural
        kinds_map_plural = {
            "col1": ("numeric", MockAccumulator("col1", 90, 10)),
            "col2": ("categorical", MockAccumulator("col2", 80, 20)),
        }

        html_plural = renderer.render_section(
            kinds_map=kinds_map_plural,
            accs={k: v[1] for k, v in kinds_map_plural.items()},
            n_rows=100,
            n_cols=2,
            total_missing_cells=30,
            per_column_chunk_metadata=None,
        )

        assert "2 variables with missing data" in html_plural

    def test_scalability_with_many_variables(self):
        """Test that the compact design handles many variables efficiently."""
        renderer = MissingValuesSectionRenderer()

        # Create 100 variables with missing data
        kinds_map = {}
        for i in range(100):
            col_name = f"column_{i:03d}"
            missing_pct = (i % 30) + 1  # Varying missing percentages 1-30%
            total = 1000
            missing_count = int(total * missing_pct / 100)
            present_count = total - missing_count
            kinds_map[col_name] = (
                "numeric",
                MockAccumulator(col_name, present_count, missing_count),
            )

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={k: v[1] for k, v in kinds_map.items()},
            n_rows=1000,
            n_cols=100,
            total_missing_cells=sum(
                int(1000 * ((i % 30) + 1) / 100) for i in range(100)
            ),
            per_column_chunk_metadata=None,
        )

        # Verify summary shows correct count
        assert "100 variables with missing data" in html

        # Verify all 100 cards are generated
        assert html.count("missing-var-card-compact") == 100

        # Verify compact structure elements exist
        assert "var-cards-grid" in html
        assert "completeness-bar-compact" in html

        # Estimate HTML size - should be reasonable
        html_size = len(html)
        # Each card is ~350-400 characters, 100 cards = ~35-40KB
        # Should be well under 100KB even with 100 variables
        assert html_size < 100_000, f"HTML too large: {html_size} bytes"

    def test_distribution_chunk_tooltips(self):
        """Test that chunk segments have correct tooltip format."""
        renderer = MissingValuesSectionRenderer()

        chunk_metadata = [(0, 99, 10), (100, 199, 50)]
        per_column_metadata = {"test_col": chunk_metadata}

        kinds_map = {
            "test_col": (
                "numeric",
                MockAccumulator("test_col", 140, 60, chunk_metadata=chunk_metadata),
            ),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"test_col": kinds_map["test_col"][1]},
            n_rows=200,
            n_cols=1,
            total_missing_cells=60,
            per_column_chunk_metadata=per_column_metadata,
        )

        # Verify tooltip format
        assert 'title="Rows 0-99: 10 missing (10.0%)"' in html
        assert 'title="Rows 100-199: 50 missing (50.0%)"' in html

    def test_distribution_per_column_chunk_metadata(self):
        """Test that per_column_chunk_metadata is preferred over chunk_metadata."""
        renderer = MissingValuesSectionRenderer()

        per_column_metadata = {"col1": [(0, 49, 5), (50, 99, 10)]}

        kinds_map = {
            "col1": (
                "numeric",
                MockAccumulator(
                    "col1",
                    85,
                    15,
                    chunk_metadata=[(0, 99, 15)],  # Old format (1 chunk)
                    per_column_chunk_metadata=per_column_metadata,  # New format (2 chunks)
                ),
            ),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"col1": kinds_map["col1"][1]},
            n_rows=100,
            n_cols=1,
            total_missing_cells=15,
            per_column_chunk_metadata=per_column_metadata,
        )

        # Should use per_column_chunk_metadata (2 chunks) not chunk_metadata (1 chunk)
        # Verify we have 2 chunk segments (not "X chunks analyzed" since we removed that text)
        assert html.count("chunk-segment") >= 2  # At least 2 segments from 2 chunks

    def test_distribution_peak_calculation(self):
        """Test that peak missing percentage is correctly calculated."""
        renderer = MissingValuesSectionRenderer()

        # Chunks with varying missing percentages: 5%, 30%, 10%
        chunk_metadata = [(0, 99, 5), (100, 199, 30), (200, 299, 10)]
        per_column_metadata = {"test_col": chunk_metadata}

        kinds_map = {
            "test_col": (
                "numeric",
                MockAccumulator("test_col", 255, 45, chunk_metadata=chunk_metadata),
            ),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={"test_col": kinds_map["test_col"][1]},
            n_rows=300,
            n_cols=1,
            total_missing_cells=45,
            per_column_chunk_metadata=per_column_metadata,
        )

        # Peak should be 30%
        assert "Peak: 30.0%" in html

    def test_distribution_multiple_variables(self):
        """Test distribution tab with multiple variables sorted by severity."""
        renderer = MissingValuesSectionRenderer()

        chunk_meta_low = [(0, 99, 2)]
        chunk_meta_high = [(0, 99, 50)]
        chunk_meta_med = [(0, 99, 15)]

        per_column_metadata = {
            "low_var": chunk_meta_low,
            "high_var": chunk_meta_high,
            "med_var": chunk_meta_med,
        }

        kinds_map = {
            "low_var": (
                "numeric",
                MockAccumulator("low_var", 98, 2, chunk_metadata=chunk_meta_low),
            ),
            "high_var": (
                "categorical",
                MockAccumulator("high_var", 50, 50, chunk_metadata=chunk_meta_high),
            ),
            "med_var": (
                "boolean",
                MockAccumulator("med_var", 85, 15, chunk_metadata=chunk_meta_med),
            ),
        }

        html = renderer.render_section(
            kinds_map=kinds_map,
            accs={k: v[1] for k, v in kinds_map.items()},
            n_rows=100,
            n_cols=3,
            total_missing_cells=67,
            per_column_chunk_metadata=per_column_metadata,
        )

        # Should have 3 chunk rows (ultra-compact)
        assert html.count("chunk-var-row") == 3

        # Should have only ONE shared legend, not per-variable
        assert html.count("chunk-legend-shared") == 1

        # Variables should be sorted by missing percentage (high, med, low)
        high_var_pos = html.index("high_var")
        med_var_pos = html.index("med_var")
        low_var_pos = html.index("low_var")

        assert high_var_pos < med_var_pos < low_var_pos, (
            "Variables should be sorted by missing percentage"
        )


class TestGetSeverityClass:
    """Test the _get_severity_class method."""

    def test_low_severity(self):
        """Test low severity classification (0-5%)."""
        renderer = MissingValuesSectionRenderer()
        assert renderer._get_severity_class(0.0) == "low"
        assert renderer._get_severity_class(2.5) == "low"
        assert renderer._get_severity_class(5.0) == "low"

    def test_medium_severity(self):
        """Test medium severity classification (5-20%)."""
        renderer = MissingValuesSectionRenderer()
        assert renderer._get_severity_class(5.1) == "medium"
        assert renderer._get_severity_class(12.5) == "medium"
        assert renderer._get_severity_class(20.0) == "medium"

    def test_high_severity(self):
        """Test high severity classification (20%+)."""
        renderer = MissingValuesSectionRenderer()
        assert renderer._get_severity_class(20.1) == "high"
        assert renderer._get_severity_class(50.0) == "high"
        assert renderer._get_severity_class(100.0) == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
