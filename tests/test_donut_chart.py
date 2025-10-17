"""Tests for donut chart rendering."""

import pytest
from pysuricata.render.donut_chart import DonutChartRenderer


class TestDonutChartRenderer:
    """Test donut chart rendering with various configurations."""

    def test_empty_donut(self):
        """Test rendering when no columns exist."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(0, 0, 0, 0)
        
        assert "dtype-donut-svg" in result
        assert "No data" in result
        assert "donut-hole" in result

    def test_single_type_100_percent_numeric(self):
        """Test rendering when 100% of columns are numeric (edge case fix)."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(75, 0, 0, 0)
        
        # Should use circle element for 100% case
        assert "dtype-donut-svg" in result
        assert '<circle cx="60" cy="60" r="60"' in result
        assert 'fill="#4ea3f1"' in result  # Numeric color
        assert 'data-percentage="100.0"' in result
        assert 'data-count="75"' in result
        assert 'data-type="Numeric"' in result
        
        # Should NOT have invalid arc path (start=end bug)
        assert 'A 60,60 0 1,1 60.00,0.00 Z' not in result

    def test_single_type_100_percent_categorical(self):
        """Test rendering when 100% of columns are categorical."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(0, 50, 0, 0)
        
        assert "dtype-donut-svg" in result
        assert '<circle cx="60" cy="60" r="60"' in result
        assert 'fill="#8ac926"' in result  # Categorical color
        assert 'data-type="Categorical"' in result

    def test_single_type_100_percent_datetime(self):
        """Test rendering when 100% of columns are datetime."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(0, 0, 30, 0)
        
        assert "dtype-donut-svg" in result
        assert '<circle cx="60" cy="60" r="60"' in result
        assert 'fill="#ffca3a"' in result  # Datetime color
        assert 'data-type="Datetime"' in result

    def test_single_type_100_percent_boolean(self):
        """Test rendering when 100% of columns are boolean."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(0, 0, 0, 20)
        
        assert "dtype-donut-svg" in result
        assert '<circle cx="60" cy="60" r="60"' in result
        assert 'fill="#ff595e"' in result  # Boolean color
        assert 'data-type="Boolean"' in result

    def test_mixed_types_normal_arcs(self):
        """Test rendering with mixed column types (normal arc paths)."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(5, 3, 1, 1)
        
        assert "dtype-donut-svg" in result
        # Should use path elements with arcs, not circles
        assert '<path d="M 60,60 L' in result
        assert 'donut-segment' in result
        
        # Should have multiple segments
        assert result.count('class="donut-segment"') == 4

    def test_two_types_50_50_split(self):
        """Test rendering with 50/50 split between two types."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(10, 10, 0, 0)
        
        assert "dtype-donut-svg" in result
        assert '<path d="M 60,60 L' in result
        
        # Should have numeric and categorical segments
        assert 'data-type="Numeric"' in result
        assert 'data-type="Categorical"' in result
        assert 'data-percentage="50.0"' in result

    def test_skips_zero_count_segments(self):
        """Test that segments with 0 count are skipped."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(5, 0, 3, 0)
        
        # Should only have numeric and datetime segments
        assert 'data-type="Numeric"' in result
        assert 'data-type="Datetime"' in result
        
        # Should not have categorical or boolean
        assert 'data-type="Categorical"' not in result
        assert 'data-type="Boolean"' not in result

    def test_single_column_edge_case(self):
        """Test with just 1 column (100% case)."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(1, 0, 0, 0)
        
        # Should render as full circle
        assert '<circle cx="60" cy="60" r="60"' in result
        assert 'data-count="1"' in result

    def test_very_large_number_of_columns(self):
        """Test with very large number of columns (all one type)."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(1000, 0, 0, 0)
        
        # Should still render as full circle
        assert '<circle cx="60" cy="60" r="60"' in result
        assert 'data-count="1000"' in result
        assert 'data-percentage="100.0"' in result

    def test_inner_segments_rendered(self):
        """Test that inner segments are rendered for visual depth."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(5, 3, 0, 0)
        
        assert "donut-inner-segments" in result
        assert 'class="segment-inner"' in result

    def test_inner_segments_for_100_percent(self):
        """Test that inner segments are rendered for 100% case."""
        renderer = DonutChartRenderer()
        result = renderer.render_dtype_donut(10, 0, 0, 0)
        
        assert "donut-inner-segments" in result
        # Should have inner circle for depth effect
        assert '<circle cx="60" cy="60" r="45"' in result
        assert 'class="segment-inner"' in result

    def test_background_circle_always_present(self):
        """Test that background circle is always rendered."""
        renderer = DonutChartRenderer()
        
        # Test with various configurations
        for args in [(10, 0, 0, 0), (5, 5, 0, 0), (1, 2, 3, 4)]:
            result = renderer.render_dtype_donut(*args)
            assert 'class="donut-background"' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

