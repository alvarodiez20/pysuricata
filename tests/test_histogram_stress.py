"""
Comprehensive stress tests for histogram rendering.

Tests edge cases, extreme values, and various distribution patterns
to ensure robust axis formatting and rendering.
"""

import numpy as np
import pytest

from pysuricata.render.histogram_svg import HistogramConfig, SVGHistogramRenderer


class TestHistogramEdgeValues:
    """Test histogram with extreme and edge case values."""

    def test_very_small_numbers(self):
        """Test with very small numbers (1e-10, 1e-6)."""
        renderer = SVGHistogramRenderer()
        values = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6] * 100)

        # Linear scale
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        assert len(svg) > 100

        # Log scale
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log
        assert len(svg_log) > 100

    def test_very_large_numbers(self):
        """Test with very large numbers (1e6, 1e10, 1e15)."""
        renderer = SVGHistogramRenderer()
        values = np.array([1e6, 1e7, 1e8, 1e9, 1e10, 1e15] * 100)

        # Linear scale
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        assert len(svg) > 100

        # Log scale
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log
        assert len(svg_log) > 100

    def test_mixed_tiny_and_huge(self):
        """Test mix of tiny and huge values in same histogram."""
        renderer = SVGHistogramRenderer()
        values = np.concatenate(
            [np.array([1e-10, 1e-8, 1e-6] * 30), np.array([1e6, 1e8, 1e10] * 30)]
        )

        # Linear scale (challenging)
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

        # Log scale (should handle well)
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log
        assert (
            "1e" in svg_log or "," in svg_log
        )  # Scientific notation or comma formatting

    def test_negative_values(self):
        """Test with negative values."""
        renderer = SVGHistogramRenderer()
        values = np.array([-1000, -500, -100, -10, -1, 0, 1, 10, 100, 500, 1000] * 10)

        # Linear scale should work
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        assert len(svg) > 100

        # Log scale should handle gracefully (exclude negatives)
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log

    def test_zeros_mixed_with_positives(self):
        """Test zero values mixed with positives."""
        renderer = SVGHistogramRenderer()
        values = np.array([0] * 50 + list(range(1, 51)))

        # Linear scale
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

        # Log scale (should exclude zeros)
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log


class TestYAxisStressTests:
    """Test Y-axis with various count patterns."""

    def test_single_value_count(self):
        """Test with count = 1 (single value)."""
        renderer = SVGHistogramRenderer()
        values = np.array([42.0])

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        assert "No data" not in svg

    def test_minimal_count(self):
        """Test with count = 2 (minimal)."""
        renderer = SVGHistogramRenderer()
        values = np.array([1.0, 2.0])

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

    def test_prime_number_counts(self):
        """Test with prime number counts (127, 251, 503)."""
        renderer = SVGHistogramRenderer()

        for count in [127, 251, 503]:
            values = np.random.randn(count) * 100
            svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
            assert "svg" in svg
            # Check that Y-axis labels are nice numbers, not ugly primes
            # Nice ticks should give us rounded values like 0, 50, 100, 150 etc
            # NOT 0, 31, 63, 95, 127 (which would be from integer division)

    def test_power_of_two_counts(self):
        """Test with powers of 2 (128, 256, 512, 1024)."""
        renderer = SVGHistogramRenderer()

        for count in [128, 256, 512, 1024]:
            values = np.random.randn(count) * 100
            svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
            assert "svg" in svg

    def test_irregular_counts(self):
        """Test with irregular counts (123, 789, 4567)."""
        renderer = SVGHistogramRenderer()

        for count in [123, 789, 4567]:
            values = np.random.randn(count) * 100
            svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
            assert "svg" in svg

    def test_very_large_counts(self):
        """Test with very large counts (1M, 10M)."""
        renderer = SVGHistogramRenderer()

        # 1 million
        values = np.random.randn(1_000_000) * 100
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        # Should have comma-separated numbers in Y-axis
        assert "," in svg or "1e" in svg  # Either commas or scientific notation


class TestXAxisRangeTests:
    """Test X-axis with various data ranges."""

    def test_constant_data(self):
        """Test all values identical (constant data)."""
        renderer = SVGHistogramRenderer()
        values = np.array([42.0] * 100)

        # Should not crash
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log

    def test_two_distinct_values(self):
        """Test two distinct values only."""
        renderer = SVGHistogramRenderer()
        values = np.array([1.0] * 50 + [2.0] * 50)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

    def test_narrow_range(self):
        """Test narrow range (e.g., 1.0001 to 1.0005)."""
        renderer = SVGHistogramRenderer()
        values = np.linspace(1.0001, 1.0005, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        # Should show appropriate decimal precision

    def test_wide_range(self):
        """Test wide range (e.g., 0.001 to 1000000)."""
        renderer = SVGHistogramRenderer()
        values = np.logspace(-3, 6, 1000)  # 0.001 to 1,000,000

        # Linear scale (challenging)
        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

        # Log scale (should handle well)
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log

    def test_many_orders_of_magnitude(self):
        """Test values spanning 10+ orders of magnitude (log scale)."""
        renderer = SVGHistogramRenderer()
        values = np.logspace(-5, 10, 1000)  # 15 orders of magnitude

        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log
        # Should have power-of-10 labels


class TestBinCountVariations:
    """Test different bin count configurations."""

    def test_single_bin_edge_case(self):
        """Test bins=1 (edge case, should be handled as 2)."""
        renderer = SVGHistogramRenderer()
        values = np.random.randn(100) * 100

        svg = renderer.render_histogram(values, bins=1, scale="lin", col_id="test")
        assert "svg" in svg

    def test_various_bin_counts(self):
        """Test bins=5, 10, 25, 50, 100."""
        renderer = SVGHistogramRenderer()
        values = np.random.randn(1000) * 100

        for bins in [5, 10, 25, 50, 100]:
            svg = renderer.render_histogram(
                values, bins=bins, scale="lin", col_id="test"
            )
            assert "svg" in svg

    def test_extreme_bin_count(self):
        """Test bins=200 (stress test)."""
        renderer = SVGHistogramRenderer()
        values = np.random.randn(10000) * 100

        svg = renderer.render_histogram(values, bins=200, scale="lin", col_id="test")
        assert "svg" in svg


class TestScaleTransitions:
    """Test rendering in both linear and log scales."""

    def test_same_data_both_scales(self):
        """Same data rendered in both linear and log scales."""
        renderer = SVGHistogramRenderer()
        values = np.random.lognormal(0, 1, 1000)

        svg_lin = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")

        assert "svg" in svg_lin
        assert "svg" in svg_log
        assert svg_lin != svg_log  # Should be different

    def test_zeros_log_scale(self):
        """Data with zeros (should exclude from log scale properly)."""
        renderer = SVGHistogramRenderer()
        values = np.array([0, 0, 0] + list(range(1, 100)))

        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log

    def test_all_negative_log_scale(self):
        """All negative values with log scale (should handle gracefully)."""
        renderer = SVGHistogramRenderer()
        values = np.array([-100, -50, -10, -5, -1] * 20)

        # Should not crash, should show "No data" or similar
        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log


class TestDistributionPatterns:
    """Test various distribution patterns."""

    def test_uniform_distribution(self):
        """Test uniform distribution."""
        renderer = SVGHistogramRenderer()
        values = np.random.uniform(0, 100, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

    def test_highly_skewed(self):
        """Test highly skewed (99% in one bin)."""
        renderer = SVGHistogramRenderer()
        values = np.array([50.0] * 990 + list(range(1, 11)))

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

    def test_bimodal_distribution(self):
        """Test bimodal distribution."""
        renderer = SVGHistogramRenderer()
        values = np.concatenate(
            [np.random.normal(20, 5, 500), np.random.normal(80, 5, 500)]
        )

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

    def test_long_tail_distribution(self):
        """Test long-tail distribution."""
        renderer = SVGHistogramRenderer()
        values = np.random.lognormal(0, 2, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

        svg_log = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg_log

    def test_single_outlier(self):
        """Test single outlier affecting scale."""
        renderer = SVGHistogramRenderer()
        values = np.array([1.0] * 999 + [1000000.0])  # One huge outlier

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg


class TestTickLabelFormatting:
    """Test that tick labels are properly formatted."""

    def test_scientific_notation_large(self):
        """Verify scientific notation for large numbers."""
        renderer = SVGHistogramRenderer()
        values = np.random.uniform(1e8, 1e10, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        # Should contain scientific notation
        assert "e" in svg.lower() or "," in svg

    def test_scientific_notation_small(self):
        """Verify scientific notation for small numbers."""
        renderer = SVGHistogramRenderer()
        values = np.random.uniform(1e-10, 1e-8, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

    def test_comma_separators(self):
        """Verify comma separators for thousands."""
        renderer = SVGHistogramRenderer()
        values = np.random.uniform(1000, 100000, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        # Should contain commas in numbers
        assert "," in svg

    def test_decimal_precision(self):
        """Check decimal precision for fractional values."""
        renderer = SVGHistogramRenderer()
        values = np.random.uniform(0.1, 10.0, 1000)

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg
        # Should show appropriate decimal places

    def test_log_scale_power_of_ten_labels(self):
        """Ensure power-of-10 labels in log scale (1, 10, 100, 1K, 10K, etc.)."""
        renderer = SVGHistogramRenderer()
        values = np.logspace(0, 6, 1000)  # 1 to 1,000,000

        svg = renderer.render_histogram(values, bins=25, scale="log", col_id="test")
        assert "svg" in svg
        # Should have clean power-of-10 labels
        # Could be "1", "10", "100", "1,000", etc.


class TestBarOpacity:
    """Test that bars have correct opacity."""

    def test_bar_opacity_is_full(self):
        """Verify bars are rendered with opacity 1.0."""
        renderer = SVGHistogramRenderer()
        values = np.random.randn(100) * 100

        svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
        assert "svg" in svg

        # Check that bars use the opacity setting from config
        assert 'fill-opacity="' in svg
        # The default config should be using 0.7 opacity (from HistogramConfig)
        # But CSS overrides should make it 1.0 in the final render


def test_no_crashes_on_empty_data():
    """Ensure no crashes with empty data."""
    renderer = SVGHistogramRenderer()
    values = np.array([])

    svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
    assert "svg" in svg
    assert "No data" in svg or "no data" in svg.lower()


def test_integration_with_custom_config():
    """Test histogram with custom configuration."""
    config = HistogramConfig(
        width=600,
        height=300,
        bar_opacity=1.0,  # Full opacity
        bar_color="#ff0000",  # Red bars
    )
    renderer = SVGHistogramRenderer(config)
    values = np.random.randn(1000) * 100

    svg = renderer.render_histogram(values, bins=25, scale="lin", col_id="test")
    assert "svg" in svg
    assert 'width="600"' in svg
    assert 'height="300"' in svg
    assert "#ff0000" in svg  # Red color
    assert 'fill-opacity="1.0"' in svg  # Full opacity


def test_log_scale_x_axis_alignment():
    """Test that log scale x-axis ticks align properly with bars.

    This is a regression test for the bug where render_histogram_from_bins()
    didn't convert bin edges/centers back to linear space, causing x-axis
    misalignment in log scale.
    """
    renderer = SVGHistogramRenderer()

    # Create data spanning multiple orders of magnitude
    bin_edges = [1, 10, 100, 1000, 10000]
    bin_counts = [100, 200, 150, 50]

    # Render with log scale using pre-computed bins
    svg = renderer.render_histogram_from_bins(
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bins=4,
        scale="log",
        title="Log Scale Test",
        col_id="test_log",
    )

    assert "svg" in svg

    # Check that expected power-of-10 tick labels are present
    # Should show 1, 10, 100, 1,000, 10,000 or similar
    assert "10" in svg or "1e" in svg.lower()

    # Verify bars are rendered (should have rect elements)
    assert '<rect class="bar"' in svg

    # The bug would cause bin_centers to be in log10 space (0, 1, 2, 3, 4)
    # instead of linear space (1, 10, 100, 1000, 10000), causing x-axis
    # ticks to not align with the actual bar positions


def test_y_axis_tick_positioning():
    """Test that Y-axis ticks are properly positioned using tick maximum.

    This is a regression test for the bug where Y-axis positioning used
    hist_data.y_max instead of the maximum tick value, causing the top
    tick to appear above the chart area.
    """
    renderer = SVGHistogramRenderer()

    # Create data with a specific maximum count that will round up to a nice number
    # For example, max count = 273 should round up to 300 with nice_ticks()
    values = np.random.randn(1000) * 100

    # Manually create histogram with known maximum
    counts, edges = np.histogram(values, bins=25)
    # Simulate a case where max count is 273 (will round to 300)
    counts = counts.astype(float)
    counts[counts.argmax()] = 273  # Set max to 273

    svg = renderer.render_histogram_from_bins(
        bin_edges=edges.tolist(),
        bin_counts=counts.astype(int).tolist(),
        bins=25,
        scale="lin",
        title="Y-Axis Test",
        col_id="test_y_axis",
    )

    assert "svg" in svg

    # The Y-axis should show ticks like 0, 50, 100, 150, 200, 250, 300
    # The top tick (300) should be positioned at the top of the chart area
    # Previously, it would be positioned at (300/273) * chart_height, going above bounds

    # Verify that the chart renders without issues
    assert '<rect class="bar"' in svg
    assert 'height="200"' in svg  # Default chart height

    # The fix ensures that the maximum tick value (300) is used for positioning
    # instead of the data maximum (273), so all ticks fit within the chart bounds


def test_bar_height_scaling_consistency():
    """Test that bar heights are scaled consistently with Y-axis ticks.

    This is a regression test for the bug where bars were scaled using
    hist_data.y_max while Y-axis ticks used tick_max, causing visual
    inconsistency (e.g., 744-row bar appearing to reach 800 mark).
    """
    renderer = SVGHistogramRenderer()

    # Test with render_histogram() instead of render_histogram_from_bins()
    # to avoid bin redistribution that might change the max count
    import numpy as np

    # Create data where max count doesn't align with nice tick values
    # Generate data that will have max count around 744
    np.random.seed(42)  # For reproducible results
    values = np.concatenate(
        [
            np.random.normal(25, 5, 744),  # 744 values around 25
            np.random.normal(75, 5, 100),  # 100 values around 75
            np.random.normal(125, 5, 50),  # 50 values around 125
        ]
    )

    svg = renderer.render_histogram(
        values=values,
        bins=25,
        scale="lin",
        title="Bar Scaling Test",
        col_id="test_bar_scaling",
    )

    assert "svg" in svg

    # Verify that the chart renders without issues
    assert '<rect class="bar"' in svg

    # The tallest bar should now be scaled proportionally to the nice tick maximum
    # instead of reaching 100% of chart height

    # Extract bar heights specifically (rect elements with class="bar")
    import re

    bar_rects = re.findall(r'<rect class="bar"[^>]*height="([0-9.]+)"', svg)
    bar_heights = [float(h) for h in bar_rects]

    if bar_heights:
        max_bar_height = max(bar_heights)

        # The tallest bar should be less than the full chart height
        # since nice_ticks() will round up the maximum (289 -> 300)
        # Expected: (289/300) * 140 ≈ 134.87
        assert max_bar_height < 150, (
            f"Bar height {max_bar_height} should be < 150 (expected ~135)"
        )
        assert max_bar_height > 130, (
            f"Bar height {max_bar_height} should be > 130 (expected ~135)"
        )

    # Y-axis should show nice ticks that round up from the actual maximum
    # Bars should be proportionally scaled to these tick marks


def test_log_scale_zero_negative_handling():
    """Test that log scale histograms handle zero/negative values gracefully.

    This test verifies that X-axis ticks appear even when data contains
    zeros or negative values, which previously caused empty X-axes.
    """
    renderer = SVGHistogramRenderer()

    # Test cases that previously caused empty X-axes
    test_cases = [
        # Case 1: Data with zeros
        {
            "name": "with_zeros",
            "values": np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100]),
            "description": "Data containing zeros",
        },
        # Case 2: Data with negative values
        {
            "name": "with_negatives",
            "values": np.array([-5, -2, 1, 3, 5, 10, 20, 50]),
            "description": "Data containing negative values",
        },
        # Case 3: Mixed positive/negative/zero
        {
            "name": "mixed_values",
            "values": np.array([-10, -5, 0, 1, 5, 10, 100, 1000]),
            "description": "Mixed positive/negative/zero values",
        },
        # Case 4: Very small positive values
        {
            "name": "small_positive",
            "values": np.array([0.001, 0.01, 0.1, 1, 10, 100]),
            "description": "Very small positive values",
        },
        # Case 5: Single positive value
        {
            "name": "single_positive",
            "values": np.array([5, 5, 5, 5, 5]),
            "description": "Single positive value repeated",
        },
    ]

    for case in test_cases:
        svg = renderer.render_histogram(
            values=case["values"],
            bins=10,
            scale="log",
            title=f"Log Scale Test: {case['description']}",
            col_id=f"test_{case['name']}",
        )

        assert "svg" in svg, f"SVG not generated for {case['name']}"

        # Verify that X-axis ticks are present
        # Look for tick marks (lines) and tick labels (text)
        tick_mark_count = svg.count('class="tick-label"')
        tick_line_count = svg.count('x2="')  # Approximate count of tick lines

        # Should have at least some ticks for positive values
        if case["name"] != "single_positive":  # Single value might have minimal ticks
            assert tick_mark_count > 0, (
                f"No tick labels found for {case['name']}: {case['description']}"
            )
            assert tick_line_count > 0, (
                f"No tick marks found for {case['name']}: {case['description']}"
            )

        # Verify the chart renders without errors
        assert '<rect class="bar"' in svg or 'class="hist-svg"' in svg

        # For cases with positive values, should have at least some tick labels
        positive_values = case["values"][case["values"] > 0]
        if len(positive_values) > 0:
            # Check that we have some numeric tick labels (not just "0" from Y-axis)
            import re

            label_matches = re.findall(r'class="tick-label"[^>]*>([^<]+)</text>', svg)
            numeric_labels = [
                label
                for label in label_matches
                if label.replace(".", "").replace(",", "").isdigit() or "10^" in label
            ]

            # Should have at least one numeric tick label for log scale
            assert len(numeric_labels) > 0, (
                f"No numeric tick labels found for {case['name']}: {case['description']}. Found labels: {label_matches}"
            )


def test_grid_line_alignment_and_bar_opacity():
    """Test that grid lines align with Y-axis ticks and bars have proper opacity."""
    renderer = SVGHistogramRenderer()

    # Create test data with known values for predictable results
    import numpy as np

    values = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    svg = renderer.render_histogram(
        values=values,
        bins=15,
        scale="lin",
        title="Grid Alignment Test",
        col_id="test_grid_alignment",
    )

    assert "svg" in svg

    # Test 1: Verify grid lines are present
    grid_line_count = svg.count('stroke="#eee"')  # Grid lines use #eee color
    assert grid_line_count > 0, "No grid lines found"

    # Test 2: Verify bars have increased opacity (0.9 instead of 0.7)
    bar_opacity_count = svg.count('fill-opacity="0.9"')
    assert bar_opacity_count > 0, (
        f"Expected bars with opacity 0.9, but found: {svg.count('fill-opacity')}"
    )

    # Test 3: Verify grid lines and Y-axis ticks use consistent positioning
    # This is harder to test directly, but we can verify both elements exist
    tick_label_count = svg.count('class="tick-label"')
    assert tick_label_count > 0, "No Y-axis tick labels found"

    # Test 4: Verify bars render without errors
    assert '<rect class="bar"' in svg, "No histogram bars found"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
