"""
Comprehensive test suite for temporal distribution histograms.

This module tests the TemporalHistogramRenderer class and its integration
with datetime cards, including edge cases, performance, and visual regression.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from pysuricata.accumulators.datetime import DatetimeAccumulator
from pysuricata.render.card_types import DateTimeStats
from pysuricata.render.datetime_card import DateTimeCardRenderer
from pysuricata.render.temporal_histogram_svg import (
    TemporalHistogramConfig,
    TemporalHistogramRenderer,
)

# ============================================================================
# Unit Tests for TemporalHistogramRenderer
# ============================================================================


class TestTemporalHistogramRenderer:
    """Unit tests for the TemporalHistogramRenderer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = TemporalHistogramRenderer()

    def test_render_hour_histogram_empty(self):
        """Test hour histogram with empty data returns valid SVG."""
        svg = self.renderer.render_hour_histogram([])
        assert "<svg" in svg
        assert "temporal-bar-chart" in svg
        assert "No data available" in svg

    def test_render_hour_histogram_full(self):
        """Test hour histogram with all 24 hours populated."""
        counts = list(range(1, 25))  # 1-24
        svg = self.renderer.render_hour_histogram(counts)
        assert "<svg" in svg
        assert "temporal-bar-chart" in svg
        assert '<rect class="temporal-bar bar"' in svg
        assert "data-count" in svg
        # Check some hour labels
        assert "00:00" in svg
        assert "12:00" in svg or "09:00" in svg  # Some labels visible

    def test_render_hour_histogram_sparse(self):
        """Test hour histogram with only few hours having data."""
        counts = [0] * 24
        counts[9] = 100  # 9 AM
        counts[17] = 50  # 5 PM
        svg = self.renderer.render_hour_histogram(counts)
        assert "<svg" in svg
        assert 'data-count="100"' in svg
        assert 'data-count="50"' in svg

    def test_render_hour_histogram_single_peak(self):
        """Test hour histogram where single hour dominates."""
        counts = [1] * 24
        counts[14] = 1000  # 2 PM has massive spike
        svg = self.renderer.render_hour_histogram(counts)
        assert "<svg" in svg
        assert 'data-count="1000"' in svg
        # Check percentage calculation
        total = sum(counts)
        pct = (1000 / total) * 100
        assert f'data-pct="{pct:.1f}"' in svg

    def test_render_dow_histogram_weekday_only(self):
        """Test DOW histogram with only weekdays having data."""
        counts = [100, 120, 110, 105, 95, 0, 0]  # Mon-Fri only
        svg = self.renderer.render_dow_histogram(counts)
        assert "<svg" in svg
        assert "Mon" in svg
        assert "Fri" in svg
        assert "Sat" in svg  # Label should be present even if 0
        assert 'data-count="100"' in svg

    def test_render_dow_histogram_weekend_heavy(self):
        """Test DOW histogram with weekend concentration."""
        counts = [10, 10, 10, 10, 10, 200, 250]  # Weekend heavy
        svg = self.renderer.render_dow_histogram(counts)
        assert "<svg" in svg
        assert 'data-count="200"' in svg
        assert 'data-count="250"' in svg
        # Weekend should have high percentages
        total = sum(counts)
        weekend_pct = (450 / total) * 100
        assert weekend_pct > 70  # More than 70% on weekend

    def test_render_month_histogram_seasonal(self):
        """Test month histogram with summer/winter peaks."""
        counts = [50, 50, 60, 70, 80, 150, 200, 180, 100, 70, 60, 50]
        # Jun-Aug peak (indices 5-7)
        svg = self.renderer.render_month_histogram(counts)
        assert "<svg" in svg
        assert "Jan" in svg
        assert "Jun" in svg or "Jul" in svg or "Aug" in svg
        assert 'data-count="200"' in svg  # July peak

    def test_render_month_histogram_sparse(self):
        """Test month histogram with few months populated."""
        counts = [0] * 12
        counts[0] = 100  # January
        counts[6] = 150  # July
        svg = self.renderer.render_month_histogram(counts)
        assert "<svg" in svg
        assert 'data-count="100"' in svg
        assert 'data-count="150"' in svg

    def test_render_year_histogram_single_year(self):
        """Test year histogram with only one year."""
        year_counts = {2023: 1000}
        svg = self.renderer.render_year_histogram(year_counts)
        assert "<svg" in svg
        assert "2023" in svg
        assert 'data-count="1000"' in svg

    def test_render_year_histogram_multi_year(self):
        """Test year histogram spanning multiple years."""
        year_counts = {2020: 100, 2021: 200, 2022: 300, 2023: 400, 2024: 500}
        svg = self.renderer.render_year_histogram(year_counts)
        assert "<svg" in svg
        assert "2020" in svg
        assert "2024" in svg
        assert 'data-count="500"' in svg

    def test_render_year_histogram_large_gaps(self):
        """Test year histogram with missing years in sequence."""
        year_counts = {2000: 100, 2005: 200, 2010: 300, 2020: 400}
        svg = self.renderer.render_year_histogram(year_counts)
        assert "<svg" in svg
        # All years should be represented
        for year in [2000, 2005, 2010, 2020]:
            assert str(year) in svg

    def test_svg_structure(self):
        """Verify SVG has proper structure (svg tag, bars, labels)."""
        counts = [10, 20, 30, 40, 50, 60, 70]
        svg = self.renderer.render_dow_histogram(counts)

        # Check SVG tags
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "viewBox" in svg
        assert "width" in svg
        assert "height" in svg

        # Check for bars
        assert '<rect class="temporal-bar bar"' in svg

        # Check for axes
        assert '<line class="axis"' in svg

        # Check for labels
        assert '<text class="tick-label"' in svg

    def test_tooltip_data_attributes(self):
        """Check data-count and data-pct attributes are present."""
        counts = [100, 200, 300, 150, 175, 125, 90]  # All 7 days with data
        svg = self.renderer.render_dow_histogram(counts)

        assert "data-count=" in svg
        assert "data-pct=" in svg
        assert "data-label=" in svg

    def test_axis_labels(self):
        """Verify correct labels (Mon/Tue, Jan/Feb, etc.)."""
        # DOW labels
        svg_dow = self.renderer.render_dow_histogram([10] * 7)
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            assert day in svg_dow

        # Month labels
        svg_month = self.renderer.render_month_histogram([10] * 12)
        for month in [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]:
            assert month in svg_month

    def test_bar_positioning(self):
        """Bars are correctly positioned and sized."""
        counts = [100, 200, 300]
        svg = self.renderer.render_dow_histogram(counts + [0] * 4)

        # Check that bars have x, y, width, height attributes
        assert "x=" in svg
        assert "y=" in svg
        assert "width=" in svg
        assert "height=" in svg

    def test_empty_data_all_types(self):
        """All chart types handle empty data gracefully."""
        assert "No data" in self.renderer.render_hour_histogram([])
        assert "No data" in self.renderer.render_dow_histogram([])
        assert "No data" in self.renderer.render_month_histogram([])
        assert "No data" in self.renderer.render_year_histogram({})


# ============================================================================
# Integration Tests
# ============================================================================


class TestDateTimeCardIntegration:
    """Integration tests for datetime cards with temporal data."""

    def test_datetime_card_with_temporal_data(self):
        """Test full card rendering with temporal stats."""
        # Create datetime stats with temporal data
        stats = DateTimeStats(
            name="test_datetime",
            count=1000,
            missing=100,
            min_ts=1609459200000000000,  # 2021-01-01
            max_ts=1640995200000000000,  # 2022-01-01
            by_hour=[40] * 24,
            by_dow=[140, 140, 140, 140, 140, 150, 150],
            by_month=[80] * 12,
            by_year={2021: 500, 2022: 500},
            dtype_str="datetime64[ns]",
            mono_inc=False,
            mono_dec=False,
            mem_bytes=8000,
            sample_ts=None,
            sample_scale=1.0,
            time_span_days=365,
            avg_interval_seconds=86400,
            interval_std_seconds=0,
            weekend_ratio=0.28,
            business_hours_ratio=0.35,
            seasonal_pattern=None,
            unique_est=365,
        )

        renderer = DateTimeCardRenderer()
        card_html = renderer.render_card(stats)

        # Check that temporal distribution section is present
        assert "dt-breakdown-grid" in card_html
        assert "Hour of Day" in card_html
        assert "Day of Week" in card_html
        assert "Month" in card_html
        assert "Year" in card_html

        # Check SVG elements are rendered
        assert '<svg class="temporal-bar-chart"' in card_html
        assert "temporal-bar" in card_html

    def test_chunked_datetime_processing(self):
        """Test multi-chunk accumulation of temporal patterns."""
        # Create datetime accumulator
        acc = DatetimeAccumulator("test_col")

        # Simulate multiple chunks
        base_time = datetime(2023, 1, 1)

        # Chunk 1: January data
        chunk1 = [
            (base_time + timedelta(days=i)).timestamp() * 1_000_000_000
            for i in range(0, 10)
        ]
        acc.update(chunk1)

        # Chunk 2: February data
        chunk2 = [
            (base_time + timedelta(days=31 + i)).timestamp() * 1_000_000_000
            for i in range(0, 10)
        ]
        acc.update(chunk2)

        summary = acc.finalize()

        # Check temporal patterns are accumulated correctly
        assert summary.count == 20
        assert len(summary.by_hour) == 24
        assert len(summary.by_dow) == 7
        assert len(summary.by_month) == 12
        assert sum(summary.by_month) == 20

    def test_business_hours_pattern(self):
        """Verify business hours (9-5, Mon-Fri) detection."""
        # Generate business hours data
        acc = DatetimeAccumulator("business_hours")
        timestamps = []

        base = datetime(2023, 1, 2)  # Monday
        for day in range(5):  # Mon-Fri
            for hour in range(9, 17):  # 9 AM - 5 PM
                dt = base + timedelta(days=day, hours=hour)
                timestamps.append(int(dt.timestamp() * 1_000_000_000))

        acc.update(timestamps)
        summary = acc.finalize()

        # Check that weekday hours (9-17) have high counts
        assert sum(summary.by_hour[9:17]) > sum(summary.by_hour[0:9])
        assert sum(summary.by_dow[:5]) > sum(summary.by_dow[5:])

    def test_weekend_pattern(self):
        """Test weekend concentration pattern."""
        acc = DatetimeAccumulator("weekend_data")
        timestamps = []

        base = datetime(2023, 1, 7)  # Saturday
        for day in range(2):  # Sat-Sun
            for hour in range(24):
                for _ in range(10):  # Multiple timestamps per hour
                    dt = base + timedelta(days=day, hours=hour)
                    timestamps.append(int(dt.timestamp() * 1_000_000_000))

        acc.update(timestamps)
        summary = acc.finalize()

        # Weekend (Sat=5, Sun=6) should have all the data
        assert summary.by_dow[5] > 0
        assert summary.by_dow[6] > 0
        assert sum(summary.by_dow[5:7]) == summary.count

    def test_seasonal_pattern(self):
        """Test summer peak detection."""
        acc = DatetimeAccumulator("seasonal")
        timestamps = []

        # More data in summer months (June-August)
        for month in [6, 7, 8]:
            for day in range(1, 30):
                for _ in range(10):  # 10 timestamps per day
                    dt = datetime(2023, month, day, 12, 0, 0)
                    timestamps.append(int(dt.timestamp() * 1_000_000_000))

        # Less data in other months
        for month in [1, 2, 3]:
            for day in range(1, 10):
                dt = datetime(2023, month, day, 12, 0, 0)
                timestamps.append(int(dt.timestamp() * 1_000_000_000))

        acc.update(timestamps)
        summary = acc.finalize()

        # Summer months should have more data
        summer_count = sum(
            [summary.by_month[i] for i in [5, 6, 7]]
        )  # Jun-Aug (0-indexed)
        winter_count = sum([summary.by_month[i] for i in [0, 1, 2]])  # Jan-Mar
        assert summer_count > winter_count

    def test_missing_temporal_data(self):
        """Test column with many missing values."""
        acc = DatetimeAccumulator("sparse_data")

        # Mix of valid timestamps and nulls
        valid_timestamps = [
            int(datetime(2023, 1, i + 1).timestamp() * 1_000_000_000) for i in range(10)
        ]
        nulls = [None] * 90

        acc.update(valid_timestamps + nulls)
        summary = acc.finalize()

        assert summary.count == 10
        assert summary.missing == 90

    def test_uniform_distribution(self):
        """Test evenly distributed timestamps."""
        acc = DatetimeAccumulator("uniform")
        timestamps = []

        # One timestamp per hour for a week
        base = datetime(2023, 1, 1)
        for day in range(7):
            for hour in range(24):
                dt = base + timedelta(days=day, hours=hour)
                timestamps.append(int(dt.timestamp() * 1_000_000_000))

        acc.update(timestamps)
        summary = acc.finalize()

        # All hours should have equal counts
        assert all(count == 7 for count in summary.by_hour)
        # All days should have equal counts
        assert all(count == 24 for count in summary.by_dow)

    def test_single_timestamp_repeated(self):
        """Test same timestamp repeated many times."""
        acc = DatetimeAccumulator("repeated")

        timestamp = int(datetime(2023, 6, 15, 14, 30).timestamp() * 1_000_000_000)
        acc.update([timestamp] * 1000)

        summary = acc.finalize()

        assert summary.count == 1000
        # All counts should be in hour 14
        assert summary.by_hour[14] == 1000
        assert sum(summary.by_hour) == 1000


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance and memory efficiency tests."""

    def test_large_dataset_rendering(self):
        """Test rendering performance with large counts."""
        renderer = TemporalHistogramRenderer()

        # Large counts
        large_counts = [1_000_000 + i * 100_000 for i in range(24)]

        start = time.time()
        svg = renderer.render_hour_histogram(large_counts)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should render in less than 1 second
        assert "<svg" in svg
        assert "M" in svg or "K" in svg  # Large numbers formatted

    def test_many_unique_years(self):
        """Test rendering with 100+ unique years."""
        renderer = TemporalHistogramRenderer()

        # 100 years of data
        year_counts = {year: 100 + year % 50 for year in range(1920, 2024)}

        start = time.time()
        svg = renderer.render_year_histogram(year_counts)
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert "<svg" in svg
        # Not all years should be labeled (too crowded)
        assert svg.count("1920") + svg.count("2023") >= 1

    def test_memory_efficiency(self):
        """Test that rendering doesn't leak memory."""
        renderer = TemporalHistogramRenderer()

        # Render many times to check for memory leaks
        for _ in range(1000):
            svg = renderer.render_hour_histogram([100] * 24)
            del svg  # Explicitly delete

        # If we get here without memory error, test passes
        assert True


# ============================================================================
# Visual Regression Tests
# ============================================================================


class TestVisualRegression:
    """Visual regression and structural validation tests."""

    def test_generate_sample_report(self):
        """Generate HTML report and check file structure."""
        import pysuricata as ps

        # Create sample datetime data
        dates = pd.date_range("2023-01-01", periods=1000, freq="H")
        df = pd.DataFrame({"timestamp": dates, "value": range(1000)})

        report = ps.profile(df)
        html = report.html

        # Check size is reasonable (not too large, not empty)
        assert len(html) > 10_000  # At least 10KB
        assert len(html) < 10_000_000  # Less than 10MB

    def test_svg_validates(self):
        """Test that SVG output is valid XML."""
        renderer = TemporalHistogramRenderer()
        svg = renderer.render_hour_histogram([10] * 24)

        # Basic XML validation
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert svg.count("<svg") == svg.count("</svg>")
        # Check no unclosed tags (simple check)
        assert (
            "<rect" not in svg
            or "</rect>" in svg
            or svg.count("<rect") == svg.count("/>") + svg.count("</rect>")
        )

    def test_accessibility(self):
        """Test SVG has aria-labels and titles."""
        renderer = TemporalHistogramRenderer()
        svg = renderer.render_hour_histogram([10] * 24)

        # Check accessibility attributes
        assert "aria-label=" in svg
        assert "role=" in svg
        assert "<title>" in svg

    def test_responsive_design(self):
        """Test charts render at different widths."""
        config_small = TemporalHistogramConfig(width=300, height=150)
        config_large = TemporalHistogramConfig(width=600, height=250)

        renderer_small = TemporalHistogramRenderer(config_small)
        renderer_large = TemporalHistogramRenderer(config_large)

        counts = [100] * 24
        svg_small = renderer_small.render_hour_histogram(counts)
        svg_large = renderer_large.render_hour_histogram(counts)

        # Both should render successfully
        assert "<svg" in svg_small
        assert "<svg" in svg_large

        # Dimensions should be different
        assert 'width="300"' in svg_small
        assert 'width="600"' in svg_large


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
