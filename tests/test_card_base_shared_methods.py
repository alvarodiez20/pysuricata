"""Unit tests for shared methods added to CardRenderer base class.

Covers _build_approx_badge, _build_chunk_distribution_simple, and
_build_missing_values_table including edge cases.
"""

import pytest

from pysuricata.render.card_base import CardRenderer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StatsStub:
    """Minimal stats-like object that carries chunk_metadata."""

    def __init__(self, chunk_metadata=None):
        self.chunk_metadata = chunk_metadata


@pytest.fixture
def renderer() -> CardRenderer:
    return CardRenderer()


# ---------------------------------------------------------------------------
# _build_approx_badge
# ---------------------------------------------------------------------------


class TestBuildApproxBadge:
    def test_true_returns_badge(self, renderer):
        html = renderer._build_approx_badge(True)
        assert '<span class="badge">approx</span>' in html

    def test_false_returns_empty_string(self, renderer):
        assert renderer._build_approx_badge(False) == ""

    def test_none_is_falsy_returns_empty_string(self, renderer):
        # None should be treated as falsy (same as False)
        assert renderer._build_approx_badge(None) == ""

    def test_zero_is_falsy_returns_empty_string(self, renderer):
        assert renderer._build_approx_badge(0) == ""

    def test_truthy_integer_returns_badge(self, renderer):
        # Any truthy value should return the badge
        html = renderer._build_approx_badge(1)
        assert '<span class="badge">approx</span>' in html


# ---------------------------------------------------------------------------
# _build_chunk_distribution_simple
# ---------------------------------------------------------------------------


class TestBuildChunkDistributionSimple:
    def test_no_chunk_metadata_returns_empty(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        assert renderer._build_chunk_distribution_simple(stats, 100) == ""

    def test_empty_chunk_list_returns_empty(self, renderer):
        stats = StatsStub(chunk_metadata=[])
        assert renderer._build_chunk_distribution_simple(stats, 100) == ""

    def test_total_values_zero_returns_empty(self, renderer):
        # Even with metadata, if total is 0 nothing should render
        stats = StatsStub(chunk_metadata=[(0, 99, 0)])
        assert renderer._build_chunk_distribution_simple(stats, 0) == ""

    def test_single_chunk_no_missing_is_low_severity(self, renderer):
        stats = StatsStub(chunk_metadata=[(0, 99, 0)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert "chunk-distribution" in html
        assert 'class="chunk-segment low"' in html
        assert "Peak: 0.0%" in html

    def test_single_chunk_all_missing_is_high_severity(self, renderer):
        stats = StatsStub(chunk_metadata=[(0, 99, 100)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'class="chunk-segment high"' in html
        assert "Peak: 100.0%" in html

    def test_boundary_low_medium_at_5_pct(self, renderer):
        # Exactly 5% missing → "low" (≤ 5)
        stats = StatsStub(chunk_metadata=[(0, 99, 5)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'class="chunk-segment low"' in html

    def test_boundary_medium_at_6_pct(self, renderer):
        # 6% missing → "medium" (5 < x ≤ 20)
        stats = StatsStub(chunk_metadata=[(0, 99, 6)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'class="chunk-segment medium"' in html

    def test_boundary_medium_high_at_20_pct(self, renderer):
        # Exactly 20% missing → "medium" (≤ 20)
        stats = StatsStub(chunk_metadata=[(0, 99, 20)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'class="chunk-segment medium"' in html

    def test_boundary_high_at_21_pct(self, renderer):
        # 21% → "high"
        stats = StatsStub(chunk_metadata=[(0, 99, 21)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'class="chunk-segment high"' in html

    def test_three_chunks_all_severity_levels(self, renderer):
        # 4% low, 15% medium, 25% high
        chunks = [(0, 99, 4), (100, 199, 15), (200, 299, 25)]
        stats = StatsStub(chunk_metadata=chunks)
        html = renderer._build_chunk_distribution_simple(stats, 300)
        assert 'class="chunk-segment low"' in html
        assert 'class="chunk-segment medium"' in html
        assert 'class="chunk-segment high"' in html

    def test_chunk_count_shown_correctly(self, renderer):
        chunks = [(0, 49, 0), (50, 99, 5), (100, 149, 2)]
        stats = StatsStub(chunk_metadata=chunks)
        html = renderer._build_chunk_distribution_simple(stats, 150)
        assert "3 chunks analyzed" in html

    def test_peak_missing_pct_shown_correctly(self, renderer):
        # Highest missing_pct across chunks is 50% (50/100)
        chunks = [(0, 99, 10), (100, 199, 50), (200, 299, 5)]
        stats = StatsStub(chunk_metadata=chunks)
        html = renderer._build_chunk_distribution_simple(stats, 300)
        assert "Peak: 50.0%" in html

    def test_data_attributes_present(self, renderer):
        stats = StatsStub(chunk_metadata=[(10, 109, 15)])
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'data-start="10"' in html
        assert 'data-end="109"' in html
        assert 'data-missing="15"' in html
        assert "data-total=" in html
        assert "data-pct=" in html

    def test_no_legend_rendered(self, renderer):
        """#294 -- the three-item severity legend belongs once, in the Missing
        values section, not repeated on every card that has a gap."""
        stats = StatsStub(chunk_metadata=[(0, 9, 0)])
        html = renderer._build_chunk_distribution_simple(stats, 10)
        assert "chunk-legend" not in html
        assert "Low (0-5%)" not in html

    # --- Edge cases ---

    def test_chunk_size_zero_no_division_error(self, renderer):
        # end_row < start_row → chunk_size = end-start+1 = 0 (malformed data)
        stats = StatsStub(chunk_metadata=[(10, 9, 0)])
        # Should not raise ZeroDivisionError; missing_pct defaults to 0.0
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert "chunk-distribution" in html

    def test_missing_count_exceeds_chunk_size_no_crash(self, renderer):
        # Data integrity issue: missing_count > chunk_size → pct > 100%
        # Should classify as "high" without crashing
        stats = StatsStub(chunk_metadata=[(0, 9, 500)])  # chunk_size=10, missing=500
        html = renderer._build_chunk_distribution_simple(stats, 100)
        assert 'class="chunk-segment high"' in html

    def test_single_row_chunk(self, renderer):
        # chunk_size = 1
        stats = StatsStub(chunk_metadata=[(5, 5, 0)])
        html = renderer._build_chunk_distribution_simple(stats, 1)
        assert "chunk-distribution" in html

    def test_width_pct_formatted_to_two_decimals(self, renderer):
        # Each chunk is 1/3 of total → 33.33%
        stats = StatsStub(chunk_metadata=[(0, 0, 0), (1, 1, 0), (2, 2, 0)])
        html = renderer._build_chunk_distribution_simple(stats, 3)
        assert "33.33%" in html

    def test_stats_without_chunk_metadata_attribute(self, renderer):
        # Object that doesn't even have chunk_metadata attribute
        class NoMetadata:
            pass

        html = renderer._build_chunk_distribution_simple(NoMetadata(), 100)
        assert html == ""


# ---------------------------------------------------------------------------
# _build_missing_values_table
# ---------------------------------------------------------------------------


class TestBuildMissingValuesTable:
    def test_no_missing_shows_100_present(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(100, 100.0, 0, 0.0, stats, 100)
        assert "Data Completeness" in html
        assert "100.0%" in html

    def test_all_missing_shows_0_present(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(0, 0.0, 100, 100.0, stats, 100)
        assert "0.0%" in html  # present_pct
        assert "100.0%" in html  # missing_pct

    def test_partial_missing(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(75, 75.0, 25, 25.0, stats, 100)
        assert "75.0%" in html
        assert "25.0%" in html

    def test_bar_widths_match_pct(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(60, 60.0, 40, 40.0, stats, 100)
        assert "width: 60.0%" in html
        assert "width: 40.0%" in html

    def test_with_chunk_metadata_includes_chunk_distribution(self, renderer):
        stats = StatsStub(chunk_metadata=[(0, 99, 10)])
        html = renderer._build_missing_values_table(90, 90.0, 10, 10.0, stats, 100)
        assert "chunk-distribution" in html
        assert "Data Completeness" in html

    def test_without_chunk_metadata_no_chunk_distribution(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(90, 90.0, 10, 10.0, stats, 100)
        assert "chunk-distribution" not in html

    def test_zero_total_values_renders_without_crash(self, renderer):
        # total_values=0 → chunk distribution returns "" but completeness still shows
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(0, 0.0, 0, 0.0, stats, 0)
        assert "Data Completeness" in html

    def test_large_counts_formatted_with_commas(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(
            1_000_000, 90.9, 100_000, 9.1, stats, 1_100_000
        )
        assert "1,000,000" in html
        assert "100,000" in html

    def test_float_pct_edge_case_sums_to_not_100(self, renderer):
        # Due to floating point, pct might not sum to exactly 100
        stats = StatsStub(chunk_metadata=None)
        # Should render without error
        html = renderer._build_missing_values_table(
            67, 66.9999999, 33, 33.0000001, stats, 100
        )
        assert "Data Completeness" in html

    def test_title_and_label_present(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(80, 80.0, 20, 20.0, stats, 100)
        assert "Data Completeness" in html
        assert "Present:" in html
        assert "Missing:" in html

    def test_present_count_shown_with_title(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(42, 42.0, 58, 58.0, stats, 100)
        assert "42" in html
        assert "58" in html

    def test_hover_titles_on_bars(self, renderer):
        stats = StatsStub(chunk_metadata=None)
        html = renderer._build_missing_values_table(70, 70.0, 30, 30.0, stats, 100)
        assert 'title="Present: 70.0%"' in html
        assert 'title="Missing: 30.0%"' in html

    def test_chunk_metadata_total_values_passed_correctly(self, renderer):
        # chunk with width proportional to total_values
        stats = StatsStub(chunk_metadata=[(0, 49, 0), (50, 99, 5)])
        html = renderer._build_missing_values_table(95, 95.0, 5, 5.0, stats, 100)
        # Each chunk is 50% of total → width 50.00%
        assert "50.00%" in html
