from pysuricata.render.missing_context import (
    _build_navigation_buttons,
    build_missing_values_context,
)
from pysuricata.render.missing_values_heatmap import (
    MissingValuesHeatmapRenderer,
    create_missing_values_heatmap_renderer,
)


class TestMissingContext:
    def test_build_missing_values_context_empty(self):
        # Empty metadata
        assert build_missing_values_context("col_a", None, ["col_a"]) == ""

        # No columns with missing values
        metadata = {"col_a": [(0, 10, 0)]}
        assert build_missing_values_context("col_a", metadata, ["col_a"]) == ""

        # Current column not in metadata or has no missing
        metadata_with_missing = {"col_b": [(0, 10, 5)]}
        assert (
            build_missing_values_context(
                "col_a", metadata_with_missing, ["col_a", "col_b"]
            )
            == ""
        )

    def test_build_missing_values_context_basic(self):
        metadata = {
            "col_a": [(0, 99, 10)],  # 10%
            "col_b": [(0, 99, 50)],  # 50%
            "col_long_name_that_exceeds_twenty_chars": [(0, 99, 90)],  # 90%
        }
        all_cols = ["col_a", "col_b", "col_long_name_that_exceeds_twenty_chars"]

        assert "Rank #3" not in build_missing_values_context(
            "col_a", metadata, all_cols
        )

        # col_long_name... is rank 1 (90%), col_b is rank 2 (50%), col_a is rank 3 (10%)
        html_a = build_missing_values_context("col_a", metadata, all_cols)
        assert "3 of 3" in html_a
        assert "col_b" in html_a  # prev column

        html_b = build_missing_values_context("col_b", metadata, all_cols)
        assert "2 of 3" in html_b
        assert "col_long_name" in html_b  # prev
        assert "col_a" in html_b  # next

        # check that line 66 is executed:
        html_first = build_missing_values_context(
            "col_long_name_that_exceeds_twenty_chars", metadata, all_cols
        )
        assert "1 of 3" in html_first
        assert "col_b" in html_first

    def test_build_navigation_buttons(self):
        # First column (no prev, has next)
        html_first = _build_navigation_buttons(None, "next_col")
        assert "disabled" in html_first
        assert "next_col" in html_first

        # Middle column
        html_mid = _build_navigation_buttons("prev_col", "next_col")
        assert "prev_col" in html_mid
        assert "next_col" in html_mid
        assert "disabled" not in html_mid

        # Last column (has prev, no next)
        html_last = _build_navigation_buttons("prev_col", None)
        assert "prev_col" in html_last
        assert "disabled" in html_last


class TestMissingValuesHeatmap:
    def test_factory(self):
        renderer = create_missing_values_heatmap_renderer()
        assert isinstance(renderer, MissingValuesHeatmapRenderer)

    def test_render_empty_states(self):
        renderer = MissingValuesHeatmapRenderer()
        assert "No chunk metadata available" in renderer.render_heatmap(None, [], 100)
        assert "No chunk metadata available" in renderer.render_heatmap(
            {"a": []}, [], 100
        )

        metadata_no_missing = {"col_a": [(0, 10, 0)]}
        assert "No missing values detected in any column!" in renderer.render_heatmap(
            metadata_no_missing, ["col_a"], 10
        )

    def test_render_basic_heatmap(self):
        renderer = MissingValuesHeatmapRenderer()
        metadata = {
            "col_a": [(0, 49, 0), (50, 99, 25)],  # None, High
            "col_b": [(0, 99, 2)],  # Low
            "col_c": [(0, 99, 10)],  # Medium
            "col_d": [(0, 99, 60)],  # Critical
            "col_long_name_exceeding_twenty_five": [(0, 99, 10)],
        }
        all_cols = list(metadata.keys())

        html = renderer.render_heatmap(metadata, all_cols, 100)
        assert "Cross-Column Missing Values Distribution" in html
        assert "col_a" in html
        assert "hm-none" in html
        assert "hm-low" in html
        assert "hm-medium" in html
        assert "hm-high" in html
        assert "hm-critical" in html

    def test_render_expanded_heatmap(self):
        renderer = MissingValuesHeatmapRenderer()
        renderer.max_columns_initial = 2  # Override for testing

        metadata = {f"col_{i}": [(0, 99, 50)] for i in range(5)}
        all_cols = list(metadata.keys())

        html = renderer.render_heatmap(metadata, all_cols, 100)
        assert "Show 3 more columns" in html
        assert "heatmap-expandable" in html
        assert "hidden" in html
