from pysuricata.render.missing_values_heatmap import (
    MissingValuesHeatmapRenderer,
    create_missing_values_heatmap_renderer,
)


def test_missing_values_heatmap_factory():
    renderer = create_missing_values_heatmap_renderer()
    assert isinstance(renderer, MissingValuesHeatmapRenderer)


def test_missing_values_heatmap_empty_states():
    renderer = MissingValuesHeatmapRenderer()

    # Empty metadata
    html = renderer.render_heatmap({}, ["colA"], 100)
    assert "No chunk metadata available" in html

    # Empty columns
    html = renderer.render_heatmap({"colA": [(0, 99, 5)]}, [], 100)
    assert "No chunk metadata available" in html

    # No missing values
    meta = {"colA": [(0, 99, 0)], "colB": [(0, 99, 0)]}
    html = renderer.render_heatmap(meta, ["colA", "colB"], 100)
    assert "No missing values detected" in html


def test_missing_values_heatmap_colors_and_rendering():
    renderer = MissingValuesHeatmapRenderer()

    cols = ["col_none", "col_low", "col_med", "col_high", "col_crit"]

    # 1 chunk of 100 rows each
    meta = {
        "col_none": [(0, 99, 0)],  # 0%
        "col_low": [(0, 99, 4)],  # 4%
        "col_med": [(0, 99, 15)],  # 15%
        "col_high": [(0, 99, 40)],  # 40%
        "col_crit": [(0, 99, 80)],  # 80%
    }

    html = renderer.render_heatmap(meta, cols, 100)

    assert "Cross-Column Missing Values Distribution" in html
    # Check all severity classes are applied
    assert "hm-none" in html
    assert "hm-low" in html
    assert "hm-medium" in html
    assert "hm-high" in html
    assert "hm-critical" in html


def test_missing_values_heatmap_long_name_truncation():
    renderer = MissingValuesHeatmapRenderer()

    long_col = "this_is_an_extremely_long_column_name_that_should_truncate"
    meta = {long_col: [(0, 99, 10)]}

    html = renderer.render_heatmap(meta, [long_col], 100)

    assert long_col in html  # present in title attr
    assert "this_is_an_extremely_l..." in html  # truncated in display


def test_missing_values_heatmap_expandable_section():
    renderer = MissingValuesHeatmapRenderer()

    # Generate 25 columns to trigger the expandable section (max_columns_initial = 20)
    cols = [f"col_{i}" for i in range(25)]
    meta = {c: [(0, 99, 5)] for c in cols}

    html = renderer.render_heatmap(meta, cols, 100)

    assert "25 columns with missing values" in html
    assert "heatmap-expandable" in html
    assert "Show 5 more columns" in html
