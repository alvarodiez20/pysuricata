from pysuricata.render.missing_context import (
    _build_navigation_buttons,
    build_missing_values_context,
)


def test_missing_values_context_empty():
    # Empty metadata
    html = build_missing_values_context("colA", None, ["colA", "colB"])
    assert html == ""

    # Dictionary is empty
    html = build_missing_values_context("colA", {}, ["colA", "colB"])
    assert html == ""

    # Metadata has no missing values
    meta = {"colA": [(0, 9, 0)], "colB": [(0, 9, 0)]}
    html = build_missing_values_context("colA", meta, ["colA", "colB"])
    assert html == ""

    # Current column has no missing values but others do
    meta = {"colA": [(0, 9, 0)], "colB": [(0, 9, 2)]}
    html = build_missing_values_context("colA", meta, ["colA", "colB"])
    assert html == ""


def test_missing_values_context_navigation():
    # Setup some columns with varying missing values to test sorting and navigation
    all_cols = ["no_miss", "low_miss", "high_miss", "med_miss"]
    meta = {
        "no_miss": [(0, 99, 0)],
        "low_miss": [(0, 99, 5)],  # 5 missing
        "med_miss": [(0, 99, 20)],  # 20 missing
        "high_miss": [(0, 99, 50)],  # 50 missing
    }

    # Ranked order should be: high_miss (1), med_miss (2), low_miss (3)

    # Test high_miss (First)
    html_high = build_missing_values_context("high_miss", meta, all_cols)
    assert "1 of 3" in html_high
    assert "disabled" in html_high  # Prev is disabled
    assert "med_miss" in html_high  # Next is med_miss

    # Test med_miss (Middle)
    html_med = build_missing_values_context("med_miss", meta, all_cols)
    assert "2 of 3" in html_med
    assert "high_miss" in html_med  # Prev is high_miss
    assert "low_miss" in html_med  # Next is low_miss

    # Test low_miss (Last)
    html_low = build_missing_values_context("low_miss", meta, all_cols)
    assert "3 of 3" in html_low
    assert "med_miss" in html_low  # Prev is med_miss
    assert "disabled" in html_low  # Next is disabled


def test_build_navigation_buttons_long_names():
    # Test truncation string formatting in navigation
    prev_col = "this_is_a_very_long_column_name_for_testing"
    next_col = "another_extremely_long_column_name"

    html = _build_navigation_buttons(prev_col, next_col)

    assert "this_is_a_very_lo..." in html
    assert "another_extremely..." in html
    assert (
        'title="Previous column with missing values: this_is_a_very_long_column_name_for_testing"'
        in html
    )
    assert (
        'title="Next column with missing values: another_extremely_long_column_name"'
        in html
    )
