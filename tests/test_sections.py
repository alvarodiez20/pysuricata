import pytest


def test_render_sample_section_pandas_basic():
    pd = pytest.importorskip("pandas", reason="pandas not installed")

    from pysuricata.render.sections import render_sample_section_pandas

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["x", "y", "z", "w"],
        }
    )
    html = render_sample_section_pandas(df, sample_rows=2)
    assert "<table" in html and "</table>" in html
    # Expect two rows sampled
    assert html.count("<tr>") >= 3  # header + >=2 rows


def test_render_sample_section_polars_basic():
    pl = pytest.importorskip("polars", reason="polars not installed")

    from pysuricata.render.sections import render_sample_section_polars

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["x", "y", "z", "w"],
        }
    )
    html = render_sample_section_polars(df, sample_rows=2)
    assert "<table" in html and "</table>" in html
    # Expect two rows sampled
    assert html.count("<tr>") >= 3


def test_render_sample_section_polars_zero_rows():
    pl = pytest.importorskip("polars", reason="polars not installed")

    from pysuricata.render.sections import render_sample_section_polars

    df = pl.DataFrame({"a": [], "b": []})
    html = render_sample_section_polars(df, sample_rows=5)
    assert "<table" in html
    # No data rows, only header
    assert html.count("<tr>") <= 2


def test_render_sample_section_dispatcher_invalid_object():
    from pysuricata.render.sections import render_sample_section

    class Dummy: ...

    html = render_sample_section(Dummy(), sample_rows=5)
    assert "Unable to render sample preview" in html


def test_render_sample_section_pandas_zero_rows():
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    from pysuricata.render.sections import render_sample_section_pandas

    df = pd.DataFrame({"x": [1, 2, 3]})
    html = render_sample_section_pandas(df, sample_rows=0)
    # Should still render a table with header and no body rows
    assert "<table" in html and "</table>" in html
    assert html.count("<tr>") <= 2


def test_render_sample_section_pandas_more_than_len():
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    from pysuricata.render.sections import render_sample_section_pandas

    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    html = render_sample_section_pandas(df, sample_rows=10)
    # Should cap at len(df)
    assert html.count("<tr>") >= 3  # header + 2 rows


def test_render_sample_section_pandas_alignment():
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    from pysuricata.render.sections import render_sample_section_pandas

    df = pd.DataFrame({"num": [1.5, 2.5, None], "txt": ["a", "b", "c"]})
    html = render_sample_section_pandas(df, sample_rows=3)
    # Numeric values should be wrapped in span.num at least once
    assert '<span class="num">' in html


def test_render_sample_section_polars_negative_rows():
    pl = pytest.importorskip("polars", reason="polars not installed")
    from pysuricata.render.sections import render_sample_section_polars

    df = pl.DataFrame({"x": [1, 2, 3]})
    html = render_sample_section_polars(df, sample_rows=-5)
    # Negative treated as zero; still renders a table structure
    assert "<table" in html
    assert html.count("<tr>") <= 2


def test_render_sample_section_polars_more_than_len_alignment():
    pl = pytest.importorskip("polars", reason="polars not installed")
    from pysuricata.render.sections import render_sample_section_polars

    df = pl.DataFrame({"num": [1, 2, 3, 4], "txt": ["a", "b", "c", "d"]})
    html = render_sample_section_polars(df, sample_rows=10)
    # Should contain a numeric span for alignment (first column is positional index)
    assert '<span class="num">' in html
