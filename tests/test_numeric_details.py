from pysuricata.accumulators.numeric import NumericSummary
from pysuricata.render.cards import render_numeric_card


def make_numeric_summary():
    return NumericSummary(
        name="x",
        count=100,
        missing=2,
        unique_est=95,
        mean=1.2,
        std=0.5,
        variance=0.25,
        se=0.05,
        cv=0.4,
        gmean=1.1,
        min=0.0,
        q1=0.5,
        median=1.0,
        q3=1.5,
        iqr=1.0,
        mad=0.3,
        skew=0.1,
        kurtosis=3.2,
        jb_chi2=1.1,
        max=10.0,
        zeros=5,
        negatives=1,
        outliers_iqr=2,
        outliers_mod_zscore=1,
        approx=False,
        inf=0,
        int_like=False,
        unique_ratio_approx=0.95,
        hist_counts=[1, 2, 3],
        top_values=[(1.0, 10), (2.0, 5)],
        sample_vals=[0.0, 0.5, 1.0, 1.5, 10.0],
        heap_pct=10.0,
        gran_decimals=1,
        gran_step=0.5,
        bimodal=False,
        ci_lo=1.1,
        ci_hi=1.3,
        mem_bytes=0,
        mono_inc=False,
        mono_dec=False,
        dtype_str="float64",
        corr_top=[("y", 0.9)],
        sample_scale=1.0,
        min_items=[("i0", 0.0)],
        max_items=[("i9", 10.0)],
    )


def test_numeric_details_tabs_present():
    s = make_numeric_summary()
    html = render_numeric_card(s)
    assert 'data-tab="stats"' in html
    assert 'data-tab="common"' in html
    assert 'data-tab="extremes"' in html
    assert 'data-tab="corr"' in html
    # Quantiles content should appear within the stats pane
    assert "P90" in html or "P95" in html
    # Content sniff
    assert "Top correlations" in html or "Correlations" in html
    # 5b.5 replaced the two `Min values` / `Max values` tables with one pane
    # plotting both tails on the Outliers pane's axis, so the headings are the
    # tail sizes and the content is the position of each value.
    assert "lowest" in html and "highest" in html


def test_numeric_card_missing_chunk_visualization():
    """The strip the numeric card actually renders.

    This used to drive `_build_dataprep_spectrum_visualization`,
    `_generate_missing_insights` and `_render_chunk_visualization` -- three
    methods #294 deleted because no code path reached any of them. The shared
    `_build_chunk_distribution_simple` is what a report contains.
    """
    from pysuricata.render.numeric_card import NumericCardRenderer

    s = make_numeric_summary()
    s.chunk_metadata = [(0, 9, 2), (10, 19, 0)]
    renderer = NumericCardRenderer()

    html = renderer._build_chunk_distribution_simple(s, 20)
    assert "chunk-segment" in html
    assert 'data-missing="2"' in html

    s.chunk_metadata = []
    assert renderer._build_chunk_distribution_simple(s, 20) == ""


def test_numeric_card_outliers_severity_and_tables():
    """The severity bands, read through the pane that now owns them.

    These used to call `_get_outlier_severity` on the renderer. 5b.2 moved the
    arithmetic into `render/outlier_fence.py` so the Min/Max pane can read the
    same bands -- a value that is `high` in one pane cannot be `moderate` in
    the other, and one implementation is the only way to guarantee it. The
    thresholds are unchanged: 3.0/2.0 IQRs, 3.5/2.5 MADs.
    """
    from pysuricata.render.outlier_fence import (
        _IQR_BANDS,
        _MAD_BANDS,
        _band,
        build_fence,
    )

    assert _band(3.0, _IQR_BANDS) == "extreme"
    assert _band(2.1, _IQR_BANDS) == "high"
    assert _band(0.6, _IQR_BANDS) == "moderate"

    assert _band(5.0, _MAD_BANDS) == "extreme"
    assert _band(3.0, _MAD_BANDS) == "high"
    assert _band(1.5, _MAD_BANDS) == "moderate"

    s = make_numeric_summary()
    s.q1, s.q3, s.iqr = 10.0, 20.0, 10.0
    s.median, s.mad = 15.0, 2.0
    s.min, s.max = 10.0, 50.0
    s.count = 40
    # A body inside the fence plus two values well past it, so the pane has
    # something to draw and something to leave alone.
    s.sample_vals = [float(v) for v in range(10, 21)] + [41.0, 50.0]

    fence = build_fence(s)
    assert fence is not None
    assert fence.hi == 35.0
    assert fence.n_high == 2
    # 10.0 is the minimum and the lower fence is -5.0, so no value can cross it.
    assert not fence.lo_possible
    assert fence.n_low == 0

    verdicts = {row.value: (row.iqr_severity, row.mad_severity) for row in fence.rows}
    assert verdicts[50.0][0] == "extreme"  # 3.0x IQR past Q3
    assert verdicts[41.0][0] == "high"  # 2.1x IQR past Q3


def test_numeric_card_correlation_and_missing_tables():
    from pysuricata.render.numeric_card import NumericCardRenderer

    s = make_numeric_summary()
    renderer = NumericCardRenderer()

    # Correlation table
    s.corr_top = [("col_b", 0.95), ("col_c", -0.85)]
    corr_html = renderer._build_correlation_table(s)
    assert "col_b" in corr_html
    assert "+0.950" in corr_html
    assert "col_c" in corr_html
    assert "-0.850" in corr_html
    assert "correlations" in corr_html

    # 5b.6: a column with no partners at all renders no pane, so the tab
    # disappears rather than repeating the section-level empty state inside
    # the card. A column *with* partners always lists them, however weak.
    s.corr_top = []
    assert renderer._build_correlation_table(s) == ""

    # Missing table empty
    s.missing = 0
    assert "0.0%" in renderer._build_missing_values_table(s)

    # Extremes pane with nothing tracked: a sentence, not a dash in a table.
    s.min_items = []
    s.max_items = []
    extremes_html = renderer._build_extremes_table(s)
    assert "No extreme values were tracked" in extremes_html
