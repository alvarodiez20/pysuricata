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
    assert "Min values" in html and "Max values" in html


def test_numeric_card_missing_chunk_visualization():
    from pysuricata.render.numeric_card import NumericCardRenderer

    s = make_numeric_summary()
    s.chunk_metadata = [(0, 9, 2), (10, 19, 0)]  # Start, end, missing
    renderer = NumericCardRenderer()

    # Test _build_dataprep_spectrum_visualization
    html = renderer._build_dataprep_spectrum_visualization(s)
    assert "spectrum-segment" in html
    assert 'data-missing="2"' in html

    # Test empty metadata behavior
    s.chunk_metadata = []
    assert renderer._build_dataprep_spectrum_visualization(s) == ""

    # Test simulate chunk distribution
    s.count = 2000
    s.missing = 100
    chunks = renderer._simulate_chunk_distribution(s)
    assert len(chunks) >= 2
    assert sum(c["missing"] for c in chunks) == 100

    # Test insights generation
    insights = renderer._generate_missing_insights(chunks, 5.0)
    assert "overall_missing_pct" in insights
    assert "max_missing_pct" in insights
    assert "severity" in insights
    assert insights["total_chunks"] == len(chunks)

    # Test complete chunk visualization render
    viz_html = renderer._render_chunk_visualization(chunks, insights, s)
    assert "chunk-bar-container" in viz_html
    assert "chunk-bar-fill" in viz_html


def test_numeric_card_outliers_severity_and_tables():
    from pysuricata.render.numeric_card import NumericCardRenderer

    s = make_numeric_summary()
    s.q1 = 10.0
    s.q3 = 20.0
    s.iqr = 10.0
    s.median = 15.0
    s.mad = 2.0
    renderer = NumericCardRenderer()

    # IQR Severity
    assert (
        renderer._get_outlier_severity(50.0, "IQR", s)[1] == "extreme"
    )  # 3x IQR (score=3.0)
    assert (
        renderer._get_outlier_severity(41.0, "IQR", s)[1] == "high"
    )  # 2.1x IQR (score=2.1)
    assert (
        renderer._get_outlier_severity(26.0, "IQR", s)[1] == "moderate"
    )  # 1.6x IQR (score=0.6)

    # MAD Severity
    assert renderer._get_outlier_severity(25.0, "MAD", s)[1] == "extreme"  # 5x MAD
    assert renderer._get_outlier_severity(21.0, "MAD", s)[1] == "high"  # 3x MAD
    assert renderer._get_outlier_severity(18.0, "MAD", s)[1] == "moderate"  # 1.5x MAD

    # Fallback Severity
    assert renderer._get_outlier_severity(20.0, "UNKNOWN", s)[1] == "moderate"

    # Test enhanced outliers table with empty list
    s.outliers_iqr = 10
    empty_summary = renderer._format_enhanced_outliers_table([], {}, s, "high")
    assert "0 outliers" in empty_summary

    # Test enhanced outliers table with actual data
    outliers = [(41.0, ["IQR"]), (21.0, ["MAD"]), (50.0, ["IQR", "MAD"])]
    idx_map = {41.0: ["i0"], 21.0: ["i1"], 50.0: ["i2"]}
    table_html = renderer._format_enhanced_outliers_table(outliers, idx_map, s, "high")
    assert "High Outliers" in table_html
    assert "Extreme:" in table_html


def test_numeric_card_correlation_and_missing_tables():
    from pysuricata.render.numeric_card import NumericCardRenderer

    s = make_numeric_summary()
    renderer = NumericCardRenderer()

    # Correlation table
    s.corr_top = [("col_b", 0.95), ("col_c", -0.85)]
    corr_html = renderer._build_correlation_table(s)
    assert "col_b" in corr_html
    assert "0.95" in corr_html
    assert "col_c" in corr_html
    assert "-0.85" in corr_html
    assert "correlation" in corr_html

    s.corr_top = []
    assert "no-correlations" in renderer._build_correlation_table(s)

    # Missing table empty
    s.missing = 0
    assert "0.0%" in renderer._build_missing_values_table(s)

    # Extremes table empty handling
    s.min_items = []
    s.max_items = []
    extremes_html = renderer._build_extremes_table(s)
    assert "—" in extremes_html
    assert "extremes" in extremes_html
