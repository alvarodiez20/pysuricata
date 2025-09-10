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
        outliers_mad=1,
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
