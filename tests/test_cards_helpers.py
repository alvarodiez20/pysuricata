from pysuricata.render.card_base import (
    compute_x_ticks_and_labels,
    format_hist_bin_labels,
)
from pysuricata.render.categorical_card import CategoricalCardRenderer
from pysuricata.render.numeric_card import NumericCardRenderer


def test_format_hist_bin_labels_linear():
    a, b = format_hist_bin_labels(0.0, 10.0, "lin")
    assert a in {"0", "0.0", "0.0000"}
    assert b in {"10", "10.0"}


def test_format_hist_bin_labels_log():
    # log10(10)=1, log10(100)=2 -> expect linear labels 10 and 100
    a, b = format_hist_bin_labels(1.0, 2.0, "log")
    assert a in {"10", "10.0"}
    assert b in {"100", "100.0", "1,000"} or b.startswith("1e") is False


def test_x_ticks_and_labels_linear():
    ticks, step, labels = compute_x_ticks_and_labels(0.0, 10.0, "lin")
    assert labels is None
    assert step > 0
    assert ticks == sorted(ticks)
    assert ticks[0] <= 0.0 + 1e-9 and ticks[-1] >= 10.0 - 1e-9


def test_x_ticks_and_labels_log():
    ticks, step, labels = compute_x_ticks_and_labels(0.0, 3.0, "log")
    assert step == 1.0
    assert isinstance(labels, list) and len(labels) == len(ticks)
    assert labels[0] in {"1", "10"}
    assert labels[-1] in {"1000", "1,000", "100"}


def test_numeric_hist_variants_html_ids():
    # Create a mock numeric stats object
    class MockNumericStats:
        def __init__(self):
            self.sample_vals = [1, 2, 3, 4, 5]
            self.sample_scale = 1.0

    renderer = NumericCardRenderer()
    stats = MockNumericStats()
    html = renderer._build_histogram_variants("col_x", "X", stats)
    assert "col_x-lin-bins-25" in html
    assert 'data-scale="log"' in html


def test_cat_variants_html_ids():
    items = [("a", 5), ("b", 3), ("c", 2)]
    renderer = CategoricalCardRenderer()
    html = renderer._build_categorical_variants(
        "col_y", items, total=10, topn_list=[2, 3], default_topn=2
    )
    assert "col_y-cat-top-2" in html
    assert 'class="cat variant active"' in html or "cat variant active" in html
