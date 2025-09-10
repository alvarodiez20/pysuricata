import math


def test_safe_col_id_basic():
    from pysuricata.render.svg_utils import safe_col_id

    assert safe_col_id("a b") == "col_a_b"
    assert safe_col_id("123") == "col_123"
    assert safe_col_id("a/b$c") == "col_a_b_c"
    assert safe_col_id(42) == "col_42"


def test_nice_ticks_monotone_and_degenerate():
    from pysuricata.render.svg_utils import nice_ticks

    ticks, step = nice_ticks(0, 1, 5)
    assert len(ticks) >= 2 and step > 0
    # swapped bounds
    t2, s2 = nice_ticks(5, 1, 4)
    assert t2[0] <= t2[-1]
    assert s2 > 0
    # degenerate bounds expand by 1
    t3, _ = nice_ticks(2, 2, 4)
    assert t3[0] <= 2 <= t3[-1]


def test_fmt_tick_ranges():
    from pysuricata.render.svg_utils import fmt_tick

    assert fmt_tick(float("nan"), 1) == ""
    assert fmt_tick(1000, 1) == "1,000"
    assert fmt_tick(1.234, 0.1) == "1.2"
    assert fmt_tick(1.2345, 0.01) == "1.23"


def test_log_ticks_basic():
    from pysuricata.render.svg_utils import nice_log_ticks_from_log10

    ticks, labels = nice_log_ticks_from_log10(0.0, 3.0, max_ticks=5)
    assert ticks == [0.0, 1.0, 2.0, 3.0]
    assert labels[0] == "1" and labels[-1] in ("1,000", "1000")


def test_log_ticks_swapped_and_degenerate():
    from pysuricata.render.svg_utils import nice_log_ticks_from_log10

    # swapped order
    t1, l1 = nice_log_ticks_from_log10(3.0, 0.0, max_ticks=4)
    assert t1[0] <= t1[-1] and len(t1) <= 4
    # degenerate
    t2, l2 = nice_log_ticks_from_log10(2.0, 2.0, max_ticks=3)
    assert len(t2) >= 2 and l2[0] != ""


def test_log_tick_labels_negative_exponents():
    from pysuricata.render.svg_utils import nice_log_ticks_from_log10

    t, labels = nice_log_ticks_from_log10(-3.0, -1.0, max_ticks=4)
    # expect small decimals like 0.001, 0.01, 0.1
    assert any(l.startswith("0.") for l in labels)
