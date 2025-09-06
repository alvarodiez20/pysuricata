import math

from pysuricata.render.format_utils import human_bytes, fmt_num, fmt_compact
from pysuricata.render.svg_utils import safe_col_id, nice_ticks, fmt_tick, svg_empty


def test_human_bytes_basic():
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(1023) == "1,023.0 B"
    assert human_bytes(1024) == "1.0 KB"
    assert human_bytes(1536) == "1.5 KB"


def test_fmt_num_variants():
    assert fmt_num(1234.5678) == "1,235" or fmt_num(1234.5678) == "1,235"  # locale-insensitive
    assert fmt_num(float("nan")) == "NaN"
    assert fmt_num(float("inf")) == "NaN"
    assert fmt_num(None) == "—"


def test_fmt_compact_variants():
    assert fmt_compact(1234.5678) in {"1235", "1.235e+03", "1.235e+03"}
    assert fmt_compact(None) == "—"
    assert fmt_compact(float("nan")) == "—"
    assert fmt_compact("42.0") in {"42", "42.0"}


def test_safe_col_id_sanitizes():
    assert safe_col_id("Amount ($)").startswith("col_")
    assert "(" not in safe_col_id("A(B)")
    assert ":" not in safe_col_id("a:b")


def test_svg_empty_shell():
    out = svg_empty("hist-svg", 420, 160)
    assert out.startswith('<svg class="hist-svg"')
    assert 'width="420"' in out and 'height="160"' in out


def test_nice_ticks_monotonic_and_equal_bounds():
    ticks, step = nice_ticks(0, 100, 5)
    assert step > 0
    assert ticks[0] <= 0 and ticks[-1] >= 100
    ticks2, step2 = nice_ticks(5, 5, 5)
    assert step2 > 0
    assert len(ticks2) >= 2


def test_fmt_tick_thresholds():
    # coarse step -> integers
    assert fmt_tick(999.4, 1.0) in {"999", "999"}
    assert fmt_tick(1000.0, 1.0) in {"1,000", "1000"}
    # fractional steps
    assert fmt_tick(0.1234, 0.1) == "0.1" or fmt_tick(0.1234, 0.1) == "0.1"
    s = fmt_tick(0.0099, 0.01)
    assert s in {"0.01", "0.0"}

