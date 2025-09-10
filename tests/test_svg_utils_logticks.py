import math

from pysuricata.render.svg_utils import nice_log_ticks_from_log10, _format_pow10_label


def test_nice_log_ticks_from_log10_basic_span():
    # Span 10^-3 .. 10^3 with ~6 ticks should skip by 2 exponents
    ticks, labels = nice_log_ticks_from_log10(-3.0, 3.0, max_ticks=6)
    # Ticks should be monotonically increasing and within bounds
    assert ticks == sorted(ticks)
    assert ticks[0] >= -3.0 - 1e-9 and ticks[-1] <= 3.0 + 1e-9
    # Labels should correspond to 10**exp in human form
    allowed = {"0.001", "0.01", "0.1", "1", "10", "100", "1,000"}
    assert all(l in allowed for l in labels)
    # Expect at least the extremes in labels
    assert labels[0] in {"0.001", "0.01"}  # depending on step selection
    assert labels[-1] in {"100", "1,000"}


def test_format_pow10_label_edges():
    # Small negative exponents -> decimals
    assert _format_pow10_label(-3) == "0.001"
    assert _format_pow10_label(-1) == "0.1"
    # Small positive exponents -> integers with grouping
    assert _format_pow10_label(3) in {"1,000", "1000"}
    # Larger magnitudes -> scientific notation
    assert _format_pow10_label(7) == "1e7"
    assert _format_pow10_label(-7) == "1e-7"
