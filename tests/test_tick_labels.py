"""A y-axis count label is at most four glyphs. Always, not usually.

The histogram's y gutter is a fixed 44px: 27px of 11px mono, a 5px tick and
8px of air. It is fixed so that the plot's left edge does not move between
columns and bars line up down the page. A five-glyph label either overflows it
or forces the gutter to breathe, and a gutter that breathes loses the
alignment the fixed one buys.

The old formatter *preferred* short and did not guarantee it: `12,500` came
out as six glyphs and `12.5M` as five. "Prefers" is the failure mode this file
exists to catch, which is why the bound is checked over a swept range rather
than over a handful of values someone thought of. Every escape found so far
was a value nobody would have listed:

* `1,000` -- the thousands separator is a glyph.
* `12,500` -- scales to 12.5, and one decimal at two digits is five glyphs.
* `999,999` -- scales to 999.999 in the K band, which *rounds to 1000*, so the
  band itself has to be promoted.
"""

from __future__ import annotations

import pytest

from pysuricata.render.histogram_svg import SVGHistogramRenderer

MAX_GLYPHS = 4


@pytest.fixture(scope="module")
def label():
    renderer = SVGHistogramRenderer()

    def format_count(value: float) -> str:
        return renderer._format_tick_label_standardized(value, is_count=True)

    return format_count


def _sweep() -> list[int]:
    """Counts across every magnitude, dense near each band edge.

    A count comes from `nice_ticks`, so the values that actually reach this
    are round-ish -- but the boundaries are where the bound breaks, and the
    boundaries are exactly what a hand-written list misses.
    """
    values: set[int] = set(range(0, 1200))
    for exponent in range(3, 19):
        base = 10**exponent
        for step in (1, 2, 5, 9):
            for delta in (-2, -1, 0, 1, 2):
                values.add(max(0, base * step + delta))
                values.add(max(0, base * step // 2 + delta))
    # Half-way cases, where rounding decides the glyph count.
    values.update({12_500, 125_000, 1_250_000, 999_999, 9_999, 10_000, 99_999})
    return sorted(values)


class TestTheBoundHolds:
    def test_every_count_in_the_sweep_fits(self, label):
        too_long = {
            value: text for value in _sweep() if len(text := label(value)) > MAX_GLYPHS
        }
        assert not too_long, (
            f"{len(too_long)} counts render wider than {MAX_GLYPHS} glyphs, "
            f"first few: {dict(list(too_long.items())[:8])}"
        )

    @pytest.mark.parametrize(
        "value",
        [1_000, 12_500, 999_999, 12_500_000, 999_999_999, 9_200_000_000_000_000_000],
        ids=[
            "comma",
            "half-scales",
            "rounds-up-a-band",
            "millions",
            "billions",
            "int64-max",
        ],
    )
    def test_the_known_escapes_are_closed(self, value, label):
        assert len(label(value)) <= MAX_GLYPHS, f"{value} -> {label(value)!r}"

    def test_a_negative_count_is_still_bounded(self, label):
        """Counts are non-negative in practice; the formatter is shared, so it
        should not produce a six-glyph label if one ever arrives."""
        assert len(label(-12_500)) <= MAX_GLYPHS + 1  # the sign is extra


class TestItStillSaysSomethingTrue:
    """Bounded and wrong would be worse than unbounded."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (999, "999"),
            (1_000, "1000"),
            (9_999, "9999"),
            (10_000, "10K"),
            (999_999, "1M"),
            (1_200_000, "1.2M"),
            (1_000_000_000, "1B"),
            (1_500_000_000_000, "1.5T"),
        ],
    )
    def test_the_value_reads_correctly(self, value, expected, label):
        assert label(value) == expected

    def test_it_is_monotonic(self, label):
        """A bigger count never prints as a smaller quantity.

        Abbreviation loses precision on purpose. Losing *order* would be a
        different thing, and it is the way a banding bug shows up.
        """
        import re

        units = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "P": 1e15, "E": 1e18}

        def magnitude(text: str) -> float:
            match = re.fullmatch(r"(-?[\d.]+)([KMBTPE]?)", text)
            assert match, f"unparseable label {text!r}"
            return float(match.group(1)) * units[match.group(2)]

        previous = -1.0
        for value in _sweep():
            current = magnitude(label(value))
            assert current >= previous - 1e-9, (
                f"{value} renders {label(value)!r}, which reads smaller than "
                f"the previous label"
            )
            previous = current


class TestTheXAxisIsNotAffected:
    """An x label is a data value and lives in a caption row as wide as the
    plot, so it keeps its precision."""

    def test_a_data_value_may_be_longer_than_four_glyphs(self):
        renderer = SVGHistogramRenderer()
        text = renderer._format_tick_label_standardized(1234.5, is_count=False)
        assert text == "1,234.5"

    def test_a_small_fraction_keeps_three_decimals(self):
        renderer = SVGHistogramRenderer()
        assert (
            renderer._format_tick_label_standardized(0.125, is_count=False) == "0.125"
        )
