"""A polars column of timestamp *strings* must profile like the pandas one.

#214. `Series.cast()` from a String silently yields nulls instead of raising,
so a `strict=False` cast reports success while producing nothing — and every
`except Exception` fallback written around one is unreachable. Measured on
polars 1.34:

| input                   | `cast(Date)` | `cast(Datetime)` | `str.to_datetime` |
|-------------------------|--------------|------------------|-------------------|
| `2020-01-01`            | ok           | **all null**     | ok                |
| `2020-01-01 12:00:00`   | **all null** | **all null**     | ok                |
| `2020-01-01T12:00:00`   | **all null** | ok               | ok                |

The conversion path tried `Date` first and kept it; inference tried both and
took the first that looked good. So the two disagreed about which strings are
datetimes, and where they disagreed a column was typed `datetime` and then
converted to nothing:

    pandas: type=datetime  count=200  missing=0
    polars: type=datetime  count=0    missing=200

200 valid timestamps reported as entirely missing, with the column still
labelled `datetime` so nothing looked structurally wrong.

Two things made this invisible. It needs **both backends** to see at all — one
backend alone is self-consistent and looks right. And the existing polars
fixtures pass already-typed `pl.Datetime` columns, which take the fast path and
never reach the cast.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pysuricata import summarize

pl = pytest.importorskip("polars")


#: The three ways a datetime arrives as text. Each hit a different cell of the
#: table above, which is why one representative string would not have done.
SHAPES = {
    "date-only": [f"2020-01-{day:02d}" for day in range(1, 21)] * 10,
    "space-separated": [
        f"2020-01-{day:02d} {hour:02d}:00:00"
        for day in range(1, 21)
        for hour in range(10)
    ],
    "iso-t": [
        f"2020-01-{day:02d}T{hour:02d}:00:00"
        for day in range(1, 21)
        for hour in range(10)
    ],
}

#: Fields that must not depend on which library parsed the text.
PARITY_FIELDS = ("type", "count", "missing", "unique_est", "min_ts", "max_ts")


@pytest.mark.parametrize("shape", sorted(SHAPES))
class TestBothBackendsAgree:
    def test_the_column_is_a_datetime_with_nothing_missing(self, shape):
        """The direct statement of the bug: 200 valid timestamps, none missing."""
        values = SHAPES[shape]

        column = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]

        assert column["type"] == "datetime"
        assert column["count"] == len(values)
        assert column["missing"] == 0, (
            f"{column['missing']} of {len(values)} valid timestamps were read as "
            "missing -- the string was never parsed"
        )

    def test_polars_matches_pandas_field_for_field(self, shape):
        """Parity, not just plausibility.

        A backend can be self-consistently wrong; only the comparison catches
        that, and this whole class of bug is invisible to a suite that
        exercises one backend at a time.
        """
        values = SHAPES[shape]

        from_pandas = summarize(pd.DataFrame({"t": values}), seed=0)["columns"]["t"]
        from_polars = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]

        divergent = {
            field: (from_pandas.get(field), from_polars.get(field))
            for field in PARITY_FIELDS
            if from_pandas.get(field) != from_polars.get(field)
        }
        assert not divergent, f"pandas vs polars: {divergent}"

    def test_a_string_column_matches_an_already_typed_one(self, shape):
        """The text and the parsed value describe the same instants.

        The existing polars fixtures pass typed `pl.Datetime` columns, which
        take the fast path -- so they agreed with each other and with nothing
        else. This is the comparison they were missing.
        """
        values = SHAPES[shape]
        typed = pl.Series(values).str.to_datetime()

        as_text = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]
        as_typed = summarize(pl.DataFrame({"t": typed}), seed=0)["columns"]["t"]

        for field in ("count", "missing", "min_ts", "max_ts"):
            assert as_text.get(field) == as_typed.get(field), field


class TestItDoesNotOverreach:
    """Parsing more strings must not mean parsing strings that are not dates."""

    def test_plain_words_are_not_datetimes(self):
        values = ["alpha", "beta", "gamma", "delta"] * 50

        column = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]

        assert column["type"] != "datetime"

    def test_identifiers_are_not_datetimes(self):
        """The polars-side neighbour of the `T1` misparse fixed in #203 -- and
        `T` is the ISO 8601 time designator, so this is the shape most likely
        to be over-parsed."""
        values = [f"T{i % 681}" for i in range(200)]

        column = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]

        assert column["type"] != "datetime"

    def test_a_mostly_unparseable_column_is_not_a_datetime(self):
        """One parseable value in twenty must not carry the column."""
        values = ["2020-01-01"] + ["not a date"] * 19

        column = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]

        assert column["type"] != "datetime"

    def test_unparseable_rows_inside_a_date_column_count_as_missing(self):
        values = [f"2020-01-{day:02d}" for day in range(1, 21)] * 10 + ["nope"] * 5

        column = summarize(pl.DataFrame({"t": values}), seed=0)["columns"]["t"]

        assert column["type"] == "datetime"
        assert column["missing"] == 5


class TestNoDeprecatedPolarsCasts:
    """The half this was filed as. Casting String -> Date/Datetime is
    deprecated from polars 1.43 and **removed in Polars 2.0**, and
    `pyproject.toml` pins `polars>=1.34.0` with no upper bound -- so an upgrade
    past 2.0 would have taken these paths out from under the library.

    Silent on polars 1.34 (nothing is deprecated yet), load-bearing above it.
    """

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_profiling_a_string_column_raises_no_deprecation_warning(self, shape):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            summarize(pl.DataFrame({"t": SHAPES[shape]}), seed=0)

    def test_the_helper_is_shared_by_the_sniff_and_the_conversion(self):
        """Inference and conversion disagreeing is the bug, so they must not be
        able to disagree: one function decides what a string datetime is."""
        from pysuricata.compute.consume_polars import polars_string_to_datetime as a
        from pysuricata.compute.processing.inference import (
            polars_string_to_datetime as b,
        )

        assert a is b
