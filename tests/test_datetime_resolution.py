"""Datetime columns are not always stored at nanosecond resolution.

`_to_datetime_ns_array_pandas` promised nanoseconds in its name and delivered
whatever unit the column happened to use, because it cast straight to int64:

    return ds.astype("int64", copy=False).to_numpy()

That was correct for as long as every datetime was `datetime64[ns]`, which was
true by default under pandas 2 and stopped being true under pandas 3, where
`pd.date_range(...)` returns `datetime64[us]`. The same cast then returned
*microseconds*, and every datetime statistic downstream came out a factor of
1,000 wrong while still looking entirely plausible -- a 2020 timestamp read as
1970, so a freshness check reported data 18,264 days old.

pandas 2 is not innocent here, which is why these tests do not skip on it:
non-nanosecond dtypes are constructible on pandas 2 as well, and arrive on
their own from parquet and pyarrow. The default hid the bug; it did not prevent
it.

Nanoseconds span 1677-09-21 to 2262-04-11. A coarser column can hold dates
outside that, and the honest answer for those is NaT -- the sentinel the
accumulator's validity window already rejects -- rather than an int64 wrapped
into a plausible wrong date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.compute.consume import (
    _NAT_INT64,
    _as_int64_nanoseconds,
    _to_datetime_ns_array_pandas,
)

#: Every resolution pandas can store a datetime at, with its scale to ns.
RESOLUTIONS = [("ns", 1), ("us", 1_000), ("ms", 1_000_000), ("s", 1_000_000_000)]


class TestEveryResolutionArrivesAsNanoseconds:
    @pytest.mark.parametrize("unit,_scale", RESOLUTIONS)
    def test_a_known_timestamp_round_trips(self, unit, _scale):
        """2020-01-01T00:00:00Z is 1_577_836_800_000_000_000 ns, whatever the
        column was stored at."""
        expected = 1_577_836_800_000_000_000
        s = pd.Series(pd.to_datetime(["2020-01-01"])).astype(f"datetime64[{unit}]")

        arr = _to_datetime_ns_array_pandas(s)

        assert arr.dtype == np.int64
        assert arr[0] == expected, (
            f"a datetime64[{unit}] column produced {arr[0]}, not {expected} -- "
            f"off by a factor of {expected / arr[0] if arr[0] else float('nan'):g}"
        )

    @pytest.mark.parametrize("unit,_scale", RESOLUTIONS)
    def test_every_resolution_agrees_with_every_other(self, unit, _scale):
        """The same instants stored differently must profile identically.

        This is the chunking invariant's cousin: a representation detail of the
        input is not allowed to change a statistic.
        """
        stamps = pd.to_datetime(
            ["2020-01-01 00:00:00", "2020-06-15 12:00:00", "2021-03-09 06:30:00"]
        )
        reference = _to_datetime_ns_array_pandas(
            pd.Series(stamps).astype("datetime64[ns]")
        )

        actual = _to_datetime_ns_array_pandas(
            pd.Series(stamps).astype(f"datetime64[{unit}]")
        )

        np.testing.assert_array_equal(actual, reference)


class TestTheSentinelAndTheEdges:
    def test_nat_stays_nat_rather_than_being_scaled(self):
        """NaT is int64 min. Multiplying it by 1,000 is undefined and would
        land somewhere real."""
        s = pd.Series([pd.Timestamp("2020-01-01"), pd.NaT]).astype("datetime64[us]")

        arr = _to_datetime_ns_array_pandas(s)

        assert arr[0] == 1_577_836_800_000_000_000
        assert arr[1] == _NAT_INT64

    def test_a_date_nanoseconds_cannot_represent_becomes_nat(self):
        """Year 3000 fits in `datetime64[us]` and not in `datetime64[ns]`.

        Saturating to the missing sentinel is the honest failure; wrapping it
        into a plausible date is not.
        """
        s = pd.Series([pd.Timestamp("3000-01-01"), pd.Timestamp("2020-01-01")]).astype(
            "datetime64[us]"
        )

        arr = _to_datetime_ns_array_pandas(s)

        assert arr[0] == _NAT_INT64
        assert arr[1] == 1_577_836_800_000_000_000

    def test_the_scaling_helper_saturates_rather_than_overflowing(self):
        """Directly, so the boundary is pinned without depending on what
        pandas will accept into a Series."""
        huge = np.array([np.iinfo(np.int64).max // 2], dtype=np.int64)

        out = _as_int64_nanoseconds(huge, "us")

        assert out[0] == _NAT_INT64, "an unrepresentable value must not wrap"

    def test_nanoseconds_are_returned_untouched(self):
        values = np.array([1_577_836_800_000_000_000, _NAT_INT64], dtype=np.int64)

        np.testing.assert_array_equal(_as_int64_nanoseconds(values, "ns"), values)


class TestItReachesTheProfile:
    """The unit tests above pin the conversion; this pins that the conversion
    is the one `summarize()` actually uses."""

    @pytest.mark.parametrize("unit,_scale", RESOLUTIONS)
    def test_the_reported_min_and_max_do_not_depend_on_the_stored_unit(
        self, unit, _scale
    ):
        stamps = pd.date_range("2020-01-01", periods=200, freq="h")
        frame = pd.DataFrame({"when": pd.Series(stamps).astype(f"datetime64[{unit}]")})

        column = summarize(frame, seed=0)["columns"]["when"]

        # 2020-01-01T00:00Z and 200 hours later, in whatever unit the payload
        # reports; the point is that it is the same number for every input unit.
        reference = summarize(
            pd.DataFrame({"when": pd.Series(stamps).astype("datetime64[ns]")}), seed=0
        )["columns"]["when"]
        assert column["min_ts"] == reference["min_ts"]
        assert column["max_ts"] == reference["max_ts"]
