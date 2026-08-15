"""The datetime accumulator's vectorised path.

Four per-row Python loops became array arithmetic. The element-wise path still
exists for object input, so the first thing to pin is that the two agree — then
the calendar decomposition itself, which is where the interesting failures live:
it must floor for pre-1970 instants, and it must not overflow at the bottom of
the representable window, where casting datetime64[ns] to a coarser unit
reports 1677-09-21 as a *positive* day number.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators.datetime import (
    _NS_MAX,
    _NS_MIN,
    DatetimeAccumulator,
    _as_ns_int64,
)


def _ns(*stamps: str) -> np.ndarray:
    return np.array([pd.Timestamp(s).value for s in stamps], dtype=np.int64)


def _fold(values, n_chunks: int = 1) -> DatetimeAccumulator:
    acc = DatetimeAccumulator("d")
    for chunk in np.array_split(np.asarray(values), n_chunks):
        acc.update(chunk)
    return acc


class TestNsCoercion:
    def test_int64_array_passes_through(self):
        arr = _ns("2020-01-01")
        assert _as_ns_int64(arr).dtype == np.int64

    def test_datetime64_is_viewed_not_reparsed(self):
        arr = pd.date_range("2020-01-01", periods=3, freq="D").to_numpy()
        assert (
            _as_ns_int64(arr).tolist()
            == arr.astype("datetime64[ns]").view("int64").tolist()
        )

    def test_a_plain_list_of_ints_takes_the_fast_path(self):
        assert _as_ns_int64([1, 2, 3]) is not None

    def test_a_list_holding_none_falls_back(self):
        assert _as_ns_int64([1, None, 3]) is None

    def test_an_object_array_falls_back(self):
        assert _as_ns_int64(np.array([1, "x"], dtype=object)) is None

    def test_nan_becomes_the_missing_sentinel_not_garbage(self):
        out = _as_ns_int64(np.array([1.0, np.nan], dtype=np.float64))
        assert out[0] == 1
        assert out[1] < _NS_MIN  # rejected by the window, not an arbitrary int


class TestVectorisedMatchesElementwise:
    """Same column, both routes, identical summary."""

    STAMPS = (
        "1700-01-01 07:00:00",
        "1880-06-15 03:17:00",
        "1969-12-31 23:59:59",
        "1970-01-01 00:00:00",
        "2020-02-29 13:45:00",
        "2026-08-15 18:30:00",
    )

    def test_counts_bounds_and_patterns_agree(self):
        ns = _ns(*self.STAMPS)
        fast = _fold(ns)
        slow = DatetimeAccumulator("d")
        slow._process_timestamps_vectorized(np.array(ns.tolist(), dtype=object))

        assert (fast.count, fast.missing) == (slow.count, slow.missing)
        assert (fast.min_ts, fast.max_ts) == (slow.min_ts, slow.max_ts)
        assert fast.by_hour == slow.by_hour
        assert fast.by_dow == slow.by_dow
        assert fast.by_month == slow.by_month
        assert fast.by_year == slow.by_year

    def test_the_distinct_estimate_agrees(self):
        """add_many and repeated add must hash a 64-bit int identically."""
        ns = np.arange(50_000, dtype=np.int64) * 1_000_000_000
        fast = _fold(ns)
        slow = DatetimeAccumulator("d")
        slow._process_timestamps_vectorized(np.array(ns.tolist(), dtype=object))
        assert fast.unique_est == slow.unique_est


class TestCalendarFields:
    @pytest.mark.parametrize(
        "stamp",
        [
            "1677-09-22 00:00:00",  # one day inside the low edge
            "1700-01-01 07:00:00",
            "1880-06-15 03:17:00",
            "1969-12-31 23:59:59",
            "1970-01-01 00:00:00",
            "2020-02-29 13:45:00",
            "2262-04-11 00:00:00",  # inside the high edge
        ],
    )
    def test_fields_match_pandas_in_utc(self, stamp):
        ts = pd.Timestamp(stamp)
        acc = _fold(_ns(stamp))
        assert acc.by_hour[ts.hour] == 1
        assert acc.by_dow[ts.dayofweek] == 1
        assert acc.by_month[ts.month - 1] == 1
        assert acc.by_year == {ts.year: 1}

    def test_the_window_edges_do_not_overflow(self):
        """datetime64 casts report the low edge as a positive day number.

        That produced hour 46 and crashed np.bincount, so the decomposition
        divides in integers instead.
        """
        acc = _fold(np.array([_NS_MIN, _NS_MAX], dtype=np.int64))
        assert acc.count == 2
        assert sum(acc.by_hour) == 2
        assert sum(acc.by_dow) == 2
        assert sum(acc.by_month) == 2
        assert acc.by_year == {1677: 1, 2262: 1}

    def test_every_hour_and_weekday_is_reachable(self):
        acc = _fold(
            pd.date_range("2020-01-01", periods=24 * 7, freq="h").to_numpy(), n_chunks=5
        )
        assert acc.by_hour == [7] * 24
        assert acc.by_dow == [24] * 7

    def test_tallies_are_utc_not_machine_local(self):
        """datetime.fromtimestamp() used the machine's zone, so the same data
        gave different hour histograms in London and Tokyo."""
        stamp = _ns("2020-06-15 23:30:00")
        original = os.environ.get("TZ")
        try:
            observed = []
            for zone in ("UTC", "Asia/Tokyo", "America/Los_Angeles"):
                os.environ["TZ"] = zone
                time.tzset()
                observed.append(_fold(stamp).by_hour)
            assert observed[1] == observed[0]
            assert observed[2] == observed[0]
            assert observed[0][23] == 1
        finally:
            if original is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original
            time.tzset()


class TestMissingAndInvalid:
    def test_the_nat_sentinel_counts_as_missing(self):
        arr = np.array([pd.Timestamp("2020-01-01").value, np.iinfo(np.int64).min])
        acc = _fold(arr)
        assert (acc.count, acc.missing) == (1, 1)

    def test_a_null_polars_style_column_is_all_missing(self):
        acc = _fold(np.full(100, np.iinfo(np.int64).min, dtype=np.int64))
        assert (acc.count, acc.missing) == (0, 100)
        assert acc.min_ts is None

    def test_nan_in_a_float_column_counts_as_missing(self):
        acc = _fold(np.array([1e18, np.nan, 2e18], dtype=np.float64))
        assert (acc.count, acc.missing) == (2, 1)

    def test_rejected_values_are_not_lost_on_the_object_path(self):
        acc = DatetimeAccumulator("d")
        acc.update([pd.Timestamp("2020-01-01").value, None, None])
        assert (acc.count, acc.missing) == (1, 2)


class TestChunkInvariance:
    VALUES = pd.date_range("1960-01-01", periods=20_000, freq="97min").to_numpy()

    @pytest.mark.parametrize("n_chunks", [2, 7, 113])
    def test_the_summary_does_not_depend_on_chunking(self, n_chunks):
        one = _fold(self.VALUES, 1)
        many = _fold(self.VALUES, n_chunks)
        assert (one.count, one.missing) == (many.count, many.missing)
        assert (one.min_ts, one.max_ts) == (many.min_ts, many.max_ts)
        assert one.by_hour == many.by_hour
        assert one.by_dow == many.by_dow
        assert one.by_month == many.by_month
        assert one.by_year == many.by_year
        assert one.unique_est == many.unique_est


class TestEndToEnd:
    def test_a_historical_column_profiles_with_the_right_span(self):
        from pysuricata import summarize

        df = pd.DataFrame(
            {"born": pd.date_range("1880-01-01", periods=5_000, freq="10D")}
        )
        col = summarize(df)["columns"]["born"]
        assert col["count"] == 5_000
        assert col["missing"] == 0
