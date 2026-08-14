"""Historical dates must not be silently counted as missing.

The validity window's lower bound was -2e18 ns, which is 1906-05-13. Birthdates
and historical records before it were reclassified as nulls, so a column of
19th-century dates looked almost entirely missing rather than old.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.accumulators.datetime import _NS_MAX, _NS_MIN, DatetimeAccumulator


class TestHistoricalDates:
    @pytest.mark.parametrize(
        "dates",
        [
            ["1800-01-01", "1850-06-15", "1899-12-31"],
            ["1700-01-01", "1690-05-05", "1680-01-01"],
            ["1678-01-01", "1679-06-30"],
            ["1900-01-01", "1905-01-01", "1906-01-01"],
        ],
    )
    def test_pre_1906_dates_are_present_not_missing(self, dates):
        df = pd.DataFrame({"d": pd.to_datetime(dates)})
        col = summarize(df)["columns"]["d"]
        assert col["count"] == len(dates)
        assert col["missing"] == 0

    def test_modern_dates_are_unaffected(self):
        dates = ["2020-01-01", "2021-06-15", "2262-01-01"]
        df = pd.DataFrame({"d": pd.to_datetime(dates)})
        col = summarize(df)["columns"]["d"]
        assert col["count"] == 3
        assert col["missing"] == 0

    def test_a_mixed_historical_and_modern_column(self):
        dates = ["1850-01-01", "2020-01-01", "1700-01-01", "1999-12-31"]
        df = pd.DataFrame({"d": pd.to_datetime(dates)})
        col = summarize(df)["columns"]["d"]
        assert col["count"] == 4
        assert col["missing"] == 0

    def test_genuine_nulls_are_still_missing(self):
        df = pd.DataFrame({"d": pd.to_datetime(["1850-01-01", None, "2020-01-01"])})
        col = summarize(df)["columns"]["d"]
        assert col["count"] == 2
        assert col["missing"] == 1


class TestValidityWindow:
    def test_window_matches_the_representable_range(self):
        """1677-09-21 to 2262-04-11, the int64 datetime64[ns] limits."""
        assert pd.Timestamp(_NS_MIN).year == 1677
        assert pd.Timestamp(_NS_MAX).year == 2262

    def test_nat_sentinel_is_excluded(self):
        """pandas reserves int64 min for NaT, so it must not be a valid value."""
        assert _NS_MIN == int(np.iinfo(np.int64).min) + 1

    def test_accumulator_accepts_the_window_edges(self):
        acc = DatetimeAccumulator("d")
        acc.update([_NS_MIN, 0, _NS_MAX])
        assert acc.count == 3
        assert acc.missing == 0

    def test_accumulator_rejects_values_outside_the_window(self):
        acc = DatetimeAccumulator("d")
        acc.update([int(np.iinfo(np.int64).min), None])
        assert acc.count == 0
        assert acc.missing == 2
