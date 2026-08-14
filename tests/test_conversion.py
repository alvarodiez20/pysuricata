import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
except ImportError:
    pl = None

from pysuricata.compute.processing.conversion import (
    ConversionStrategy,
    UnifiedConverter,
)


class TestUnifiedConverter:
    @pytest.fixture
    def converter(self):
        return UnifiedConverter(strategy=ConversionStrategy.SAFE)

    def test_unsupported_type(self, converter):
        res = converter.to_numeric([1, 2, 3])
        assert not res.success
        assert "Unsupported" in res.error or "failed" in res.error.lower()

        res = converter.to_boolean((1, 2, 3))
        assert not res.success

        res = converter.to_datetime_ns({"A": 1})
        assert not res.success

        res = converter.to_categorical_iter("A")
        assert not res.success

    def test_pandas_to_numeric(self, converter):
        s = pd.Series([1, 2, "3", "bad"])
        res = converter.to_numeric(s)
        assert res.success
        # "bad" gets coerced to NaN
        assert np.isnan(res.data[3])
        assert res.data[2] == 3.0

        # Fast path
        s2 = pd.Series([1.0, 2.0, 3.0])
        res2 = converter.to_numeric(s2)
        assert res2.success
        assert np.allclose(res2.data, [1.0, 2.0, 3.0])

        # Zero copy strategy
        z_converter = UnifiedConverter(strategy=ConversionStrategy.ZERO_COPY)
        res_z = z_converter.to_numeric(s2)
        assert res_z.success

        # Test FAST strategy on string data
        fast_converter = UnifiedConverter(strategy=ConversionStrategy.FAST)
        res_fast = fast_converter.to_numeric(pd.Series(["1", "2"]))
        assert res_fast.success

    def test_pandas_to_boolean(self, converter):
        s = pd.Series([True, False, None, 1, 0, "yes", "no", "bad"])
        res = converter.to_boolean(s)
        assert res.success
        expected = [True, False, None, True, False, True, False, None]
        assert res.data == expected

    def test_pandas_to_datetime_ns(self, converter):
        s = pd.Series(["2020-01-01", "2020-01-02", "bad"])
        res = converter.to_datetime_ns(s)
        assert res.success
        assert res.data[0] is not None
        # In pandas, the NAT_INT value or None might be returned depending on code path
        NAT_INT = -9223372036854775808
        assert res.data[2] is None or res.data[2] == NAT_INT

        # Already datetime
        s2 = pd.to_datetime(pd.Series(["2020-01-01"]))
        res2 = converter.to_datetime_ns(s2)
        assert res2.success
        assert res2.data[0] == res.data[0]

    def test_pandas_to_categorical_iter(self, converter):
        s = pd.Series(["A", "B", None])
        res = converter.to_categorical_iter(s)
        assert res.success
        data = list(res.data)
        assert data[0] == "A"
        assert data[1] == "B"
        assert data[2] in ("None", "", "nan")

    def test_cache(self, converter):
        assert converter.get_cache_stats()["cache_size"] == 0
        converter._conversion_cache["test"] = True
        assert converter.get_cache_stats()["cache_size"] == 1
        converter.clear_cache()
        assert converter.get_cache_stats()["cache_size"] == 0

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_to_numeric(self, converter):
        s = pl.Series(["1", "2", "bad"])
        res = converter.to_numeric(s)
        assert res.success
        assert np.isnan(res.data[2])
        assert res.data[0] == 1.0

        s2 = pl.Series([1.5, 2.5])
        res2 = converter.to_numeric(s2)
        assert res2.success
        assert np.allclose(res2.data, [1.5, 2.5])

        # Fast strategy
        fast_converter = UnifiedConverter(strategy=ConversionStrategy.FAST)
        res3 = fast_converter.to_numeric(pl.Series(["1"]))
        assert res3.success

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_to_boolean(self, converter):
        s = pl.Series([True, False, None])
        res = converter.to_boolean(s)
        assert res.success
        assert res.data == [True, False, None]

        s2 = pl.Series([1, 0, None])
        res2 = converter.to_boolean(s2)
        assert res2.success
        assert res2.data == [True, False, None]

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_to_datetime_ns(self, converter):
        # We need to test the cast fallback safely.
        # Using cast with strict=False will return null on bad strings.
        s = pl.Series(["2020-01-01", "bad"])
        res = converter.to_datetime_ns(s)
        assert res.success
        # In Polars cast to Datetime gives us something (or null), we then call to_list
        # But we really want to check logic execution.
        assert len(res.data) == 2

        # Already Datetime
        from datetime import datetime

        s2 = pl.Series([datetime(2020, 1, 1)])
        res2 = converter.to_datetime_ns(s2)
        assert res2.success

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_polars_to_categorical_iter(self, converter):
        s = pl.Series(["A", None, "C"])
        res = converter.to_categorical_iter(s)
        assert res.success
        data = list(res.data)
        assert data[0] == "A"
        assert data[1] == ""
        assert data[2] == "C"

        # Edge case: ints
        sint = pl.Series([1, 2])
        resint = converter.to_categorical_iter(sint)
        data_int = list(resint.data)
        assert data_int == ["1", "2"]
