import pandas as pd
import pytest

try:
    import polars as pl
except ImportError:
    pl = None

from pysuricata.compute.core.types import ColumnKinds
from pysuricata.compute.processing.inference import (
    InferenceStrategy,
    UnifiedTypeInferrer,
    should_reclassify_numeric_as_boolean,
    should_reclassify_numeric_as_categorical,
)


class DummyConfig:
    def __init__(self):
        self.enable_auto_boolean_detection = True
        self.boolean_detection_min_samples = 5
        self.boolean_detection_max_zero_ratio = 0.9
        self.boolean_detection_require_name_pattern = False


class TestUnifiedTypeInferrer:
    @pytest.fixture
    def inferrer(self):
        return UnifiedTypeInferrer(strategy=InferenceStrategy.BALANCED)

    def test_infer_kinds_unsupported(self, inferrer):
        res = inferrer.infer_kinds([1, 2, 3])
        assert not res.success
        assert "Unsupported" in res.error or "failed" in str(res.error).lower()

    def test_infer_series_type_unsupported(self, inferrer):
        res = inferrer.infer_series_type([1, 2, 3])
        assert not res.success
        assert "Unsupported" in res.error or "failed" in str(res.error).lower()

    def test_infer_pandas_dataframe(self, inferrer):
        df = pd.DataFrame(
            {
                "num": [1, 2, 3],
                "cat": ["A", "B", "C"],
                "bools": [True, False, True],
                "dates": pd.to_datetime(["2020", "2021", "2022"]),
            }
        )
        res = inferrer.infer_kinds(df)
        assert res.success
        assert isinstance(res.data, ColumnKinds)
        assert "num" in res.data.numeric
        assert "cat" in res.data.categorical
        assert "bools" in res.data.boolean
        assert "dates" in res.data.datetime
        assert res.metrics["confidence"] > 0.0

    def test_infer_pandas_series_fast_paths(self, inferrer):
        s_num = pd.Series([1, 2, 3])
        assert inferrer.infer_series_type(s_num).data == "numeric"

        s_bool = pd.Series([True, False])
        assert inferrer.infer_series_type(s_bool).data == "boolean"

        s_dt = pd.to_datetime(pd.Series(["2020-01-01"]))
        assert inferrer.infer_series_type(s_dt).data == "datetime"

        s_td = pd.to_timedelta(pd.Series(["1 days", "2 days"]))
        assert inferrer.infer_series_type(s_td).data == "numeric"

    def test_infer_pandas_series_sample_based(self, inferrer):
        # A string series that looks entirely like dates
        s_dates_str = pd.Series(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
        )
        assert inferrer.infer_series_type(s_dates_str).data == "datetime"

        # A string series that looks like numbers
        s_num_str = pd.Series(["1.1", "2.2", "3.3", "4.4"])
        assert inferrer.infer_series_type(s_num_str).data == "numeric"

        # A string series that is categorically boolean (needs AGGRESSIVE strategy)
        agg_inferrer = UnifiedTypeInferrer(strategy=InferenceStrategy.AGGRESSIVE)
        s_bool_str = pd.Series(["true", "false", "yes", "no"])
        assert agg_inferrer.infer_series_type(s_bool_str).data == "boolean"

        # Fast strategy bypasses sample inference and defaults to categorical for generic objects
        fast_inferrer = UnifiedTypeInferrer(strategy=InferenceStrategy.FAST)
        assert fast_inferrer.infer_series_type(s_bool_str).data == "categorical"

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_infer_polars_dataframe(self, inferrer):
        df = pl.DataFrame(
            {
                "num": [1, 2, 3],
                "cat": ["A", "B", "C"],
                "bools": [True, False, True],
            }
        )
        res = inferrer.infer_kinds(df)
        assert res.success
        assert "num" in res.data.numeric
        assert "cat" in res.data.categorical
        assert "bools" in res.data.boolean

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_infer_polars_series_fast_paths(self, inferrer):
        assert inferrer.infer_series_type(pl.Series([1, 2])).data == "numeric"
        assert inferrer.infer_series_type(pl.Series([True, False])).data == "boolean"
        assert inferrer.infer_series_type(pl.Series(["A", "B"])).data == "categorical"

        from datetime import datetime

        assert (
            inferrer.infer_series_type(pl.Series([datetime(2020, 1, 1)])).data
            == "datetime"
        )

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_infer_polars_series_sample_based(self, inferrer):
        s_dates_str = pl.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        assert inferrer.infer_series_type(s_dates_str).data == "datetime"

        s_num_str = pl.Series(["1.5", "2.5", "3.5"])
        assert inferrer.infer_series_type(s_num_str).data == "numeric"

        agg_inferrer = UnifiedTypeInferrer(strategy=InferenceStrategy.AGGRESSIVE)
        s_bool_str = pl.Series(["true", "false", "true", "false"])
        assert agg_inferrer.infer_series_type(s_bool_str).data == "boolean"

        fast_inferrer = UnifiedTypeInferrer(strategy=InferenceStrategy.FAST)
        assert fast_inferrer.infer_series_type(s_bool_str).data == "categorical"

    def test_cache(self, inferrer):
        assert inferrer.get_cache_stats()["cache_size"] == 0
        inferrer._inference_cache["test"] = True
        assert inferrer.get_cache_stats()["cache_size"] == 1
        inferrer.clear_cache()
        assert inferrer.get_cache_stats()["cache_size"] == 0


class TestReclassificationRules:
    def test_should_reclassify_numeric_as_categorical(self):
        # A cardinality ceiling, not a ratio: the answer must not depend on how
        # many rows there are. See tests/test_public_surface.py for the
        # end-to-end version of this property.
        assert (
            should_reclassify_numeric_as_categorical(unique_count=5, total_count=100)
            is True
        )
        assert (
            should_reclassify_numeric_as_categorical(unique_count=50, total_count=10000)
            is True
        )
        assert (
            should_reclassify_numeric_as_categorical(unique_count=50, total_count=100)
            is True
        )
        assert (
            should_reclassify_numeric_as_categorical(unique_count=67, total_count=100)
            is False
        )
        assert (
            should_reclassify_numeric_as_categorical(unique_count=67, total_count=20000)
            is False
        )
        assert (
            should_reclassify_numeric_as_categorical(unique_count=0, total_count=0)
            is False
        )

    def test_should_reclassify_numeric_as_boolean_pandas(self):
        config = DummyConfig()

        # Valid boolean
        s_valid = pd.Series([0, 1, 0, 1, 1, 0], name="flag_active")
        assert should_reclassify_numeric_as_boolean(s_valid, config) is True

        # Too few samples
        s_few = pd.Series([0, 1], name="is_small")
        assert should_reclassify_numeric_as_boolean(s_few, config) is False

        # Not just 0 and 1
        s_other = pd.Series([0, 1, 2, 1, 0, 0], name="is_other")
        assert should_reclassify_numeric_as_boolean(s_other, config) is False

        # Too many zeros (unbalanced)
        config.boolean_detection_max_zero_ratio = 0.5
        s_zeros = pd.Series([0, 0, 0, 0, 0, 1], name="is_unbalanced")
        assert should_reclassify_numeric_as_boolean(s_zeros, config) is False

        # Name pattern required
        config.boolean_detection_max_zero_ratio = 0.9
        config.boolean_detection_require_name_pattern = True
        s_bad_name = pd.Series([0, 1, 0, 1, 1, 0], name="some_random_col")
        assert should_reclassify_numeric_as_boolean(s_bad_name, config) is False

        s_good_name = pd.Series([0, 1, 0, 1, 1, 0], name="is_active")
        assert should_reclassify_numeric_as_boolean(s_good_name, config) is True

    @pytest.mark.skipif(pl is None, reason="polars not installed")
    def test_should_reclassify_numeric_as_boolean_polars(self):
        config = DummyConfig()
        s_valid = pl.Series("flag_active", [0, 1, 0, 1, 1, 0])
        assert should_reclassify_numeric_as_boolean(s_valid, config) is True

        s_other = pl.Series("other", [0, 1, 2, 0, 1, 0])
        assert should_reclassify_numeric_as_boolean(s_other, config) is False


try:
    import pyarrow as pa
except ImportError:
    pa = None


class TestArrowBackedDtypes:
    """Coverage for the pyarrow-backed (ArrowDtype) inference branch."""

    @pytest.fixture
    def inferrer(self):
        return UnifiedTypeInferrer(strategy=InferenceStrategy.BALANCED)

    @pytest.mark.skipif(pa is None, reason="pyarrow not installed")
    @pytest.mark.parametrize(
        ("arrow_type_name", "values", "expected"),
        [
            ("bool_", [True, False], "boolean"),
            ("int64", [1, 2], "numeric"),
            ("uint8", [1, 2], "numeric"),
            ("float32", [1.0, 2.0], "numeric"),
            ("string", ["a", "b"], "categorical"),
        ],
    )
    def test_arrow_dtype_classification(
        self, inferrer, arrow_type_name, values, expected
    ):
        # Build the dtype explicitly: the "<type>[pyarrow]" string form maps to
        # pandas' StringDtype for strings, which is not an ArrowDtype at all.
        dtype = pd.ArrowDtype(getattr(pa, arrow_type_name)())
        s = pd.Series(values, dtype=dtype)
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == expected

    @pytest.mark.skipif(pa is None, reason="pyarrow not installed")
    def test_arrow_temporal_dtypes(self, inferrer):
        cases = [
            (
                pa.timestamp("ns"),
                pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "datetime",
            ),
            (
                pa.date32(),
                pd.to_datetime(["2020-01-01", "2020-01-02"]).date,
                "datetime",
            ),
            (pa.duration("ns"), pd.to_timedelta(["1 days", "2 days"]), "numeric"),
        ]
        for pa_type, values, expected in cases:
            s = pd.Series(values, dtype=pd.ArrowDtype(pa_type))
            res = inferrer.infer_series_type(s)
            assert res.success
            assert res.data == expected, f"{pa_type} -> {res.data}"

    @pytest.mark.skipif(pa is None, reason="pyarrow not installed")
    def test_arrow_decimal_is_numeric(self, inferrer):
        import decimal

        s = pd.Series(
            [decimal.Decimal("1.5")], dtype=pd.ArrowDtype(pa.decimal128(10, 2))
        )
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == "numeric"


class TestPandasTemporalDtypes:
    @pytest.fixture
    def inferrer(self):
        return UnifiedTypeInferrer(strategy=InferenceStrategy.BALANCED)

    def test_bool_is_not_classified_as_numeric(self, inferrer):
        """pandas ``is_numeric_dtype`` returns True for bool, so order matters."""
        res = inferrer.infer_series_type(pd.Series([True, False, True]))
        assert res.success
        assert res.data == "boolean"

    def test_timedelta_is_numeric(self, inferrer):
        s = pd.to_timedelta(pd.Series(["1 days", "2 days"]))
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == "numeric"

    def test_tz_aware_datetime_is_datetime(self, inferrer):
        s = pd.to_datetime(pd.Series(["2020-01-01", "2020-01-02"])).dt.tz_localize(
            "UTC"
        )
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == "datetime"


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsDtypes:
    @pytest.fixture
    def inferrer(self):
        return UnifiedTypeInferrer(strategy=InferenceStrategy.BALANCED)

    @pytest.mark.parametrize(
        ("values", "dtype_name", "expected"),
        [
            ([1, 2, 3], "Int8", "numeric"),
            ([1, 2, 3], "UInt16", "numeric"),
            ([1.0, 2.0], "Float32", "numeric"),
            ([True, False], "Boolean", "boolean"),
            (["a", "b"], "String", "categorical"),
        ],
    )
    def test_polars_scalar_dtypes(self, inferrer, values, dtype_name, expected):
        s = pl.Series("c", values, dtype=getattr(pl, dtype_name))
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == expected

    def test_polars_duration_is_numeric(self, inferrer):
        import datetime as _dt

        s = pl.Series("c", [_dt.timedelta(days=1), _dt.timedelta(days=2)])
        assert s.dtype == pl.Duration
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == "numeric"

    def test_polars_time_is_numeric(self, inferrer):
        import datetime as _dt

        s = pl.Series("c", [_dt.time(1, 0), _dt.time(2, 0)])
        assert s.dtype == pl.Time
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == "numeric"

    def test_polars_date_is_datetime(self, inferrer):
        import datetime as _dt

        s = pl.Series("c", [_dt.date(2020, 1, 1), _dt.date(2020, 1, 2)])
        res = inferrer.infer_series_type(s)
        assert res.success
        assert res.data == "datetime"

    def test_polars_nested_types_fall_back_to_categorical(self, inferrer):
        for s in (
            pl.Series("c", [[1, 2], [3]]),
            pl.Series("c", [{"a": 1}, {"a": 2}]),
        ):
            res = inferrer.infer_series_type(s)
            assert res.success
            assert res.data == "categorical"

    def test_polars_aggressive_string_boolean(self, inferrer):
        aggressive = UnifiedTypeInferrer(strategy=InferenceStrategy.AGGRESSIVE)
        s = pl.Series("c", ["yes", "no", "yes", "no", "yes"])
        res = aggressive.infer_series_type(s)
        assert res.success
        assert res.data == "boolean"

    def test_polars_aggressive_plain_strings_stay_categorical(self, inferrer):
        aggressive = UnifiedTypeInferrer(strategy=InferenceStrategy.AGGRESSIVE)
        s = pl.Series("c", ["alpha", "beta", "gamma", "delta", "epsilon"])
        res = aggressive.infer_series_type(s)
        assert res.success
        assert res.data == "categorical"


class TestDateSniffing:
    """Object columns are probed for dates with fixed formats before dateutil.

    ``format="mixed"`` disables pandas' vectorised parser and parses row by row
    in Python, which was 20.7% of total runtime. These assert the classification
    is unchanged now that fixed formats are tried first over a small probe.
    """

    @pytest.fixture
    def inferrer(self):
        return UnifiedTypeInferrer(strategy=InferenceStrategy.BALANCED)

    @pytest.mark.parametrize(
        "values,expected",
        [
            (["2020-01-01", "2020-06-15", "2021-12-31"], "datetime"),
            (["2020-01-01 12:30:00", "2020-06-15 08:00:00"], "datetime"),
            (["2020-01-01T12:30:00", "2020-06-15T08:00:00"], "datetime"),
            (["2020/01/01", "2020/06/15"], "datetime"),
            (["01/02/2020", "15/06/2020"], "datetime"),
            (["alpha", "beta", "gamma"], "categorical"),
            (["1.5", "2.5", "3.5"], "numeric"),
            (["", "", ""], "categorical"),
        ],
    )
    def test_classification_is_unchanged(self, inferrer, values, expected):
        s = pd.Series(values * 40, dtype=object)
        assert inferrer.infer_series_type(s).data == expected

    def test_unusual_format_still_reaches_the_dateutil_fallback(self, inferrer):
        """A format not in the fixed list must still be detected."""
        s = pd.Series(["Jan 5, 2020", "Feb 17, 2021", "Mar 3, 2019"] * 40)
        assert inferrer.infer_series_type(s).data == "datetime"

    def test_mostly_junk_is_not_called_a_date(self, inferrer):
        """One parseable value in twenty must not carry the column."""
        s = pd.Series(["2020-01-01"] + ["not a date"] * 19)
        assert inferrer.infer_series_type(s).data != "datetime"

    def test_empty_and_all_null_columns_do_not_divide_by_zero(self, inferrer):
        assert inferrer.infer_series_type(pd.Series([], dtype=object)).success
        assert inferrer.infer_series_type(
            pd.Series([None, None, None], dtype=object)
        ).success

    def test_probe_is_bounded(self, inferrer):
        """A large column must not be parsed in full just to answer yes/no."""
        import time

        s = pd.Series(["Jan 5, 2020"] * 50_000)
        start = time.perf_counter()
        assert inferrer.infer_series_type(s).data == "datetime"
        # Parsing 50k rows through dateutil takes seconds; the bounded probe is
        # milliseconds. A generous ceiling still catches a regression to
        # full-column parsing.
        assert time.perf_counter() - start < 1.0
