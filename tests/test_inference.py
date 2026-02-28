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
        assert (
            should_reclassify_numeric_as_categorical(unique_count=5, total_count=100)
            is True
        )
        assert (
            should_reclassify_numeric_as_categorical(unique_count=50, total_count=10000)
            is True
        )  # 0.5% ratio
        assert (
            should_reclassify_numeric_as_categorical(unique_count=50, total_count=100)
            is False
        )  # 50% ratio
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
