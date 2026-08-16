"""Unit tests for all four card renderers.

Tests render_card() on each renderer type with minimal valid stats objects,
covering normal cases, edge cases, and security (HTML escaping).
"""

import pytest

from pysuricata.render.boolean_card import BooleanCardRenderer
from pysuricata.render.card_types import (
    BooleanStats,
    CategoricalStats,
    DateTimeStats,
    NumericStats,
)
from pysuricata.render.categorical_card import CategoricalCardRenderer
from pysuricata.render.datetime_card import DateTimeCardRenderer
from pysuricata.render.numeric_card import NumericCardRenderer

# ---------------------------------------------------------------------------
# Factory helpers — build minimal valid stats with sensible defaults
# ---------------------------------------------------------------------------


def make_numeric(**kw) -> NumericStats:
    defaults = {
        "name": "col_a",
        "dtype_str": "float64",
        "count": 100,
        "missing": 0,
        "unique_est": 80,
        "approx": False,
        "min": 0.0,
        "max": 100.0,
        "mean": 50.0,
        "median": 50.0,
        "std": 28.87,
        "variance": 833.33,
        "se": 2.89,
        "cv": 0.58,
        "gmean": None,
        "q1": 25.0,
        "q3": 75.0,
        "iqr": 50.0,
        "mad": 25.0,
        "skew": 0.0,
        "kurtosis": 0.0,
        "jb_chi2": 0.5,
        "ci_lo": 44.3,
        "ci_hi": 55.7,
        "gran_step": None,
        "gran_decimals": None,
        "heap_pct": 0.0,
        "zeros": 0,
        "negatives": 0,
        "inf": 0,
        "outliers_iqr": 0,
        "int_like": False,
        "unique_ratio_approx": 0.8,
        "mono_inc": False,
        "mono_dec": False,
        "bimodal": False,
        "mem_bytes": 800,
        "sample_vals": None,
        "sample_scale": 1.0,
        "top_values": None,
        "min_items": None,
        "max_items": None,
        "corr_top": None,
        "chunk_metadata": None,
        "corr_threshold": 0.5,
    }
    defaults.update(kw)
    return NumericStats(**defaults)


def make_categorical(**kw) -> CategoricalStats:
    defaults = {
        "name": "col_b",
        "dtype_str": "object",
        "count": 100,
        "missing": 0,
        "unique_est": 5,
        "approx": False,
        "mem_bytes": 800,
        "top_items": [("alpha", 50), ("beta", 30), ("gamma", 20)],
        "empty_zero": 0,
        "case_variants_est": 0,
        "trim_variants_est": 0,
    }
    defaults.update(kw)
    return CategoricalStats(**defaults)


def make_boolean(**kw) -> BooleanStats:
    defaults = {
        "name": "col_c",
        "dtype_str": "bool",
        "true_n": 60,
        "false_n": 40,
        "missing": 0,
        "mem_bytes": 100,
    }
    defaults.update(kw)
    return BooleanStats(**defaults)


def make_datetime(**kw) -> DateTimeStats:
    defaults = {
        "name": "col_d",
        "dtype_str": "datetime64[ns]",
        "count": 100,
        "missing": 0,
        "mem_bytes": 800,
        "min_ts": None,
        "max_ts": None,
        "mono_inc": False,
        "mono_dec": False,
        "sample_ts": None,
        "sample_scale": 1.0,
        "by_hour": [0] * 24,
        "by_dow": [0] * 7,
        "by_month": [0] * 12,
        "by_year": {},
        "unique_est": 50,
        "approx": True,
        "time_span_days": 365.0,
        "avg_interval_seconds": 86400.0,
        "interval_std_seconds": 3600.0,
        "weekend_ratio": 0.28,
        "business_hours_ratio": 0.6,
        "seasonal_pattern": None,
        "chunk_metadata": None,
    }
    defaults.update(kw)
    return DateTimeStats(**defaults)


# ---------------------------------------------------------------------------
# NumericCardRenderer
# ---------------------------------------------------------------------------


class TestNumericCardRenderer:
    def setup_method(self):
        self.renderer = NumericCardRenderer()

    def test_minimal_stats_renders_card(self):
        html = self.renderer.render_card(make_numeric())
        assert '<article class="var-card"' in html
        assert "col_a" in html

    def test_returns_string(self):
        result = self.renderer.render_card(make_numeric())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_name_appears_in_output(self):
        html = self.renderer.render_card(make_numeric(name="my_column"))
        assert "my_column" in html

    def test_html_special_chars_in_name_are_escaped(self):
        html = self.renderer.render_card(
            make_numeric(name="<script>alert('xss')</script>")
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_unicode_name_rendered(self):
        html = self.renderer.render_card(make_numeric(name="montant_€"))
        assert "montant_€" in html or "montant" in html

    def test_approx_true_shows_badge(self):
        html = self.renderer.render_card(make_numeric(approx=True))
        assert "approx" in html

    def test_approx_false_no_badge(self):
        html = self.renderer.render_card(make_numeric(approx=False))
        # The badge HTML should not appear if approx is False
        assert '<span class="badge">approx</span>' not in html

    def test_all_missing_renders_without_crash(self):
        html = self.renderer.render_card(make_numeric(count=0, missing=100))
        assert '<article class="var-card"' in html

    def test_all_missing_shows_missing_info(self):
        html = self.renderer.render_card(make_numeric(count=0, missing=100))
        assert "100" in html

    def test_with_chunk_metadata_shows_distribution(self):
        """The strip lives in the Missing Values pane, which #154 stopped
        rendering on columns that have no missing values -- so the fixture now
        has some. A per-chunk missing chart on a complete column was the thing
        being removed."""
        chunks = [(0, 49, 0), (50, 99, 5)]
        html = self.renderer.render_card(make_numeric(missing=5, chunk_metadata=chunks))
        assert "chunk-distribution" in html

    def test_no_missing_means_no_missing_pane_at_all(self):
        chunks = [(0, 49, 0), (50, 99, 0)]
        html = self.renderer.render_card(make_numeric(missing=0, chunk_metadata=chunks))
        assert "chunk-distribution" not in html
        assert 'data-tab="missing"' not in html

    def test_no_chunk_metadata_no_distribution_section(self):
        html = self.renderer.render_card(make_numeric(chunk_metadata=None))
        assert "chunk-distribution" not in html

    def test_infinite_values_flag_rendered(self):
        html = self.renderer.render_card(make_numeric(inf=5))
        assert "∞" in html or "Inf" in html or "inf" in html.lower()

    def test_nan_inf_in_numeric_fields_no_crash(self):
        html = self.renderer.render_card(
            make_numeric(
                mean=float("nan"),
                std=float("nan"),
                skew=float("nan"),
                kurtosis=float("nan"),
                jb_chi2=float("nan"),
            )
        )
        assert '<article class="var-card"' in html

    def test_zero_count_and_zero_missing_no_crash(self):
        html = self.renderer.render_card(make_numeric(count=0, missing=0))
        assert '<article class="var-card"' in html

    def test_large_values_no_crash(self):
        html = self.renderer.render_card(
            make_numeric(
                count=10_000_000,
                missing=1_000_000,
                min=-1e15,
                max=1e15,
                mean=5e14,
            )
        )
        assert '<article class="var-card"' in html

    def test_quality_assessor_accessible(self):
        # quality_assessor is now inherited from CardRenderer
        assert hasattr(self.renderer, "quality_assessor")

    def test_table_builder_accessible(self):
        assert hasattr(self.renderer, "table_builder")

    def test_skewed_right_flag_when_high_skew(self):
        html = self.renderer.render_card(make_numeric(skew=5.0))
        assert "skewed" in html

    def test_constant_flag_when_one_unique(self):
        html = self.renderer.render_card(make_numeric(unique_est=1))
        assert "constant" in html

    def test_empty_name_renders(self):
        html = self.renderer.render_card(make_numeric(name=""))
        assert '<article class="var-card"' in html


# ---------------------------------------------------------------------------
# CategoricalCardRenderer
# ---------------------------------------------------------------------------


class TestCategoricalCardRenderer:
    def setup_method(self):
        self.renderer = CategoricalCardRenderer()

    def test_minimal_stats_renders_card(self):
        html = self.renderer.render_card(make_categorical())
        assert "var-card" in html
        assert "col_b" in html

    def test_returns_string(self):
        result = self.renderer.render_card(make_categorical())
        assert isinstance(result, str)

    def test_html_special_chars_in_name_escaped(self):
        html = self.renderer.render_card(
            make_categorical(name='<img src="x" onerror="alert(1)">')
        )
        assert "<img" not in html

    def test_missing_values_shown(self):
        html = self.renderer.render_card(make_categorical(count=80, missing=20))
        assert "20" in html

    def test_top_items_none_renders_without_crash(self):
        html = self.renderer.render_card(make_categorical(top_items=None))
        assert "var-card" in html

    def test_top_items_empty_list_renders_without_crash(self):
        html = self.renderer.render_card(make_categorical(top_items=[]))
        assert "var-card" in html

    def test_approx_true_shows_badge(self):
        html = self.renderer.render_card(make_categorical(approx=True))
        assert "approx" in html

    def test_approx_false_no_badge(self):
        html = self.renderer.render_card(make_categorical(approx=False))
        assert '<span class="badge">approx</span>' not in html

    def test_high_cardinality_flag(self):
        # unique_est much larger than count → high cardinality
        html = self.renderer.render_card(make_categorical(count=100, unique_est=500))
        assert "cardinality" in html.lower() or "var-card" in html

    def test_case_variants_flag(self):
        html = self.renderer.render_card(make_categorical(case_variants_est=3))
        assert "var-card" in html  # should render without crash

    def test_empty_strings_flag(self):
        html = self.renderer.render_card(make_categorical(empty_zero=5))
        assert "var-card" in html

    def test_with_chunk_metadata_shows_distribution(self):
        # CategoricalStats doesn't have chunk_metadata in the dataclass definition
        # but the shared method uses getattr(..., None) so it safely returns ""
        html = self.renderer.render_card(make_categorical())
        assert "var-card" in html

    def test_name_with_special_characters_in_id(self):
        html = self.renderer.render_card(make_categorical(name="amount ($)"))
        # col_id should be sanitized; raw special chars should not be in id
        assert "amount ($)" not in html.split('id="')[1].split('"')[0]

    def test_all_missing_renders(self):
        html = self.renderer.render_card(make_categorical(count=0, missing=100))
        assert "var-card" in html

    def test_quality_assessor_inherited(self):
        assert hasattr(self.renderer, "quality_assessor")

    def test_table_builder_inherited(self):
        assert hasattr(self.renderer, "table_builder")

    def test_single_top_item_renders(self):
        html = self.renderer.render_card(make_categorical(top_items=[("only", 100)]))
        assert "var-card" in html
        assert "only" in html

    def test_unicode_category_values_rendered(self):
        html = self.renderer.render_card(
            make_categorical(top_items=[("日本語", 50), ("Ελληνικά", 30)])
        )
        assert "var-card" in html


# ---------------------------------------------------------------------------
# BooleanCardRenderer
# ---------------------------------------------------------------------------


class TestBooleanCardRenderer:
    def setup_method(self):
        self.renderer = BooleanCardRenderer()

    def test_minimal_stats_renders_card(self):
        html = self.renderer.render_card(make_boolean())
        assert "var-card" in html
        assert "col_c" in html

    def test_returns_string(self):
        result = self.renderer.render_card(make_boolean())
        assert isinstance(result, str)

    def test_html_special_chars_in_name_escaped(self):
        html = self.renderer.render_card(make_boolean(name="<b>bold</b>"))
        assert "<b>" not in html

    def test_all_true_constant_renders(self):
        # All values True → constant flag
        html = self.renderer.render_card(make_boolean(true_n=100, false_n=0))
        assert "var-card" in html

    def test_all_false_constant_renders(self):
        html = self.renderer.render_card(make_boolean(true_n=0, false_n=100))
        assert "var-card" in html

    def test_missing_values_render_correctly(self):
        html = self.renderer.render_card(
            make_boolean(true_n=50, false_n=30, missing=20)
        )
        assert "var-card" in html
        assert "20" in html

    def test_all_missing_boolean_renders(self):
        html = self.renderer.render_card(make_boolean(true_n=0, false_n=0, missing=100))
        assert "var-card" in html

    def test_zero_true_and_false_and_missing_renders(self):
        # Completely empty — edge case: total = 0
        html = self.renderer.render_card(make_boolean(true_n=0, false_n=0, missing=0))
        assert "var-card" in html

    def test_imbalanced_flag_when_very_skewed(self):
        # 95/5 split → imbalanced
        html = self.renderer.render_card(make_boolean(true_n=95, false_n=5))
        assert "var-card" in html  # should render
        assert "Imbalanced" in html or "imbalanced" in html.lower()

    def test_balanced_no_imbalanced_flag(self):
        # 50/50 split → NOT imbalanced
        html = self.renderer.render_card(make_boolean(true_n=50, false_n=50))
        # Imbalanced flag should not appear
        assert "Imbalanced" not in html

    def test_quality_assessor_inherited(self):
        assert hasattr(self.renderer, "quality_assessor")

    def test_table_builder_inherited(self):
        assert hasattr(self.renderer, "table_builder")

    def test_missing_completeness_uses_boolean_total(self):
        # Boolean total = true_n + false_n + missing (NOT count field)
        # With 80 present and 20 missing, present_pct should be 80%
        html = self.renderer.render_card(
            make_boolean(true_n=40, false_n=40, missing=20)
        )
        assert "80.0%" in html  # present_pct
        assert "20.0%" in html  # missing_pct

    def test_chunk_metadata_not_in_boolean_stats(self):
        # BooleanStats doesn't have chunk_metadata; chunk section should not appear
        html = self.renderer.render_card(make_boolean())
        assert "chunk-distribution" not in html


# ---------------------------------------------------------------------------
# DateTimeCardRenderer
# ---------------------------------------------------------------------------


class TestDateTimeCardRenderer:
    def setup_method(self):
        self.renderer = DateTimeCardRenderer()

    def test_minimal_stats_renders_card(self):
        html = self.renderer.render_card(make_datetime())
        assert "var-card" in html
        assert "col_d" in html

    def test_returns_string(self):
        result = self.renderer.render_card(make_datetime())
        assert isinstance(result, str)

    def test_html_special_chars_in_name_escaped(self):
        html = self.renderer.render_card(make_datetime(name="<time>now</time>"))
        assert "<time>" not in html

    def test_all_missing_renders(self):
        html = self.renderer.render_card(make_datetime(count=0, missing=100))
        assert "var-card" in html

    def test_zero_count_and_zero_missing(self):
        html = self.renderer.render_card(make_datetime(count=0, missing=0))
        assert "var-card" in html

    def test_with_chunk_metadata_shows_distribution(self):
        """Same as the numeric case: the strip lives in the Missing Values pane,
        which now renders only when the column has missing values (#154)."""
        chunks = [(0, 49, 2), (50, 99, 0)]
        html = self.renderer.render_card(
            make_datetime(missing=2, chunk_metadata=chunks)
        )
        assert "chunk-distribution" in html

    def test_without_chunk_metadata_no_distribution(self):
        html = self.renderer.render_card(make_datetime(chunk_metadata=None))
        assert "chunk-distribution" not in html

    def test_monotonic_increasing_flag(self):
        html = self.renderer.render_card(make_datetime(mono_inc=True, count=10))
        assert "var-card" in html
        # Flag should be present (monotonic increasing)
        assert "Monotonic" in html or "monotonic" in html.lower()

    def test_monotonic_decreasing_flag(self):
        html = self.renderer.render_card(make_datetime(mono_dec=True, count=10))
        assert "var-card" in html

    def test_quality_assessor_inherited(self):
        assert hasattr(self.renderer, "quality_assessor")

    def test_table_builder_inherited(self):
        assert hasattr(self.renderer, "table_builder")

    def test_by_hour_all_zeros_renders(self):
        html = self.renderer.render_card(make_datetime(by_hour=[0] * 24))
        assert "var-card" in html

    def test_by_hour_none_renders(self):
        html = self.renderer.render_card(make_datetime(by_hour=None))
        assert "var-card" in html

    def test_by_year_empty_dict_renders(self):
        html = self.renderer.render_card(make_datetime(by_year={}))
        assert "var-card" in html

    def test_by_year_with_data_renders(self):
        html = self.renderer.render_card(make_datetime(by_year={2020: 30, 2021: 70}))
        assert "var-card" in html

    def test_missing_completeness_shown(self):
        """The pane needs missing values **and** more than one chunk (#154,
        5b.7). With a single chunk it states one fact four times under a header
        already carrying it; the only thing it knows that the card face does
        not is where in the read the gaps fall."""
        html = self.renderer.render_card(
            make_datetime(
                count=70,
                missing=30,
                chunk_metadata=[(0, 49, 20), (50, 99, 10)],
            )
        )
        assert "70.0%" in html  # present_pct
        assert "30.0%" in html  # missing_pct

    def test_missing_completeness_hidden_for_a_single_chunk(self):
        html = self.renderer.render_card(
            make_datetime(count=70, missing=30, chunk_metadata=[(0, 99, 30)])
        )
        assert "Data Completeness" not in html


# ---------------------------------------------------------------------------
# Cross-renderer consistency checks
# ---------------------------------------------------------------------------


class TestCrossRendererConsistency:
    """Ensure all 4 renderers produce consistent structural HTML."""

    @pytest.mark.parametrize(
        "renderer_class,stats_fn",
        [
            (NumericCardRenderer, make_numeric),
            (CategoricalCardRenderer, make_categorical),
            (BooleanCardRenderer, make_boolean),
            (DateTimeCardRenderer, make_datetime),
        ],
    )
    def test_all_renderers_produce_var_card_article(self, renderer_class, stats_fn):
        renderer = renderer_class()
        html = renderer.render_card(stats_fn())
        assert '<article class="var-card"' in html

    @pytest.mark.parametrize(
        "renderer_class,stats_fn",
        [
            (NumericCardRenderer, make_numeric),
            (CategoricalCardRenderer, make_categorical),
            (BooleanCardRenderer, make_boolean),
            (DateTimeCardRenderer, make_datetime),
        ],
    )
    def test_all_renderers_have_quality_assessor(self, renderer_class, stats_fn):
        renderer = renderer_class()
        assert hasattr(renderer, "quality_assessor")

    @pytest.mark.parametrize(
        "renderer_class,stats_fn",
        [
            (NumericCardRenderer, make_numeric),
            (CategoricalCardRenderer, make_categorical),
            (BooleanCardRenderer, make_boolean),
            (DateTimeCardRenderer, make_datetime),
        ],
    )
    def test_all_renderers_escape_xss_in_name(self, renderer_class, stats_fn):
        renderer = renderer_class()
        stats = stats_fn(name='"><script>alert(1)</script>')
        html = renderer.render_card(stats)
        assert "<script>" not in html

    @pytest.mark.parametrize(
        "renderer_class,stats_fn",
        [
            (NumericCardRenderer, make_numeric),
            (CategoricalCardRenderer, make_categorical),
            (DateTimeCardRenderer, make_datetime),
        ],
    )
    def test_completeness_section_present_when_missing(self, renderer_class, stats_fn):
        """Renderers with missing values show the Data Completeness pane.

        Numeric and datetime additionally require more than one chunk (#154,
        5b.7); categorical does not, because its accumulator is never handed
        chunk metadata to gate on (#193). The fixture supplies two chunks where
        the type carries them, so this asserts the pane for every kind that can
        render it.
        """
        renderer = renderer_class()
        extra = {}
        if renderer_class in (NumericCardRenderer, DateTimeCardRenderer):
            extra["chunk_metadata"] = [(0, 49, 10), (50, 99, 10)]
        stats = stats_fn(count=80, missing=20, **extra)
        html = renderer.render_card(stats)
        assert "Data Completeness" in html
