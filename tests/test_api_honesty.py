"""Four ways the library said something untrue about its own behaviour.

#98 a healthy column flagged as a data-quality problem; #100 an error outside
the exception hierarchy, in internal vocabulary, naming the wrong type; #101 a
validation rule that disagreed with itself depending on which door you came
through; #102 a failure message printed by a call that succeeded.

None of them is a wrong number. All of them cost trust in the numbers.
"""

from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pandas as pd
import pytest

from pysuricata import (
    ComputeOptions,
    ConfigurationError,
    ProfileConfig,
    PySuricataError,
    UnsupportedDataError,
    profile,
)


def _flags(html: str) -> list[str]:
    """The quality chips, by label."""
    return re.findall(r'<li class="flag[^"]*"[^>]*>([^<]+)</li>', html)


class TestQualityFlagsDoNotDependOnTheRowCount:
    """#98. The classifier stopped using a unique *ratio* in #84; the flag layer
    did not, so the same column changed verdict as the frame grew — and #86 put
    the chips in a triage block at the top, where a false alarm is the first
    thing anyone reads."""

    @pytest.mark.parametrize("rows", [1_000, 20_000, 200_000])
    def test_age_is_never_flagged_quasi_constant(self, rows):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"age": rng.integers(18, 86, rows)})
        assert "Quasi‑constant" not in _flags(profile(frame, seed=0).html)

    @pytest.mark.parametrize("rows", [1_000, 20_000, 200_000])
    def test_age_gets_the_same_verdict_at_every_size(self, rows):
        """The UX-1 test, applied to the flag layer."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"age": rng.integers(18, 86, rows)})
        assert _flags(profile(frame, seed=0).html) == ["Positive‑only"]

    def test_a_genuinely_quasi_constant_column_is_still_flagged(self):
        """99.5% one value. Misra-Gries counts are lower bounds, so a share
        computed from them understates dominance and never invents it."""
        rng = np.random.default_rng(0)
        values = np.concatenate([np.ones(19_900), rng.standard_normal(100)])
        frame = pd.DataFrame({"stuck": values})
        assert "Quasi‑constant" in _flags(profile(frame, seed=0).html)

    def test_a_constant_column_is_flagged_constant(self):
        """Streamed, so it stays numeric: a single-valued column in a whole
        frame is reclassified as categorical and gets a categorical card."""

        def chunks():
            for _ in range(4):
                yield pd.DataFrame({"one": np.ones(2_500)})

        assert "Constant" in _flags(profile(chunks(), seed=0).html)

    def test_a_continuous_column_is_neither(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"z": rng.standard_normal(20_000)})
        flags = _flags(profile(frame, seed=0).html)
        assert "Quasi‑constant" not in flags
        assert "Discrete" not in flags

    @pytest.mark.parametrize("rows", [5_000, 50_000])
    def test_discrete_uses_the_classifier_ceiling(self, rows):
        """A streamed integer column with few levels stays numeric, and Discrete
        is what says so. The ceiling is the classifier's own, so the flag and
        the classification cannot disagree."""
        rng = np.random.default_rng(0)

        def chunks():
            for _ in range(4):
                yield pd.DataFrame({"grade": rng.integers(0, 12, rows // 4)})

        assert "Discrete" in _flags(profile(chunks(), seed=0).html)

    def test_discrete_does_not_fire_above_the_ceiling(self):
        rng = np.random.default_rng(0)

        def chunks():
            for _ in range(4):
                yield pd.DataFrame({"age": rng.integers(18, 86, 12_500)})

        assert "Discrete" not in _flags(profile(chunks(), seed=0).html)

    def test_no_cardinality_ratio_survives_in_the_flag_layer(self):
        """The acceptance criterion from #98, as a test: one cardinality rule."""
        from pysuricata.render import card_config

        assert not hasattr(card_config.QualityThresholds(), "unique_ratio_threshold")
        assert not hasattr(card_config.QualityThresholds(), "quasi_constant_threshold")


class TestBadInputStaysInTheHierarchy:
    """#100. `except PySuricataError` missed the most common way to get an input
    wrong, and the message described the first element rather than the argument."""

    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"a": np.arange(100.0)})

    def test_a_list_of_the_wrong_thing(self):
        with pytest.raises(UnsupportedDataError, match="list of int"):
            profile([1, 2, 3])

    def test_the_list_error_is_in_the_hierarchy(self):
        with pytest.raises(PySuricataError):
            profile([1, 2, 3])

    def test_a_scalar_still_reports_well(self):
        with pytest.raises(UnsupportedDataError, match="Cannot profile int"):
            profile(42)

    def test_a_generator_of_the_wrong_thing(self):
        """A generator cannot be inspected without consuming it, so this one is
        caught by the engine — and has to arrive as the same kind of error."""

        def chunks():
            yield 1

        with pytest.raises(UnsupportedDataError, match="yielding int"):
            profile(chunks())

    def test_no_internal_vocabulary_reaches_the_caller(self):
        with pytest.raises(UnsupportedDataError) as caught:
            profile([1, 2, 3])
        assert "Adapter" not in str(caught.value)
        assert "Unsupported input type" not in str(caught.value)

    def test_a_list_of_frames_still_works(self, frame):
        assert profile([frame, frame]).stats["dataset"]["rows_est"] == 200

    def test_an_empty_list_is_not_refused_early(self):
        """Nothing to inspect, so it goes to the engine like any other stream."""
        assert profile([]).stats == {}

    def test_a_bad_config_object(self, frame):
        with pytest.raises(ConfigurationError, match="must be a ProfileConfig"):
            profile(frame, config="oops")

    def test_the_bad_config_error_suggests_the_keyword_form(self, frame):
        with pytest.raises(ConfigurationError, match="chunk_size"):
            profile(frame, config="oops")


class TestValidationAgreesWithItself:
    """#101. The constructor's rule guarded a door nobody walks through: the
    options are mutable, so people build one and then adjust a field."""

    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"a": np.arange(100.0)})

    def test_the_constructor_rejects_zero(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ComputeOptions(chunk_size=0)

    def test_mutation_is_rejected_too(self, frame):
        options = ComputeOptions()
        options.chunk_size = 0
        with pytest.raises(ConfigurationError, match="chunk_size must be positive"):
            profile(frame, config=ProfileConfig(compute=options))

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("numeric_sample_size", 0),
            ("max_uniques", 0),
            ("top_k", 0),
            ("chunk_size", -1),
            ("log_every_n_chunks", 0),
            ("corr_max_cols", 0),
            ("corr_max_per_col", 0),
            ("corr_threshold", 1.5),
        ],
    )
    def test_construction_and_mutation_agree(self, frame, field, bad):
        """The rule is the same rule, whichever way the value arrives."""
        with pytest.raises(ValueError):
            ComputeOptions(**{field: bad})

        options = ComputeOptions()
        setattr(options, field, bad)
        with pytest.raises(ValueError):
            profile(frame, config=ProfileConfig(compute=options))

    def test_none_still_means_no_chunking(self, frame):
        options = ComputeOptions()
        options.chunk_size = None
        assert (
            profile(frame, config=ProfileConfig(compute=options)).stats["dataset"][
                "rows_est"
            ]
            == 100
        )

    def test_a_valid_mutation_still_works(self, frame):
        options = ComputeOptions()
        options.chunk_size = 25
        assert (
            profile(frame, config=ProfileConfig(compute=options)).stats["dataset"][
                "rows_est"
            ]
            == 100
        )


class TestASuccessfulCallSaysNothingAboutFailing:
    """#102. `profile(pd.DataFrame())` announced "Stream processing failed" and
    then returned a usable report. In CI — where `pysuricata check` now puts this
    library on purpose — a line containing "failed" on a green run is exactly
    what gets grepped for."""

    def test_an_empty_frame_is_silent(self):
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            profile(pd.DataFrame())
        assert err.getvalue() == ""
        assert out.getvalue() == ""

    def test_it_still_returns_a_report(self):
        with contextlib.redirect_stderr(io.StringIO()):
            report = profile(pd.DataFrame())
        assert report.html
        assert report.stats == {}

    def test_a_real_failure_still_says_so(self):
        """The word is reserved for calls that raise."""

        def chunks():
            yield 1

        err = io.StringIO()
        with contextlib.redirect_stderr(err), pytest.raises(UnsupportedDataError):
            profile(chunks())
        assert "failed" in err.getvalue().lower()
