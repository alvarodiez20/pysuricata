"""An estimate must not be published as an exact count.

The first of the three things to fix before anyone screenshots the report: the
distinct count exceeded the row count and claimed to be exact.

```
column                  count    reported unique   true    approx
score — 20,000 floats  20,000            20,197  20,000    False
cid   — a primary key  20,000            19,478  20,000    False
```

More distinct values than rows is arithmetically impossible. The sketch is not
at fault — KMV at k=2048 has a relative standard error near 1/sqrt(k-2), about
2.2%, and both figures are inside that. The *reporting* was at fault: it
presented an estimate as an exact integer and then set `approx: False`, which
asserts an exactness the value does not have.

There is a second consequence, which is why the identifier tolerance is tested
here too: `cid` failed the identifier check for exactly this reason. The check
required 0.98 of the row count, which is *inside* the estimator's own error, so
a perfect key was a coin flip and came back a measurement with a mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import profile, summarize
from pysuricata.accumulators import (
    CategoricalAccumulator,
    DatetimeAccumulator,
    NumericAccumulator,
)


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 20_000
    return pd.DataFrame(
        {
            "score": rng.normal(0, 1, n),
            "cid": np.arange(n),
            "age": rng.integers(18, 86, n),
            "label": np.array([f"v{i}" for i in range(n)], dtype=object),
            "seen": pd.date_range("2026-01-01", periods=n, freq="min"),
        }
    )


@pytest.fixture(scope="module")
def payload(frame):
    return summarize(frame, seed=0)


class TestTheImpossibleNumberIsGone:
    """`unique <= count`, for every column of every kind."""

    def test_the_reported_case(self, payload):
        """20,000 standard normals estimated at 20,197 before the clamp."""
        score = payload["columns"]["score"]
        assert score["unique_est"] <= score["count"]

    @pytest.mark.parametrize("column", ["score", "cid", "age", "label", "seen"])
    def test_no_column_exceeds_its_row_count(self, payload, column):
        stats = payload["columns"][column]
        assert stats["unique_est"] <= stats["count"], column

    def test_the_numeric_accumulator_clamps(self):
        acc = NumericAccumulator("x", seed=1)
        acc.update(np.random.default_rng(0).normal(0, 1, 20_000))
        summary = acc.finalize()
        assert summary.unique_est <= summary.count

    def test_the_categorical_accumulator_clamps(self):
        acc = CategoricalAccumulator("c", seed=1)
        acc.update(np.array([f"v{i}" for i in range(20_000)], dtype=object))
        summary = acc.finalize()
        assert summary.unique_est <= summary.count

    def test_the_datetime_accumulator_clamps(self):
        acc = DatetimeAccumulator("t", seed=1)
        acc.update(pd.date_range("2026-01-01", periods=20_000, freq="min").values)
        summary = acc.finalize()
        assert summary.unique_est <= summary.count

    def test_the_clamp_does_not_flatten_a_real_answer(self):
        """It must bound the estimate, not replace it. A column with few levels
        keeps its own count, nowhere near the row count."""
        rng = np.random.default_rng(0)
        acc = NumericAccumulator("x", seed=1)
        acc.update(rng.integers(0, 200, 20_000).astype(float))
        summary = acc.finalize()
        assert 150 <= summary.unique_est <= 200

    def test_an_empty_column_is_zero_not_negative(self):
        acc = NumericAccumulator("x", seed=1)
        assert acc.finalize().unique_est == 0


class TestApproximationIsDeclared:
    """`approx` used to mean sampling alone, so a column small enough to hold
    every value in the reservoir reported `False` while still publishing a
    sketched distinct count."""

    def test_a_sketched_count_is_marked_approximate(self, payload):
        assert payload["columns"]["score"]["approx"] is True

    def test_a_key_column_is_marked_approximate(self, payload):
        assert payload["columns"]["cid"]["approx"] is True

    def test_an_exactly_counted_column_is_not(self, payload):
        """68 distinct integers fit KMV's exact counter, so nothing here is an
        estimate and saying `approx` would be its own kind of dishonesty."""
        assert payload["columns"]["age"]["approx"] is False

    def test_a_small_frame_is_exact(self):
        stats = summarize(pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.0]}), seed=0)
        assert stats["columns"]["x"]["approx"] is False

    def test_a_categorical_sketch_is_declared(self, payload):
        assert payload["columns"]["label"]["approx"] is True

    def test_the_report_marks_the_figure(self, frame):
        """The card prints `Unique (≈)`, so the page no longer claims a
        precision it does not have."""
        html = profile(frame, seed=0).html
        assert "Unique (≈)" in html


class TestTheIdentifierCheckClearsTheSketchError:
    """A threshold has to sit further from 1.0 than the estimator's own error,
    not inside it."""

    def test_a_perfect_key_is_recognised(self, payload):
        assert payload["columns"]["cid"]["type"] == "identifier"

    def test_it_still_says_no_to_a_column_that_is_merely_mostly_distinct(self):
        rng = np.random.default_rng(0)
        values = np.sort(np.r_[np.arange(16_000), rng.integers(0, 16_000, 4_000)])
        stats = summarize(pd.DataFrame({"x": values}), seed=0)
        assert stats["columns"]["x"]["type"] == "numeric"

    def test_a_measurement_is_not_an_identifier(self, payload):
        assert payload["columns"]["score"]["type"] == "numeric"

    def test_a_short_frame_is_never_an_identifier(self):
        """Three rows of 1, 2, 3 are monotonic, integral and fully distinct,
        and are not a key."""
        stats = summarize(pd.DataFrame({"x": np.arange(50)}), seed=0)
        assert stats["columns"]["x"]["type"] != "identifier"


class TestChunkingDoesNotChangeTheAnswer:
    """The clamp is applied at finalize, so it must not depend on how the
    stream was split."""

    @pytest.mark.parametrize("chunk_size", [1_000, 7_000, 100_000])
    def test_unique_stays_within_bounds_at_every_chunk_size(self, frame, chunk_size):
        stats = summarize(frame, seed=0, chunk_size=chunk_size)
        for name, column in stats["columns"].items():
            assert column["unique_est"] <= column["count"], (name, chunk_size)

    def test_the_identifier_verdict_is_stable_across_chunk_sizes(self, frame):
        verdicts = {
            summarize(frame, seed=0, chunk_size=size)["columns"]["cid"]["type"]
            for size in (1_000, 7_000, 100_000)
        }
        assert verdicts == {"identifier"}
