"""A column scaled to [0, 1] is numeric, and a 0.0/1.0 column is not empty.

Two defects, found while replacing the demo dataset (#150). The bike-sharing
frame has `temp`, `feels_like` and `humidity` normalised to [0, 1], and the
report said each was **100% missing** -- `count=0, missing=17,379` against a
frame pandas says has no gaps at all.

**The promotion truncated instead of comparing.** The rule that decides whether
a numeric column is really boolean read `{int(v) for v in unique_values}`,
added to handle numpy types. `int()` on a float truncates, so `int(0.24)` is 0
and `int(1.0)` is 1: every value of a normalised column collapsed onto {0, 1}
and the column was promoted. A column maxing at 0.85 escaped only because
nothing in it truncated to 1.

**The conversion stringified.** Once promoted, `_to_bool_array_pandas` fell
through to a string coercion where `astype(str)` turns 1.0 into `"1.0"` --
which is in neither the true set nor the false set, so every row became None.
That half is independent of the first: a genuine 0.0/1.0 float column, which
*should* be promoted, was also reported entirely missing. An integer 0/1 column
escaped it because `str(1)` is `"1"`, which is in the set.

So the two dtypes disagreed about the same data, and **so did the two
adapters**: polars casts with `cast(pl.Boolean, strict=False)` and was right
all along, which is why nothing caught this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize

N = 500


def _column(frame) -> dict:
    return summarize(frame, seed=0)["columns"]["c"]


def _frames(values) -> list:
    """The same values as pandas, and as polars when it is installed."""
    frames = [pd.DataFrame({"c": values})]
    try:
        import polars as pl

        frames.append(pl.DataFrame({"c": values}))
    except ImportError:  # pragma: no cover - polars is a test dependency
        pass
    return frames


class TestANormalisedColumnStaysNumeric:
    """The reported case, and the reason it matters: a feature scaled to
    [0, 1] is about as ordinary as data gets, and the failure was total."""

    @pytest.mark.parametrize(
        ("name", "values"),
        [
            ("spans exactly 0 to 1", np.round(np.linspace(0.0, 1.0, N), 2)),
            ("starts above zero", np.round(np.linspace(0.02, 1.0, N), 2)),
            ("stops below one", np.round(np.linspace(0.0, 0.85, N), 2)),
        ],
    )
    def test_it_is_numeric_and_not_missing(self, name, values):
        for frame in _frames(values):
            stats = _column(frame)

            assert stats["type"] == "numeric", f"{name}: {type(frame).__name__}"
            assert stats["missing"] == 0, f"{name}: {type(frame).__name__}"
            assert stats["count"] == N

    def test_the_reported_column_shape(self):
        """`humidity`: 89 distinct values, min 0.0, max 1.0, some genuine
        zeros. Every one of those is what made it look boolean."""
        rng = np.random.default_rng(0)
        values = np.round(rng.uniform(0.0, 1.0, N), 2)
        values[:22] = 0.0
        values[22:30] = 1.0

        stats = _column(pd.DataFrame({"c": values}))

        assert stats["type"] == "numeric"
        assert stats["missing"] == 0


class TestAGenuineBooleanColumnStillWorks:
    """The guard must not have bought its correctness by refusing to promote
    anything."""

    @pytest.mark.parametrize(
        ("name", "values"),
        [
            ("0/1 integers", np.array([0, 1] * (N // 2))),
            ("0.0/1.0 floats", np.array([0.0, 1.0] * (N // 2))),
            ("real booleans", np.array([True, False] * (N // 2))),
        ],
    )
    def test_it_is_boolean_and_fully_counted(self, name, values):
        for frame in _frames(values):
            stats = _column(frame)

            assert stats["type"] == "boolean", f"{name}: {type(frame).__name__}"
            assert stats["missing"] == 0, f"{name}: {type(frame).__name__}"
            assert stats["true"] + stats["false"] == N

    def test_missing_values_are_still_missing(self):
        """The float path must not turn a NaN into a False on its way through
        the numeric comparison."""
        values = np.array([0.0, 1.0, np.nan] * (N // 3))

        stats = _column(pd.DataFrame({"c": values}))

        assert stats["type"] == "boolean"
        assert stats["missing"] == int(pd.Series(values).isna().sum())
        assert stats["true"] + stats["false"] + stats["missing"] == len(values)


class TestTheTwoDtypesAgree:
    """The defect was visible only as a disagreement: the same data as int and
    as float profiled differently, and nothing asserted they should not."""

    def test_int_and_float_columns_of_the_same_values_agree(self):
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 2, N)

        as_int = _column(pd.DataFrame({"c": raw}))
        as_float = _column(pd.DataFrame({"c": raw.astype(float)}))

        assert as_int["type"] == as_float["type"]
        assert as_int["missing"] == as_float["missing"] == 0
        assert as_int["true"] == as_float["true"]

    def test_pandas_and_polars_agree(self):
        """polars was right all along -- `cast(pl.Boolean, strict=False)` --
        which is why nothing caught this. Pinning the pair means the next
        divergence fails somewhere."""
        pl = pytest.importorskip("polars")
        values = np.round(np.linspace(0.0, 1.0, N), 2)

        pandas_stats = _column(pd.DataFrame({"c": values}))
        polars_stats = _column(pl.DataFrame({"c": values}))

        assert pandas_stats["type"] == polars_stats["type"] == "numeric"
        assert pandas_stats["missing"] == polars_stats["missing"] == 0
