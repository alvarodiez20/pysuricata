"""Type inference must not extrapolate from an unrepresentative first chunk.

Reclassifying a numeric column as categorical reads the distinct-value ratio of
the first chunk. That is evidence about the column only when the chunk *is* the
column. On a stream it is not, and nothing revisits the decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize


def _low_then_high():
    """A stream whose prefix looks low-cardinality and whose body is not.

    This is what a sorted column, or one with a leading run of a single value,
    looks like when it arrives in chunks.
    """
    yield pd.DataFrame({"measure": np.repeat(np.arange(9.0), 5_000)})
    for start in range(0, 240_000, 40_000):
        yield pd.DataFrame({"measure": np.arange(start + 100.0, start + 40_100.0)})


class TestStreamedInference:
    def test_unrepresentative_prefix_does_not_mislabel_the_column(self):
        """Regression: 244,255 distinct values were reported as categorical
        because the first 45,000 rows held nine."""
        col = summarize(_low_then_high())["columns"]["measure"]
        assert col["type"] == "numeric"
        assert col["unique_est"] > 100_000

    def test_streamed_low_cardinality_stays_numeric(self):
        """The accepted trade-off, asserted so it is a decision and not a
        surprise: a genuinely low-cardinality streamed column keeps a numeric
        card. Numeric still tracks top values via Misra-Gries."""

        def gen():
            rng = np.random.default_rng(0)
            for _ in range(4):
                yield pd.DataFrame({"rating": rng.integers(1, 6, 5_000).astype(float)})

        assert summarize(gen())["columns"]["rating"]["type"] == "numeric"

    def test_streamed_numeric_statistics_are_still_correct(self):
        values = np.concatenate(
            [np.repeat(np.arange(9.0), 5_000), np.arange(100.0, 40_100.0)]
        )

        def gen():
            yield pd.DataFrame({"measure": values[:45_000]})
            yield pd.DataFrame({"measure": values[45_000:]})

        col = summarize(gen())["columns"]["measure"]
        assert col["min"] == pytest.approx(values.min())
        assert col["max"] == pytest.approx(values.max())
        assert col["mean"] == pytest.approx(values.mean(), rel=1e-9)


class TestInMemoryInferenceUnchanged:
    """A whole frame is trustworthy evidence, so behaviour there must not move."""

    def test_low_cardinality_is_still_reclassified(self):
        df = pd.DataFrame(
            {"rating": np.random.default_rng(0).integers(1, 6, 5_000).astype(float)}
        )
        assert summarize(df)["columns"]["rating"]["type"] == "categorical"

    def test_high_cardinality_stays_numeric(self):
        df = pd.DataFrame({"x": np.arange(5_000.0)})
        assert summarize(df)["columns"]["x"]["type"] == "numeric"

    def test_a_single_chunk_list_source_is_still_treated_as_a_stream(self):
        """A one-element list is an iterable of frames, so it is a stream: we
        cannot know it is complete without consuming it."""
        df = pd.DataFrame(
            {"rating": np.random.default_rng(0).integers(1, 6, 5_000).astype(float)}
        )
        assert summarize([df])["columns"]["rating"]["type"] == "numeric"
