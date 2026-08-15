"""Extreme values must be exact, and their row indices global.

Two defects: indices came from ``np.arange(len(chunk))``, so they named a
position within whichever chunk the value arrived in; and the consume layer only
sampled extremes every fifth chunk, so the reported min/max could miss the true
ones entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuricata.accumulators import NumericAccumulator
from pysuricata.compute.consume import consume_chunk_pandas
from pysuricata.compute.core.types import ColumnKinds


def _consume_in_chunks(values, n_chunks, reset_index=True):
    kinds = ColumnKinds(numeric=["x"])
    accs = {"x": NumericAccumulator("x")}
    offset = 0
    for chunk in np.array_split(values, n_chunks):
        df = pd.DataFrame({"x": chunk})
        if reset_index:
            # What a generator source gives: each chunk indexed from zero.
            df.index = range(len(chunk))
        consume_chunk_pandas(df, accs, kinds, None, None, row_offset=offset)
        offset += len(chunk)
    return accs["x"].finalize()


class TestGlobalExtremeIndices:
    @pytest.mark.parametrize("position", [0, 120, 750, 999])
    def test_index_is_global_not_chunk_local(self, position):
        values = np.zeros(1_000)
        values[position] = 999.0
        summary = _consume_in_chunks(values, 5)
        assert summary.max == 999.0
        assert summary.max_items[0][0] == position

    def test_minimum_index_is_global(self):
        values = np.zeros(1_000)
        values[813] = -999.0
        summary = _consume_in_chunks(values, 7)
        assert summary.min == -999.0
        assert summary.min_items[0][0] == 813

    def test_indices_are_not_duplicated_across_paths(self):
        """The consume layer used to feed the tracker a second, chunk-local
        copy of each extreme."""
        values = np.zeros(1_000)
        values[750] = 999.0
        summary = _consume_in_chunks(values, 5)
        top = [i for i, v in summary.max_items if v == 999.0]
        assert top == [750]


class TestExtremesAreExact:
    @pytest.mark.parametrize("n_chunks", [1, 2, 3, 5, 11, 50])
    def test_true_extremes_are_never_missed(self, n_chunks):
        """Throttling to every fifth chunk meant four chunks in five could hide
        the true minimum and maximum."""
        rng = np.random.default_rng(7)
        values = rng.standard_normal(5_000)
        values[137] = -50.0
        values[4_021] = 50.0
        summary = _consume_in_chunks(values, n_chunks)
        assert summary.min == pytest.approx(-50.0)
        assert summary.max == pytest.approx(50.0)

    def test_extreme_in_an_early_chunk_survives(self):
        values = np.zeros(1_000)
        values[5] = 42.0
        summary = _consume_in_chunks(values, 20)
        assert summary.max == 42.0
        assert summary.max_items[0][0] == 5

    def test_matches_the_unchunked_result(self):
        rng = np.random.default_rng(11)
        values = rng.standard_normal(3_000)
        whole = _consume_in_chunks(values, 1)
        split = _consume_in_chunks(values, 17)
        assert split.min == pytest.approx(whole.min)
        assert split.max == pytest.approx(whole.max)
        assert split.max_items[0][0] == whole.max_items[0][0]
