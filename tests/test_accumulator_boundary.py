"""The surface the engine may use on an accumulator, and the options' shape.

#64 and #105.

#64 is not tidiness. A PyO3 accumulator cannot satisfy
`isinstance(acc, NumericAccumulator)`, cannot expose a `_uniques` attribute
holding a Python `KMV`, and cannot be pickled by copying `__dict__`. Each of
those was how something outside the accumulator package reached inside it, and
each was a place the crate could not be swapped in without editing the caller.

So these tests are written against a **fake accumulator** that implements the
protocol and inherits nothing — which is the closest a Python test gets to
proving the boundary would hold for a type from another language.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from pysuricata import ComputeOptions, ConfigurationError, ProfileConfig, summarize
from pysuricata.accumulators import (
    BooleanAccumulator,
    CategoricalAccumulator,
    DatetimeAccumulator,
    NumericAccumulator,
)
from pysuricata.accumulators.protocols import StreamingAccumulator


def _accumulators():
    rng = np.random.default_rng(0)
    return [
        (NumericAccumulator("n", seed=1), rng.standard_normal(3_000)),
        (
            CategoricalAccumulator("c", seed=1),
            np.array(["north", "south", "east"] * 1_000, dtype=object),
        ),
        (
            DatetimeAccumulator("d", seed=1),
            pd.date_range("2024-01-01", periods=3_000, freq="h").values,
        ),
        (BooleanAccumulator("b"), np.array([True, False] * 1_500)),
    ]


class TestKindReplacesIsinstance:
    @pytest.mark.parametrize(
        "cls,expected",
        [
            (NumericAccumulator, "numeric"),
            (CategoricalAccumulator, "categorical"),
            (DatetimeAccumulator, "datetime"),
            (BooleanAccumulator, "boolean"),
        ],
    )
    def test_every_accumulator_declares_its_kind(self, cls, expected):
        assert cls("x").kind == expected

    def test_the_kinds_are_the_four_the_engine_knows(self):
        kinds = {acc.kind for acc, _ in _accumulators()}
        assert kinds == {"numeric", "categorical", "datetime", "boolean"}

    def test_the_consume_layer_no_longer_tests_types(self):
        """The acceptance criterion from #64, as a test."""
        from pathlib import Path

        for name in ("consume.py", "consume_polars.py"):
            source = Path("pysuricata/compute") / name
            assert "isinstance(acc" not in source.read_text(), name


class TestTheProtocolIsSatisfied:
    @pytest.mark.parametrize("index", range(4))
    def test_each_accumulator_matches_the_protocol(self, index):
        acc, _ = _accumulators()[index]
        assert isinstance(acc, StreamingAccumulator)

    @pytest.mark.parametrize("index", range(4))
    def test_unique_est_is_a_property_on_every_kind(self, index):
        acc, data = _accumulators()[index]
        acc.update(data)
        assert isinstance(acc.unique_est, int)

    def test_the_render_layer_reads_no_private_sketch(self):
        """The access, not the word: the comment explaining why it is gone
        naturally mentions the name."""
        from pathlib import Path

        assert "acc._uniques" not in Path("pysuricata/render/html.py").read_text()

    def test_the_report_layer_reads_no_private_flag(self):
        from pathlib import Path

        assert "acc._track_top_k" not in Path("pysuricata/report.py").read_text()

    def test_top_k_tracking_is_public(self):
        acc = NumericAccumulator("n", seed=1)
        assert acc.tracks_top_values is True
        acc.update(np.arange(500_000.0))
        assert acc.tracks_top_values is False


class TestCheckpointingSurvivesPickling:
    """Checkpointing pickles the accumulator dict. A native type needs an
    explicit reduce, or checkpointing breaks the moment the fast path is
    enabled — at the *end* of a long run, which is the worst time to find out."""

    @pytest.mark.parametrize("index", range(4))
    def test_a_round_trip_preserves_the_summary(self, index):
        acc, data = _accumulators()[index]
        acc.update(data)
        restored = pickle.loads(pickle.dumps(acc))
        assert restored.finalize() == acc.finalize()

    @pytest.mark.parametrize("index", range(4))
    def test_a_restored_accumulator_keeps_accumulating(self, index):
        acc, data = _accumulators()[index]
        half = len(data) // 2
        acc.update(data[:half])
        restored = pickle.loads(pickle.dumps(acc))
        restored.update(data[half:])
        acc.update(data[half:])
        assert restored.finalize().count == acc.finalize().count

    def test_the_reduce_names_a_rebuilder_rather_than_the_class(self):
        """What makes it implementable by a type that has no `__dict__`."""
        from pysuricata.accumulators.protocols import rebuild_accumulator

        acc = NumericAccumulator("n", seed=1)
        rebuilder, args = acc.__reduce__()
        assert rebuilder is rebuild_accumulator
        assert args[0] is NumericAccumulator

    def test_state_is_a_mapping(self):
        state = NumericAccumulator("n", seed=1).__getstate__()
        assert isinstance(state, dict)
        assert state["name"] == "n"


class TestAForeignAccumulatorWouldWork:
    """The real test of a boundary: something that inherits nothing."""

    class Foreign:
        """What a PyO3 accumulator looks like from Python."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.count = 0
            self.missing = 0
            self._bytes = 0

        @property
        def kind(self) -> str:
            return "numeric"

        @property
        def unique_est(self) -> int:
            return self.count

        def update(self, arr, row_offset: int = 0) -> None:
            self.count += len(arr)

        def add_mem(self, nbytes: int) -> None:
            self._bytes += nbytes

        def finalize(self):
            return {"count": self.count}

    def test_it_satisfies_the_protocol_without_inheriting(self):
        assert isinstance(self.Foreign("x"), StreamingAccumulator)

    def test_the_consume_layer_dispatches_on_it(self):
        """It would have fallen through every `isinstance` branch before."""
        from pysuricata.compute.consume import consume_chunk_pandas
        from pysuricata.compute.core.types import ColumnKinds

        frame = pd.DataFrame({"x": np.arange(100.0)})
        foreign = self.Foreign("x")
        consume_chunk_pandas(
            frame,
            {"x": foreign},
            ColumnKinds(numeric=["x"], categorical=[], datetime=[], boolean=[]),
        )
        assert foreign.count == 100


class TestOptionsShape:
    """#105. Twenty-two fields, and two of the problems they caused."""

    @pytest.fixture
    def frame(self):
        return pd.DataFrame({"a": np.arange(200.0)})

    def test_the_dataclass_field_name_points_at_the_keyword(self, frame):
        """Someone who read `ComputeOptions` and typed what they found there was
        told it was unknown — true, and useless."""
        with pytest.raises(ConfigurationError, match=r"the keyword for it is sample="):
            summarize(frame, numeric_sample_size=5_000)

    def test_a_field_with_no_keyword_says_where_to_set_it(self, frame):
        with pytest.raises(ConfigurationError, match="ProfileConfig"):
            summarize(frame, max_uniques=4_096)

    def test_a_plain_typo_still_lists_the_keywords(self, frame):
        with pytest.raises(ConfigurationError, match="Available: chunk_size"):
            summarize(frame, chunck_size=100)

    def test_the_keyword_itself_still_works(self, frame):
        assert summarize(frame, sample=5_000)["dataset"]["rows_est"] == 200

    def test_checkpoint_settings_are_reachable_as_a_group(self):
        options = ComputeOptions()
        assert options.checkpoint.every_n_chunks == 0
        assert options.checkpoint.prefix == "pysuricata_ckpt"

    def test_writing_through_the_group_sets_the_field(self):
        options = ComputeOptions()
        options.checkpoint.every_n_chunks = 10
        options.checkpoint.dir = "./ckpt"
        assert options.checkpoint_every_n_chunks == 10
        assert options.checkpoint_dir == "./ckpt"

    def test_writing_the_field_shows_through_the_group(self):
        """A lens, not a copy: a copy would be a second place for the settings
        to live, and they would disagree the first time someone set one."""
        options = ComputeOptions()
        options.checkpoint_max_to_keep = 7
        assert options.checkpoint.max_to_keep == 7

    def test_a_typo_in_the_group_lists_the_settings(self):
        options = ComputeOptions()
        with pytest.raises(AttributeError, match="every_n_chunks"):
            options.checkpoint.every_n_chunk = 5

    def test_the_group_reads_back(self):
        assert "every_n_chunks=0" in repr(ComputeOptions().checkpoint)

    def test_existing_code_still_works(self, frame):
        """The fields did not move, so nothing that set them breaks."""
        options = ComputeOptions(checkpoint_every_n_chunks=0, checkpoint_max_to_keep=3)
        assert (
            summarize(frame, config=ProfileConfig(compute=options))["dataset"][
                "rows_est"
            ]
            == 200
        )
