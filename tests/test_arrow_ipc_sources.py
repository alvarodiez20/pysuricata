"""Arrow IPC files load, which is the one format another language writes (#247).

`.arrow`, `.feather` and `.ipc` all raised `UnsupportedDataError`, and
`pa.ipc.open_file(path)` raised `Cannot profile RecordBatchFileReader`. The
split fell exactly on the line between *in-process* Arrow and *on-disk* Arrow:
a `pa.Table` handed over inside one process worked, and the file another
runtime writes did not.

That line is the one that matters. `arrow::write_ipc_file()` in R,
`Arrow.write()` in Julia and the `arrow` crate in Rust all produce on-disk IPC,
so "make Arrow the boundary, not pandas" could not be documentation alone --
the one format it names was the one that did not load.

## Three framings, one set of extensions

The extension does not say which framing a file uses, and dispatching on it
would load R's `write_ipc_file()` output while failing on Julia's, which
defaults to the stream framing. The magic bytes decide:

| magic | framing | reader | streams? |
|---|---|---|---|
| `ARROW1` | IPC file, footer with a batch index | `pa.ipc.open_file` | yes, by index |
| `\\xff\\xff\\xff\\xff` | IPC stream, no footer | `pa.ipc.open_stream` | yes, forward-only |
| `FEA1` | Feather V1 -- not IPC at all | `feather.read_table` | no |

## Why the file reader needed its own branch

`RecordBatchStreamReader` subclasses `pa.RecordBatchReader` and already
qualified. `RecordBatchFileReader` does not subclass it and has no
`to_batches`: the IPC file format's footer indexes every batch, so it offers
random access -- `num_record_batches` and `get_batch(i)` -- instead of an
iterator. Reading by index keeps the bounded-memory promise, where the
`read_all()` its API leads with would materialise the file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.api import UnsupportedDataError

pa = pytest.importorskip("pyarrow")

ROWS = 300

#: Every extension the loader now accepts, and the one it still refuses. `.txt`
#: is here so a change that accepts everything fails rather than looking right.
IPC_SUFFIXES = (".arrow", ".feather", ".ipc")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [i * 1.5 for i in range(ROWS)],
            "region": [f"r{i % 7}" for i in range(ROWS)],
        }
    )


@pytest.fixture(scope="module")
def table(frame: pd.DataFrame):
    return pa.Table.from_pandas(frame, preserve_index=False)


def _write_ipc_file(table, path, *, batches: int = 1) -> str:
    """The `ARROW1` framing -- what `arrow::write_ipc_file()` writes."""
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            for batch in table.to_batches(max_chunksize=max(1, len(table) // batches)):
                writer.write_batch(batch)
    return str(path)


def _write_ipc_stream(table, path) -> str:
    """The `\\xff\\xff\\xff\\xff` framing -- Julia's `Arrow.write()` default."""
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def _rows(payload: dict) -> int:
    return payload["dataset"]["rows_est"]


class TestEveryEntryPointLoads:
    """Four independent entry points. One passing says nothing about the
    others, which is why the issue asked for a test per shape."""

    @pytest.mark.parametrize("suffix", IPC_SUFFIXES)
    def test_a_path_loads(self, table, tmp_path, suffix):
        path = _write_ipc_file(table, tmp_path / f"data{suffix}")

        assert _rows(summarize(path, seed=0)) == ROWS

    def test_a_record_batch_file_reader_loads(self, table, tmp_path):
        """The reader `pa.ipc.open_file()` hands back. Not a
        `RecordBatchReader`, and has no `to_batches`."""
        reader = pa.ipc.open_file(_write_ipc_file(table, tmp_path / "d.arrow"))

        assert not isinstance(reader, pa.RecordBatchReader)
        assert _rows(summarize(reader, seed=0)) == ROWS

    def test_a_record_batch_stream_reader_loads(self, table, tmp_path):
        reader = pa.ipc.open_stream(_write_ipc_stream(table, tmp_path / "d.arrow"))

        assert _rows(summarize(reader, seed=0)) == ROWS


class TestTheFramingIsSniffedNotAssumed:
    """The extension is not evidence. All three framings are legal under all
    three extensions, and two of the three are what a non-Python runtime
    actually writes."""

    @pytest.mark.parametrize("suffix", IPC_SUFFIXES)
    def test_the_stream_framing_loads_under_every_extension(
        self, table, tmp_path, suffix
    ):
        """Julia's `Arrow.write()` writes this one. Dispatching on the suffix
        alone would send it to `open_file`, which raises `ArrowInvalid`."""
        path = _write_ipc_stream(table, tmp_path / f"julia{suffix}")

        assert _rows(summarize(path, seed=0)) == ROWS

    def test_the_two_ipc_framings_are_actually_different_on_disk(self, table, tmp_path):
        """Guards the premise. If both writers produced the same bytes the
        sniffing above would be untested ceremony."""
        as_file = open(_write_ipc_file(table, tmp_path / "f.arrow"), "rb").read(6)
        as_stream = open(_write_ipc_stream(table, tmp_path / "s.arrow"), "rb").read(6)

        assert as_file.startswith(b"ARROW1")
        assert not as_stream.startswith(b"ARROW1")

    def test_feather_v1_still_loads(self, table, tmp_path):
        """Not Arrow IPC at all -- a different container that pyarrow
        deprecated writing in 25.0.0. It has no streaming reader, so it is read
        whole; refusing it would be a regression for anyone holding one."""
        feather = pytest.importorskip("pyarrow.feather")
        path = tmp_path / "legacy.feather"
        with pytest.warns(Warning):  # pyarrow's own deprecation notice
            feather.write_feather(table, str(path), version=1)

        assert open(path, "rb").read(4) == b"FEA1"
        assert _rows(summarize(str(path), seed=0)) == ROWS


class TestItStreamsRatherThanMaterialising:
    """The point of reading these directly. If a multi-batch file came back as
    one frame, this would be `pd.read_feather` with extra steps."""

    def test_a_multi_batch_file_is_read_one_batch_at_a_time(self, table, tmp_path):
        from pysuricata import sources

        path = _write_ipc_file(table, tmp_path / "many.arrow", batches=6)
        batches = list(sources.stream_ipc(path))

        assert len(batches) > 1, (
            "the file was read as a single batch, so nothing about this path "
            "is bounded in memory"
        )
        assert sum(len(b) for b in batches) == ROWS

    def test_a_single_batch_file_arrives_as_a_frame(self, table, tmp_path):
        """The documented consequence in `sources.py`: a stream cannot support
        the distinct-value evidence type inference uses, so a file that fits in
        one batch is handed over as a frame and behaves exactly as before."""
        from pysuricata import sources

        path = _write_ipc_file(table, tmp_path / "one.arrow")

        assert isinstance(
            sources.first_batch_or_stream(sources.stream_ipc(path)), pd.DataFrame
        )


class TestTheResultMatchesTheSameDataViaPandas:
    """A loader that returns *something* is not enough -- it has to be the same
    dataset."""

    @pytest.mark.parametrize("suffix", IPC_SUFFIXES)
    def test_the_payload_agrees_with_the_frame(self, frame, table, tmp_path, suffix):
        path = _write_ipc_file(table, tmp_path / f"same{suffix}")

        from_disk = summarize(path, seed=0)
        from_memory = summarize(frame, seed=0)

        assert _rows(from_disk) == _rows(from_memory)
        assert set(from_disk["columns"]) == set(from_memory["columns"])
        for name in from_memory["columns"]:
            assert (
                from_disk["columns"][name]["missing"]
                == from_memory["columns"][name]["missing"]
            )


class TestTheRefusalStillRefuses:
    def test_an_unknown_suffix_is_still_an_error(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")

        with pytest.raises(UnsupportedDataError) as raised:
            summarize(str(path), seed=0)

        assert ".txt" in str(raised.value)

    def test_the_message_lists_the_formats_that_do_work(self, tmp_path):
        """The message named three formats and the loader now takes six. A
        message that undersells is how someone converts a file they did not
        need to convert."""
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")

        with pytest.raises(UnsupportedDataError) as raised:
            summarize(str(path), seed=0)

        for suffix in (*IPC_SUFFIXES, ".csv", ".parquet", ".json"):
            assert suffix in str(raised.value), suffix
