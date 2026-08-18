"""Stream Arrow, Parquet and DuckDB without materialising them first.

Every source used to arrive as a pandas or polars frame, which for a Parquet
file or a DuckDB query meant reading the whole thing into memory before
profiling it. That contradicts the one claim the library is positioned on --
bounded memory regardless of dataset size -- for exactly the inputs where the
claim matters most.

The engine already consumes an iterable of chunks, so this is a reader per
source rather than a change to the core. All three arrive as Arrow record
batches, so there is really one reader and two entry points.

**What this is not.** The accumulators take numpy arrays, so each batch is
converted on its way through; this is not a zero-copy Arrow path. What changes
is that one batch is materialised at a time instead of the entire file.

**One behavioural consequence, stated up front.** Type inference reclassifies a
numeric column as categorical from the distinct values it can see, which is
sound evidence only when the whole column is in hand. A stream cannot offer
that: a leading run of one value looks low-cardinality while the column is not.
So a file that arrives in a single batch is handed to the engine as a frame --
identical results to `pd.read_parquet` -- and a file that does not is treated as
what it is, a stream.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "is_arrow_source",
    "is_duckdb_relation",
    "stream_arrow",
    "stream_duckdb",
    "stream_ipc",
    "stream_parquet",
]

# Rows per batch when the caller does not say. Large enough that per-batch
# overhead is noise, small enough that a batch of a wide frame stays modest:
# 64k rows x 20 float64 columns is about 10 MB.
DEFAULT_BATCH_ROWS = 65_536


def stream_parquet(
    path: str | os.PathLike,
    *,
    batch_size: int | None = None,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield a Parquet file one batch at a time.

    Args:
        path: Path to a `.parquet` file.
        batch_size: Rows per batch. Defaults to `DEFAULT_BATCH_ROWS`.
        columns: Optional subset to read. Columns not read are never decoded,
            which is where most of the saving is on a wide file.

    Yields:
        One pandas DataFrame per batch.

    Raises:
        ImportError: If pyarrow is not installed.
        FileNotFoundError: If the file does not exist.
    """
    pq = _require("pyarrow.parquet", "reading Parquet")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    # `pre_buffer=True` is pyarrow's default and it is aimed at remote storage:
    # it schedules ahead-of-time reads for row groups the caller has not asked
    # for yet, trading memory for hiding read latency. `stream_parquet` only
    # ever sees a local path (`resolved.exists()` above), so there is no
    # latency to hide and the prefetch is pure retained memory -- measured at
    # roughly 2x this reader's own working set on a text-heavy file under a
    # 512 MB ceiling (#92).
    handle = pq.ParquetFile(resolved, pre_buffer=False)
    batches = handle.iter_batches(
        batch_size=batch_size or DEFAULT_BATCH_ROWS, columns=columns
    )
    for batch in batches:
        yield batch.to_pandas()


#: The first bytes of each on-disk framing that can wear a `.arrow`,
#: `.feather` or `.ipc` extension. The extension does not say which one it is,
#: so the magic is what decides -- see `stream_ipc`.
_IPC_FILE_MAGIC = b"ARROW1"
_FEATHER_V1_MAGIC = b"FEA1"


def stream_ipc(
    path: str | os.PathLike, *, batch_size: int | None = None
) -> Iterator[pd.DataFrame]:
    """Yield an Arrow IPC file one batch at a time.

    This is the format another runtime writes -- `arrow::write_ipc_file()` in
    R, `Arrow.write()` in Julia, the `arrow` crate in Rust -- which is what
    makes it worth reading directly rather than through a conversion (#247).

    **Three different framings can wear these extensions, and the extension
    does not say which.** Dispatching on the suffix would load an R
    `write_ipc_file()` and fail on an `Arrow.write()`, since Julia defaults to
    the stream framing. So the first bytes decide:

    | magic | framing | read by |
    |---|---|---|
    | `ARROW1` | IPC file, with a footer | `pa.ipc.open_file` |
    | `\xff\xff\xff\xff` | IPC stream, no footer | `pa.ipc.open_stream` |
    | `FEA1` | Feather V1, not IPC at all | `feather.read_table` |

    Only the last materialises, and that format is legacy -- pyarrow itself
    deprecated writing it in 25.0.0 -- so it has no streaming reader to use.

    Args:
        path: Path to a `.arrow`, `.feather` or `.ipc` file.
        batch_size: Rows per batch, where the framing lets us choose. The two
            IPC framings carry the batching their writer chose and are read as
            they were written.

    Yields:
        One pandas DataFrame per batch.

    Raises:
        ImportError: If pyarrow is not installed.
        FileNotFoundError: If the file does not exist.
    """
    pa = _require("pyarrow", "reading Arrow IPC")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    with open(resolved, "rb") as handle:
        magic = handle.read(6)

    if magic.startswith(_FEATHER_V1_MAGIC):
        feather = _require("pyarrow.feather", "reading Feather V1")
        yield from stream_arrow(feather.read_table(resolved), batch_size=batch_size)
        return

    opener = (
        pa.ipc.open_file if magic.startswith(_IPC_FILE_MAGIC) else pa.ipc.open_stream
    )
    yield from stream_arrow(opener(resolved), batch_size=batch_size)


def stream_arrow(
    source: Any, *, batch_size: int | None = None
) -> Iterator[pd.DataFrame]:
    """Yield an Arrow source one batch at a time.

    Args:
        source: A `RecordBatchReader`, `Table`, `RecordBatch`, or a
            `pyarrow.dataset.Dataset`.
        batch_size: Rows per batch, where the source lets us choose. A
            `RecordBatchReader` already carries its own batching and is passed
            through as it is.

    Yields:
        One pandas DataFrame per batch.

    Raises:
        ImportError: If pyarrow is not installed.
        TypeError: If the object is not an Arrow source.
    """
    pa = _require("pyarrow", "reading Arrow data")
    rows = batch_size or DEFAULT_BATCH_ROWS

    if isinstance(source, pa.RecordBatchReader):
        # Its batching is a property of whatever produced it; re-chunking here
        # would mean buffering, which is the thing being avoided.
        for batch in source:
            yield batch.to_pandas()
        return

    if isinstance(source, pa.Table):
        for batch in source.to_batches(max_chunksize=rows):
            yield batch.to_pandas()
        return

    if isinstance(source, pa.RecordBatch):
        yield source.to_pandas()
        return

    # `RecordBatchFileReader` -- what `pa.ipc.open_file()` returns (#247). It
    # is the one Arrow reader that is *not* a `RecordBatchReader`: the IPC file
    # format has a footer listing every batch, so it offers random access by
    # index instead of a forward-only iterator. Reading by index keeps the
    # bounded-memory promise; `read_all()` would materialise the file.
    #
    # Duck-typed on the pair of members rather than the class name, for the
    # reason `_duckdb_reader` is: a check that names a class breaks when the
    # class moves, and these two members are specific enough together.
    count = getattr(source, "num_record_batches", None)
    get_batch = getattr(source, "get_batch", None)
    if isinstance(count, int) and callable(get_batch):
        for index in range(count):
            yield get_batch(index).to_pandas()
        return

    to_batches = getattr(source, "to_batches", None)
    if callable(to_batches):  # pyarrow.dataset.Dataset and friends
        for batch in to_batches(batch_size=rows):
            yield batch.to_pandas()
        return

    if hasattr(source, "__arrow_c_stream__"):
        # The PyCapsule interface: whatever produced this, pyarrow can read it.
        for batch in pa.RecordBatchReader.from_stream(source):
            yield batch.to_pandas()
        return

    raise TypeError(
        f"{type(source).__name__} is not an Arrow source. Pass a Table, a "
        "RecordBatch, a RecordBatchReader or a Dataset."
    )


def stream_duckdb(
    relation: Any, *, batch_size: int | None = None
) -> Iterator[pd.DataFrame]:
    """Yield a DuckDB relation or query result one batch at a time.

    A relation is a query that has not run yet, so this profiles a result set
    without ever landing it in memory:

    ```python
    con.sql("SELECT * FROM 'events/*.parquet' WHERE ts > '2026-01-01'")
    ```

    Args:
        relation: A DuckDB relation or result — anything with
            `to_arrow_reader`, or `fetch_record_batch` on older DuckDB.
        batch_size: Rows per batch.

    Yields:
        One pandas DataFrame per batch.

    Raises:
        TypeError: If the object cannot produce Arrow batches.
    """
    fetch = _duckdb_reader(relation)
    if fetch is None:
        raise TypeError(
            f"{type(relation).__name__} is not a DuckDB relation. Pass the "
            "result of con.sql(...) or con.execute(...)."
        )
    for batch in fetch(batch_size or DEFAULT_BATCH_ROWS):
        yield batch.to_pandas()


def _duckdb_reader(relation: Any):
    """The relation's batch-reader method, whichever name this DuckDB uses.

    `fetch_record_batch` is deprecated in favour of `to_arrow_reader`; both are
    accepted so the reader works either side of that rename.
    """
    for name in ("to_arrow_reader", "fetch_record_batch"):
        method = getattr(relation, name, None)
        if callable(method):
            return method
    return None


def is_arrow_source(obj: Any) -> bool:
    """Whether `stream_arrow` can read this object.

    Two ways to qualify. A pyarrow object is recognised by module name, without
    importing pyarrow — an import inside a type check would make every
    `profile()` call pay for a dependency the caller may not have. Anything
    else qualifies by exporting `__arrow_c_stream__`, the Arrow PyCapsule
    interface, which is how the rest of the ecosystem hands data over without
    anyone agreeing on a type.

    pandas and polars are excluded explicitly rather than by being checked
    first somewhere else: both export the capsule (pandas since 2.2), both have
    their own adapter, and a predicate that is only correct because of the order
    its caller happens to use it in is a trap for the next caller.
    """
    module = type(obj).__module__ or ""
    if module.startswith(("pandas", "polars")):
        return False
    if hasattr(obj, "__arrow_c_stream__"):
        return True
    if not module.startswith("pyarrow"):
        return False
    return type(obj).__name__ in {
        "Table",
        "RecordBatch",
        "RecordBatchReader",
        # `pa.ipc.open_file()` and `open_stream()` (#247). The stream reader is
        # a `RecordBatchReader` subclass and would qualify anyway; the file
        # reader is not, and named here is the only way it qualifies.
        "RecordBatchFileReader",
        "RecordBatchStreamReader",
        "Dataset",
        "FileSystemDataset",
        "InMemoryDataset",
        "UnionDataset",
    }


def is_duckdb_relation(obj: Any) -> bool:
    """Whether `stream_duckdb` can read this object.

    Duck-typed on the batch-reader method, which is specific enough to be safe
    and survives DuckDB moving its classes between modules -- they live in
    `_duckdb` rather than `duckdb`, which a module-name check gets wrong.
    """
    return _duckdb_reader(obj) is not None


def first_batch_or_stream(batches: Iterator[Any]) -> Any:
    """Collapse a single-batch stream back into one frame.

    Type inference treats a stream conservatively, because a prefix is not
    evidence about a column: a leading run of one value looks low-cardinality
    while the column is not, and the decision is never revisited. When the
    source turns out to fit in one batch there is no prefix to be wrong about,
    so handing the engine a frame rather than a generator keeps the result
    identical to reading the file with pandas.

    Args:
        batches: The batch iterator.

    Returns:
        A single DataFrame when the source held exactly one batch, otherwise an
        iterator yielding every batch including the one already read.
    """
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        return _empty_frame()

    try:
        second = next(iterator)
    except StopIteration:
        return first

    def _rest() -> Iterator[Any]:
        yield first
        yield second
        yield from iterator

    return _rest()


def _empty_frame():
    import pandas as pd

    return pd.DataFrame()


def _require(module: str, purpose: str):
    """Import an optional dependency, or say plainly what to install."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        package = module.split(".")[0]
        raise ImportError(
            f"{purpose} needs {package}. Install it with `pip install {package}`."
        ) from e
