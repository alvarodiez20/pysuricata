"""Arrow, Parquet and DuckDB read without being materialised first.

#66. Every source used to arrive as a pandas or polars frame, so profiling a
Parquet file meant reading the whole thing into memory — which contradicts the
one claim the library is positioned on, for exactly the inputs where the claim
matters most.

Two properties are tested hardest:

* the numbers must match profiling the same data through pandas, or the reader
  is a second implementation with its own bugs;
* a file that fits in one batch must still be treated as a **frame**, because
  type inference on a stream cannot use distinct counts and would classify it
  differently.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pysuricata import UnsupportedDataError, profile, summarize
from pysuricata.sources import (
    first_batch_or_stream,
    is_arrow_source,
    is_duckdb_relation,
    stream_arrow,
    stream_duckdb,
    stream_parquet,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _frame(n: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "amount": rng.standard_normal(n),
            "region": rng.choice(["north", "south", "east"], n).astype(object),
            "code": rng.integers(0, 5_000, n),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory, frame):
    """Written with small row groups, so it arrives as many batches."""
    path = tmp_path_factory.mktemp("data") / "many.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path, row_group_size=2_000)
    return path


@pytest.fixture(scope="module")
def single_batch_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one.parquet"
    pq.write_table(pa.Table.from_pandas(_frame(500)), path)
    return path


class TestParquet:
    def test_a_path_profiles(self, parquet_path):
        assert summarize(str(parquet_path), seed=0)["dataset"]["rows_est"] == 20_000

    def test_the_statistics_match_pandas(self, parquet_path, frame):
        via_pandas = summarize(frame, seed=0)["columns"]["amount"]
        via_file = summarize(str(parquet_path), seed=0)["columns"]["amount"]
        assert via_file["count"] == via_pandas["count"]
        assert via_file["mean"] == pytest.approx(via_pandas["mean"], rel=1e-9)
        assert via_file["std"] == pytest.approx(via_pandas["std"], rel=1e-9)
        assert via_file["min"] == via_pandas["min"]
        assert via_file["max"] == via_pandas["max"]

    def test_the_distinct_estimate_matches(self, parquet_path, frame):
        via_pandas = summarize(frame, seed=0)["columns"]["code"]["unique_est"]
        assert summarize(str(parquet_path), seed=0)["columns"]["code"][
            "unique_est"
        ] == (via_pandas)

    def test_a_pathlib_path_works(self, parquet_path):
        assert summarize(parquet_path, seed=0)["dataset"]["rows_est"] == 20_000

    def test_it_renders(self, parquet_path):
        assert "<html" in profile(str(parquet_path), seed=0).html.lower()

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(Exception, match="not found"):
            summarize(str(tmp_path / "nope.parquet"))

    def test_only_the_requested_columns_are_decoded(self, parquet_path):
        batches = list(stream_parquet(parquet_path, columns=["amount"]))
        assert list(batches[0].columns) == ["amount"]

    def test_the_batch_size_is_respected(self, parquet_path):
        batches = list(stream_parquet(parquet_path, batch_size=1_000))
        assert max(len(b) for b in batches) <= 1_000
        assert sum(len(b) for b in batches) == 20_000


class TestSingleBatchFilesStayFrames:
    """The behavioural cliff this reader has to avoid.

    Type inference reclassifies a low-cardinality integer column as categorical
    from the distinct values it can see, which is only sound when the whole
    column is in hand. A stream cannot offer that, so a file small enough to
    arrive whole is handed over as a frame — otherwise upgrading would silently
    change the column types of every small Parquet file.
    """

    def test_a_single_batch_file_comes_back_as_a_frame(self, single_batch_path):
        from pysuricata.api import _coerce_input

        assert isinstance(_coerce_input(str(single_batch_path)), pd.DataFrame)

    def test_a_file_larger_than_one_batch_comes_back_as_a_stream(self, tmp_path):
        """Row groups do not decide this — `iter_batches` reads across them, so
        the line is `DEFAULT_BATCH_ROWS`. Anything under that keeps whole-frame
        semantics, which covers most files people point at by hand."""
        from pysuricata.api import _coerce_input
        from pysuricata.sources import DEFAULT_BATCH_ROWS

        path = tmp_path / "wide.parquet"
        pq.write_table(pa.Table.from_pandas(_frame(DEFAULT_BATCH_ROWS + 1_000)), path)
        assert not isinstance(_coerce_input(str(path)), pd.DataFrame)

    def test_a_small_file_classifies_exactly_as_pandas_does(self, tmp_path):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"grade": rng.integers(0, 12, 800)})
        path = tmp_path / "grades.parquet"
        pq.write_table(pa.Table.from_pandas(frame), path)
        assert (
            summarize(str(path), seed=0)["columns"]["grade"]["type"]
            == summarize(frame, seed=0)["columns"]["grade"]["type"]
            == "categorical"
        )

    def test_first_batch_or_stream_collapses_one_batch(self):
        frame = _frame(10)
        assert first_batch_or_stream(iter([frame])) is frame

    def test_first_batch_or_stream_keeps_both_batches(self):
        result = first_batch_or_stream(iter([_frame(10), _frame(10)]))
        assert sum(len(chunk) for chunk in result) == 20

    def test_an_empty_source_gives_an_empty_frame(self):
        assert first_batch_or_stream(iter([])).empty


class TestArrow:
    def test_a_table_profiles(self, frame):
        assert summarize(pa.Table.from_pandas(frame), seed=0)["dataset"][
            "rows_est"
        ] == (20_000)

    def test_a_record_batch_reader_profiles(self, parquet_path, frame):
        reader = pa.RecordBatchReader.from_batches(
            pa.Table.from_pandas(frame).schema,
            pq.ParquetFile(parquet_path).iter_batches(batch_size=2_000),
        )
        assert summarize(reader, seed=0)["dataset"]["rows_est"] == 20_000

    def test_a_record_batch_profiles(self, frame):
        batch = pa.RecordBatch.from_pandas(frame.head(100))
        assert summarize(batch, seed=0)["dataset"]["rows_est"] == 100

    def test_a_table_matches_pandas(self, frame):
        via_arrow = summarize(pa.Table.from_pandas(frame), seed=0)["columns"]["amount"]
        via_pandas = summarize(frame, seed=0)["columns"]["amount"]
        assert via_arrow["mean"] == pytest.approx(via_pandas["mean"], rel=1e-9)

    def test_a_dataset_profiles(self, parquet_path):
        import pyarrow.dataset as ds

        assert summarize(ds.dataset(parquet_path), seed=0)["dataset"]["rows_est"] == (
            20_000
        )

    def test_recognition_does_not_import_pyarrow_for_a_plain_object(self):
        assert not is_arrow_source(object())

    def test_a_pandas_frame_is_not_an_arrow_source(self):
        """It exports `__arrow_c_stream__` since pandas 2.2, and it has its own
        adapter. Excluded here rather than by call ordering, so the predicate is
        correct on its own."""
        assert not is_arrow_source(pd.DataFrame({"a": [1]}))

    def test_a_polars_frame_is_not_either(self):
        pl = pytest.importorskip("polars")
        assert not is_arrow_source(pl.DataFrame({"a": [1]}))

    def test_a_capsule_exporter_is_recognised(self, frame):
        """The Arrow PyCapsule interface, which is how everything else in the
        ecosystem hands data over without agreeing on a type."""
        assert is_arrow_source(pa.Table.from_pandas(frame))

    def test_something_else_entirely_is_refused(self):
        with pytest.raises(TypeError, match="not an Arrow source"):
            list(stream_arrow(object()))


class TestPolarsStillTakesItsOwnPath:
    """Polars frames export the Arrow capsule too, so ordering matters: they
    must keep their own adapter rather than being streamed through Arrow."""

    def test_a_polars_frame_is_not_diverted(self, frame):
        pl = pytest.importorskip("polars")
        from pysuricata.api import _coerce_input

        assert isinstance(_coerce_input(pl.DataFrame(frame)), pl.DataFrame)


class TestDuckDB:
    @pytest.fixture(scope="class")
    def con(self):
        duckdb = pytest.importorskip("duckdb")
        return duckdb.connect()

    def test_a_relation_profiles(self, con, parquet_path, frame):
        relation = con.sql(f"SELECT * FROM '{parquet_path}'")
        assert summarize(relation, seed=0)["dataset"]["rows_est"] == len(frame)

    def test_a_filtered_query_never_lands_in_memory(self, con, parquet_path, frame):
        """The point of the DuckDB path: profile a result set, not a table."""
        relation = con.sql(f"SELECT * FROM '{parquet_path}' WHERE amount > 0")
        expected = int((frame["amount"] > 0).sum())
        assert summarize(relation, seed=0)["dataset"]["rows_est"] == expected

    def test_the_statistics_match_the_equivalent_pandas_filter(
        self, con, parquet_path, frame
    ):
        relation = con.sql(f"SELECT * FROM '{parquet_path}' WHERE amount > 0")
        via_duckdb = summarize(relation, seed=0)["columns"]["amount"]
        via_pandas = summarize(frame[frame["amount"] > 0], seed=0)["columns"]["amount"]
        assert via_duckdb["mean"] == pytest.approx(via_pandas["mean"], rel=1e-9)
        assert via_duckdb["max"] == via_pandas["max"]

    def test_a_relation_is_recognised(self, con):
        assert is_duckdb_relation(con.sql("SELECT 1 AS a"))

    def test_a_dataframe_is_not(self):
        assert not is_duckdb_relation(pd.DataFrame({"a": [1]}))


class TestTheReadersDirectly:
    """The branches integration tests never reach: bad input, and a missing
    optional dependency. Codecov flagged these on #96, and they are exactly the
    paths a user hits first when something is wrong."""

    def test_stream_parquet_on_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            next(stream_parquet(tmp_path / "absent.parquet"))

    def test_stream_duckdb_on_something_that_is_not_a_relation(self):
        with pytest.raises(TypeError, match="not a DuckDB relation"):
            next(stream_duckdb(pd.DataFrame({"a": [1]})))

    def test_stream_duckdb_yields_frames(self, parquet_path):
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect()
        relation = con.sql(f"SELECT * FROM '{parquet_path}'")
        batches = list(stream_duckdb(relation, batch_size=5_000))
        assert sum(len(b) for b in batches) == 20_000
        assert all(isinstance(b, pd.DataFrame) for b in batches)

    def test_a_missing_optional_dependency_says_what_to_install(self):
        from pysuricata.sources import _require

        with pytest.raises(ImportError, match="pip install nosuchpkg"):
            _require("nosuchpkg.reader", "reading nothing")

    def test_the_capsule_branch_is_reachable_on_its_own(self, frame):
        """An object that is not a pyarrow type but exports the Arrow C stream
        interface — which is how polars, DuckDB and the rest hand data over."""

        class Exporter:
            def __init__(self, table):
                self._table = table

            def __arrow_c_stream__(self, requested_schema=None):
                return self._table.__arrow_c_stream__(requested_schema)

        batches = list(stream_arrow(Exporter(pa.Table.from_pandas(frame))))
        assert sum(len(b) for b in batches) == len(frame)


class TestTheCliStreamsParquetToo:
    """`pysuricata check data.parquet` in CI is the case #92 turns on, so the
    CLI must take the streaming path rather than `pd.read_parquet`."""

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pysuricata.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_summarize_reads_a_parquet_file(self, parquet_path):
        done = self._run("summarize", str(parquet_path), "--quiet")
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout)["dataset"]["rows_est"] == 20_000

    def test_check_writes_a_baseline_from_a_parquet_file(self, parquet_path, tmp_path):
        baseline = tmp_path / "b.json"
        done = self._run(
            "check", str(parquet_path), "--write-baseline", str(baseline), "--quiet"
        )
        assert done.returncode == 0, done.stderr
        assert baseline.exists()

    def test_the_cli_loader_returns_a_stream_for_a_large_file(self, tmp_path):
        from pysuricata.cli import load_data
        from pysuricata.sources import DEFAULT_BATCH_ROWS

        path = tmp_path / "big.parquet"
        pq.write_table(pa.Table.from_pandas(_frame(DEFAULT_BATCH_ROWS + 1_000)), path)
        assert not isinstance(load_data(str(path)), pd.DataFrame)

    def test_an_unsupported_suffix_is_still_refused(self, tmp_path):
        from pysuricata.cli import load_data

        path = tmp_path / "data.docx"
        path.write_text("not really a document")
        with pytest.raises(ValueError, match="unsupported format"):
            load_data(str(path))


class TestCoercionTakesTheRightBranch:
    def test_a_duckdb_relation_becomes_a_generator(self, parquet_path):
        duckdb = pytest.importorskip("duckdb")
        from pysuricata.api import _coerce_input

        con = duckdb.connect()
        coerced = _coerce_input(con.sql(f"SELECT * FROM '{parquet_path}'"))
        assert not isinstance(coerced, pd.DataFrame)
        assert sum(len(chunk) for chunk in coerced) == 20_000

    def test_an_arrow_table_becomes_a_frame_when_it_fits(self, frame):
        from pysuricata.api import _coerce_input

        assert isinstance(
            _coerce_input(pa.Table.from_pandas(frame.head(100))), pd.DataFrame
        )


class TestTheErrorMessageNamesTheNewSources:
    def test_an_unsupported_object_lists_them(self):
        with pytest.raises(UnsupportedDataError, match="DuckDB relation"):
            summarize(object())

    def test_arrow_is_named_too(self):
        with pytest.raises(UnsupportedDataError, match="Arrow"):
            summarize(object())
