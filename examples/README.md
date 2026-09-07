# Examples

Three notebooks, in the order they are worth reading.

| notebook | what it shows |
|---|---|
| [`pandas_example.ipynb`](pandas_example.ipynb) | The whole surface on 891 rows: a report, `summarize()`, configuration, streaming an iterable of chunks, and a CI quality gate. |
| [`polars_example.ipynb`](polars_example.ipynb) | The same, for polars `DataFrame` and `LazyFrame`. Needs `pip install pysuricata[polars]`. |
| [`billion_rows.ipynb`](billion_rows.ipynb) | One billion rows in 225 MB, under a 4 GB ceiling the kernel enforces, with the memory sampled throughout and graphed. |

The first two run in seconds and fetch the Titanic CSV over the network.

The third one **reads recorded results rather than producing them**. Its inputs
live in `benchmarks/results/` and were written by `benchmarks/billion.py`; the
headline run takes about 80 minutes and the Parquet fixture is several GB, so
neither is rebuilt on open. Every figure in it is reproducible with the commands
in its last section.

## A note on the configuration name

Both of the first two notebooks use `ProfileConfig`. `ReportConfig` is a
deprecated alias for the same object, removed in 1.0.0, and using it emits a
`DeprecationWarning`. If you find `ReportConfig` in older material, that is
what it is.
