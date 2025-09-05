from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional, Sequence, Union


FrameLike = Any  # engine-native frame (e.g., pandas.DataFrame)


def iter_chunks(
    data: Union[FrameLike, Iterable[FrameLike]],
    *,
    chunk_size: Optional[int] = 200_000,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[FrameLike]:
    """Yield pandas DataFrame chunks from in-memory objects only.

    Supports:
    - In-memory pandas.DataFrame (sliced by rows)
    - In-memory polars.DataFrame (native chunks, no pandas conversion)
    - An iterable of pandas DataFrames (pass-through)

    Notes:
    - This adapter yields pandas DataFrames as the common interchange format,
      which is what the compute pipeline consumes today.
    - Optional dependencies (pandas/polars) are imported lazily.
    """

    # Iterable of frames (assumed pandas DataFrames); pass-through
    try:
        if hasattr(data, "__iter__") and not hasattr(data, "__array__"):
            # Avoid mistaking numpy arrays for iterables of frames
            # Avoid strings/bytes which were handled above
            if not isinstance(data, (dict, list, tuple, set)):
                for ch in data:  # type: ignore[assignment]
                    yield ch
                return
    except Exception:
        pass

    # In-memory pandas DataFrame
    try:
        import pandas as pd  # type: ignore
        if isinstance(data, pd.DataFrame):
            n = len(data)
            if n == 0:
                return
            step = int(chunk_size or n)
            for i in range(0, n, step):
                df = data.iloc[i : i + step]
                if columns is not None:
                    # best-effort selection
                    try:
                        df = df[list(columns)]
                    except Exception:
                        pass
                yield df
            return
    except Exception:
        pass

    # In-memory polars DataFrame -> polars chunks
    try:
        import polars as pl  # type: ignore
        if isinstance(data, pl.DataFrame):
            step = int(chunk_size or len(data))
            n = data.height
            use_cols = list(columns) if columns is not None else None
            for i in range(0, n, step):
                ch = data.slice(i, min(step, n - i))
                if use_cols is not None:
                    # best-effort selection
                    try:
                        ch = ch.select(use_cols)
                    except Exception:
                        pass
                yield ch
            return
        if hasattr(data, "collect") and hasattr(data, "slice") and data.__class__.__name__ == "LazyFrame":  # lazy polars
            lf = data
            if columns is not None:
                try:
                    lf = lf.select(list(columns))
                except Exception:
                    pass
            df = lf.collect()
            yield from iter_chunks(df, chunk_size=chunk_size, columns=columns)
            return
    except Exception:
        pass

    raise TypeError(
        "Unsupported input for iter_chunks. Provide an in-memory pandas/polars DataFrame, or an iterable of pandas DataFrames."
    )
