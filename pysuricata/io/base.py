from __future__ import annotations

import warnings
from itertools import tee
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

FrameLike = Any  # engine-native frame (e.g., pandas.DataFrame)


def iter_chunks(
    data: Union[FrameLike, Iterable[FrameLike]],
    *,
    chunk_size: Optional[int] = 200_000,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[FrameLike]:
    """Yield DataFrame chunks from in-memory objects only.

    Supports:
    - In-memory pandas.DataFrame (sliced by rows)
    - In-memory polars.DataFrame (native chunks, no conversion)
    - An iterable of pandas or polars DataFrames (pass-through)

    Notes:
    - This adapter prefers to preserve the native frame backend when possible,
      allowing the engine to consume either pandas or polars chunks efficiently.
    - Optional dependencies (pandas/polars) are imported lazily.
    """

    # Iterable of frames (pandas or polars); light validation via peek
    try:
        if (
            hasattr(data, "__iter__")
            and not hasattr(data, "__array__")
            and not isinstance(data, (str, bytes, bytearray))
        ):
            it1, it2 = tee(iter(data))  # type: ignore[arg-type]
            try:
                first = next(it2)
            except StopIteration:
                return
            ok = False
            try:
                import pandas as pd  # type: ignore

                if isinstance(first, pd.DataFrame):
                    ok = True
            except Exception:
                pass
            try:
                import polars as pl  # type: ignore

                if not ok and isinstance(first, pl.DataFrame):
                    ok = True
            except Exception:
                pass
            if not ok:
                warnings.warn(
                    "iter_chunks received an iterable whose first element is not a pandas/polars DataFrame; passing through anyway.",
                    RuntimeWarning,
                )
            # yield first and the rest
            yield first
            for ch in it2:
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
        # Lazy polars: windowed slice+collect to avoid full materialization
        if (
            hasattr(data, "collect")
            and hasattr(data, "slice")
            and data.__class__.__name__ == "LazyFrame"
        ):  # lazy polars
            lf = data
            if columns is not None:
                try:
                    lf = lf.select(list(columns))
                except Exception:
                    pass
            step = int(chunk_size or 200_000)
            offset = 0
            while True:
                try:
                    ch = lf.slice(offset, step).collect()
                except Exception:
                    break
                h = getattr(ch, "height", None)
                if h is None:
                    try:
                        h = len(ch)
                    except Exception:
                        h = 0
                if not h:
                    break
                yield ch
                if h < step:
                    break
                offset += step
            return
    except Exception:
        pass

    raise TypeError(
        "Unsupported input for iter_chunks. Provide an in-memory pandas/polars DataFrame, or an iterable of pandas DataFrames."
    )
