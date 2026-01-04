import pytest

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from pysuricata.api import ComputeOptions, ProfileConfig, profile


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_profile_polars_basic():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10.0, 20.0, None, 40.0, 50.0],
            "c": [True, False, True, None, False],
            "d": pl.date_range(
                low=pl.datetime(2024, 1, 1),
                high=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "e": ["x", "y", "x", "z", None],
        }
    )
    rep = profile(df, config=ProfileConfig())
    assert rep.html and isinstance(rep.html, str)
    assert isinstance(rep.stats, dict)
    assert rep.stats.get("dataset") is not None


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_profile_polars_iterable_chunks():
    # Build a slightly larger frame and split manually into polars chunks
    n = 100
    df = pl.DataFrame(
        {
            "a": list(range(n)),
            "b": [float(i) if i % 7 else None for i in range(n)],
            "c": [(i % 2) == 0 for i in range(n)],
            "e": ["x" if i % 3 == 0 else "y" for i in range(n)],
        }
    )

    step = 17
    chunks = [df.slice(i, min(step, n - i)) for i in range(0, n, step)]
    rep = profile(iter(chunks), config=ProfileConfig())
    assert rep.html and isinstance(rep.html, str)
    assert isinstance(rep.stats, dict)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_profile_polars_lazyframe_windowed():
    n = 123
    lf = pl.LazyFrame(
        {
            "x": list(range(n)),
            "y": [float(i) if i % 5 else None for i in range(n)],
            "z": ["a" if i % 2 else "b" for i in range(n)],
        }
    ).with_columns(pl.col("x") * 2)
    # LazyFrame should be supported (collected internally)
    rep = profile(lf, config=ProfileConfig(compute=ComputeOptions(chunk_size=17)))
    assert rep.html and isinstance(rep.html, str)
    assert isinstance(rep.stats, dict)

