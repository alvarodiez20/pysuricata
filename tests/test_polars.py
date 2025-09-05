import pytest

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from pysuricata.api import profile, ReportConfig


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_profile_polars_basic():
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, None, 40.0, 50.0],
        "c": [True, False, True, None, False],
        "d": pl.date_range(low=pl.datetime(2024,1,1), high=pl.datetime(2024,1,5), interval="1d", eager=True),
        "e": ["x", "y", "x", "z", None],
    })
    rep = profile(df, config=ReportConfig())
    assert rep.html and isinstance(rep.html, str)
    assert rep.stats and isinstance(rep.stats, dict)
    assert rep.stats.get("dataset") is not None
