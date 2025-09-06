import pandas as pd

from pysuricata import profile, ReportConfig


def test_profile_generates_html_snapshot():
    df = pd.DataFrame(
        {
            "amount": [1.0, 2.5, None, 4.0, 5.5],
            "country": ["US", "US", "DE", None, "FR"],
            "ts": pd.to_datetime([
                "2021-01-01", "2021-01-02", None, "2021-01-04", "2021-01-05"
            ]),
            "flag": [True, False, True, None, False],
        }
    )

    cfg = ReportConfig()
    rep = profile(df, config=cfg)

    html = rep.html
    assert html.startswith("<!DOCTYPE html>")
    # basic anchors / headers
    assert "Variables" in html
    assert "Top missing" in html or "missing" in html.lower()
    # presence of cards grid
    assert "cards-grid" in html
    # some values and labels should be present
    assert "amount" in html
    assert "country" in html
    assert "flag" in html

