"""Rebuild `docs/assets/bike_sharing.csv` from the UCI source (#150).

Titanic could not exercise what the report does: no datetime column, so the
datetime card and its temporal panels never rendered in the one example anybody
looks at, and no numeric pair above 0.5, so the correlations section always took
the weak-result route.

This dataset is chosen for what it has rather than for what it is about:

* a real timestamp, hourly across two calendar years, so all four temporal
  panels render -- the year panel is dropped by the renderer inside a single
  year, which is why both years are kept
* ``temp`` and ``feels_like`` at r = 0.99, and ``registered`` with ``rentals``
  at 0.97, so the ranked list and the triangle matrix have something to draw
* two boolean columns and two categoricals, so all four card kinds appear
* 17,379 rows, which is 20x Titanic and enough that chunking is doing something

**The vendored file is the deliverable; this script is how it was made.** CI
never runs it and never reaches the network -- that is the constraint #150 sets
and the reason the CSV is committed rather than downloaded on demand.

Run it only to regenerate:

    python scripts/build_demo_dataset.py path/to/hour.csv

Source: UCI Machine Learning Repository, Bike Sharing Dataset, Fanaee-T and
Gama (2013), <https://doi.org/10.24432/C5W894>. Use in publications must cite
that paper; see `docs/assets/bike_sharing.NOTICE`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs" / "assets" / "bike_sharing.csv"

#: The source encodes both as small integers. A bar chart of `1, 2, 3, 4` is a
#: chart of nothing, and the demo exists to be read.
SEASON = {1: "winter", 2: "spring", 3: "summer", 4: "autumn"}
WEATHER = {1: "clear", 2: "mist", 3: "light rain", 4: "heavy rain"}


def build(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source)
    return pd.DataFrame(
        {
            # The source splits the timestamp across `dteday` and `hr`, which
            # profiles as a datetime column whose every value is midnight --
            # the hour-of-day panel would be one bar. Recombining them is the
            # single change that makes this dataset worth vendoring.
            "rented_at": pd.to_datetime(raw["dteday"])
            + pd.to_timedelta(raw["hr"], unit="h"),
            "season": raw["season"].map(SEASON),
            "weather": raw["weathersit"].map(WEATHER),
            "holiday": raw["holiday"].astype(bool),
            "working_day": raw["workingday"].astype(bool),
            # Normalised in the source, and left that way: rescaling to degrees
            # would be inventing units the file does not carry.
            "temp": raw["temp"].round(2),
            "feels_like": raw["atemp"].round(2),
            "humidity": raw["hum"].round(2),
            "windspeed": raw["windspeed"].round(2),
            "casual": raw["casual"],
            "registered": raw["registered"],
            "rentals": raw["cnt"],
        }
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print("error: pass the path to the UCI hour.csv", file=sys.stderr)
        return 2

    frame = build(Path(sys.argv[1]))
    # Seconds are always zero and so are minutes; printing them costs 51 KB to
    # say nothing.
    frame.to_csv(DEST, index=False, date_format="%Y-%m-%dT%H:%M")
    print(
        f"wrote {DEST.relative_to(REPO)}: {len(frame):,} rows, "
        f"{DEST.stat().st_size / 1024:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
