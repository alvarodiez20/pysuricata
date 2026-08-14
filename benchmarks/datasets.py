"""Synthetic datasets for PySuricata benchmarks.

Every generator is deterministic given a seed, so a benchmark run is
reproducible and two runs on different machines compare like with like.

The shapes here are chosen to isolate specific costs rather than to look
realistic:

* ``numeric_wide``     — isolates the numeric accumulator and the O(p^2)
                         correlation loop.
* ``categorical_heavy``— isolates string hashing and Misra-Gries, the path
                         that currently goes through ``Series.tolist()``.
* ``datetime_heavy``   — isolates the datetime accumulator, which builds one
                         Python ``datetime`` object per row.
* ``duplicate_rows``   — isolates ``RowKMV``, where u64 row hashes are
                         currently stringified and SHA-1'd.
* ``mixed``            — the end-to-end case, roughly the column mix of a
                         real analytics table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "numeric_wide",
    "categorical_heavy",
    "datetime_heavy",
    "duplicate_rows",
    "mixed",
    "sprinkle_nulls",
    "SUITES",
    "describe",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def numeric_wide(rows: int = 1_000_000, cols: int = 20, seed: int = 0) -> pd.DataFrame:
    """Float64 columns with a deliberately varied distribution mix.

    Column 0 is heavy-tailed (lognormal) so skew/kurtosis are non-trivial;
    column 1 is a large-mean, small-variance series that breaks naive
    ``sum(x^2) - sum(x)^2/n`` variance and correlation formulas; the rest are
    correlated normals so the correlation loop has real work to do.
    """
    g = _rng(seed)
    data = {}
    base = g.standard_normal(rows)
    for c in range(cols):
        if c == 0:
            data[f"num_{c}"] = g.lognormal(0.0, 1.4, rows)
        elif c == 1:
            # mean 1e9, sd 1 — catastrophic cancellation bait
            data[f"num_{c}"] = 1e9 + g.standard_normal(rows)
        elif c == 2:
            data[f"num_{c}"] = g.integers(0, 100, rows).astype("float64")
        else:
            rho = 0.9 - 0.05 * c
            data[f"num_{c}"] = rho * base + np.sqrt(
                max(1e-9, 1 - rho**2)
            ) * g.standard_normal(rows)
    return pd.DataFrame(data)


def categorical_heavy(
    rows: int = 1_000_000, cardinality: int = 50_000, seed: int = 0
) -> pd.DataFrame:
    """Zipf-distributed strings — a few very frequent values, a long tail.

    This is the shape that makes Misra-Gries and KMV both work hard: the head
    hits existing counters, the tail forces evictions.
    """
    g = _rng(seed)
    zipf = g.zipf(1.3, rows) % cardinality
    return pd.DataFrame(
        {
            "sku": np.array([f"SKU-{i:07d}" for i in range(cardinality)])[zipf],
            "country": g.choice(["ES", "US", "DE", "FR", "GB", "IT", "PT"], rows),
            "free_text": np.array(
                [
                    f"note about item {i} recorded by operator {i % 37}"
                    for i in range(2000)
                ]
            )[g.integers(0, 2000, rows)],
        }
    )


def datetime_heavy(rows: int = 1_000_000, seed: int = 0) -> pd.DataFrame:
    """Timestamps: one monotonic, one shuffled, one with a pre-1906 tail.

    ``historic`` exists to catch the validity-window bug: the datetime
    accumulator drops anything before roughly 1906-05-13 and counts it as
    missing rather than as data.
    """
    g = _rng(seed)
    start = np.datetime64("2020-01-01T00:00:00", "ns")
    step = np.timedelta64(37, "s")
    monotonic = start + np.arange(rows) * step
    shuffled = monotonic.copy()
    g.shuffle(shuffled)
    historic = np.datetime64("1850-01-01", "ns") + (
        g.integers(0, 200 * 365, rows) * np.timedelta64(1, "D")
    )
    return pd.DataFrame(
        {"event_at": monotonic, "seen_at": shuffled, "historic": historic}
    )


def duplicate_rows(
    rows: int = 1_000_000, distinct: int = 100_000, seed: int = 0
) -> pd.DataFrame:
    """A frame with a known exact duplicate rate, for RowKMV accuracy checks."""
    g = _rng(seed)
    idx = g.integers(0, distinct, rows)
    return pd.DataFrame(
        {
            "a": idx.astype("int64"),
            "b": (idx * 7 % 1013).astype("float64"),
            "c": np.array([f"g{i % 977}" for i in range(distinct)])[idx],
        }
    )


def mixed(rows: int = 1_000_000, seed: int = 0) -> pd.DataFrame:
    """Roughly the column mix of a real analytics table."""
    g = _rng(seed)
    n = numeric_wide(rows, cols=8, seed=seed)
    c = categorical_heavy(rows, cardinality=20_000, seed=seed + 1)
    d = datetime_heavy(rows, seed=seed + 2)[["event_at"]]
    b = pd.DataFrame(
        {
            "is_active": g.random(rows) > 0.3,
            "is_flagged": g.random(rows) > 0.95,
        }
    )
    return pd.concat([n, c, d, b], axis=1)


def sprinkle_nulls(df: pd.DataFrame, frac: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Replace ``frac`` of each column with nulls, in place of a copy where possible."""
    g = _rng(seed)
    out = df.copy()
    for col in out.columns:
        mask = g.random(len(out)) < frac
        if out[col].dtype.kind == "b":
            out[col] = out[col].astype("object").mask(mask)
        else:
            out.loc[mask, col] = None
    return out


# Suite definitions used by run_all.py. Sizes are deliberately small by
# default so a full run finishes on a laptop; pass --scale to grow them.
SUITES: dict[str, dict] = {
    "numeric_wide": {"fn": numeric_wide, "kwargs": {"rows": 500_000, "cols": 20}},
    "numeric_verywide": {"fn": numeric_wide, "kwargs": {"rows": 100_000, "cols": 200}},
    "categorical_heavy": {"fn": categorical_heavy, "kwargs": {"rows": 500_000}},
    "datetime_heavy": {"fn": datetime_heavy, "kwargs": {"rows": 500_000}},
    "duplicate_rows": {"fn": duplicate_rows, "kwargs": {"rows": 500_000}},
    "mixed": {"fn": mixed, "kwargs": {"rows": 500_000}},
}


def describe(df: pd.DataFrame) -> dict:
    """Shape metadata recorded alongside every benchmark result."""
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "bytes": int(df.memory_usage(deep=True).sum()),
        "dtypes": {
            str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()
        },
    }


def build(name: str, scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    spec = SUITES[name]
    kwargs = dict(spec["kwargs"])
    if "rows" in kwargs:
        kwargs["rows"] = max(1_000, int(kwargs["rows"] * scale))
    kwargs["seed"] = seed
    return spec["fn"](**kwargs)
