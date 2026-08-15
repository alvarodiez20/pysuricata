"""Gate a dataset on shape drift, with an exit code.

`profile` and `summarize` both exit 0 no matter what they found, which makes
them useless in a pipeline. This module compares a run against a stored
baseline and reports what moved, so CI can fail on it.

The position this occupies: every existing gate (Great Expectations, Soda,
pointblank) asks you to author expectations first. A profiler already knows the
shape of yesterday's data, so it can gate with no configuration at all — the
baseline *is* the expectation.

Three things shape the defaults.

**Approximate quantities need loose thresholds.** `unique_est` is a KMV
estimate with relative error about `1/sqrt(k)` — 2.2% at the default k=2048. A
cardinality threshold anywhere near that fires on sketch noise rather than on
drift, so the default sits an order of magnitude above it and
`Thresholds.warnings()` names any threshold that does not.

**Drift in a distribution is measured in standard deviations, not percent.** A
relative change in the mean is meaningless when the mean is near zero, and
incomparable across columns with different units. `|Δmean| / σ_baseline` is
neither.

**Growth is not drift.** Appending rows is the normal life of a dataset, so row
count drift is off by default; a *distribution* that moved is what you want to
hear about. The same reasoning keeps "a column appeared" off by default and "a
column vanished" on.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from . import __version__

# The gate is thresholds applied to the deltas `comparison` computes. Sharing
# the readers is what keeps `check` and `compare` from disagreeing about what
# a change is.
from .comparison import DEFAULT_UNIQUES_K, KMV_RELATIVE_ERROR_PCT
from .comparison import missing_pct as _missing_pct
from .comparison import number as _number
from .comparison import true_rate as _true_rate
from .report import SUMMARY_SCHEMA_VERSION

__all__ = [
    "BASELINE_VERSION",
    "Baseline",
    "CheckResult",
    "Finding",
    "Thresholds",
    "compare",
    "make_baseline",
    "parse_duration",
    "read_baseline",
    "read_thresholds",
    "write_baseline",
]

# Bumped when the baseline envelope changes shape. The payload inside carries
# its own `schema_version`; both are checked on read.
BASELINE_VERSION = 1

# KMV relative error is ~1/sqrt(k). Surfaced so a threshold below the noise
# floor can be called out instead of silently producing a flaky gate. Defined
# with the deltas, so the gate and the diff agree on where noise ends.
_DEFAULT_UNIQUES_K = DEFAULT_UNIQUES_K
_KMV_RELATIVE_ERROR_PCT = KMV_RELATIVE_ERROR_PCT

# `min_ts` and `max_ts` are epoch **nanoseconds** in the payload, which is the
# kind of thing a consumer guesses wrong exactly once.
_NS_PER_SECOND = 1_000_000_000

# Thresholds that accept "26h" as well as a number of seconds.
_DURATION_FIELDS = frozenset({"max_age"})

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400, "w": 604_800}


def parse_duration(text: str | float | int) -> float:
    """Parse `"26h"`, `"3d"`, `"90m"` or a bare number of seconds.

    Args:
        text: A duration string, or a number already in seconds.

    Returns:
        Seconds.

    Raises:
        ValueError: If the string is not a number followed by s, m, h, d or w.
    """
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    raw = str(text).strip().lower()
    if not raw:
        raise ValueError("duration must not be empty")
    unit = _DURATION_UNITS.get(raw[-1])
    number, factor = (raw[:-1], unit) if unit is not None else (raw, 1)
    try:
        value = float(number)
    except ValueError:
        raise ValueError(
            f"cannot read {text!r} as a duration. Use a number of seconds, or a "
            "number followed by s, m, h, d or w -- for example '26h'."
        ) from None
    if value < 0:
        raise ValueError(f"duration must not be negative, got {text!r}")
    return value * factor


@dataclass(frozen=True)
class Thresholds:
    """What counts as a breach.

    Every field is a ceiling: a value at or below it passes. `None` disables
    the check entirely, which is how the growth-shaped ones ship.

    Attributes:
        max_missing_pct: Absolute gate. Any column missing more than this
            percentage fails, with or without a baseline.
        min_rows: Absolute gate. Fewer rows than this fails.
        max_rows_drift_pct: Row count change against the baseline. Off by
            default: appending rows is not drift.
        max_missing_drift_pp: Change in a column's missing rate, in
            **percentage points** — 1% to 7% is a 6-point move.
        max_unique_drift_pct: Change in the approximate distinct count. Must
            stay well above the KMV error of ~2.2% at the default k.
        max_mean_shift_sigma: Mean shift in baseline standard deviations.
        max_median_shift_sigma: Median shift in baseline standard deviations.
        max_std_ratio: Spread change, as the larger of the two ratios. 2.0
            means "the column may not double or halve its standard deviation".
        max_true_rate_drift_pp: Boolean columns: change in the share of True,
            in percentage points.
        max_age: Absolute gate, in seconds, or a duration string like `"26h"`.
            Fails when the newest timestamp in a datetime column is older than
            this. Needs no baseline.
        fail_on_new_column: Whether an added column is a breach. Off by
            default — new columns are usually a feature landing.
        fail_on_range_expansion: Whether a new min below, or max above, the
            baseline is a breach. Off by default, since honest new data
            routinely widens a range.
        require_max_ts_advances: Whether a datetime column whose newest
            timestamp did not move past the baseline's is a breach. This is the
            one that catches a re-run of yesterday's extract, where every
            distribution matches because the data is literally the same.

    Both time-shaped thresholds are off by default: a datetime column can be a
    birth date rather than an event time, and failing a build because nobody
    was born since the baseline would be absurd.
    """

    max_missing_pct: float | None = None
    min_rows: int | None = None
    max_rows_drift_pct: float | None = None
    max_missing_drift_pp: float | None = 5.0
    max_unique_drift_pct: float | None = 25.0
    max_mean_shift_sigma: float | None = 0.5
    max_median_shift_sigma: float | None = 0.5
    max_std_ratio: float | None = 2.0
    max_true_rate_drift_pp: float | None = 10.0
    max_age: float | None = None
    fail_on_new_column: bool = False
    fail_on_range_expansion: bool = False
    require_max_ts_advances: bool = False

    def __post_init__(self) -> None:
        """Accept a duration string wherever seconds are expected.

        `Thresholds(max_age="26h")` is the natural thing to write, so it has to
        work in Python and not only in a thresholds file.
        """
        for name in _DURATION_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, parse_duration(value))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Thresholds:
        """Build from a parsed thresholds file.

        Args:
            data: Mapping of field name to value. A `[thresholds]` or
                `[tool.pysuricata.check]` table is unwrapped by the reader, not
                here.

        Returns:
            The thresholds.

        Raises:
            ValueError: If a key is not a threshold, or a value is the wrong
                type. An unknown key is an error rather than a no-op: a typo in
                a gate's configuration must not silently loosen the gate.
        """
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(
                f"unknown threshold(s): {', '.join(unknown)}. "
                f"Known thresholds: {', '.join(sorted(known))}"
            )
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            # `field.type` is a string under `from __future__ import
            # annotations`, so the default's type is the reliable discriminator.
            if isinstance(known[name].default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be true or false, got {value!r}")
                kwargs[name] = value
            elif name in _DURATION_FIELDS and isinstance(value, str):
                kwargs[name] = parse_duration(value)
            elif value is None:
                kwargs[name] = None
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number or null, got {value!r}")
            elif value < 0:
                raise ValueError(f"{name} must not be negative, got {value!r}")
            else:
                kwargs[name] = value
        return cls(**kwargs)

    def warnings(self) -> list[str]:
        """Thresholds tight enough to fire on estimator noise rather than drift."""
        notes: list[str] = []
        limit = self.max_unique_drift_pct
        if limit is not None and limit < 2 * _KMV_RELATIVE_ERROR_PCT:
            notes.append(
                f"max_unique_drift_pct={limit:g} is close to the KMV sketch error "
                f"(~{_KMV_RELATIVE_ERROR_PCT:.1f}% at k={_DEFAULT_UNIQUES_K}); this "
                "gate may fail on estimation noise. Raise it, or profile with a "
                "larger uniques_k."
            )
        return notes


@dataclass(frozen=True)
class Finding:
    """One threshold that was crossed.

    Attributes:
        kind: Machine-readable category — `schema`, `rows`, `missing`,
            `cardinality`, `distribution`, `range` or `boolean`.
        column: Column name, or None for dataset-level findings.
        message: One line, naming what moved and by how much.
        baseline: The baseline value, when there is one.
        current: The observed value.
        approximate: Whether the comparison rests on a sketch estimate rather
            than an exact count.
    """

    kind: str
    column: str | None
    message: str
    baseline: Any = None
    current: Any = None
    approximate: bool = False

    def render(self) -> str:
        """Format for a terminal."""
        where = self.column if self.column is not None else "dataset"
        tail = " (approximate)" if self.approximate else ""
        return f"{where}: {self.message}{tail}"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of a comparison.

    Attributes:
        findings: Every threshold crossed, dataset-level first.
        checked_columns: Columns compared against the baseline.
        notes: Non-fatal remarks — an unusable threshold, a column skipped.
    """

    findings: tuple[Finding, ...] = ()
    checked_columns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """True when nothing crossed a threshold."""
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, for `--json`."""
        return {
            "passed": self.passed,
            "findings": [asdict(f) for f in self.findings],
            "checked_columns": list(self.checked_columns),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Baseline:
    """A stored `summarize()` payload plus the provenance to trust it.

    Attributes:
        summary: The payload, exactly as `summarize()` returned it.
        created_at: UTC ISO-8601 timestamp.
        pysuricata_version: The version that produced it.
        source: What it was produced from, for the error message when it does
            not match what is being checked.
    """

    summary: Mapping[str, Any]
    created_at: str = ""
    pysuricata_version: str = ""
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The on-disk envelope."""
        return {
            "baseline_version": BASELINE_VERSION,
            "created_at": self.created_at,
            "pysuricata_version": self.pysuricata_version,
            "source": self.source,
            "summary": self.summary,
        }


def make_baseline(summary: Mapping[str, Any], source: str | None = None) -> Baseline:
    """Wrap a `summarize()` payload as a baseline.

    Args:
        summary: The payload.
        source: Optional description of where the data came from.

    Returns:
        The baseline, stamped with the current time and version.
    """
    return Baseline(
        summary=summary,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        pysuricata_version=__version__,
        source=source,
    )


def write_baseline(baseline: Baseline, path: str | os.PathLike) -> None:
    """Write a baseline as JSON.

    Args:
        baseline: The baseline.
        path: Destination path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline.to_dict(), indent=2, default=_jsonable))


def read_baseline(path: str | os.PathLike) -> Baseline:
    """Read a baseline written by `write_baseline`.

    Args:
        path: Source path.

    Returns:
        The baseline.

    Raises:
        ValueError: If the file is not a baseline, or was written by an
            incompatible version. Both messages say to regenerate it, because
            that is always the fix and it costs one command.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or "summary" not in raw:
        raise ValueError(
            f"{path} is not a pysuricata baseline. Write one with "
            "`pysuricata check <data> --write-baseline <path>`."
        )
    version = raw.get("baseline_version")
    if version != BASELINE_VERSION:
        raise ValueError(
            f"{path} has baseline_version {version!r}, this pysuricata reads "
            f"{BASELINE_VERSION}. Regenerate it with --write-baseline."
        )
    summary = raw["summary"]
    stored_schema = summary.get("schema_version")
    if stored_schema != SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"{path} holds a summary with schema_version {stored_schema!r}, this "
            f"pysuricata produces {SUMMARY_SCHEMA_VERSION}. Regenerate it with "
            "--write-baseline."
        )
    return Baseline(
        summary=summary,
        created_at=raw.get("created_at", ""),
        pysuricata_version=raw.get("pysuricata_version", ""),
        source=raw.get("source"),
    )


def compare(
    current: Mapping[str, Any],
    baseline: Baseline | Mapping[str, Any] | None = None,
    thresholds: Thresholds | None = None,
    *,
    now: float | None = None,
) -> CheckResult:
    """Compare a summary against a baseline and any absolute thresholds.

    Args:
        current: A `summarize()` payload for the data under test.
        baseline: A `Baseline`, a bare summary payload, or None to run only the
            absolute thresholds.
        thresholds: What counts as a breach. Defaults to `Thresholds()`.
        now: Unix time to measure freshness against, for tests. Defaults to the
            wall clock.

    Returns:
        The findings. An empty `findings` means the gate passes.
    """
    limits = thresholds if thresholds is not None else Thresholds()
    findings: list[Finding] = []
    notes: list[str] = list(limits.warnings())

    findings.extend(_absolute_findings(current, limits))
    findings.extend(_freshness_findings(current, limits, now))

    if baseline is None:
        return CheckResult(tuple(findings), (), tuple(notes))

    before = baseline.summary if isinstance(baseline, Baseline) else baseline
    findings.extend(_dataset_findings(before, current, limits))

    old_cols = before.get("columns", {}) or {}
    new_cols = current.get("columns", {}) or {}
    findings.extend(_schema_findings(old_cols, new_cols, limits))

    old_rows = _number((before.get("dataset") or {}).get("rows_est"))
    new_rows = _number((current.get("dataset") or {}).get("rows_est"))

    shared = [name for name in new_cols if name in old_cols]
    for name in shared:
        old, new = old_cols[name], new_cols[name]
        if old.get("type") != new.get("type"):
            # Already reported as a retype; comparing a mean against a category
            # count would only add noise.
            continue
        findings.extend(
            _column_findings(
                name, old, new, limits, old_rows=old_rows, new_rows=new_rows
            )
        )

    return CheckResult(tuple(findings), tuple(shared), tuple(notes))


def _absolute_findings(current: Mapping[str, Any], limits: Thresholds) -> list[Finding]:
    """Gates that need no baseline."""
    out: list[Finding] = []
    dataset = current.get("dataset", {}) or {}
    columns = current.get("columns", {}) or {}

    if limits.min_rows is not None:
        rows = _number(dataset.get("rows_est"))
        if rows is not None and rows < limits.min_rows:
            out.append(
                Finding(
                    kind="rows",
                    column=None,
                    message=(
                        f"{int(rows):,} rows, below the required minimum of "
                        f"{limits.min_rows:,}"
                    ),
                    baseline=limits.min_rows,
                    current=rows,
                )
            )

    if limits.max_missing_pct is not None:
        for name, stats in columns.items():
            pct = _missing_pct(stats)
            if pct is not None and pct > limits.max_missing_pct:
                out.append(
                    Finding(
                        kind="missing",
                        column=name,
                        message=(
                            f"{pct:.2f}% missing, above the limit of "
                            f"{limits.max_missing_pct:g}%"
                        ),
                        baseline=limits.max_missing_pct,
                        current=pct,
                    )
                )
    return out


def _dataset_findings(
    before: Mapping[str, Any], current: Mapping[str, Any], limits: Thresholds
) -> list[Finding]:
    """Dataset-level drift."""
    out: list[Finding] = []
    if limits.max_rows_drift_pct is None:
        return out
    old = _number((before.get("dataset") or {}).get("rows_est"))
    new = _number((current.get("dataset") or {}).get("rows_est"))
    if old is None or new is None or old == 0:
        return out
    drift = abs(new - old) / old * 100.0
    if drift > limits.max_rows_drift_pct:
        direction = "more" if new > old else "fewer"
        out.append(
            Finding(
                kind="rows",
                column=None,
                message=(
                    f"{int(new):,} rows, {drift:.1f}% {direction} than the baseline's "
                    f"{int(old):,} (limit {limits.max_rows_drift_pct:g}%)"
                ),
                baseline=old,
                current=new,
            )
        )
    return out


def _schema_findings(
    old_cols: Mapping[str, Any], new_cols: Mapping[str, Any], limits: Thresholds
) -> list[Finding]:
    """Columns that vanished, appeared, or changed kind."""
    out: list[Finding] = []
    for name in old_cols:
        if name not in new_cols:
            out.append(
                Finding(
                    kind="schema",
                    column=name,
                    message="column is missing from the data",
                    baseline=old_cols[name].get("type"),
                    current=None,
                )
            )
    for name, stats in new_cols.items():
        if name in old_cols:
            old_type, new_type = old_cols[name].get("type"), stats.get("type")
            if old_type != new_type:
                out.append(
                    Finding(
                        kind="schema",
                        column=name,
                        message=f"type changed from {old_type} to {new_type}",
                        baseline=old_type,
                        current=new_type,
                    )
                )
        elif limits.fail_on_new_column:
            out.append(
                Finding(
                    kind="schema",
                    column=name,
                    message=f"column is new ({stats.get('type')})",
                    baseline=None,
                    current=stats.get("type"),
                )
            )
    return out


def _column_findings(
    name: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    limits: Thresholds,
    old_rows: float | None = None,
    new_rows: float | None = None,
) -> list[Finding]:
    """Per-column drift, for a column whose type did not change."""
    out: list[Finding] = []
    out.extend(_missing_drift(name, old, new, limits))
    out.extend(_cardinality_drift(name, old, new, limits, old_rows, new_rows))

    kind = new.get("type")
    if kind == "numeric":
        out.extend(_numeric_drift(name, old, new, limits))
    elif kind == "boolean":
        out.extend(_boolean_drift(name, old, new, limits))
    elif kind == "datetime":
        out.extend(_staleness(name, old, new, limits))
    return out


def _staleness(
    name: str, old: Mapping[str, Any], new: Mapping[str, Any], limits: Thresholds
) -> list[Finding]:
    """Did the newest timestamp move past the baseline's?

    The most common failure of a scheduled pipeline is not that the numbers
    drifted — it is that the job produced **yesterday's data again**. Every
    distribution matches, every column is present, and every other check here
    passes, because the data is literally the same.
    """
    if not limits.require_max_ts_advances:
        return []
    before, after = _number(old.get("max_ts")), _number(new.get("max_ts"))
    if before is None or after is None:
        return []
    if after > before:
        return []
    if after < before:
        message = f"newest timestamp moved backwards, {_when(before)} to {_when(after)}"
    else:
        message = f"newest timestamp did not advance past {_when(before)}"
    return [
        Finding(
            kind="freshness",
            column=name,
            message=message,
            baseline=before,
            current=after,
        )
    ]


def _freshness_findings(
    current: Mapping[str, Any], limits: Thresholds, now: float | None = None
) -> list[Finding]:
    """Absolute freshness: how old is the newest timestamp, right now."""
    if limits.max_age is None:
        return []
    now_seconds = time.time() if now is None else now
    out: list[Finding] = []
    for name, stats in (current.get("columns") or {}).items():
        if stats.get("type") != "datetime":
            continue
        newest = _number(stats.get("max_ts"))
        if newest is None:
            continue
        age = now_seconds - newest / _NS_PER_SECOND
        if age > limits.max_age:
            out.append(
                Finding(
                    kind="freshness",
                    column=name,
                    message=(
                        f"newest value is {_duration(age)} old ({_when(newest)}), "
                        f"past the limit of {_duration(limits.max_age)}"
                    ),
                    baseline=limits.max_age,
                    current=age,
                )
            )
    return out


def _when(nanoseconds: float) -> str:
    """Format an epoch-nanosecond timestamp in UTC.

    UTC, not local: the payload stores timestamps as epoch values, and a gate
    that reads them through the runner's timezone would fail differently
    depending on where CI happens to run. The datetime accumulator was bitten
    by exactly that once.
    """
    try:
        moment = _dt.datetime.fromtimestamp(
            nanoseconds / _NS_PER_SECOND, _dt.timezone.utc
        )
    except (OverflowError, OSError, ValueError):
        return "an unrepresentable time"
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _duration(seconds: float) -> str:
    """Format a span of seconds the way a person would say it."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5_400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172_800:
        return f"{seconds / 3_600:.1f}h"
    return f"{seconds / 86_400:.1f}d"


def _missing_drift(
    name: str, old: Mapping[str, Any], new: Mapping[str, Any], limits: Thresholds
) -> list[Finding]:
    if limits.max_missing_drift_pp is None:
        return []
    before, after = _missing_pct(old), _missing_pct(new)
    if before is None or after is None:
        return []
    move = abs(after - before)
    if move <= limits.max_missing_drift_pp:
        return []
    return [
        Finding(
            kind="missing",
            column=name,
            message=(
                f"missing rate moved {move:.2f} points, {before:.2f}% to {after:.2f}% "
                f"(limit {limits.max_missing_drift_pp:g})"
            ),
            baseline=before,
            current=after,
        )
    ]


def _cardinality_drift(
    name: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    limits: Thresholds,
    old_rows: float | None = None,
    new_rows: float | None = None,
) -> list[Finding]:
    """Distinct-count drift, immune to the dataset simply having grown.

    Neither the distinct *count* nor the distinct *rate* is stable under growth
    on its own, and they fail in opposite directions:

    * a three-level categorical keeps its count when rows double, and halves
      its rate;
    * a continuous column keeps its rate, and doubles its count.

    So a gate on either one alone fails every build that appends data.
    Requiring *both* to move is stable: growth moves exactly one of them.

    The cost, stated rather than left to be discovered: when the row count also
    moved a lot, a small change in levels sits inside the band that growth
    alone could explain and is not reported. Three levels becoming five while
    the rows double is an example. When the row count holds — the common CI
    shape, the same query run a day later — this is exactly as sensitive as
    gating on the raw count. `max_rows_drift_pct` is the gate for volume;
    this one is for shape.

    When the row count is unknown, the count is all there is and it is used
    alone.
    """
    if limits.max_unique_drift_pct is None:
        return []
    before, after = _number(old.get("unique_est")), _number(new.get("unique_est"))
    if before is None or after is None or before <= 0:
        return []
    count_drift = abs(after - before) / before * 100.0
    if count_drift <= limits.max_unique_drift_pct:
        return []

    if old_rows and new_rows:
        before_rate = before / old_rows
        after_rate = after / new_rows
        if before_rate > 0:
            rate_drift = abs(after_rate - before_rate) / before_rate * 100.0
            if rate_drift <= limits.max_unique_drift_pct:
                # The count moved and the rate did not: the dataset grew.
                return []

    return [
        Finding(
            kind="cardinality",
            column=name,
            message=(
                f"distinct values moved {count_drift:.1f}%, ~{int(before):,} to "
                f"~{int(after):,} (limit {limits.max_unique_drift_pct:g}%)"
            ),
            baseline=before,
            current=after,
            approximate=True,
        )
    ]


def _numeric_drift(
    name: str, old: Mapping[str, Any], new: Mapping[str, Any], limits: Thresholds
) -> list[Finding]:
    """Shift and spread, measured against the baseline's own scale."""
    out: list[Finding] = []
    sigma = _number(old.get("std"))

    for key, limit, label in (
        ("mean", limits.max_mean_shift_sigma, "mean"),
        ("median", limits.max_median_shift_sigma, "median"),
    ):
        if limit is None or sigma is None or sigma <= 0:
            continue
        before, after = _number(old.get(key)), _number(new.get(key))
        if before is None or after is None:
            continue
        shift = abs(after - before) / sigma
        if shift > limit:
            out.append(
                Finding(
                    kind="distribution",
                    column=name,
                    message=(
                        f"{label} moved {shift:.2f}σ, {before:.6g} to {after:.6g} "
                        f"(limit {limit:g}σ, baseline σ={sigma:.6g})"
                    ),
                    baseline=before,
                    current=after,
                )
            )

    if limits.max_std_ratio is not None:
        after_sigma = _number(new.get("std"))
        if sigma is not None and after_sigma is not None and sigma > 0:
            if after_sigma > 0:
                ratio = max(after_sigma / sigma, sigma / after_sigma)
            else:
                ratio = math.inf
            if ratio > limits.max_std_ratio:
                out.append(
                    Finding(
                        kind="distribution",
                        column=name,
                        message=(
                            f"spread changed {ratio:.2f}×, σ {sigma:.6g} to "
                            f"{after_sigma:.6g} (limit {limits.max_std_ratio:g}×)"
                        ),
                        baseline=sigma,
                        current=after_sigma,
                    )
                )

    if limits.fail_on_range_expansion:
        out.extend(_range_expansion(name, old, new))
    return out


def _range_expansion(
    name: str, old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[Finding]:
    out: list[Finding] = []
    old_min, new_min = _number(old.get("min")), _number(new.get("min"))
    if old_min is not None and new_min is not None and new_min < old_min:
        out.append(
            Finding(
                kind="range",
                column=name,
                message=f"minimum fell below the baseline: {old_min:.6g} to {new_min:.6g}",
                baseline=old_min,
                current=new_min,
            )
        )
    old_max, new_max = _number(old.get("max")), _number(new.get("max"))
    if old_max is not None and new_max is not None and new_max > old_max:
        out.append(
            Finding(
                kind="range",
                column=name,
                message=f"maximum rose above the baseline: {old_max:.6g} to {new_max:.6g}",
                baseline=old_max,
                current=new_max,
            )
        )
    return out


def _boolean_drift(
    name: str, old: Mapping[str, Any], new: Mapping[str, Any], limits: Thresholds
) -> list[Finding]:
    if limits.max_true_rate_drift_pp is None:
        return []
    before, after = _true_rate(old), _true_rate(new)
    if before is None or after is None:
        return []
    move = abs(after - before)
    if move <= limits.max_true_rate_drift_pp:
        return []
    return [
        Finding(
            kind="boolean",
            column=name,
            message=(
                f"true rate moved {move:.2f} points, {before:.2f}% to {after:.2f}% "
                f"(limit {limits.max_true_rate_drift_pp:g})"
            ),
            baseline=before,
            current=after,
        )
    ]


def _jsonable(value: Any) -> Any:
    """Fallback encoder for the numpy scalars the payload still carries."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)


def read_thresholds(path: str | os.PathLike) -> Thresholds:
    """Read a thresholds file.

    JSON and TOML are both accepted. A `[thresholds]` table, or the
    `[tool.pysuricata.check]` table of a `pyproject.toml`, is unwrapped so the
    same file can hold other configuration.

    Args:
        path: Path to a `.json` or `.toml` file.

    Returns:
        The thresholds.

    Raises:
        ValueError: If the suffix is unrecognised, the file cannot be parsed,
            or TOML is asked for on a Python without `tomllib` (3.10) and
            without `tomli` installed.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        data = json.loads(source.read_text())
    elif suffix in (".toml", ".tml"):
        data = _load_toml(source)
    else:
        raise ValueError(f"unsupported thresholds file '{suffix}'. Use .json or .toml.")
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a table of thresholds")
    data = _unwrap_thresholds(data)
    return Thresholds.from_mapping(data)


def _load_toml(source: Path) -> Mapping[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise ValueError(
                "reading TOML needs Python 3.11+ (for tomllib) or the tomli "
                "package. Use a .json thresholds file instead."
            ) from None
    return tomllib.loads(source.read_text())


def _unwrap_thresholds(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the thresholds table inside a file that may hold other things."""
    nested = (
        data.get("tool", {}).get("pysuricata", {}).get("check")
        if isinstance(data.get("tool"), Mapping)
        else None
    )
    if isinstance(nested, Mapping):
        return nested
    inner = data.get("thresholds")
    if isinstance(inner, Mapping):
        return inner
    return data


# Kept out of __all__ deliberately: `render_findings` is the CLI's presentation
# and not something to promise stability on.
def render_findings(result: CheckResult) -> str:
    """Format a result for a terminal."""
    lines: list[str] = []
    for note in result.notes:
        lines.append(f"note: {note}")
    if result.passed:
        checked = len(result.checked_columns)
        lines.append(
            f"check passed — {checked} column{'s' if checked != 1 else ''} compared"
            if checked
            else "check passed"
        )
        return "\n".join(lines)
    count = len(result.findings)
    lines.append(f"check failed — {count} finding{'s' if count != 1 else ''}")
    for finding in result.findings:
        lines.append(f"  {finding.render()}")
    return "\n".join(lines)
