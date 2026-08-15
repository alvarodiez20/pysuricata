"""What changed between two datasets, as a structure rather than a verdict.

`pysuricata.check` answers "should this build fail?". This answers "what moved?"
— every delta, whether or not it crosses a threshold, so a person or a tool can
decide what it means.

The two share their arithmetic. `check` is thresholds applied to the deltas
computed here, which is what keeps the gate and the diff from disagreeing about
what counts as a change. Where they differ is deliberate and worth stating:

* **Category churn** is here and not in the gate. Which values fall in and out
  of a top-k table moves for reasons that are not drift — Misra-Gries keeps
  bounded counters, and the tail of a long list reshuffles on noise — so it is a
  poor thing to fail a build on and exactly the right thing to show a reader.
* **Quantile shift** is reported for every quartile, not just the median. A gate
  wants one number; a diff wants the shape.
* Nothing here is a pass or a fail. `Comparison` has no `passed`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ColumnDelta",
    "Comparison",
    "DatasetDelta",
    "SchemaDelta",
    "compare",
]

# Shared with `check`: both sides of a comparison are `summarize()` payloads, so
# the readers of that payload live in one place.

# KMV relative standard error is ~1/sqrt(k), about 2.2% at the default
# uniques_k=2048. A distinct-count change smaller than that is the estimator
# breathing, not the column moving, and printing it as a finding is the same
# mistake as printing a sketch estimate as an exact integer.
DEFAULT_UNIQUES_K = 2048
KMV_RELATIVE_ERROR_PCT = 100.0 / math.sqrt(DEFAULT_UNIQUES_K)


def number(value: Any) -> float | None:
    """Coerce to float, treating None and NaN alike as "no value"."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def missing_pct(stats: Mapping[str, Any]) -> float | None:
    """Missing as a percentage of the rows the column was seen in."""
    missing = number(stats.get("missing"))
    count = number(stats.get("count"))
    if missing is None or count is None:
        return None
    total = count + missing
    if total <= 0:
        return 0.0
    return missing / total * 100.0


def true_rate(stats: Mapping[str, Any]) -> float | None:
    """Share of True among the non-missing values, as a percentage."""
    true_n = number(stats.get("true"))
    false_n = number(stats.get("false"))
    if true_n is None or false_n is None:
        return None
    total = true_n + false_n
    if total <= 0:
        return None
    return true_n / total * 100.0


def relative_change(before: float | None, after: float | None) -> float | None:
    """Percentage change from `before` to `after`, or None if it has no meaning."""
    if before is None or after is None or before == 0:
        return None
    return (after - before) / abs(before) * 100.0


def shift_in_sigma(
    before: float | None, after: float | None, sigma: float | None
) -> float | None:
    """A shift measured in the baseline's own standard deviations.

    Percent is meaningless when the value is near zero and incomparable across
    columns with different units; sigma is neither.
    """
    if before is None or after is None or sigma is None or sigma <= 0:
        return None
    return (after - before) / sigma


@dataclass(frozen=True)
class SchemaDelta:
    """Columns that appeared, vanished, or changed kind.

    Attributes:
        added: Columns present only in the second dataset, with their type.
        removed: Columns present only in the first, with their type.
        retyped: `column -> (before, after)` for columns whose type changed.
        unchanged: Columns present in both, with the same type.
    """

    added: dict[str, str] = field(default_factory=dict)
    removed: dict[str, str] = field(default_factory=dict)
    retyped: dict[str, tuple[str, str]] = field(default_factory=dict)
    unchanged: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether the schema moved at all."""
        return bool(self.added or self.removed or self.retyped)


@dataclass(frozen=True)
class DatasetDelta:
    """Whole-frame movement.

    Attributes:
        rows_before: Rows in the first dataset.
        rows_after: Rows in the second.
        rows_change_pct: Relative change, or None when the first was empty.
        missing_pct_before: Missing cells as a percentage of all cells.
        missing_pct_after: The same, after.
        memory_bytes_before: Approximate in-memory size before.
        memory_bytes_after: And after.
    """

    rows_before: float | None = None
    rows_after: float | None = None
    rows_change_pct: float | None = None
    missing_pct_before: float | None = None
    missing_pct_after: float | None = None
    memory_bytes_before: float | None = None
    memory_bytes_after: float | None = None


@dataclass(frozen=True)
class ColumnDelta:
    """What moved in one column present in both datasets.

    Every field is a *delta*, not a verdict. `None` means the statistic was not
    available on one side or the other — a categorical column has no mean — and
    is distinct from a delta of zero.

    Attributes:
        name: Column name.
        kind: The column's type, which is the same on both sides by
            construction: a retyped column is a schema finding, not a delta.
        count_before, count_after: Non-missing values.
        missing_pct_before, missing_pct_after: Missing rate, in percent.
        missing_pct_change: Change in **percentage points**.
        unique_before, unique_after: **Approximate** distinct counts.
        unique_change_pct: Relative change in the distinct count.
        distinct_rate_change_pct: Relative change in distinct-per-row, which is
            what separates "the data grew" from "the column changed shape".
        mean_shift_sigma, median_shift_sigma: Shift in baseline sigmas.
        q1_shift_sigma, q3_shift_sigma: The rest of the shape.
        std_ratio: Spread after over spread before.
        range_before, range_after: `(min, max)` for numeric columns.
        true_rate_change_pp: Boolean columns, in percentage points.
        categories_added, categories_removed: Values that entered or left the
            tracked top-k. **Approximate**: top-k membership is not a census.
        top_category_before, top_category_after: The most common value.
        span_days_before, span_days_after: Datetime columns.
        newest_before, newest_after: Newest timestamp, epoch nanoseconds.
    """

    name: str
    kind: str
    count_before: float | None = None
    count_after: float | None = None
    missing_pct_before: float | None = None
    missing_pct_after: float | None = None
    missing_pct_change: float | None = None
    unique_before: float | None = None
    unique_after: float | None = None
    unique_change_pct: float | None = None
    distinct_rate_change_pct: float | None = None
    mean_shift_sigma: float | None = None
    median_shift_sigma: float | None = None
    q1_shift_sigma: float | None = None
    q3_shift_sigma: float | None = None
    std_ratio: float | None = None
    range_before: tuple[float, float] | None = None
    range_after: tuple[float, float] | None = None
    true_rate_change_pp: float | None = None
    categories_added: tuple[str, ...] = ()
    categories_removed: tuple[str, ...] = ()
    top_category_before: str | None = None
    top_category_after: str | None = None
    span_days_before: float | None = None
    span_days_after: float | None = None
    newest_before: float | None = None
    newest_after: float | None = None

    @property
    def approximate(self) -> bool:
        """Whether any delta on this column rests on a sketch estimate."""
        return (
            self.unique_change_pct is not None
            or bool(self.categories_added)
            or bool(self.categories_removed)
        )


@dataclass(frozen=True)
class Comparison:
    """The full diff between two datasets.

    Attributes:
        dataset: Whole-frame movement.
        schema: Columns added, removed and retyped.
        columns: Per-column deltas, for columns present in both.
    """

    dataset: DatasetDelta
    schema: SchemaDelta
    columns: dict[str, ColumnDelta]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "dataset": asdict(self.dataset),
            "schema": {
                "added": dict(self.schema.added),
                "removed": dict(self.schema.removed),
                "retyped": {k: list(v) for k, v in self.schema.retyped.items()},
                "unchanged": list(self.schema.unchanged),
            },
            "columns": {name: asdict(delta) for name, delta in self.columns.items()},
        }

    def __repr__(self) -> str:
        moved = sum(1 for delta in self.columns.values() if _moved(delta))
        return (
            f"<Comparison {len(self.columns)} columns compared, {moved} moved, "
            f"{len(self.schema.added)} added, {len(self.schema.removed)} removed>"
        )


def compare(
    before: Any,
    after: Any,
    *,
    seed: int = 0,
    **options: Any,
) -> Comparison:
    """Compare two datasets and report what changed.

    Args:
        before: The reference. A DataFrame, a path, an Arrow or DuckDB source,
            or a `summarize()` payload.
        after: The dataset to compare against it, in any of the same forms.
        seed: Passed to `summarize()` for both sides. Defaulted rather than left
            to chance so that comparing a dataset against itself is a no-op
            rather than a set of sampling wobbles.
        **options: Any other keyword `summarize()` accepts, applied to both
            sides — comparing two profiles taken with different settings would
            report the settings as drift.

    Returns:
        The comparison.

    Example:
        ```python
        from pysuricata import compare

        diff = compare(january, february)
        diff.schema.added
        diff.columns["amount"].median_shift_sigma
        ```
    """
    # Profiled once each, not once per section: a source may be a generator or a
    # Parquet file, and reading it three times is between wasteful and wrong.
    first = _payload(before, seed, options)
    second = _payload(after, seed, options)
    return Comparison(
        dataset=_dataset_delta(first, second),
        schema=_schema_delta(first, second),
        columns=_column_deltas(first, second),
    )


def _payload(data: Any, seed: int, options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept a profiled payload, or profile the data to get one."""
    if isinstance(data, Mapping) and "columns" in data and "dataset" in data:
        return data
    from .api import summarize

    return summarize(data, seed=seed, **options)


def _dataset_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> DatasetDelta:
    first = before.get("dataset") or {}
    second = after.get("dataset") or {}
    rows_before = number(first.get("rows_est"))
    rows_after = number(second.get("rows_est"))
    return DatasetDelta(
        rows_before=rows_before,
        rows_after=rows_after,
        rows_change_pct=relative_change(rows_before, rows_after),
        missing_pct_before=number(first.get("missing_cells_pct")),
        missing_pct_after=number(second.get("missing_cells_pct")),
        memory_bytes_before=number(first.get("memory_bytes")),
        memory_bytes_after=number(second.get("memory_bytes")),
    )


def _schema_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> SchemaDelta:
    old = before.get("columns") or {}
    new = after.get("columns") or {}
    added = {name: new[name].get("type", "") for name in new if name not in old}
    removed = {name: old[name].get("type", "") for name in old if name not in new}
    retyped: dict[str, tuple[str, str]] = {}
    unchanged: list[str] = []
    for name in new:
        if name not in old:
            continue
        before_type, after_type = old[name].get("type"), new[name].get("type")
        if before_type != after_type:
            retyped[name] = (str(before_type), str(after_type))
        else:
            unchanged.append(name)
    return SchemaDelta(
        added=added, removed=removed, retyped=retyped, unchanged=tuple(unchanged)
    )


def _column_deltas(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, ColumnDelta]:
    old = before.get("columns") or {}
    new = after.get("columns") or {}
    old_rows = number((before.get("dataset") or {}).get("rows_est"))
    new_rows = number((after.get("dataset") or {}).get("rows_est"))

    deltas: dict[str, ColumnDelta] = {}
    for name, stats in new.items():
        if name not in old:
            continue
        previous = old[name]
        # A retyped column is a schema finding. Comparing a mean against a
        # category count would be noise on top of a fact the reader has.
        if previous.get("type") != stats.get("type"):
            continue
        deltas[name] = _column_delta(name, previous, stats, old_rows, new_rows)
    return deltas


def _column_delta(
    name: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    old_rows: float | None,
    new_rows: float | None,
) -> ColumnDelta:
    kind = str(new.get("type", ""))
    fields: dict[str, Any] = {"name": name, "kind": kind}

    fields["count_before"] = number(old.get("count"))
    fields["count_after"] = number(new.get("count"))

    before_missing, after_missing = missing_pct(old), missing_pct(new)
    fields["missing_pct_before"] = before_missing
    fields["missing_pct_after"] = after_missing
    if before_missing is not None and after_missing is not None:
        fields["missing_pct_change"] = after_missing - before_missing

    before_unique, after_unique = (
        number(old.get("unique_est")),
        number(new.get("unique_est")),
    )
    fields["unique_before"] = before_unique
    fields["unique_after"] = after_unique
    fields["unique_change_pct"] = relative_change(before_unique, after_unique)
    if before_unique is not None and after_unique is not None and old_rows and new_rows:
        fields["distinct_rate_change_pct"] = relative_change(
            before_unique / old_rows, after_unique / new_rows
        )

    if kind in ("numeric", "identifier"):
        fields.update(_numeric_delta(old, new))
    elif kind == "boolean":
        before_rate, after_rate = true_rate(old), true_rate(new)
        if before_rate is not None and after_rate is not None:
            fields["true_rate_change_pp"] = after_rate - before_rate
    elif kind == "categorical":
        fields.update(_categorical_delta(old, new))
    elif kind == "datetime":
        fields["span_days_before"] = number(old.get("time_span_days"))
        fields["span_days_after"] = number(new.get("time_span_days"))
        fields["newest_before"] = number(old.get("max_ts"))
        fields["newest_after"] = number(new.get("max_ts"))

    return ColumnDelta(**fields)


def _numeric_delta(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Shift and spread, measured against the baseline's own scale."""
    sigma = number(old.get("std"))
    out: dict[str, Any] = {
        "mean_shift_sigma": shift_in_sigma(
            number(old.get("mean")), number(new.get("mean")), sigma
        ),
        "median_shift_sigma": shift_in_sigma(
            number(old.get("median")), number(new.get("median")), sigma
        ),
        "q1_shift_sigma": shift_in_sigma(
            number(old.get("q1")), number(new.get("q1")), sigma
        ),
        "q3_shift_sigma": shift_in_sigma(
            number(old.get("q3")), number(new.get("q3")), sigma
        ),
    }
    after_sigma = number(new.get("std"))
    if sigma is not None and after_sigma is not None and sigma > 0:
        out["std_ratio"] = after_sigma / sigma

    before_min, before_max = number(old.get("min")), number(old.get("max"))
    after_min, after_max = number(new.get("min")), number(new.get("max"))
    if before_min is not None and before_max is not None:
        out["range_before"] = (before_min, before_max)
    if after_min is not None and after_max is not None:
        out["range_after"] = (after_min, after_max)
    return out


def _categorical_delta(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    """Which values entered and left the tracked top-k.

    Deliberately absent from the gate: top-k membership is not a census, and its
    tail reshuffles on counting noise rather than on drift. As a *report* it is
    the single most legible thing about a categorical column that changed.
    """
    before_items = _items(old.get("top_items"))
    after_items = _items(new.get("top_items"))
    before_names = {value for value, _ in before_items}
    after_names = {value for value, _ in after_items}
    return {
        "categories_added": tuple(sorted(after_names - before_names)),
        "categories_removed": tuple(sorted(before_names - after_names)),
        "top_category_before": before_items[0][0] if before_items else None,
        "top_category_after": after_items[0][0] if after_items else None,
    }


def _items(raw: Any) -> list[tuple[str, int]]:
    """Normalise a top-items list, which may be absent or hold tuples or lists."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    out: list[tuple[str, int]] = []
    for entry in raw:
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            pair = list(entry)
            if len(pair) == 2:
                out.append((str(pair[0]), int(pair[1])))
    return out


def _moved(delta: ColumnDelta) -> bool:
    """Whether anything about this column is different at all."""
    return any(
        value not in (None, 0.0, (), 1.0)
        for name, value in asdict(delta).items()
        if name.endswith(("_change", "_change_pct", "_sigma", "_pp", "_ratio"))
    ) or bool(delta.categories_added or delta.categories_removed)


def render(comparison: Comparison) -> str:
    """Format a comparison for a terminal.

    Not a verdict: every line is a delta, and it is the reader's business what
    the numbers mean.
    """
    lines: list[str] = []
    dataset = comparison.dataset
    if dataset.rows_before is not None and dataset.rows_after is not None:
        change = (
            f" ({dataset.rows_change_pct:+.1f}%)"
            if dataset.rows_change_pct is not None
            else ""
        )
        lines.append(
            f"rows: {int(dataset.rows_before):,} → {int(dataset.rows_after):,}{change}"
        )

    schema = comparison.schema
    for name, kind in schema.added.items():
        lines.append(f"  + {name} ({kind})")
    for name, kind in schema.removed.items():
        lines.append(f"  - {name} ({kind})")
    for name, (was, now) in schema.retyped.items():
        lines.append(f"  ~ {name}: {was} → {now}")

    for name, delta in comparison.columns.items():
        for line in _column_lines(delta):
            lines.append(f"  {name}: {line}")

    return "\n".join(lines) if lines else "no differences"


def _column_lines(delta: ColumnDelta) -> list[str]:
    lines: list[str] = []
    if delta.missing_pct_change:
        lines.append(
            f"missing {delta.missing_pct_before:.2f}% → "
            f"{delta.missing_pct_after:.2f}% ({delta.missing_pct_change:+.2f} pts)"
        )
    if delta.median_shift_sigma:
        lines.append(f"median {delta.median_shift_sigma:+.2f}σ")
    if delta.std_ratio is not None and abs(delta.std_ratio - 1.0) > 1e-9:
        lines.append(f"spread ×{delta.std_ratio:.2f}")
    if delta.true_rate_change_pp:
        lines.append(f"true rate {delta.true_rate_change_pp:+.2f} pts")
    if delta.categories_added:
        lines.append(f"new categories: {', '.join(delta.categories_added)} (approx)")
    if delta.categories_removed:
        lines.append(f"gone: {', '.join(delta.categories_removed)} (approx)")
    if (
        delta.unique_change_pct
        and abs(delta.unique_change_pct) > KMV_RELATIVE_ERROR_PCT
    ):
        # Below the sketch's own error this is noise; the structured delta still
        # carries the raw number for anyone who wants it.
        lines.append(f"~distinct {delta.unique_change_pct:+.1f}% (approx)")
    if delta.newest_before is not None and delta.newest_after is not None:
        moved_days = (delta.newest_after - delta.newest_before) / 86_400e9
        if abs(moved_days) >= 1.0:
            lines.append(f"newest value moved {moved_days:+.1f} days")
    return lines
