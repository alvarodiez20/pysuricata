#!/usr/bin/env python3
"""Derive the `summarize()` JSON Schema from a live payload, not from prose.

`docs/summary-schema.md` documents the payload's shape by hand, and a hand
maintained document can say one thing while the code does another -- the same
class of problem as a hand-written changelog number. This script builds a
synthetic frame that exercises every column kind (numeric, categorical,
datetime, boolean, identifier) plus the dataset-level features that only show
up conditionally (duplicate rows, missing cells, `top_missing`), calls
`summarize()`, and infers a JSON Schema from the actual keys and value types
in the result.

The output is versioned by `schema_version`: `docs/schemas/summary.v{N}.schema.json`.

Usage:
    python scripts/generate_summary_schema.py --check   # CI: fail if stale
    python scripts/generate_summary_schema.py --write   # regenerate in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"

#: Keys whose value is a sketch's own opaque payload, or otherwise not worth
#: constraining beyond "present". Kept short and explicit rather than a rule
#: guessing at intent.
_LOOSE_ARRAY_ITEMS = {"min_items", "max_items", "top_values", "top_items"}

#: Dicts keyed by data-dependent values (a year, seen only for frames that
#: span one) rather than a fixed set of field names. `required` on a fixed
#: property list is the wrong shape for these -- the next frame's dict has
#: different keys entirely.
_DYNAMIC_KEY_OBJECTS = {"by_year"}

#: Keys documented in docs/summary-schema.md as `<type> | None`, where the
#: `None` branch depends on data this exemplar frame does not happen to
#: contain (a numeric column with no positive value, a datetime column with
#: no detected season). Inferring from one sample would bake in whichever
#: branch the exemplar landed on; these are pinned from the documented
#: contract instead.
_NULLABLE_TYPES = {
    "min_positive": "number",
    "seasonal_pattern": "string",
    "source_timezone": "string",
    "top_values": "array",
}


def _build_dataset() -> Any:
    """A frame exercising every column kind and every conditional dataset field."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 3_000
    df = pd.DataFrame(
        {
            "amount": rng.lognormal(3.0, 1.2, n),
            "id": np.arange(n, dtype=np.int64),
            "category": rng.choice(["alpha", "beta", "gamma", "delta"], size=n),
            "flag": rng.choice([True, False], size=n),
            "when": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )
    df.loc[0:20, "amount"] = np.nan
    df.loc[0:5, "category"] = None
    # A handful of exact-duplicate rows, and enough of them that the KMV
    # sketch resolves the count rather than suppressing it to 0.
    df = pd.concat([df, df.iloc[:200]], ignore_index=True)
    return df


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"summarize() produced a JSON-incompatible value: {value!r}")


def _infer(key: str, value: Any) -> dict[str, Any]:
    if key in _NULLABLE_TYPES:
        return {"type": [_NULLABLE_TYPES[key], "null"]}
    kind = _json_type(value)
    if kind == "object":
        if key in _DYNAMIC_KEY_OBJECTS:
            value_types = {_json_type(v) for v in value.values()}
            item_schema = {"type": value_types.pop()} if len(value_types) == 1 else {}
            return {"type": "object", "additionalProperties": item_schema}
        return {
            "type": "object",
            "properties": {k: _infer(k, v) for k, v in value.items()},
            "required": sorted(value.keys()),
        }
    if kind == "array":
        if not value or key in _LOOSE_ARRAY_ITEMS:
            return {"type": "array"}
        # Homogeneous-enough arrays (by_hour, by_dow, true_histogram_counts, ...):
        # constrain the element type, not the length.
        item_types = {_json_type(v) for v in value}
        if len(item_types) == 1:
            return {"type": "array", "items": {"type": item_types.pop()}}
        return {"type": "array"}
    return {"type": kind}


def _column_schema(payload: dict[str, Any], type_names: list[str]) -> dict[str, Any]:
    schema = _infer("<column>", payload)
    schema["properties"]["type"] = {"type": "string", "enum": sorted(type_names)}
    return schema


def build_schema() -> dict[str, Any]:
    from pysuricata import summarize
    from pysuricata.report import SUMMARY_SCHEMA_VERSION

    payload = dict(summarize(_build_dataset(), seed=0))

    columns = payload["columns"]
    by_type: dict[str, dict[str, Any]] = {}
    for col in columns.values():
        by_type.setdefault(col["type"], col)
    # `identifier` shares the numeric shape (docs/summary-schema.md), so it is
    # folded into the numeric branch's `type` enum rather than sampled fresh --
    # a frame is not guaranteed to produce one on every run.
    if "identifier" not in by_type and "numeric" in by_type:
        by_type["identifier"] = by_type["numeric"]

    column_branches = []
    for type_name, exemplar in sorted(by_type.items()):
        if type_name == "identifier":
            continue  # folded into numeric's enum below, not a separate branch
        branch_types = ["identifier"] if type_name == "numeric" else [type_name]
        if type_name == "numeric":
            branch_types = ["numeric", "identifier"]
        column_branches.append(_column_schema(exemplar, branch_types))

    dataset_schema = _infer("dataset", payload["dataset"])

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://alvarodiez20.github.io/pysuricata/schemas/"
            f"summary.v{SUMMARY_SCHEMA_VERSION}.schema.json"
        ),
        "title": "pysuricata summarize() payload",
        "description": (
            "Generated by scripts/generate_summary_schema.py from a live "
            "summarize() payload -- see docs/summary-schema.md for the "
            "narrative version of this same contract."
        ),
        "type": "object",
        "properties": {
            "schema_version": {"const": SUMMARY_SCHEMA_VERSION},
            "dataset": dataset_schema,
            "columns": {
                "type": "object",
                "additionalProperties": {"oneOf": column_branches},
            },
        },
        "required": ["schema_version", "dataset", "columns"],
    }
    return schema


def _schema_path(schema: dict[str, Any]) -> Path:
    version = schema["properties"]["schema_version"]["const"]
    return SCHEMA_DIR / f"summary.v{version}.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="fail if the checked-in schema is stale"
    )
    mode.add_argument(
        "--write", action="store_true", help="regenerate the schema file in place"
    )
    args = parser.parse_args()

    schema = build_schema()
    text = json.dumps(schema, indent=2, sort_keys=False) + "\n"
    path = _schema_path(schema)

    if args.write:
        SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 0

    if not path.exists():
        print(
            f"{path.relative_to(REPO_ROOT)} does not exist -- run --write",
            file=sys.stderr,
        )
        return 1
    current = path.read_text()
    if current != text:
        print(
            f"{path.relative_to(REPO_ROOT)} is stale -- run "
            "`python scripts/generate_summary_schema.py --write` and commit the diff",
            file=sys.stderr,
        )
        return 1
    print(f"{path.relative_to(REPO_ROOT)} matches the live payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
