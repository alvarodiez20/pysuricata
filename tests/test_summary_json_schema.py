"""The JSON Schema for `summarize()` cannot drift from the payload it describes.

#250. `docs/summary-schema.md` is prose; a non-Python consumer needs a machine
readable document, and a hand-written one is the same class of problem as a
hand-written changelog number -- it says one thing while the code does
another. `scripts/generate_summary_schema.py` derives the schema from a live
payload instead. This test pins two things: that the checked-in schema is
exactly what that script produces right now (so a payload change without a
regeneration fails loud), and that a live payload actually validates against
it (so the schema, once regenerated, is not merely self-consistent but
correct).
"""

from __future__ import annotations

import json

import jsonschema
import numpy as np
import pandas as pd
import pytest

from pysuricata import summarize
from pysuricata.report import SUMMARY_SCHEMA_VERSION
from scripts.generate_summary_schema import _build_dataset, _schema_path, build_schema


def _as_json(payload):
    """Round-trip through JSON, the form a schema actually describes.

    `summarize()` is JSON-*safe*, not JSON itself -- tuples and other
    non-JSON containers survive in the returned mapping and only become
    plain lists once something calls `json.dumps` on it, same as `pysuricata
    summarize --output stats.json` does. Validating the raw mapping would
    reject a tuple as "not an array", which is true of the Python object and
    false of the JSON it produces.
    """
    return json.loads(json.dumps(payload))


@pytest.fixture(scope="module")
def generated_schema():
    return build_schema()


class TestTheCheckedInSchemaMatchesTheGenerator:
    def test_the_file_on_disk_is_exactly_what_generation_produces(
        self, generated_schema
    ):
        path = _schema_path(generated_schema)
        assert path.exists(), (
            f"{path} is missing -- run "
            "`python scripts/generate_summary_schema.py --write`"
        )
        on_disk = json.loads(path.read_text())
        assert on_disk == generated_schema, (
            "docs/schemas/summary.v*.schema.json is stale -- run "
            "`python scripts/generate_summary_schema.py --write` and commit the diff"
        )

    def test_the_schema_is_versioned_by_the_live_schema_version(self, generated_schema):
        assert (
            generated_schema["properties"]["schema_version"]["const"]
            == SUMMARY_SCHEMA_VERSION
        )


class TestALivePayloadValidatesAgainstIt:
    """Self-consistency (the schema matches its own generator) is not the same
    claim as correctness (the schema accepts a real payload). A `oneOf` branch
    that never matches anything, or a `required` list one key too long, passes
    the test above and fails this one.
    """

    def test_the_exemplar_payload_used_to_generate_it_validates(self, generated_schema):
        payload = _as_json(dict(summarize(_build_dataset(), seed=0)))
        jsonschema.validate(instance=payload, schema=generated_schema)

    def test_an_independently_built_frame_also_validates(self, generated_schema):
        # A different frame than the generator's own exemplar, so this is not
        # merely re-checking the fixture the schema was fitted to.
        rng = np.random.default_rng(7)
        n = 500
        frame = pd.DataFrame(
            {
                "price": rng.exponential(2.0, n),
                "warehouse": rng.choice(["north", "south"], size=n),
                "shipped_at": pd.date_range("2023-06-01", periods=n, freq="min"),
                "returned": rng.random(n) > 0.8,
            }
        )
        frame.loc[:20, "price"] = np.nan
        payload = _as_json(dict(summarize(frame, seed=0)))
        jsonschema.validate(instance=payload, schema=generated_schema)

    def test_a_payload_missing_a_required_dataset_key_is_rejected(
        self, generated_schema
    ):
        payload = _as_json(dict(summarize(_build_dataset(), seed=0)))
        broken = {**payload, "dataset": dict(payload["dataset"])}
        del broken["dataset"]["rows_est"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=generated_schema)

    def test_a_column_of_an_unknown_type_is_rejected(self, generated_schema):
        payload = _as_json(dict(summarize(_build_dataset(), seed=0)))
        columns = dict(payload["columns"])
        some_column = next(iter(columns.values()))
        broken_column = {**some_column, "type": "not-a-real-type"}
        broken = {**payload, "columns": {**columns, "bogus": broken_column}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=generated_schema)
