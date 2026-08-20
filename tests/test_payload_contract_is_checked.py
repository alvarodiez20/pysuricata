"""The contract check has its own guard, for the reason #151's does.

`test_readme_is_checked.py` puts it plainly: *a guard that does not run on the
file it guards is not a guard*. `check_payload_contract` lives in
`benchmarks/check_docs.py`, which runs from one workflow triggered on a path
list -- so the check can be silently defeated three ways that leave every test
green: unwire it from `main()`, narrow the frame it profiles until a column kind
is no longer represented, or widen what counts as a declaration until any
mention anywhere satisfies it.

Each of those is pinned below, alongside the thing itself: the contract and the
payload agree today.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# A plain import, deliberately. `importorskip` would have been the reflex, and
# it is wrong here: `benchmarks` is a package and `check_docs` imports nothing
# but the standard library at module level, so the import cannot fail for a
# reason worth tolerating -- it can only fail because something broke. Skipping
# on that would report a broken guard as an absent one, which is the exact
# confusion this file exists to prevent one level down.
from benchmarks import check_docs


class TestTheContractAndThePayloadAgree:
    def test_no_key_is_undocumented_or_invented(self):
        """The check, run for real. A failure here names the key and the
        direction, which is the whole point of it reporting both."""
        findings: list = []
        check_docs.check_payload_contract(findings)

        assert not findings, "\n".join(f.detail for f in findings)


class TestTheCheckCannotBeQuietlyDefeated:
    def test_it_is_wired_into_the_run(self):
        """Defining a check and not calling it is the failure this whole file
        is modelled on."""
        source = (REPO / "benchmarks" / "check_docs.py").read_text(encoding="utf-8")
        main = source[source.index("def main(") :]

        assert "check_payload_contract(findings)" in main, (
            "check_payload_contract is defined but main() never calls it, so "
            "the contract is unchecked no matter what CI reports"
        )

    def test_the_frame_carries_every_column_kind(self):
        """The checker's other fixture is numeric and string only. Profiling
        that here would leave the datetime and boolean halves of the contract
        free to say anything, and the check would still pass."""
        from pysuricata import summarize

        columns = summarize(check_docs._contract_frame())["columns"]
        kinds = {c["type"] for c in columns.values()}

        assert kinds == {"numeric", "categorical", "datetime", "boolean"}, (
            f"the contract frame reaches {sorted(kinds)}; the kinds it misses "
            f"are unchecked"
        )

    def test_only_the_first_table_column_declares_a_key(self):
        """Later columns carry prose that names other keys -- `unique_est /
        count`, `true + false + missing`. Counting those would let a key be
        documented by being mentioned in someone else's note, which is the
        failure mode one notch subtler than not checking at all."""
        table = (
            "| Key | Type | Notes |\n"
            "|---|---|---|\n"
            "| `count`, `missing` | int | Exact |\n"
            "| `unique_ratio_approx` | float | `unique_est` / `count` |\n"
        )
        declared = check_docs._documented_keys(table)

        assert declared == {"count", "missing", "unique_ratio_approx"}, declared
        assert "unique_est" not in declared, (
            "`unique_est` appears only in the Notes column here, so it is "
            "described rather than declared"
        )
