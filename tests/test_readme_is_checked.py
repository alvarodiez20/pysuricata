"""The README is documentation, and for a long time nothing checked it.

#151. Every page under `docs/` is executed and resolved against the live API by
`benchmarks/check_docs.py`, and the README was not in that set. So it drifted
while the rest stayed correct, and claimed:

| Claim | Actual |
|---|---|
| sketch `k` = 1,024 | **2,048** |
| numeric sample = 10,000 | **20,000** |
| two CLI subcommands | **three** |

It is the most-read page in the project and the one PyPI renders as the package
description, which makes it the last file that should have been unchecked.

The rewrite is the smaller half of the fix; this is the other half. These tests
guard the *wiring* -- that the README is in the checked set and that editing it
triggers the check -- because the rewrite is only correct until the next person
edits it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
WORKFLOW = REPO / ".github" / "workflows" / "docs-check.yml"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


class TestTheReadmeIsInTheCheckedSet:
    def test_check_docs_collects_it(self):
        """The mechanism, asserted directly rather than through the workflow."""
        import sys

        sys.path.insert(0, str(REPO))
        from benchmarks.check_docs import _pages

        assert README in _pages(), (
            "README.md is not collected by check_docs, so nothing verifies its "
            "numbers against the live API -- which is how it came to be several "
            "releases out of date"
        )

    def test_editing_it_triggers_the_check(self):
        """A guard that never runs on the file it guards is not a guard."""
        workflow = WORKFLOW.read_text(encoding="utf-8")

        assert "README.md" in workflow, (
            "docs-check.yml does not list README.md in its trigger paths, so a "
            "README-only PR would skip the check entirely"
        )

    def test_the_checker_itself_triggers_the_check(self):
        """Changing the checker must re-run it over everything it covers."""
        workflow = WORKFLOW.read_text(encoding="utf-8")

        assert "benchmarks/check_docs.py" in workflow


class TestTheNumbersMatchTheLibrary:
    """The specific claims #151 measured. `check_docs` executes the fences and
    resolves the option names; these pin the prose figures beside them, which
    no fence contains."""

    def _defaults(self):
        from pysuricata import ComputeOptions

        return ComputeOptions()

    def test_the_sketch_size_is_the_live_default(self, readme):
        expected = self._defaults().max_uniques

        assert f"{expected:,}" in readme, (
            f"the README should quote the live sketch size ({expected:,})"
        )
        assert "default 1024" not in readme and "default 1,024" not in readme

    def test_the_sample_size_is_the_live_default(self, readme):
        expected = self._defaults().numeric_sample_size

        assert f"{expected:,}" in readme
        assert "10 000" not in readme and "10,000" not in readme

    def test_the_chunk_size_is_the_live_default(self, readme):
        expected = self._defaults().chunk_size

        assert f"{expected:,}" in readme


class TestTheDifferentiatorIsDocumented:
    """`pysuricata check` is the feature that separates this from a report
    generator, and the README did not mention it at all."""

    def test_the_check_subcommand_appears(self, readme):
        assert "pysuricata check" in readme

    def test_every_cli_subcommand_appears(self, readme):
        import pysuricata.cli as cli

        source = Path(cli.__file__).read_text(encoding="utf-8")
        subcommands = set(re.findall(r'add_parser\(\s*["\'](\w+)["\']', source))
        assert subcommands, "no subcommands found; the parser moved"

        missing = [s for s in sorted(subcommands) if f"pysuricata {s}" not in readme]
        assert not missing, f"CLI subcommands absent from the README: {missing}"

    def test_compare_is_documented(self, readme):
        assert "compare()" in readme

    def test_it_does_not_promise_a_method_comparison_lacks(self, readme):
        """The draft this replaced claimed `Comparison.save_html`. It has only
        `to_dict()`, and executing the example is what caught it."""
        from pysuricata import Comparison

        if not hasattr(Comparison, "save_html"):
            assert (
                "Comparison" not in readme
                or "save_html" not in readme.split("Comparison")[1][:200]
            ), "the README offers Comparison.save_html, which does not exist"


class TestNoStaleAliasArtifacts:
    def test_it_does_not_say_a_name_is_aliased_as_itself(self, readme):
        """The bulk `ReportConfig` -> `ProfileConfig` rename in #216 left the
        line 'via `ProfileConfig` (aliased as `ProfileConfig`)' behind. A
        find-and-replace can produce a sentence that parses and means nothing."""
        assert "aliased as `ProfileConfig`" not in readme

    def test_it_teaches_the_current_config_name(self, readme):
        assert "ProfileConfig" in readme
        assert "ReportConfig" not in readme, (
            "ReportConfig is deprecated and removed in 0.3.0 (#210); the README "
            "should not teach it"
        )
