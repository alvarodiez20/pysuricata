"""The README is documentation, and for a long time nothing checked it.

#151 rewrote it and put it inside `benchmarks/check_docs.py`, which executes
every self-contained fence and resolves every option name against the live API.
That covers the code in the README. It does not cover the *prose beside* the
code, which is where the drift actually was:

| Claim | Actual |
|---|---|
| sketch `k` = 1,024 | **2,048** |
| numeric sample = 10,000 | **20,000** |
| two CLI subcommands | **three** |

Not one of those three lives in a fence. `k = sketch size (default 2048)` is a
line of italic text under a table; the CLI section is a `bash` block, which
`check_docs` cannot execute. So the checker that closed #151 would not have
caught any of the errors #151 was filed about.

These tests are that second half. They read the expected values from
`ComputeOptions()` and from the argument parser's own source, so a renamed
default or a fourth subcommand fails here rather than drifting -- and they pin
the *wiring*, because a guard that does not run on the file it guards is not a
guard.

Digit grouping is deliberately normalised away. `20 000`, `20,000`, `20_000` and
`20000` are the same claim, and a test that insists on one of them fails the
next person to restyle a sentence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
WORKFLOW = REPO / ".github" / "workflows" / "docs-check.yml"

#: Separators that mean "this is one number": space, comma, underscore. Stripping
#: them lets the assertions be about the value rather than the house style.
_GROUPING = re.compile(r"[,_ ](?=\d)")


def _digits_normalised(text: str) -> str:
    """`20 000`, `20,000` and `20_000` all become `20000`."""
    return _GROUPING.sub("", text)


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme_numbers(readme: str) -> str:
    return _digits_normalised(readme)


@pytest.fixture(scope="module")
def defaults():
    from pysuricata import ComputeOptions

    return ComputeOptions()


class TestTheReadmeIsInTheCheckedSet:
    """#151's durable half, asserted at the mechanism rather than by reading the
    workflow file and hoping."""

    def test_check_docs_collects_it(self):
        import sys

        sys.path.insert(0, str(REPO))
        from benchmarks.check_docs import _checked_pages, _pages

        assert README in _checked_pages(), (
            "README.md is not collected by check_docs, so nothing verifies its "
            "fences against the live API -- which is how it came to be several "
            "releases out of date"
        )
        assert README not in _pages(), (
            "`_pages()` is the mkdocs tree and feeds the nav-coverage check; the "
            "README is not an mkdocs page, so putting it there reports it as "
            "missing from the nav forever"
        )

    def test_editing_it_triggers_the_check(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        assert "README.md" in workflow, (
            "docs-check.yml does not list README.md in its trigger paths, so a "
            "README-only pull request would skip the check entirely"
        )

    def test_the_checker_itself_triggers_the_check(self):
        """Changing the checker must re-run it over everything it covers.

        Otherwise the pull request that narrows what is checked -- or breaks it
        outright -- is the one pull request the check does not run on."""
        workflow = WORKFLOW.read_text(encoding="utf-8")

        for script in (
            "benchmarks/check_docs.py",
            "scripts/build_docs_assets.py",
            "scripts/regenerate_example_report.py",
        ):
            assert script in workflow, (
                f"docs-check.yml runs {script} but does not trigger on it"
            )

    def test_the_workflow_triggers_on_itself(self):
        """One step further out: editing the trigger list is exactly the edit
        that most needs the job to run."""
        workflow = WORKFLOW.read_text(encoding="utf-8")

        assert ".github/workflows/docs-check.yml" in workflow


class TestTheProseNumbersMatchTheLibrary:
    """The three claims #151 measured. None of them is inside a fence, so
    `check_docs` executes right past all three."""

    def test_the_sketch_size_is_the_live_default(self, readme_numbers, defaults):
        expected = defaults.max_uniques

        assert str(expected) in readme_numbers, (
            f"the README should quote the live sketch size ({expected})"
        )
        assert "default 1024" not in readme_numbers, (
            "1024 was the claim; the relative standard error 1/sqrt(k-2) quoted "
            "beside it is 3.1% at 1024 and 2.2% at 2048, so the two disagreed"
        )

    def test_the_numeric_sample_size_is_the_live_default(
        self, readme_numbers, defaults
    ):
        expected = defaults.numeric_sample_size

        assert str(expected) in readme_numbers
        assert "default 10000" not in readme_numbers

    def test_the_chunk_size_is_the_live_default(self, readme_numbers, defaults):
        expected = defaults.chunk_size

        assert str(expected) in readme_numbers, (
            f"the README should quote the live chunk size ({expected}); it once "
            f"passed chunk_size=250000 while calling it ordinary"
        )


class TestTheDifferentiatorIsDocumented:
    """`pysuricata check` is what separates this from a report generator, and
    the README did not mention it at all. The CLI section is a `bash` fence,
    which `check_docs` cannot execute."""

    def test_every_cli_subcommand_appears(self, readme):
        import pysuricata.cli as cli

        source = Path(cli.__file__).read_text(encoding="utf-8")
        subcommands = set(re.findall(r'add_parser\(\s*["\'](\w+)["\']', source))
        assert subcommands, "no subcommands found; the parser moved"

        missing = [s for s in sorted(subcommands) if f"pysuricata {s}" not in readme]
        assert not missing, f"CLI subcommands absent from the README: {missing}"

    def test_the_exit_codes_are_documented(self, readme):
        """A gate whose exit codes are undocumented needs a wrapper, which is
        the thing `check` exists to avoid. They are covered by
        `docs/versioning.md`, so they are a promise, not an implementation
        detail."""
        section = readme[readme.find("pysuricata check") :]

        assert "exit" in section, (
            "the README documents `pysuricata check` without saying what its "
            "exit codes mean"
        )

    def test_compare_is_documented(self, readme):
        assert "compare(" in readme

    def test_it_does_not_promise_a_method_comparison_lacks(self, readme):
        """An earlier draft offered `Comparison.save_html`. It has only
        `to_dict()`, and executing the example is what caught it."""
        from pysuricata import Comparison

        if not hasattr(Comparison, "save_html"):
            assert "save_html" not in readme[readme.find("compare(") :], (
                "the README offers Comparison.save_html, which does not exist"
            )


class TestNoStaleRenameArtifacts:
    def test_it_does_not_say_a_name_is_aliased_as_itself(self, readme):
        """The bulk `ReportConfig` -> `ProfileConfig` rename left the line
        'via `ProfileConfig` (aliased as `ProfileConfig`)' behind. A
        find-and-replace can produce a sentence that parses and means nothing."""
        assert "aliased as `ProfileConfig`" not in readme

    def test_it_teaches_the_current_config_name(self, readme):
        assert "ProfileConfig" in readme
        assert "ReportConfig" not in readme, (
            "ReportConfig warns since 0.1.1 and goes in 0.3.0 (#210); the README "
            "should not teach a name with a removal date"
        )


class TestTheMemoryClaimNamesItsAxis:
    """ "Bounded memory" is true in rows and false in columns, and the README
    said neither (#207).

    Measured on the current code: a 1,000,000 x 14 frame costs 56 MB of
    marginal RSS, and a 20,000 x 600 frame -- **fewer cells** -- costs 856 MB.
    Both memory and report size are linear in the column count, because every
    column holds its own sketches for the whole run and gets its own card.

    A claim that does not say which axis it describes is not a weaker claim, it
    is one a reader will apply to the axis it is false on. These check the
    qualification is still there, not that the numbers are right -- the numbers
    belong to `benchmarks/columns.py`.
    """

    def test_the_headline_claim_says_rows(self, readme):
        claim = re.search(r"^Data is processed in chunks.*$", readme, re.M)
        assert claim, "the streaming claim is gone from the README"

        text = claim.group(0).lower()
        assert "row" in text, (
            "the memory claim does not name the axis it holds on. It is bounded "
            f"in rows and unbounded in columns: {claim.group(0)!r}"
        )

    def test_the_claim_does_not_say_regardless_of_size(self, readme):
        """The exact wording that was wrong. "Regardless of dataset size" reads
        as *any* dataset, and a 600-column frame is a dataset."""
        assert "bounded regardless of dataset size" not in readme

    def test_the_column_axis_is_named_as_the_exception(self, readme):
        # `[*_]*` between words: the sentence carries markdown emphasis, and
        # which word is emphasised is an editorial choice this should not pin.
        pattern = (
            r"not[*_\s]+bounded[*_\s]+in[*_\s]+the[*_\s]+number[*_\s]+of[*_\s]+columns"
        )
        assert re.search(pattern, readme, re.I), (
            "the README no longer states the exception, so a reader has only "
            "the half of the claim that is favourable"
        )

    def test_the_feature_bullet_agrees_with_the_headline(self, readme):
        """Two places make the claim. They drifted apart once already."""
        bullet = re.search(r"^- \*\*Streaming architecture\*\*.*$", readme, re.M)
        assert bullet, "the streaming feature bullet is gone"
        assert "row" in bullet.group(0).lower(), bullet.group(0)

    def test_the_column_benchmark_exists(self):
        """The claim is only honest while something measures it."""
        assert (REPO / "benchmarks" / "columns.py").exists(), (
            "benchmarks/columns.py is what keeps the column axis measured; "
            "without it the README's numbers are unfalsifiable again"
        )
