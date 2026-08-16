"""The two scripts the release pipeline is built on.

Sixty-odd releases happened to this project and none of them were decisions.
`version-check` required a bump on every pull request; `cd.yml` published on
every push to `main`. Each rule is defensible alone, and together they made one
merged PR exactly one PyPI release, unconditionally -- so a rewritten kernel and
a fixed typo were the same size of event and `0.0.71 -> 0.0.72` described
nothing.

Publishing now happens on a tag. That frees `version-check` to be a weaker rule
that catches more: bump if you like, but the step has to be legal.

The first version of `check_step` got its own central rule wrong -- it asked
that only one component *change*, which rejects `0.0.72 -> 0.1.0`, the exact
release the reform exists to enable, because bumping minor forces patch from 72
to 0. `TestAResetIsPartOfTheBump` is that bug, kept.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_version import VersionError, changelog_has, check_step  # noqa: E402
from release_notes import extract  # noqa: E402


class TestALegalStep:
    @pytest.mark.parametrize(
        ("previous", "current", "kind"),
        [
            ("0.0.72", "0.0.73", "patch"),
            ("0.1.0", "0.1.1", "patch"),
            ("0.1.9", "0.1.10", "patch"),
            ("0.0.72", "0.1.0", "minor"),
            ("0.1.3", "0.2.0", "minor"),
            ("0.1.0", "1.0.0", "major"),
            ("0.9.9", "1.0.0", "major"),
        ],
    )
    def test_it_is_accepted_and_named(self, previous, current, kind):
        assert check_step(previous, current) == kind

    def test_no_bump_is_legal(self):
        """The whole point of the reform: a pull request need not bump."""
        assert check_step("0.1.0", "0.1.0") == "unchanged"


class TestAResetIsPartOfTheBump:
    """The bug the first version of this check shipped with."""

    def test_the_release_this_reform_exists_for_is_allowed(self):
        assert check_step("0.0.72", "0.1.0") == "minor"

    def test_one_point_oh_is_allowed_from_any_patch(self):
        assert check_step("0.9.41", "1.0.0") == "major"

    def test_failing_to_reset_is_refused(self):
        with pytest.raises(VersionError, match="resets the ones below"):
            check_step("0.1.3", "0.2.3")


class TestAnIllegalStep:
    def test_a_downgrade(self):
        with pytest.raises(VersionError, match="only go up"):
            check_step("0.0.73", "0.0.72")

    def test_two_components_at_once(self):
        with pytest.raises(VersionError, match="exactly one component"):
            check_step("0.1.0", "0.2.1")

    @pytest.mark.parametrize(
        ("previous", "current"),
        [("0.1.0", "0.1.2"), ("0.1.0", "0.3.0"), ("0.1.0", "2.0.0")],
    )
    def test_a_skipped_number(self, previous, current):
        with pytest.raises(VersionError, match="skips"):
            check_step(previous, current)

    @pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "v1.2.3", "1.2.3a1", ""])
    def test_something_that_is_not_a_version(self, bad):
        with pytest.raises(VersionError, match="MAJOR.MINOR.PATCH"):
            check_step("0.1.0", bad)


class TestTheChangelogGate:
    def test_a_version_with_a_section_passes(self, tmp_path):
        path = tmp_path / "CHANGELOG.md"
        path.write_text(
            "# Changelog\n\n## [0.1.0] - 2026-01-01\n\n### Added\n- a thing\n"
        )
        assert changelog_has("0.1.0", path)

    def test_a_version_without_one_does_not(self, tmp_path):
        path = tmp_path / "CHANGELOG.md"
        path.write_text("# Changelog\n\n## [0.1.0] - 2026-01-01\n\n- a thing\n")
        assert not changelog_has("0.2.0", path)

    def test_a_missing_file_is_not_a_pass(self, tmp_path):
        assert not changelog_has("0.1.0", tmp_path / "nope.md")

    def test_the_repo_changelog_has_the_current_version(self):
        """A live check, so this cannot pass on a fixture while the real file
        drifts."""
        import tomllib

        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ]
        assert changelog_has(version, ROOT / "CHANGELOG.md")


class TestReleaseNotes:
    CHANGELOG = (
        "# Changelog\n\n"
        "## [Unreleased]\n\nNothing yet.\n\n"
        "## [0.2.0] - 2026-02-01\n\n### Added\n- the new thing\n\n"
        "## [0.1.0] - 2026-01-01\n\n### Added\n- the first thing\n"
    )

    def test_it_lifts_only_that_version(self):
        body = extract(self.CHANGELOG, "0.2.0")
        assert "the new thing" in body
        assert "the first thing" not in body
        assert "Nothing yet" not in body

    def test_the_last_section_runs_to_the_end(self):
        assert "the first thing" in extract(self.CHANGELOG, "0.1.0")

    def test_a_missing_version_raises(self):
        with pytest.raises(KeyError):
            extract(self.CHANGELOG, "9.9.9")

    def test_an_empty_section_counts_as_missing(self):
        """A heading with nothing under it would publish a blank release page,
        which is the thing this script exists to prevent."""
        with pytest.raises(KeyError):
            extract("## [0.3.0] - 2026-03-01\n\n## [0.2.0] - x\n\n- real\n", "0.3.0")

    def test_the_real_changelog_yields_notes_for_the_current_version(self):
        import tomllib

        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ]
        body = extract((ROOT / "CHANGELOG.md").read_text(), version)
        assert len(body) > 20


class TestTheScriptsRunAsCommands:
    """They are invoked by the workflow as commands, so the exit codes are the
    contract, not the functions."""

    def _run(self, script: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args],
            capture_output=True,
            text=True,
        )

    def test_a_legal_bump_exits_zero(self):
        got = self._run("check_version.py", "--previous", "0.0.1", "--current", "0.0.1")
        assert got.returncode == 0, got.stderr

    def test_an_illegal_bump_exits_one(self):
        got = self._run("check_version.py", "--previous", "0.2.0", "--current", "0.1.0")
        assert got.returncode == 1
        assert "only go up" in got.stderr

    def test_missing_release_notes_exit_one(self):
        got = self._run("release_notes.py", "99.99.99")
        assert got.returncode == 1
        assert "Refusing to publish" in got.stderr
