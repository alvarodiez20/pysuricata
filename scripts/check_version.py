#!/usr/bin/env python3
"""Validate a version bump, instead of demanding one.

The old rule was `every pull request must bump the version or it cannot merge`,
paired with `every push to main publishes`. Each is defensible alone; together
they make one merged pull request exactly one PyPI release, unconditionally. A
rewritten kernel and a fixed typo become the same size of event, so `0.0.71 ->
0.0.72` carries no information and nobody can pin against anything.

Semantic versioning is not a convention you adopt on top of that. It is
impossible while it holds, because the version is incremented by the *act of
merging* rather than by a judgement about what merged.

So the rule is weaker and catches more. A pull request need not bump. If it
does, the step has to be legal:

* a real increase, never a downgrade;
* exactly one component *raised*, never `0.1.0 -> 0.2.1`;
* the components below it reset to zero, so `0.1.3 -> 0.2.3` is refused while
  `0.0.72 -> 0.1.0` is fine -- a reset is part of the bump, not a second one;
* no skipped numbers, so `0.1.0 -> 0.1.2` is refused;
* and `CHANGELOG.md` must carry a matching `## [X.Y.Z]` section, because the
  release notes are lifted from it and a release with no notes is not a release.

Usage:
    python scripts/check_version.py --base origin/main
    python scripts/check_version.py --previous 0.1.0 --current 0.1.1
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: `1.2.3`. Pre-release and build metadata are deliberately unsupported: this
#: project has never published one, and accepting a syntax nothing produces
#: means writing ordering rules nobody has tested.
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class VersionError(Exception):
    """A bump that is not a legal step."""


def parse(version: str) -> tuple[int, int, int]:
    match = _VERSION.match(version.strip())
    if not match:
        raise VersionError(f"{version!r} is not MAJOR.MINOR.PATCH")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


#: `version = "1.2.3"` inside the `[project]` table. Scoped to that table so a
#: `version` key under `[tool.something]` cannot be picked up by accident.
_PROJECT_VERSION = re.compile(
    r"^\[project\]\s*$.*?^version\s*=\s*[\"']([^\"']+)[\"']",
    re.M | re.S,
)


def read_version(text: str) -> str:
    """The `[project] version` from a `pyproject.toml`, without a TOML parser.

    `tomllib` is standard library from 3.11 and this project supports 3.10, so
    importing it made the whole test module unimportable on the oldest Python
    the package claims to run on -- caught by CI on 3.10 and nowhere else,
    because every local run is on a newer interpreter. Adding `tomli` for one
    string is a dependency this does not need.
    """
    match = _PROJECT_VERSION.search(text)
    if not match:
        raise VersionError("no `version` under `[project]` in pyproject.toml")
    return match.group(1)


def version_at(ref: str) -> str:
    """The version recorded in `pyproject.toml` at a git ref."""
    out = subprocess.run(
        ["git", "show", f"{ref}:pyproject.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return read_version(out.stdout)


def check_step(previous: str, current: str) -> str:
    """Return a description of the step, or raise if it is not a legal one."""
    old, new = parse(previous), parse(current)
    if new == old:
        return "unchanged"
    if new < old:
        raise VersionError(f"{current} is lower than {previous} -- versions only go up")

    # Exactly one component *increases*; the ones below it must reset to zero.
    #
    # Getting this wrong is easy and the first version of this check did: it
    # asked that only one component *change*, which rejects `0.0.72 -> 0.1.0`
    # -- the very release it exists to permit -- because bumping minor requires
    # patch to go 72 -> 0, and that is a second changed component. Reset is not
    # a second decision, it is part of the first one.
    increased = [i for i in range(3) if new[i] > old[i]]
    if len(increased) != 1:
        names = ", ".join(("major", "minor", "patch")[i] for i in increased)
        raise VersionError(
            f"{previous} -> {current} increases {names or 'nothing'}. "
            "A bump raises exactly one component."
        )

    index = increased[0]
    name = ("major", "minor", "patch")[index]
    if new[index] != old[index] + 1:
        raise VersionError(
            f"{previous} -> {current} skips {name} numbers "
            f"({old[index]} -> {new[index]}). A skipped version is one nobody can "
            "install, and it makes the history lie about what shipped."
        )
    for lower in range(index + 1, 3):
        if new[lower] != 0:
            raise VersionError(
                f"{previous} -> {current} raises {name} but leaves "
                f"{('major', 'minor', 'patch')[lower]} at {new[lower]}. "
                "Bumping a component resets the ones below it to 0."
            )
    return name


def changelog_has(version: str, changelog: Path) -> bool:
    if not changelog.exists():
        return False
    return (
        re.search(
            rf"^## \[{re.escape(version)}\]",
            changelog.read_text(encoding="utf-8"),
            re.M,
        )
        is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="git ref to compare against, e.g. origin/main")
    parser.add_argument("--previous")
    parser.add_argument("--current")
    parser.add_argument(
        "--changelog", default=str(REPO_ROOT / "CHANGELOG.md"), help="path to CHANGELOG"
    )
    args = parser.parse_args()

    try:
        current = args.current or read_version(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        previous = args.previous or version_at(args.base or "origin/main")
    except (subprocess.CalledProcessError, KeyError, OSError) as exc:
        print(f"could not read a version: {exc}", file=sys.stderr)
        return 2

    try:
        step = check_step(previous, current)
    except VersionError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    if step == "unchanged":
        print(
            f"✓ version unchanged at {current}. Nothing publishes on merge, so a "
            "pull request does not have to bump."
        )
        return 0

    if not changelog_has(current, Path(args.changelog)):
        print(
            f"✗ {previous} -> {current} is a legal {step} bump, but CHANGELOG.md has "
            f"no `## [{current}]` section.\n"
            "  The release notes are lifted from that section, so a version without "
            "one would publish with an empty release page.",
            file=sys.stderr,
        )
        return 1

    print(f"✓ {previous} -> {current} is a legal {step} bump, with a changelog section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
