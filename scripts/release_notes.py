#!/usr/bin/env python3
"""Lift a version's section out of CHANGELOG.md, for the GitHub release body.

Releases were created with an empty body. The changelog is already enforced on
every pull request, so the notes exist -- they were simply never carried the
last few feet to the release page.

**It refuses to release a version with no section.** That is the point rather
than a side effect: a release with no notes is a tag pretending to be an
announcement, and the failure has to happen before publishing, not after.

Usage:
    python scripts/release_notes.py 0.1.0
    python scripts/release_notes.py 0.1.0 --output notes.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract(changelog: str, version: str) -> str:
    """The body of `## [version]`, up to the next `## ` heading.

    Raises:
        KeyError: when the version has no section.
    """
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, changelog, re.S | re.M)
    if not match:
        raise KeyError(version)
    body = match.group(1).strip()
    if not body:
        raise KeyError(version)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--changelog", default=str(REPO_ROOT / "CHANGELOG.md"))
    parser.add_argument("--output", help="write here instead of stdout")
    args = parser.parse_args()

    path = Path(args.changelog)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        body = extract(text, args.version)
    except KeyError:
        print(
            f"✗ CHANGELOG.md has no section for {args.version}.\n"
            "  Refusing to publish a release with an empty body -- a tag with no "
            "notes is not an announcement.\n"
            f"  Add a `## [{args.version}]` section and re-tag.",
            file=sys.stderr,
        )
        return 1

    if args.output:
        Path(args.output).write_text(body + "\n", encoding="utf-8")
        print(f"✓ wrote {len(body)} chars of notes for {args.version}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
