"""The install metadata is a contract, and nothing was checking it.

Two defects shipped in the published `0.1.0` metadata and PyPI versions are
immutable, so neither could be corrected in place:

- `psutil>=7.1.0` sat in `dependencies` and no code path under `pysuricata/`
  imported it (#204). psutil publishes no WASM wheel, so `micropip.install`
  could not resolve the package at all, and the browser demo carried a
  hand-written mock distribution purely to get past a dependency the library
  never used.
- pandas was capped `<3.0` on every Python version (#203). Installing into a
  pandas 3 environment silently pulled pandas back to 2.3.3 -- a downgrade the
  user would discover only when something else in their project broke. The cap
  was accidental rather than defensive: it predated pandas 3 and was never
  reconsidered. It did turn out to be sitting on two real incompatibilities,
  fixed alongside it in `tests/test_datetime_resolution.py` and
  `tests/test_inference.py`, which is why CI now runs a pandas 3 leg.

Both are one line of metadata each, and both were invisible to a test suite
that only ever exercised imported code. These tests read `pyproject.toml`
itself.

`tomllib` is standard library only from 3.11 and this project's floor is 3.10,
where importing it would fail collection of this whole module. The requirement
lines are parsed with a regex for the same reason `scripts/check_version.py`
reads the version with one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE = ROOT / "pysuricata"

#: `dependencies = [...]` inside `[project]`. Scoped to the opening bracket and
#: the first line that closes it, so `[project.optional-dependencies]` and the
#: `[dependency-groups]` tables below cannot be picked up by accident.
_RUNTIME_DEPS = re.compile(
    r"^\[project\]\s*$.*?^dependencies\s*=\s*\[(.*?)^\]",
    re.M | re.S,
)

#: The distribution name at the head of a PEP 508 requirement -- everything
#: before the first version specifier, extra, marker or whitespace.
_DIST_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")

#: Distributions whose import name differs from the name pip installs.
_IMPORT_NAME = {"markdown": "markdown"}


def _requirement_lines() -> list[str]:
    """Every runtime requirement string in `[project] dependencies`."""
    match = _RUNTIME_DEPS.search(PYPROJECT.read_text(encoding="utf-8"))
    assert match, "no `dependencies` list under `[project]` in pyproject.toml"

    lines = []
    for raw in match.group(1).splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(",").strip()
        #: Only the matching outer pair is removed. `.strip("\"'")` would also
        #: eat the closing quote of an environment marker, turning
        #: `python_version < '3.13'` into `python_version < '3.13`.
        if len(line) >= 2 and line[0] in "\"'" and line[-1] == line[0]:
            lines.append(line[1:-1])
    return lines


def _imported_modules() -> set[str]:
    """Every top-level module name imported anywhere under `pysuricata/`.

    Parsed rather than grepped: a grep for a name finds it in a docstring, a
    comment or a string literal, and this test's whole point is to distinguish
    a dependency that is used from one that is merely mentioned.
    """
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


class TestEveryRuntimeDependencyIsActuallyImported:
    """#204. A dependency nothing imports is not free.

    It is resolved on every install, it constrains every environment the
    package lands in, and on a platform with no wheel for it -- Pyodide -- it
    is the difference between the package installing and not.
    """

    def test_no_declared_dependency_is_unused(self):
        imported = _imported_modules()

        unused = []
        for requirement in _requirement_lines():
            match = _DIST_NAME.match(requirement)
            assert match, f"cannot read a distribution name from {requirement!r}"
            dist = match.group(1)
            module = _IMPORT_NAME.get(dist.lower(), dist.lower().replace("-", "_"))
            if module not in imported:
                unused.append(f"{dist} (no `import {module}` under pysuricata/)")

        assert not unused, (
            "declared as runtime dependencies but never imported: "
            + ", ".join(unused)
            + ". Move it to an optional extra or a dependency group."
        )

    def test_psutil_specifically_is_not_a_runtime_dependency(self):
        """The regression that shipped, pinned by name.

        The general test above would catch it again only while psutil stays
        unimported. This one states the finding.
        """
        names = [_DIST_NAME.match(r).group(1).lower() for r in _requirement_lines()]
        assert "psutil" not in names, (
            "psutil is back in `[project] dependencies`. It has no WASM wheel, "
            "so this breaks `micropip.install('pysuricata')` in the browser demo."
        )

    def test_psutil_is_still_installable_as_an_extra(self):
        """Removing it from `dependencies` must not remove it from the project.

        The memory tests import it, and docs/performance.md hands the reader a
        recipe that does too.
        """
        text = PYPROJECT.read_text(encoding="utf-8")
        assert "[project.optional-dependencies]" in text
        extras = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
        assert "psutil" in extras, "psutil should remain available as an extra"


class TestThePandasCeilingAdmitsPandas3:
    """#203. `pandas~=2.0` and `pandas>=2.2.3,<3.0` both exclude pandas 3.

    Under pandas 3.0.5 the full suite passes once the datetime-resolution and
    date-inference differences are fixed; only the echoed `dtype` labels differ
    (`object` -> `str`, `datetime64[ns]` -> `[us]`), and those faithfully
    report a genuinely different input rather than a changed statistic.
    """

    def _pandas_requirements(self) -> list[str]:
        #: Runs of whitespace collapsed: the requirement lines are aligned for
        #: reading, and a matcher that counts spaces would fail on formatting.
        reqs = [
            " ".join(r.split())
            for r in _requirement_lines()
            if r.lower().startswith("pandas")
        ]
        assert reqs, "no pandas requirement found under `[project] dependencies`"
        return reqs

    def test_no_pandas_requirement_caps_below_4(self):
        offenders = [
            r for r in self._pandas_requirements() if "<3" in r.replace(" ", "")
        ]
        assert not offenders, (
            "these pandas requirements exclude pandas 3, which works: "
            f"{offenders}. Installing into a pandas 3 environment silently "
            "downgrades the user to 2.3.3."
        )

    def test_no_pandas_requirement_uses_a_compatible_release_clause(self):
        """`pandas~=2.0` means `>=2.0, <3.0` -- a `<3` cap wearing a disguise.

        It reads as a floor, which is why the cap survived a review that was
        looking for one.
        """
        offenders = [r for r in self._pandas_requirements() if "~=" in r]
        assert not offenders, (
            f"`~=` on pandas pins the major version implicitly: {offenders}. "
            "Write the floor and the ceiling out."
        )

    @pytest.mark.parametrize(
        "marker,floor", [("< '3.13'", "2.0"), (">= '3.13'", "2.2.3")]
    )
    def test_the_floors_are_preserved(self, marker, floor):
        """Widening the ceiling must not disturb the floor.

        The `python_version` split exists for the floor: 2.2.3 is the first
        pandas publishing cp313 wheels, so collapsing the two lines into one
        `>=2.2` would let a constrained resolver build 2.2.0 from source
        against a Python it never supported -- the same failure the numpy
        floor below it already documents.
        """
        matching = [r for r in self._pandas_requirements() if marker in r]
        assert len(matching) == 1, (
            f"expected one pandas line for {marker}, got {matching}"
        )
        assert f">={floor}," in matching[0].replace(" ", ""), (
            f"the pandas floor for {marker} should still be {floor}: {matching[0]}"
        )
