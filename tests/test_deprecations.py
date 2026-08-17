"""Deprecated names must say so, and say when they go.

`ReportConfig = ProfileConfig` was a bare alias exported in `__all__`. Anyone
using the old name got no signal that it was going away, and anyone reading
`__init__.py` could not tell whether it was deprecated or simply a second
spelling intended to stay. #82 removed the two-constructor ceremony and this
alias is what was left holding the door open; the door now has a closing date.

The warning fires on **use**, via a module-level `__getattr__` (PEP 562), not
on `import pysuricata` -- an import-time warning would fire for every user
including the ones who never touch the old name, which is how deprecation
warnings train people to filter them out.
"""

from __future__ import annotations

import warnings

import pytest

import pysuricata

#: Every deprecated name, the name replacing it, and the release removing it.
DEPRECATIONS = [("ReportConfig", "ProfileConfig", "0.3.0")]


@pytest.mark.parametrize("old,new,removal", DEPRECATIONS)
class TestADeprecatedNameWarnsAndStillWorks:
    def test_using_it_raises_a_deprecation_warning(self, old, new, removal):
        with pytest.warns(DeprecationWarning, match=old):
            getattr(pysuricata, old)

    def test_the_warning_names_the_removal_release(self, old, new, removal):
        """A deprecation without a date is a deprecation nobody acts on."""
        with pytest.warns(DeprecationWarning) as record:
            getattr(pysuricata, old)
        assert removal in str(record[0].message)

    def test_the_warning_names_the_replacement(self, old, new, removal):
        with pytest.warns(DeprecationWarning) as record:
            getattr(pysuricata, old)
        assert new in str(record[0].message)

    def test_it_still_resolves_to_the_replacement(self, old, new, removal):
        """Deprecated is not removed. Until `removal` it must keep working, and
        be the *same object* so `isinstance` and identity checks hold."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert getattr(pysuricata, old) is getattr(pysuricata, new)

    def test_it_is_still_exported(self, old, new, removal):
        assert old in pysuricata.__all__
        assert old in dir(pysuricata)


class TestTheWarningFiresWhereItShould:
    def test_importing_the_package_does_not_warn(self):
        """The reason for `__getattr__` over an eager alias.

        Re-imported in a subprocess: `pysuricata` is already in `sys.modules`
        here, so an in-process import would not re-execute the module and would
        pass whether or not the warning were at import time.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning",
                "-c",
                "import pysuricata",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_dir_does_not_warn(self):
        """Tab-completing in a REPL is not use."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert "ReportConfig" in dir(pysuricata)

    def test_an_unknown_attribute_is_still_an_attribute_error(self):
        """`__getattr__` must not turn every typo into a deprecation."""
        # Via a variable rather than `pysuricata.NoSuchName`: a bare attribute
        # access is a useless expression (B018) and a literal `getattr` is
        # flagged too (B009), while the name is the whole point of the test.
        missing = "NoSuchName"
        with pytest.raises(AttributeError, match=missing):
            getattr(pysuricata, missing)
