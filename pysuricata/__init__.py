"""pysuricata package exports.

Preferred high-level API:
    from pysuricata import profile, summarize, ProfileConfig
"""

import warnings

# Expose package version in the conventional place
from ._version import resolve_version as _resolve_version

__version__ = _resolve_version()


# Backwards-compatibility shim for polars.date_range(low/high → start/end)
def _patch_polars_date_range() -> None:
    try:
        import polars as pl  # type: ignore
    except Exception:
        return
    try:
        orig = getattr(pl, "date_range", None)
        if not callable(orig) or getattr(orig, "_pysuricata_patched", False):
            return

        def compat_date_range(*args, **kwargs):  # type: ignore[override]
            if "low" in kwargs or "high" in kwargs:
                # Map old argument names to new ones if necessary
                if "start" not in kwargs and "low" in kwargs:
                    kwargs["start"] = kwargs.pop("low")
                if "end" not in kwargs and "high" in kwargs:
                    kwargs["end"] = kwargs.pop("high")
            return orig(*args, **kwargs)  # type: ignore[misc]

        compat_date_range._pysuricata_patched = True
        pl.date_range = compat_date_range  # type: ignore[attr-defined]
    except Exception:
        # Best-effort shim; silently ignore if API differs
        return


_patch_polars_date_range()

# High-level API wrappers
from .api import (
    ComputeOptions,
    ConfigurationError,
    ProfileConfig,
    PySuricataError,
    RenderOptions,
    Report,
    UnsupportedDataError,
    profile,
    summarize,
)
from .comparison import Comparison, compare

#: Deprecated name -> (replacement name, the object, the release that removes it).
#:
#: `ReportConfig` is what was left holding the door open when #82 removed the
#: two-constructor ceremony. As a bare alias it gave no signal that it was going
#: away, and a reader of this module could not tell whether it was deprecated or
#: simply a second spelling intended to stay. The door now has a closing date on
#: it: the clock starts at 0.1.0, so by 0.3.0 a full minor has passed with a
#: warning in place -- the deprecation policy being run rather than described.
_DEPRECATED_NAMES = {
    "ReportConfig": ("ProfileConfig", ProfileConfig, "0.3.0"),
}


def __getattr__(name: str):
    """Resolve deprecated aliases, warning on **use** rather than on import.

    An eager `ReportConfig = ProfileConfig` cannot warn at all, and warning at
    import time would fire on `import pysuricata` for every user including the
    ones who never touch the old name. PEP 562 puts the warning exactly where
    the deprecated name is actually read.
    """
    entry = _DEPRECATED_NAMES.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    replacement, obj, removal = entry
    warnings.warn(
        f"{name} is deprecated and will be removed in {removal}; "
        f"use {replacement} instead. It is the same object, so the only change "
        f"needed is the name.",
        DeprecationWarning,
        stacklevel=2,
    )
    return obj


def __dir__() -> list[str]:
    """Keep the deprecated names discoverable without resolving them.

    `dir()` must not fire the warning -- tab-completing in a REPL is not use.
    """
    return sorted(set(globals()) | set(_DEPRECATED_NAMES))


# Without this, `dir(pysuricata)` is mostly internal submodules and `from
# pysuricata import *` drags them in. Everything listed is public and covered by
# the compatibility promise; anything absent is an implementation detail.
__all__ = [
    "Comparison",
    "ComputeOptions",
    "ConfigurationError",
    "ProfileConfig",
    "PySuricataError",
    "RenderOptions",
    "Report",
    "ReportConfig",
    "UnsupportedDataError",
    "__version__",
    "compare",
    "profile",
    "summarize",
]
