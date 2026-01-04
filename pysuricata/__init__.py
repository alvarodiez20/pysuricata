"""pysuricata package exports.

Preferred high-level API:
    from pysuricata import profile, summarize, ProfileConfig
"""

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

        setattr(compat_date_range, "_pysuricata_patched", True)
        pl.date_range = compat_date_range  # type: ignore[attr-defined]
    except Exception:
        # Best-effort shim; silently ignore if API differs
        return


_patch_polars_date_range()

# High-level API wrappers
from .api import (
    ComputeOptions,
    ProfileConfig,
    RenderOptions,
    Report,
    profile,
    summarize,
)

# Alias for discoverability
ReportConfig = ProfileConfig
