from __future__ import annotations

"""Lightweight version resolver used across the package.

Keeps version lookup isolated to avoid import cycles and heavy imports in
rendering/engine modules.
"""

import os

try:  # Python 3.8+
    try:
        from importlib.metadata import (  # type: ignore
            PackageNotFoundError as _PkgNotFound,
        )
        from importlib.metadata import version as _pkg_version  # type: ignore
    except Exception:  # pragma: no cover
        from importlib_metadata import (  # type: ignore
            PackageNotFoundError as _PkgNotFound,
        )
        from importlib_metadata import (
            version as _pkg_version,
        )
except Exception:  # very last resort
    _pkg_version = None  # type: ignore[assignment]

    class _PkgNotFound(Exception):
        pass


def resolve_version() -> str:
    """Return the installed package version or a sensible fallback.

    Order of resolution:
    1) importlib.metadata version for the installed package
    2) environment variable ``PYSURICATA_VERSION``
    3) ``"dev"`` as a final fallback
    """
    if _pkg_version is not None:
        try:
            return _pkg_version("pysuricata")
        except _PkgNotFound:
            pass
        except Exception:
            pass
    return os.getenv("PYSURICATA_VERSION", "dev")
