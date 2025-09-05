"""pysuricata package exports.

Preferred high-level API:
    from pysuricata import profile, summarize, ReportConfig
"""

# High-level API wrappers
from .api import (
    Report,
    ReportConfig,
    ComputeOptions,
    RenderOptions,
    profile,
    summarize,
)

# Intentionally no public generate_report; use profile()/summarize() only.
