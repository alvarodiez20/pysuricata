"""Per-type accumulators and streaming sketches.

This package contains implementations for numeric, categorical, datetime,
and boolean accumulators, as well as reusable sketch algorithms.

Initial extraction moves the sketch algorithms here; subsequent steps can
move the per-type accumulators and register them.
"""

from .sketches import KMV, ReservoirSampler, MisraGries  # re-export
from .numeric import NumericAccumulator, NumericSummary
from .boolean import BooleanAccumulator, BooleanSummary
from .categorical import CategoricalAccumulator, CategoricalSummary
from .datetime import DatetimeAccumulator, DatetimeSummary
