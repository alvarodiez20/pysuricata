from __future__ import annotations

from typing import Any, Dict, Optional

from ..compute.core.types import ColumnKinds
from ..config import EngineConfig
from .boolean import BooleanAccumulator
from .categorical import CategoricalAccumulator
from .datetime import DatetimeAccumulator
from .numeric import NumericAccumulator

# Import new system for enhanced functionality
try:
    from .config import AccumulatorConfig
    from .factory2 import build_accumulators as build_accumulators_v2

    _NEW_SYSTEM_AVAILABLE = True
except ImportError:
    _NEW_SYSTEM_AVAILABLE = False


def build_accumulators(
    kinds: ColumnKinds,
    cfg: EngineConfig,
    use_new_system: bool = True,
    accumulator_config: Optional[AccumulatorConfig] = None,
) -> Dict[str, Any]:
    """Instantiate high-performance accumulator objects optimized for big data processing.

    This function provides production-grade accumulator implementations with
    enterprise features including vectorized operations, memory optimization,
    and comprehensive error handling.

    Args:
        kinds: Column kinds information
        cfg: Engine configuration
        use_new_system: Whether to use the new accumulator system (default: True)
        accumulator_config: Optional modern accumulator configuration

    Returns:
        Dictionary mapping column names to accumulator instances
    """
    # Use new system if available and requested
    if use_new_system and _NEW_SYSTEM_AVAILABLE:
        return build_accumulators_v2(kinds, cfg, accumulator_config)

    # Fallback to direct instantiation with optimized configuration
    accs: Dict[str, Any] = {}
    for name in kinds.numeric:
        accs[name] = NumericAccumulator(name)
    for name in kinds.boolean:
        accs[name] = BooleanAccumulator(name)
    for name in kinds.datetime:
        accs[name] = DatetimeAccumulator(name)
    for name in kinds.categorical:
        accs[name] = CategoricalAccumulator(name)
    return accs
