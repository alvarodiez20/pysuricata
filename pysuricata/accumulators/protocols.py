from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CorrelatableAccumulator(Protocol):
    """Accumulator that can receive a list of top correlated columns.

    Structural protocol used for typing; at runtime we duck-type via
    ``hasattr(acc, 'set_corr_top')``.
    """

    def set_corr_top(self, items: Any) -> None:  # type: ignore[override]
        ...


@runtime_checkable
class FinalizableAccumulator(Protocol):
    def finalize(self) -> Any: ...
