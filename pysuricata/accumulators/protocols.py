"""The surface the engine is allowed to use on an accumulator.

Prerequisites for the native core (#64, for #44). The point is not tidiness: a
PyO3 type cannot satisfy `isinstance(acc, NumericAccumulator)`, cannot expose a
`_uniques` attribute holding a Python `KMV`, and cannot be pickled by copying
`__dict__`. Each of those is currently how something outside the accumulator
package reaches inside it, so each is a place the crate could not be swapped in
without editing the caller.

Everything here is behaviour the pure-Python path already had; this states it as
an interface so a second implementation has something to match.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

# What kind of column an accumulator handles. Dispatch reads this rather than
# testing types, because a native accumulator will not be an instance of the
# Python class -- and because the isinstance chain's *order* was load-bearing
# without saying so anywhere.
AccumulatorKind = Literal["numeric", "categorical", "datetime", "boolean"]


def rebuild_accumulator(cls: type, state: dict[str, Any]) -> Any:
    """Reconstruct an accumulator from pickled state, without calling `__init__`.

    The reduce target for every accumulator type. Checkpointing pickles the
    accumulator dict, so a native type needs an explicit reduce or checkpointing
    breaks the moment the fast path is enabled -- and it breaks at the *end* of a
    long run, which is the worst possible time to find out.

    Args:
        cls: The accumulator class.
        state: What `__getstate__` returned.

    Returns:
        The reconstructed accumulator.
    """
    instance = cls.__new__(cls)
    instance.__setstate__(state)
    return instance


class PicklableAccumulator:
    """Explicit pickle protocol, in place of relying on `__dict__` copying.

    Written out rather than inherited from the default because the default is a
    CPython implementation detail that a Rust type does not get for free.
    """

    def __getstate__(self) -> dict[str, Any]:
        """The accumulator's full state."""
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state produced by `__getstate__`."""
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce to (rebuilder, class, state)."""
        return (rebuild_accumulator, (type(self), self.__getstate__()))


@runtime_checkable
class StreamingAccumulator(Protocol):
    """What the engine and the consume layer may rely on.

    Deliberately small. Anything not here is internal to an accumulator, and a
    caller reaching for it is a caller the native core would break.
    """

    name: str

    @property
    def kind(self) -> AccumulatorKind:
        """Which column kind this accumulator handles."""
        ...

    @property
    def unique_est(self) -> int:
        """Approximate distinct values seen."""
        ...

    def update(self, arr: Any) -> None:
        """Fold a chunk of values in."""
        ...

    def add_mem(self, nbytes: int) -> None:
        """Record bytes seen, for the memory estimate."""
        ...

    def finalize(self) -> Any:
        """Produce the summary dataclass."""
        ...


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
