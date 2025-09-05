"""In-memory chunk adapters.

Exports a unified interface that yields pandas DataFrame chunks from
in-memory inputs (DataFrames or iterables of DataFrames).
"""

from .base import iter_chunks  # re-export convenience

__all__ = [
    "iter_chunks",
]
