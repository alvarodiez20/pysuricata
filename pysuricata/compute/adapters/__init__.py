"""Data adapters for different backends.

This module provides adapter implementations for different data backends,
including pandas and polars, with a unified interface.
"""

from .base import BaseAdapter
from .pandas import PandasAdapter
from .polars import PolarsAdapter

__all__ = [
    "BaseAdapter",
    "PandasAdapter",
    "PolarsAdapter",
]
