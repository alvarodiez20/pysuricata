"""Card rendering module - refactored for better maintainability.

This module provides backward-compatible access to card rendering functionality
that has been refactored into specialized modules for better organization.
"""

from __future__ import annotations

from typing import Any

# Import the new modular card renderers
from .card_factory import (
    render_boolean_card as _render_boolean_card,
)
from .card_factory import (
    render_categorical_card as _render_categorical_card,
)
from .card_factory import (
    render_datetime_card as _render_datetime_card,
)
from .card_factory import (
    render_numeric_card as _render_numeric_card,
)

# Re-export the main rendering functions for backward compatibility
__all__ = [
    "render_numeric_card",
    "render_categorical_card",
    "render_datetime_card",
    "render_boolean_card",
]


# Backward compatibility functions
def render_numeric_card(s: Any) -> str:
    """Render numeric card - backward compatibility function."""
    return _render_numeric_card(s)


def render_categorical_card(s: Any) -> str:
    """Render categorical card - backward compatibility function."""
    return _render_categorical_card(s)


def render_datetime_card(s: Any) -> str:
    """Render datetime card - backward compatibility function."""
    return _render_datetime_card(s)


def render_boolean_card(s: Any) -> str:
    """Render boolean card - backward compatibility function."""
    return _render_boolean_card(s)
