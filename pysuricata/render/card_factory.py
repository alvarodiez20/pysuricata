"""Card factory for creating appropriate card renderers."""

from typing import Any, Union

from .boolean_card import BooleanCardRenderer
from .card_types import BooleanStats, CategoricalStats, DateTimeStats, NumericStats
from .categorical_card import CategoricalCardRenderer
from .datetime_card import DateTimeCardRenderer
from .numeric_card import NumericCardRenderer


class CardFactory:
    """Factory for creating card renderers based on data type."""

    def __init__(self):
        self._renderers = {
            "numeric": NumericCardRenderer(),
            "categorical": CategoricalCardRenderer(),
            "datetime": DateTimeCardRenderer(),
            "boolean": BooleanCardRenderer(),
        }

    def get_renderer(self, data_type: str):
        """Get the appropriate renderer for the data type."""
        return self._renderers.get(data_type)

    def render_card(
        self, stats: Union[NumericStats, CategoricalStats, DateTimeStats, BooleanStats]
    ) -> str:
        """Render a card using the appropriate renderer."""
        # Determine data type from stats object
        if isinstance(stats, NumericStats):
            renderer = self._renderers["numeric"]
        elif isinstance(stats, CategoricalStats):
            renderer = self._renderers["categorical"]
        elif isinstance(stats, DateTimeStats):
            renderer = self._renderers["datetime"]
        elif isinstance(stats, BooleanStats):
            renderer = self._renderers["boolean"]
        else:
            # Fallback: try to determine from attributes
            if hasattr(stats, "true_n") and hasattr(stats, "false_n"):
                renderer = self._renderers["boolean"]
            elif hasattr(stats, "min_ts") and hasattr(stats, "max_ts"):
                renderer = self._renderers["datetime"]
            elif hasattr(stats, "top_items"):
                renderer = self._renderers["categorical"]
            else:
                renderer = self._renderers["numeric"]

        return renderer.render_card(stats)


# Convenience functions for backward compatibility
def render_numeric_card(stats: Any) -> str:
    """Render numeric card - backward compatibility function."""
    factory = CardFactory()
    return factory.render_card(stats)


def render_categorical_card(stats: Any) -> str:
    """Render categorical card - backward compatibility function."""
    factory = CardFactory()
    return factory.render_card(stats)


def render_datetime_card(stats: Any) -> str:
    """Render datetime card - backward compatibility function."""
    factory = CardFactory()
    return factory.render_card(stats)


def render_boolean_card(stats: Any) -> str:
    """Render boolean card - backward compatibility function."""
    factory = CardFactory()
    return factory.render_card(stats)
