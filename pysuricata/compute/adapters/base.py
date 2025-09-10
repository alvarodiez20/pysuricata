"""Base adapter class for data backends.

This module provides the base adapter class that all backend-specific
adapters should inherit from, ensuring consistent interface implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.protocols import DataAdapter
from ..core.types import ColumnKinds
from ..processing.conversion import ConversionStrategy, UnifiedConverter
from ..processing.inference import InferenceStrategy, UnifiedTypeInferrer


class BaseAdapter(DataAdapter, ABC):
    """Base adapter class for data backends.

    This abstract base class provides common functionality and enforces
    the DataAdapter protocol for all backend-specific adapters.

    Attributes:
        converter: Unified data converter.
        inferrer: Unified type inferrer.
        logger: Logger for adapter operations.
    """

    def __init__(
        self,
        converter: Optional[UnifiedConverter] = None,
        inferrer: Optional[UnifiedTypeInferrer] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the base adapter.

        Args:
            converter: Unified data converter.
            inferrer: Unified type inferrer.
            logger: Logger for adapter operations.
        """
        self.converter = converter or UnifiedConverter(
            strategy=ConversionStrategy.ZERO_COPY
        )
        self.inferrer = inferrer or UnifiedTypeInferrer(
            strategy=InferenceStrategy.BALANCED
        )
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def infer_and_build(
        self, data: Any, config: Any
    ) -> tuple[ColumnKinds, Dict[str, Any]]:
        """Infer column types and build accumulators.

        Args:
            data: Input data to analyze.
            config: Configuration object with processing parameters.

        Returns:
            Tuple of (column_kinds, accumulators_dict).
        """
        pass

    @abstractmethod
    def estimate_mem(self, frame: Any) -> int:
        """Estimate memory usage of a data frame.

        Args:
            frame: Data frame to estimate memory for.

        Returns:
            Estimated memory usage in bytes.
        """
        pass

    @abstractmethod
    def missing_cells(self, frame: Any) -> int:
        """Count missing cells in a data frame.

        Args:
            frame: Data frame to count missing cells in.

        Returns:
            Number of missing cells.
        """
        pass

    @abstractmethod
    def consume_chunk(
        self,
        data: Any,
        accs: Dict[str, Any],
        kinds: ColumnKinds,
        logger: Optional[Any] = None,
    ) -> None:
        """Consume a data chunk and update accumulators.

        Args:
            data: Data chunk to process.
            accs: Dictionary of accumulators to update.
            kinds: Column type information.
            logger: Optional logger for progress tracking.
        """
        pass

    @abstractmethod
    def update_corr(
        self, frame: Any, corr_est: Any, logger: Optional[Any] = None
    ) -> None:
        """Update correlation estimator with new data.

        Args:
            frame: Data frame to process.
            corr_est: Correlation estimator to update.
            logger: Optional logger for progress tracking.
        """
        pass

    @abstractmethod
    def sample_section_html(self, first: Any, cfg: Any) -> str:
        """Generate HTML for sample data section.

        Args:
            first: First data chunk for sampling.
            cfg: Configuration object.

        Returns:
            HTML string for sample section.
        """
        pass

    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter information.
        """
        return {
            "adapter_type": self.__class__.__name__,
            "converter_strategy": self.converter.strategy.value,
            "inferrer_strategy": self.inferrer.strategy.value,
        }

    def validate_data(self, data: Any) -> bool:
        """Validate that data is compatible with this adapter.

        Args:
            data: Data to validate.

        Returns:
            True if data is compatible, False otherwise.
        """
        return self._is_compatible_data(data)

    @abstractmethod
    def _is_compatible_data(self, data: Any) -> bool:
        """Check if data is compatible with this adapter.

        Args:
            data: Data to check.

        Returns:
            True if compatible, False otherwise.
        """
        pass
