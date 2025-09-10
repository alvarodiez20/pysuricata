"""Custom exceptions for the compute module.

This module defines specific exception types for different error conditions
in the compute pipeline, enabling better error handling and debugging.
"""

from typing import Optional


class ComputeError(Exception):
    """Base exception for all compute-related errors.

    This is the base class for all exceptions raised by the compute module.
    It provides a common interface for error handling and logging.

    Attributes:
        message: Human-readable error message.
        details: Optional additional error details.
        error_code: Optional error code for programmatic handling.
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize the compute error.

        Args:
            message: Human-readable error message.
            details: Optional additional error details.
            error_code: Optional error code for programmatic handling.
        """
        super().__init__(message)
        self.message = message
        self.details = details
        self.error_code = error_code

    def __str__(self) -> str:
        """Return string representation of the error."""
        result = self.message
        if self.details:
            result += f" (Details: {self.details})"
        if self.error_code:
            result += f" [Code: {self.error_code}]"
        return result


class ChunkingError(ComputeError):
    """Exception raised when chunking operations fail.

    This exception is raised when there are issues with data chunking,
    such as invalid chunk sizes, unsupported data types, or memory issues.
    """

    def __init__(self, message: str, chunk_size: Optional[int] = None, **kwargs):
        """Initialize the chunking error.

        Args:
            message: Human-readable error message.
            chunk_size: The chunk size that caused the error.
            **kwargs: Additional arguments passed to ComputeError.
        """
        super().__init__(message, **kwargs)
        self.chunk_size = chunk_size


class InferenceError(ComputeError):
    """Exception raised when type inference fails.

    This exception is raised when there are issues with column type inference,
    such as ambiguous data types or inference failures.
    """

    def __init__(self, message: str, column_name: Optional[str] = None, **kwargs):
        """Initialize the inference error.

        Args:
            message: Human-readable error message.
            column_name: The column that caused the inference error.
            **kwargs: Additional arguments passed to ComputeError.
        """
        super().__init__(message, **kwargs)
        self.column_name = column_name


class ConversionError(ComputeError):
    """Exception raised when data conversion fails.

    This exception is raised when there are issues with data type conversion,
    such as invalid data formats or conversion failures.
    """

    def __init__(
        self,
        message: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the conversion error.

        Args:
            message: Human-readable error message.
            source_type: The source data type that failed to convert.
            target_type: The target data type that was attempted.
            **kwargs: Additional arguments passed to ComputeError.
        """
        super().__init__(message, **kwargs)
        self.source_type = source_type
        self.target_type = target_type
