"""Enhanced logging utilities for pysuricata.

This module provides high-performance, thread-safe logging utilities optimized
for data processing workflows. It includes context managers for timing operations,
decorators for automatic function timing, and integration with checkpointing systems.

Key Features:
- Thread-safe SectionTimer with nanosecond precision
- Automatic exception handling and logging
- Performance-optimized timing with minimal overhead
- Integration with checkpointing systems
- Configurable logging levels and formats
- Memory-efficient logging for large datasets

Example:
    Basic usage:
        logger = get_logger(__name__)
        with SectionTimer(logger, "Data processing"):
            process_data()

    Function timing:
        @timeit(logger)
        def expensive_operation():
            return compute_result()

    Checkpoint integration:
        with CheckpointTimer(logger, "Processing", checkpoint_manager):
            process_chunk()
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, Union

# Type variables for generic function decorators
F = TypeVar("F", bound=Callable[..., Any])


class PerformanceTimer:
    """High-performance timer with nanosecond precision.

    This class provides optimized timing functionality with minimal overhead,
    using the most precise timing available on the platform.

    Attributes:
        start_time: The start time in seconds (high precision).
        end_time: The end time in seconds (high precision).
        duration: The calculated duration in seconds.
    """

    def __init__(self) -> None:
        """Initialize the performance timer."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0

    def start(self) -> None:
        """Start the timer with maximum precision."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

    def elapsed(self) -> float:
        """Get the elapsed time without stopping the timer.

        Returns:
            The elapsed time in seconds since start() was called.
        """
        return time.perf_counter() - self.start_time


class SectionTimer:
    """Thread-safe context manager for timing and logging code sections.

    This context manager provides automatic timing, logging, and exception
    handling for code sections. It's optimized for performance with minimal
    overhead and provides detailed logging information.

    Features:
    - Automatic start/end logging with timing information
    - Exception handling with detailed error logging
    - Thread-safe operation
    - Configurable log levels
    - Memory-efficient logging

    Attributes:
        logger: The logger instance to use for output.
        label: The label/name for this section.
        level: The logging level to use (default: INFO).
        timer: Internal performance timer.
        _lock: Thread lock for thread safety.

    Example:
        logger = get_logger(__name__)
        with SectionTimer(logger, "Data processing"):
            process_large_dataset()

        # With custom log level
        with SectionTimer(logger, "Debug operation", level=logging.DEBUG):
            debug_operation()
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        level: int = logging.INFO,
        *,
        show_start: bool = True,
        show_end: bool = True,
    ) -> None:
        """Initialize the section timer.

        Args:
            logger: The logger instance to use for output.
            label: The label/name for this section.
            level: The logging level to use (default: INFO).
            show_start: Whether to log the start of the section.
            show_end: Whether to log the end of the section.

        Raises:
            ValueError: If logger is None or label is empty.
        """
        if logger is None:
            raise ValueError("Logger cannot be None")
        if not label or not label.strip():
            raise ValueError("Label cannot be empty")

        self.logger = logger
        self.label = label.strip()
        self.level = level
        self.show_start = show_start
        self.show_end = show_end
        self.timer = PerformanceTimer()
        self._lock = threading.Lock()
        self._started = False

    def __enter__(self) -> SectionTimer:
        """Enter the context and start timing.

        Returns:
            Self for method chaining.
        """
        with self._lock:
            if self._started:
                self.logger.warning("SectionTimer '%s' already started", self.label)
                return self

            self.timer.start()
            self._started = True

            if self.show_start and self.logger.isEnabledFor(self.level):
                self.logger.log(self.level, "▶ %s...", self.label)

        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit the context and log timing information.

        Args:
            exc_type: The exception type if an exception occurred.
            exc_val: The exception value if an exception occurred.
            exc_tb: The exception traceback if an exception occurred.

        Returns:
            False to allow exceptions to propagate normally.
        """
        with self._lock:
            if not self._started:
                self.logger.warning("SectionTimer '%s' was not started", self.label)
                return False

            self.timer.stop()
            duration = self.timer.duration

            if exc_type is not None:
                # Log exception with timing information
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error(
                        "✖ %s failed after %.3fs: %s",
                        self.label,
                        duration,
                        exc_val or exc_type.__name__,
                        exc_info=(exc_type, exc_val, exc_tb),
                    )
            elif self.show_end and self.logger.isEnabledFor(self.level):
                # Log successful completion with timing
                self.logger.log(
                    self.level,
                    "✓ %s completed in %.3fs",
                    self.label,
                    duration,
                )

            self._started = False

        return False  # Allow exceptions to propagate

    def get_elapsed(self) -> float:
        """Get the elapsed time without stopping the timer.

        Returns:
            The elapsed time in seconds since the timer started.

        Raises:
            RuntimeError: If the timer has not been started.
        """
        if not self._started:
            raise RuntimeError("Timer has not been started")
        return self.timer.elapsed()


class CheckpointTimer(SectionTimer):
    """Section timer with integrated checkpointing support.

    This timer extends SectionTimer with checkpointing capabilities,
    allowing automatic checkpoint creation at configurable intervals.

    Attributes:
        checkpoint_manager: Optional checkpoint manager for saving state.
        checkpoint_interval: Interval in seconds between checkpoints.
        last_checkpoint: Time of the last checkpoint.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        checkpoint_manager: Optional[Any] = None,
        checkpoint_interval: float = 30.0,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        """Initialize the checkpoint timer.

        Args:
            logger: The logger instance to use for output.
            label: The label/name for this section.
            checkpoint_manager: Optional checkpoint manager instance.
            checkpoint_interval: Minimum interval between checkpoints in seconds.
            level: The logging level to use.
            **kwargs: Additional arguments passed to SectionTimer.
        """
        super().__init__(logger, label, level, **kwargs)
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval = max(0.1, checkpoint_interval)  # Minimum 100ms
        self.last_checkpoint: float = 0.0

    def __enter__(self) -> CheckpointTimer:
        """Enter the context and start timing with checkpointing.

        Returns:
            Self for method chaining.
        """
        super().__enter__()
        self.last_checkpoint = time.perf_counter()
        return self

    def maybe_checkpoint(
        self, chunk_idx: int, state: dict[str, Any], html: Optional[str] = None
    ) -> bool:
        """Create a checkpoint if enough time has passed.

        Args:
            chunk_idx: The chunk index for the checkpoint.
            state: The state dictionary to checkpoint.
            html: Optional HTML content to checkpoint.

        Returns:
            True if a checkpoint was created, False otherwise.
        """
        if self.checkpoint_manager is None:
            return False

        current_time = time.perf_counter()
        if current_time - self.last_checkpoint >= self.checkpoint_interval:
            try:
                # Create checkpoint using the checkpoint manager's save method
                if hasattr(self.checkpoint_manager, "save"):
                    self.checkpoint_manager.save(chunk_idx, state, html)
                    self.last_checkpoint = current_time
                    self.logger.debug(
                        "Checkpoint created for section '%s' (chunk %d)",
                        self.label,
                        chunk_idx,
                    )
                    return True
            except Exception as e:
                self.logger.warning(
                    "Failed to create checkpoint for section '%s' (chunk %d): %s",
                    self.label,
                    chunk_idx,
                    e,
                )

        return False


def timeit(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    *,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator for automatic function timing and logging.

    This decorator automatically times function execution and logs the results.
    It's optimized for performance with minimal overhead.

    Args:
        logger: The logger to use (defaults to module logger).
        level: The logging level to use.
        include_args: Whether to include function arguments in the log.
        include_result: Whether to include function result in the log.

    Returns:
        A decorator function that wraps the target function.

    Example:
        @timeit(logger)
        def expensive_computation(data):
            return process_data(data)

        # With custom options
        @timeit(logger, level=logging.DEBUG, include_args=True)
        def debug_function(x, y):
            return x + y
    """
    if logger is None:
        logger = get_logger(__name__)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer = PerformanceTimer()
            timer.start()

            try:
                result = func(*args, **kwargs)
                timer.stop()

                # Build log message
                msg_parts = [f"{func.__name__} executed in {timer.duration:.4f}s"]

                if include_args and (args or kwargs):
                    args_str = ", ".join(repr(arg) for arg in args)
                    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    msg_parts.append(f"with args: ({all_args})")

                if include_result:
                    msg_parts.append(f"result: {result!r}")

                logger.log(level, " ".join(msg_parts))
                return result

            except Exception as e:
                timer.stop()
                logger.error(
                    "%s failed after %.4fs: %s",
                    func.__name__,
                    timer.duration,
                    e,
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


@contextmanager
def log_exceptions(
    logger: logging.Logger,
    message: str,
    level: int = logging.ERROR,
    reraise: bool = True,
):
    """Context manager for logging exceptions with custom messages.

    Args:
        logger: The logger instance to use.
        message: The message to log when an exception occurs.
        level: The logging level to use.
        reraise: Whether to reraise the exception after logging.

    Example:
        with log_exceptions(logger, "Failed to process data"):
            risky_operation()
    """
    try:
        yield
    except Exception as e:
        logger.log(level, "%s: %s", message, e, exc_info=True)
        if reraise:
            raise


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with optimal configuration.

    This function creates and configures a logger with performance-optimized
    settings suitable for data processing workflows. It automatically detects
    Jupyter notebook environments and uses stdout for better visibility.

    Args:
        name: The logger name (typically __name__).

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    # Configure logger with proper handlers if not already configured
    if not logger.handlers:
        # Set the logger level
        logger.setLevel(logging.INFO)
        
        # Detect if we're in a Jupyter notebook environment
        def _is_jupyter_environment():
            try:
                # Check for IPython/Jupyter
                get_ipython()  # This will raise NameError if not in IPython
                return True
            except NameError:
                return False
        
        # Use stdout for Jupyter notebooks, stderr for regular scripts
        if _is_jupyter_environment():
            stream = sys.stdout
        else:
            stream = sys.stderr
        
        # Create a console handler with appropriate stream
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

    return logger


def configure_logging(
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    stream: Any = sys.stderr,
) -> None:
    """Configure global logging settings for the application.

    Args:
        level: The logging level (string or int).
        format_string: Custom format string for log messages.
        stream: The output stream for log messages.

    Example:
        configure_logging(level="DEBUG")
        configure_logging(level=logging.INFO, format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        stream=stream,
        force=True,  # Override existing configuration
    )


# Module-level logger with null handler to prevent warnings
logger = get_logger(__name__)


# Backward compatibility aliases
__all__ = [
    "SectionTimer",
    "CheckpointTimer",
    "PerformanceTimer",
    "timeit",
    "log_exceptions",
    "get_logger",
    "configure_logging",
    "logger",
]
