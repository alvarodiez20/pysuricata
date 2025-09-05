# eda_tool/eda_tool/logger.py
import time
import logging

# Library code should not configure global logging. Attach a NullHandler so
# users can opt-in to log output by configuring logging in their application.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def timeit(func):
    """Decorator that logs the time a function takes to run.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function with timing logging.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper
