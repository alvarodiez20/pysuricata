from __future__ import annotations

from typing import Tuple

import numpy as np


def mad(arr: np.ndarray) -> float:
    """Calculates the Median Absolute Deviation (MAD) of an array.

    The MAD is a robust measure of the variability of a univariate sample of
    quantitative data. It is defined as the median of the absolute deviations
    from the data's median.

    Args:
        arr: A numpy array of quantitative data.

    Returns:
        The MAD of the array.
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))
