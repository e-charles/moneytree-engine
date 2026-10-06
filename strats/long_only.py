# strats/long_only 

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
PositionArray = NDArray[np.int8]


def long_only_targets(
    entry_signal: BoolArray,
    exit_signal: BoolArray,
) -> PositionArray:
    """
    Generate long-only close-based target positions.

    Parameters
    ----------
    entry_signal:
        True when the strategy wants to enter long.

    exit_signal:
        True when the strategy wants to exit to cash.

    Returns
    -------
    NDArray[np.int8]
        Close-based target positions.

        0 = flat
        1 = long

    Notes
    -----
    Execution timing and portfolio accounting are handled by
    the backtesting engine.

    target[t] becomes executable at Open[t + 1].
    """

    bar_count = entry_signal.size

    if exit_signal.size != bar_count:
        raise ValueError(
            "entry_signal and exit_signal must have equal length"
        )

    targets = np.empty(
        bar_count,
        dtype=np.int8,
    )

    position = 0

    for bar_number in range(
        bar_count
    ):

        if entry_signal[
            bar_number
        ]:
            position = 1

        elif exit_signal[
            bar_number
        ]:
            position = 0

        targets[
            bar_number
        ] = position

    return targets