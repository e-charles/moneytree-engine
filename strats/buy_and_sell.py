from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
PositionArray = NDArray[np.int8]

def long_short_targets(
    buy_signal: BoolArray,
    sell_signal: BoolArray,
) -> PositionArray:

    targets = np.empty(
        buy_signal.size,
        dtype=np.int8,
    )

    position = 0

    for bar_number in range(
        buy_signal.size
    ):

        if buy_signal[
            bar_number
        ]:
            position = 1

        elif sell_signal[
            bar_number
        ]:
            position = -1

        targets[
            bar_number
        ] = position

    return targets