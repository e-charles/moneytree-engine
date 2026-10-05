# strats/double_dip.py

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rsi_double_dip_targets(
    rsi: NDArray[np.float64],
    oversold: float = 30.0,
    overbought: float = 70.0,
    max_bars_between_dips: int = 10,
) -> NDArray[np.int8]:
    """
    Generate long-only RSI Double Dip target positions.

    Parameters
    ----------
    rsi:
        One-dimensional NumPy array containing RSI values.

        NaN values are allowed, primarily for the RSI warm-up
        period.

    oversold:
        RSI level at or below which an oversold episode occurs.

    overbought:
        RSI level at or above which an overbought episode occurs.

    max_bars_between_dips:
        Maximum number of bars allowed between the first
        oversold observation and the second oversold episode.

    Returns
    -------
    NDArray[np.int8]
        Target position decided at each bar's close.

        0 = flat
        1 = long

    Notes
    -----
    These are close-based target positions.

    The backtesting engine is responsible for converting:

        target[t]

    into execution at:

        Open[t + 1]
    """

    # =========================================================
    # VALIDATION
    # =========================================================

    # if not isinstance(rsi, np.ndarray):
    #     raise TypeError(
    #         "rsi must be a NumPy ndarray"
    #     )

    # if rsi.ndim != 1:
    #     raise ValueError(
    #         "rsi must be one-dimensional"
    #     )

    # if rsi.size == 0:
    #     raise ValueError(
    #         "rsi cannot be empty"
    #     )

    # if not (
    #     0
    #     < oversold
    #     < overbought
    #     < 100
    # ):
    #     raise ValueError(
    #         "Thresholds must satisfy "
    #         "0 < oversold < overbought < 100."
    #     )

    # if max_bars_between_dips < 1:
    #     raise ValueError(
    #         "max_bars_between_dips must be at least 1."
    #     )

    # =========================================================
    # OUTPUT
    # =========================================================
    #
    # We know the exact number of output observations ahead of
    # time, so allocate the result array once rather than
    # repeatedly growing a Python list.
    # =========================================================

    bar_count = rsi.size

    targets = np.empty(
        bar_count,
        dtype=np.int8,
    )

    # =========================================================
    # POSITION STATE
    # =========================================================

    position = 0

    # =========================================================
    # ENTRY STATE
    # =========================================================
    #
    # -1 means there is currently no first oversold bar.
    #
    # This avoids Optional[int] / None checks inside the hot
    # state-machine loop.
    # =========================================================

    first_oversold_bar = -1

    first_oversold_recovered = False
    second_oversold_seen = False

    # =========================================================
    # EXIT STATE
    # =========================================================

    first_overbought_seen = False
    first_overbought_recovered = False
    second_overbought_seen = False

    # =========================================================
    # PREVIOUS RSI
    # =========================================================
    #
    # NaN represents:
    #
    #     no usable previous RSI observation
    #
    # This naturally handles the RSI warm-up period.
    # =========================================================

    previous_rsi = np.nan

    # =========================================================
    # STRATEGY STATE MACHINE
    # =========================================================

    for bar_number in range(bar_count):

        value = rsi[bar_number]

        # -----------------------------------------------------
        # Missing RSI
        # -----------------------------------------------------
        #
        # Primarily occurs during indicator warm-up.
        #
        # Preserve the current target and break continuity with
        # the previous RSI observation.
        # -----------------------------------------------------

        if np.isnan(value):

            targets[
                bar_number
            ] = position

            previous_rsi = np.nan

            continue

        # -----------------------------------------------------
        # Current RSI state
        # -----------------------------------------------------

        is_oversold = (
            value <= oversold
        )

        is_overbought = (
            value >= overbought
        )

        has_previous_rsi = (
            not np.isnan(
                previous_rsi
            )
        )

        # -----------------------------------------------------
        # Threshold crossings
        # -----------------------------------------------------

        crossed_above_oversold = (
            has_previous_rsi
            and previous_rsi <= oversold
            and value > oversold
        )

        crossed_below_overbought = (
            has_previous_rsi
            and previous_rsi >= overbought
            and value < overbought
        )

        # =====================================================
        # FLAT
        # =====================================================

        if position == 0:

            # -------------------------------------------------
            # Expire an old first-dip setup
            # -------------------------------------------------

            if (
                first_oversold_bar >= 0
                and (
                    bar_number
                    - first_oversold_bar
                    > max_bars_between_dips
                )
            ):
                first_oversold_bar = -1
                first_oversold_recovered = False
                second_oversold_seen = False

            # -------------------------------------------------
            # First oversold episode
            # -------------------------------------------------

            if (
                first_oversold_bar < 0
                and is_oversold
            ):
                first_oversold_bar = (
                    bar_number
                )

            # -------------------------------------------------
            # Recovery after first oversold episode
            # -------------------------------------------------

            elif (
                first_oversold_bar >= 0
                and not first_oversold_recovered
                and crossed_above_oversold
            ):
                first_oversold_recovered = True

            # -------------------------------------------------
            # Second oversold episode
            # -------------------------------------------------

            elif (
                first_oversold_bar >= 0
                and first_oversold_recovered
                and is_oversold
            ):
                second_oversold_seen = True

            # -------------------------------------------------
            # Recovery after second oversold episode
            #
            # ENTRY SIGNAL
            # -------------------------------------------------

            elif (
                second_oversold_seen
                and crossed_above_oversold
            ):
                position = 1

                # Reset entry state.
                first_oversold_bar = -1
                first_oversold_recovered = False
                second_oversold_seen = False

        # =====================================================
        # LONG
        # =====================================================

        else:

            # -------------------------------------------------
            # First overbought episode
            # -------------------------------------------------

            if (
                not first_overbought_seen
                and is_overbought
            ):
                first_overbought_seen = True

            # -------------------------------------------------
            # First drop below overbought
            # -------------------------------------------------

            elif (
                first_overbought_seen
                and not first_overbought_recovered
                and crossed_below_overbought
            ):
                first_overbought_recovered = True

            # -------------------------------------------------
            # Second overbought episode
            # -------------------------------------------------

            elif (
                first_overbought_seen
                and first_overbought_recovered
                and is_overbought
            ):
                second_overbought_seen = True

            # -------------------------------------------------
            # Second drop below overbought
            #
            # EXIT SIGNAL
            # -------------------------------------------------

            elif (
                second_overbought_seen
                and crossed_below_overbought
            ):
                position = 0

                # Reset exit state.
                first_overbought_seen = False
                first_overbought_recovered = False
                second_overbought_seen = False

        # =====================================================
        # RECORD CLOSE-BASED TARGET
        # =====================================================

        targets[
            bar_number
        ] = position

        previous_rsi = value

    return targets