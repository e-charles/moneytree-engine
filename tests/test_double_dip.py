import numpy as np

from strats.double_dip import (
    rsi_double_dip_targets,
)


# ============================================================
# Helpers
# ============================================================


def make_rsi(
    values: list[float],
) -> np.ndarray:
    """
    Create a NumPy RSI array for strategy tests.
    """
    return np.asarray(
        values,
        dtype=np.float64,
    )


# ============================================================
# ENTRY BEHAVIOR
# ============================================================


def test_enters_after_second_oversold_recovery() -> None:
    rsi = make_rsi(
        [
            50.0,
            29.0,
            25.0,
            31.0,
            45.0,
            28.0,
            22.0,
            32.0,
            40.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    expected = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_does_not_enter_after_only_one_oversold_episode() -> None:
    rsi = make_rsi(
        [
            50.0,
            29.0,
            25.0,
            31.0,
            45.0,
            55.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    expected = np.zeros(
        rsi.size,
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_double_dip_expires_if_second_dip_arrives_too_late() -> None:
    rsi = make_rsi(
        [
            50.0,
            25.0,
            32.0,
            45.0,
            50.0,
            50.0,
            50.0,
            25.0,
            32.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=3,
    )

    expected = np.zeros(
        rsi.size,
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_consecutive_oversold_bars_count_as_one_episode() -> None:
    rsi = make_rsi(
        [
            50.0,
            29.0,
            28.0,
            27.0,
            25.0,
            23.0,
            29.0,
            31.0,
            45.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    expected = np.zeros(
        rsi.size,
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


# ============================================================
# EXIT BEHAVIOR
# ============================================================


def test_exits_after_second_overbought_rejection() -> None:
    rsi = make_rsi(
        [
            25.0,
            32.0,
            25.0,
            33.0,
            60.0,
            72.0,
            68.0,
            74.0,
            69.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    expected = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


# ============================================================
# RSI WARM-UP
# ============================================================


def test_nan_rsi_values_do_not_generate_signals() -> None:
    rsi = make_rsi(
        [
            np.nan,
            np.nan,
            np.nan,
            50.0,
            45.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    expected = np.zeros(
        rsi.size,
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_nan_breaks_threshold_crossing_continuity() -> None:
    rsi = make_rsi(
        [
            25.0,
            np.nan,
            35.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    # The transition:
    #
    #     25 -> NaN -> 35
    #
    # must not be interpreted as a direct crossing above the
    # oversold threshold.
    expected = np.asarray(
        [
            0,
            0,
            0,
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


# ============================================================
# OUTPUT CONTRACT
# ============================================================


def test_targets_use_int8_dtype() -> None:
    rsi = make_rsi(
        [
            50.0,
            25.0,
            32.0,
            25.0,
            33.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    assert actual.dtype == np.int8


def test_target_length_matches_rsi_length() -> None:
    rsi = make_rsi(
        [
            50.0,
            25.0,
            32.0,
            25.0,
            33.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    assert actual.size == rsi.size


def test_targets_contain_only_binary_positions() -> None:
    rsi = make_rsi(
        [
            25.0,
            32.0,
            25.0,
            33.0,
            60.0,
            72.0,
            68.0,
            74.0,
            69.0,
        ]
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    assert np.all(
        (actual == 0)
        | (actual == 1)
    )