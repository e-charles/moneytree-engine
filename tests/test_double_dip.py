import pandas as pd
# import sys
# from pathlib import Path

# Adds the parent directory (/workspaces/moneytree-optimizer) to the path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from strats.double_dip import rsi_double_dip_targets


def test_enters_after_second_oversold_recovery() -> None:
    rsi = pd.Series(
        [50, 29, 25, 31, 45, 28, 22, 32, 40],
        index=pd.date_range("2025-01-01", periods=9, freq="D"),
        dtype=float,
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30,
        overbought=70,
        max_bars_between_dips=10,
    )

    expected = pd.Series(
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        index=rsi.index,
        dtype="int8",
    )

    pd.testing.assert_series_equal(actual, expected)

def test_does_not_enter_after_only_one_oversold_episode() -> None:
    rsi = pd.Series(
        [50, 29, 25, 31, 45, 55],
        dtype=float,
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30,
        overbought=70,
        max_bars_between_dips=10,
    )

    assert actual.eq(0).all()

def test_double_dip_expires_if_second_dip_arrives_too_late() -> None:
    rsi = pd.Series(
        [50, 25, 32, 45, 50, 50, 50, 25, 32],
        dtype=float,
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30,
        overbought=70,
        max_bars_between_dips=3,
    )

    assert actual.eq(0).all()

def test_exits_after_second_overbought_rejection() -> None:
    rsi = pd.Series(
        [25, 32, 25, 33, 60, 72, 68, 74, 69],
        dtype=float,
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30,
        overbought=70,
        max_bars_between_dips=10,
    )

    expected = pd.Series(
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        dtype="int8",
    )

    pd.testing.assert_series_equal(actual, expected)

def test_target_is_shifted_for_next_open_execution() -> None:
    targets_at_close = pd.Series([0, 0, 1, 1], dtype="int8")

    executable_target = targets_at_close.shift(1).fillna(0).astype("int8")

    expected = pd.Series([0, 0, 0, 1], dtype="int8")

    pd.testing.assert_series_equal(executable_target, expected)

def test_consecutive_oversold_bars_count_as_one_episode() -> None:
    rsi = pd.Series(
        [50, 29, 28, 27, 25, 23, 29, 31, 45],
        dtype=float,
    )

    actual = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30,
        overbought=70,
        max_bars_between_dips=10,
    )

    assert actual.eq(0).all()