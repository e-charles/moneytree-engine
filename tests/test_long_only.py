import numpy as np

from strats.long_only import (
    long_only_targets,
)


def make_signal(
    values: list[bool],
) -> np.ndarray:
    return np.asarray(
        values,
        dtype=np.bool_,
    )


def test_entry_signal_enters_long() -> None:
    entry_signal = make_signal(
        [False, True, False, False]
    )

    exit_signal = make_signal(
        [False, False, False, False]
    )

    actual = long_only_targets(
        entry_signal=entry_signal,
        exit_signal=exit_signal,
    )

    expected = np.asarray(
        [0, 1, 1, 1],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_exit_signal_returns_to_flat() -> None:
    entry_signal = make_signal(
        [False, True, False, False]
    )

    exit_signal = make_signal(
        [False, False, True, False]
    )

    actual = long_only_targets(
        entry_signal=entry_signal,
        exit_signal=exit_signal,
    )

    expected = np.asarray(
        [0, 1, 0, 0],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_repeated_entry_signal_does_not_change_position() -> None:
    entry_signal = make_signal(
        [True, True, True, False]
    )

    exit_signal = make_signal(
        [False, False, False, False]
    )

    actual = long_only_targets(
        entry_signal=entry_signal,
        exit_signal=exit_signal,
    )

    expected = np.asarray(
        [1, 1, 1, 1],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_repeated_exit_signal_remains_flat() -> None:
    entry_signal = make_signal(
        [False, False, False, False]
    )

    exit_signal = make_signal(
        [True, True, True, False]
    )

    actual = long_only_targets(
        entry_signal=entry_signal,
        exit_signal=exit_signal,
    )

    expected = np.asarray(
        [0, 0, 0, 0],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_targets_are_int8() -> None:
    entry_signal = make_signal(
        [False, True]
    )

    exit_signal = make_signal(
        [False, False]
    )

    actual = long_only_targets(
        entry_signal=entry_signal,
        exit_signal=exit_signal,
    )

    assert actual.dtype == np.int8


def test_signal_length_mismatch_rejected() -> None:
    entry_signal = make_signal(
        [False, True, False]
    )

    exit_signal = make_signal(
        [False, True]
    )

    try:
        long_only_targets(
            entry_signal=entry_signal,
            exit_signal=exit_signal,
        )
    except ValueError as error:
        assert str(error) == (
            "entry_signal and exit_signal must have equal length"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )