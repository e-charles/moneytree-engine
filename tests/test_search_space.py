import pytest

from optimization.search_space import (
    DoubleDipSearchSpace,
    DoubleDipParameters,
    default_double_dip_search_space,
)


def test_candidate_count() -> None:
    search_space = DoubleDipSearchSpace(
        rsi_lengths=(10, 14),
        oversold_levels=(25.0, 30.0),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5, 10),
    )

    assert search_space.candidate_count() == 8


def test_candidates_are_double_dip_parameters() -> None:
    search_space = DoubleDipSearchSpace(
        rsi_lengths=(14,),
        oversold_levels=(30.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(10,),
    )

    candidates = list(
        search_space.candidates()
    )

    assert len(candidates) == 1

    assert candidates[0] == DoubleDipParameters(
        rsi_length=14,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )


def test_invalid_threshold_combination_is_skipped() -> None:
    search_space = DoubleDipSearchSpace(
        rsi_lengths=(14,),
        oversold_levels=(30.0, 75.0),
        overbought_levels=(70.0,),
        max_bars_between_dips=(10,),
    )

    candidates = list(
        search_space.candidates()
    )

    assert len(candidates) == 1

    assert candidates[0].oversold == 30.0
    assert candidates[0].overbought == 70.0


def test_empty_rsi_lengths_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="rsi_lengths cannot be empty",
    ):
        DoubleDipSearchSpace(
            rsi_lengths=(),
            oversold_levels=(30.0,),
            overbought_levels=(70.0,),
            max_bars_between_dips=(10,),
        )


def test_invalid_rsi_length_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="all RSI lengths must be at least 2",
    ):
        DoubleDipSearchSpace(
            rsi_lengths=(1,),
            oversold_levels=(30.0,),
            overbought_levels=(70.0,),
            max_bars_between_dips=(10,),
        )


def test_invalid_max_bars_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        DoubleDipSearchSpace(
            rsi_lengths=(14,),
            oversold_levels=(30.0,),
            overbought_levels=(70.0,),
            max_bars_between_dips=(0,),
        )


def test_default_search_space_has_expected_candidate_count() -> None:
    search_space = default_double_dip_search_space()

    assert search_space.candidate_count() == 2080