# /optimization/search_space.py

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator


@dataclass(frozen=True)
class DoubleDipParameters:
    """
    One complete RSI Double Dip parameter configuration.
    """

    rsi_length: int
    oversold: float
    overbought: float
    max_bars_between_dips: int


@dataclass(frozen=True)
class DoubleDipSearchSpace:
    """
    Defines all parameter values that the optimizer may test.

    Example
    -------
    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5, 10, 14),
        oversold_levels=(25.0, 30.0),
        overbought_levels=(70.0, 75.0),
        max_bars_between_dips=(5, 10),
    )
    """

    rsi_lengths: tuple[int, ...]
    oversold_levels: tuple[float, ...]
    overbought_levels: tuple[float, ...]
    max_bars_between_dips: tuple[int, ...]

    def __post_init__(self) -> None:
        """
        Validate the search-space definition when it is created.
        """

        if not self.rsi_lengths:
            raise ValueError(
                "rsi_lengths cannot be empty"
            )

        if not self.oversold_levels:
            raise ValueError(
                "oversold_levels cannot be empty"
            )

        if not self.overbought_levels:
            raise ValueError(
                "overbought_levels cannot be empty"
            )

        if not self.max_bars_between_dips:
            raise ValueError(
                "max_bars_between_dips cannot be empty"
            )

        if any(
            length < 2
            for length in self.rsi_lengths
        ):
            raise ValueError(
                "all RSI lengths must be at least 2"
            )

        if any(
            not 0 < level < 100
            for level in self.oversold_levels
        ):
            raise ValueError(
                "all oversold levels must be between 0 and 100"
            )

        if any(
            not 0 < level < 100
            for level in self.overbought_levels
        ):
            raise ValueError(
                "all overbought levels must be between 0 and 100"
            )

        if any(
            bars < 1
            for bars in self.max_bars_between_dips
        ):
            raise ValueError(
                "all max_bars_between_dips values "
                "must be at least 1"
            )

    def candidates(
        self,
    ) -> Iterator[DoubleDipParameters]:
        """
        Generate every valid Double Dip parameter combination.

        Combinations where:
            oversold >= overbought

        are invalid and therefore skipped.
        """

        for (
            rsi_length,
            oversold,
            overbought,
            max_bars,
        ) in product(
            self.rsi_lengths,
            self.oversold_levels,
            self.overbought_levels,
            self.max_bars_between_dips,
        ):
            if oversold >= overbought:
                continue

            yield DoubleDipParameters(
                rsi_length=rsi_length,
                oversold=oversold,
                overbought=overbought,
                max_bars_between_dips=max_bars,
            )

    def candidate_count(self) -> int:
        """
        Return the number of valid candidate configurations.
        """

        return sum(1 for _ in self.candidates())


def default_double_dip_search_space() -> DoubleDipSearchSpace:
    """
    Return the default RSI Double Dip research search space.

    This search space is intentionally constrained while the
    optimization framework is being developed and validated.
    """

    return DoubleDipSearchSpace(
        rsi_lengths=tuple(range(5, 31)),
        oversold_levels=(
            20.0,
            25.0,
            30.0,
            35.0,
        ),
        overbought_levels=(
            65.0,
            70.0,
            75.0,
            80.0,
        ),
        max_bars_between_dips=(
            3,
            5,
            7,
            10,
            15,
        ),
    )