
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Configuration controlling walk-forward optimization.

    train_bars:
        Number of bars used to optimize parameters before
        each out-of-sample test window.

    test_bars:
        Number of unseen bars on which the selected parameters
        are evaluated.

    step_bars:
        Number of bars to move forward after each fold.

        None means step_bars == test_bars, producing
        non-overlapping test windows.

    expanding:
        If False, use a rolling training window.

        Example:

            Fold 1 train: 0:500
            Fold 2 train: 100:600

        If True, keep the beginning of the training period fixed
        and expand the training history.

        Example:

            Fold 1 train: 0:500
            Fold 2 train: 0:600

    allow_partial_last_test:
        If True, the final test window may contain fewer than
        test_bars.

        If False, incomplete final test windows are discarded.
    """

    train_bars: int
    test_bars: int
    step_bars: int | None = None

    expanding: bool = False
    allow_partial_last_test: bool = False

    def __post_init__(self) -> None:

        if self.train_bars < 2:
            raise ValueError(
                "train_bars must be at least 2"
            )

        if self.test_bars < 1:
            raise ValueError(
                "test_bars must be at least 1"
            )

        if (
            self.step_bars is not None
            and self.step_bars < 1
        ):
            raise ValueError(
                "step_bars must be at least 1"
            )

        if (
            self.step_bars is not None
            and self.step_bars < self.test_bars
        ):
            raise ValueError(
                "step_bars cannot be smaller than test_bars"
            )





def _validate_market_data(
    data: pd.DataFrame,
) -> None:
    """
    Validate market data required by walk-forward optimization.

    This validation occurs once before any walk-forward folds
    are evaluated.
    """

    if data.empty:
        raise ValueError(
            "data cannot be empty"
        )

    required_columns = {
        "Open",
        "Close",
    }

    missing = required_columns.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if data.index.has_duplicates:
        raise ValueError(
            "data index cannot contain duplicate timestamps"
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data index must be sorted in ascending order"
        )

    prices = data[
        ["Open", "Close"]
    ]

    if prices.isna().any().any():
        raise ValueError(
            "Open and Close cannot contain NaN values"
        )

    if (
        prices <= 0
    ).any().any():
        raise ValueError(
            "Open and Close prices must be greater than zero"
        )

