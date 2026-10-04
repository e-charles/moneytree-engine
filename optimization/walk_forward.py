# optimization/walk_forward

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ta.momentum import RSIIndicator

from backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from backtesting.metrics import calculate_metrics

from optimization.robustness import (
    NeighborhoodConfig,
    add_neighborhood_robustness,
)
from optimization.search_space import (
    DoubleDipParameters,
    DoubleDipSearchSpace,
)
from optimization.selection import (
    SelectedDoubleDipCandidate,
    SelectionConfig,
    select_double_dip_candidate,
)
from optimization.static import optimize_double_dip

from strats.double_dip import rsi_double_dip_targets


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

        # Overlapping test folds make aggregate out-of-sample
        # performance difficult to interpret because the same
        # market bars would be evaluated more than once.
        if (
            self.step_bars is not None
            and self.step_bars < self.test_bars
        ):
            raise ValueError(
                "step_bars cannot be smaller than test_bars"
            )


@dataclass(frozen=True)
class WalkForwardFold:
    """
    Complete result of one walk-forward fold.
    """

    fold: int

    train_start: object
    train_end: object

    test_start: object
    test_end: object

    selected_candidate: SelectedDoubleDipCandidate

    test_metrics: dict[str, float | int]

    test_backtest: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    """
    Complete walk-forward optimization result.

    folds:
        Detailed fold-by-fold results.

    summary:
        Flat DataFrame containing selected parameters,
        in-sample ranking information, and out-of-sample
        performance for every fold.
    """

    folds: tuple[WalkForwardFold, ...]
    summary: pd.DataFrame


def _validate_market_data(
    data: pd.DataFrame,
) -> None:
    """
    Validate market data required by walk-forward optimization.
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
            f"Missing required columns: {sorted(missing)}"
        )

    if data.index.has_duplicates:
        raise ValueError(
            "data index cannot contain duplicate timestamps"
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data index must be sorted in ascending order"
        )

    if data[
        ["Open", "Close"]
    ].isna().any().any():
        raise ValueError(
            "Open and Close cannot contain NaN values"
        )

    if (
        data[
            ["Open", "Close"]
        ] <= 0
    ).any().any():
        raise ValueError(
            "Open and Close prices must be greater than zero"
        )


def _evaluate_test_window(
    data: pd.DataFrame,
    test_start_position: int,
    test_end_position: int,
    parameters: DoubleDipParameters,
    backtest_config: BacktestConfig,
    periods_per_year: int,
) -> tuple[
    dict[str, float | int],
    BacktestResult,
]:
    """
    Evaluate frozen parameters on one unseen test window.

    Historical Close data before the test window is included
    when calculating RSI and rebuilding strategy state.

    No future data beyond the end of the test window is used.

    The returned backtest itself contains only test-window bars.
    """

    if test_start_position < 0:
        raise ValueError(
            "test_start_position cannot be negative"
        )

    if test_end_position <= test_start_position:
        raise ValueError(
            "test_end_position must be greater than "
            "test_start_position"
        )

    # ---------------------------------------------------------
    # 1. Build historical context
    # ---------------------------------------------------------
    #
    # Everything through the end of the test window is legal:
    #
    #     history ... train | test
    #
    # No observation after test_end_position is included.
    # ---------------------------------------------------------

    history_through_test = data.iloc[
        :test_end_position
    ]

    rsi = RSIIndicator(
        close=history_through_test["Close"],
        window=parameters.rsi_length,
    ).rsi()

    # ---------------------------------------------------------
    # 2. Reconstruct strategy state
    # ---------------------------------------------------------
    #
    # Running the strategy over historical RSI allows state such
    # as the first oversold dip to exist immediately before the
    # test window.
    #
    # Parameters are already frozen at this point.
    # ---------------------------------------------------------

    target_with_history = rsi_double_dip_targets(
        rsi=rsi,
        oversold=parameters.oversold,
        overbought=parameters.overbought,
        max_bars_between_dips=(
            parameters.max_bars_between_dips
        ),
    )

    # ---------------------------------------------------------
    # 3. Extract ONLY the unseen test period
    # ---------------------------------------------------------

    test_data = data.iloc[
        test_start_position:test_end_position
    ]

    test_target = target_with_history.reindex(
        test_data.index
    )

    # ---------------------------------------------------------
    # 4. Run isolated out-of-sample backtest
    # ---------------------------------------------------------
    #
    # Every fold starts with backtest_config.initial_cash.
    #
    # Therefore each fold measures the frozen parameter
    # configuration independently.
    # ---------------------------------------------------------

    result = run_backtest(
        data=test_data,
        target_at_close=test_target,
        config=backtest_config,
    )

    # ---------------------------------------------------------
    # 5. Calculate out-of-sample metrics
    # ---------------------------------------------------------

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=periods_per_year,
    )

    return metrics, result


def _build_summary_row(
    fold_number: int,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    selected: SelectedDoubleDipCandidate,
    test_metrics: dict[str, float | int],
) -> dict[str, object]:
    """
    Construct one flat walk-forward summary row.
    """

    parameters = selected.parameters

    return {
        "fold": fold_number,

        "train_start": train_data.index[0],
        "train_end": train_data.index[-1],

        "test_start": test_data.index[0],
        "test_end": test_data.index[-1],

        "train_bars": len(train_data),
        "test_bars": len(test_data),

        "rsi_length": parameters.rsi_length,
        "oversold": parameters.oversold,
        "overbought": parameters.overbought,
        "max_bars_between_dips": (
            parameters.max_bars_between_dips
        ),

        "selection_metric": (
            selected.ranking_metric
        ),

        "train_ranking_value": (
            selected.ranking_value
        ),

        "train_closed_trades": (
            selected.closed_trades
        ),

        "train_total_return": (
            selected.total_return
        ),

        "train_annualized_return": (
            selected.annualized_return
        ),

        "train_max_drawdown": (
            selected.max_drawdown
        ),

        "train_sharpe": (
            selected.sharpe
        ),

        "train_profit_factor": (
            selected.profit_factor
        ),

        "test_closed_trades": int(
            test_metrics["closed_trades"]
        ),

        "test_total_return": float(
            test_metrics["total_return"]
        ),

        "test_annualized_return": float(
            test_metrics["annualized_return"]
        ),

        "test_annualized_volatility": float(
            test_metrics["annualized_volatility"]
        ),

        "test_max_drawdown": float(
            test_metrics["max_drawdown"]
        ),

        "test_sharpe": float(
            test_metrics["sharpe"]
        ),

        "test_profit_factor": float(
            test_metrics["profit_factor"]
        ),
    }


def walk_forward_optimize_double_dip(
    data: pd.DataFrame,
    search_space: DoubleDipSearchSpace,
    walk_forward_config: WalkForwardConfig,
    selection_config: SelectionConfig | None = None,
    neighborhood_config: NeighborhoodConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """
    Perform walk-forward optimization of the RSI Double Dip
    strategy.

    For each fold:

        1. Construct the in-sample training window.
        2. Optimize every candidate using ONLY training data.
        3. Measure parameter-neighborhood robustness using ONLY
           training results.
        4. Apply eligibility rules and select the winner.
        5. Freeze the selected parameters.
        6. Evaluate those parameters on the following unseen
           test window.

    Test-window results never participate in parameter selection.
    """

    _validate_market_data(data)

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero"
        )

    if selection_config is None:
        selection_config = SelectionConfig()

    if neighborhood_config is None:
        neighborhood_config = NeighborhoodConfig()

    if backtest_config is None:
        backtest_config = BacktestConfig()

    if len(data) <= walk_forward_config.train_bars:
        raise ValueError(
            "data must contain more bars than train_bars"
        )

    step_bars = (
        walk_forward_config.step_bars
        if walk_forward_config.step_bars is not None
        else walk_forward_config.test_bars
    )

    folds: list[WalkForwardFold] = []
    summary_rows: list[dict[str, object]] = []

    # First unseen bar immediately follows the initial
    # training window.
    test_start_position = (
        walk_forward_config.train_bars
    )

    fold_number = 0

    while test_start_position < len(data):

        test_end_position = (
            test_start_position
            + walk_forward_config.test_bars
        )

        # -----------------------------------------------------
        # Handle final incomplete test window
        # -----------------------------------------------------

        if test_end_position > len(data):

            if (
                not walk_forward_config
                .allow_partial_last_test
            ):
                break

            test_end_position = len(data)

        # -----------------------------------------------------
        # Construct training boundaries
        # -----------------------------------------------------

        if walk_forward_config.expanding:
            train_start_position = 0
        else:
            train_start_position = (
                test_start_position
                - walk_forward_config.train_bars
            )

        train_end_position = (
            test_start_position
        )

        train_data = data.iloc[
            train_start_position:train_end_position
        ]

        test_data = data.iloc[
            test_start_position:test_end_position
        ]

        # =====================================================
        # 1. STATIC OPTIMIZATION
        # =====================================================

        optimization_results = optimize_double_dip(
            data=train_data,
            search_space=search_space,
            backtest_config=backtest_config,
            periods_per_year=periods_per_year,
        )

        # =====================================================
        # 2. NEIGHBORHOOD ROBUSTNESS
        # =====================================================

        robust_results = add_neighborhood_robustness(
            results=optimization_results,
            config=neighborhood_config,
        )

        # =====================================================
        # 3. SELECTION
        # =====================================================

        selected = select_double_dip_candidate(
            results=robust_results,
            config=selection_config,
        )

        # =====================================================
        # 4. FREEZE PARAMETERS AND TEST OUT OF SAMPLE
        # =====================================================

        test_metrics, test_backtest = (
            _evaluate_test_window(
                data=data,
                test_start_position=(
                    test_start_position
                ),
                test_end_position=(
                    test_end_position
                ),
                parameters=selected.parameters,
                backtest_config=backtest_config,
                periods_per_year=periods_per_year,
            )
        )

        # =====================================================
        # 5. RECORD FOLD
        # =====================================================

        fold = WalkForwardFold(
            fold=fold_number,

            train_start=train_data.index[0],
            train_end=train_data.index[-1],

            test_start=test_data.index[0],
            test_end=test_data.index[-1],

            selected_candidate=selected,

            test_metrics=test_metrics,

            test_backtest=test_backtest,
        )

        folds.append(fold)

        summary_rows.append(
            _build_summary_row(
                fold_number=fold_number,
                train_data=train_data,
                test_data=test_data,
                selected=selected,
                test_metrics=test_metrics,
            )
        )

        # =====================================================
        # 6. ADVANCE THROUGH TIME
        # =====================================================

        fold_number += 1

        test_start_position += step_bars

    if not folds:
        raise ValueError(
            "walk-forward configuration produced no "
            "complete evaluation folds"
        )

    return WalkForwardResult(
        folds=tuple(folds),
        summary=pd.DataFrame(summary_rows),
    )