from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator

from backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
    validate_backtest_config,
)

from backtesting.metrics import (
    calculate_metrics,
)

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

from optimization.static import (
    optimize_double_dip,
)

from strats.double_dip import (
    rsi_double_dip_targets,
)


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

    Historical Close data through the test window is used to
    calculate RSI and rebuild strategy state.

    No data after test_end_position is used.

    Once RSI has been calculated, strategy generation,
    backtesting, and metric calculation operate entirely on
    NumPy arrays.
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

    if test_end_position > len(data):
        raise ValueError(
            "test_end_position cannot exceed data length"
        )

    # =========================================================
    # 1. BUILD HISTORICAL CONTEXT
    # =========================================================
    #
    # We allow information through:
    #
    #     test_end_position - 1
    #
    # because each RSI observation is calculated only from
    # current and past Close prices.
    #
    # No future observations after the test window are included.
    # =========================================================

    history_close = data[
        "Close"
    ].iloc[
        :test_end_position
    ]

    # =========================================================
    # 2. CALCULATE RSI
    # =========================================================
    #
    # RSIIndicator currently uses pandas, so this is the one
    # remaining pandas calculation in the test numerical path.
    #
    # Immediately convert the result to NumPy.
    # =========================================================

    rsi = RSIIndicator(
        close=history_close,
        window=parameters.rsi_length,
    ).rsi().to_numpy(
        dtype=np.float64,
        copy=False,
    )

    # =========================================================
    # 3. REBUILD STRATEGY STATE WITH NUMPY
    # =========================================================
    #
    # The strategy runs over the historical RSI context so state
    # immediately before the test window is reconstructed.
    # =========================================================

    target_with_history = (
        rsi_double_dip_targets(
            rsi=rsi,
            oversold=parameters.oversold,
            overbought=parameters.overbought,
            max_bars_between_dips=(
                parameters.max_bars_between_dips
            ),
        )
    )

    # =========================================================
    # 4. EXTRACT TEST-WINDOW NUMPY ARRAYS
    # =========================================================

    open_prices = data[
        "Open"
    ].iloc[
        test_start_position:test_end_position
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    close_prices = data[
        "Close"
    ].iloc[
        test_start_position:test_end_position
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    test_target = target_with_history[
        test_start_position:test_end_position
    ]

    # =========================================================
    # 5. RUN OUT-OF-SAMPLE BACKTEST
    # =========================================================
    #
    # The full walk-forward market dataset and configuration
    # were validated before entering the fold loop.
    #
    # rsi_double_dip_targets() is responsible for producing
    # binary target arrays.
    # =========================================================

    result = run_backtest(
        open_prices=open_prices,
        close_prices=close_prices,
        target_at_close=test_target,
        config=backtest_config,
        inputs_prevalidated=True,
    )

    # =========================================================
    # 6. CALCULATE OUT-OF-SAMPLE METRICS
    # =========================================================

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=periods_per_year,
        inputs_prevalidated=True,
    )

    return (
        metrics,
        result,
    )


def _build_summary_row(
    fold_number: int,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    selected: SelectedDoubleDipCandidate,
    test_metrics: dict[str, float | int],
) -> dict[str, object]:
    """
    Construct one flat human-readable walk-forward summary row.

    This function belongs to the orchestration/reporting side of
    walk-forward processing, so use of pandas indexes here is
    intentional.
    """

    parameters = (
        selected.parameters
    )

    return {
        "fold": fold_number,

        "train_start": (
            train_data.index[0]
        ),

        "train_end": (
            train_data.index[-1]
        ),

        "test_start": (
            test_data.index[0]
        ),

        "test_end": (
            test_data.index[-1]
        ),

        "train_bars": len(
            train_data
        ),

        "test_bars": len(
            test_data
        ),

        "rsi_length": (
            parameters.rsi_length
        ),

        "oversold": (
            parameters.oversold
        ),

        "overbought": (
            parameters.overbought
        ),

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
            test_metrics[
                "closed_trades"
            ]
        ),

        "test_total_return": float(
            test_metrics[
                "total_return"
            ]
        ),

        "test_annualized_return": float(
            test_metrics[
                "annualized_return"
            ]
        ),

        "test_annualized_volatility": float(
            test_metrics[
                "annualized_volatility"
            ]
        ),

        "test_max_drawdown": float(
            test_metrics[
                "max_drawdown"
            ]
        ),

        "test_sharpe": float(
            test_metrics[
                "sharpe"
            ]
        ),

        "test_profit_factor": float(
            test_metrics[
                "profit_factor"
            ]
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
        2. Optimize candidates using only training data.
        3. Calculate neighborhood robustness using only
           training-period results.
        4. Apply eligibility rules and select the winner.
        5. Freeze the selected parameters.
        6. Evaluate those frozen parameters on the following
           unseen test window.

    Test-window results never participate in candidate
    selection.
    """

    # =========================================================
    # 1. VALIDATE GLOBAL INPUTS ONCE
    # =========================================================

    _validate_market_data(
        data
    )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero"
        )

    if selection_config is None:
        selection_config = (
            SelectionConfig()
        )

    if neighborhood_config is None:
        neighborhood_config = (
            NeighborhoodConfig()
        )

    if backtest_config is None:
        backtest_config = (
            BacktestConfig()
        )

    validate_backtest_config(
        backtest_config
    )

    if (
        len(data)
        <= walk_forward_config.train_bars
    ):
        raise ValueError(
            "data must contain more bars than train_bars"
        )

    # =========================================================
    # 2. DETERMINE FOLD STEP
    # =========================================================

    step_bars = (
        walk_forward_config.step_bars
        if walk_forward_config.step_bars
        is not None
        else walk_forward_config.test_bars
    )

    folds: list[
        WalkForwardFold
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    # The first unseen bar follows the initial training window.
    test_start_position = (
        walk_forward_config.train_bars
    )

    fold_number = 0

    # =========================================================
    # 3. WALK FORWARD THROUGH TIME
    # =========================================================

    while (
        test_start_position
        < len(data)
    ):

        test_end_position = (
            test_start_position
            + walk_forward_config.test_bars
        )

        # -----------------------------------------------------
        # Final partial test window
        # -----------------------------------------------------

        if (
            test_end_position
            > len(data)
        ):

            if (
                not walk_forward_config
                .allow_partial_last_test
            ):
                break

            test_end_position = (
                len(data)
            )

        # -----------------------------------------------------
        # Training boundaries
        # -----------------------------------------------------

        if (
            walk_forward_config.expanding
        ):

            train_start_position = 0

        else:

            train_start_position = (
                test_start_position
                - walk_forward_config.train_bars
            )

        train_end_position = (
            test_start_position
        )

        # -----------------------------------------------------
        # Build train/test views
        # -----------------------------------------------------
        #
        # pandas remains useful here because fold construction
        # is orchestration, not the numerical candidate hot
        # path.
        # -----------------------------------------------------

        train_data = data.iloc[
            train_start_position:
            train_end_position
        ]

        test_data = data.iloc[
            test_start_position:
            test_end_position
        ]

        # =====================================================
        # 4. STATIC TRAINING OPTIMIZATION
        # =====================================================

        optimization_results = (
            optimize_double_dip(
                data=train_data,
                search_space=search_space,
                backtest_config=backtest_config,
                periods_per_year=periods_per_year,
            )
        )

        # =====================================================
        # 5. NEIGHBORHOOD ROBUSTNESS
        # =====================================================

        robust_results = (
            add_neighborhood_robustness(
                results=optimization_results,
                config=neighborhood_config,
            )
        )

        # =====================================================
        # 6. SELECT TRAINING-PERIOD WINNER
        # =====================================================

        selected = (
            select_double_dip_candidate(
                results=robust_results,
                config=selection_config,
            )
        )

        # =====================================================
        # 7. FREEZE PARAMETERS AND TEST OUT OF SAMPLE
        # =====================================================

        (
            test_metrics,
            test_backtest,
        ) = _evaluate_test_window(
            data=data,
            test_start_position=(
                test_start_position
            ),
            test_end_position=(
                test_end_position
            ),
            parameters=(
                selected.parameters
            ),
            backtest_config=(
                backtest_config
            ),
            periods_per_year=(
                periods_per_year
            ),
        )

        # =====================================================
        # 8. RECORD FOLD
        # =====================================================

        fold = WalkForwardFold(
            fold=fold_number,

            train_start=(
                train_data.index[0]
            ),

            train_end=(
                train_data.index[-1]
            ),

            test_start=(
                test_data.index[0]
            ),

            test_end=(
                test_data.index[-1]
            ),

            selected_candidate=(
                selected
            ),

            test_metrics=(
                test_metrics
            ),

            test_backtest=(
                test_backtest
            ),
        )

        folds.append(
            fold
        )

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
        # 9. ADVANCE
        # =====================================================

        fold_number += 1

        test_start_position += (
            step_bars
        )

    # =========================================================
    # 10. REQUIRE AT LEAST ONE FOLD
    # =========================================================

    if not folds:
        raise ValueError(
            "walk-forward configuration produced no "
            "complete evaluation folds"
        )

    # =========================================================
    # 11. BUILD FINAL HUMAN-READABLE SUMMARY
    # =========================================================

    return WalkForwardResult(
        folds=tuple(
            folds
        ),
        summary=pd.DataFrame(
            summary_rows
        ),
    )