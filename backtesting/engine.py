# backtesting/engine.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

def validate_backtest_config(
    config: BacktestConfig,
) -> None:
    """
    Validate backtest configuration.

    This validation is independent of market data and strategy
    targets, so an optimizer can perform it once before evaluating
    thousands of candidates.
    """

    if config.initial_cash <= 0:
        raise ValueError(
            "initial_cash must be greater than zero"
        )

    if config.commission_rate < 0:
        raise ValueError(
            "commission_rate cannot be negative"
        )

    if config.slippage_rate < 0:
        raise ValueError(
            "slippage_rate cannot be negative"
        )

    if config.slippage_rate >= 1:
        raise ValueError(
            "slippage_rate must be less than 1"
        )


def validate_backtest_market_data(
    data: pd.DataFrame,
) -> None:
    """
    Validate market data used by the backtest.

    During optimization this can be called once per training
    window rather than once per candidate.
    """

    if data.empty:
        raise ValueError(
            "data cannot be empty"
        )

    required_columns = {
        "Open",
        "Close",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        data[
            ["Open", "Close"]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Open and Close prices cannot contain "
            "NaN values"
        )

    if (
        (
            data[
                ["Open", "Close"]
            ]
            <= 0
        )
        .any()
        .any()
    ):
        raise ValueError(
            "Open and Close prices must be "
            "greater than zero"
        )


def validate_backtest_target(
    data: pd.DataFrame,
    target_at_close: pd.Series,
) -> None:
    """
    Validate one strategy target series.
    """

    if not target_at_close.index.equals(
        data.index
    ):
        raise ValueError(
            "target_at_close index must exactly "
            "match data index"
        )

    if target_at_close.isna().any():
        raise ValueError(
            "target_at_close cannot contain NaN values"
        )

    invalid_targets = (
        ~target_at_close.isin([0, 1])
    )

    if invalid_targets.any():
        raise ValueError(
            "target_at_close may contain only 0 or 1"
        )
    
@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration controlling execution assumptions.

    initial_cash:
        Starting portfolio value.

    commission_rate:
        Commission charged on each transaction as a fraction
        of transaction notional.

    slippage_rate:
        Simulated adverse price movement on execution.

        Buys execute above the quoted price.
        Sells execute below the quoted price.

    close_open_position_at_end:
        If True, any remaining long position is liquidated
        at the final bar's Close.
    """

    initial_cash: float = 100_000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0002
    close_open_position_at_end: bool = True


@dataclass
class BacktestResult:
    """
    Complete output of a backtest.

    equity:
        Portfolio value marked at each bar's Close.

    positions:
        Position actually held after execution at each bar's Open.

        0 = flat
        1 = long

    trades:
        One row for every completed trade.
    """

    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame


def _fill_price(
    raw_price: float,
    side: int,
    slippage_rate: float,
) -> float:
    """
    Apply adverse slippage to an execution price.

    side:
        +1 = buy
        -1 = sell
    """

    if side not in (-1, 1):
        raise ValueError(
            "side must be either -1 or +1"
        )

    return raw_price * (
        1 + side * slippage_rate
    )


def _validate_inputs(
    data: pd.DataFrame,
    target_at_close: pd.Series,
    config: BacktestConfig,
) -> None:
    """
    Validate market data, strategy targets, and backtest
    configuration.
    """

    if data.empty:
        raise ValueError(
            "data cannot be empty"
        )

    required_columns = {
        "Open",
        "Close",
    }

    missing_columns = (
        required_columns.difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not target_at_close.index.equals(
        data.index
    ):
        raise ValueError(
            "target_at_close index must exactly "
            "match data index"
        )

    if target_at_close.isna().any():
        raise ValueError(
            "target_at_close cannot contain NaN values"
        )

    invalid_targets = (
        ~target_at_close.isin([0, 1])
    )

    if invalid_targets.any():
        raise ValueError(
            "target_at_close may contain only 0 or 1"
        )

    if data[
        ["Open", "Close"]
    ].isna().any().any():
        raise ValueError(
            "Open and Close prices cannot contain "
            "NaN values"
        )

    if (
        data[
            ["Open", "Close"]
        ] <= 0
    ).any().any():
        raise ValueError(
            "Open and Close prices must be "
            "greater than zero"
        )

    if config.initial_cash <= 0:
        raise ValueError(
            "initial_cash must be greater than zero"
        )

    if config.commission_rate < 0:
        raise ValueError(
            "commission_rate cannot be negative"
        )

    if config.slippage_rate < 0:
        raise ValueError(
            "slippage_rate cannot be negative"
        )

    if config.slippage_rate >= 1:
        raise ValueError(
            "slippage_rate must be less than 1"
        )


def run_backtest(
    data: pd.DataFrame,
    target_at_close: pd.Series,
    config: BacktestConfig | None = None,
    inputs_prevalidated: bool = False
) -> BacktestResult:
    """
    Run a long-only backtest.

    Strategy targets are generated using information available
    at each bar's Close and become executable at the following
    bar's Open.

    inputs_prevalidated:
    If False, perform complete market-data, target, and
    configuration validation.

    If True, skip validation.

    This option is intended for trusted internal optimization
    loops where the inputs have already been validated before
    candidate evaluation.

    Normal callers should leave this False.
    """

    if config is None:
        config = BacktestConfig()

    if not inputs_prevalidated:
        _validate_inputs(
            data=data,
            target_at_close=target_at_close,
            config=config,
    )

    # =========================================================
    # 1. CONVERT INPUTS TO NUMPY ARRAYS
    # =========================================================
    #
    # Pandas remains our public data interface, but the
    # performance-sensitive execution loop operates on raw
    # arrays.
    # =========================================================

    open_prices = data[
        "Open"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    close_prices = data[
        "Close"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    timestamps = data.index.to_numpy(
        copy=False
    )

    target_values = (
        target_at_close
        .to_numpy(
            dtype=np.int8,
            copy=False,
        )
    )

    bar_count = len(data)

    # =========================================================
    # 2. CLOSE TARGET -> NEXT OPEN POSITION
    # =========================================================
    #
    # Previous implementation:
    #
    # position_at_open = (
    #     target_at_close
    #     .shift(1)
    #     .fillna(0)
    #     .astype("int8")
    # )
    #
    # Equivalent NumPy representation:
    #
    # target:
    #     [1, 0, 0, 1]
    #
    # execution:
    #     [0, 1, 0, 0]
    #
    # The final target has no next bar on which to execute.
    # =========================================================

    position_at_open = np.empty(
        bar_count,
        dtype=np.int8,
    )

    position_at_open[0] = 0

    if bar_count > 1:
        position_at_open[1:] = (
            target_values[:-1]
        )

    # =========================================================
    # 3. PORTFOLIO STATE
    # =========================================================

    cash = float(
        config.initial_cash
    )

    units = 0.0

    current_position = 0

    # =========================================================
    # 4. CURRENT TRADE STATE
    # =========================================================

    entry_price: float | None = None
    entry_time = None

    entry_fee = 0.0
    entry_notional = 0.0

    entry_bar_number: int | None = None

    # =========================================================
    # 5. PREALLOCATE BAR OUTPUTS
    # =========================================================
    #
    # We already know how many equity and position observations
    # there will be, so lists do not need to grow dynamically.
    # =========================================================

    equity_values = np.empty(
        bar_count,
        dtype=np.float64,
    )

    position_values = np.empty(
        bar_count,
        dtype=np.int8,
    )

    completed_trades: list[dict] = []

    # =========================================================
    # 6. NUMPY-BASED EXECUTION LOOP
    # =========================================================

    for bar_number in range(bar_count):

        desired_position = int(
            position_at_open[
                bar_number
            ]
        )

        open_price = float(
            open_prices[
                bar_number
            ]
        )

        close_price = float(
            close_prices[
                bar_number
            ]
        )

        timestamp = timestamps[
            bar_number
        ]

        # =====================================================
        # EXIT LONG
        # =====================================================

        if (
            current_position == 1
            and desired_position == 0
        ):

            exit_price = _fill_price(
                raw_price=open_price,
                side=-1,
                slippage_rate=(
                    config.slippage_rate
                ),
            )

            exit_notional = (
                units * exit_price
            )

            exit_fee = (
                exit_notional
                * config.commission_rate
            )

            cash += (
                exit_notional
                - exit_fee
            )

            # entry_price cannot logically be None while a
            # position is open.
            assert entry_price is not None

            gross_pnl = (
                (
                    exit_price
                    - entry_price
                )
                * units
            )

            total_fees = (
                entry_fee
                + exit_fee
            )

            net_pnl = (
                gross_pnl
                - total_fees
            )

            return_pct = (
                net_pnl
                / entry_notional
                if entry_notional > 0
                else 0.0
            )

            bars_held = (
                bar_number
                - entry_bar_number
                if entry_bar_number
                is not None
                else 0
            )

            completed_trades.append(
                {
                    "entry_time": (
                        entry_time
                    ),
                    "exit_time": (
                        timestamp
                    ),
                    "direction": "long",
                    "entry_price": (
                        entry_price
                    ),
                    "exit_price": (
                        exit_price
                    ),
                    "units": units,
                    "entry_notional": (
                        entry_notional
                    ),
                    "gross_pnl": (
                        gross_pnl
                    ),
                    "fees": total_fees,
                    "net_pnl": net_pnl,
                    "return_pct": (
                        return_pct
                    ),
                    "bars_held": (
                        bars_held
                    ),
                    "exit_reason": (
                        "target"
                    ),
                }
            )

            # -----------------------------------------------
            # Reset position state
            # -----------------------------------------------

            units = 0.0
            current_position = 0

            entry_price = None
            entry_time = None

            entry_fee = 0.0
            entry_notional = 0.0

            entry_bar_number = None

        # =====================================================
        # ENTER LONG
        # =====================================================

        if (
            current_position == 0
            and desired_position == 1
        ):

            entry_price = _fill_price(
                raw_price=open_price,
                side=1,
                slippage_rate=(
                    config.slippage_rate
                ),
            )

            # entry_notional + commission <= cash

            entry_notional = (
                cash
                / (
                    1
                    + config.commission_rate
                )
            )

            units = (
                entry_notional
                / entry_price
            )

            entry_fee = (
                entry_notional
                * config.commission_rate
            )

            cash -= (
                entry_notional
                + entry_fee
            )

            # Eliminate tiny floating-point residuals.
            if abs(cash) < 1e-10:
                cash = 0.0

            current_position = 1

            entry_time = timestamp

            entry_bar_number = (
                bar_number
            )

        # =====================================================
        # MARK TO MARKET AT CLOSE
        # =====================================================

        holdings_value = (
            units
            * close_price
        )

        equity = (
            cash
            + holdings_value
        )

        equity_values[
            bar_number
        ] = equity

        position_values[
            bar_number
        ] = current_position

    # =========================================================
    # 7. OPTIONAL FINAL LIQUIDATION
    # =========================================================

    if (
        config.close_open_position_at_end
        and current_position == 1
    ):

        final_bar_number = (
            bar_count - 1
        )

        final_timestamp = (
            timestamps[
                final_bar_number
            ]
        )

        raw_close_price = float(
            close_prices[
                final_bar_number
            ]
        )

        exit_price = _fill_price(
            raw_price=raw_close_price,
            side=-1,
            slippage_rate=(
                config.slippage_rate
            ),
        )

        exit_notional = (
            units
            * exit_price
        )

        exit_fee = (
            exit_notional
            * config.commission_rate
        )

        cash += (
            exit_notional
            - exit_fee
        )

        assert entry_price is not None

        gross_pnl = (
            (
                exit_price
                - entry_price
            )
            * units
        )

        total_fees = (
            entry_fee
            + exit_fee
        )

        net_pnl = (
            gross_pnl
            - total_fees
        )

        return_pct = (
            net_pnl
            / entry_notional
            if entry_notional > 0
            else 0.0
        )

        bars_held = (
            final_bar_number
            - entry_bar_number
            if entry_bar_number
            is not None
            else 0
        )

        completed_trades.append(
            {
                "entry_time": (
                    entry_time
                ),
                "exit_time": (
                    final_timestamp
                ),
                "direction": "long",
                "entry_price": (
                    entry_price
                ),
                "exit_price": (
                    exit_price
                ),
                "units": units,
                "entry_notional": (
                    entry_notional
                ),
                "gross_pnl": (
                    gross_pnl
                ),
                "fees": total_fees,
                "net_pnl": net_pnl,
                "return_pct": (
                    return_pct
                ),
                "bars_held": (
                    bars_held
                ),
                "exit_reason": (
                    "end_of_data"
                ),
            }
        )

        units = 0.0
        current_position = 0

        # The portfolio is entirely cash after final
        # liquidation.
        equity_values[
            final_bar_number
        ] = cash

        position_values[
            final_bar_number
        ] = 0

    # =========================================================
    # 8. CONSTRUCT PUBLIC PANDAS OUTPUT
    # =========================================================

    equity = pd.Series(
        equity_values,
        index=data.index,
        name="equity",
        dtype=float,
    )

    positions = pd.Series(
        position_values,
        index=data.index,
        name="position",
        dtype="int8",
    )

    trade_columns = [
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "units",
        "entry_notional",
        "gross_pnl",
        "fees",
        "net_pnl",
        "return_pct",
        "bars_held",
        "exit_reason",
    ]

    trades = pd.DataFrame(
        completed_trades,
        columns=trade_columns,
    )

    return BacktestResult(
        equity=equity,
        positions=positions,
        trades=trades,
    )