# backtesting/engine.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration controlling execution assumptions.

    initial_cash:
        Starting portfolio value.

    commission_rate:
        Commission charged on each transaction as a fraction
        of transaction notional.
        Example: 0.0005 = 0.05% = 5 basis points.

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

    A buy pays slightly more than the quoted price.
    A sell receives slightly less than the quoted price.
    """

    if side not in (-1, 1):
        raise ValueError("side must be either -1 or +1")

    return raw_price * (1 + side * slippage_rate)


def _validate_inputs(
    data: pd.DataFrame,
    target_at_close: pd.Series,
    config: BacktestConfig,
) -> None:
    """
    Validate market data, strategy targets, and backtest configuration.
    """

    if data.empty:
        raise ValueError("data cannot be empty")

    required_columns = {"Open", "Close"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not target_at_close.index.equals(data.index):
        raise ValueError(
            "target_at_close index must exactly match data index"
        )

    if target_at_close.isna().any():
        raise ValueError("target_at_close cannot contain NaN values")

    invalid_targets = ~target_at_close.isin([0, 1])

    if invalid_targets.any():
        raise ValueError(
            "target_at_close may contain only 0 or 1"
        )

    if data[["Open", "Close"]].isna().any().any():
        raise ValueError(
            "Open and Close prices cannot contain NaN values"
        )

    if (data[["Open", "Close"]] <= 0).any().any():
        raise ValueError(
            "Open and Close prices must be greater than zero"
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
) -> BacktestResult:
    """
    Run a long-only backtest.

    Strategy targets are assumed to be generated using information
    available at each bar's Close.

    Therefore:

        target_at_close[t]

    becomes executable at:

        Open[t + 1]

    The engine performs that conversion internally to prevent
    accidental look-ahead bias.

    Target meanings:

        0 = flat
        1 = long

    Execution rules:

        0 -> 1 : buy at that bar's Open
        1 -> 1 : hold
        1 -> 0 : sell at that bar's Open
        0 -> 0 : remain flat

    Returns:
        BacktestResult containing the equity curve,
        executed positions, and completed trades.
    """

    if config is None:
        config = BacktestConfig()

    _validate_inputs(
        data=data,
        target_at_close=target_at_close,
        config=config,
    )

    # ---------------------------------------------------------
    # Convert Close-based strategy decisions into positions
    # executable at the following bar's Open.
    # ---------------------------------------------------------

    position_at_open = (
        target_at_close
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    # ---------------------------------------------------------
    # Portfolio state
    # ---------------------------------------------------------

    cash = float(config.initial_cash)
    units = 0.0
    current_position = 0

    # ---------------------------------------------------------
    # Current trade state
    # ---------------------------------------------------------

    entry_price: float | None = None
    entry_time = None
    entry_fee = 0.0
    entry_notional = 0.0
    entry_bar_number: int | None = None

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    equity_values: list[float] = []
    position_values: list[int] = []
    completed_trades: list[dict] = []

    # ---------------------------------------------------------
    # Main execution loop
    # ---------------------------------------------------------

    for bar_number, (timestamp, row) in enumerate(data.iterrows()):

        desired_position = int(position_at_open.loc[timestamp])

        open_price = float(row["Open"])
        close_price = float(row["Close"])

        # =====================================================
        # EXIT LONG
        # =====================================================

        if current_position == 1 and desired_position == 0:

            exit_price = _fill_price(
                raw_price=open_price,
                side=-1,
                slippage_rate=config.slippage_rate,
            )

            exit_notional = units * exit_price

            exit_fee = (
                exit_notional
                * config.commission_rate
            )

            # Money received from selling the shares.
            cash += exit_notional - exit_fee

            gross_pnl = (
                (exit_price - entry_price)
                * units
            )

            total_fees = entry_fee + exit_fee

            net_pnl = (
                gross_pnl
                - total_fees
            )

            return_pct = (
                net_pnl / entry_notional
                if entry_notional > 0
                else 0.0
            )

            bars_held = (
                bar_number - entry_bar_number
                if entry_bar_number is not None
                else 0
            )

            completed_trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "direction": "long",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "units": units,
                    "entry_notional": entry_notional,
                    "gross_pnl": gross_pnl,
                    "fees": total_fees,
                    "net_pnl": net_pnl,
                    "return_pct": return_pct,
                    "bars_held": bars_held,
                    "exit_reason": "target",
                }
            )

            # Reset position state.
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

        if current_position == 0 and desired_position == 1:

            entry_price = _fill_price(
                raw_price=open_price,
                side=1,
                slippage_rate=config.slippage_rate,
            )

            # We want:
            #
            # entry_notional + entry_fee <= cash
            #
            # where:
            #
            # entry_fee = entry_notional * commission_rate
            #
            # Therefore:
            #
            # entry_notional =
            # cash / (1 + commission_rate)

            entry_notional = (
                cash
                / (1 + config.commission_rate)
            )

            units = (
                entry_notional
                / entry_price
            )

            entry_fee = (
                entry_notional
                * config.commission_rate
            )

            # Pay for the shares AND the transaction fee.
            cash -= (
                entry_notional
                + entry_fee
            )

            # Floating-point calculations can leave an extremely
            # small residual such as -1e-11.
            if abs(cash) < 1e-10:
                cash = 0.0

            current_position = 1
            entry_time = timestamp
            entry_bar_number = bar_number

        # =====================================================
        # MARK PORTFOLIO TO MARKET AT CLOSE
        # =====================================================

        holdings_value = (
            units * close_price
        )

        equity = (
            cash + holdings_value
        )

        equity_values.append(equity)
        position_values.append(current_position)

    # ---------------------------------------------------------
    # Optional forced liquidation at final Close
    # ---------------------------------------------------------

    if (
        config.close_open_position_at_end
        and current_position == 1
    ):
        final_bar_number = len(data) - 1
        final_timestamp = data.index[-1]

        raw_close_price = float(
            data["Close"].iloc[-1]
        )

        exit_price = _fill_price(
            raw_price=raw_close_price,
            side=-1,
            slippage_rate=config.slippage_rate,
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

        gross_pnl = (
            (exit_price - entry_price)
            * units
        )

        total_fees = (
            entry_fee + exit_fee
        )

        net_pnl = (
            gross_pnl - total_fees
        )

        return_pct = (
            net_pnl / entry_notional
            if entry_notional > 0
            else 0.0
        )

        bars_held = (
            final_bar_number - entry_bar_number
            if entry_bar_number is not None
            else 0
        )

        completed_trades.append(
            {
                "entry_time": entry_time,
                "exit_time": final_timestamp,
                "direction": "long",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "units": units,
                "entry_notional": entry_notional,
                "gross_pnl": gross_pnl,
                "fees": total_fees,
                "net_pnl": net_pnl,
                "return_pct": return_pct,
                "bars_held": bars_held,
                "exit_reason": "end_of_data",
            }
        )

        # The final portfolio is now entirely cash.
        units = 0.0
        current_position = 0

        equity_values[-1] = cash
        position_values[-1] = 0

    # ---------------------------------------------------------
    # Construct output
    # ---------------------------------------------------------

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