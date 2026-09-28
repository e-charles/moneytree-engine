# src/trading_research/engine.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0005   # 5 bps per transaction side
    slippage_rate: float = 0.0002     # 2 bps per transaction side
    close_open_position_at_end: bool = True


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame


def _fill_price(raw_price: float, side: int, slippage_rate: float) -> float:
    """
    side: +1 means buy, -1 means sell.
    A buy fills higher; a sell fills lower.
    """
    return raw_price * (1 + side * slippage_rate)


def run_backtest(
    data: pd.DataFrame,
    target_position: pd.Series,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """
    Executes target position changes at each bar's Open.

    `target_position` must already have been shifted if derived from Close.
    Expected values are -1, 0, +1.
    """
    required_columns = {"Open", "Close"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    target_position = target_position.reindex(data.index).fillna(0).astype(int)

    invalid = ~target_position.isin([-1, 0, 1])
    if invalid.any():
        raise ValueError("target_position may contain only -1, 0, or +1")

    cash = config.initial_cash
    units = 0.0
    current_position = 0
    entry_price: float | None = None
    entry_time = None
    entry_fee = 0.0

    equity_values: list[float] = []
    position_values: list[int] = []
    completed_trades: list[dict] = []

    for timestamp, row in data.iterrows():
        desired = int(target_position.loc[timestamp])
        open_price = float(row["Open"])
        close_price = float(row["Close"])

        if desired != current_position:
            # Close existing position first.
            if current_position != 0:
                exit_side = -current_position
                exit_price = _fill_price(
                    open_price,
                    exit_side,
                    config.slippage_rate,
                )
                exit_notional = abs(units) * exit_price
                exit_fee = exit_notional * config.commission_rate

                if current_position == 1:
                    gross_pnl = (exit_price - entry_price) * units
                else:
                    gross_pnl = (entry_price - exit_price) * abs(units)

                net_pnl = gross_pnl - entry_fee - exit_fee

                cash += gross_pnl - exit_fee

                completed_trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "direction": "long" if current_position == 1 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "fees": entry_fee + exit_fee,
                        "net_pnl": net_pnl,
                    }
                )

                units = 0.0
                current_position = 0
                entry_price = None
                entry_time = None
                entry_fee = 0.0

            # Open desired position using all available equity as a simple baseline.
            if desired != 0:
                entry_side = desired
                fill = _fill_price(open_price, entry_side, config.slippage_rate)
                equity_before_entry = cash
                units = (equity_before_entry / fill) * desired
                entry_notional = abs(units) * fill
                entry_fee = entry_notional * config.commission_rate
                cash -= entry_fee

                current_position = desired
                entry_price = fill
                entry_time = timestamp

        marked_position_value = units * close_price
        equity = cash + marked_position_value

        equity_values.append(equity)
        position_values.append(current_position)

    # Optional forced final close at the final close.
    if config.close_open_position_at_end and current_position != 0:
        timestamp = data.index[-1]
        close_price = float(data["Close"].iloc[-1])
        exit_side = -current_position
        exit_price = _fill_price(close_price, exit_side, config.slippage_rate)
        exit_fee = abs(units) * exit_price * config.commission_rate

        if current_position == 1:
            gross_pnl = (exit_price - entry_price) * units
        else:
            gross_pnl = (entry_price - exit_price) * abs(units)

        net_pnl = gross_pnl - entry_fee - exit_fee

        completed_trades.append(
            {
                "entry_time": entry_time,
                "exit_time": timestamp,
                "direction": "long" if current_position == 1 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "fees": entry_fee + exit_fee,
                "net_pnl": net_pnl,
            }
        )

        equity_values[-1] += gross_pnl - exit_fee
        position_values[-1] = 0

    trades = pd.DataFrame(completed_trades)
    return BacktestResult(
        equity=pd.Series(equity_values, index=data.index, name="equity"),
        positions=pd.Series(position_values, index=data.index, name="position"),
        trades=trades,
    )