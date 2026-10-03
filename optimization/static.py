# /optimization/static.py

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator

from strats.double_dip import rsi_double_dip_targets

from backtesting.engine import BacktestConfig, run_backtest

from backtesting.metrics import calculate_metrics

from optimization.search_space import DoubleDipParameters, DoubleDipSearchSpace


def _validate_market_data(data: pd.DataFrame) -> None:
    """
    Validate the market data required by the optimizer.
    """

    if data.empty:
        raise ValueError("data cannot be empty")

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

    if data[["Open", "Close"]].isna().any().any():
        raise ValueError(
            "Open and Close cannot contain NaN values"
        )

    if (data[["Open", "Close"]] <= 0).any().any():
        raise ValueError(
            "Open and Close prices must be greater than zero"
        )


def _calculate_rsi(
    close: pd.Series,
    rsi_length: int,
) -> pd.Series:
    """
    Calculate RSI for one candidate configuration.
    """

    return RSIIndicator(
        close=close,
        window=rsi_length,
    ).rsi()


def evaluate_double_dip_candidate(
    data: pd.DataFrame,
    parameters: DoubleDipParameters,
    backtest_config: BacktestConfig | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """
    Evaluate one RSI Double Dip parameter configuration.

    Pipeline:

        market data
            ↓
        RSI
            ↓
        Double Dip targets
            ↓
        backtest engine
            ↓
        metrics

    Returns one flat dictionary suitable for conversion
    into a DataFrame row.
    """

    if backtest_config is None:
        backtest_config = BacktestConfig()

    # ---------------------------------------------------------
    # RSI
    # ---------------------------------------------------------

    rsi = _calculate_rsi(
        close=data["Close"],
        rsi_length=parameters.rsi_length,
    )

    # ---------------------------------------------------------
    # Strategy
    # ---------------------------------------------------------

    target_at_close = rsi_double_dip_targets(
        rsi=rsi,
        oversold=parameters.oversold,
        overbought=parameters.overbought,
        max_bars_between_dips=(
            parameters.max_bars_between_dips
        ),
    )

    # ---------------------------------------------------------
    # Backtest
    # ---------------------------------------------------------

    result = run_backtest(
        data=data,
        target_at_close=target_at_close,
        config=backtest_config,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=periods_per_year,
    )

    # ---------------------------------------------------------
    # Flatten result
    # ---------------------------------------------------------

    return {
        "rsi_length": parameters.rsi_length,
        "oversold": parameters.oversold,
        "overbought": parameters.overbought,
        "max_bars_between_dips": (
            parameters.max_bars_between_dips
        ),

        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualized_return"],
        "max_drawdown": metrics["max_drawdown"],
        "annualized_volatility": (
            metrics["annualized_volatility"]
        ),
        "sharpe": metrics["sharpe"],

        "closed_trades": metrics["closed_trades"],
        "winning_trades": metrics["winning_trades"],
        "losing_trades": metrics["losing_trades"],

        "win_rate": metrics["win_rate"],

        "gross_profit": metrics["gross_profit"],
        "gross_loss": metrics["gross_loss"],
        "profit_factor": metrics["profit_factor"],

        "net_profit": metrics["net_profit"],
        "average_trade_return": (
            metrics["average_trade_return"]
        ),

        # Useful diagnostic columns.
        "final_equity": float(
            result.equity.iloc[-1]
        ),
    }


def optimize_double_dip(
    data: pd.DataFrame,
    search_space: DoubleDipSearchSpace,
    backtest_config: BacktestConfig | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Evaluate every valid RSI Double Dip candidate.

    IMPORTANT:
    This function does NOT select a winner.

    It returns the complete result table so that candidate
    eligibility, ranking, robustness, and walk-forward
    selection can be handled independently.
    """

    _validate_market_data(data)

    if backtest_config is None:
        backtest_config = BacktestConfig