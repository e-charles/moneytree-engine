# backtesting/metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Calculate performance statistics from a backtest.

    Parameters
    ----------
    equity:
    Portfolio equity measured at each bar's Close.

    trades:
    DataFrame containing completed trades produced by
    the backtesting engine.

    periods_per_year:
    Number of observations expected per year.
    252 is appropriate for daily U.S. equity data.

    Notes
    -----
    Sharpe ratio assumes a zero risk-free rate.

    Maximum drawdown is returned as a negative number.
    Example:
    -0.20 means a 20% maximum drawdown.
    """

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if equity.empty:
        raise ValueError("equity cannot be empty")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be greater than zero")

    if (equity.isna().any()):
        raise ValueError("equity cannot contain NaN values")

    if (equity <= 0).any():
        raise ValueError("equity values must be greater than zero")

    required_trade_cols = {"net_pnl"}

    missing = required_trade_cols.difference( trades.columns)

    if missing and not trades.empty:
        raise ValueError(f"Missing required trade columns: {sorted(missing)}")

    # ---------------------------------------------------------
    # Portfolio return
    # ---------------------------------------------------------

    returns = equity.pct_change().dropna()

    starting_equity = float(equity.iloc[0]) # assumption that equity.iloc[0] == the intial_cash 
    ending_equity = float(equity.iloc[-1])

    total_return = ending_equity / starting_equity - 1.0 

    # ---------------------------------------------------------
    # Annualized return
    # ---------------------------------------------------------
    years = len(returns) / periods_per_year
    annualized_return = (
        (ending_equity / starting_equity) ** (1 / years) - 1
        if years > 0 and starting_equity > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # Drawdown
    # ---------------------------------------------------------

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    # ---------------------------------------------------------
    # Volatility and Sharpe
    # ---------------------------------------------------------

    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # No completed trades
    # ---------------------------------------------------------
    if trades.empty:
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(volatility), 
            "max_drawdown": max_drawdown,
            "sharpe": float(sharpe),

            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,

            "win_rate": np.nan,

            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": np.nan,

            "net_profit": 0.0,
            "average_trade_return": np.nan
        }

    # ---------------------------------------------------------
    # Trade Stats
    # ---------------------------------------------------------
    pnl = trades["net_pnl"]
    net_profit = float( pnl.sum())
    winning_trades = int( (pnl > 0).sum())
    losing_trades = int( (pnl < 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())

    average_trade_return = (
        float(trades["return_pct"].mean())
        if "return_pct" in trades.columns
        else np.nan
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf if gross_profit > 0 else np.nan
    )

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(volatility), 
        "max_drawdown": max_drawdown,
        "sharpe": float(sharpe),

        "closed_trades": int(len(trades)),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,

        "win_rate": float((pnl > 0).mean()),

        "gross_profit": gross_profit, 
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),

        "net_profit": net_profit,
        "average_trade_return": average_trade_return
    }