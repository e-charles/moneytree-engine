# src/trading_research/metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float]:
    if equity.empty:
        raise ValueError("equity cannot be empty")

    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    years = len(equity) / periods_per_year
    annualized_return = (
        (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
        if years > 0 and equity.iloc[0] > 0
        else np.nan
    )

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )

    if trades.empty:
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "max_drawdown": max_drawdown,
            "sharpe": float(sharpe),
            "closed_trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "net_profit": 0.0,
        }

    pnl = trades["net_pnl"]
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf if gross_profit > 0 else np.nan
    )

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": max_drawdown,
        "sharpe": float(sharpe),
        "closed_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(profit_factor),
        "net_profit": float(pnl.sum()),
    }