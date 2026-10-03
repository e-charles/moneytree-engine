# scripts/run_double_dip.py

from __future__ import annotations

import sys
from pathlib import Path

# Adds the parent directory (/workspaces/moneytree-optimizer) to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator

from strats.double_dip import rsi_double_dip_targets
from backtesting.engine import BacktestConfig, run_backtest
from backtesting.metrics import calculate_metrics


def fetch_stock_data(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download historical stock data.
    """

    stock = yf.Ticker(ticker)

    data = stock.history(
        start=start,
        end=end,
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError(
            f"No market data returned for {ticker}"
        )

    required_columns = {"Open", "Close"}

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return data


def run_rsi_double_dip(
    ticker: str,
    start: str,
    end: str,
    rsi_length: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    max_bars_between_dips: int = 10,
) -> None:
    """
    Run the complete RSI Double Dip research pipeline.
    """

    # =========================================================
    # 1. Fetch market data
    # =========================================================

    data = fetch_stock_data(
        ticker=ticker,
        start=start,
        end=end,
    )

    # =========================================================
    # 2. Calculate RSI
    # =========================================================

    rsi = RSIIndicator(
        close=data["Close"],
        window=rsi_length,
    ).rsi()

    # =========================================================
    # 3. Generate strategy targets
    # =========================================================

    target_at_close = rsi_double_dip_targets(
        rsi=rsi,
        oversold=oversold,
        overbought=overbought,
        max_bars_between_dips=max_bars_between_dips,
    )

    # =========================================================
    # 4. Configure execution assumptions
    # =========================================================

    config = BacktestConfig(
        initial_cash=100_000.0,

        # 5 basis points per transaction
        commission_rate=0.0005,

        # 2 basis points of adverse slippage
        slippage_rate=0.0002,

        close_open_position_at_end=True,
    )

    # =========================================================
    # 5. Run backtest
    # =========================================================

    result = run_backtest(
        data=data,
        target_at_close=target_at_close,
        config=config,
    )

    # =========================================================
    # 6. Calculate metrics
    # =========================================================

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=252,
    )

    # =========================================================
    # 7. Print configuration
    # =========================================================

    print()
    print("=" * 60)
    print("RSI DOUBLE DIP BACKTEST")
    print("=" * 60)

    print(f"Ticker:                  {ticker}")
    print(f"Start:                   {data.index[0]}")
    print(f"End:                     {data.index[-1]}")
    print()

    print("Strategy Parameters")
    print("-" * 60)

    print(f"RSI Length:              {rsi_length}")
    print(f"Oversold:                {oversold}")
    print(f"Overbought:              {overbought}")
    print(
        f"Max Bars Between Dips:   "
        f"{max_bars_between_dips}"
    )

    # =========================================================
    # 8. Print performance
    # =========================================================

    print()
    print("Performance")
    print("-" * 60)

    print(
        f"Initial Equity:          "
        f"${config.initial_cash:,.2f}"
    )

    print(
        f"Final Equity:            "
        f"${result.equity.iloc[-1]:,.2f}"
    )

    print(
        f"Total Return:            "
        f"{metrics['total_return']:.2%}"
    )

    print(
        f"Annualized Return:       "
        f"{metrics['annualized_return']:.2%}"
    )

    print(
        f"Maximum Drawdown:        "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Annualized Volatility:   "
        f"{metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe Ratio:            "
        f"{metrics['sharpe']:.2f}"
    )

    # =========================================================
    # 9. Print trade statistics
    # =========================================================

    print()
    print("Trade Statistics")
    print("-" * 60)

    print(
        f"Closed Trades:           "
        f"{metrics['closed_trades']}"
    )

    print(
        f"Winning Trades:          "
        f"{metrics['winning_trades']}"
    )

    print(
        f"Losing Trades:           "
        f"{metrics['losing_trades']}"
    )

    win_rate = metrics["win_rate"]

    if pd.isna(win_rate):
        print("Win Rate:                N/A")
    else:
        print(
            f"Win Rate:                "
            f"{win_rate:.2%}"
        )

    profit_factor = metrics["profit_factor"]

    if pd.isna(profit_factor):
        print("Profit Factor:           N/A")
    elif profit_factor == float("inf"):
        print("Profit Factor:           inf")
    else:
        print(
            f"Profit Factor:           "
            f"{profit_factor:.2f}"
        )

    print(
        f"Gross Profit:            "
        f"${metrics['gross_profit']:,.2f}"
    )

    print(
        f"Gross Loss:              "
        f"${metrics['gross_loss']:,.2f}"
    )

    print(
        f"Net Profit:              "
        f"${metrics['net_profit']:,.2f}"
    )

    average_trade_return = metrics[
        "average_trade_return"
    ]

    if pd.isna(average_trade_return):
        print("Average Trade Return:    N/A")
    else:
        print(
            f"Average Trade Return:    "
            f"{average_trade_return:.2%}"
        )

    # =========================================================
    # 10. Print completed trades
    # =========================================================

    print()
    print("Completed Trades")
    print("-" * 60)

    if result.trades.empty:
        print("No completed trades.")

    else:
        print(
            result.trades.to_string(
                index=False,
            )
        )


if __name__ == "__main__":

    run_rsi_double_dip(
        ticker="TSLA",
        start="2015-01-01",
        end="2026-01-01",
        rsi_length=14,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )