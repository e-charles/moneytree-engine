from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from backtesting.engine import BacktestTrades


FloatArray = NDArray[np.float64]


def calculate_metrics(
    equity: FloatArray,
    trades: BacktestTrades,
    periods_per_year: int = 252,
    *,
    inputs_prevalidated: bool = False,
) -> dict[str, float | int]:
    """
    Calculate performance statistics from a NumPy backtest result.

    Parameters
    ----------
    equity:
        One-dimensional NumPy array containing portfolio equity
        measured at each bar's Close.

    trades:
        Completed trades produced by the NumPy backtesting
        engine.

    periods_per_year:
        Number of observations expected per year.

        252 is appropriate for daily U.S. equity data.

    inputs_prevalidated:
        If False, validate the supplied equity and trade arrays.

        During optimization, the backtesting engine guarantees
        its output contract, so callers may set this to True to
        avoid repeated validation.

    Notes
    -----
    Sharpe ratio assumes a zero risk-free rate.

    Maximum drawdown is returned as a negative number.

    Example:

        -0.20 = 20% maximum drawdown
    """

    # =========================================================
    # VALIDATION
    # =========================================================

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero"
        )

    if not inputs_prevalidated:
        _validate_metric_inputs(
            equity=equity,
            trades=trades,
        )

    # =========================================================
    # PORTFOLIO RETURNS
    # =========================================================
    #
    # Pandas equivalent:
    #
    #     equity.pct_change().dropna()
    #
    # For:
    #
    #     equity = [100, 102, 101]
    #
    # returns becomes:
    #
    #     [
    #         102 / 100 - 1,
    #         101 / 102 - 1,
    #     ]
    #
    # No Series needs to be created.
    # =========================================================

    if equity.size > 1:

        returns = (
            equity[1:]
            / equity[:-1]
            - 1.0
        )

    else:

        returns = np.empty(
            0,
            dtype=np.float64,
        )

    # =========================================================
    # TOTAL RETURN
    # =========================================================

    starting_equity = float(
        equity[0]
    )

    ending_equity = float(
        equity[-1]
    )

    equity_ratio = (
        ending_equity
        / starting_equity
    )

    total_return = (
        equity_ratio
        - 1.0
    )

    # =========================================================
    # ANNUALIZED RETURN
    # =========================================================

    years = (
        returns.size
        / periods_per_year
    )

    if years > 0:

        annualized_return = (
            equity_ratio
            ** (1.0 / years)
            - 1.0
        )

    else:

        annualized_return = np.nan

    # =========================================================
    # MAXIMUM DRAWDOWN
    # =========================================================
    #
    # Pandas equivalent:
    #
    #     running_peak = equity.cummax()
    #
    # NumPy can perform the cumulative maximum directly.
    # =========================================================

    running_peak = (
        np.maximum.accumulate(
            equity
        )
    )

    drawdown = (
        equity
        / running_peak
        - 1.0
    )

    max_drawdown = float(
        np.min(
            drawdown
        )
    )

    # =========================================================
    # VOLATILITY AND SHARPE
    # =========================================================
    #
    # Calculate standard deviation ONCE.
    #
    # The previous implementation called:
    #
    #     returns.std(ddof=1)
    #
    # multiple times.
    # =========================================================

    if returns.size > 1:

        mean_return = float(
            np.mean(
                returns
            )
        )

        return_std = float(
            np.std(
                returns,
                ddof=1,
            )
        )

    else:

        mean_return = np.nan
        return_std = np.nan

    annualization_factor = (
        np.sqrt(
            periods_per_year
        )
    )

    if np.isfinite(return_std):

        annualized_volatility = (
            return_std
            * annualization_factor
        )

    else:

        annualized_volatility = np.nan

    if (
        returns.size > 1
        and return_std > 0
    ):

        sharpe = (
            mean_return
            / return_std
            * annualization_factor
        )

    else:

        sharpe = np.nan

    # =========================================================
    # TRADE COUNT
    # =========================================================

    closed_trades = (
        trades.count
    )

    # =========================================================
    # NO COMPLETED TRADES
    # =========================================================

    if closed_trades == 0:

        return {
            "total_return": float(
                total_return
            ),
            "annualized_return": float(
                annualized_return
            ),
            "annualized_volatility": float(
                annualized_volatility
            ),
            "max_drawdown": (
                max_drawdown
            ),
            "sharpe": float(
                sharpe
            ),

            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,

            "win_rate": np.nan,

            "gross_profit": 0.0,
            "gross_loss": 0.0,

            "profit_factor": np.nan,

            "net_profit": 0.0,

            "average_trade_return": np.nan,
        }

    # =========================================================
    # TRADE STATISTICS
    # =========================================================

    pnl = trades.net_pnl

    winning_mask = (
        pnl > 0
    )

    losing_mask = (
        pnl < 0
    )

    winning_trades = int(
        np.count_nonzero(
            winning_mask
        )
    )

    losing_trades = int(
        np.count_nonzero(
            losing_mask
        )
    )

    net_profit = float(
        np.sum(
            pnl
        )
    )

    gross_profit = float(
        np.sum(
            pnl[
                winning_mask
            ]
        )
    )

    gross_loss = float(
        -np.sum(
            pnl[
                losing_mask
            ]
        )
    )

    # =========================================================
    # WIN RATE
    # =========================================================
    #
    # This preserves the previous behavior:
    #
    #     (pnl > 0).mean()
    #
    # Breakeven trades therefore count as non-winning trades.
    # =========================================================

    win_rate = (
        winning_trades
        / closed_trades
    )

    # =========================================================
    # AVERAGE TRADE RETURN
    # =========================================================

    average_trade_return = float(
        np.mean(
            trades.return_pct
        )
    )

    # =========================================================
    # PROFIT FACTOR
    # =========================================================

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = np.inf

    else:

        profit_factor = np.nan

    # =========================================================
    # RESULT
    # =========================================================

    return {
        "total_return": float(
            total_return
        ),
        "annualized_return": float(
            annualized_return
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "max_drawdown": (
            max_drawdown
        ),
        "sharpe": float(
            sharpe
        ),

        "closed_trades": (
            closed_trades
        ),
        "winning_trades": (
            winning_trades
        ),
        "losing_trades": (
            losing_trades
        ),

        "win_rate": float(
            win_rate
        ),

        "gross_profit": (
            gross_profit
        ),
        "gross_loss": (
            gross_loss
        ),

        "profit_factor": float(
            profit_factor
        ),

        "net_profit": (
            net_profit
        ),

        "average_trade_return": (
            average_trade_return
        ),
    }


def _validate_metric_inputs(
    equity: FloatArray,
    trades: BacktestTrades,
) -> None:
    """
    Validate numerical backtest results before metric
    calculation.

    Optimization may skip this validation because the trusted
    backtesting engine already guarantees these invariants.
    """

    if not isinstance(
        equity,
        np.ndarray,
    ):
        raise TypeError(
            "equity must be a NumPy ndarray"
        )

    if equity.ndim != 1:
        raise ValueError(
            "equity must be one-dimensional"
        )

    if equity.size == 0:
        raise ValueError(
            "equity cannot be empty"
        )

    if not np.isfinite(
        equity
    ).all():
        raise ValueError(
            "equity cannot contain NaN or infinite values"
        )

    if np.any(
        equity <= 0
    ):
        raise ValueError(
            "equity values must be greater than zero"
        )

    trade_count = (
        trades.count
    )

    trade_arrays = (
        trades.exit_bars,
        trades.entry_prices,
        trades.exit_prices,
        trades.units,
        trades.entry_notionals,
        trades.gross_pnl,
        trades.fees,
        trades.net_pnl,
        trades.return_pct,
        trades.bars_held,
        trades.exit_reasons,
    )

    if any(
        array.size != trade_count
        for array in trade_arrays
    ):
        raise ValueError(
            "all trade arrays must have equal length"
        )

    if (
        trade_count > 0
        and not np.isfinite(
            trades.net_pnl
        ).all()
    ):
        raise ValueError(
            "trade net_pnl cannot contain "
            "NaN or infinite values"
        )

    if (
        trade_count > 0
        and not np.isfinite(
            trades.return_pct
        ).all()
    ):
        raise ValueError(
            "trade return_pct cannot contain "
            "NaN or infinite values"
        )