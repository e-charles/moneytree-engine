import pandas as pd
import numpy as np
# import sys
# from pathlib import Path
import pytest

# Adds the parent directory (/workspaces/moneytree-optimizer) to the path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import calculate_metrics


# ============================================================
# Helpers
# ============================================================

def make_equity(values: list[float]) -> pd.Series:
    """
    Create a simple synthetic equity curve.
    """
    return pd.Series(
        values,
        dtype=float,
    )


def make_trades(
    net_pnl: list[float],
    return_pct: list[float] | None = None,
) -> pd.DataFrame:
    """
    Create synthetic completed trades for metrics testing.
    """

    data = {
        "net_pnl": net_pnl,
    }

    if return_pct is not None:
        data["return_pct"] = return_pct

    return pd.DataFrame(data)


# ============================================================
# PORTFOLIO RETURN TESTS
# ============================================================

def test_zero_total_return() -> None:
    equity = make_equity(
        [100_000.0, 100_000.0, 100_000.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics["total_return"] == pytest.approx(0.0)


def test_positive_total_return() -> None:
    equity = make_equity(
        [100_000.0, 110_000.0, 120_000.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    # $100,000 -> $120,000 = +20%
    assert metrics["total_return"] == pytest.approx(0.20)


def test_negative_total_return() -> None:
    equity = make_equity(
        [100_000.0, 95_000.0, 80_000.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    # $100,000 -> $80,000 = -20%
    assert metrics["total_return"] == pytest.approx(-0.20)


# ============================================================
# ANNUALIZED RETURN
# ============================================================

def test_annualized_return() -> None:
    # Three equity observations produce two return periods.
    #
    # By specifying periods_per_year=2, those two return periods
    # represent exactly one year.
    equity = make_equity(
        [100.0, 110.0, 121.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
        periods_per_year=2,
    )

    # 100 -> 121 over one simulated year = +21%
    assert metrics["annualized_return"] == pytest.approx(
        0.21
    )


def test_annualized_return_is_nan_with_only_one_equity_value() -> None:
    equity = make_equity(
        [100_000.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert np.isnan(
        metrics["annualized_return"]
    )


# ============================================================
# DRAWDOWN
# ============================================================

def test_constant_equity_has_zero_max_drawdown() -> None:
    equity = make_equity(
        [100.0, 100.0, 100.0, 100.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_max_drawdown() -> None:
    equity = make_equity(
        [
            100.0,
            120.0,
            110.0,
            90.0,
            105.0,
        ]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    # Peak = 120
    # Trough = 90
    #
    # drawdown = 90 / 120 - 1
    #          = -0.25
    assert metrics["max_drawdown"] == pytest.approx(
        -0.25
    )


def test_drawdown_uses_previous_peak() -> None:
    equity = make_equity(
        [
            100.0,
            150.0,
            140.0,
            130.0,
            160.0,
        ]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    # Maximum decline:
    #
    # 150 -> 130
    #
    # 130 / 150 - 1 = -13.333...%
    expected = (
        130.0 / 150.0 - 1.0
    )

    assert metrics["max_drawdown"] == pytest.approx(
        expected
    )


# ============================================================
# VOLATILITY AND SHARPE
# ============================================================

def test_constant_equity_has_zero_volatility() -> None:
    equity = make_equity(
        [100.0, 100.0, 100.0, 100.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
        periods_per_year=252,
    )

    assert metrics["annualized_volatility"] == pytest.approx(
        0.0
    )


def test_constant_equity_has_nan_sharpe() -> None:
    equity = make_equity(
        [100.0, 100.0, 100.0, 100.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    # Zero standard deviation means the Sharpe denominator
    # is zero, so Sharpe is undefined.
    assert np.isnan(
        metrics["sharpe"]
    )


def test_annualized_volatility() -> None:
    equity = make_equity(
        [
            100.0,
            110.0,
            99.0,
            108.9,
        ]
    )

    trades = pd.DataFrame()

    periods_per_year = 3

    metrics = calculate_metrics(
        equity,
        trades,
        periods_per_year=periods_per_year,
    )

    returns = equity.pct_change().dropna()

    expected_volatility = (
        returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    assert metrics["annualized_volatility"] == pytest.approx(expected_volatility)


def test_sharpe_ratio() -> None:
    equity = make_equity(
        [
            100.0,
            110.0,
            104.5,
            114.95,
        ]
    )

    trades = pd.DataFrame()

    periods_per_year = 3

    metrics = calculate_metrics(
        equity,
        trades,
        periods_per_year=periods_per_year,
    )

    returns = equity.pct_change().dropna()

    expected_sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    assert metrics["sharpe"] == pytest.approx(
        expected_sharpe
    )


# ============================================================
# TRADE METRICS
# ============================================================

def test_trade_statistics() -> None:
    equity = make_equity(
        [100_000.0, 100_100.0]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
            -50.0,
            75.0,
            -25.0,
        ],
        return_pct=[
            0.01,
            -0.005,
            0.0075,
            -0.0025,
        ],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    # Trades:
    #
    # +100
    # -50
    # +75
    # -25
    #
    # Winners = 2
    # Losers  = 2
    # Gross profit = 175
    # Gross loss   = 75
    # Net profit   = 100
    # Profit factor = 175 / 75

    assert metrics["closed_trades"] == 4

    assert metrics["winning_trades"] == 2
    assert metrics["losing_trades"] == 2

    assert metrics["win_rate"] == pytest.approx(
        0.50
    )

    assert metrics["gross_profit"] == pytest.approx(
        175.0
    )

    assert metrics["gross_loss"] == pytest.approx(
        75.0
    )

    assert metrics["profit_factor"] == pytest.approx(
        175.0 / 75.0
    )

    assert metrics["net_profit"] == pytest.approx(
        100.0
    )


def test_average_trade_return() -> None:
    equity = make_equity(
        [100_000.0, 101_000.0]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
            -50.0,
            75.0,
            -25.0,
        ],
        return_pct=[
            0.10,
            -0.05,
            0.075,
            -0.025,
        ],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    expected = np.mean(
        [
            0.10,
            -0.05,
            0.075,
            -0.025,
        ]
    )

    assert metrics["average_trade_return"] == pytest.approx(expected)


def test_missing_return_pct_produces_nan_average_trade_return() -> None:
    equity = make_equity(
        [100_000.0, 101_000.0]
    )

    trades = make_trades(
        net_pnl=[1000.0],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert np.isnan(
        metrics["average_trade_return"]
    )


# ============================================================
# PROFIT FACTOR EDGE CASES
# ============================================================

def test_no_losing_trades_produces_infinite_profit_factor() -> None:
    equity = make_equity(
        [100_000.0, 102_000.0]
    )

    trades = make_trades(
        net_pnl=[
            1000.0,
            500.0,
        ],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics["gross_profit"] == pytest.approx(
        1500.0
    )

    assert metrics["gross_loss"] == pytest.approx(
        0.0
    )

    assert np.isinf(
        metrics["profit_factor"]
    )


def test_only_losing_trades_produces_zero_profit_factor() -> None:
    equity = make_equity(
        [100_000.0, 98_500.0]
    )

    trades = make_trades(
        net_pnl=[
            -1000.0,
            -500.0,
        ],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics["gross_profit"] == pytest.approx(
        0.0
    )

    assert metrics["gross_loss"] == pytest.approx(
        1500.0
    )

    assert metrics["profit_factor"] == pytest.approx(
        0.0
    )


def test_all_breakeven_trades_produce_nan_profit_factor() -> None:
    equity = make_equity(
        [100_000.0, 100_000.0]
    )

    trades = make_trades(
        net_pnl=[
            0.0,
            0.0,
        ],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert np.isnan(
        metrics["profit_factor"]
    )

    assert metrics["win_rate"] == pytest.approx(
        0.0
    )


# ============================================================
# NO-TRADE BEHAVIOR
# ============================================================

def test_no_trades_returns_expected_trade_metrics() -> None:
    equity = make_equity(
        [100_000.0, 100_000.0, 100_000.0]
    )

    trades = pd.DataFrame()

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics["closed_trades"] == 0
    assert metrics["winning_trades"] == 0
    assert metrics["losing_trades"] == 0

    assert metrics["gross_profit"] == pytest.approx(0.0)
    assert metrics["gross_loss"] == pytest.approx(0.0)
    assert metrics["net_profit"] == pytest.approx(0.0)

    assert np.isnan(metrics["win_rate"])
    assert np.isnan(metrics["profit_factor"])
    assert np.isnan(metrics["average_trade_return"])


# ============================================================
# BREAKEVEN TRADE BEHAVIOR
# ============================================================

def test_breakeven_trade_counts_as_closed_but_not_as_win_or_loss() -> None:
    equity = make_equity(
        [100_000.0, 100_000.0]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
            -50.0,
            0.0,
        ],
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics["closed_trades"] == 3

    assert metrics["winning_trades"] == 1
    assert metrics["losing_trades"] == 1

    # Breakeven trade remains in the denominator.
    assert metrics["win_rate"] == pytest.approx(
        1.0 / 3.0
    )


# ============================================================
# VALIDATION
# ============================================================

def test_empty_equity_rejected() -> None:
    equity = pd.Series(
        [],
        dtype=float,
    )

    trades = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="equity cannot be empty",
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_nan_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            float("nan"),
            101_000.0,
        ]
    )

    trades = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="equity cannot contain NaN values",
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_zero_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            0.0,
        ]
    )

    trades = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="equity values must be greater than zero",
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_negative_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            -10_000.0,
        ]
    )

    trades = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="equity values must be greater than zero",
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_zero_periods_per_year_rejected() -> None:
    equity = make_equity(
        [100_000.0, 101_000.0]
    )

    trades = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="periods_per_year must be greater than zero",
    ):
        calculate_metrics(
            equity,
            trades,
            periods_per_year=0,
        )


def test_negative_periods_per_year_rejected() -> None:
    equity = make_equity(
        [100_000.0, 101_000.0]
    )

    trades = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="periods_per_year must be greater than zero",
    ):
        calculate_metrics(
            equity,
            trades,
            periods_per_year=-252,
        )


def test_missing_net_pnl_rejected() -> None:
    equity = make_equity(
        [100_000.0, 101_000.0]
    )

    trades = pd.DataFrame(
        {
            "some_other_column": [1000.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required trade columns",
    ):
        calculate_metrics(
            equity,
            trades,
        )