import numpy as np
import pytest

from backtesting.engine import BacktestTrades
from backtesting.metrics import calculate_metrics


# ============================================================
# Helpers
# ============================================================


def make_equity(
    values: list[float],
) -> np.ndarray:
    """
    Create a NumPy equity curve for metrics testing.
    """
    return np.asarray(
        values,
        dtype=np.float64,
    )


def make_empty_trades() -> BacktestTrades:
    """
    Create an empty BacktestTrades result.
    """
    return BacktestTrades(
        entry_bars=np.empty(
            0,
            dtype=np.int64,
        ),
        exit_bars=np.empty(
            0,
            dtype=np.int64,
        ),
        entry_prices=np.empty(
            0,
            dtype=np.float64,
        ),
        exit_prices=np.empty(
            0,
            dtype=np.float64,
        ),
        units=np.empty(
            0,
            dtype=np.float64,
        ),
        entry_notionals=np.empty(
            0,
            dtype=np.float64,
        ),
        gross_pnl=np.empty(
            0,
            dtype=np.float64,
        ),
        fees=np.empty(
            0,
            dtype=np.float64,
        ),
        net_pnl=np.empty(
            0,
            dtype=np.float64,
        ),
        return_pct=np.empty(
            0,
            dtype=np.float64,
        ),
        bars_held=np.empty(
            0,
            dtype=np.int64,
        ),
        exit_reasons=np.empty(
            0,
            dtype=np.int8,
        ),
    )


def make_trades(
    net_pnl: list[float],
    return_pct: list[float] | None = None,
) -> BacktestTrades:
    """
    Create synthetic completed trades for metrics testing.

    Fields not directly relevant to metric calculations are
    populated with valid placeholder values.
    """
    trade_count = len(
        net_pnl
    )

    pnl = np.asarray(
        net_pnl,
        dtype=np.float64,
    )

    if return_pct is None:
        returns = np.zeros(
            trade_count,
            dtype=np.float64,
        )
    else:
        if len(return_pct) != trade_count:
            raise ValueError(
                "return_pct must match net_pnl length"
            )

        returns = np.asarray(
            return_pct,
            dtype=np.float64,
        )

    return BacktestTrades(
        entry_bars=np.arange(
            trade_count,
            dtype=np.int64,
        ),
        exit_bars=np.arange(
            1,
            trade_count + 1,
            dtype=np.int64,
        ),
        entry_prices=np.full(
            trade_count,
            100.0,
            dtype=np.float64,
        ),
        exit_prices=np.full(
            trade_count,
            100.0,
            dtype=np.float64,
        ),
        units=np.ones(
            trade_count,
            dtype=np.float64,
        ),
        entry_notionals=np.full(
            trade_count,
            100.0,
            dtype=np.float64,
        ),
        gross_pnl=pnl.copy(),
        fees=np.zeros(
            trade_count,
            dtype=np.float64,
        ),
        net_pnl=pnl,
        return_pct=returns,
        bars_held=np.ones(
            trade_count,
            dtype=np.int64,
        ),
        exit_reasons=np.zeros(
            trade_count,
            dtype=np.int8,
        ),
    )


# ============================================================
# PORTFOLIO RETURN TESTS
# ============================================================


def test_zero_total_return() -> None:
    equity = make_equity(
        [
            100_000.0,
            100_000.0,
            100_000.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    assert metrics[
        "total_return"
    ] == pytest.approx(
        0.0
    )


def test_positive_total_return() -> None:
    equity = make_equity(
        [
            100_000.0,
            110_000.0,
            120_000.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    # $100,000 -> $120,000 = +20%
    assert metrics[
        "total_return"
    ] == pytest.approx(
        0.20
    )


def test_negative_total_return() -> None:
    equity = make_equity(
        [
            100_000.0,
            95_000.0,
            80_000.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    # $100,000 -> $80,000 = -20%
    assert metrics[
        "total_return"
    ] == pytest.approx(
        -0.20
    )


# ============================================================
# ANNUALIZED RETURN
# ============================================================


def test_annualized_return() -> None:
    # Three equity observations produce two return periods.
    #
    # periods_per_year=2 means those two periods represent
    # exactly one year.
    equity = make_equity(
        [
            100.0,
            110.0,
            121.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
        periods_per_year=2,
    )

    assert metrics[
        "annualized_return"
    ] == pytest.approx(
        0.21
    )


def test_annualized_return_is_nan_with_only_one_equity_value() -> None:
    equity = make_equity(
        [100_000.0]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    assert np.isnan(
        metrics[
            "annualized_return"
        ]
    )


# ============================================================
# DRAWDOWN
# ============================================================


def test_constant_equity_has_zero_max_drawdown() -> None:
    equity = make_equity(
        [
            100.0,
            100.0,
            100.0,
            100.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    assert metrics[
        "max_drawdown"
    ] == pytest.approx(
        0.0
    )


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

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    # Peak = 120
    # Trough = 90
    #
    # 90 / 120 - 1 = -0.25
    assert metrics[
        "max_drawdown"
    ] == pytest.approx(
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

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    expected = (
        130.0 / 150.0 - 1.0
    )

    assert metrics[
        "max_drawdown"
    ] == pytest.approx(
        expected
    )


# ============================================================
# VOLATILITY AND SHARPE
# ============================================================


def test_constant_equity_has_zero_volatility() -> None:
    equity = make_equity(
        [
            100.0,
            100.0,
            100.0,
            100.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
        periods_per_year=252,
    )

    assert metrics[
        "annualized_volatility"
    ] == pytest.approx(
        0.0
    )


def test_constant_equity_has_nan_sharpe() -> None:
    equity = make_equity(
        [
            100.0,
            100.0,
            100.0,
            100.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    assert np.isnan(
        metrics[
            "sharpe"
        ]
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

    periods_per_year = 3

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
        periods_per_year=periods_per_year,
    )

    returns = (
        equity[1:]
        / equity[:-1]
        - 1.0
    )

    expected_volatility = (
        np.std(
            returns,
            ddof=1,
        )
        * np.sqrt(
            periods_per_year
        )
    )

    assert metrics[
        "annualized_volatility"
    ] == pytest.approx(
        expected_volatility
    )


def test_sharpe_ratio() -> None:
    equity = make_equity(
        [
            100.0,
            110.0,
            104.5,
            114.95,
        ]
    )

    periods_per_year = 3

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
        periods_per_year=periods_per_year,
    )

    returns = (
        equity[1:]
        / equity[:-1]
        - 1.0
    )

    expected_sharpe = (
        np.mean(
            returns
        )
        / np.std(
            returns,
            ddof=1,
        )
        * np.sqrt(
            periods_per_year
        )
    )

    assert metrics[
        "sharpe"
    ] == pytest.approx(
        expected_sharpe
    )


def test_single_return_has_nan_volatility_and_sharpe() -> None:
    equity = make_equity(
        [
            100.0,
            110.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    assert np.isnan(
        metrics[
            "annualized_volatility"
        ]
    )

    assert np.isnan(
        metrics[
            "sharpe"
        ]
    )


# ============================================================
# TRADE METRICS
# ============================================================


def test_trade_statistics() -> None:
    equity = make_equity(
        [
            100_000.0,
            100_100.0,
        ]
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

    assert metrics[
        "closed_trades"
    ] == 4

    assert metrics[
        "winning_trades"
    ] == 2

    assert metrics[
        "losing_trades"
    ] == 2

    assert metrics[
        "win_rate"
    ] == pytest.approx(
        0.50
    )

    assert metrics[
        "gross_profit"
    ] == pytest.approx(
        175.0
    )

    assert metrics[
        "gross_loss"
    ] == pytest.approx(
        75.0
    )

    assert metrics[
        "profit_factor"
    ] == pytest.approx(
        175.0 / 75.0
    )

    assert metrics[
        "net_profit"
    ] == pytest.approx(
        100.0
    )


def test_average_trade_return() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
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

    assert metrics[
        "average_trade_return"
    ] == pytest.approx(
        expected
    )


# ============================================================
# PROFIT FACTOR EDGE CASES
# ============================================================


def test_no_losing_trades_produces_infinite_profit_factor() -> None:
    equity = make_equity(
        [
            100_000.0,
            102_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            1000.0,
            500.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics[
        "gross_profit"
    ] == pytest.approx(
        1500.0
    )

    assert metrics[
        "gross_loss"
    ] == pytest.approx(
        0.0
    )

    assert np.isinf(
        metrics[
            "profit_factor"
        ]
    )


def test_only_losing_trades_produces_zero_profit_factor() -> None:
    equity = make_equity(
        [
            100_000.0,
            98_500.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            -1000.0,
            -500.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics[
        "gross_profit"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "gross_loss"
    ] == pytest.approx(
        1500.0
    )

    assert metrics[
        "profit_factor"
    ] == pytest.approx(
        0.0
    )


def test_all_breakeven_trades_produce_nan_profit_factor() -> None:
    equity = make_equity(
        [
            100_000.0,
            100_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            0.0,
            0.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert np.isnan(
        metrics[
            "profit_factor"
        ]
    )

    assert metrics[
        "win_rate"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# NO-TRADE BEHAVIOR
# ============================================================


def test_no_trades_returns_expected_trade_metrics() -> None:
    equity = make_equity(
        [
            100_000.0,
            100_000.0,
            100_000.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        make_empty_trades(),
    )

    assert metrics[
        "closed_trades"
    ] == 0

    assert metrics[
        "winning_trades"
    ] == 0

    assert metrics[
        "losing_trades"
    ] == 0

    assert metrics[
        "gross_profit"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "gross_loss"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "net_profit"
    ] == pytest.approx(
        0.0
    )

    assert np.isnan(
        metrics[
            "win_rate"
        ]
    )

    assert np.isnan(
        metrics[
            "profit_factor"
        ]
    )

    assert np.isnan(
        metrics[
            "average_trade_return"
        ]
    )


# ============================================================
# BREAKEVEN TRADE BEHAVIOR
# ============================================================


def test_breakeven_trade_counts_as_closed_but_not_as_win_or_loss() -> None:
    equity = make_equity(
        [
            100_000.0,
            100_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
            -50.0,
            0.0,
        ]
    )

    metrics = calculate_metrics(
        equity,
        trades,
    )

    assert metrics[
        "closed_trades"
    ] == 3

    assert metrics[
        "winning_trades"
    ] == 1

    assert metrics[
        "losing_trades"
    ] == 1

    # Breakeven trades remain in the win-rate denominator.
    assert metrics[
        "win_rate"
    ] == pytest.approx(
        1.0 / 3.0
    )


# ============================================================
# VALIDATION
# ============================================================


def test_equity_must_be_numpy_array() -> None:
    equity = [
        100_000.0,
        101_000.0,
    ]

    with pytest.raises(
        TypeError,
        match="equity must be a NumPy ndarray",
    ):
        calculate_metrics(
            equity,  # type: ignore[arg-type]
            make_empty_trades(),
        )


def test_empty_equity_rejected() -> None:
    equity = make_equity(
        []
    )

    with pytest.raises(
        ValueError,
        match="equity cannot be empty",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
        )


def test_multidimensional_equity_rejected() -> None:
    equity = np.asarray(
        [
            [
                100_000.0,
                101_000.0,
            ]
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="equity must be one-dimensional",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
        )


def test_nan_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            float("nan"),
            101_000.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="equity cannot contain NaN or infinite values",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
        )


def test_infinite_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            float("inf"),
            101_000.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="equity cannot contain NaN or infinite values",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
        )


def test_zero_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            0.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="equity values must be greater than zero",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
        )


def test_negative_equity_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            -10_000.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="equity values must be greater than zero",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
        )


def test_zero_periods_per_year_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="periods_per_year must be greater than zero",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
            periods_per_year=0,
        )


def test_negative_periods_per_year_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="periods_per_year must be greater than zero",
    ):
        calculate_metrics(
            equity,
            make_empty_trades(),
            periods_per_year=-252,
        )


def test_trade_arrays_must_have_equal_length() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
            200.0,
        ],
        return_pct=[
            0.01,
            0.02,
        ],
    )

    # BacktestTrades.count is determined by entry_bars.
    #
    # Create an invalid BacktestTrades object where net_pnl
    # contains one observation but the other trade arrays
    # contain two.
    invalid_trades = BacktestTrades(
        entry_bars=trades.entry_bars,
        exit_bars=trades.exit_bars,
        entry_prices=trades.entry_prices,
        exit_prices=trades.exit_prices,
        units=trades.units,
        entry_notionals=trades.entry_notionals,
        gross_pnl=trades.gross_pnl,
        fees=trades.fees,
        net_pnl=np.asarray(
            [100.0],
            dtype=np.float64,
        ),
        return_pct=trades.return_pct,
        bars_held=trades.bars_held,
        exit_reasons=trades.exit_reasons,
    )

    with pytest.raises(
        ValueError,
        match="all trade arrays must have equal length",
    ):
        calculate_metrics(
            equity,
            invalid_trades,
        )


def test_nan_trade_net_pnl_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            float("nan"),
        ],
        return_pct=[
            0.01,
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "trade net_pnl cannot contain "
            "NaN or infinite values"
        ),
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_infinite_trade_net_pnl_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            float("inf"),
        ],
        return_pct=[
            0.01,
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "trade net_pnl cannot contain "
            "NaN or infinite values"
        ),
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_nan_trade_return_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
        ],
        return_pct=[
            float("nan"),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "trade return_pct cannot contain "
            "NaN or infinite values"
        ),
    ):
        calculate_metrics(
            equity,
            trades,
        )


def test_infinite_trade_return_rejected() -> None:
    equity = make_equity(
        [
            100_000.0,
            101_000.0,
        ]
    )

    trades = make_trades(
        net_pnl=[
            100.0,
        ],
        return_pct=[
            float("inf"),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "trade return_pct cannot contain "
            "NaN or infinite values"
        ),
    ):
        calculate_metrics(
            equity,
            trades,
        )