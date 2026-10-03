import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestConfig
from optimization.search_space import (
    DoubleDipParameters,
    DoubleDipSearchSpace,
)
from optimization.static import (
    evaluate_double_dip_candidate_from_rsi,
    optimize_double_dip,
)


# ============================================================
# Helpers
# ============================================================


def make_market_data(
    periods: int = 100,
) -> pd.DataFrame:
    """
    Create deterministic market data with enough movement
    for RSI calculations.

    This is not intended to model a real stock. It simply
    provides valid market data for optimizer tests.
    """

    index = pd.date_range(
        "2025-01-01",
        periods=periods,
        freq="D",
    )

    # Repeating movements prevent the RSI from remaining
    # permanently at one extreme.
    pattern = np.array(
        [
            0.0,
            -2.0,
            -4.0,
            -6.0,
            -3.0,
            1.0,
            4.0,
            6.0,
            3.0,
            0.0,
        ]
    )

    repeated = np.resize(
        pattern,
        periods,
    )

    close = 100.0 + repeated

    # Slightly different Open prices.
    open_ = close - 0.5

    return pd.DataFrame(
        {
            "Open": open_,
            "Close": close,
        },
        index=index,
    )


def no_cost_config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=100_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        close_open_position_at_end=True,
    )


# ============================================================
# STATIC OPTIMIZER
# ============================================================


def test_optimizer_evaluates_every_candidate() -> None:
    data = make_market_data()

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5, 10),
        oversold_levels=(25.0, 30.0),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    # 2 RSI lengths
    # x 2 oversold values
    # x 1 overbought value
    # x 1 gap
    #
    # = 4 candidates

    results = optimize_double_dip(
        data=data,
        search_space=search_space,
        backtest_config=no_cost_config(),
    )

    assert len(results) == 4


def test_optimizer_returns_expected_parameter_columns() -> None:
    data = make_market_data()

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5,),
        oversold_levels=(25.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    results = optimize_double_dip(
        data=data,
        search_space=search_space,
        backtest_config=no_cost_config(),
    )

    expected_columns = {
        "rsi_length",
        "oversold",
        "overbought",
        "max_bars_between_dips",
    }

    assert expected_columns.issubset(
        results.columns
    )


def test_optimizer_returns_expected_metric_columns() -> None:
    data = make_market_data()

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5,),
        oversold_levels=(25.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    results = optimize_double_dip(
        data=data,
        search_space=search_space,
        backtest_config=no_cost_config(),
    )

    expected_columns = {
        "total_return",
        "annualized_return",
        "max_drawdown",
        "annualized_volatility",
        "sharpe",
        "closed_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "gross_profit",
        "gross_loss",
        "profit_factor",
        "net_profit",
        "average_trade_return",
        "final_equity",
    }

    assert expected_columns.issubset(
        results.columns
    )


def test_optimizer_preserves_candidate_parameters() -> None:
    data = make_market_data()

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5, 10),
        oversold_levels=(25.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(3, 7),
    )

    results = optimize_double_dip(
        data=data,
        search_space=search_space,
        backtest_config=no_cost_config(),
    )

    actual = {
        (
            int(row["rsi_length"]),
            float(row["oversold"]),
            float(row["overbought"]),
            int(row["max_bars_between_dips"]),
        )
        for _, row in results.iterrows()
    }

    expected = {
        (5, 25.0, 70.0, 3),
        (5, 25.0, 70.0, 7),
        (10, 25.0, 70.0, 3),
        (10, 25.0, 70.0, 7),
    }

    assert actual == expected


def test_candidate_evaluator_returns_given_parameters() -> None:
    data = make_market_data()

    parameters = DoubleDipParameters(
        rsi_length=5,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=5,
    )

    # For this particular test we can provide a manually
    # constructed RSI series because we're testing the
    # candidate evaluator rather than RSIIndicator itself.
    rsi_values = np.resize(
        [
            50.0,
            25.0,
            32.0,
            25.0,
            33.0,
            60.0,
            72.0,
            68.0,
            74.0,
            69.0,
        ],
        len(data),
    )

    rsi = pd.Series(
        rsi_values,
        index=data.index,
        dtype=float,
    )

    row = evaluate_double_dip_candidate_from_rsi(
        data=data,
        rsi=rsi,
        parameters=parameters,
        backtest_config=no_cost_config(),
    )

    assert row["rsi_length"] == 5
    assert row["oversold"] == pytest.approx(30.0)
    assert row["overbought"] == pytest.approx(70.0)
    assert row["max_bars_between_dips"] == 5


# ============================================================
# VALIDATION
# ============================================================


def test_optimizer_rejects_empty_market_data() -> None:
    data = pd.DataFrame(
        columns=["Open", "Close"],
        dtype=float,
    )

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5,),
        oversold_levels=(30.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    with pytest.raises(
        ValueError,
        match="data cannot be empty",
    ):
        optimize_double_dip(
            data=data,
            search_space=search_space,
        )


def test_optimizer_rejects_missing_open() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    data = pd.DataFrame(
        {
            "Close": np.arange(
                100.0,
                120.0,
            ),
        },
        index=index,
    )

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5,),
        oversold_levels=(30.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        optimize_double_dip(
            data=data,
            search_space=search_space,
        )


def test_optimizer_rejects_unsorted_data() -> None:
    data = make_market_data(
        periods=20,
    )

    data = data.sort_index(
        ascending=False
    )

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5,),
        oversold_levels=(30.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    with pytest.raises(
        ValueError,
        match="data index must be sorted",
    ):
        optimize_double_dip(
            data=data,
            search_space=search_space,
        )


def test_optimizer_rejects_nonpositive_periods_per_year() -> None:
    data = make_market_data()

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(5,),
        oversold_levels=(30.0,),
        overbought_levels=(70.0,),
        max_bars_between_dips=(5,),
    )

    with pytest.raises(
        ValueError,
        match="periods_per_year must be greater than zero",
    ):
        optimize_double_dip(
            data=data,
            search_space=search_space,
            periods_per_year=0,
        )