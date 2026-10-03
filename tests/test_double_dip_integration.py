# tests/test_double_dip_integration.py


import pandas as pd
import pytest

from strats.double_dip import rsi_double_dip_targets
from backtesting.engine import BacktestConfig, run_backtest
from backtesting.metrics import calculate_metrics


def test_double_dip_full_pipeline() -> None:
    """
    Integration test:

        RSI
         ↓
        Double Dip targets
         ↓
        Backtesting engine
         ↓
        Completed trade
         ↓
        Performance metrics

    Results can be calculated by hand.
    """

    index = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    # ---------------------------------------------------------
    # 1. Synthetic RSI
    # ---------------------------------------------------------
    #
    # Oversold threshold = 30
    # Overbought threshold = 70
    #
    # Bar    RSI       Meaning
    # ---------------------------------------------------------
    # 0      50        Neutral
    # 1      25        First oversold episode
    # 2      32        First oversold recovery
    # 3      25        Second oversold episode
    # 4      33        Second recovery -> target LONG
    # 5      60        Remain LONG
    # 6      72        First overbought episode
    # 7      68        First overbought recovery
    # 8      74        Second overbought episode
    # 9      69        Second recovery -> target FLAT
    #
    rsi = pd.Series(
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
        index=index,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 2. Synthetic market prices
    # ---------------------------------------------------------
    #
    # Double Dip becomes LONG at bar 4's Close.
    #
    # Because the engine executes Close-generated targets at
    # the following Open:
    #
    #     BUY at bar 5 Open = $100
    #
    # Double Dip becomes FLAT at bar 9's Close.
    #
    # That exit cannot execute at the next Open because the
    # dataset ends at bar 9.
    #
    # Therefore, with close_open_position_at_end=True,
    # the engine liquidates at bar 9 Close = $120.
    #
    data = pd.DataFrame(
        {
            "Open": [
                90.0,
                91.0,
                92.0,
                93.0,
                94.0,
                100.0,  # actual entry
                104.0,
                108.0,
                112.0,
                118.0,
            ],
            "Close": [
                90.0,
                91.0,
                92.0,
                93.0,
                94.0,
                102.0,
                106.0,
                110.0,
                115.0,
                120.0,  # forced final liquidation
            ],
        },
        index=index,
    )

    # ---------------------------------------------------------
    # 3. Generate strategy targets
    # ---------------------------------------------------------

    target_at_close = rsi_double_dip_targets(
        rsi=rsi,
        oversold=30.0,
        overbought=70.0,
        max_bars_between_dips=10,
    )

    expected_targets = pd.Series(
        [
            0,
            0,
            0,
            0,
            1,  # second oversold recovery
            1,
            1,
            1,
            1,
            0,  # second overbought rejection
        ],
        index=index,
        dtype="int8",
    )

    pd.testing.assert_series_equal(
        target_at_close,
        expected_targets,
    )

    # ---------------------------------------------------------
    # 4. Run backtest
    # ---------------------------------------------------------

    config = BacktestConfig(
        initial_cash=100_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        close_open_position_at_end=True,
    )

    result = run_backtest(
        data=data,
        target_at_close=target_at_close,
        config=config,
    )

    # ---------------------------------------------------------
    # 5. Verify executed positions
    # ---------------------------------------------------------
    #
    # Targets at Close:
    #
    # 0 0 0 0 1 1 1 1 1 0
    #
    # Positions at following Open:
    #
    # 0 0 0 0 0 1 1 1 1 1
    #
    # Then the final open position is forcibly liquidated,
    # so the engine reports position 0 on the final bar.
    #

    expected_positions = [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
    ]

    assert result.positions.tolist() == expected_positions

    # ---------------------------------------------------------
    # 6. Verify trade
    # ---------------------------------------------------------

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    # Double Dip completes at bar 4 Close.
    # Trade executes on bar 5 Open.
    assert trade["entry_time"] == index[5]
    assert trade["entry_price"] == pytest.approx(100.0)

    # The target becomes flat at the LAST Close.
    #
    # There is no bar 10 Open where that signal could execute,
    # so the engine's end-of-data liquidation closes at
    # bar 9 Close.
    assert trade["exit_time"] == index[9]
    assert trade["exit_price"] == pytest.approx(120.0)

    assert trade["exit_reason"] == "end_of_data"

    # ---------------------------------------------------------
    # 7. Verify P&L
    # ---------------------------------------------------------
    #
    # Starting cash = $100,000
    # Entry price   = $100
    #
    # Units:
    #
    #     100,000 / 100 = 1,000 shares
    #
    # Exit price:
    #
    #     $120
    #
    # Gross profit:
    #
    #     ($120 - $100) * 1,000
    #     = $20,000
    #

    assert trade["units"] == pytest.approx(1_000.0)

    assert trade["entry_notional"] == pytest.approx(
        100_000.0
    )

    assert trade["gross_pnl"] == pytest.approx(
        20_000.0
    )

    assert trade["fees"] == pytest.approx(
        0.0
    )

    assert trade["net_pnl"] == pytest.approx(
        20_000.0
    )

    assert trade["return_pct"] == pytest.approx(
        0.20
    )

    # Entire portfolio should finish at $120,000.
    assert result.equity.iloc[-1] == pytest.approx(
        120_000.0
    )

    # ---------------------------------------------------------
    # 8. Calculate metrics
    # ---------------------------------------------------------

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=252,
    )

    # ---------------------------------------------------------
    # 9. Verify core metrics
    # ---------------------------------------------------------

    assert metrics["total_return"] == pytest.approx(
        0.20
    )

    assert metrics["closed_trades"] == 1

    assert metrics["winning_trades"] == 1
    assert metrics["losing_trades"] == 0

    assert metrics["win_rate"] == pytest.approx(
        1.0
    )

    assert metrics["gross_profit"] == pytest.approx(
        20_000.0
    )

    assert metrics["gross_loss"] == pytest.approx(
        0.0
    )

    assert metrics["net_profit"] == pytest.approx(
        20_000.0
    )

    # One winner and no losing trades gives infinite
    # profit factor.
    assert metrics["profit_factor"] == float("inf")

    assert metrics["average_trade_return"] == pytest.approx(
        0.20
    )

def test_double_dip_executes_normal_exit_at_following_open() -> None:
    """
    Integration test for a normal target-driven exit.

    Verifies the complete pipeline:

        RSI
         ↓
        Double Dip targets
         ↓
        BUY at following Open
         ↓
        HOLD
         ↓
        Double Dip exit target
         ↓
        SELL at following Open
         ↓
        Trade record
         ↓
        Metrics

    Unlike the end-of-data integration test, the exit signal
    occurs before the final bar, so the strategy-generated FLAT
    target can execute normally at the next Open.
    """

    index = pd.date_range(
        "2025-01-01",
        periods=12,
        freq="D",
    )

    # ---------------------------------------------------------
    # 1. Synthetic RSI
    # ---------------------------------------------------------
    #
    # Oversold = 30
    # Overbought = 70
    #
    # Bar   RSI    Meaning
    # ---------------------------------------------------------
    # 0     50     Neutral
    # 1     25     First oversold episode
    # 2     32     First oversold recovery
    # 3     25     Second oversold episode
    # 4     33     Second recovery -> LONG target
    #
    # 5     60     Long
    # 6     72     First overbought episode
    # 7     68     First overbought recovery
    # 8     74     Second overbought episode
    # 9     69     Second recovery -> FLAT target
    #
    # 10    55     Flat target can now execute
    # 11    50     Remain flat
    #
    rsi = pd.Series(
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
            55.0,
            50.0,
        ],
        index=index,
        dtype=float,
    )

    # ----------------------------