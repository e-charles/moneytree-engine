import numpy as np
import pytest

from backtesting.engine import (
    BacktestConfig,
    run_backtest,
)


# ============================================================
# Helpers
# ============================================================


def make_prices(
    opens: list[float],
    closes: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create NumPy Open and Close arrays for engine tests.
    """
    return (
        np.asarray(
            opens,
            dtype=np.float64,
        ),
        np.asarray(
            closes,
            dtype=np.float64,
        ),
    )


def make_target(
    values: list[int],
) -> np.ndarray:
    """
    Create a close-based target array.
    """
    return np.asarray(
        values,
        dtype=np.int8,
    )


def no_cost_config(
    initial_cash: float = 100_000.0,
    close_open_position_at_end: bool = False,
) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=initial_cash,
        commission_rate=0.0,
        slippage_rate=0.0,
        close_open_position_at_end=(
            close_open_position_at_end
        ),
    )


# ============================================================
# EXECUTION TIMING
# ============================================================


def test_close_signal_executes_at_following_open() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 110.0, 120.0],
        closes=[100.0, 115.0, 125.0],
    )

    # Strategy decides long at bar 0 Close.
    #
    # Engine shifts the target internally:
    #
    # target[0] -> position_at_open[1]
    #
    # Therefore the purchase occurs at bar 1 Open = 110.
    target_at_close = make_target(
        [1, 1, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [
        0,
        1,
        1,
    ]

    # Position remains open because forced final liquidation
    # is disabled.
    assert result.trades.count == 0

    # No order can execute on the first bar because no prior
    # close-based target exists.
    assert result.equity[0] == pytest.approx(
        100_000.0
    )

    # Buy at 110 and mark the position at the same bar's
    # Close of 115.
    expected_equity_bar_1 = (
        100_000.0
        * (115.0 / 110.0)
    )

    assert result.equity[1] == pytest.approx(
        expected_equity_bar_1
    )


def test_first_bar_cannot_execute_prior_signal() -> None:
    """
    The engine receives an independent backtest window.

    There is no target before target_at_close[0], so the first
    bar must always begin flat.

    This behavior is important at walk-forward test boundaries.
    """
    open_prices, close_prices = make_prices(
        opens=[100.0, 100.0],
        closes=[100.0, 100.0],
    )

    target_at_close = make_target(
        [1, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions[0] == 0
    assert result.positions[1] == 1

    assert result.equity[0] == pytest.approx(
        100_000.0
    )


def test_last_bar_close_signal_cannot_execute() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0],
    )

    # Turns long only at the final Close.
    #
    # There is no following Open, so no trade can execute.
    target_at_close = make_target(
        [0, 0, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [
        0,
        0,
        0,
    ]

    assert result.trades.count == 0

    assert result.equity[-1] == pytest.approx(
        100_000.0
    )


def test_holding_target_does_not_create_multiple_entries() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    # bar 0 Close -> long
    # bar 1 Open  -> enter
    #
    # bars 1 and 2 continue targeting long.
    #
    # bar 3 Close -> flat
    # bar 4 Open  -> exit
    target_at_close = make_target(
        [1, 1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [
        0,
        1,
        1,
        1,
        0,
    ]

    assert result.trades.count == 1

    assert result.trades.entry_bars[0] == 1
    assert result.trades.exit_bars[0] == 4


# ============================================================
# ACCOUNTING
# ============================================================


def test_initial_equity_does_not_double_after_purchase() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 100.0],
        closes=[100.0, 100.0],
    )

    target_at_close = make_target(
        [1, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [
        0,
        1,
    ]

    # All cash is converted into shares.
    #
    # Since Open == Close and there are no costs, equity must
    # remain exactly equal to initial capital.
    assert result.equity[1] == pytest.approx(
        100_000.0
    )


def test_profitable_trade_increases_final_equity_correctly() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            120.0,
            120.0,
        ],
        closes=[
            100.0,
            100.0,
            120.0,
            120.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.trades.count == 1

    assert result.trades.entry_prices[0] == pytest.approx(
        100.0
    )

    assert result.trades.exit_prices[0] == pytest.approx(
        120.0
    )

    assert result.trades.units[0] == pytest.approx(
        1_000.0
    )

    assert result.trades.gross_pnl[0] == pytest.approx(
        20_000.0
    )

    assert result.trades.net_pnl[0] == pytest.approx(
        20_000.0
    )

    assert result.equity[-1] == pytest.approx(
        120_000.0
    )

    assert result.final_equity == pytest.approx(
        120_000.0
    )


def test_losing_trade_decreases_final_equity_correctly() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            80.0,
            80.0,
        ],
        closes=[
            100.0,
            100.0,
            80.0,
            80.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.trades.count == 1

    assert result.trades.gross_pnl[0] == pytest.approx(
        -20_000.0
    )

    assert result.trades.net_pnl[0] == pytest.approx(
        -20_000.0
    )

    assert result.equity[-1] == pytest.approx(
        80_000.0
    )

    assert result.final_equity == pytest.approx(
        80_000.0
    )


def test_selling_restores_position_to_flat() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [
        0,
        1,
        1,
        0,
    ]

    assert result.positions[-1] == 0

    assert result.equity[-1] == pytest.approx(
        100_000.0
    )


# ============================================================
# EXECUTION COSTS
# ============================================================


def test_buy_slippage_worsens_entry_price() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.0,
            slippage_rate=0.01,
            close_open_position_at_end=False,
        ),
    )

    assert result.trades.count == 1

    # Quoted Open = 100.
    # 1% adverse buy slippage -> 101.
    assert result.trades.entry_prices[0] == pytest.approx(
        101.0
    )

    assert result.trades.entry_prices[0] > 100.0


def test_sell_slippage_worsens_exit_price() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.0,
            slippage_rate=0.01,
            close_open_position_at_end=False,
        ),
    )

    assert result.trades.count == 1

    # Quoted Open = 100.
    # 1% adverse sell slippage -> 99.
    assert result.trades.exit_prices[0] == pytest.approx(
        99.0
    )

    assert result.trades.exit_prices[0] < 100.0


def test_entry_commission_reduces_equity() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 100.0],
        closes=[100.0, 100.0],
    )

    target_at_close = make_target(
        [1, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.001,
            slippage_rate=0.0,
            close_open_position_at_end=False,
        ),
    )

    entry_notional = (
        100_000.0 / 1.001
    )

    expected_equity = (
        entry_notional
    )

    assert result.positions[1] == 1

    assert result.equity[1] == pytest.approx(
        expected_equity
    )

    assert result.equity[1] < 100_000.0


def test_exit_commission_reduces_equity() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.001,
            slippage_rate=0.0,
            close_open_position_at_end=False,
        ),
    )

    assert result.trades.count == 1

    expected_entry_notional = (
        100_000.0 / 1.001
    )

    expected_entry_fee = (
        expected_entry_notional
        * 0.001
    )

    expected_exit_fee = (
        expected_entry_notional
        * 0.001
    )

    expected_total_fees = (
        expected_entry_fee
        + expected_exit_fee
    )

    expected_final_equity = (
        expected_entry_notional
        - expected_exit_fee
    )

    assert result.trades.entry_prices[0] == pytest.approx(
        100.0
    )

    assert result.trades.exit_prices[0] == pytest.approx(
        100.0
    )

    assert result.trades.gross_pnl[0] == pytest.approx(
        0.0
    )

    assert result.trades.fees[0] == pytest.approx(
        expected_total_fees
    )

    assert result.trades.net_pnl[0] == pytest.approx(
        -expected_total_fees
    )

    assert result.positions.tolist() == [
        0,
        1,
        1,
        0,
    ]

    assert result.equity[-1] == pytest.approx(
        expected_final_equity
    )

    assert result.equity[-1] < result.equity[1]
    assert result.equity[-1] < 100_000.0


# ============================================================
# TRADE LIFECYCLE
# ============================================================


def test_target_exit_records_completed_trade() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            110.0,
            120.0,
            130.0,
        ],
        closes=[
            100.0,
            115.0,
            125.0,
            135.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [
        0,
        1,
        1,
        0,
    ]

    assert result.trades.count == 1

    # The numerical engine records bar positions rather than
    # timestamps. Timestamp translation belongs to reporting.
    assert result.trades.entry_bars[0] == 1
    assert result.trades.exit_bars[0] == 3

    assert result.trades.entry_prices[0] == pytest.approx(
        110.0
    )

    assert result.trades.exit_prices[0] == pytest.approx(
        130.0
    )

    # 0 = strategy target exit.
    assert result.trades.exit_reasons[0] == 0

    units = (
        100_000.0 / 110.0
    )

    expected_pnl = (
        (130.0 - 110.0)
        * units
    )

    assert result.trades.gross_pnl[0] == pytest.approx(
        expected_pnl
    )

    assert result.trades.net_pnl[0] == pytest.approx(
        expected_pnl
    )

    assert result.trades.bars_held[0] == 2

    assert result.equity[-1] == pytest.approx(
        100_000.0 + expected_pnl
    )


def test_forced_final_liquidation_records_completed_trade() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            110.0,
            120.0,
        ],
        closes=[
            100.0,
            115.0,
            125.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(
            close_open_position_at_end=True
        ),
    )

    assert result.trades.count == 1

    # Entry executes at bar 1 Open.
    assert result.trades.entry_bars[0] == 1
    assert result.trades.entry_prices[0] == pytest.approx(
        110.0
    )

    # Forced liquidation occurs at the final bar Close.
    assert result.trades.exit_bars[0] == 2
    assert result.trades.exit_prices[0] == pytest.approx(
        125.0
    )

    units = (
        100_000.0 / 110.0
    )

    expected_pnl = (
        (125.0 - 110.0)
        * units
    )

    assert result.trades.net_pnl[0] == pytest.approx(
        expected_pnl
    )

    assert result.equity[-1] == pytest.approx(
        100_000.0 + expected_pnl
    )

    assert result.final_equity == pytest.approx(
        result.equity[-1]
    )

    assert result.positions[-1] == 0


def test_forced_liquidation_has_end_of_data_exit_reason() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            110.0,
            120.0,
        ],
        closes=[
            100.0,
            115.0,
            125.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 1]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(
            close_open_position_at_end=True
        ),
    )

    assert result.trades.count == 1

    # 1 = forced end-of-data liquidation.
    assert result.trades.exit_reasons[0] == 1


def test_trade_bars_held_are_based_on_execution_bars() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    target_at_close = make_target(
        [1, 1, 1, 0, 0]
    )

    result = run_backtest(
        open_prices,
        close_prices,
        target_at_close,
        no_cost_config(),
    )

    assert result.trades.count == 1

    # Entry at bar 1 and exit at bar 4.
    assert result.trades.entry_bars[0] == 1
    assert result.trades.exit_bars[0] == 4
    assert result.trades.bars_held[0] == 3


# ============================================================
# VALIDATION
# ============================================================


def test_empty_price_arrays_rejected() -> None:
    open_prices = np.asarray(
        [],
        dtype=np.float64,
    )

    close_prices = np.asarray(
        [],
        dtype=np.float64,
    )

    target_at_close = np.asarray(
        [],
        dtype=np.int8,
    )

    with pytest.raises(
        ValueError,
        match="price arrays cannot be empty",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_open_prices_must_be_numpy_array() -> None:
    close_prices = np.asarray(
        [100.0, 101.0],
        dtype=np.float64,
    )

    target_at_close = make_target(
        [0, 0]
    )

    with pytest.raises(
        TypeError,
        match="open_prices must be a NumPy ndarray",
    ):
        run_backtest(
            [100.0, 101.0],  # type: ignore[arg-type]
            close_prices,
            target_at_close,
        )


def test_close_prices_must_be_numpy_array() -> None:
    open_prices = np.asarray(
        [100.0, 101.0],
        dtype=np.float64,
    )

    target_at_close = make_target(
        [0, 0]
    )

    with pytest.raises(
        TypeError,
        match="close_prices must be a NumPy ndarray",
    ):
        run_backtest(
            open_prices,
            [100.0, 101.0],  # type: ignore[arg-type]
            target_at_close,
        )


def test_target_must_be_numpy_array() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 101.0],
        closes=[100.0, 101.0],
    )

    with pytest.raises(
        TypeError,
        match="target_at_close must be a NumPy ndarray",
    ):
        run_backtest(
            open_prices,
            close_prices,
            [0, 0],  # type: ignore[arg-type]
        )


def test_price_array_lengths_must_match() -> None:
    open_prices = np.asarray(
        [100.0, 101.0, 102.0],
        dtype=np.float64,
    )

    close_prices = np.asarray(
        [100.0, 101.0],
        dtype=np.float64,
    )

    target_at_close = make_target(
        [0, 0, 0]
    )

    with pytest.raises(
        ValueError,
        match=(
            "open_prices and close_prices must "
            "have equal length"
        ),
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_target_length_must_match_price_arrays() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 101.0, 102.0],
        closes=[100.0, 101.0, 102.0],
    )

    target_at_close = make_target(
        [0, 0]
    )

    with pytest.raises(
        ValueError,
        match=(
            "target_at_close must have the same "
            "length as the price arrays"
        ),
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_nonpositive_open_price_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 0.0, 100.0],
        closes=[100.0, 100.0, 100.0],
    )

    target_at_close = make_target(
        [0, 0, 0]
    )

    with pytest.raises(
        ValueError,
        match="open_prices must be greater than zero",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_nonpositive_close_price_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 100.0, 100.0],
        closes=[100.0, -5.0, 100.0],
    )

    target_at_close = make_target(
        [0, 0, 0]
    )

    with pytest.raises(
        ValueError,
        match="close_prices must be greater than zero",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_nan_open_price_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            float("nan"),
            100.0,
        ],
        closes=[
            100.0,
            105.0,
            115.0,
        ],
    )

    target_at_close = make_target(
        [0, 0, 0]
    )

    with pytest.raises(
        ValueError,
        match=(
            "open_prices must contain only "
            "finite values"
        ),
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_infinite_close_price_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[
            100.0,
            100.0,
            100.0,
        ],
        closes=[
            100.0,
            float("inf"),
            100.0,
        ],
    )

    target_at_close = make_target(
        [0, 0, 0]
    )

    with pytest.raises(
        ValueError,
        match=(
            "close_prices must contain only "
            "finite values"
        ),
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_invalid_target_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 101.0, 102.0],
        closes=[100.0, 101.0, 102.0],
    )

    target_at_close = make_target(
        [0, 2, 0]
    )

    with pytest.raises(
        ValueError,
        match="target_at_close may contain only 0 or 1",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
        )


def test_multidimensional_open_prices_rejected() -> None:
    open_prices = np.asarray(
        [
            [100.0, 101.0],
        ],
        dtype=np.float64,
    )

    close_prices = np.asarray(
        [100.0, 101.0],
        dtype=np.float64,
    )

    target_at_close = make_target(
        [0, 0]
    )

    with pytest.raises(
        ValueError,
        match="open_prices must be one-dimensional",
    ):
        run_backtest(
            open_prices,  # type: ignore[arg-type]
            close_prices,
            target_at_close,
        )


def test_multidimensional_target_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0, 101.0],
        closes=[100.0, 101.0],
    )

    target_at_close = np.asarray(
        [
            [0, 1],
        ],
        dtype=np.int8,
    )

    with pytest.raises(
        ValueError,
        match="target_at_close must be one-dimensional",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,  # type: ignore[arg-type]
        )


# ============================================================
# CONFIG VALIDATION
# ============================================================


def test_nonpositive_initial_cash_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0],
        closes=[100.0],
    )

    target_at_close = make_target(
        [0]
    )

    with pytest.raises(
        ValueError,
        match="initial_cash must be greater than zero",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
            BacktestConfig(
                initial_cash=0.0,
            ),
        )


def test_negative_commission_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0],
        closes=[100.0],
    )

    target_at_close = make_target(
        [0]
    )

    with pytest.raises(
        ValueError,
        match="commission_rate cannot be negative",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
            BacktestConfig(
                commission_rate=-0.001,
            ),
        )


def test_negative_slippage_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0],
        closes=[100.0],
    )

    target_at_close = make_target(
        [0]
    )

    with pytest.raises(
        ValueError,
        match="slippage_rate cannot be negative",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
            BacktestConfig(
                slippage_rate=-0.001,
            ),
        )


def test_slippage_of_one_or_more_rejected() -> None:
    open_prices, close_prices = make_prices(
        opens=[100.0],
        closes=[100.0],
    )

    target_at_close = make_target(
        [0]
    )

    with pytest.raises(
        ValueError,
        match="slippage_rate must be less than 1",
    ):
        run_backtest(
            open_prices,
            close_prices,
            target_at_close,
            BacktestConfig(
                slippage_rate=1.0,
            ),
        )