import pandas as pd
# import sys
# from pathlib import Path
import pytest

# Adds the parent directory (/workspaces/moneytree-optimizer) to the path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

from backtesting.engine import BacktestConfig, run_backtest


def make_data(
        opens: list[float],
        closes: list[float],
        index: pd.Index | None = None 
) -> pd.DataFrame: 
    """Minimal market-data frame sufficient for backtesting"""
    if index is None: 
            index = pd.date_range("2025-01-01", periods=len(opens), freq="D")

    return pd.DataFrame(
         {
              "Open": opens,
              "Close": closes
         }, 
         index=index
    )

def make_target(
        data: pd.DataFrame,
        values: list[int | float]
) -> pd.Series: 
     """Create a close-based target series aligned with market-data"""
     return pd.Series(values, index=data.index, name="target_at_close")


def no_cost_config(
    intial_cash: float = 100_000.0,
    close_open_position_at_end: bool = False, 
) -> BacktestConfig:
    return BacktestConfig(
         initial_cash=intial_cash,
         commission_rate=0,
         slippage_rate=0,
         close_open_position_at_end=close_open_position_at_end
    )

     
# Execution timing
def test_close_signal_executes_at_following_open() -> None:
   data = make_data(
        opens= [100.0, 110.0, 120.0],
        closes= [100.0, 115.0, 125.0]
   )

   # strategy decides long at bar 0's close 
   # engine shifts internally and buys at bar 1's open = 110

   target_at_close = make_target(data, [1, 1, 1])

   result = run_backtest(data, 
                         target_at_close, 
                         no_cost_config())

   assert result.positions.tolist() == [0, 1, 1]
   assert result.trades.empty 

   # no order can execute on the first bar since no prior signal exists 
   assert result.equity.iloc[0] == pytest.approx(100_000.0)

   # buys at 110, then mark to the same bar's close of 115 
   # units = 100,000 / 110
   expected_equity_bar_1 = 100_000.0 * (115.0/110.0)
   assert result.equity.iloc[1] == pytest.approx(expected_equity_bar_1)

def test_last_bar_close_signal_cannot_execute() -> None:
     data = make_data(
          opens= [100.0, 100.0, 100.0],
          closes= [100.0, 100.0, 100.0]
     )

     # turns long only at the last close 
     # there is no following open bar so no trade should execute 
     target_at_close = make_target(data, [0, 0, 1])

     result = run_backtest(
          data, 
          target_at_close,
          no_cost_config()
     ) 

     assert result.positions.tolist() == [0, 0, 0]
     assert result.trades.empty 
     assert result.equity.iloc[-1] == pytest.approx(100_000.0)

def test_holding_target_does_not_create_multiple_entries() -> None: 
    data = make_data(
            opens=[100.0, 100.0, 100.0, 100.0, 100.0],
            closes=[100.0, 100.0, 100.0, 100.0, 100.0],
        )

    # Signals are evaluated at Close:
    # bar 0 says long -> buy at bar 1 open
    # bars 1 and 2 remain long -> hold
    # bar 3 says flat -> sell at bar 4 open
    target_at_close = make_target(data, [1, 1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(),
    )


    assert result.positions.tolist() == [0, 1, 1, 1, 0]
    assert len(result.trades) == 1

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == data.index[1]
    assert trade["exit_time"] == data.index[4]
    assert trade["direction"] == "long"



# Accounting 
def test_initial_equity_does_not_double_after_pruchase() -> None: 
    data = make_data(
        opens=[100.0, 100.0],
        closes=[100.0, 100.0],
    )

    # Signal at close of first bar; enter at second bar open.
    target_at_close = make_target(data, [1, 1])

    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [0, 1]

    # At entry, all cash is correctly converted into shares.
    # Because Open == Close and fees are zero, equity remains unchanged.
    assert result.equity.iloc[1] == pytest.approx(100_000.0)




def test_profitable_trade_increases_final_equity_correctly() -> None: 
    data = make_data(
        opens=[100.0, 100.0, 120.0, 120.0],
        closes=[100.0, 100.0, 120.0, 120.0],
    )

    # Signal long at bar 0 close -> enter at bar 1 open = 100.
    # Signal flat at bar 2 close -> exit at bar 3 open = 120.
    target_at_close = make_target(data, [1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(),
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    # $100,000 buys 1,000 units at 100.
    # Selling 1,000 units at 120 returns $120,000.
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(120.0)
    assert trade["units"] == pytest.approx(1_000.0)
    assert trade["gross_pnl"] == pytest.approx(20_000.0)
    assert trade["net_pnl"] == pytest.approx(20_000.0)
    assert result.equity.iloc[-1] == pytest.approx(120_000.0)




def test_losing_trade_decreases_final_equity_correctly() -> None: 
    data = make_data(
        opens=[100.0, 100.0, 80.0, 80.0],
        closes=[100.0, 100.0, 80.0, 80.0],
    )

    # Enter at bar 1 Open = 100 and exit at bar 3 Open = 80.
    target_at_close = make_target(data, [1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(),
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["gross_pnl"] == pytest.approx(-20_000.0)
    assert trade["net_pnl"] == pytest.approx(-20_000.0)
    assert result.equity.iloc[-1] == pytest.approx(80_000.0)


def test_selling_restores_postion_to_flat() -> None: 
    data = make_data(
        opens=[100.0, 100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0, 100.0],
    )

    # Long signal at bar 0 -> buy at bar 1 Open.
    # Flat signal at bar 2 -> sell at bar 3 Open.
    target_at_close = make_target(data, [1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(),
    )

    assert result.positions.tolist() == [0, 1, 1, 0]
    assert result.positions.iloc[-1] == 0
    assert result.equity.iloc[-1] == pytest.approx(100_000.0)


# Execution costs 
def test_buy_slippage_worsens_entry_price() -> None: 
    data = make_data(
        opens=[100.0, 100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0, 100.0],
    )

    # Enter at bar 1; exit at bar 3.
    target_at_close = make_target(data, [1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.0,
            slippage_rate=0.01,
            close_open_position_at_end=False,
        ),
    )

    trade = result.trades.iloc[0]

    # Buy fill = quoted Open * (1 + slippage).
    assert trade["entry_price"] == pytest.approx(101.0)

    # Without slippage, it would have entered at 100.
    assert trade["entry_price"] > 100.0


def test_sell_slippage_worsens_exit_price() -> None: 
    data = make_data(
        opens=[100.0, 100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0, 100.0],
    )
    target_at_close = make_target(data, [1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.0,
            slippage_rate=0.01,
            close_open_position_at_end=False,
        ),
    )

    trade = result.trades.iloc[0]

    # Sell fill = quoted Open * (1 - slippage).
    assert trade["exit_price"] == pytest.approx(99.0)
    assert trade["exit_price"] < 100.0


def test_entry_commission_reduces_equity() -> None: 
    data = make_data(
        opens=[100.0, 100.0],
        closes=[100.0, 100.0],
    )
    target_at_close = make_target(data, [1, 1])

    result = run_backtest(
        data,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.001,
            slippage_rate=0.0,
            close_open_position_at_end=False,
        ),
    )

    # Entry notional is sized to ensure:
    #
    # entry_notional + entry_fee = $100,000
    #
    # entry_notional = 100,000 / 1.001
    # entry fee = entry_notional * 0.001
    entry_notional = 100_000.0 / 1.001
    expected_equity = entry_notional

    assert result.positions.iloc[1] == 1
    assert result.equity.iloc[1] == pytest.approx(expected_equity)

    # This is approximately a $99.90 reduction, not exactly $100,
    # because the engine sizes the purchase to include commission.
    assert result.equity.iloc[1] < 100_000.0


def test_exit_commission_reduces_equity() -> None: 
    data = make_data(
        opens=[100.0, 100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0, 100.0],
    )
    target_at_close = make_target(data, [1, 1, 0, 0])

    result = run_backtest(
        data,
        target_at_close,
        BacktestConfig(
            initial_cash=100_000.0,
            commission_rate=0.001,
            slippage_rate=0.0,
            close_open_position_at_end=False,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
   

    expected_entry_notional = 100_000.0 / 1.001
    expected_entry_fee = expected_entry_notional * 0.001
    expected_exit_fee = expected_entry_notional * 0.001

    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(100.0)
    assert trade["gross_pnl"] == pytest.approx(0.0)
    assert trade["fees"] == pytest.approx(expected_entry_fee + expected_exit_fee)
    assert trade["net_pnl"] == pytest.approx(-(expected_entry_fee + expected_exit_fee))

    expected_final_equity = (expected_entry_notional - expected_exit_fee)

    assert result.positions.tolist() ==[0, 1, 1, 0]
    assert result.equity.iloc[-1] == pytest.approx(expected_final_equity)

    # entry and exit loses money 
    assert result.equity.iloc[-1] < result.equity.iloc[1]
    assert result.equity.iloc[-1] < 100_000.0



# # Lifecycle 
def test_target_exit_records_completed_trade() -> None:
    data = make_data(
        opens=[100.0, 110.0, 120.0, 130.0],
        closes=[100.0, 115.0, 125.0, 135.0],
    )

    target_at_close = make_target(data, [1, 1, 0, 0])

    # Enter at bar 1; exit at bar 3.
    result = run_backtest(
        data,
        target_at_close,
        no_cost_config()
    )

    assert result.positions.tolist() == [0, 1, 1, 0]
    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["entry_time"] == data.index[1]
    assert trade["exit_time"] == data.index[3]
    assert trade["entry_price"] == pytest.approx(110.0)
    assert trade["exit_price"] == pytest.approx(130.0)
    assert trade["direction"] == "long"
    assert trade["exit_reason"] == "target"

    units = 100_000.0 / 110.0
    expected_pnl = ((130.0 - 110.0) * units)

    assert trade["gross_pnl"] == pytest.approx(expected_pnl)
    assert trade["net_pnl"] == pytest.approx(expected_pnl)
    assert result.equity.iloc[-1] == pytest.approx(100_000.0 + expected_pnl)



def test_forced_final_liquidation_records_completed_trade() -> None:
    data = make_data(
            opens=[100.0, 110.0, 120.0],
            closes=[100.0, 115.0, 125.0],
    )
    
    target_at_close = make_target(data, [1, 1, 1])

    # Enter at bar 1; exit at bar 3.
    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(close_open_position_at_end=True)
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    # entry is at open bar 1 
    assert trade["entry_time"] == data.index[1]
    assert trade["exit_price"] == pytest.approx(125.0)

    units = 100_000.0 / 110.0
    expected_pnl = ( (125.0 - 110.0) * units)

    assert trade["net_pnl"] == pytest.approx(expected_pnl)
    assert result.equity.iloc[-1] == pytest.approx( 100_000.0 + expected_pnl)

    assert result.positions.iloc[-1] == 0 


def test_forced_liquidation_has_exit_reason_equal_to_end_of_data() -> None:
    data = make_data(
            opens=[100.0, 110.0, 120.0],
            closes=[100.0, 115.0, 125.0],
    )
    
    target_at_close = make_target(data, [1, 1, 1])

    # Enter at bar 1; exit at bar 3.
    result = run_backtest(
        data,
        target_at_close,
        no_cost_config(close_open_position_at_end=True)
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "end_of_data"


# Validation 
def test_empty_data_rejected() -> None:
    data = pd.DataFrame(
        columns=["Open", "Close"],
        dtype=float
    )

    target_at_close = pd.Series([], index= data.index, dtype="int8")

    with pytest.raises(
        ValueError, 
        match="data cannot be empty"
    ): run_backtest(data, target_at_close)

def test_missing_open_rejected() -> None: 
    index = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="D"
    )

    data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0]
        },
        index=index,
    )

    target_at_close = pd.Series(
        [0, 0, 0],
        index=index, 
        dtype="int8"
    )

    with pytest.raises(
        ValueError, 
        match="Missing required columns"
    ): run_backtest(
        data, 
        target_at_close
    )

def test_missing_close_rejected() -> None: 
    data = make_data(
        opens=[100.0, 0.0, 100.0],
        closes=[100.0, 100.0, 100.0],
    )
    
    target_at_close = make_target(data, [0, 0, 0])

    with pytest.raises(
        ValueError, 
        match= "Open and Close prices must be greater than zero",
    ): run_backtest(
        data,
        target_at_close,
    )

def test_negative_or_zero_prices_rejected() -> None:
    data = make_data(
        opens=[100.0, -5.0, 100.0],
        closes=[100.0, 100.0, 100.0],
    )
    
    target_at_close = make_target(data, [0, 0, 0])

    with pytest.raises(
        ValueError, 
        match= "Open and Close prices must be greater than zero",
    ): run_backtest(
        data,
        target_at_close,
    )

def test_nan_prices_rejected() -> None:
    data = make_data(
        opens=[100.0, float("nan"), 100.0],
        closes=[100.0, 105.0, 115.0],
    )
    
    target_at_close = make_target(data, [0, 0, 0])

    with pytest.raises(
        ValueError, 
        match= "Open and Close prices cannot contain NaN values",
    ): run_backtest(
        data,
        target_at_close,
    )

def test_invalid_target_rejected() -> None:
    data = make_data(
        opens=[100.0, 101.0, 102.0],
        closes=[100.0, 101.0, 102.0],
    )
    
    target_at_close = make_target(data, [0, 2, 0])

    with pytest.raises(
        ValueError, 
        match= "target_at_close may contain only 0 or 1",
    ): run_backtest(
        data,
        target_at_close,
    )

def test_target_or_data_index_mismatch_rejected() -> None:
    data = make_data(
        opens=[100.0, 101.0, 102.0],
        closes=[100.0, 101.0, 102.0],
    )

    wrong_index = pd.date_range(
        "2030-01-01",
        periods=3,
        freq="D"
    )
    
    target_at_close = pd.Series([0, 1, 1], index=wrong_index, dtype="int8")

    with pytest.raises(
        ValueError, 
        match= "target_at_close index must exactly match data index",
    ): run_backtest(
        data,
        target_at_close,
    )
