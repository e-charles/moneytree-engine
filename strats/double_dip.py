from __future__ import annotations

import pandas as pd

# import numpy as np


# # double dip in SMA does not really make sense since 
# def sma_strategy(data, crossover, crossunder):
#     # Initialize variables
#     entry_price_long = np.nan
#     entry_price_short = np.nan
#     total_trades = 0
#     winning_trades = 0
#     total_profit = 0.0
#     total_loss = 0.0

#     i = 1
#     while i < len(data):
#         trade_profit = np.nan

#         if i + 2 >= len(data): # make sure i + 1 is within bounds 
#             break # no more pairs to check 
#         # checks [crossunder, no crossover, crossunder] for cross unders
#         if (crossunder.iloc[i] and not crossover.iloc[i + 1] and crossunder.iloc[i + 2] and np.isnan(entry_price_long) or \
#             crossunder.iloc[i] and crossunder.iloc[i + 1]  and np.isnan(entry_price_long)): # crosses under consecutivly 
#             entry_price_long = data['Close'].iloc[i + 1]
#             total_trades += 1
#             #i += 2 # skips an iteration 
#             continue
#         # checks [Low, None, Low]
#         elif (crossover.iloc[i] and not crossunder.iloc[i + 1] and crossover.iloc[i + 1] and not np.isnan(entry_price_long) or \
#         crossover.iloc[i] and crossover.iloc[i + 1]  and not np.isnan(entry_price_long)): # crosses above twice 
#             trade_profit = data['Close'].iloc[i + 1] - entry_price_long
            
#             if trade_profit > 0:
#                 total_profit += trade_profit
#                 winning_trades += 1
#             else:
#                 total_loss -= trade_profit

#             entry_price_long = np.nan # closes the active long position     
#             #i += 2 # skips an iteration 
#             continue

#         i += 1 # normal increment 
                
#     return total_trades, winning_trades, total_profit, total_loss

# def rsi_strategy(data, buy_signal, sell_signal):
#     # Initialize variables
#     entry_price_long = np.nan
#     total_trades = 0
#     winning_trades = 0
#     total_profit = 0.0
#     total_loss = 0.0

#     i = 0
#     while i < len(data) - 2:
#         # Buy on consecutive oversold signals
#         # signal + nuetral + signal 
#         if np.isnan(entry_price_long): 
#             if (buy_signal.iloc[i] and not sell_signal.iloc[i + 1] and buy_signal.iloc[i + 2]):
#                 entry_price_long = data['Close'].iloc[i+2]
#                 total_trades += 1
#                 i += 2
#                 continue
            
#             # signal + signal 
#             if (buy_signal.iloc[i] and buy_signal.iloc[i + 1]):
#                 entry_price_long = data['Close'].iloc[i + 1]
#                 total_trades += 1
#                 i += 1
#                 continue

#         # Sell on consecutive overbought signals
#         # signal + nuetral + signal 
#         if not np.isnan(entry_price_long):
#             if (sell_signal.iloc[i] and not buy_signal.iloc[i + 1] and sell_signal.iloc[i + 2]):
#                 trade_profit = data['Close'].iloc[i + 2] - entry_price_long
#                 if trade_profit > 0:
#                     total_profit += trade_profit
#                     winning_trades += 1
#                 else:
#                     total_loss -= trade_profit
#                 entry_price_long = np.nan
#                 i += 2
#                 continue

#             # signal + signal 
#             if (sell_signal.iloc[i] and sell_signal.iloc[i + 1]):
#                 trade_profit = data['Close'].iloc[i + 1] - entry_price_long
#                 if trade_profit > 0:
#                     total_profit += trade_profit
#                     winning_trades += 1
#                 else:
#                     total_loss -= trade_profit
#                 entry_price_long = np.nan
#                 i += 1
#                 continue
#         i += 1
                
#     return total_trades, winning_trades, total_profit, total_loss



# strats/double_dip.py

def rsi_double_dip_targets(
    rsi: pd.Series,
    oversold: float = 30.0,
    overbought: float = 70.0,
    max_bars_between_dips: int = 10
) -> pd.Series:
    """
    Long-only RSI double-dip strategy.

    Entry:
      1. RSI reaches or falls below `oversold` (first dip).
      2. RSI later recovers above `oversold`.
      3. RSI reaches or falls below `oversold` again within
         `max_bars_between_dips` bars of the first dip.
      4. RSI recovers above `oversold` a second time.
      5. Target becomes LONG on that completed bar.

    Exit:
      1. RSI reaches or rises above `overbought` (first peak).
      2. RSI later falls below `overbought`.
      3. RSI reaches or rises above `overbought` again.
      4. RSI falls below `overbought` a second time.
      5. Target becomes FLAT on that completed bar.

    The returned series is a target position decided at each bar's close.
    Shift it by one bar before backtesting if fills occur at the next open.

    Returns:
        A Series of 0 (flat) and 1 (long).
    """
    if rsi.empty:
        raise ValueError("rsi cannot be empty")

    if not 0 < oversold < overbought < 100:
        raise ValueError(
            "Thresholds must satisfy 0 < oversold < overbought < 100."
        )

    if max_bars_between_dips < 1:
        raise ValueError("max_bars_between_dips must be at least 1.")

    position = 0
    targets: list[int] = []

    # Entry-side state.
    first_oversold_bar: int | None = None
    first_oversold_recovered = False
    second_oversold_seen = False

    # Exit-side state.
    first_overbought_seen = False
    first_overbought_recovered = False
    second_overbought_seen = False

    previous_rsi: float | None = None

    for bar_number, value in enumerate(rsi):
        if pd.isna(value):
            targets.append(position)
            previous_rsi = None
            continue

        is_oversold = value <= oversold
        is_overbought = value >= overbought

        crossed_above_oversold = (
            previous_rsi is not None
            and previous_rsi <= oversold
            and value > oversold
        )

        crossed_below_overbought = (
            previous_rsi is not None
            and previous_rsi >= overbought
            and value < overbought
        )

        if position == 0:
            # Expire an old first-dip setup.
            if (
                first_oversold_bar is not None
                and bar_number - first_oversold_bar > max_bars_between_dips
            ):
                first_oversold_bar = None
                first_oversold_recovered = False
                second_oversold_seen = False

            # First oversold episode arms the setup.
            if first_oversold_bar is None and is_oversold:
                first_oversold_bar = bar_number

            # RSI must leave oversold after first dip before a second dip counts.
            elif (
                first_oversold_bar is not None
                and not first_oversold_recovered
                and crossed_above_oversold
            ):
                first_oversold_recovered = True

            # Second oversold episode occurs after first recovery.
            elif (
                first_oversold_bar is not None
                and first_oversold_recovered
                and is_oversold
            ):
                second_oversold_seen = True

            # Enter upon recovery from the second oversold episode.
            elif second_oversold_seen and crossed_above_oversold:
                position = 1

                # Reset entry state after entering.
                first_oversold_bar = None
                first_oversold_recovered = False
                second_oversold_seen = False

        else:
            # First overbought episode arms the exit setup.
            if not first_overbought_seen and is_overbought:
                first_overbought_seen = True

            # RSI must leave overbought before a second peak counts.
            elif (
                first_overbought_seen
                and not first_overbought_recovered
                and crossed_below_overbought
            ):
                first_overbought_recovered = True

            # Second overbought episode.
            elif (
                first_overbought_seen
                and first_overbought_recovered
                and is_overbought
            ):
                second_overbought_seen = True

            # Exit after second fall below overbought.
            elif second_overbought_seen and crossed_below_overbought:
                position = 0

                # Reset exit state after closing.
                first_overbought_seen = False
                first_overbought_recovered = False
                second_overbought_seen = False

        targets.append(position)
        previous_rsi = float(value)

    return pd.Series(targets, index=rsi.index, dtype="int8")