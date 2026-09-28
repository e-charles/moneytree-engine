import numpy as np

def sma_strategy(data, crossover, crossunder):
    # Initialize variables
    entry_price_long = np.nan
    entry_price_short = np.nan
    total_trades = 0
    winning_trades = 0
    total_profit = 0.0
    total_loss = 0.0

    i = 1
    while i < len(data):
        trade_profit = np.nan

        # Open long position
        if crossover.iloc[i] and np.isnan(entry_price_long): # opens long, no existing position 
            entry_price_long = data['Close'].iloc[i]
            total_trades += 1
        # Close long position
        elif crossunder.iloc[i] and not np.isnan(entry_price_long): # closes long 
            trade_profit = data['Close'].iloc[i] - entry_price_long
            if trade_profit > 0:
                total_profit += trade_profit
                winning_trades += 1
            else:
                total_loss -= trade_profit
            entry_price_long = np.nan # closes the active long position 
        i += 1
        
    return total_trades, winning_trades, total_profit, total_loss


def rsi_strategy(data, buy_signal, sell_signal):
    # Initialize variables
    entry_price_long = np.nan
    entry_price_short = np.nan
    total_trades = 0
    winning_trades = 0
    total_profit = 0.0
    total_loss = 0.0

    i = 1
    while i < len(data):
        trade_profit = np.nan
        # Open long position
        if buy_signal.iloc[i] and np.isnan(entry_price_long):
            entry_price_long = data['Close'].iloc[i]
            total_trades += 1
        # Close long position
        elif sell_signal.iloc[i] and not np.isnan(entry_price_long):
            trade_profit = data['Close'].iloc[i] - entry_price_long
            if trade_profit > 0:
                total_profit += trade_profit
                winning_trades += 1
            else:
                total_loss -= trade_profit
            entry_price_long = np.nan
        i += 1
        
    return total_trades, winning_trades, total_profit, total_loss