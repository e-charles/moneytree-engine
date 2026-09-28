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
        if crossover.iloc[i] and np.isnan(entry_price_long) and np.isnan(entry_price_short): # opens long, no existing position
            entry_price_long = data['Close'].iloc[i]
            total_trades += 1
        # Close long and open short
        elif crossunder.iloc[i] and not np.isnan(entry_price_long): # closes long and open short position (there was an active long position)
            trade_profit = data['Close'].iloc[i] - entry_price_long
            if trade_profit > 0: 
                total_profit += trade_profit
                winning_trades += 1
            else:
                total_loss -= trade_profit
            entry_price_long = np.nan # closes the active long
            entry_price_short = data['Close'].iloc[i] # opens a short position 
            total_trades += 1
        # Open short position
        elif crossunder.iloc[i] and np.isnan(entry_price_short) and np.isnan(entry_price_long): # opens short, no existing position 
            entry_price_short = data['Close'].iloc[i]
            total_trades += 1
        # Close short and open long
        elif crossover.iloc[i] and not np.isnan(entry_price_short): # closes short and opens long (there was an active short position)
            trade_profit = entry_price_short - data['Close'].iloc[i]
            if trade_profit > 0:
                total_profit += trade_profit
                winning_trades += 1
            else:
                total_loss -= trade_profit
            entry_price_short = np.nan # closes the active short 
            entry_price_long = data['Close'].iloc[i] # opens a long position 
            total_trades += 1
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
        if buy_signal.iloc[i] and np.isnan(entry_price_long) and np.isnan(entry_price_short): # buy signal + no current long or short entry 
            entry_price_long = data['Close'].iloc[i]
            total_trades += 1
        # Close long and open short
        elif sell_signal.iloc[i] and not np.isnan(entry_price_long): # sell signal + current long entry 
            trade_profit = data['Close'].iloc[i] - entry_price_long
            if trade_profit > 0:
                total_profit += trade_profit
                winning_trades += 1
            else:
                total_loss -= trade_profit
            entry_price_long = np.nan
            entry_price_short = data['Close'].iloc[i]
            total_trades += 1
        # Open short position
        elif sell_signal.iloc[i] and np.isnan(entry_price_short) and np.isnan(entry_price_long): # sell signal + no current long or short entry 
            entry_price_short = data['Close'].iloc[i]
            total_trades += 1
        # Close short and open long
        elif buy_signal.iloc[i] and not np.isnan(entry_price_short): # buy signal + current short entry 
            trade_profit = entry_price_short - data['Close'].iloc[i]
            if trade_profit > 0:
                total_profit += trade_profit
                winning_trades += 1
            else:
                total_loss -= trade_profit
            entry_price_short = np.nan
            entry_price_long = data['Close'].iloc[i]
            total_trades += 1
        i += 1
    return total_trades, winning_trades, total_profit, total_loss
