# test for the optimal RSI of a stock 

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import yfinance as yf

import sys 
import os 
# Get the directory of the current script (test/sma.py)
current_dir = os.path.dirname(__file__)

# Add the parent directory (my_project/) to sys.path
# This makes 'sma_strats' discoverable as a top-level package
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


# sys.path.insert(0, "../strats")
from strats import double_dip, long_only, buy_and_sell


# Function to calculate robustness score
def f_robustness(pf, trades, win_rate, min_trades):
    if trades >= min_trades and not np.isnan(pf) and pf >= 0 and not np.isnan(win_rate):
        return pf * np.log(trades) * np.sqrt(win_rate)
    return -1e10

def backtest(data, rsi_len, strategy_type, min_trades, overbought=70, oversold=30):
    # Calculate RSI
    rsi = RSIIndicator(data['Close'], window=rsi_len).rsi()

    # Detect RSI signals
    buy_signal = (rsi > oversold) & (rsi.shift(1) <= oversold)  # RSI crosses above oversold
    sell_signal = (rsi < overbought) & (rsi.shift(1) >= overbought)  # RSI crosses below overbought
    
    if strategy_type == 'Buy & Sell':
        total_trades, winning_trades, total_profit, total_loss = buy_and_sell.rsi_strategy(data, buy_signal, sell_signal)
    elif strategy_type == 'Long Only':
        total_trades, winning_trades, total_profit, total_loss = long_only.rsi_strategy(data, buy_signal, sell_signal)
    elif strategy_type == 'Double Dip':
        total_trades, winning_trades, total_profit, total_loss = double_dip.rsi_strategy(data, buy_signal, sell_signal)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

     # Calculate metrics
    profit_factor = total_profit / total_loss if total_loss > 0 else (10000.0 if total_profit > 0 else 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    robustness = f_robustness(profit_factor, total_trades, win_rate, min_trades)

    return total_trades, profit_factor, win_rate, robustness

# Main function to find the best RSI period
def find_best_rsi(data, strategy_type='Long Only', min_trades=100, show_table=True, 
                    overbought_range=range(60, 81, 5), oversold_range=range(20, 41, 5)):

    best_score = -1e10
    best_len = np.nan
    best_trades = np.nan
    best_pf = np.nan
    best_wr = np.nan
    best_overbought = np.nan
    best_oversold = np.nan
    
    # Test RSI periods and thresholds
    for rsi_len in range(5, 51, 1):
        for overbought in overbought_range:
            for oversold in oversold_range:
                if oversold >= overbought: # skip any invalid range 
                    continue 

                total_trades, profit_factor, win_rate, robustness = backtest(
                    data, rsi_len, strategy_type, min_trades, overbought, oversold
                )
                if not np.isnan(robustness) and robustness > best_score:
                    best_score = robustness
                    best_len = rsi_len
                    best_trades = total_trades
                    best_pf = profit_factor
                    best_wr = win_rate
                    best_overbought = overbought
                    best_oversold = oversold
    
    # Display results
    if show_table:
        if not np.isnan(best_len):
            print("Optimal RSI Period:", best_len)
            print("Optimal Overbought Threshold:", best_overbought)
            print("Optimal Oversold Threshold:", best_oversold)
            print("Total Trades:", best_trades)
            print(f"Profit Factor: {best_pf:.2f}")
            print(f"Win Rate (%): {best_wr * 100:.1f}")
            print(f"Robustness Score: {best_score:.2f}")
        else:
            print("Optimal RSI Period: Calculating...")
            print("Optimal Overbought Threshold: --")
            print("Optimal Oversold Threshold: --")
            print("Total Trades: --")
            print("Profit Factor: --")
            print("Win Rate (%): --")
            print("Robustness Score: --")
    
    return best_len, best_overbought, best_oversold, best_trades, best_pf, best_wr, best_score

# Fetch data from yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data retrieved for ticker {ticker}")
    return data

# Example usage
if __name__ == "__main__":
    # User inputs
    ticker = 'OKLO'
    start_date = '2023-01-01'
    end_date = '2026-08-01'
    strategy_type = 'Double Dip'
    min_trades = 1
    show_table = True
    overbought_range = range(60, 81, 5)  # Test 60, 65, 70, 75, 80
    oversold_range = range(20, 41, 5)    # Test 20, 25, 30, 35, 40
    print(f'Symbol: {ticker}')
    print(f'Strategy: {strategy_type}')
    
    # Fetch stock data
    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        # Run the optimization
        best_len, best_overbought, best_oversold, best_trades, best_pf, best_wr, best_score = find_best_rsi(
            data, strategy_type, min_trades, show_table, overbought_range, oversold_range
        )
    except Exception as e:
        print(f"Error fetching data: {e}")