# Test for the optimal SMA for a stock 

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
import yfinance as yf

import sys 
import os 
# Get the directory of the current script (test/sma.py)
# current_dir = os.path.dirname(__file__a)

# Add the parent directory (my_project/) to sys.path
# This makes 'sma_strats' discoverable as a top-level package
# sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.insert(0, "../strats")
from strats import double_dip, long_only, buy_and_sell


# Function to calculate robustness score
def f_robustness(pf, trades, win_rate, min_trades):
    if trades >= min_trades and not np.isnan(pf) and pf >= 0 and not np.isnan(win_rate):
        return pf * np.log(trades) * np.sqrt(win_rate)
    return -1e10


def backtest(data, ma_len, strategy_type, min_trades):
    # calc SMA
    sma = SMAIndicator(data['Close'], window=ma_len).sma_indicator()

    # Detect crossovers and crossunders
    crossover = (data['Close'] > sma) & (data['Close'].shift(1) <= sma.shift(1)) # if price moves above SMA from below
    crossunder = (data['Close'] < sma) & (data['Close'].shift(1) >= sma.shift(1)) # if price moves below SMA from above

    
    if strategy_type == 'Buy & Sell':
        total_trades, winning_trades, total_profit, total_loss = buy_and_sell.sma_strategy(data, crossover, crossunder)
    elif strategy_type == 'Long Only':
        total_trades, winning_trades, total_profit, total_loss = long_only.sma_strategy(data, crossover, crossunder)
    elif strategy_type == 'Double Dip':
        total_trades, winning_trades, total_profit, total_loss = double_dip.sma_strategy(data, crossover, crossunder)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

     # Calculate metrics
    profit_factor = total_profit / total_loss if total_loss > 0 else (10000.0 if total_profit > 0 else 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    robustness = f_robustness(profit_factor, total_trades, win_rate, min_trades)

    return total_trades, profit_factor, win_rate, robustness

# Main function to find the best SMA length
def find_best_sma(data, strategy_type='Long Only', min_trades=100, show_table=True):
    best_score = -1e10
    best_len = np.nan
    best_trades = np.nan
    best_pf = np.nan
    best_wr = np.nan
    
    # Test SMA lengths from 10 to 1000 in steps of 10
    for ma_len in range(10, 1000, 10): 
        total_trades, profit_factor, win_rate, robustness = backtest(data, ma_len, strategy_type, min_trades)
        if not np.isnan(robustness) and robustness > best_score:
            best_score = robustness
            best_len = ma_len
            best_trades = total_trades
            best_pf = profit_factor
            best_wr = win_rate
    
    # Display results
    if show_table:
        if not np.isnan(best_len):
            print("Optimal SMA Length:", best_len)
            print("Total Trades:", best_trades)
            print(f"Profit Factor: {best_pf:.2f}")
            print(f"Win Rate (%): {best_wr * 100:.1f}")
            print(f"Robustness Score: {best_score:.2f}")
        else:
            print("Optimal SMA Length: Calculating...")
            print("Total Trades: --")
            print("Profit Factor: --")
            print("Win Rate (%): --")
            print("Robustness Score: --")
    
    return best_len, best_trades, best_pf, best_wr, best_score

# Fetch data from yfinance
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    # Ensure the DataFrame has 'Close' column (yfinance uses 'Close' for closing prices)
    if data.empty:
        raise ValueError(f"No data retrieved for ticker {ticker}")
    return data

# Example usage
if __name__ == "__main__":
    # User inputs
    ticker = 'AAPL'  # Stock ticker symbol
    start_date = '2020-01-01'  # Start date for historical data
    end_date = '2025-06-28'  # End date (current date as per context)
    strategy_type = 'Double Dip'  # Options: 'Buy & Sell', 'Long Only'
    min_trades = 5 # default = 100
    show_table = True
    print(f'Symbol: {ticker}')
    print(f'Strategy: {strategy_type}')
    
    # Fetch stock data
    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        # Run the optimization
        best_len, best_trades, best_pf, best_wr, best_score = find_best_sma(
            data, strategy_type, min_trades, show_table
        )
    except Exception as e:
        print(f"Error fetching data: {e}")