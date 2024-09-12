import pandas as pd
import numpy as np
import yfinance as yf
from multiprocessing import Pool, cpu_count

# Load your data
data = yf.download(tickers='SBIN.NS', period='2y', interval='1h')

# Define the range of periods for EMA, RSI, and MACD
ema_periods = range(10, 50, 2)  # 20 settings: 10, 12, ..., 48
rsi_periods = range(7, 47, 2)   # 20 settings: 7, 9, ..., 45
macd_fast_periods = range(8, 48, 2)  # 20 settings: 8, 10, ..., 46
macd_slow_periods = range(17, 57, 2) # 20 settings: 17, 19, ..., 55
macd_signal_periods = range(7, 47, 2) # 20 settings: 7, 9, ..., 45

# Initialize an empty DataFrame to store results
full_results = pd.DataFrame(columns=['EMA_Period', 'RSI_Period', 'MACD_Fast', 'MACD_Slow', 'MACD_Signal', 'Cumulative_Returns'])

def calculate_indicators(data, ema_period, rsi_period, macd_fast, macd_slow, macd_signal):
    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    average_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = average_gain / average_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    fast_ema = data['Close'].ewm(span=macd_fast, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=macd_slow, adjust=False).mean()
    data['MACD'] = fast_ema - slow_ema
    data['MACD_signal'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
def generate_signals(data):
    data['signal'] = np.where(
        (data['Close'] > data['EMA']) & 
        (data['RSI'] > 50) & 
        (data['MACD'] > data['MACD_signal']), 1, 
        np.where(
            (data['Close'] < data['EMA']) & 
            (data['RSI'] < 50) & 
            (data['MACD'] < data['MACD_signal']), -1, 
            0
        )
    )
    
def apply_sl_tsl_target(data, initial_capital=50000, sl_pct=0.01, tsl_pct=0.005, target_pct=0.02):
    data['position'] = data['signal'].shift(1)
    data['entry_price'] = np.where(data['position'] != 0, data['Close'], np.nan)
    data['entry_price'].fillna(method='ffill', inplace=True)
    
    # Stop Loss and Target
    data['stop_loss'] = data['entry_price'] * (1 - sl_pct * data['position'])
    data['target'] = data['entry_price'] * (1 + target_pct * data['position'])
    
    # Trailing Stop Loss
    data['trailing_stop'] = data['Close'].where(data['position'] != 0).cummax() * (1 - tsl_pct * data['position'])
    
    # Exit conditions
    data['exit'] = np.where(
        (data['Close'] <= data['stop_loss']) & (data['position'] == 1) |
        (data['Close'] >= data['target']) & (data['position'] == 1) |
        (data['Close'] >= data['stop_loss']) & (data['position'] == -1) |
        (data['Close'] <= data['target']) & (data['position'] == -1) |
        (data['Close'] <= data['trailing_stop']) & (data['position'] == 1) |
        (data['Close'] >= data['trailing_stop']) & (data['position'] == -1),
        1, 0
    )
    
    data['trade_return'] = (data['Close'] - data['entry_price']) * data['position'] / data['entry_price']
    data['trade_return'] = np.where(data['exit'] == 1, data['trade_return'], 0)
    
    data['cumulative_returns'] = (1 + data['trade_return']).cumprod()
    
def backtest(data, ema_period, rsi_period, macd_fast, macd_slow, macd_signal):
    calculate_indicators(data, ema_period, rsi_period, macd_fast, macd_slow, macd_signal)
    generate_signals(data)
    apply_sl_tsl_target(data)
    return data['cumulative_returns'].iloc[-1]

# Perform backtest for all combinations
def backtest_combinations(combo):
    ema_period, rsi_period, macd_fast, macd_slow, macd_signal = combo
    data_copy = data.copy()
    cumulative_return = backtest(data_copy, ema_period, rsi_period, macd_fast, macd_slow, macd_signal)
    return {
        'EMA_Period': ema_period,
        'RSI_Period': rsi_period,
        'MACD_Fast': macd_fast,
        'MACD_Slow': macd_slow,
        'MACD_Signal': macd_signal,
        'Cumulative_Returns': cumulative_return
    }

# Generate all combinations
combinations = [
    (ema_period, rsi_period, macd_fast, macd_slow, macd_signal)
    for ema_period in ema_periods
    for rsi_period in rsi_periods
    for macd_fast in macd_fast_periods
    for macd_slow in macd_slow_periods
    for macd_signal in macd_signal_periods
    if macd_fast < macd_slow  # Valid MACD settings
]

# Use multiprocessing to parallelize backtests
if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        results = pool.map(backtest_combinations, combinations)
    
    # Convert results to DataFrame
    full_results = pd.DataFrame(results)
    
    # Save the results to a CSV file for further analysis
    full_results.to_csv('sensitivity_analysis_results.csv', index=False)

    # Display the top 5 results sorted by cumulative returns
    top_results = full_results.sort_values(by='Cumulative_Returns', ascending=False).head(5)
    print(top_results)
