import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas_ta as ta
import datetime
from logzero import logger, loglevel
import logzero
from concurrent.futures import ThreadPoolExecutor, as_completed
from st2 import custom_resample, apply_pivots

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.Datetime = pd.to_datetime(data.Datetime)
    data.set_index('Datetime', inplace=True)
    return data

def preprocess_data(data):
    data_hourly = custom_resample(data, freq='1h')
    data_hourly['RSI'] = ta.rsi(data_hourly['Close'], timeperiod=14)

    data['Date'] = data.index.date
    data['datetime'] = data.index

    merged_data = pd.merge(data, data_hourly[['RSI']], left_index=True, right_index=True, how='left').ffill()
    daily_data = custom_resample(data, freq='D')
    daily_data['Date'] = daily_data.index.date
    daily_data = apply_pivots(daily_data)

    merged_data = pd.merge(merged_data, daily_data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']], 
                           left_on='Date', right_on='Date', how='left')
    merged_data.set_index('datetime', inplace=True)
    merged_data['wma'] = ta.vwma(merged_data['Close'], merged_data['Volume'], length=5)
    
    return merged_data

def define_trading_conditions(data):
    """Define the buy and sell conditions."""
    start_time = datetime.time(10, 14)
    end_time = datetime.time(14, 16)
    buy_condition = (
        (data['%CPR'] < 0.01) &
        (data.RSI > 70) &
        (data.Close > data.R1) &
        (data.Close.shift(1) < data.R1) &
        (data.index.time >= start_time) &
        (data.index.time <= end_time)
    )
                    
    sell_condition = (data.Close > data.R3) | (data.Close < data.TC) 
    
    data['Signal'] = np.select([buy_condition, sell_condition], ['Buy', 'Sell'])
    data['Shifted_close'] = data['Close'].shift()
    
    return data

def backtest(df, initial_capital=100000, risk_per_trade=0.01, stop_loss=0.01, take_profit=0.03):
    """Backtest the trading strategy with given conditions."""
    position = False
    trades = {'Buy Date': [], 'Buy Price': [], 'Sell Date': [], 'Sell Price': [], 'Quantity': []}
    capital = initial_capital
    close_time = datetime.time(15, 10)
    
    for index, row in df.iterrows():
        if row['Signal'] == 'Buy' and not position:
            position = True
            risk_amount = capital * risk_per_trade
            qty = math.floor(risk_amount / (stop_loss * row['Close']))
            capital -= qty * row['Close']
            trades['Buy Date'].append(index)
            trades['Buy Price'].append(row['Close'])
            trades['Quantity'].append(qty)
            logger.info(f"Buy Signal: {index}, Price: {row['Close']}, Quantity: {qty}")

        if position:
            if row['Signal'] == 'Sell' or row['Shifted_close'] < (1 - stop_loss) * trades['Buy Price'][-1] or \
               row['Shifted_close'] > (1 + take_profit) * trades['Buy Price'][-1] or index.time() > close_time:
                trades['Sell Date'].append(index)
                trades['Sell Price'].append(row['Close'])
                capital += qty * row['Close']
                logger.info(f"Sell Signal: {index}, Price: {row['Close']}, PnL: {qty * (row['Close'] - trades['Buy Price'][-1])}")
                position = False

    # Ensure lists are of equal length by filling with NaN
    max_len = max(len(trades['Buy Date']), len(trades['Sell Date']))
    for key in trades.keys():
        trades[key].extend([np.nan] * (max_len - len(trades[key])))

    return pd.DataFrame(trades)

def calculate_trade_metrics(trade_history_df):
    """Calculate metrics like Return, PnL, Cumulative Profit, etc."""
    trade_history_df['Return'] = (trade_history_df['Sell Price'] - trade_history_df['Buy Price']) / trade_history_df['Buy Price']
    trade_history_df['Days'] = (trade_history_df['Sell Date'] - trade_history_df['Buy Date']).dt.days
    trade_history_df['PnL'] = trade_history_df['Sell Price'] - trade_history_df['Buy Price']
    trade_history_df['Realized Profit'] = trade_history_df['PnL'] * trade_history_df['Quantity']
    trade_history_df['Cum Profit'] = trade_history_df['Realized Profit'].cumsum()
    trade_history_df['Cumulative Return'] = (1 + trade_history_df['Return']).cumprod() - 1
    trade_history_df['Drawdown'] = trade_history_df['Cum Profit'].cummax() - trade_history_df['Cum Profit']
    
    # Calculate KPIs
    total_trades = len(trade_history_df.dropna())
    winning_trades = len(trade_history_df[trade_history_df['PnL'] > 0].dropna())
    losing_trades = len(trade_history_df[trade_history_df['PnL'] < 0].dropna())
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_pnl = trade_history_df['PnL'].mean()
    total_profit = trade_history_df['Realized Profit'].sum()
    
    kpi = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': win_rate,
        'Average PnL': avg_pnl,
        'Total Profit': total_profit,
        'Max Drawdown': trade_history_df['Drawdown'].max()
    }
    
    return trade_history_df, kpi

def plot_metrics(trade_history_df, stock_symbol):
    """Plot cumulative profit and drawdown."""
    plt.figure(figsize=(14, 7))

    # Plot Cumulative Profit
    plt.subplot(2, 1, 1)
    plt.plot(trade_history_df['Buy Date'], trade_history_df['Cum Profit'], label='Cumulative Profit')
    plt.title(f'Cumulative Profit for {stock_symbol}')
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.legend()

    # Plot Drawdown
    plt.subplot(2, 1, 2)
    plt.plot(trade_history_df.index, trade_history_df['Drawdown'], label='Drawdown', color='red')
    plt.title(f'Drawdown for {stock_symbol}')
    plt.xlabel('Trade')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_file(file_path):
    file_name = os.path.basename(file_path).replace('.csv', '')
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    signal_data = define_trading_conditions(processed_data)
    backtest_results = backtest(signal_data)
    trade_history_df, kpi = calculate_trade_metrics(backtest_results)
    
    # Save the trade history to a CSV file
    output_dir = 'Backtest/FnO/Results/Charts'
    os.makedirs(output_dir, exist_ok=True)
    trade_history_df.to_csv(os.path.join(output_dir, f'{file_name}_trade_history.csv'), index=False)
    
    # Plot the metrics
    # plot_metrics(trade_history_df, file_name)

    return kpi

def main(folder_path):
    result_list = []
    stock_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file): file for file in stock_files}

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                result_list.append(result)
            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")

    # Save the results to a CSV file
    result_df = pd.DataFrame(result_list)
    
    result_df.to_csv('Backtest/FnO/Results/backtesting_results.csv', index=False)
    print("Backtesting results saved to 'backtesting_results.csv'")

if __name__ == "__main__":
    main('FnO/5M')
