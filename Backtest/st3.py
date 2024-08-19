import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas_ta as ta
import datetime
from logzero import logger, loglevel
import logzero
import st2
# Set up logzero
logzero.logfile("backtest.log", maxBytes=1e6, backupCount=3)
loglevel(logzero.INFO)

# Create the directory to save trade history if it doesn't exist
def create_trade_history_folder(folder_name="trade_history"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Data Processing Functions

def load_and_prepare_data(file_path):
    """Load 5-minute data and prepare hourly and daily data with pivot points."""
    
    min5_data = st2.load_data(file_path)
    indexed_5min = min5_data.set_index('Datetime')
    indexed_5min.index = pd.to_datetime(indexed_5min.index)
    
    hourly_data = st2.custom_resample(indexed_5min, freq='h').dropna()
    daily_data = st2.custom_resample(hourly_data, freq='D').dropna()
    daily_data = st2.apply_pivots(daily_data)
    
    hourly_data['Date'] = hourly_data.index.date
    hourly_data['datetime'] = hourly_data.index
    
    merged_data = pd.merge()
    
    merged_data = pd.merge(hourly_data, daily_data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']], on='Date')
    merged_data.set_index('datetime', inplace=True)
    merged_data.dropna(inplace=True)
    
    return merged_data

def calculate_indicators(df):
    """Calculate technical indicators."""
    df['rsi'] = ta.rsi(df['Close'], length=14).astype('float64')
    df['mfi'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9'].astype('float64')
    df['macd_signal'] = macd['MACDs_12_26_9'].astype('float64')
    
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0'].astype('float64')
    df['bb_lower'] = bb['BBL_20_2.0'].astype('float64')
    df['percent_b'] = ((df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).astype('float64')
    
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3'].astype('float64')
    df['stoch_d'] = stoch['STOCHd_14_3_3'].astype('float64')
    
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['adx'] = adx['ADX_14'].astype('float64')
    
    return df

def define_trading_conditions(data):
    """Define the buy and sell conditions."""
    start_time = datetime.time(10, 14)
    end_time = datetime.time(14, 16)
    close_time = datetime.time(15, 10)
    
    buy_condition = (data.rsi > 70) & (data.mfi > 80) & (data.macd > data.macd_signal) & \
                    (data.percent_b > 80) & (data.stoch_k > 80) & (data.stoch_d > 80) & \
                    (data.adx > 20) & (data.Close > data.R2) & \
                    (data.index.time > start_time) & (data.index.time < end_time)
                    
    sell_condition = (data.Close > data.R3) | (data.Close < data.TC) | (data.index.time >= close_time)
    
    data['Signal'] = np.select([buy_condition, sell_condition], ['Buy', 'Sell'])
    data['Shifted_close'] = data['Close'].shift()
    
    
    return data

# Backtesting Function

def backtest(df, initial_capital=100000, risk_per_trade=0.01, stop_loss=0.01, take_profit=0.03):
    """Backtest the trading strategy with given conditions."""
    position = False
    trades = {'Buy Date': [], 'Buy Price': [], 'Sell Date': [], 'Sell Price': [], 'Quantity': []}
    capital = initial_capital
    
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
            if row['Signal'] == 'Sell' or row['Shifted_close'] < (1 - stop_loss) * trades['Buy Price'][-1]  or row['Shifted_close'] > (1 + take_profit) * trades['Buy Price'][-1]:
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

# Metrics Calculation and Visualization

def calculate_trade_metrics(trade_history_df):
    """Calculate metrics like Return, PnL, Cumulative Profit, etc."""
    trade_history_df['Return'] = (trade_history_df['Sell Price'] - trade_history_df['Buy Price']) / trade_history_df['Buy Price']
    trade_history_df['Days'] = (trade_history_df['Sell Date'] - trade_history_df['Buy Date']).dt.days
    trade_history_df['PnL'] = trade_history_df['Sell Price'] - trade_history_df['Buy Price']
    trade_history_df['Realized Profit'] = trade_history_df['PnL'] * trade_history_df['Quantity']
    trade_history_df['Cum Profit'] = trade_history_df['Realized Profit'].cumsum()
    trade_history_df['Cumulative Return'] = (1 + trade_history_df['Return']).cumprod() - 1
    trade_history_df['Drawdown'] = trade_history_df['Cumulative Return'].cummax() - trade_history_df['Cumulative Return']
    
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
    plt.plot(trade_history_df.index, trade_history_df['Cum Profit'], label='Cumulative Profit')
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

def run_backtest_for_multiple_stocks(stock_files):
    """Run backtests for multiple stocks."""
    kpi_list = []
    trade_history_folder = create_trade_history_folder()
    for file_path in stock_files:
        stock_symbol = file_path.split('/')[-1].split('.')[0]
        logger.info(f"Starting backtest for {stock_symbol}")
        
        # Load and prepare data
        merged_data = load_and_prepare_data(file_path)
        
        # Calculate indicators and define trading conditions
        data_with_indicators = calculate_indicators(merged_data)
        data_with_signals = define_trading_conditions(data_with_indicators)
        
        # Perform backtesting
        trade_history_df = backtest(data_with_signals)
        
        # Calculate metrics and plot results
        trade_history_df, kpi = calculate_trade_metrics(trade_history_df)
        # plot_metrics(trade_history_df, stock_symbol)
        
        # Save trade history
        trade_history_file = os.path.join(trade_history_folder, f"{stock_symbol}_trade_history.csv")
        trade_history_df.to_csv(trade_history_file)
        
        # Store KPIs
        kpi['Stock'] = stock_symbol
        kpi_list.append(kpi)
        
        logger.info(f"Completed backtest for {stock_symbol}")

    # Create a DataFrame with KPIs for all stocks
        kpi_df = pd.DataFrame(kpi_list)
        kpi_df.to_csv(os.path.join(trade_history_folder, "stocks_kpi_summary.csv"))
        logger.info("All backtests completed. KPI summary saved.")
        
        
if __name__ == "__main__":
    # Logging
    file_directory = 'nifty_200/5M'
    stock_files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith('.csv')]
    run_backtest_for_multiple_stocks(stock_files)