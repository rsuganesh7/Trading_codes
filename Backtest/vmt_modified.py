# -*- coding: utf-8 -*-
"""
Angel One - Opening Range Breakout strategy backtesting with KPI

Further refactored version with saving trades in a file.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import pandas_ta as ta
import sys
import os
import datetime as dt
sys.path.append(os.path.abspath('/Users/suganeshr/Trading/'))

from Trading_codes.Live.login_manager import LoginManager
from Trading_codes.Live.instruments_list import InstrumentDetail

lm = LoginManager()
obj, data, usr = lm.login_and_retrieve_info()

tickers = pd.read_csv('/Users/suganeshr/Trading/Trading_codes/FnO.csv')['SYMBOL'].to_list()
bktst_start_dt = "2023-06-01 09:15"
bktst_end_dt = "2024-09-06 15:30"

# Initialize an empty DataFrame to store the trades
trades_df = pd.DataFrame(columns=["Date", "Ticker", "Direction", "Entry_Price", "Exit_Price", "PnL"])


def custom_resample(df, start='09:15', end='15:30', freq='1h'):
    """Resample data within a given time range."""
    df = df.between_time(start, end)
    return df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()


def calculate_pivot_levels(high, low, close):
    """Calculate pivot levels for given high, low, and close prices."""
    pp = (high + low + close) / 3
    tc = (pp + high) / 2
    bc = (pp + low) / 2
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)
    return pp, tc, bc, r1, r2, r3, s1, s2, s3


def apply_pivots(df):
    """Apply pivot calculations to the dataframe."""
    df['Prev_High'] = df['high'].shift(1)
    df['Prev_Low'] = df['low'].shift(1)
    df['Prev_Close'] = df['close'].shift(1)

    pivot_columns = df.apply(
        lambda row: calculate_pivot_levels(row['Prev_High'], row['Prev_Low'], row['Prev_Close']), 
        axis=1, result_type='expand'
    )

    df[['PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']] = pivot_columns
    df['%CPR'] = abs(df['TC'] - df['BC']) / df['PP']
    df['Date'] = df.index.date
    return df


def fetch_token_for_ticker(ticker):
    """Fetch the token for a given ticker."""
    token = id.token_lookup(ticker=ticker)
    print(f'Token for {ticker}: {token}')
    return token


def create_params_for_ticker(token, interval, exchange):
    """Create request parameters for fetching historical data."""
    return {
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": bktst_start_dt,
        "todate": bktst_end_dt
    }


def fetch_historical_data_for_ticker(ticker, token, params):
    """Fetch historical data for a single ticker."""
    try:
        hist_data = obj.getCandleData(params)
        df = pd.DataFrame(hist_data["data"], columns=["date", "open", "high", "low", "close", "volume"])
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["gap"] = ((df["open"] / df["close"].shift(1)) - 1) * 100
        df["avvol"] = df["volume"].rolling(10).mean().shift(1)
        df = apply_pivots(df)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_historical_data(tickers, instrument_list, interval="ONE_DAY", exchange="NSE"):
    """Fetch historical data for multiple tickers."""
    hist_data_tickers = {}
    for ticker in tickers:
        
        token = fetch_token_for_ticker(ticker)
        params = create_params_for_ticker(token, interval, exchange)
        df = fetch_historical_data_for_ticker(ticker, token, params)
        if df is not None:
            hist_data_tickers[ticker] = df
    return hist_data_tickers


def fetch_intraday_data_for_ticker(ticker, date, params):
    """Fetch intraday data for a single ticker."""
    try:
        time.sleep(0.3) 
        hist_data = obj.getCandleData(params)
        df = pd.DataFrame(hist_data["data"], columns=["date", "open", "high", "low", "close", "volume"])
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["Date"] = df.index.date
        return df
    except Exception as e:
        print(f"Error fetching intraday data for {ticker} on {date}: {e}")
        return None


def create_intraday_params(ticker, date, interval, exchange):
    """Create parameters for intraday data fetch."""
    return {
        "exchange": exchange,
        "symboltoken": id.token_lookup(ticker=ticker),
        "interval": interval,
        "fromdate": (date - timedelta(days=6)).strftime("%Y-%m-%d") + ' 09:15',
        "todate": date.strftime("%Y-%m-%d") + ' 15:30'
    }


def fetch_intraday_data(ticker, date, interval, instrument_list, exchange="NSE"):
    """Fetch intraday data for a given ticker and date."""
    params = create_intraday_params(ticker, date, interval, exchange)
    return fetch_intraday_data_for_ticker(ticker, date, params)


def preprocess_data(data):
    """Preprocess the data by resampling, applying RSI, and adding pivot levels."""
    data_hourly = custom_resample(data, freq='1h')
    data_hourly['RSI'] = ta.rsi(data_hourly['close'], timeperiod=14)

    data['Date'] = data.index.date
    data['datetime'] = data.index

    merged_data = merge_hourly_data(data, data_hourly)
    daily_data = custom_resample(data, freq='D')
    daily_data['Date'] = daily_data.index.date
    daily_data = apply_pivots(daily_data)

    return finalize_preprocessed_data(merged_data, daily_data)

# Initialize the trades_df DataFrame globally
trades_df = pd.DataFrame(columns=["Date", "Ticker", "Direction", "Entry_Price", "Exit_Price", "PnL"])

def merge_hourly_data(data, data_hourly):
    """Merge hourly data with the original data."""
    return pd.merge(data, data_hourly[['RSI']], left_index=True, right_index=True, how='left').ffill()


def finalize_preprocessed_data(merged_data, daily_data):
    """Finalize the preprocessed data by merging with daily data."""
    merged_data = pd.merge(merged_data, daily_data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']], 
                           left_on='Date', right_on='Date', how='left')
    merged_data.set_index('datetime', inplace=True)
    merged_data['vwma'] = ta.vwma(merged_data['close'], merged_data['volume'], length=5)
    return merged_data


def find_top_gap_stocks(data):
    """Identify top gap stocks with %CPR less than 1.5%."""
    top_gap_by_date = {}
    dates = data[tickers[0]].index.to_list()

    for date in dates:
        temp = find_gap_stocks_on_date(data, date)
        if not temp.empty:
            top_gap_by_date[date] = temp.index.to_list()

    return top_gap_by_date


def find_gap_stocks_on_date(data, date):
    """Identify gap stocks for a specific date."""
    temp = pd.Series()
    for ticker in data:
        try:
            if data[ticker].loc[date, '%CPR'] < 1.5:
                temp[ticker] = data[ticker].loc[date, '%CPR']
        except KeyError:
            pass
    return temp
def process_ticker_for_backtest(ticker, date, date_stats, date_str, candle_data):
    global trades_df
    """Process a single ticker for backtesting."""
    try:
        intraday_df = fetch_intraday_data(ticker, date, 'FIVE_MINUTE', instrument_list)
        if intraday_df is None:
            print(f"Skipping {ticker} for {date} due to missing data.")
            return

        intraday_df = preprocess_data(intraday_df)
        intraday_df = define_trading_conditions(intraday_df)
        open_price, direction = '', ''
        date_stats[date][ticker] = 0

        # Iterate over the signals to make trading decisions
        for i in range(1, len(intraday_df)):
            signal = intraday_df.iloc[i]['Signal']

            if signal == 'Buy' and open_price == '':
                open_price = intraday_df.iloc[i]['close']
                direction = 'long'

            elif signal == 'Sell' and direction == 'long':
                ticker_return = (intraday_df.iloc[i]['close'] / open_price) - 1
                date_stats[date][ticker] = ticker_return
                trade_row = pd.DataFrame({
                    "Date": [date_str],
                    "Ticker": [ticker],
                    "Direction": ["long"],
                    "Entry_Price": [open_price],
                    "Exit_Price": [intraday_df.iloc[i]['close']],
                    "PnL": [ticker_return]
                })
                trades_df = pd.concat([trades_df, trade_row], ignore_index=True)
                open_price = ''
                direction = ''

            elif signal == 'Short' and open_price == '':
                open_price = intraday_df.iloc[i]['close']
                direction = 'short'

            elif signal == 'Cover' and direction == 'short':
                ticker_return = 1 - (intraday_df.iloc[i]['close'] / open_price)
                date_stats[date][ticker] = ticker_return
                trade_row = pd.DataFrame({
                    "Date": [date_str],
                    "Ticker": [ticker],
                    "Direction": ["short"],
                    "Entry_Price": [open_price],
                    "Exit_Price": [intraday_df.iloc[i]['close']],
                    "PnL": [ticker_return]
                })
                trades_df = pd.concat([trades_df, trade_row], ignore_index=True)
                open_price = ''
                direction = ''

    except KeyError as e:
        print(f"KeyError: {ticker} on {date}: {e}")
    except Exception as e:
        print(f"An error occurred for {ticker} on {date}: {e}")     
    
def execute_backtest(top_gap_by_date, candle_data):
    """Execute the backtest strategy using pivot points and top gap stocks."""
    
    
    date_stats = {}

    for date in top_gap_by_date:
        date_str = date.strftime("%Y-%m-%d %H:%M")
        date_stats[date] = {}

        for ticker in top_gap_by_date[date]:
            process_ticker_for_backtest(ticker, date, date_stats, date_str, candle_data)

    return date_stats
def define_trading_conditions(data):
    """Define the buy and sell conditions."""
    start_time = dt.time(10, 14)
    end_time = dt.time(14, 16)
    
    # Buy condition
    buy_condition = (
        (data['%CPR'] < 0.01) &
        (data.RSI > 70) &
        (data.close > data.R1) &
        (data.close.shift(1) < data.R1) &
        (data.index.time >= start_time) &
        (data.index.time <= end_time)
    )
    
    # Sell condition
    sell_condition = (data.close > data.R3) | (data.close < data.TC) | (data.RSI < 50)
    
    # Short condition
    short_condition = (
        (data['%CPR'] < 0.01) &
        (data.RSI < 30) &
        (data.close < data.S1) &
        (data.close.shift(1) > data.S1) &
        (data.index.time >= start_time) &
        (data.index.time <= end_time)
    )
    
    # Cover condition
    cover_condition  = (data.close < data.S3) | (data.close > data.BC) | (data.RSI > 50)
    
    # Create a 'Signal' column with Buy, Sell, Short, and Cover signals
    data['Signal'] = np.select([buy_condition, sell_condition, short_condition, cover_condition], ['Buy', 'Sell', 'Short', 'Cover'])
    
    # Shift the close column for comparison
    data['Shifted_close'] = data['close'].shift()
    
    return data



# KPIs and Performance Metrics
def abs_return(date_stats):
    """Calculate the absolute return of the strategy."""
    df = pd.DataFrame(date_stats).T
    df["ret"] = 1 + df.mean(axis=1)
    return (df["ret"].cumprod() - 1).iloc[-1]


def win_rate(date_stats):
    """Calculate the win rate of the strategy."""
    win_count = sum(1 for i in date_stats for t in date_stats[i] if date_stats[i][t] > 0)
    total_count = sum(1 for i in date_stats for t in date_stats[i])
    return (win_count / total_count) * 100


def mean_return(date_stats, condition):
    """Calculate the mean return for winning or losing trades."""
    returns = [date_stats[i][t] for i in date_stats for t in date_stats[i] if condition(date_stats[i][t])]
    return sum(returns) / len(returns)


def return_curve(date_stats):
    """Plot the cumulative return curve."""
    df = pd.DataFrame(date_stats).T
    df["ret"] = 1 + df.mean(axis=1)
    df["ret"].fillna(1, inplace=True)
    df["cum_ret"] = df["ret"].cumprod() - 1
    df["cum_ret"].plot(title="Return Profile")


# Main code execution
id = InstrumentDetail()
instrument_list = id.get_instrument_list()
candle_data = fetch_historical_data(tickers, instrument_list)

top_gap_by_date = find_top_gap_stocks(candle_data)
date_stats = execute_backtest(top_gap_by_date, candle_data)

# KPIs
print("**********Strategy Performance Statistics**********")
print(f"Total cumulative return = {abs_return(date_stats):.4f}")
print(f"Total win rate = {win_rate(date_stats):.2f}%")
print(f"Mean return per win trade = {mean_return(date_stats, lambda x: x > 0):.4f}")
print(f"Mean return per loss trade = {mean_return(date_stats, lambda x: x < 0):.4f}")
return_curve(date_stats)

# Save the trades to a CSV file after the backtest
trades_df.to_csv("backtest_trades.csv", index=False)
print("Trades have been saved to 'backtest_trades.csv'.")